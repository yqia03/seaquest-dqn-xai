"""Decision-specific explanations on the saved Seaquest inputs."""
import copy
import hashlib
import time

import cv2
import numpy as np
import torch
from captum.attr import IntegratedGradients


def predict(model, batch, batch_size=32):
    with torch.inference_mode():
        values = np.concatenate([
            model(torch.as_tensor(batch[i:i + batch_size], dtype=torch.float32)).cpu().numpy()
            for i in range(0, len(batch), batch_size)
        ])
    if not np.isfinite(values).all():
        raise ValueError("Non-finite Q-values")
    return values


def action_pair(q):
    # Match argmax: the lowest action index wins a tie.
    return tuple(int(i) for i in np.argsort(-np.asarray(q), kind="stable")[:2])


def grid(height=84, width=84, patch=6):
    if patch <= 0:
        raise ValueError("Patch size must be positive")
    return [(y, min(y + patch, height), x, min(x + patch, width))
            for y in range(0, height, patch) for x in range(0, width, patch)]


def replacement(stack, fill):
    if fill == "black":
        return np.zeros_like(stack, dtype=np.float32)
    if fill == "blur":
        return np.stack([cv2.GaussianBlur(f.astype(np.float32), (0, 0), 5)
                         for f in stack])
    raise ValueError(f"Unknown fill: {fill}")


def log_softmax(q):
    q = np.asarray(q, dtype=np.float64)
    shifted = q - q.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def sarfa_components(before, after, action):
    """Author's SARFA formula, with stable softmax and KL(before || after)."""
    other = np.arange(len(before)) != action
    dp = np.maximum(np.exp(log_softmax(before))[action]
                    - np.exp(log_softmax(after))[:, action], 0)
    lp, lq = log_softmax(before[other]), log_softmax(after[:, other])
    kl = np.maximum((np.exp(lp) * (lp - lq)).sum(-1), 0)
    specificity = 1 / (1 + kl)
    scores = 2 * dp * specificity / np.maximum(dp + specificity, 1e-30)
    return scores, dp, specificity, kl


def explain(model, stack, pair=None, patch=6):
    stack = np.asarray(stack, dtype=np.float32)
    before = predict(model, stack[None])[0]
    action, alternative = action_pair(before) if pair is None else pair
    base = replacement(stack, "blur")
    cells = grid(*stack.shape[-2:], patch)

    def target(x):
        q = model(x)
        return q[:, action] - q[:, alternative]

    started = time.perf_counter()
    x, baseline = torch.tensor(stack[None]), torch.tensor(base[None])
    with torch.no_grad():
        difference = float((target(x) - target(baseline)).item())
    tolerance = max(1e-4, .02 * abs(difference))
    for steps in [64, 256, 1024, 4096]:
        attribution, delta = IntegratedGradients(target).attribute(
            x, baselines=baseline, n_steps=steps, method="gausslegendre",
            internal_batch_size=16, return_convergence_delta=True)
        if abs(delta.item()) <= tolerance:
            break
    ig = attribution.detach().cpu().numpy()[0].sum(0)
    ig_seconds = time.perf_counter() - started

    started = time.perf_counter()
    batch = np.repeat(stack[None], len(cells), axis=0)
    for i, (y0, y1, x0, x1) in enumerate(cells):
        batch[i, :, y0:y1, x0:x1] = base[:, y0:y1, x0:x1]
    after = predict(model, batch)
    perturb_seconds = time.perf_counter() - started
    started = time.perf_counter()
    drops = before[action] - before[alternative] - (after[:, action] - after[:, alternative])
    occlusion_seconds = perturb_seconds + time.perf_counter() - started
    started = time.perf_counter()
    scores, dp, specificity, kl = sarfa_components(before, after, action)
    sarfa_seconds = perturb_seconds + time.perf_counter() - started
    maps = {"ig": ig}
    for name, values in [("occlusion", drops), ("sarfa", scores)]:
        heat = np.zeros(stack.shape[-2:], dtype=np.float64)
        for (y0, y1, x0, x1), value in zip(cells, values):
            heat[y0:y1, x0:x1] = value
        maps[name] = heat
    finite = {name: bool(np.isfinite(heat).all()) for name, heat in maps.items()}
    if not all(finite.values()):
        raise ValueError(f"Non-finite explanation: {finite}")
    info = {
        "action": action, "alternative": alternative, "finite": finite,
        "seconds": {"ig": ig_seconds, "occlusion": occlusion_seconds, "sarfa": sarfa_seconds},
        "shared_perturbation_seconds": perturb_seconds,
        "ig": {"steps": steps, "output_difference": difference,
               "attribution_sum": float(ig.sum()), "residual": float(delta.item()),
               "tolerance": tolerance, "converged": bool(abs(delta.item()) <= tolerance)},
    }
    details = {"before": before, "after": after, "sarfa": scores,
               "probability_drop": dp, "specificity": specificity, "kl": kl,
               "contrastive_drop": drops}
    return maps, info, details


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def randomized_copy(model, seed):
    changed = copy.deepcopy(model)
    layer = [m for m in changed.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layer.reset_parameters()
    return changed.eval()
