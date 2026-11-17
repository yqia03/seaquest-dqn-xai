"""Common-grid deletion and insertion with a fixed original action."""
import time

import numpy as np

from explanations import predict


def metrics(q, action):
    q = np.asarray(q)
    others = np.arange(q.shape[-1]) != action
    return {"margin": q[..., action] - q[..., others].max(axis=-1),
            "centered": q[..., action] - q.mean(axis=-1),
            "flipped": q.argmax(axis=-1) != action}


def ranked_cells(heat, cells):
    # Signed mean importance; row-major cell index breaks equal scores.
    scores = [heat[y0:y1, x0:x1].mean() for y0, y1, x0, x1 in cells]
    return np.argsort(-np.asarray(scores), kind="stable")


def intervention_path(stack, base, cells, order, counts, direction):
    if direction not in ["deletion", "insertion"]:
        raise ValueError("Unknown direction")
    if sorted(order) != list(range(len(cells))):
        raise ValueError("Order must be a permutation of grid cells")
    if counts[0] != 0 or counts[-1] != len(cells) or np.any(np.diff(counts) <= 0):
        raise ValueError("Counts must increase from zero to the complete grid")
    current = (stack if direction == "deletion" else base).copy()
    source = base if direction == "deletion" else stack
    images, done = [], 0
    for count in counts:
        for cell_index in order[done:count]:
            y0, y1, x0, x1 = cells[cell_index]
            current[:, y0:y1, x0:x1] = source[:, y0:y1, x0:x1]
        images.append(current.copy())
        done = count
    return np.stack(images)


def evaluate(model, stack, base, cells, order, counts, direction, action):
    started = time.perf_counter()
    path = intervention_path(stack, base, cells, order, counts, direction)
    q = predict(model, path)
    expected = predict(model, np.stack([stack, base]))
    endpoints = expected if direction == "deletion" else expected[::-1]
    np.testing.assert_allclose(q[[0, -1]], endpoints, rtol=1e-5, atol=1e-5)
    values = metrics(q, action)
    progress = np.asarray(counts) / len(cells)
    summary = {f"{name}_auc": float(np.trapezoid(values[name], progress))
               for name in ["margin", "centered"]}
    summary.update(flip_rate=float(values["flipped"].mean()),
                   flipped_fraction_auc=float(np.trapezoid(values["flipped"].astype(float), progress)),
                   seconds=time.perf_counter() - started)
    return q, values, summary
