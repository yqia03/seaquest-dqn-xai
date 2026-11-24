# Does the Agent Watch Its Oxygen?

Does a pretrained Atari Seaquest agent use the on-screen oxygen gauge when it chooses an action? We compared three explanation methods and measured how masking the gauge changed the agent’s action scores. **In a 20-game follow-up, switching from black fill to blur reversed the gauge-versus-control result at every oxygen level.**

## Demo

https://github.com/user-attachments/assets/14cc3317-741b-406c-9116-0f3bc0e738cd

[Download the 58.8-second MP4](assets/demo.mp4?raw=true) · [Silent preview](assets/demo-preview.gif) · [Report (PDF)](report/Group65-Seaquest.pdf) · [Portable notebook](report/Group65-Seaquest.ipynb)

## What we found

- **120 decisions for method selection.** Across 10 games, we compared Integrated Gradients, contrastive occlusion and SARFA with a common-grid deletion and insertion benchmark. No method consistently outperformed the others under the predefined criteria, so we retained all three.
- **0 of 300 planned holdout decisions.** All eligible four-frame inputs from the original 20 analysis games had already appeared in the selection trajectories. The input audit caught this overlap even though the game seeds were separate. The original holdout yielded no estimate of oxygen-gauge reliance.
- **300 distinct decisions in the follow-up.** With a 25% chance per emulator frame of repeating the previous action, 20 new games yielded 100 decisions at each of three oxygen levels. All 300 four-frame inputs were distinct and absent from the original and pilot trajectories.

## Black fill and blur reversed the comparison

On each decision, we masked the oxygen gauge and an equal-area neighboring HUD region separately across all four input frames. We measured the drop in the original action’s Q-value lead over the strongest alternative, then subtracted the HUD drop from the gauge drop. Each entry averages the paired differences across the 20 games, giving each game equal weight.

| Oxygen level | Decisions | Black fill (Q) | Blur (Q) |
| --- | ---: | ---: | ---: |
| Low | 100 | −0.386 | +0.037 |
| Medium | 100 | −0.814 | +0.039 |
| High | 100 | −1.125 | +0.042 |

Blackening the gauge weakened support for the original action less than blackening the neighboring HUD. Blurring the gauge weakened it more. The interventions changed the pixels presented to the network while the game’s oxygen and dynamics stayed fixed. The conclusions apply to immediate action scores in the sticky-action setting. Two exploratory life-loss cases had different oxygen levels before the loss, with the cause unresolved in both.

## Technical approach

We used the public [Stable-Baselines3 Seaquest DQN checkpoint](https://huggingface.co/sb3/dqn-SeaquestNoFrameskip-v4) with frozen weights. The policy chose from 18 actions using stacks of four 84 × 84 grayscale frames. Emulator RAM supplied oxygen levels and life-loss markers for analysis. We hashed the exact frame stacks to check data overlap, replayed recorded decisions to verify Q-values and used whole-game resampling to preserve within-game dependence.

## Reproduce the results

To read all three analyses, open the [PDF report](report/Group65-Seaquest.pdf). To regenerate the tables and figures, open the [notebook](report/Group65-Seaquest.ipynb) in Jupyter with NumPy 2+, Matplotlib, Pillow and IPython, restart the kernel and run all cells. The notebook includes the required observations and model outputs.

For the six gauge-versus-HUD values above and the twelve additional control comparisons, run the following commands from the repository root. The script resamples the 20 games and checks all 18 point estimates and intervals against the [saved results](results/sticky_published_estimates.csv).

```sh
python -m pip install 'numpy>=2'
python analysis/oxygen_contrasts.py
```

The [explanation methods](xai/explanations.py) and [faithfulness metrics](xai/faithfulness.py) can also be tested independently.

```sh
python -m pip install -r xai/requirements.txt
python -m unittest discover -s xai -v
```

## Team

This was a joint CS5495 course project by **QIAN Yuwei** and **JIN Yupeng**.

- **QIAN Yuwei** developed the original collection, replay and input-isolation pipeline, analyzed the revisited states, cross-checked the follow-up cohort, assembled the final report and produced the animated presentation cut.
- **JIN Yupeng** built the explanation benchmark, collected and analyzed the sticky-action cohort, developed the life-loss case comparisons, independently verified results and recorded the original presentation and narration.

OpenAI Codex assisted Qian’s work on implementation, analysis, report production and presentation animation.
