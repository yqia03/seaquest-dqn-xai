"""Rebuild the 18 sticky-action gauge-minus-control contrasts from per-game means."""
import csv
from pathlib import Path

import numpy as np


RESULTS = Path(__file__).resolve().parents[1] / "results"
STRATA = ("low", "medium", "high")
FILLS = ("black", "blur")
CONTROLS = ("neighboring_hud", "score", "playfield")


def read_csv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    rows = read_csv("sticky_game_contrasts.csv")
    by_game = {(row["stratum"], row["game_id"]): row for row in rows}
    games = sorted({row["game_id"] for row in rows})
    expected_cells = {(s, g) for s in STRATA for g in games}
    if (len(games) != 20 or len(rows) != 60 or set(by_game) != expected_cells
            or any(int(row["states"]) != 5 for row in rows)):
        raise ValueError("Expected five decisions per oxygen stratum in each of twenty games")
    published = {(row["stratum"], row["fill"], row["control"]): row
                 for row in read_csv("sticky_published_estimates.csv")}
    # Each draw reuses whole game IDs across strata, fills and controls.
    draws = np.random.default_rng(5495).integers(len(games), size=(20000, len(games)))
    print(f"{'Oxygen':<7} {'Fill':<6} {'Control':<16} {'Mean':>8} {'95% interval':>21}")
    for stratum in STRATA:
        for fill in FILLS:
            for control in CONTROLS:
                column = f"{fill}_oxygen_minus_{control}"
                values = np.array([float(by_game[stratum, game][column]) for game in games])
                mean = values.mean()
                lower, upper = np.quantile(values[draws].mean(axis=1), [.025, .975])
                ref = published[stratum, fill, control]
                np.testing.assert_allclose(
                    [mean, lower, upper], [float(ref[k]) for k in ("estimate", "lo95", "hi95")],
                    rtol=0, atol=1e-10,
                )
                print(f"{stratum:<7} {fill:<6} {control:<16} {mean:+8.4f}   [{lower:+.4f}, {upper:+.4f}]")


if __name__ == "__main__":
    main()
