"""Tune rating parameters on seasons 2005-2014 only; validate untouched on 2015-2025.
Score = log loss of a small multinomial logistic model on rating differences (fitted on the
tuning period, applied unchanged to validation).
Stage 1: goal/result Elo - update speed k, season reset strength and target, early-season speed-up.
Stage 2: separate shot rating (learns from shots-on-target share only), judged by how much it adds
         on top of the stage-1 rating, on matches that have shot data.
Writes reports/ratings_tuning.md and data/processed/rating_params.json
"""
import itertools, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent))
from ratings import compute_ratings  # noqa: E402
from features import COUNTRY  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TUNE, VALID = (2005, 2014), (2015, 2025)
V1 = dict(k=20.0, regress=0.2, k_early=1.0, regress_to="global")
SHOT_FIXED = dict(w_shots=1.0, gd_mult=False, hfa=40.0)

df = pd.read_parquet(ROOT / "data/processed/matches.parquet")
ctry = df.league.map(COUNTRY).fillna(df.league)
df["home_id"] = ctry + ":" + df.home
df["away_id"] = ctry + ":" + df.away
y = df.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()
season = df.season.to_numpy()
SHOTS = df.hst.notna().to_numpy() & df.ast.notna().to_numpy()
T = (season >= TUNE[0]) & (season <= TUNE[1])
V = (season >= VALID[0]) & (season <= VALID[1])


def diff_feats(r):
    d = (r[0] - r[1]) / 100.0
    return [d, d ** 2]


def score(X, ih, ia, subset=None):
    sub = np.ones(len(y), bool) if subset is None else subset
    t, v = T & sub, V & sub
    p = LogisticRegression(max_iter=1000).fit(X[t], y[t]).predict_proba(X)
    early = v & (ih < 10) & (ia < 10)
    return {name: log_loss(y[m], p[m], labels=[0, 1, 2])
            for name, m in [("tune", t), ("valid", v), ("valid_early", early), ("valid_late", v & ~early)]}


def eval_goal(params):
    r = compute_ratings(df, **params)
    return {**params, **score(np.column_stack(diff_feats(r)), r[2], r[3])}


_goal = None


def eval_shot(params):
    s = compute_ratings(df, **SHOT_FIXED, **params)
    X = np.column_stack(diff_feats(_goal) + diff_feats(s))
    return {**params, **score(X, _goal[2], _goal[3], SHOTS)}


def init_shot(goal_params):
    global _goal
    _goal = compute_ratings(df, **goal_params)


if __name__ == "__main__":
    g_grid = [dict(k=k, regress=rg, k_early=ke, regress_to=to) for k, rg, ke, to in itertools.product(
        [10.0, 15.0, 20.0], [0.0, 0.1, 0.2, 0.33, 0.5], [1.0, 1.5, 2.0], ["global", "league"])]
    with Pool(4) as pool:
        G = pd.DataFrame(pool.map(eval_goal, g_grid))
    best_g = G.sort_values("tune").iloc[0]
    goal_params = {c: best_g[c] for c in ["k", "regress", "k_early", "regress_to"]}
    v1 = G[(G.k == 20) & (G.regress == 0.2) & (G.k_early == 1) & (G.regress_to == "global")].iloc[0]

    s_grid = [dict(k=k, regress=rg, k_early=ke, regress_to="league")
              for k, rg, ke in itertools.product([30.0, 60.0, 100.0], [0.0, 0.2, 0.5], [1.0, 2.0])]
    with Pool(4, initializer=init_shot, initargs=(goal_params,)) as pool:
        S = pd.DataFrame(pool.map(eval_shot, s_grid))
    best_s = S.sort_values("tune").iloc[0]
    init_shot(goal_params)
    goal_only_shots = score(np.column_stack(diff_feats(_goal)), _goal[2], _goal[3], SHOTS)
    r1 = compute_ratings(df, **V1)
    v1_shots = score(np.column_stack(diff_feats(r1)), r1[2], r1[3], SHOTS)

    cols = ["tune", "valid", "valid_early", "valid_late"]
    md = ["# Rating tuning\n",
          f"Parameters chosen on seasons {TUNE[0]}-{TUNE[1]} (`tune`) only; `valid*` = seasons {VALID[0]}-{VALID[1]}, "
          "untouched during selection. Log loss of a rating-only model, lower is better. "
          "`valid_early` = both teams within their first 10 matches of the season.\n",
          "## Stage 1: season reset and early-season speed-up (all matches)\n",
          pd.DataFrame([v1, best_g], index=["v1 (global reset 20%, no speed-up)", "tuned"])[
              ["k", "regress", "k_early", "regress_to"] + cols].round(4).to_markdown(), "\n",
          "### Reset strength x target (best k and speed-up per row)\n",
          G.loc[G.groupby(["regress_to", "regress"]).tune.idxmin()][["regress_to", "regress", "k", "k_early"] + cols]
          .round(4).to_markdown(index=False), "\n",
          "## Stage 2: separate shot rating on top of the tuned rating (matches with shot data)\n",
          pd.DataFrame([v1_shots, goal_only_shots, best_s[cols].to_dict()],
                       index=["v1 rating", "tuned rating", "tuned rating + shot rating"])[cols].round(4).to_markdown(), "\n",
          f"Shot rating parameters: k={best_s.k}, reset={best_s.regress} toward league mean, early speed-up={best_s.k_early}.\n",
          "\n### All shot-rating settings\n", S.sort_values("tune")[["k", "regress", "k_early"] + cols].round(4).to_markdown(index=False), "\n"]
    (ROOT / "reports/ratings_tuning.md").write_text("\n".join(md))
    params = {"goal": {**{k: float(goal_params[k]) for k in ["k", "regress", "k_early"]}, "regress_to": goal_params["regress_to"]},
              "shot": {**SHOT_FIXED, **{k: float(best_s[k]) for k in ["k", "regress", "k_early"]}, "regress_to": "league"}}
    (ROOT / "data/processed/rating_params.json").write_text(json.dumps(params, indent=2))
    print("\n".join(md)); print(json.dumps(params, indent=2))
