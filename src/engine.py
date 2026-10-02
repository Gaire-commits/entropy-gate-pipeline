"""The decision engine: a policy that learns, from realized rewards, when to trade.

Inputs are the saved walk-forward predictions, one row per sample, so every
model's output and the entropy gate's verdict are available as state. The policy
picks an action per sample (short, flat, long) and is paid

    reward = position * forward_return_bps - cost_bps * |position|

Flat pays exactly zero. That is the reference to beat: a policy that trades only
loses money to costs unless it has something better than "always up".

Learning is online and fold by fold. Before scoring fold k the policy is updated
on the feedback from folds < k only, which is what it would really have seen. The
update is REINFORCE with a baseline on bandit feedback: the policy samples an
action, observes only that action's reward, and moves toward actions that beat
its running average. Prices do not react to our trades, so any action's reward
for a historical sample is known and a sampled action can be scored by lookup;
that makes replaying the logged data a faithful simulator, not an approximation.

The same fact allows full feedback (`feedback="full"`): every past sample scores
all three actions, including the trades the policy did not take, so a missed
opportunity is as visible as a losing trade. Bandit feedback cannot see one:
flat pays zero and says nothing about what trading would have paid. In a
backtest with a flat cost the missed rewards are exact; live, the fill a skipped
trade would have got is unknown, which is when bandit feedback is the honest one.

What this is not: a sequential agent with inventory. Each sample is decided on
its own, so the problem is a contextual bandit (RL with a one-step horizon).
Position-dependent costs, holding limits and risk budgets would need a stepwise
environment, and are the natural next extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .baselines import is_rule
from .experiment import KEY, ensemble

ACTIONS = np.array([-1.0, 0.0, 1.0])  # short, flat, long
GATE_FLAGS = ["pass_trend", "pass_entropy", "pass_entropy_weighted"]
FIXED_ACTIONS = ("long", "short", "follow signal", "fade signal")


# ------------------------------------------------------------------ state

def build_state(pred: pd.DataFrame, archs: list[str] | None = None, use_gate: bool = True) -> pd.DataFrame:
    """One row per sample with the engine's inputs and the realized return.

    Each model contributes its signed confidence, 2*(p-0.5) in [-1, 1], averaged
    over seeds. The gate contributes its three pass flags and the trend p-value
    (1.0 when there is no reading yet). Nothing here looks at the realized
    return; `ret_bps` and `y` ride along for scoring only.
    """
    archs = archs or sorted(pred["arch"].unique())
    base = None
    for arch in archs:
        sub = ensemble(pred[pred["arch"] == arch])
        sub[f"conf_{arch}"] = 2 * (sub["prob"] - 0.5)
        keep = KEY + [f"conf_{arch}"]
        if base is None:
            base = sub[KEY + ["y", "ret"] + [c for c in GATE_FLAGS + ["p_trend", "has_reading"] if c in sub.columns]
                        + [f"conf_{arch}"]]
        else:
            base = base.merge(sub[keep], on=KEY, how="inner")
    base = base.sort_values(["date", "symbol", "bar_index"]).reset_index(drop=True)
    base["ret_bps"] = base["ret"] * 1e4
    base["slot"] = base["bar_index"] / 78.0
    if use_gate and "pass_trend" in base.columns:
        for col in GATE_FLAGS:
            base[col] = base[col].astype(float)
        base["p_trend"] = base["p_trend"].fillna(1.0)
        base["has_reading"] = base["has_reading"].astype(float)
    return base


def market_neutral(state: pd.DataFrame, min_peers: int = 20) -> pd.DataFrame:
    """Express returns and model confidences relative to the other symbols at the same moment.

    Every symbol sampled at the same date and bar shares that moment's market move. In a
    quarter where the market drifts up, "long" pays on every sample, so one quarter's drift
    looks like tens of thousands of confirmations of a rule that will not hold next quarter
    (the engine collapsed into always-long on the S&P 500 this way). Subtracting the
    cross-sectional mean of the forward return (what the engine is paid) and of each model's
    confidence (what it sees) leaves what is specific to the stock, a long-short book.

    The inputs stay causal: the demeaned confidences use only model outputs, which are all
    known at decision time. The demeaned return is a reward, never an input. Moments with
    fewer than `min_peers` symbols are dropped, since a mean over three stocks is not a market.
    Costs are charged on the stock leg only; hedging the market leg would cost extra, so the
    result is optimistic by that amount.
    """
    keys = ["date", "bar_index"]
    peers = state.groupby(keys)["ret_bps"].transform("size")
    if peers.median() < min_peers:
        raise ValueError(f"market-neutral needs a wide cross-section: the median date and bar has "
                         f"{peers.median():.0f} symbols, at least {min_peers} are required")
    out = state[peers >= min_peers].copy()
    for col in ["ret_bps"] + [c for c in out.columns if c.startswith("conf_")]:
        out[col] -= out.groupby(keys)[col].transform("mean")
    return out.reset_index(drop=True)


def feature_columns(state: pd.DataFrame, use_gate: bool = True) -> list[str]:
    cols = [c for c in state.columns if c.startswith("conf_")] + ["slot"]
    if use_gate:
        cols += [c for c in GATE_FLAGS + ["p_trend", "has_reading"] if c in state.columns]
    return cols


# ------------------------------------------------------------------ policy

class Policy(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        with torch.no_grad():
            self.net[2].bias.zero_()  # neutral start: a flat-leaning prior saturates the softmax and stops learning

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EngineConfig:
    cost_bps: float = 2.0
    hidden: int = 32
    lr: float = 1e-2
    epochs: int = 15             # passes over each newly revealed fold
    batch: int = 2048
    entropy_bonus: float = 0.10  # keeps the softmax from saturating on 'flat'; decays with folds
    reward_scale: float = 10.0   # bps -> network units
    feedback: str = "bandit"     # bandit: only the sampled action's reward; full: all three, trades not taken included
    seed: int = 0


def action_rewards(ret_bps: np.ndarray, cost_bps: float) -> np.ndarray:
    """(n, 3) reward of short, flat, long for each sample."""
    return np.column_stack([-ret_bps - cost_bps, np.zeros_like(ret_bps), ret_bps - cost_bps])


def learn_from(policy: Policy, opt, X: np.ndarray, rewards: np.ndarray, cfg: EngineConfig,
               baseline: float, rng: np.random.Generator, bonus: float) -> float:
    """One round of policy-gradient updates on newly revealed feedback. Returns the baseline.

    bandit: REINFORCE on the sampled action, advantage = reward - running baseline,
    scaled by its batch standard deviation so the step size does not depend on how
    noisy returns are. full: gradient of expected reward under the policy, using
    all three action rewards (the logged data allows it because prices do not react
    to our trades); lower variance, but it uses feedback the agent never "tried".
    """
    Xt = torch.from_numpy(X.astype(np.float32))
    R = torch.from_numpy(rewards.astype(np.float32)) / cfg.reward_scale
    for _ in range(cfg.epochs):
        order = rng.permutation(len(X))
        for s in range(0, len(order), cfg.batch):
            idx = torch.from_numpy(order[s : s + cfg.batch])
            dist = torch.distributions.Categorical(logits=policy(Xt[idx]))
            if cfg.feedback == "full":
                loss = -(dist.probs * R[idx]).sum(dim=1).mean()
            else:
                a = dist.sample()
                r = R[idx][torch.arange(len(idx)), a]
                baseline = 0.95 * baseline + 0.05 * float(r.mean())
                adv = (r - baseline) / (r.std() + 1e-6)
                loss = -(dist.log_prob(a) * adv).mean()
            loss = loss - bonus * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return baseline


@torch.no_grad()
def decide(policy: Policy, X: np.ndarray) -> np.ndarray:
    """Greedy position (-1, 0, +1) for each sample."""
    logits = policy(torch.from_numpy(X.astype(np.float32)))
    return ACTIONS[logits.argmax(dim=1).numpy()]


# ------------------------------------------------------------------ walk-forward

def run_engine(state: pd.DataFrame, cfg: EngineConfig, use_gate: bool = True, log=print) -> pd.DataFrame:
    """Online walk-forward. Adds `pos` and `pnl_bps` (net of cost) to every scored sample.

    The first fold has nothing to learn from, so the policy stays out and earns
    zero there; scoring starts from fold 1. Rows keep the state's index.
    """
    if cfg.feedback not in ("bandit", "full"):
        raise ValueError(f"unknown feedback '{cfg.feedback}'; options: bandit, full")
    torch.manual_seed(cfg.seed)
    cols = feature_columns(state, use_gate)
    X_all = state[cols].to_numpy(float)
    rewards = action_rewards(state["ret_bps"].to_numpy(), cfg.cost_bps)
    folds = np.sort(state["fold"].unique())

    policy = Policy(len(cols), cfg.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    baseline = 0.0
    out = []
    for k, fold in enumerate(folds):
        rows = np.flatnonzero(state["fold"].to_numpy() == fold)
        if k > 0:
            prev = np.flatnonzero(state["fold"].to_numpy() == folds[k - 1])
            bonus = cfg.entropy_bonus / (1 + k)
            baseline = learn_from(policy, opt, X_all[prev], rewards[prev], cfg, baseline, rng, bonus)
        pos = decide(policy, X_all[rows]) if k > 0 else np.zeros(len(rows))
        scored = state.iloc[rows][["fold", "date", "symbol", "ret_bps", "y"]].copy()
        scored["pos"] = pos
        scored["pnl_bps"] = pos * scored["ret_bps"].to_numpy() - cfg.cost_bps * np.abs(pos)
        out.append(scored)
        log(f"  fold {fold:3d}  trades {np.mean(pos != 0):6.1%}  net {scored['pnl_bps'].mean():+6.2f} bps/opportunity")
    return pd.concat(out)


# ------------------------------------------------------------------ baselines and scoring

def baseline_positions(state: pd.DataFrame, arch: str | None = None) -> dict[str, np.ndarray]:
    """Fixed rules, no learning: hold long, or trade every model signal."""
    pos = {"always_up": np.ones(len(state))}
    for c in [c for c in state.columns if c.startswith("conf_")]:
        pos[f"follow_{c[5:]}"] = np.sign(state[c].to_numpy())
    return pos


def threshold_rule(state: pd.DataFrame, cost_bps: float, grid=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7)) -> pd.DataFrame:
    """The strong supervised baseline: each fold, pick the (model, confidence cutoff)
    that earned the most net on earlier folds, then trade only above it.

    This is what "just filter by confidence" looks like when the cutoff has to be
    chosen from the past. The engine has to beat this to justify itself.
    """
    confs = [c for c in state.columns if c.startswith("conf_")]
    folds = np.sort(state["fold"].unique())
    out = []
    for k, fold in enumerate(folds):
        rows = state[state["fold"] == fold]
        if k == 0:
            pos = np.zeros(len(rows))
        else:
            past = state[state["fold"] < fold]
            best, best_net = (confs[0], 1.1), -np.inf
            for c in confs:
                for t in grid:
                    p = np.sign(past[c]) * (past[c].abs() > t)
                    net = float((p * past["ret_bps"] - cost_bps * p.abs()).mean())
                    if net > best_net:
                        best, best_net = (c, t), net
            c, t = best
            pos = (np.sign(rows[c]) * (rows[c].abs() > t)).to_numpy()
            if best_net <= 0:
                pos = np.zeros(len(rows))
        scored = rows[["fold", "date", "symbol", "ret_bps", "y"]].copy()
        scored["pos"] = pos
        scored["pnl_bps"] = pos * scored["ret_bps"].to_numpy() - cost_bps * np.abs(pos)
        out.append(scored)
    return pd.concat(out, ignore_index=True)


def score_positions(frame: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Net bps per opportunity (flat samples count as zero) with a day-block bootstrap interval."""
    days = np.unique(frame["date"])
    by_day = frame.groupby("date").agg(pnl=("pnl_bps", "sum"), n=("pnl_bps", "size"), trades=("pos", lambda p: (p != 0).sum()))
    by_day = by_day.reindex(days, fill_value=0)
    pnl, n = by_day["pnl"].to_numpy(float), by_day["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), (n_boot, len(days)))
    boot = pnl[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1)
    return {
        "net_bps": float(pnl.sum() / n.sum()),
        "net_lo": float(np.percentile(boot, 2.5)),
        "net_hi": float(np.percentile(boot, 97.5)),
        "trade_rate": float(by_day["trades"].sum() / n.sum()),
        "samples": int(n.sum()),
    }


# ------------------------------------------------------------------ missed opportunities

def model_signal(state: pd.DataFrame, arch: str | None = None) -> np.ndarray:
    """The signal situations are graded by: one model's signed confidence, or by default
    the average over the trained models. Rules are left out unless nothing else is there;
    their 'confidence' is a fixed +-1."""
    if arch is not None:
        return state[f"conf_{arch}"].to_numpy(float)
    confs = [c for c in state.columns if c.startswith("conf_")]
    trained = [c for c in confs if not is_rule(c[5:])] or confs
    return state[trained].to_numpy(float).mean(axis=1)


def situations(state: pd.DataFrame, signal: np.ndarray, gate_flag: str = "pass_trend",
               n_strength: int = 5) -> pd.DataFrame:
    """Gate verdict and signal-strength quintile (1 = weakest) for every row."""
    if gate_flag in state.columns:
        gate = np.where(state[gate_flag].to_numpy(float) > 0, "passed", "not passed")
    else:
        gate = np.full(len(state), "-")
    rank = pd.Series(np.abs(signal)).rank(method="first")
    strength = pd.qcut(rank, n_strength, labels=False).to_numpy() + 1
    return pd.DataFrame({"gate": gate, "strength": strength})


def missed_opportunities(state: pd.DataFrame, engines: dict[str, list[np.ndarray]], cost_bps: float,
                         signal: np.ndarray, gate_flag: str = "pass_trend", n_strength: int = 5,
                         n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Where trading paid on average while an engine stayed out or traded the wrong way.

    `state` holds the scored rows and `engines` maps a name to that engine's positions
    on them, one array per training seed. Rows are grouped by situation: the gate's
    verdict and the strength of the model signal. In each group four fixed actions are
    scored on the realized returns (long, short, follow the signal, fade the signal);
    the best of them, or staying flat if none pays after cost, is what a policy would
    have earned had it known which action suits that kind of situation, though nothing
    about any single sample. `left_<name>` is that value minus what the engine earned
    there, per opportunity.

    The best action is picked after seeing the outcomes, which flatters it, so a group
    is a real opportunity only when the best action's interval is above zero, and with
    ten groups one can clear that bar by luck.
    """
    ret = state["ret_bps"].to_numpy(float)
    side = np.sign(signal)
    fixed = {"long": np.ones(len(ret)), "short": -np.ones(len(ret)), "follow signal": side, "fade signal": -side}
    labels = situations(state, signal, gate_flag, n_strength)
    rows = []
    for (gate, strength), idx in labels.groupby(["gate", "strength"]).indices.items():
        nets = {a: float(np.mean(p[idx] * ret[idx] - cost_bps * np.abs(p[idx]))) for a, p in fixed.items()}
        best = max(nets, key=nets.get)
        row = {"gate": gate, "strength": int(strength), "samples": len(idx)}
        if nets[best] > 0:
            pos = fixed[best][idx]
            frame = pd.DataFrame({"date": state["date"].to_numpy()[idx], "pos": pos,
                                  "pnl_bps": pos * ret[idx] - cost_bps * np.abs(pos)})
            s = score_positions(frame, n_boot=n_boot, seed=seed)
            row.update(best_action=best, best_net=s["net_bps"], best_lo=s["net_lo"], best_hi=s["net_hi"])
        else:
            row.update(best_action="stay flat", best_net=0.0, best_lo=np.nan, best_hi=np.nan)
        for name, runs in engines.items():
            pos = [np.asarray(p, float)[idx] for p in runs]
            row[f"trades_{name}"] = float(np.mean([(p != 0).mean() for p in pos]))
            row[f"net_{name}"] = float(np.mean([np.mean(p * ret[idx] - cost_bps * np.abs(p)) for p in pos]))
            row[f"left_{name}"] = row["best_net"] - row[f"net_{name}"]
        rows.append(row)
    order = {"passed": 0, "not passed": 1, "-": 2}
    table = pd.DataFrame(rows)
    return table.sort_values(["gate", "strength"], key=lambda c: c.map(order) if c.name == "gate" else c,
                             ignore_index=True)
