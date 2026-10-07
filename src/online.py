"""An online agent: learns from a stream of bars, one bar at a time, and keeps exploring on purpose.

Everything the walk-forward pipeline does in batch happens here as the bars arrive:

1. `StreamFeaturizer` sees each bar once, in time order, and keeps only short ring buffers. At each
   signal bar it cuts the same windows `dataset.build_dataset` would (same channels, same session
   rules); the realized return is attached only when the exit bar arrives, 31 bars later. A replay
   test checks that features and returns equal the batch pipeline's sample for sample, so the
   stream can neither see the future nor quietly differ from what the offline models saw.
2. `OnlineAgent` picks short / flat / long per stock and learns only from outcomes that have matured.
   Rewards are market-neutral by default (each stock's return minus the cross-sectional average at
   that moment), net of cost per position.
3. Exploration is controlled through the policy's entropy H(pi) = -sum pi log pi, not a fixed bonus:
   the bonus weight beta adapts so the policy's entropy tracks a target (beta rises when the policy
   gets too sure of itself, falls when it is too random), and a Page-Hinkley drift detector raises the
   target for a while when the greedy policy's results drop, so the agent explores again after the
   market changes. Without that, an agent paid only for the actions it takes ("bandit" feedback)
   can settle on "flat", which pays exactly zero and teaches nothing, and never notice a new signal.

Two kinds of feedback, as in `engine.py`: `full` scores all three actions on every matured sample
(prices do not react to our trades, so what a skipped trade would have paid is known from the bars,
ignoring fills); `bandit` learns only from the action actually taken, which is the honest one live.
With full feedback exploration costs nothing for information; entropy then only keeps the policy
from locking in. Positions here are paper positions: nothing in this module places orders.

`run_agent` drives an agent through recorded events in time order (outcomes of a bar before
decisions at that bar, both known at its close), so the same event list can be replayed with the
returns shuffled across stocks within each moment: a placebo world with the same features,
volatility and timing but nothing to learn. If the agent "earns" there as well, its learning is fake.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .engine import ACTIONS, Policy, action_rewards
from .ml import window_summary

CHANNELS = ("log_return", "session_return", "hl_range", "close_open", "volume_z", "signed_volume")
VOLUME_LOOKBACK = 78     # features._volume_z


# ------------------------------------------------------------------ the stream

@dataclass
class Moment:
    """One signal bar: the stocks with a complete window, their inputs, and (once matured) their returns."""
    mid: int
    date: np.datetime64
    bar_index: int
    symbols: np.ndarray
    F: np.ndarray                       # (n, d) agent inputs, known at the signal bar's close
    X: np.ndarray | None = None         # (n, channels, window) raw windows, kept for the replay check
    ret: np.ndarray | None = None       # (n,) log return entry -> exit, set at maturity only
    entry_close: np.ndarray | None = None
    exit_offset: int = 0
    entry_offset: int = 0


def panel_days(bars: dict[str, pd.DataFrame]):
    """Yield (date, fields, slots) per trading day across the universe, in date order.

    fields[name] is (max_slots, n_symbols) with NaN where a symbol has no bar; slots[s] is the
    number of grid bars symbol s has that day (0 if none, 39 on an early close). Bars come from
    `data.to_session_grid`, as everywhere else. Built a day at a time so a 500-stock history
    never has to sit in memory as one panel.
    """
    symbols = list(bars)
    per = []
    for sym in symbols:
        df = bars[sym]
        ts = df.index.get_level_values("timestamp")
        day = ts.tz_localize(None).normalize() if ts.tz is not None else ts.normalize()
        day = day.values.astype("datetime64[D]")
        starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
        per.append({
            "days": day[starts], "starts": starts, "ends": np.r_[starts[1:], len(day)],
            **{k: df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")},
        })
    all_days = np.unique(np.concatenate([p["days"] for p in per])) if per else np.array([], "datetime64[D]")
    cursor = np.zeros(len(symbols), dtype=int)
    for d in all_days:
        slots = np.zeros(len(symbols), dtype=int)
        rows = []
        for s, p in enumerate(per):
            c = cursor[s]
            if c < len(p["days"]) and p["days"][c] == d:
                a, b = p["starts"][c], p["ends"][c]
                slots[s] = b - a
                rows.append((s, a, b))
                cursor[s] += 1
        width = int(slots.max()) if len(slots) else 0
        fields = {k: np.full((width, len(symbols)), np.nan) for k in ("open", "high", "low", "close", "volume")}
        for s, a, b in rows:
            for k in fields:
                fields[k][: b - a, s] = per[s][k][a:b]
        yield d, fields, slots


class RunningStandardizer:
    """Exponentially weighted mean and variance per feature. `transform` uses only what came before."""

    def __init__(self, d: int, halflife: float = 500.0):
        self.alpha = 1 - 0.5 ** (1 / halflife)
        self.mean, self.var, self.n = np.zeros(d), np.ones(d), 0

    def transform(self, F: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return np.zeros_like(F)
        return np.clip((F - self.mean) / np.sqrt(np.maximum(self.var, 1e-12)), -5, 5)

    def update(self, F: np.ndarray) -> None:
        m, v = F.mean(axis=0), F.var(axis=0)
        if self.n == 0:
            self.mean, self.var = m, np.where(v > 0, v, 1.0)
        else:
            a = self.alpha
            delta = m - self.mean
            self.mean = self.mean + a * delta
            self.var = (1 - a) * (self.var + a * delta**2) + a * v
        self.n += 1


def cross_section_z(F: np.ndarray) -> np.ndarray:
    """Each feature standardized across the stocks at one moment, clipped to +-5."""
    sd = F.std(axis=0)
    return np.clip((F - F.mean(axis=0)) / np.where(sd > 0, sd, 1.0), -5, 5)


class StreamFeaturizer:
    """Cuts the batch pipeline's windows from bars that arrive one at a time.

    Per symbol it keeps the last `window + 78` grid rows (enough for the first window bar's
    rolling volume statistics and its previous close) and the previous session's last close.
    Nothing later than the current bar is ever stored. Decisions are emitted at the same signal
    bars as `features.make_windows` (window start on a multiple of `stride` from the open), only
    when the whole span to the exit fits in that symbol's session, which the trading calendar
    (early closes) tells in advance.

    `normalize`: `cross_section` standardizes each input across the stocks at that moment (needs
    `min_peers` stocks), `running` uses exponentially weighted statistics of earlier moments only.
    Inputs are the window summary the tree models use (last, mean, spread, min, max per channel)
    plus the time of day.
    """

    def __init__(self, symbols: list[str], features_cfg, normalize: str = "cross_section", min_peers: int = 20,
                 keep_windows: bool = False):
        f = features_cfg
        if getattr(f, "allow_overnight", False):
            raise ValueError("the stream follows allow_overnight: false (windows inside one session)")
        unknown = set(f.channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"channels not available in the stream: {sorted(unknown)}")
        if normalize not in ("cross_section", "running"):
            raise ValueError(f"unknown normalize '{normalize}'; options: cross_section, running")
        self.symbols = np.asarray(symbols)
        self.channels = list(f.channels)
        self.window, self.embargo, self.horizon = int(f.window), int(f.embargo), int(f.horizon)
        self.stride = int(getattr(f, "stride", 1))
        self.normalize, self.min_peers, self.keep_windows = normalize, min_peers, keep_windows
        S, self.L = len(symbols), self.window + VOLUME_LOOKBACK
        self.ring = {k: np.full((S, self.L), np.nan) for k in ("open", "high", "low", "close", "volume")}
        self.ptr = np.zeros(S, dtype=int)                 # rows written per symbol
        self.prev_close = np.full(S, np.nan)              # last valid close of the previous session
        self.cur_last = np.full(S, np.nan)                # last valid close so far this session
        self.d = 5 * len(self.channels) + 1
        self.scaler = RunningStandardizer(self.d) if normalize == "running" else None
        self.pending: list[Moment] = []
        self.mid = 0

    # -- per bar
    def _append(self, row: dict[str, np.ndarray], present: np.ndarray) -> None:
        idx = np.flatnonzero(present)
        slot = self.ptr[idx] % self.L
        for k, buf in self.ring.items():
            buf[idx, slot] = row[k][idx]
        self.ptr[idx] += 1

    def _ordered(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Each listed symbol's last L rows, oldest first (NaN before its first row)."""
        order = (self.ptr[idx, None] + np.arange(self.L)[None, :]) % self.L
        return {k: np.take_along_axis(buf[idx], order, axis=1) for k, buf in self.ring.items()}

    def _windows(self, idx: np.ndarray) -> np.ndarray:
        """(n, channels, window) for the listed symbols, exactly as features.build_channels computes them."""
        r = self._ordered(idx)
        w, L = self.window, self.L
        c, prev = r["close"][:, L - w:], r["close"][:, L - w - 1: L - 1]
        out = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for name in self.channels:
                if name == "log_return":
                    out.append(np.log(c / prev))
                elif name == "session_return":
                    out.append(np.log(c / self.prev_close[idx, None]))
                elif name == "hl_range":
                    out.append((r["high"][:, L - w:] - r["low"][:, L - w:]) / c)
                elif name == "close_open":
                    o = r["open"][:, L - w:]
                    out.append((c - o) / o)
                elif name == "volume_z":
                    v = r["volume"]
                    view = np.lib.stride_tricks.sliding_window_view(v[:, L - w - VOLUME_LOOKBACK + 1:], VOLUME_LOOKBACK, axis=1)
                    finite = np.isfinite(view)
                    count = finite.sum(axis=2)
                    mu = np.where(finite, view, 0.0).sum(axis=2) / np.maximum(count, 1)
                    dev = np.where(finite, view - mu[..., None], 0.0)
                    sd = np.sqrt((dev**2).sum(axis=2) / np.maximum(count - 1, 1))
                    ok = (count >= VOLUME_LOOKBACK // 2) & (sd > 0)
                    out.append(np.where(ok, (v[:, L - w:] - mu) / np.where(ok, sd, 1.0), np.nan))
                elif name == "signed_volume":
                    direction = np.nan_to_num(np.sign(c - prev), nan=0.0)
                    out.append(direction * np.log1p(r["volume"][:, L - w:]))
        return np.stack(out, axis=1)

    def push(self, date: np.datetime64, offset: int, row: dict[str, np.ndarray], slots: np.ndarray):
        """Feed one bar for every symbol. Returns (matured moments, new moment or None).

        A symbol is present at `offset` if its session has that many bars. Its row may still be all
        NaN (no IEX trade in those 5 minutes), which the batch grid keeps as a row too.
        """
        present = offset < slots
        if offset == 0:
            first = present
            self.prev_close[first] = self.cur_last[first]
            self.cur_last[first] = np.nan
        close = row["close"]
        valid_close = present & np.isfinite(close)
        self.cur_last[valid_close] = close[valid_close]
        self._append(row, present)

        matured = []
        for m in self.pending:
            if m.entry_offset == offset:
                m.entry_close = row["close"][m._idx]
            if m.exit_offset == offset:
                with np.errstate(divide="ignore", invalid="ignore"):
                    m.ret = np.log(row["close"][m._idx] / m.entry_close)
                matured.append(m)
        self.pending = [m for m in self.pending if m.exit_offset > offset]

        moment = self._decide(date, offset, present, slots)
        if moment is not None:
            self.pending.append(moment)
        return matured, moment

    def _decide(self, date, offset, present, slots) -> Moment | None:
        start = offset - (self.window - 1)
        if start < 0 or start % self.stride:
            return None
        exit_offset = offset + self.embargo + self.horizon
        idx = np.flatnonzero(present & (exit_offset < slots))
        if not len(idx):
            return None
        X = self._windows(idx)
        ok = np.isfinite(X).all(axis=(1, 2))
        idx, X = idx[ok], X[ok]
        if self.normalize == "cross_section" and len(idx) < self.min_peers:
            return None
        if not len(idx):
            return None
        F = np.column_stack([window_summary(X), np.full(len(idx), offset / 78.0)])
        if self.scaler is not None:
            Z = self.scaler.transform(F)
            self.scaler.update(F)
        else:
            Z = cross_section_z(F)
            Z[:, -1] = offset / 78.0
        m = Moment(mid=self.mid, date=np.datetime64(date, "D"), bar_index=offset, symbols=self.symbols[idx],
                   F=Z.astype(np.float32), X=X if self.keep_windows else None,
                   entry_offset=offset + self.embargo, exit_offset=exit_offset)
        m._idx = idx
        self.mid += 1
        return m


def stream_events(bars: dict[str, pd.DataFrame], features_cfg, normalize: str = "cross_section",
                  min_peers: int = 20, keep_windows: bool = False):
    """Replay cached bars through a StreamFeaturizer. Returns (moments by id, events in time order).

    An event is (time, "outcome" | "decide", moment id). At one bar the outcomes come first:
    both are known at that bar's close. The list can be replayed through any number of agents.
    """
    feat = StreamFeaturizer(list(bars), features_cfg, normalize, min_peers, keep_windows)
    moments, events, t = {}, [], 0
    for date, fields, slots in panel_days(bars):
        for offset in range(fields["close"].shape[0]):
            row = {k: v[offset] for k, v in fields.items()}
            matured, new = feat.push(date, offset, row, slots)
            for m in matured:
                events.append((t, "outcome", m.mid))
            if new is not None:
                moments[new.mid] = new
                events.append((t, "decide", new.mid))
            t += 1
        for m in feat.pending:            # a session that ended before its exit bar (bad grid): never matures
            m.ret = None
        feat.pending = []
    for m in moments.values():
        m.entry_close = None
        m.__dict__.pop("_idx", None)
    return moments, events


def compare_to_batch(moments: dict[int, Moment], data: dict) -> dict:
    """Match stream moments to `build_dataset` samples by (symbol, date, bar) and compare.

    Returns counts plus the largest differences in raw windows and returns. Every batch sample
    must have a stream twin; stream samples without a batch twin must be exactly the ones whose
    return turned out not finite (an entry or exit bar with no trade), which the batch drops.
    """
    rows = []
    for m in moments.values():
        for i, s in enumerate(m.symbols):
            rows.append((s, m.date, m.bar_index, m.mid, i))
    stream = pd.DataFrame(rows, columns=["symbol", "date", "bar_index", "mid", "i"])
    stream["date"] = pd.to_datetime(stream["date"])
    batch = pd.DataFrame({"symbol": data["symbol"], "date": pd.to_datetime(data["date"]),
                          "bar_index": data["bar_index"], "row": np.arange(len(data["symbol"]))})
    joined = batch.merge(stream, on=["symbol", "date", "bar_index"], how="left")
    missing = int(joined["mid"].isna().sum())
    joined = joined.dropna(subset=["mid"])
    dx, dr = 0.0, 0.0
    for mid, grp in joined.groupby("mid"):
        m = moments[int(mid)]
        i = grp["i"].to_numpy(int)
        xb = data["X"][grp["row"].to_numpy()].astype(np.float64)      # (n, channels, window)
        if m.X is not None:
            dx = max(dx, float(np.max(np.abs(m.X[i] - xb) / (1 + np.abs(xb)))))
        dr = max(dr, float(np.max(np.abs(m.ret[i] - data["ret"][grp["row"].to_numpy()]))))
    matched = set(zip(joined["mid"].astype(int), joined["i"].astype(int)))
    extra = [(mid, i) for mid, i in zip(stream["mid"], stream["i"]) if (mid, i) not in matched]
    extra_finite = sum(1 for mid, i in extra if moments[mid].ret is not None and np.isfinite(moments[mid].ret[i]))
    return {"batch": len(batch), "stream": len(stream), "missing_in_stream": missing, "extra_in_stream": len(extra),
            "extra_with_finite_return": extra_finite, "max_window_diff": dx, "max_return_diff": dr}


# ------------------------------------------------------------------ exploration control

class PageHinkley:
    """Alarm when a stream's mean drops. Values are standardized by a running spread first.

    S_t = max(0, S_{t-1} + (mean_t - x_t - delta)); alarm when S_t > threshold, then reset.
    `delta` (in standard deviations) is the drop it ignores; `threshold` trades delay for false alarms.
    """

    def __init__(self, delta: float = 0.5, threshold: float = 25.0, halflife: float = 200.0, warmup: int = 30):
        self.delta, self.threshold, self.warmup = delta, threshold, warmup
        self.alpha = 1 - 0.5 ** (1 / halflife)
        self.mean, self.var, self.n, self.s = 0.0, 1.0, 0, 0.0

    def update(self, x: float) -> bool:
        if not np.isfinite(x):
            return False
        self.n += 1
        if self.n == 1:
            self.mean, self.var = x, 1.0
            return False
        z = (x - self.mean) / np.sqrt(max(self.var, 1e-12))
        a = self.alpha if self.n > self.warmup else 1.0 / self.n
        delta = x - self.mean
        self.mean += a * delta
        self.var = (1 - a) * (self.var + a * delta**2)
        if self.n <= self.warmup:
            return False
        self.s = max(0.0, self.s + (-z - self.delta))
        if self.s > self.threshold:
            self.s = 0.0
            return True
        return False


@dataclass
class EntropyControl:
    """Adapts the entropy bonus beta so the policy's mean entropy tracks a target.

    beta <- clip(beta * exp(lr * (target - H)), beta_min, beta_max). Multiplicative, so beta stays
    positive and moves by the same factor at any scale. ln 3 = 1.10 nats is a uniform choice among
    short / flat / long; 0.5 is roughly (0.08, 0.84, 0.08). A drift alarm adds `boost` to the target
    for `boost_moments` moments. `enabled=False` is the ablation: beta fixed at 0.
    """
    target: float = 0.5
    beta: float = 0.05
    beta_min: float = 1e-4
    beta_max: float = 2.0
    lr: float = 0.5
    boost: float = 0.4
    boost_moments: int = 60
    enabled: bool = True
    boost_left: int = 0

    def current_target(self) -> float:
        return min(self.target + (self.boost if self.boost_left > 0 else 0.0), float(np.log(3)) - 0.05)

    def step(self, entropy: float) -> float:
        if not self.enabled:
            self.beta = 0.0
            return 0.0
        self.beta = float(np.clip(self.beta * np.exp(self.lr * (self.current_target() - entropy)), self.beta_min, self.beta_max))
        return self.beta

    def alarm(self) -> None:
        self.boost_left = self.boost_moments

    def tick(self) -> None:
        self.boost_left = max(0, self.boost_left - 1)


# ------------------------------------------------------------------ the agent

@dataclass
class OnlineConfig:
    cost_bps: float = 2.0
    feedback: str = "full"         # full: all three actions' rewards; bandit: only the action taken
    neutral: bool = True           # reward relative to the cross-sectional average at that moment
    hidden: int = 32
    lr: float = 3e-3
    buffer: int = 50_000           # most recent matured samples kept; older ones are forgotten
    min_buffer: int = 2_000        # no updates before this many matured samples
    update_every: int = 1          # moments between update rounds
    steps: int = 2                 # gradient steps per round
    batch: int = 1024
    reward_scale: float = 10.0     # bps -> network units
    max_ratio: float = 5.0         # bandit: cap on the importance weight pi_now / pi_when_taken
    warmup_moments: int = 180      # ~60 days at 3 a day: scoring starts here, and the frozen copy is taken
    drift: bool = True
    entropy: EntropyControl = field(default_factory=EntropyControl)
    seed: int = 0


class OnlineAgent:
    """Decides on each moment, learns from each matured outcome, never sees a return before maturity.

    `decide` returns three paper books for the same moment: `explore` samples from the policy (the
    agent that learns; its exploration costs money), `greedy` takes the most likely action of the same
    policy (what would be deployed), and `frozen` is the greedy policy as it was at the end of warm-up,
    never updated afterwards (does continuing to learn help?).
    """

    def __init__(self, d: int, cfg: OnlineConfig):
        if cfg.feedback not in ("full", "bandit"):
            raise ValueError(f"unknown feedback '{cfg.feedback}'; options: full, bandit")
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.policy = Policy(d, cfg.hidden)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.ctrl = copy.deepcopy(cfg.entropy)
        self.drift = PageHinkley() if cfg.drift else None
        self.frozen: Policy | None = None
        W = cfg.buffer
        self.bF = np.zeros((W, d), np.float32)
        self.bR = np.zeros((W, 3), np.float32)
        self.bA = np.zeros(W, np.int64)
        self.bP = np.ones(W, np.float32)
        self.filled, self.head = 0, 0
        self.open: dict[int, tuple] = {}
        self.decided = 0
        self.baseline, self.spread = 0.0, 1.0
        self.trace: list[dict] = []

    @torch.no_grad()
    def _probs(self, policy: Policy, F: np.ndarray) -> np.ndarray:
        return torch.softmax(policy(torch.from_numpy(F)), dim=1).numpy().astype(np.float64)

    def decide(self, mid: int, F: np.ndarray) -> dict[str, np.ndarray]:
        p = self._probs(self.policy, F)
        p /= p.sum(axis=1, keepdims=True)
        u = self.rng.random(len(F))[:, None]
        a = np.minimum((u > np.cumsum(p, axis=1)).sum(axis=1), 2)
        self.open[mid] = (F, a, p[np.arange(len(F)), a])
        if self.decided == self.cfg.warmup_moments:
            self.frozen = copy.deepcopy(self.policy)
        self.decided += 1
        self.ctrl.tick()
        greedy = p.argmax(axis=1)
        frozen = self._probs(self.frozen, F).argmax(axis=1) if self.frozen is not None else np.ones(len(F), int)
        H = float(-(p * np.log(p + 1e-12)).sum(axis=1).mean())
        self.last_entropy = H
        return {"explore": ACTIONS[a], "greedy": ACTIONS[greedy], "frozen": ACTIONS[frozen],
                "probs": p, "entropy": H}

    def observe(self, mid: int, ret_bps: np.ndarray) -> None:
        """The moment's realized returns, in bps, available now at maturity. NaN = no usable price."""
        F, a, pb = self.open.pop(mid)
        ok = np.isfinite(ret_bps)
        if not ok.any():
            return
        r = ret_bps[ok] - (ret_bps[ok].mean() if self.cfg.neutral else 0.0)
        R = action_rewards(r, self.cfg.cost_bps)
        if self.drift is not None:
            greedy = self._probs(self.policy, F[ok]).argmax(axis=1)
            if self.drift.update(float(R[np.arange(len(r)), greedy].mean())):
                self.ctrl.alarm()
        n = len(r)
        idx = (self.head + np.arange(n)) % self.cfg.buffer
        self.bF[idx], self.bR[idx], self.bA[idx], self.bP[idx] = F[ok], R, a[ok], pb[ok]
        self.head = (self.head + n) % self.cfg.buffer
        self.filled = min(self.filled + n, self.cfg.buffer)
        self.observed = getattr(self, "observed", 0) + 1
        if self.filled >= self.cfg.min_buffer and self.observed % self.cfg.update_every == 0:
            self._learn()

    def _learn(self) -> None:
        cfg = self.cfg
        Hs = []
        for _ in range(cfg.steps):
            idx = self.rng.integers(0, self.filled, min(cfg.batch, self.filled))
            logits = self.policy(torch.from_numpy(self.bF[idx]))
            dist = torch.distributions.Categorical(logits=logits)
            R = torch.from_numpy(self.bR[idx]) / cfg.reward_scale
            if cfg.feedback == "full":
                gain = (dist.probs * R).sum(dim=1).mean()
            else:
                a = torch.from_numpy(self.bA[idx])
                r = R[torch.arange(len(idx)), a]
                self.baseline = 0.99 * self.baseline + 0.01 * float(r.mean())
                self.spread = 0.99 * self.spread + 0.01 * float(r.std() + 1e-6)
                adv = (r - self.baseline) / self.spread
                logp = dist.log_prob(a)
                ratio = torch.clamp(torch.exp(logp.detach()) / torch.from_numpy(self.bP[idx]), max=cfg.max_ratio)
                gain = (ratio * adv * logp).mean()
            H = dist.entropy().mean()
            loss = -gain - self.ctrl.beta * H
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            Hs.append(float(H.detach()))
        self.ctrl.step(float(np.mean(Hs)))


# ------------------------------------------------------------------ replay

def run_agent(moments: dict[int, Moment], events: list, d: int, cfg: OnlineConfig,
              placebo_seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drive one agent through the events. Returns (decisions, trace).

    decisions: one row per stock and moment with the market-neutral return in bps (shuffled across
    the moment's stocks in a placebo run, for scoring as for learning) and the three books' positions.
    trace: per moment, the policy's entropy, beta, the entropy target and whether a drift boost is on.
    """
    agent = OnlineAgent(d, cfg)
    shuffle = np.random.default_rng(placebo_seed) if placebo_seed is not None else None
    books, trace, seen = {}, [], {}
    for _, kind, mid in events:
        m = moments[mid]
        if kind == "decide":
            out = agent.decide(mid, m.F)
            books[mid] = out
            trace.append({"mid": mid, "date": m.date, "bar_index": m.bar_index, "entropy": out["entropy"],
                          "beta": agent.ctrl.beta, "target": agent.ctrl.current_target(),
                          "boost": agent.ctrl.boost_left > 0, "trades_explore": float(np.mean(out["explore"] != 0))})
        else:
            ret = m.ret * 1e4
            if shuffle is not None:
                ret = ret.copy()
                ok = np.flatnonzero(np.isfinite(ret))
                ret[ok] = ret[shuffle.permutation(ok)]
            seen[mid] = ret
            agent.observe(mid, ret)
    rows = []
    for mid, out in books.items():
        if mid not in seen:
            continue
        m, ret = moments[mid], seen[mid]
        ok = np.isfinite(ret)
        if not ok.any():
            continue
        r = ret[ok] - (ret[ok].mean() if cfg.neutral else 0.0)
        rows.append(pd.DataFrame({"mid": mid, "date": pd.Timestamp(m.date), "bar_index": m.bar_index,
                                  "symbol": m.symbols[ok], "ret_bps": r, "scored": mid >= cfg.warmup_moments,
                                  **{f"pos_{k}": out[k][ok] for k in ("explore", "greedy", "frozen")}}))
    decisions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return decisions, pd.DataFrame(trace)


def book_pnl(decisions: pd.DataFrame, book: str, cost_bps: float) -> pd.DataFrame:
    """Frame for engine.score_positions: scored rows only, P&L net of cost per position."""
    d = decisions[decisions["scored"]]
    pos = d[f"pos_{book}"].to_numpy(float)
    return pd.DataFrame({"date": d["date"].to_numpy(), "pos": pos,
                         "pnl_bps": pos * d["ret_bps"].to_numpy() - cost_bps * np.abs(pos)})
