"""
Rebuilds the cumulative geometric mean price figure (originally Fig. 5) using
configs 1-4 (150-step baseline conditions, R=50 replicates each) instead of
the fertility/finite-lifespan configs 5/6, which only have 1 saved replicate.

Trade prices are inferred from consecutive AgentData snapshots (the same
logic as simulation_manager.price_series), not from the stale currentPrice
field, since currentPrice only updates when an agent actually trades and
otherwise carries a value from a previous, possibly much earlier, step.

For each replicate r, p_bar_r(t) = exp( (1/k_r(t)) * sum(ln p) ) over all
trade prices pooled from step 1 through step t (cumulative geometric mean,
matching the manuscript's Section 3.5 definition). mu(t)/sigma(t) are then
the cross-replicate mean/std of p_bar_r(t) across all 50 replicates.
"""

import pickle
import sys
import types
from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class AgentData:
    agentId: int
    remainSugar: int
    remainSpice: int
    currentMrs: float
    currentPrice: float
    isOccupied: bool
    Age: int
    SugarMetabolism: int
    SpiceMetabolism: int
    SugarCapacity: int
    SpiceCapacity: int


_stub = types.ModuleType("simulation_manager")
_stub.AgentData = AgentData
sys.modules["simulation_manager"] = _stub


def idx(agents: Sequence[AgentData]) -> Dict[int, AgentData]:
    return {a.agentId: a for a in agents}


def price_series(history: List[Sequence[AgentData]]) -> List[List[float]]:
    list_step_prices: List[List[float]] = []
    prev = idx(history[0])

    for step in history[1:]:
        now = idx(step)
        prices: List[float] = []

        for aid in prev.keys() & now.keys():
            before, after = prev[aid], now[aid]
            d_sugar = after.remainSugar - before.remainSugar
            d_spice = after.remainSpice - before.remainSpice

            if d_sugar != 0 and d_spice != 0 and d_sugar * d_spice < 0:
                price = abs(d_spice) / abs(d_sugar)
                prices.append(price)

        list_step_prices.append(prices)
        prev = now

    return list_step_prices


def cumulative_geom_mean(step_prices: List[List[float]]) -> np.ndarray:
    n = len(step_prices)
    out = np.full(n, np.nan)
    log_sum = 0.0
    count = 0
    for t, prices in enumerate(step_prices):
        for p in prices:
            if p > 0:
                log_sum += np.log(p)
                count += 1
        if count > 0:
            out[t] = np.exp(log_sum / count)
    return out


def load_replicates(config: int) -> List[List[Sequence[AgentData]]]:
    with open(f"submission_data/list_step_agents_config{config}.pkl", "rb") as f:
        return pickle.load(f)


def build_condition(config: int, label: str):
    replicates = load_replicates(config)
    n_rep = len(replicates)
    n_steps = len(replicates[0]) - 1

    p_bar = np.full((n_rep, n_steps), np.nan)
    trade_counts_final = np.zeros(n_rep)

    for r, history in enumerate(replicates):
        step_prices = price_series(history)
        p_bar[r] = cumulative_geom_mean(step_prices)
        trade_counts_final[r] = sum(len(p) for p in step_prices)

    mu = np.nanmean(p_bar, axis=0)
    sigma = np.nanstd(p_bar, axis=0, ddof=1)
    steps = np.arange(1, n_steps + 1)

    print(f"--- config{config} ({label}) ---")
    print(f"replicates: {n_rep}, steps: {n_steps}")
    print(f"total trades per replicate (mean +/- std): "
          f"{trade_counts_final.mean():.1f} +/- {trade_counts_final.std(ddof=1):.1f}")
    print(f"mu(final step) = {mu[-1]:.4f}, sigma(final step) = {sigma[-1]:.4f}")
    print(f"mu(step 10)    = {mu[9]:.4f}, sigma(step 10)    = {sigma[9]:.4f}")
    first_valid = np.argmax(~np.isnan(mu))
    print(f"first step with a defined cumulative mean across all replicates: {steps[first_valid]}")

    return steps, mu, sigma


def plot_condition(steps, mu, sigma, title, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, mu, lw=2, label="cross-replicate mean")
    plt.fill_between(steps, mu - sigma, mu + sigma, alpha=0.3, label=r"$\pm1\sigma$")
    plt.axhline(1.0, ls="--", lw=1, color="gray", label="benchmark")
    plt.xlabel("step")
    plt.ylabel(r"$\bar p_r(t)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    steps_rb, mu_rb, sigma_rb = build_condition(1, "rule-based, deterministic")
    steps_rl, mu_rl, sigma_rl = build_condition(2, "DRL, deterministic")

    plot_condition(steps_rb, mu_rb, sigma_rb,
                   "(a) Prices in rule-based simulation (config 1, R=50)",
                   "images/price_dynamics_v2_rulebased.png")
    plot_condition(steps_rl, mu_rl, sigma_rl,
                   "(b) Prices in RL simulation (config 2, R=50)",
                   "images/price_dynamics_v2_drl.png")
