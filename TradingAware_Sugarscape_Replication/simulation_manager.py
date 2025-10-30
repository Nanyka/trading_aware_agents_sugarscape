# from __future__ import annotations
import math
import statistics
import powerlaw

from scipy import stats
from typing import Sequence, Dict, Optional
from data_visualizer import *


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

@dataclass
class Metric:
    TradeCount: float
    AliveAgent: int
    MarketPrice: float
    Inequality: float
    AverageWelfare: float
    CRRatio: float
    HardCodeAgentPercentage: float
    IsEnd: bool
    Agents: List[AgentData]

HOST = '127.0.0.1'
PORT = 50007

def run_simulation():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("Waiting for Unity...")
        conn, addr = s.accept()
        print(f"Connected by {addr}")

        buffer = ""
        episode_metrics = []
        episode_agents = []
        current_metrics = []
        current_agents = []

        def parse_metric(data: str) -> Metric:
            raw = json.loads(data)
            agents = [AgentData(**agent) for agent in raw.get("Agents", [])]
            raw.pop("Agents", None)
            return Metric(Agents=agents, **raw)

        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        metric = parse_metric(line)
                        current_metrics.append(metric)
                        current_agents.extend(metric.Agents)
                        if metric.IsEnd:
                            episode_metrics.append(current_metrics)
                            episode_agents.append(current_agents)
                            current_metrics = []
                            current_agents = []
                    except json.JSONDecodeError as e:
                        print("Bad JSON:", e)

        print("Stop record!")
        return episode_metrics


def idx(agents: Sequence[AgentData]) -> Dict[int, AgentData]:
    return {a.agentId: a for a in agents}

def price_dispersion_series(history: List[Sequence[AgentData]]) -> List[Optional[float]]:
    """
    Returns a list `sigma[t]` where
        sigma[t] = st.dev. of ln(p_i,t) across all trades in tick t  (p = ΔSpice / ΔSugar)
    The first entry (tick 0) is None because no 'previous' tick exists yet.
    """
    sigma_log_price: List[Optional[float]] = []      # σ₀ undefined
    prev = idx(history[0])

    for step in history[1:]:
        now = idx(step)
        log_prices: List[float] = []

        # iterate over agents that exist in both ticks
        for aid in prev.keys() & now.keys():
            before, after = prev[aid], now[aid]
            d_sugar = after.remainSugar - before.remainSugar
            d_spice = after.remainSpice - before.remainSpice

            # genuine barter: non-zero & opposite-signed deltas
            if d_sugar != 0 and d_spice != 0 and d_sugar * d_spice < 0:
                price = abs(d_spice) / abs(d_sugar)       # spice per unit sugar
                log_prices.append(math.log(price))

        # sample st.dev. requires ≥2 data points; else treat as 0 dispersion
        sigma = statistics.stdev(log_prices) if len(log_prices) > 1 else 0.0
        sigma_log_price.append(sigma)
        prev = now                                        # advance window

    return sigma_log_price

def price_series(history: List[Sequence[AgentData]]) -> List[Optional[float]]:
    """
    Returns a list `price[t]` where
        price[t] across all trades in tick t  (p = ΔSpice / ΔSugar)
    The first entry (tick 0) is None because no 'previous' tick exists yet.
    """
    list_step_prices: List[Optional[float]] = []      # σ₀ undefined
    prev = idx(history[0])

    for step in history[1:]:
        now = idx(step)
        prices: List[float] = []

        # iterate over agents that exist in both ticks
        for aid in prev.keys() & now.keys():
            before, after = prev[aid], now[aid]
            d_sugar = after.remainSugar - before.remainSugar
            d_spice = after.remainSpice - before.remainSpice

            # genuine barter: non-zero & opposite-signed deltas
            if d_sugar != 0 and d_spice != 0 and d_sugar * d_spice < 0:
                price = abs(d_spice) / abs(d_sugar)       # spice per unit sugar
                prices.append(price)

        list_step_prices.append(prices)
        prev = now                                        # advance window

    return list_step_prices

def calculate_welfare(agent, sugar=None, spice=None):
    alpha = agent.SugarMetabolism/(agent.SugarMetabolism + agent.SpiceMetabolism)
    beta = agent.SpiceMetabolism/(agent.SugarMetabolism + agent.SpiceMetabolism)
    if sugar is None:
        return (agent.remainSugar**alpha)*(agent.remainSpice**beta)
    elif spice is None:
        return (sugar ** alpha) * (agent.remainSpice ** beta)
    else:
        return (sugar ** alpha) * (spice ** beta)

def population_skew(arr):
    arr = np.asarray(arr, dtype=float)
    mu = arr.mean()
    sigma = arr.std(ddof=0)          # population σ
    return np.mean(((arr - mu) / sigma) ** 3)


def fit_pareto(wealth):
    fit = powerlaw.Fit(wealth)   # Clauset et al. method
    return fit.alpha, fit.xmin, fit.D               # slope α, cut-off xmin, KS-distance D

def gini(x):         # handy to confirm overall inequality
    x = np.sort(np.asarray(x))
    n = x.size
    return (2*np.arange(1,n+1)*x).sum() / (n*x.sum()) - (n+1)/n

from scipy.stats import ttest_rel, wilcoxon, shapiro
import numpy as np


# Example: convert to NumPy arrays  ➜ element-wise subtraction works

def t_test(data1, data2):
    # ── 1.  Data ─────────────────────────────────────────────────────────────
    x = np.asarray(data1, dtype=float)  # config 1
    y = np.asarray(data2, dtype=float)  # config 2

    assert x.size == y.size, "Paired samples must have the same length"
    diff = x - y
    n = diff.size
    df = n - 1

    # ── 2.  Paired t-test ─────────────────────────────────────────────────────
    t_stat, p_val = stats.ttest_rel(x, y)

    # ── 3.  Descriptive statistics ────────────────────────────────────────────
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)  # sample SD
    se_diff = std_diff / np.sqrt(n)

    # 95 % confidence interval
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    # Cohen’s d for paired samples
    cohen_d = mean_diff / std_diff

    # ── 4.  Paper-ready output ────────────────────────────────────────────────
    print(
        f"Paired t-test (n = {n}): "
        f"mean Δ = {mean_diff:.2f} "
        f"95% CI [{ci_low:.2f}, {ci_high:.2f}], "
        f"t({df}) = {t_stat:.2f}, p = {p_val:.3e}, "
        f"Cohen's d = {cohen_d:.2f}"
    )

