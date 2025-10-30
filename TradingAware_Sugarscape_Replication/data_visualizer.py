import json
import pickle
import socket
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gmean

def show_one(data, index: int):
    # ❶ Collect the six time-series for this run
    metric_series = [
        data[0][index],
        data[1][index],
        data[2][index],
        data[3][index],
        data[4][index],
        data[5][index],
    ]
    titles = [
        "Trade",
        "Market Price",
        "Alive",
        "Inequality",
        "Welfare",
        "Consumption-Regrowth Ratio",
    ]

    # ❷ Lay out a 3 × 2 grid of axes
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    # ❸ Plot each series in its own panel
    for ax, title, series in zip(axes, titles, metric_series):
        ax.plot(series, label="hard_code", color="#1f77b4")   # <-- draw the line
        ax.set_title(title, weight="bold", fontsize=11)
        # ax.set_xlabel("Step")
        ax.xaxis.set_tick_params(labelbottom=True)   # make tick labels visible
        ax.xaxis.label.set_visible(True)
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")                          # valid loc string

    fig.tight_layout(pad=2.0)
    plt.show()

def compare_two(data, first_index=0, second_index=1, group_one='hard_code', group_two='drl_agent'):
    # Titles and labels
    metrics = ['trade_counts','market_prices', 'alive_agents', 'inequality', 'average_welfare', 'cr_ratios']
    titles = ['Trade','Market Price',  'Alive', 'Inequality', 'Welfare', 'Consumption-Regrowth Ratio']

    # Create long-format DataFrame with separate x's for each source
    data_frames = []
    for i, (y1, y2) in enumerate(zip(
        [data[0][first_index][1:], data[1][first_index][1:], data[2][first_index][1:], data[3][first_index][1:], data[4][first_index][1:], data[5][first_index][1:]],
        [data[0][second_index][1:], data[1][second_index][1:], data[2][second_index][1:], data[3][second_index][1:], data[4][second_index][1:], data[5][second_index][1:]]
    )):
        x1 = np.linspace(0, 100, len(y1))[1:]
        x2 = np.linspace(0, 100, len(y2))[1:]

        df1 = pd.DataFrame({
            'x': x1,
            'y': y1[1:],  # skip first value if needed
            'source': group_one,
            'metric': metrics[i],
            'title': titles[i]
        })

        df2 = pd.DataFrame({
            'x': x2,
            'y': y2[1:],
            'source': group_two,
            'metric': metrics[i],
            'title': titles[i]
        })

        data_frames.append(pd.concat([df1, df2]))

    df_all = pd.concat(data_frames, ignore_index=True)

    # Plot in 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, title in enumerate(titles):
        df_plot = df_all[df_all['title'] == title]
        sns.lineplot(data=df_plot, x='x', y='y', hue='source', ax=axes[i])
        axes[i].set_title(title)
        axes[i].legend(title='Configuration')
        axes[i].set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

def compare_three(data,first_index,second_index,third_index):
    # Titles and labels
    metrics = ['trade_counts','market_prices', 'alive_agents', 'inequality', 'average_welfare', 'cr_ratios']
    titles = ['Trade','Market Price',  'Alive', 'Inequality', 'Welfare', 'Consumption-Regrowth Ratio']

    # Create long-format DataFrame with separate x's for each source
    data_frames = []
    for i, (y1, y2, y3) in enumerate(zip(
        [data[0][first_index][1:],data[1][first_index][1:], data[2][first_index][1:], data[3][first_index][1:], data[4][first_index][1:], data[5][first_index][1:]],
        [data[0][second_index][1:], data[1][second_index][1:], data[2][second_index][1:], data[3][second_index][1:], data[4][second_index][1:], data[5][second_index][1:]],
        [data[0][third_index][1:], data[1][third_index][1:], data[2][third_index][1:], data[3][third_index][1:], data[4][third_index][1:], data[5][third_index][1:]]
    )):
        x1 = np.linspace(0, 100, len(y1))[1:]
        x2 = np.linspace(0, 100, len(y2))[1:]
        x3 = np.linspace(0, 100, len(y3))[1:]

        df1 = pd.DataFrame({
            'x': x1,
            'y': y1[1:],  # skip first value if needed
            'source': 'hard_code',
            'metric': metrics[i],
            'title': titles[i]
        })

        df2 = pd.DataFrame({
            'x': x2,
            'y': y2[1:],
            'source': 'stochastic_agent',
            'metric': metrics[i],
            'title': titles[i]
        })

        df3 = pd.DataFrame({
            'x': x3,
            'y': y3[1:],
            'source': 'static_agent',
            'metric': metrics[i],
            'title': titles[i]
        })

        data_frames.append(pd.concat([df1, df2, df3]))

    df_all = pd.concat(data_frames, ignore_index=True)

    # Plot in 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, title in enumerate(titles):
        df_plot = df_all[df_all['title'] == title]
        sns.lineplot(data=df_plot, x='x', y='y', hue='source', ax=axes[i])
        axes[i].set_title(title)
        axes[i].legend(title='Configuration')

    plt.tight_layout()
    plt.show()

def plot_distribution(data, bins=20):
    plt.figure(figsize=(8, 5))
    sns.histplot(data, bins=bins, kde=True, color="#1f77b4")

    plt.title("Distribution of remaining sugar", weight="bold")
    plt.xlabel("remainSugar")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_progress_two(data1, data2, color="#1f77b4", label1="Rule-based", label2="RL", sub_title=None):
    datasets = [data1, data2]
    titles   = [label1, label2]

    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    axes = axes.flatten()                      # easier to iterate

    for ax, data, title in zip(axes, datasets, titles):
        sns.histplot(data,bins=20,kde=True,ax=ax,color=color)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        ax.set_title(title, weight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    fig.tight_layout(pad=2.0)                  # avoid overlaps
    if sub_title is not None: plt.suptitle(sub_title)
    plt.show()

def plot_comparison_bars(data1, data2, label1="Rule-based", label2="RL", colors=("#1f77b4", "#ff7f0e"), sub_title=None):
    """
    Plot side-by-side frequency bars for two data series.
    The horizontal axis shows the *actual* integer values, not the
    automatically-generated index.
    """
    # 1) Round to nearest int and make Series
    v1 = pd.Series(np.rint(data1).astype(int))
    v2 = pd.Series(np.rint(data2).astype(int))

    # 2) Frequency tables (no sorting by count)
    c1 = v1.value_counts().sort_index()
    c2 = v2.value_counts().sort_index()

    # 3) All distinct values that ever appeared
    all_vals = sorted(set(c1.index).union(c2.index))

    # 4) Re-index so the two Series have identical value grids
    c1 = c1.reindex(all_vals, fill_value=0)
    c2 = c2.reindex(all_vals, fill_value=0)

    # 5) Long form for Seaborn
    df_long = pd.DataFrame({
        "Value": np.tile(all_vals, 2),
        "Group": np.repeat([label1, label2], len(all_vals)),
        "Count": np.concatenate([c1.values, c2.values])
    })

    order = all_vals
    df_long["Value"] = pd.Categorical(df_long["Value"], categories=order, ordered=True)

    # 6) Plot
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_long, x="Value", y="Count", hue="Group", palette=colors, ax=ax, dodge=True, order=order)

    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(order))))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(order))

    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend(title="")

    if sub_title:
        ax.set_title(sub_title)

    plt.tight_layout()
    plt.show()


def plot_progress_three(data1,data2,data3,bins=20, color="#1f77b4", sub_title=None):
    datasets = [data1, data2, data3]
    titles   = ["First", "Middle", "Last"]

    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    axes = axes.flatten()                      # easier to iterate

    for ax, data, title in zip(axes, datasets, titles):
        sns.histplot(
            data,
            bins=20,
            kde=True,
            ax=ax,
            color=color                    # same blue as before
        )
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    fig.tight_layout(pad=2.0)                  # avoid overlaps
    if sub_title is not None: plt.suptitle(sub_title)
    plt.show()

def show_mean_overtime(data,title, benchmark = 0):
    means, stds = [], []
    for prices in data:
        if prices:  # skip empty steps
            arr = np.asarray(prices, dtype=float)
            means.append(arr.mean())
            stds.append(arr.std(ddof=0))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    means, stds = np.asarray(means), np.asarray(stds)
    steps = np.arange(len(data))

    valid = ~np.isnan(means)
    steps, means, stds = steps[valid], means[valid], stds[valid]

    # ── Plot ──────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(steps, means, lw=2, label='mean')
    plt.fill_between(steps, means - stds, means + stds,
                     alpha=0.3, label='±1 σ')

    # 👉 horizontal dashed line at y = 0
    plt.axhline(benchmark, linestyle='--', linewidth=1,color='gray', label='zero baseline')

    plt.xlabel('step')
    plt.ylabel(title)
    plt.title(f'Mean price trajectory ({len(data)} steps)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def show_mean_overtime(data,title, benchmark = 0):
    means, stds = [], []
    for prices in data:
        if prices:  # skip empty steps
            arr = np.asarray(prices, dtype=float)
            means.append(arr.mean())
            stds.append(arr.std(ddof=0))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    means, stds = np.asarray(means), np.asarray(stds)
    steps = np.arange(len(data))

    valid = ~np.isnan(means)
    steps, means, stds = steps[valid], means[valid], stds[valid]

    # ── Plot ──────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(steps, means, lw=2, label='mean')
    plt.fill_between(steps, means - stds, means + stds,
                     alpha=0.3, label='±1 σ')

    # 👉 horizontal dashed line at y = 0
    plt.axhline(benchmark, linestyle='--', linewidth=1,color='gray', label='zero baseline')

    plt.xlabel('step')
    plt.ylabel(title)
    plt.title(f'Mean price trajectory ({len(data)} steps)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def geometric_sd(arr, axis=0):
    if np.any(arr <= 0):
        return np.full(arr.shape[axis], np.nan)
    ln_x  = np.log(arr)
    ln_gm = ln_x.mean(axis=axis, keepdims=True)
    var   = ((ln_x - ln_gm)**2).mean(axis=axis)
    return np.exp(np.sqrt(var))

def show_geom_mean_overtime(data, title, benchmark=0):
    g_means, g_stds = [], []

    for prices in data:
        if prices:                                   # skip empty step
            arr = np.asarray(prices, dtype=float)

            # geometric mean & σ_g require positive numbers
            if np.all(arr > 0):
                g_means.append(gmean(arr))
                g_stds.append(geometric_sd(arr))
            else:
                g_means.append(np.nan)
                g_stds.append(np.nan)
        else:
            g_means.append(np.nan)
            g_stds.append(np.nan)

    g_means, g_stds = np.asarray(g_means), np.asarray(g_stds)
    steps           = np.arange(len(data))

    valid = ~np.isnan(g_means)
    steps, g_means, g_stds = steps[valid], g_means[valid], g_stds[valid]

    # ── Plot ──────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(steps, g_means, lw=2, label='geometric mean')
    plt.fill_between(steps, g_means / g_stds, g_means * g_stds,
                     alpha=0.3, label='±1 σ\u2093')  # σᵍ

    plt.axhline(benchmark, ls='--', lw=1, color='gray', label='benchmark')

    plt.xlabel('step')
    plt.ylabel(title)
    plt.title(f'Geometric mean trajectory ({len(data)} steps)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_mean_overtime(data1, data2,group1, group2, title,
                          x_begin=0, y_begin=0, y_end=None,
                          is_show_overall=False, annotate_last_value=False,
                          ax=None):

    # ── 1. Stats ───────────────────────────────────────────────
    arr_A = np.asarray(data1)
    arr_B = np.asarray(data2)

    mean_A, std_A = arr_A.mean(0), arr_A.std(0)
    mean_B, std_B = arr_B.mean(0), arr_B.std(0)

    overall_mean_A = mean_A.mean()
    overall_mean_B = mean_B.mean()

    steps = np.arange(arr_A.shape[1])

    # ── 2. Figure / Axes handling ──────────────────────────────
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True

    # ── 3. Plot ────────────────────────────────────────────────
    ax.plot(steps, mean_A, lw=2, label=f"mean – {group1}")
    ax.fill_between(steps, mean_A - std_A, mean_A + std_A,
                    alpha=0.25, label=f"±1 σ – {group1}")

    ax.plot(steps, mean_B, lw=2, label=f"mean – {group2}")
    ax.fill_between(steps, mean_B - std_B, mean_B + std_B,
                    alpha=0.25, label=f"±1 σ – {group2}")

    if is_show_overall:
        ax.axhline(overall_mean_A, ls="--", color="C0",
                   label=f"overall μ {group1} = {overall_mean_A:.2f}")
        ax.axhline(overall_mean_B, ls="--", color="C1",
                   label=f"overall μ {group2} = {overall_mean_B:.2f}")

    if annotate_last_value:
        # plot a small marker so the reader sees what we’re labelling
        last_a, last_b = mean_A[-1], mean_B[-1]
        ax.scatter(steps[-1], last_a, color="C0")
        ax.scatter(steps[-1], last_b, color="C1")
        # tweak offsets so the numbers don’t overlap the markers
        ax.text(steps[-1], last_a,f"{last_a:.2f}", color="C0", va="bottom", ha="left", fontsize=9)
        ax.text(steps[-1], last_b,f"{last_b:.2f}", color="C1", va="bottom", ha="left", fontsize=9)

    ax.set(
        xlabel="step",
        ylabel=title,
        title=f"Time series of {title}",
        xlim=(x_begin, steps[-1])
    )
    if y_end is not None:
        ax.set_ylim(y_begin, y_end)
    ax.legend()

    if created_fig:                # stand-alone call
        fig.tight_layout()
        plt.show()

    return ax                      # lets caller keep chaining if desired

def plot_step_stats(steps_data, ylabel="value", title=None):
    """
    steps_data : list[list[float]]
        Outer index = step, inner lists = scalar observations for that step.
    ylabel : str
        Label for the y-axis.
    """
    # --- 1. Mean and σ for each step -----------------------------------
    means = [np.mean(step) if step else np.nan for step in steps_data]
    stds  = [np.std(step)  if step else np.nan for step in steps_data]

    steps = np.arange(len(steps_data))

    # --- 2. Plot --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(steps, means, lw=2, label="mean")
    plt.fill_between(steps, np.array(means) - stds,
                              np.array(means) + stds,
                              alpha=0.3, label="±1 σ")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_two_step_series(steps_A, steps_B, labels=("A", "B"),
                            ylabel="value", title="Mean ±1 σ per step (two datasets)"):
    # ------------------------------------------------------------------
    # 1. Pad the shorter dataset with empty lists so indices line up
    # ------------------------------------------------------------------
    max_len = max(len(steps_A), len(steps_B))
    pad = lambda seq, L: seq + [[]]*(L - len(seq))
    steps_A, steps_B = pad(steps_A, max_len), pad(steps_B, max_len)

    # ------------------------------------------------------------------
    # 2. Per-step means and standard deviations
    # ------------------------------------------------------------------
    def stats(series):
        means = [np.mean(s) if s else np.nan for s in series]
        stds  = [np.std(s)  if s else np.nan for s in series]
        return np.array(means), np.array(stds)

    mean_A, std_A = stats(steps_A)
    mean_B, std_B = stats(steps_B)

    x = np.arange(max_len)

    # ------------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))

    # Dataset A
    plt.plot(x, mean_A, lw=2, label=f"{labels[0]} mean")
    plt.fill_between(x, mean_A - std_A, mean_A + std_A,
                     alpha=0.25, label=f"{labels[0]} ±1 σ")

    # Dataset B
    plt.plot(x, mean_B, lw=2, label=f"{labels[1]} mean")
    plt.fill_between(x, mean_B - std_B, mean_B + std_B,
                     alpha=0.25, label=f"{labels[1]} ±1 σ")

    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_distribution_by_group(data1, data2,step,is_sugar=True,label=('Group1','Group2'), axis=('x_axis','y_axis'), title=None, ax=None):
    def extract_agent_data(list_step_agents_config, config_label):
        data = []
        for simulation in list_step_agents_config:
            if step < len(simulation):  # safety check
                for agent in simulation[step]:
                    if is_sugar:
                        data.append({
                            'remain': agent.remainSugar,
                            'metabolism': agent.SugarMetabolism,
                            'config': config_label
                        })
                    else:
                        data.append({
                            'remain': agent.remainSpice,
                            'metabolism': agent.SpiceMetabolism,
                            'config': config_label
                        })
        return data

    # Get data from both configs
    data1 = extract_agent_data(data1, label[0])
    data2 = extract_agent_data(data2, label[1])

    # Combine and convert to DataFrame
    df = pd.DataFrame(data1 + data2)

    # Draw comparison plot (boxplot or violinplot)
    # plt.figure(figsize=(12, 6))
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True

    sns.boxplot(x='metabolism', y='remain', hue='config', data=df, ax=ax)
    ax.set_title(title or "")
    ax.set_xlabel(axis[0])
    ax.set_ylabel(axis[1])
    ax.legend(title="Configuration")

    # plt.title(title)
    # plt.xlabel(axis[0])
    # plt.ylabel(axis[1])
    # plt.legend(title='Configuration')

    if created_fig:                # stand-alone call
        fig.tight_layout()
        plt.show()

    return ax

def dot_graph(step_values, title, jitter=0.15, dot_size=30, alpha=0.7, color='tab:blue'):
    plt.figure(figsize=(10, 5))

    rng = np.random.default_rng()         # fast NumPy random generator

    for step, values in enumerate(step_values):
        if not values:                    # skip empty steps
            continue
        x = np.full(len(values), step, dtype=float)

        if jitter:
            x += rng.uniform(-jitter, jitter, size=len(values))

        plt.scatter(x, values, s=dot_size, alpha=alpha, color=color)

    # Optional: show step ticks even if some steps were empty
    # plt.xticks(np.arange(len(step_values)))

    plt.xlabel('step')
    plt.ylabel(title)
    # plt.title(f'Dot graph ({len(step_values)} steps)')
    plt.tight_layout()
    plt.show()