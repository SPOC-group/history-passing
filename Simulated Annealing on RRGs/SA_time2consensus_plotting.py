import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import joblib


ds = [3, 4, 5, 6]
ns = [100, 500, 1000]
pmax_expected = 3

colors = ["#4471C4", "#ED7D31", "#7030A0"]
p_labels = [r"$p=1$", r"$p=2$", r"$p=3$"]

data_cache = joblib.load("SAdata_cache_combined.pkl")
#SAdata_cache_combined.pkl contains the aggregated simulation results used in Figure 15.
#It was produced by combining the raw .npz outputs of simulated annealing runs as described in the figures caption.
#The file can be loaded using joblib.load (Python ≥3.10, joblib ≥1.3).

# --------------------------------------------------
# Precompute axis limits per column (per d)
# --------------------------------------------------
x_limits = {}
y_limits = {}

for d in ds:
    all_m_list = []
    all_t_list = []

    for n in ns:
        combined_t, combined_m, pmax = data_cache[(d, n)]

        t_arrays = [t if t is not None else np.array([]) for t in combined_t]
        m_arrays = [m if m is not None else np.array([]) for m in combined_m]

        if t_arrays:
            all_t_list.append(np.concatenate(t_arrays) if t_arrays else np.array([]))
        if m_arrays:
            all_m_list.append(np.concatenate(m_arrays) if m_arrays else np.array([]))

    if all_t_list and all_m_list:
        all_t = np.concatenate(all_t_list)
        all_m = np.concatenate(all_m_list)

        x_limits[d] = (all_m.min() * 0.98, all_m.max() * 1.08)
        y_limits[d] = (0, all_t.max() * 1.08)
    else:
        x_limits[d] = (0, 1)
        y_limits[d] = (0, 1)

# --------------------------------------------------
# Plotting the Figure 15
# --------------------------------------------------
fig, axes = plt.subplots(
    nrows=len(ns),
    ncols=len(ds),
    figsize=(18, 9),
    sharex=False,
    sharey=False
)

for row_idx, n in enumerate(ns):
    for col_idx, d in enumerate(ds):
        ax = axes[row_idx, col_idx]

        combined_t, combined_m, pmax = data_cache[(d, n)]

        print(f"\nCombination: d={d}, n={n}")

        for p in range(pmax):
            t_i = combined_t[p]
            m_i = combined_m[p]
            if t_i.size == 0:
                continue

            color = colors[p]

            ax.scatter(m_i, t_i, s=10, c=color, alpha=0.28, edgecolors='none')

            # ----------------------
            # Compute min per unique t
            # ----------------------
            unique_ts, inverse_idx = np.unique(t_i, return_inverse=True)
            min_m = np.full(unique_ts.shape, np.inf)
            np.minimum.at(min_m, inverse_idx, m_i)

            ax.plot(min_m, unique_ts, color=color, linewidth=1.8)

            # ----------------------
            # horizontal line and black point
            # ----------------------
            target_t = p + 1
            ax.axhline(y=target_t, color=color, linewidth=1.0, alpha=0.7)

            mask = (t_i == target_t)
            if np.any(mask):
                idx_min = np.argmin(m_i[mask])
                ax.scatter(
                    m_i[mask][idx_min],
                    t_i[mask][idx_min],
                    s=90,
                    color="black",
                    zorder=5
                )

            if np.any(mask):
                best_m = np.min(m_i[mask])
                print(f"d={d}, n={n}, t={target_t}: minimal m = {best_m:.6f}")
            else:
                print(f"d={d}, n={n}, t={target_t}: NO DATA")


        ax.set_xlim(*x_limits[d])
        ax.set_ylim(*y_limits[d])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if row_idx == 0:
            ax.set_title(rf"$d = {d}$", fontsize=15)

        if col_idx == 0:
            ax.set_ylabel(rf"$n = {n}$", fontsize=15, rotation=90, labelpad=10)

        if row_idx == len(ns) - 1:
            ax.set_xlabel(
                r"$m_{\operatorname{init}}\!\left(\mathbf{x}_t^{\operatorname{sol}}\right)$",
                fontsize=15
            )

        if col_idx != 0:
            ax.set_ylabel("")

leg_ax = axes[-1, -1]
legend_handles = [
    plt.Line2D([], [], color=colors[p], marker='o', linestyle='none',
               markersize=8, label=p_labels[p])
    for p in range(pmax_expected)
]

leg_ax.legend(
    handles=legend_handles,
    title="SA variants",
    fontsize=15,
    title_fontsize=15,
    frameon=False,
    loc="upper right"
)

fig.supylabel(
    r"First consensus time $T_{\mathrm{eff}}\!\left(\mathbf{x}_t^{\operatorname{sol}}\right)$",
    fontsize=16,
    x=0.02
)

plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.show()

