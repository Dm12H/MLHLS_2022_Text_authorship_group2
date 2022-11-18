import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def draw_boxplot(ax, data, labels=None, **_):
    sns.boxplot(ax=ax, data=data, y=labels)


def draw_distribution(ax, data, color=None):
    sns.kdeplot(data, ax=ax, color=color)


def draw_ridge3d(ax, data, n_rows=10, labels=None, **_):
    # TODO Bregman divergence kmeans for representative clusters
    n_rows = min(n_rows, len(data)) if n_rows is not None else len(data)
    # пока просто n штук, чтобы было видно
    data = data[:n_rows]
    mn = np.min([np.min(line) for line in data])
    mx = np.max([np.max(line) for line in data])

    subspec = ax.get_subplotspec()
    gs = subspec.subgridspec(n_rows, 1, hspace=-.5)
    pal = sns.cubehelix_palette(n_rows, rot=-.25, light=.7)
    axs = gs.subplots()
    # ensuring axs is iterable
    try:
        iter(axs)
    except TypeError:
        axs = (axs,)
    ax.set_xlim(mn, mx)
    for i, sub_ax in enumerate(axs):
        sns.kdeplot(data[i], ax=sub_ax, bw_adjust=.5, clip=(mn,mx), color=pal[i], fill=True, alpha=1, linewidth=1.5)
        sns.kdeplot(data[i], ax=sub_ax, bw_adjust=.5, clip=(mn, mx), color="w")
        # remove x and y ticks
        sub_ax.set_yticks([])
        sub_ax.set_xticks([])
        sub_ax.set_xlim(mn, mx)
        sub_ax.set_ylabel('')
        rect = sub_ax.patch
        rect.set_alpha(0)
        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            sub_ax.spines[s].set_visible(False)
    if labels:
        if len(labels) != n_rows:
            raise ValueError("labels should match number of rows for visualization")
        for label, sub_ax in zip(labels, axs):
            sub_ax.text(mn, 0, label, fontsize=14, ha="right")


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create the data
    rs = np.random.RandomState(1979)
    x = rs.randn(500).reshape(10, -1)
    mn = np.max(x, axis=1)[..., np.newaxis]

    x -= mn
    x /= np.sum(x, axis=1)[..., np.newaxis]

    labels = [f"line_{i+1}" for i in range(10)]
    draw_ridge3d(ax=ax, data=x, n_rows=10, labels=labels)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.show()
