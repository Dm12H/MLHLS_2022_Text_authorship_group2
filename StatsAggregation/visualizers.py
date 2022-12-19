import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def draw_boxplot(ax, data, labels=None, **_):
    sns.boxplot(ax=ax, data=data, y=labels)


def draw_distribution(ax, data, color=None):
    sns.kdeplot(data, ax=ax, color=color)


def draw_3d(ax, data, n_rows=10, labels=None, mode="ridge", overlap=True, **_):
    # TODO Bregman divergence kmeans for representative clusters
    n_rows = min(n_rows, len(data)) if n_rows is not None else len(data)
    # пока просто n штук, чтобы было видно
    data = data[:n_rows]

    subspec = ax.get_subplotspec()
    hspace = -.5 if overlap else 0
    gs = subspec.subgridspec(n_rows, 1, hspace=hspace)
    pal = sns.cubehelix_palette(n_rows, rot=-.25, light=.7)
    axs = gs.subplots()
    if mode == "ridge":
        mn = np.min([np.min(line) for line in data])
        mx = np.max([np.max(line) for line in data])
        ax.set_xlim(mn, mx)
    # ensuring axs is iterable
    try:
        iter(axs)
    except TypeError:
        axs = (axs,)
    for i, sub_ax in enumerate(axs):
        if mode == "ridge":
            sns.kdeplot(data[i], ax=sub_ax, bw_adjust=.5, clip=(mn, mx), color=pal[i], fill=True, alpha=1, linewidth=1.5)
            sns.kdeplot(data[i], ax=sub_ax, bw_adjust=.5, clip=(mn, mx), color="w")
            sub_ax.set_xlim(mn, mx)
            sub_ax.set_yticks([])
        elif mode == "hist":
            x = np.arange(len(data[i]))
            sns.barplot(x=x, y=data[i], ax=sub_ax, facecolor=pal[i], edgecolor=pal[i],  errorbar=None)
            ax.set_xticks([])
        else:
            raise ValueError("invalid mode")
        # remove x and y ticks
        sub_ax.set_xticks([])
        sub_ax.set_ylabel('')
        rect = sub_ax.patch
        rect.set_alpha(0)
        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            sub_ax.spines[s].set_visible(False)
    ax.set_yticks([])
    if labels:
        if len(labels) != n_rows:
            raise ValueError("labels should match number of rows for visualization")
        for label, sub_ax in zip(labels, axs):
            left, right = sub_ax.get_xlim()
            sub_ax.text(right, 0, label, fontsize=14, ha="left")


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create the data
    rs = np.random.RandomState(1979)
    x = rs.randn(500).reshape(10, -1)
    mn = np.max(x, axis=1)[..., np.newaxis]

    x -= mn
    x /= np.sum(x, axis=1)[..., np.newaxis]

    labels = [f"line_{i+1}" for i in range(10)]
    draw_3d(ax=ax, data=x, n_rows=10, labels=labels)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.show()
