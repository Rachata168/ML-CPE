# plot_results.py
import matplotlib.pyplot as plt

def _plot_with_labels(ax, series_dict, title, ylabel):
    """วาดกราฟเส้นพร้อม label ตัวเลขกำกับปลายเส้น (สไตล์เดียวกับ plot_compare.py)"""
    colors = ["#e74c3c", "#3498db"]  # แดง = train, ฟ้า = val

    for i, (label, values) in enumerate(series_dict.items()):
        color = colors[i % len(colors)]
        epochs_range = range(1, len(values) + 1)

        ax.plot(epochs_range, values, label=label, color=color, linewidth=2)

        last_epoch = len(values)
        last_value = values[-1]
        ax.annotate(
            f"{label} {last_value:.3f}",
            xy=(last_epoch, last_value),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
            va="center"
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_history(history, title="Model"):
    """
    history: dict จาก model.fit() -> history.history
    วาด 2 กราฟ: accuracy (train vs val) และ loss (train vs val)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _plot_with_labels(
        axes[0],
        {"train acc": history["accuracy"], "val acc": history["val_accuracy"]},
        title=f"{title} - Accuracy",
        ylabel="Accuracy"
    )

    _plot_with_labels(
        axes[1],
        {"train loss": history["loss"], "val loss": history["val_loss"]},
        title=f"{title} - Loss",
        ylabel="Loss"
    )

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()