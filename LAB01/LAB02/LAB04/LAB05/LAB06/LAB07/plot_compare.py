# plot_compare.py
import matplotlib.pyplot as plt

def plot_comparison(results, metric="val_accuracy", title="CNN Config Comparison"):
    plt.figure(figsize=(10, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, r in enumerate(results):
        history = r["history"]
        values = history[metric]
        label = f"e{r['epochs']}_L{r['conv_layers']}_F{r['filters']}"
        color = colors[i % len(colors)]

        plt.plot(range(1, len(values) + 1), values,
                 label=label, color=color, linewidth=2)

        last_epoch = len(values)
        last_value = values[-1]
        plt.annotate(
            f"{label} {last_value:.3f}",
            xy=(last_epoch, last_value),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
            va="center"
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png", dpi=150)
    plt.show()