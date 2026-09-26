# plot_compare.py
import matplotlib.pyplot as plt

def plot_comparison(results, metric="val_accuracy", title="CNN Config Comparison"):
    """
    results: list of dicts จาก compare_configs.py
             แต่ละ dict มี keys: epochs, conv_layers, filters, history
    metric: "val_accuracy" หรือ "val_loss" หรือ "accuracy" หรือ "loss"
    """
    plt.figure(figsize=(10, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, r in enumerate(results):
        history = r["history"]
        values = history[metric]
        label = f"e{r['epochs']}_L{r['conv_layers']}_F{r['filters']}"
        color = colors[i % len(colors)]

        # เส้นกราฟ
        plt.plot(range(1, len(values) + 1), values,
                 label=label, color=color, linewidth=2)

        # label ตัวเลขกำกับที่ปลายเส้น (แบบในภาพตัวอย่าง)
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


if __name__ == "__main__":
    # ตัวอย่างการเรียกใช้ร่วมกับ compare_configs.py
    from compare_configs import run_experiments

    results = run_experiments()

    plot_comparison(results, metric="val_accuracy", title="Validation Accuracy per Config")
    plot_comparison(results, metric="val_loss", title="Validation Loss per Config")