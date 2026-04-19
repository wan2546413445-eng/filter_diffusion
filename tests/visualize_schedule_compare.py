import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_CURVE_PROFILE = {
    "linear": [
        1.00000000, 0.95000000, 0.90000000, 0.85000000, 0.80000000,
        0.75000000, 0.70000000, 0.65000000, 0.60000000, 0.55000000,
        0.50000000, 0.45000000, 0.40000000, 0.35000000, 0.30000000,
        0.25000000, 0.20000000, 0.15000000, 0.10000000, 0.05000000,
    ],
    "sparse": [
        1.00000000, 0.98787021, 0.97059083, 0.95115667, 0.93037302,
        0.90817255, 0.88399033, 0.85705806, 0.82661769, 0.79205430,
        0.75294826, 0.70904674, 0.66015452, 0.60594406, 0.54568489,
        0.47789235, 0.39989555, 0.30732471, 0.19351777, 0.04884625,
    ],
    "dense": [
        1.00000000, 0.85648223, 0.74267529, 0.65010445, 0.57210765,
        0.50431511, 0.44405594, 0.38984548, 0.34095326, 0.29705174,
        0.25794570, 0.22338231, 0.19294194, 0.16600967, 0.14182745,
        0.11962698, 0.09884333, 0.07940917, 0.06212979, 0.04913874,
    ],
}


def build_exact_curve(schedule_type: str, timesteps: int):
    base = np.asarray(_CURVE_PROFILE[schedule_type], dtype=np.float32)

    if timesteps == 19:
        curve = base
    elif timesteps == 20:
        curve = np.concatenate([base, [base[-1]]], axis=0)
    else:
        raise ValueError("Only timesteps=19 or 20 is supported for exact step visualization.")

    x = np.arange(len(curve))
    return x, curve


def main():
    timesteps = 19   # 改成 19 也可以
    save_dir = Path("./schedule_vis_outputs")
    save_dir.mkdir(parents=True, exist_ok=True)

    x_linear, y_linear = build_exact_curve("linear", timesteps)
    x_sparse, y_sparse = build_exact_curve("sparse", timesteps)
    x_dense, y_dense = build_exact_curve("dense", timesteps)

    plt.figure(figsize=(10, 6))
    plt.plot(x_linear, y_linear, "-o", label="linear")
    plt.plot(x_sparse, y_sparse, "-^", label="sparse")
    plt.plot(x_dense, y_dense, "-s", label="dense")
    plt.axhline(0.05, linestyle="--", label="r_min=0.05")

    for xi, yi in zip(x_linear, y_linear):
        plt.text(xi, yi + 0.015, f"{yi:.5f}", fontsize=8, ha="center")
    for xi, yi in zip(x_sparse, y_sparse):
        plt.text(xi, yi + 0.015, f"{yi:.5f}", fontsize=8, ha="center")
    for xi, yi in zip(x_dense, y_dense):
        plt.text(xi, yi - 0.04, f"{yi:.5f}", fontsize=8, ha="center")

    plt.xlabel("Timesteps")
    plt.ylabel("Center Frequency Ratio")
    plt.title(f"Exact step-matched schedule (timesteps={timesteps})")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_path = save_dir / f"exact_schedule_timesteps_{timesteps}.png"
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"Saved figure to: {save_path}")
    print("\nDense exact table:")
    for i, v in enumerate(y_dense):
        print(f"t={i:2d} -> {v:.2f}")


if __name__ == "__main__":
    main()