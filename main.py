"""Entrena, evalúa y guarda métricas/gráficos del experimento."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.agents import train_agent
from src.evaluation import evaluate_one, generate_arrivals

def run(episodes=10_000, test_days=1_000, robustness=True):
    output = Path("results/metrics")
    figures = Path("results/figures")
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    q, qr = train_agent("q_learning", episodes=episodes, seed=42)
    s, sr = train_agent("sarsa", episodes=episodes, seed=42)
    history = pd.DataFrame({"episode": np.arange(1, episodes + 1),
                            "Q-Learning": qr, "SARSA": sr})
    history.to_csv(output / "training_rewards.csv", index=False)
    ax = history[["Q-Learning", "SARSA"]].rolling(100, min_periods=1).mean().plot(figsize=(9, 5))
    ax.set(title="Recompensa media móvil de entrenamiento", xlabel="Episodio", ylabel="Recompensa")
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "learning_curves.png", dpi=160)
    plt.close(ax.figure)

    records = []
    for seed in range(10_000, 10_000 + test_days):
        arrivals = generate_arrivals(seed)
        for name, table in [("Fija", None), ("Q-Learning", q), ("SARSA", s)]:
            metrics = evaluate_one(name, arrivals, table)
            metrics.pop("allocation")
            records.append({"seed": seed, "policy": name, **metrics})
    results = pd.DataFrame(records)
    assert (results.groupby("seed")["arrivals"].nunique() == 1).all()
    assert (results.arrivals == results.attended + results.pending).all()
    results.to_csv(output / "evaluation_by_day.csv", index=False)
    columns = ["reward", "staff_cost", "queue_cost", "terminal_cost", "pending", "agent_intervals", "service_rate"]
    summary = results.groupby("policy")[columns].agg(["mean", "std"])
    summary.to_csv(output / "evaluation_summary.csv")
    print("\nEvaluación principal:\n", summary.round(2).to_string())
    for metric in ["reward", "staff_cost", "pending", "service_rate"]:
        ax = results.groupby("policy")[metric].mean().plot.bar(figsize=(7, 4), title=metric)
        ax.set_xlabel("Política")
        ax.figure.tight_layout()
        ax.figure.savefig(figures / f"{metric}.png", dpi=160)
        plt.close(ax.figure)
    if robustness:
        robust = []
        for training_seed in [11, 22, 33, 44, 55]:
            for algorithm, label in [("q_learning", "Q-Learning"), ("sarsa", "SARSA")]:
                table, _ = train_agent(algorithm, episodes=episodes, seed=training_seed)
                for test_seed in range(20_000, 20_200):
                    result = evaluate_one(label, generate_arrivals(test_seed), table)
                    robust.append({"train_seed": training_seed, "policy": label,
                                   "reward": result["reward"], "pending": result["pending"]})
        robust_df = pd.DataFrame(robust)
        robust_df.to_csv(output / "robustness_by_day.csv", index=False)
        robust_summary = robust_df.groupby(["policy", "train_seed"])[["reward", "pending"]].mean()
        robust_summary.to_csv(output / "robustness_summary.csv")
        print("\nRobustez por semilla:\n", robust_summary.round(2).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimento RL para mesa de ayuda")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--test-days", type=int, default=1_000)
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.test_days < 1:
        parser.error("--episodes y --test-days deben ser positivos")
    run(args.episodes, args.test_days, not args.skip_robustness)
