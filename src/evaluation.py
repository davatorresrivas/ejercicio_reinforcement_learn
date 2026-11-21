"""Evaluación de políticas en jornadas de demanda compartida."""
import numpy as np
from .environment import HelpDeskEnvironment, greedy_action, DEMAND_LEVELS, DEMAND_MEANS

def evaluate_one(policy, arrivals, q_table=None):
    """Evalúa una jornada usando llegadas fijadas externamente."""
    env = HelpDeskEnvironment(arrivals=arrivals)
    state = env.reset()
    metrics = dict(reward=0.0, arrivals=0, attended=0, pending=0,
                   agent_intervals=0, staff_cost=0, queue_cost=0, terminal_cost=0)
    allocation = []
    while state is not None:
        action = 0 if policy == "Fija" else greedy_action(q_table, state)
        state, reward, done, info = env.step(action)
        metrics["reward"] += reward
        for key in ("arrivals", "attended", "agent_intervals", "staff_cost", "queue_cost", "terminal_cost"):
            source = {"agent_intervals": "agents"}.get(key, "cost_" + key.replace("_cost", "") if key.endswith("_cost") else key)
            metrics[key] += info[source]
        allocation.append(info["agents"])
    metrics["pending"] = env.pending
    metrics["service_rate"] = (metrics["attended"] / metrics["arrivals"]
                               if metrics["arrivals"] else 1.0)
    metrics["allocation"] = allocation
    return metrics


def generate_arrivals(seed):
    """Genera ocho llegadas de solicitudes, reproducibles por semilla."""
    return tuple(int(x) for x in np.random.default_rng(seed).poisson(
        [DEMAND_MEANS[level] for level in DEMAND_LEVELS]))
