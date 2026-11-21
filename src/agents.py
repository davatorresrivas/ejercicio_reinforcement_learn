"""Entrenamiento tabular Q-Learning y SARSA."""
from collections import defaultdict
import numpy as np
from .environment import HelpDeskEnvironment, valid_actions_for_state, greedy_action

def epsilon_greedy(q_table, state, epsilon, rng):
    """Selecciona una acción válida con exploración epsilon-greedy."""
    candidates = valid_actions_for_state(state)
    if rng.random() < epsilon:
        return int(rng.choice(candidates))
    return greedy_action(q_table, state)


def train_agent(algorithm, episodes=10_000, alpha=0.1, gamma=0.95,
                epsilon_start=1.0, epsilon_min=0.05, seed=42):
    """Entrena Q-Learning o SARSA y devuelve tabla Q e historial."""
    if algorithm not in ("q_learning", "sarsa"):
        raise ValueError("Algoritmo no soportado")
    rng = np.random.default_rng(seed)
    env = HelpDeskEnvironment(seed=seed + 1_000_000)
    q_table = defaultdict(lambda: np.zeros(3, dtype=float))
    history = []
    for episode in range(episodes):
        state = env.reset()
        epsilon = max(epsilon_min,
                      epsilon_start - (epsilon_start - epsilon_min) * episode / max(1, episodes - 1))
        total_reward = 0.0
        action = epsilon_greedy(q_table, state, epsilon, rng)
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            if done:
                target = reward
                next_action = None
            elif algorithm == "q_learning":
                target = reward + gamma * max(
                    q_table[next_state][a + 1] for a in valid_actions_for_state(next_state)
                )
                next_action = None
            else:
                next_action = epsilon_greedy(q_table, next_state, epsilon, rng)
                target = reward + gamma * q_table[next_state][next_action + 1]
            idx = action + 1
            q_table[state][idx] += alpha * (target - q_table[state][idx])
            total_reward += reward
            state = next_state
            if not done:
                action = (next_action if algorithm == "sarsa" else
                          epsilon_greedy(q_table, state, epsilon, rng))
        history.append(total_reward)
    return q_table, np.asarray(history)
