"""Entorno discreto de la mesa de ayuda."""
import numpy as np

DEMAND_LEVELS = (0, 0, 1, 2, 2, 1, 1, 0)
DEMAND_MEANS = (3, 7, 12)
ACTIONS = (-1, 0, 1)

class HelpDeskEnvironment:
    """Simulador discreto de asignación de agentes de soporte."""

    def __init__(self, seed=42, arrivals=None):
        self.max_agents = 5
        self.capacity_per_agent = 3
        self.max_steps = 8
        self.rng = np.random.default_rng(seed)
        self.arrivals_override = arrivals
        self.reset()

    def reset(self):
        """Reinicia el episodio y devuelve el estado inicial."""
        self.time = 0
        self.pending = 0
        self.agents = 3
        if self.arrivals_override is None:
            self.arrivals = tuple(int(self.rng.poisson(DEMAND_MEANS[level]))
                                  for level in DEMAND_LEVELS)
        else:
            if len(self.arrivals_override) != self.max_steps or any(
                int(x) != x or x < 0 for x in self.arrivals_override
            ):
                raise ValueError("Se requieren 8 llegadas enteras no negativas")
            self.arrivals = tuple(int(x) for x in self.arrivals_override)
        return self.get_state()

    def get_state(self):
        """Estado observable; None indica episodio finalizado."""
        if self.time >= self.max_steps:
            return None
        return (self.time, DEMAND_LEVELS[self.time], self.pending, self.agents)

    def valid_actions(self):
        """Acciones que mantienen los agentes en el rango permitido."""
        if self.time >= self.max_steps:
            return []
        return [a for a in ACTIONS if 0 <= self.agents + a <= self.max_agents]

    def step(self, action):
        """Aplica una decisión, simula atención y retorna (estado, recompensa, fin, info)."""
        if self.time >= self.max_steps:
            raise RuntimeError("El episodio ya terminó; llama a reset()")
        if action not in self.valid_actions():
            raise ValueError(f"Acción inválida: {action}")
        self.agents += action
        arrivals = self.arrivals[self.time]
        available = self.pending + arrivals
        attended = min(available, self.agents * self.capacity_per_agent)
        self.pending = available - attended
        cost_staff = 10 * self.agents
        cost_queue = 5 * self.pending
        self.time += 1
        done = self.time == self.max_steps
        cost_terminal = 10 * self.pending if done else 0
        reward = -(cost_staff + cost_queue + cost_terminal)
        info = dict(arrivals=arrivals, attended=attended, pending=self.pending,
                    agents=self.agents, cost_staff=cost_staff,
                    cost_queue=cost_queue, cost_terminal=cost_terminal)
        return self.get_state(), reward, done, info


def valid_actions_for_state(state, max_agents=5):
    """Máscara de acciones válidas a partir del estado observado."""
    agents = state[3]
    return [a for a in ACTIONS if 0 <= agents + a <= max_agents]


def greedy_action(q_table, state):
    """Elige la acción válida de mayor valor; empates favorecen mantener personal."""
    candidates = valid_actions_for_state(state)
    values = q_table[state]
    return max(candidates, key=lambda a: (values[a + 1], -abs(a), -a))
