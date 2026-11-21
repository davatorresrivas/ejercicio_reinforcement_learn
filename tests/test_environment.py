"""Pruebas de invariantes del entorno y evaluación."""
import unittest
from src.environment import HelpDeskEnvironment
from src.evaluation import evaluate_one, generate_arrivals

class TestHelpDesk(unittest.TestCase):
    def test_empty_day(self):
        env = HelpDeskEnvironment(arrivals=[0]*8)
        self.assertEqual(env.reset(), (0, 0, 0, 3))
        self.assertEqual(env.step(0)[1], -30)

    def test_zero_agents(self):
        env = HelpDeskEnvironment(arrivals=[5]*8)
        for action in (-1, -1, -1):
            _, _, _, info = env.step(action)
        self.assertEqual(env.agents, 0)
        self.assertEqual(env.pending, 7)
        self.assertEqual(info["attended"], 0)

    def test_conservation_capacity_and_terminal(self):
        env = HelpDeskEnvironment(seed=42)
        received = served = 0
        done = False
        while not done:
            _, _, done, info = env.step(0)
            received += info["arrivals"]
            served += info["attended"]
            self.assertLessEqual(info["attended"], 3 * info["agents"])
        self.assertEqual(received, served + env.pending)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_invalid_action(self):
        env = HelpDeskEnvironment(arrivals=[0]*8)
        with self.assertRaises(ValueError):
            env.step(3)

    def test_reproducible_fixed_policy(self):
        arrivals = generate_arrivals(123)
        a = evaluate_one("Fija", arrivals)
        b = evaluate_one("Fija", arrivals)
        self.assertEqual(a, b)
        self.assertEqual(a["arrivals"], a["attended"] + a["pending"])

if __name__ == "__main__":
    unittest.main()
