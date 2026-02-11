import numpy as np
import random
from gridworld import GridWorld

class OffPolicyMCAgent:

    def __init__(self, env, gamma=0.9):
        self.env = env
        self.env_size = env.get_size()
        self.gamma = gamma

        # Value function V(s)
        self.V = np.zeros((self.env_size, self.env_size))

        # Cumulative weights C(s) for weighted importance sampling
        self.C = np.zeros((self.env_size, self.env_size))

        # Target policy π(a|s): greedy w.r.t. V(s), stored as action indices
        self.num_actions = len(self.env.actions)
        self.pi = np.zeros((self.env_size, self.env_size), dtype=int)

        # Behavior policy b(a|s): uniform random over actions
        self.b_prob = 1.0 / self.num_actions

        # For readable printing
        self.pi_str = []

    def generate_episode(self):
        """Generate an episode using the behavior policy b(a|s) (uniform random)."""
        # Start from a random non-terminal state
        while True:
            i = random.randint(0, self.env_size - 1)
            j = random.randint(0, self.env_size - 1)
            if not self.env.is_terminal_state(i, j):
                break

        episode = []

        while True:
            # Behavior policy: choose random action
            action = random.randint(0, self.num_actions - 1)

            next_i, next_j, reward, done = self.env.step(action, i, j)

            episode.append((i, j, action, reward))

            if done:
                break

            i, j = next_i, next_j

        return episode

    def update_greedy_policy(self):
        """Update target policy π to be greedy w.r.t. current V(s)."""
        self.pi_str = []

        for i in range(self.env_size):
            row = []
            for j in range(self.env_size):

                if self.env.is_terminal_state(i, j):
                    self.pi[i, j] = -1
                    row.append("X")
                    continue

                best_value = float('-inf')
                best_action = 0
                best_str = ""

                for a in range(self.num_actions):
                    next_i, next_j, reward, done = self.env.step(a, i, j)
                    value = reward + self.gamma * self.V[next_i, next_j]

                    if value > best_value:
                        best_value = value
                        best_action = a
                        best_str = self.env.action_description[a]
                    elif value == best_value:
                        best_str += "|" + self.env.action_description[a]

                self.pi[i, j] = best_action
                row.append(best_str)

            self.pi_str.append(row)

    def run_off_policy_mc(self, max_episodes=5000):
        """Off-policy Monte Carlo prediction with importance sampling."""
        # Initialize target policy (e.g., greedy w.r.t. initial V = 0)
        self.update_greedy_policy()

        for _ in range(max_episodes):

            episode = self.generate_episode()
            G = 0.0
            W = 1.0

            # Process episode backwards
            for t in reversed(range(len(episode))):
                i, j, action, reward = episode[t]
                G = reward + self.gamma * G

                # If behavior took an action target policy would never take → stop
                if self.pi[i, j] != action:
                    break

                # Update cumulative weight
                self.C[i, j] += W

                # Weighted importance sampling update
                self.V[i, j] += (W / self.C[i, j]) * (G - self.V[i, j])

                # Update importance weight: π(a|s)/b(a|s)
                # π(a|s) = 1 for greedy action, 0 otherwise (we already checked equality)
                W *= 1.0 / self.b_prob

            # Improve target policy after each episode
            self.update_greedy_policy()

    def get_value_function(self):
        return self.V

    def print_policy(self):
        for row in self.pi_str:
            print(row)