import numpy as np
from gridworld import GridWorld

ENV_SIZE = 5

class ValueIteration:
    def __init__(self, env, theta=0.001):
        self.env = env
        self.env_size = env.get_size()
        self.V = np.zeros((self.env_size, self.env_size))
        self.gamma = 1.0
        self.theta = theta
        self.policy = np.zeros((self.env_size, self.env_size), dtype=int)

    def run(self):
        while True:
            delta = 0
            new_V = np.copy(self.V)

            for i in range(self.env_size):
                for j in range(self.env_size):

                    if self.env.is_terminal_state(i,j):
                        continue

                    max_value = float("-inf")
                    best_action = 0

                    for a in range(len(self.env.actions)):
                        next_i, next_j, reward, _ = self.env.step(a, i, j)
                        value = reward + self.gamma * self.V[next_i, next_j]

                        if value > max_value:
                            max_value = value
                            best_action = a

                    new_V[i,j] = max_value
                    delta = max(delta, abs(self.V[i,j] - new_V[i,j]))

            self.V = new_V

            if delta < self.theta:
                break

        self.extract_policy()

    def extract_policy(self):
        for i in range(self.env_size):
            for j in range(self.env_size):

                if self.env.is_terminal_state(i,j):
                    self.policy[i,j] = -1
                    continue

                max_value = float("-inf")
                best_action = 0

                for a in range(len(self.env.actions)):
                    next_i, next_j, reward, _ = self.env.step(a, i, j)
                    value = reward + self.gamma * self.V[next_i, next_j]

                    if value > max_value:
                        max_value = value
                        best_action = a

                self.policy[i,j] = best_action


# Run value iteration
env = GridWorld(ENV_SIZE)
agent = ValueIteration(env)
agent.run()

print("Optimal Value Function:")
print(agent.V)

print("\nOptimal Policy (0=Right,1=Left,2=Down,3=Up,-1=Terminal):")
print(agent.policy)
