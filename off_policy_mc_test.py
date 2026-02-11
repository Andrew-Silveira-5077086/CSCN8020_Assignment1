from gridworld import GridWorld
from off_policy_mc_agent import OffPolicyMCAgent

def main():
    ENV_SIZE = 5
    env = GridWorld(ENV_SIZE)

    agent = OffPolicyMCAgent(env, gamma=0.9)
    agent.run_off_policy_mc(max_episodes=5000)

    print("Off-Policy MC Value Function (Importance Sampling):")
    print(agent.get_value_function())

    print("\nGreedy Policy from Off-Policy MC:")
    agent.print_policy()

if __name__ == "__main__":
    main()