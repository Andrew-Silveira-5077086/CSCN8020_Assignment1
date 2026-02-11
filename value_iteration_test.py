import numpy as np
from gridworld import GridWorld
#from value_iteration import GridWorld
from value_iteration_agent import Agent

def main():
    ENV_SIZE = 5
    THETA_THRESHOLD = 0.05
    MAX_ITERATIONS = 1000
    env = GridWorld(ENV_SIZE)
    agent = Agent(env, THETA_THRESHOLD)

    # Perform value iteration
    done = False
    for iter in range(MAX_ITERATIONS):
        # Add stopping criteria if change in value function is small
        if done: break
        # Make a copy of the value function
        new_V = np.copy(agent.get_value_function())
        # Loop over all states
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                if not env.is_terminal_state(i, j):
                    new_V[i, j], _, _= agent.calculate_max_value(i,j)
        #done = agent.is_done(new_V)
        agent.update_value_function(new_V)

    # Print the optimal value function
    print("Optimal Value Function Found in %d iterations:"%(iter+1))
    print(agent.get_value_function())

    agent.update_greedy_policy()
    
    print("\nOptimal policy:")
    agent.print_policy()

if __name__=="__main__":
    main()