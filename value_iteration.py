import numpy as np

ENV_SIZE = 5

class GridWorld():

    def __init__(self, env_size):
        self.env_size = env_size
        # Initialize the value function and set terminal state value to 0
        self.V = np.zeros((env_size, env_size))
        self.terminal_state = (4, 4)
        self.V[self.terminal_state] = 0

        # Define the transition probabilities and rewards
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
        self.action_description = ["Right", "Left", "Down", "Up"]
        self.gamma = 1.0  # Discount factor
        self.reward = np.ones((self.env_size, self.env_size)) * -1  # Reward for non-terminal states
        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int)
        
        # Grey states
        grey_states = [(1,2), (3,0), (0,4)]
        
        for (i, j) in grey_states:
            self.reward[i, j] = -5
    
    '''@brief Checks if there is the change in V is less than preset threshold
    '''
    def is_done(self, i, j):
        pass
    
    '''@brief Returns True if the state is a terminal state
    '''
    def is_terminal_state(self, i, j):
        return (i, j) == self. terminal_state
    
    '''
    @brief Overwrites the current state-value function with a new one
    '''
    def update_value_function(self, V):
        self.V = np.copy(V)

    '''
    @brief Returns the full state-value function V_pi
    '''
    def get_value_function(self):
        return self.V

    '''@brief Returns the stored greedy policy
    '''
    def get_policy(self):
        return self.pi_greedy
    
    '''@brief Prints the policy using the action descriptions
    '''
    def print_policy(self):
        for row in self.pi_str:
            print(row)

    '''@brief Calculate the maximim value by following a greedy policy
    '''
    def calculate_max_value(self, i, j):
        # TODO: Find the maximum value for the current state using Bellman's equation
        # HINT #1 start with a - infinite value as the max
        max_value = float('-inf')
        best_action = None
        best_actions_str = ""
        # HINT #2: Loop over all actions
        for action_index in range(len(self.actions)):
            # TODO: Find Next state
            next_i, next_j, reward, done = self.step(action_index, i, j)
            
            # If invalid, stay in place
            if not self.is_valid_state(next_i, next_j):
                next_i, next_j = i, j
            # Calculate value function if state is valid
            #reward = self.reward[next_i, next_j]      
          
            value = reward + self.gamma * self.V[next_i, next_j]
          
            # TODO: Update the max_value as required
          
            if value > max_value:
                max_value = value
                best_action = action_index
                best_actions_str = self.action_description[action_index]
            elif value == max_value:
                best_actions_str += "|" + self.action_description[action_index]  
        
        
            '''
            TODO: Optional - You can also update the best action and best_actions_str to update the policy
            Otherwise, feel free to change the return values and add any extra methods to calculate the greedy policy
            '''

        return max_value, best_action, best_actions_str
    
    '''@brief Returns the next state given the chosen action and current state
    '''
    def step(self, action_index, i, j):
        # We are assuming a Transition Probability Matrix where
        # P(s'|s) = 1.0 for a single state and 0 otherwise
        action = self.actions[action_index]
        next_i, next_j = i + action[0], j + action[1]
        if not self.is_valid_state(next_i, next_j):
            next_i, next_j = i, j
        
        done = self.is_terminal_state(next_i, next_j)
        reward = self.reward[next_i, next_j]
        return next_i, next_j, reward, done
    
    '''@brief Checks if a state is within the acceptable bounds of the environment
    '''
    def is_valid_state(self, i, j):
        valid = 0 <= i < self.env_size and 0 <= j < self.env_size
        return valid
    
    def update_greedy_policy(self):
        self.pi_str = []
        
        for i in range(ENV_SIZE):
            row = []
            for j in range(ENV_SIZE):
                # TODO: calculate the greedy policy and populate self.pi_greedy
                if self.is_terminal_state(i, j):
                    self.pi_greedy[i, j] = -1
                    row.append("X")
                    continue

                # TODO: Optional - Add the optimal action description to self.pi_str to be able to print it
                _, best_action, best_actions_str = self.calculate_max_value(i, j)
                # Store numeric policy
                
                self.pi_greedy[i, j] = best_action

                # Store readable policy
                row.append(best_actions_str)
            self.pi_str.append(row)   

    def get_size(self):
        return self.env_size
    
    def get_actions(self):
        return self.actions

gridworld = GridWorld(ENV_SIZE)
# Perform value iteration
num_iterations = 1000

for _ in range(num_iterations):
    # TODO: Make a copy of the value function
    old_V = np.copy(gridworld.get_value_function())
    new_V = np.copy(old_V)
    
    # TODO: For all states, update the *copied* value function using GridWorld's calculate_max_value
    for i in range(ENV_SIZE):
        for j in range(ENV_SIZE):
            # Skip terminal state
            if gridworld.is_terminal_state(i, j):
                continue
            
            max_value, _, _ = gridworld.calculate_max_value(i, j)
            new_V[i, j] = max_value
    # TODO: After updating all states, update the value function using GridlWorld's update_value_function
    gridworld.update_value_function(new_V)
    
    # TODO: Add another stopping criteria (implement in GridWorld's is_done)
    if np.max(np.abs(new_V - old_V)) < 1e-4:
        break

# Print the optimal value function
print("Optimal Value Function:")
print(gridworld.get_value_function())

print("Updating greedy Function:")
gridworld.update_greedy_policy()

print("Printing policy:")
gridworld.print_policy()

