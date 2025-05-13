import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from tqdm import tqdm
from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from MemoryWrappers.KOrderMemory import KOrderMemory
from MemoryWrappers.BinaryMemory import BinaryMemory
from Environments.AbstractEnv import AbstractEnv

# Create a new memory wrapper: CustomBufferMemory
class CustomBufferMemory(AbstractEnv):
    """
    Custom buffer memory that allows the agent to decide when to save observations.
    
    The agent can choose between two control actions: "save" or "ignore".
    If "save" is chosen, the current observation is added to the buffer.
    If "ignore" is chosen, the current observation is not saved.
    
    The state provided to the agent consists of (current_observation, buffer_contents).
    """
    def __init__(self, env, buffer_size=1):
        self.__env = env
        self.__buffer_size = buffer_size
        self.__buffer = []
        self.__current_obs = None
        
    @property
    def action_space(self):
        env_actions = self.__env.action_space
        control_actions = ["save", "ignore"]
        return [(env_action, control_action) for env_action in env_actions 
                for control_action in control_actions]
    
    def reset(self):
        self.__current_obs = self.__env.reset()
        self.__buffer = []
        return self.__get_state()
    
    def __get_state(self):
        # State is current observation plus buffer contents
        return (self.__current_obs, tuple(self.__buffer))
    
    def step(self, action):
        env_action, control_action = action
        # Take action in environment
        next_obs, reward, done = self.__env.step(env_action)
        
        # Update buffer based on control action
        if control_action == "save" and len(self.__buffer) < self.__buffer_size:
            self.__buffer.append(self.__current_obs)
        elif control_action == "save" and len(self.__buffer) == self.__buffer_size:
            # If buffer is full, remove oldest observation
            self.__buffer.pop(0)
            self.__buffer.append(self.__current_obs)
            
        # Update current observation
        self.__current_obs = next_obs
        
        return self.__get_state(), reward, done
    
    def show(self):
        self.__env.show()
        print(f"Current obs: {self.__current_obs}, Buffer: {self.__buffer}")


# Agent implementations for our experiments
class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount=0.99, epsilon=0.01, init_value=1.0):
        self.action_space = action_space
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(lambda: init_value))
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space[np.random.randint(len(self.action_space))]
        return self.get_greedy_action(state)
    
    def get_greedy_action(self, state):
        state_key = self._get_state_key(state)
        return max(self.action_space, key=lambda a: self.Q[state_key][a])
    
    def update(self, state, action, reward, next_state, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Q-learning update (off-policy)
        best_next_action = max(self.action_space, key=lambda a: self.Q[next_state_key][a])
        best_next_value = self.Q[next_state_key][best_next_action]
        
        # If episode ended, no future value
        if done:
            best_next_value = 0
            
        # Q-learning update
        self.Q[state_key][action] += self.alpha * (
            reward + self.gamma * best_next_value - self.Q[state_key][action])
    
    def _get_state_key(self, state):
        # Convert state to hashable tuple for use as dictionary key
        return str(state)


class SarsaAgent(QLearningAgent):
    def __init__(self, action_space, learning_rate=0.1, discount=0.99, epsilon=0.01, init_value=1.0):
        super().__init__(action_space, learning_rate, discount, epsilon, init_value)
        
    def update(self, state, action, reward, next_state, next_action, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # SARSA update (on-policy)
        next_value = self.Q[next_state_key][next_action] if not done else 0
        
        # SARSA update
        self.Q[state_key][action] += self.alpha * (
            reward + self.gamma * next_value - self.Q[state_key][action])


class NStepSarsaAgent:
    def __init__(self, action_space, n_steps=16, learning_rate=0.1, discount=0.99, epsilon=0.01, init_value=1.0):
        self.action_space = action_space
        self.n = n_steps
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(lambda: init_value))
        self.buffer = []  # Store transitions (s, a, r)
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space[np.random.randint(len(self.action_space))]
        return self.get_greedy_action(state)
    
    def get_greedy_action(self, state):
        state_key = self._get_state_key(state)
        return max(self.action_space, key=lambda a: self.Q[state_key][a])
    
    def start_episode(self):
        self.buffer = []
        
    def add_transition(self, state, action, reward):
        self.buffer.append((state, action, reward))
        
    def update(self, current_state, current_action):
        # Only update if we have enough transitions in buffer
        if len(self.buffer) < self.n:
            return
            
        # Calculate n-step return
        G = 0
        for i in range(min(self.n, len(self.buffer))):
            G += self.gamma**i * self.buffer[i][2]  # buffer[i][2] is reward
            
        # Add bootstrap estimate if there are more states
        if len(self.buffer) > self.n:
            bootstrap_state = self.buffer[self.n][0]
            bootstrap_action = self.buffer[self.n][1]
            bootstrap_key = self._get_state_key(bootstrap_state)
            G += self.gamma**self.n * self.Q[bootstrap_key][bootstrap_action]
            
        # Update oldest state in buffer
        oldest_state = self.buffer[0][0]
        oldest_action = self.buffer[0][1]
        oldest_key = self._get_state_key(oldest_state)
        
        self.Q[oldest_key][oldest_action] += self.alpha * (G - self.Q[oldest_key][oldest_action])
        
        # Remove oldest transition
        self.buffer.pop(0)
        
    def end_episode(self):
        # Process remaining transitions at end of episode
        while self.buffer:
            # Calculate return for remainder of episode (no bootstrap)
            G = 0
            for i, (_, _, r) in enumerate(self.buffer):
                G += self.gamma**i * r
                
            # Update oldest state
            oldest_state = self.buffer[0][0]
            oldest_action = self.buffer[0][1]
            oldest_key = self._get_state_key(oldest_state)
            
            self.Q[oldest_key][oldest_action] += self.alpha * (G - self.Q[oldest_key][oldest_action])
            
            self.buffer.pop(0)
    
    def _get_state_key(self, state):
        # Convert state to hashable format
        return str(state)


# Run experiments
def run_qlearning(env, n_episodes=1000, n_runs=30):
    episode_lengths = np.zeros((n_runs, n_episodes))
    
    for run in tqdm(range(n_runs), desc="Q-Learning"):
        agent = QLearningAgent(env.action_space, learning_rate=0.1, discount=0.99, epsilon=0.01, init_value=1.0)
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            steps = 0
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                steps += 1
                
                # Prevent infinite loops
                if steps > 10000:
                    done = True
                
            episode_lengths[run, episode] = steps
    
    return np.mean(episode_lengths, axis=0)


def run_sarsa(env, n_episodes=1000, n_runs=30):
    episode_lengths = np.zeros((n_runs, n_episodes))
    
    for run in tqdm(range(n_runs), desc="SARSA"):
        agent = SarsaAgent(env.action_space, learning_rate=0.1, discount=0.99, epsilon=0.01, init_value=1.0)
        
        for episode in range(n_episodes):
            state = env.reset()
            action = agent.get_action(state)
            done = False
            steps = 0
            
            while not done:
                next_state, reward, done = env.step(action)
                next_action = agent.get_action(next_state) if not done else None
                agent.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                steps += 1
                
                # Prevent infinite loops
                if steps > 10000:
                    done = True
                
            episode_lengths[run, episode] = steps
    
    return np.mean(episode_lengths, axis=0)


def run_nstep_sarsa(env, n_steps=16, n_episodes=1000, n_runs=30):
    episode_lengths = np.zeros((n_runs, n_episodes))
    
    for run in tqdm(range(n_runs), desc=f"{n_steps}-step SARSA"):
        agent = NStepSarsaAgent(env.action_space, n_steps=n_steps, learning_rate=0.1, discount=0.99, epsilon=0.01, init_value=1.0)
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            steps = 0
            agent.start_episode()
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                # Add transition to buffer
                agent.add_transition(state, action, reward)
                
                # Update Q-values for oldest transition (if enough in buffer)
                agent.update(state, action)
                
                state = next_state
                steps += 1
                
                # Prevent infinite loops
                if steps > 10000:
                    done = True
            
            # Process remaining transitions at end of episode
            agent.end_episode()
            
            episode_lengths[run, episode] = steps
    
    return np.mean(episode_lengths, axis=0)


def plot_results(results, title, filename):
    plt.figure(figsize=(12, 8))
    
    for algorithm, data in results.items():
        plt.plot(range(len(data)), data, label=algorithm)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Length')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def main():
    n_episodes = 1000
    n_runs = 30
    
    # Experiment 1: No memory
    print("Running experiments without memory...")
    env = InvisibleDoorEnv()
    
    results_no_memory = {
        'Q-Learning': run_qlearning(env, n_episodes, n_runs),
        'SARSA': run_sarsa(env, n_episodes, n_runs),
        '16-step SARSA': run_nstep_sarsa(env, 16, n_episodes, n_runs)
    }
    
    plot_results(results_no_memory, 'InvisibleDoorEnv: Without Memory', 'invisible_door_no_memory.png')
    
    # Experiment 2: K-order memory
    print("\nRunning experiments with K-order memory (k=2)...")
    memory_size = 2
    env = InvisibleDoorEnv()
    env = KOrderMemory(env, memory_size)
    
    results_korder_memory = {
        'Q-Learning': run_qlearning(env, n_episodes, n_runs),
        'SARSA': run_sarsa(env, n_episodes, n_runs),
        '16-step SARSA': run_nstep_sarsa(env, 16, n_episodes, n_runs)
    }
    
    plot_results(results_korder_memory, 'InvisibleDoorEnv: With K-order Memory (k=2)', 'invisible_door_korder_memory.png')
    
    # Experiment 3: Binary memory
    print("\nRunning experiments with Binary memory (1 bit)...")
    num_of_bits = 1
    env = InvisibleDoorEnv()
    env = BinaryMemory(env, num_of_bits)
    
    results_binary_memory = {
        'Q-Learning': run_qlearning(env, n_episodes, n_runs),
        'SARSA': run_sarsa(env, n_episodes, n_runs),
        '16-step SARSA': run_nstep_sarsa(env, 16, n_episodes, n_runs)
    }
    
    plot_results(results_binary_memory, 'InvisibleDoorEnv: With Binary Memory (1 bit)', 'invisible_door_binary_memory.png')
    
    # Experiment 4: Custom Buffer memory
    print("\nRunning experiments with Custom Buffer memory (size=1)...")
    env = InvisibleDoorEnv()
    env = CustomBufferMemory(env, buffer_size=1)
    
    results_custom_memory = {
        'Q-Learning': run_qlearning(env, n_episodes, n_runs),
        'SARSA': run_sarsa(env, n_episodes, n_runs),
        '16-step SARSA': run_nstep_sarsa(env, 16, n_episodes, n_runs)
    }
    
    plot_results(results_custom_memory, 'InvisibleDoorEnv: With Custom Buffer Memory', 'invisible_door_custom_memory.png')
    
    # Compare the best algorithms across different memory types
    print("\nComparing best algorithms across different memory types...")
    best_results = {
        'No Memory': results_no_memory['16-step SARSA'],
        'K-order Memory': results_korder_memory['16-step SARSA'],
        'Binary Memory': results_binary_memory['16-step SARSA'],
        'Custom Buffer': results_custom_memory['16-step SARSA']
    }
    
    plot_results(best_results, 'InvisibleDoorEnv: Comparing Memory Types (16-step SARSA)', 'invisible_door_memory_comparison.png')
    
    print("All experiments completed!")


if __name__ == "__main__":
    main()
