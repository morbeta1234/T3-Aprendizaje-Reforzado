import numpy as np
import matplotlib.pyplot as plt
from Environments.SimpleEnvs.CliffEnv import CliffEnv
from tqdm import tqdm
import time

def show(env, current_state, reward=None):
    env.show()
    print(f"Estado actual: {current_state}")
    if reward:
        print(f"Recompensa: {reward}")

class CliffWalkingAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount=1.0, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

def state_to_index(state, width):
    return state[0] * width + state[1]

def index_to_action(action_idx):
    actions = ["up", "right", "down", "left"]
    return actions[action_idx]

def run_q_learning(env, n_episodes=500, n_runs=100, visualize=False):
    returns = np.zeros((n_runs, n_episodes))
    
    for run in tqdm(range(n_runs), desc="Q-learning runs"):
        agent = CliffWalkingAgent(env._height * env._width, 4)
        
        for episode in range(n_episodes):
            state = env.reset()
            state_idx = state_to_index(state, env._width)
            done = False
            episode_return = 0
            
            if visualize and episode % 100 == 0:  # Visualizar cada 100 episodios
                print(f"\nEpisode {episode + 1}")
                show(env, state)
                time.sleep(0.5)
            
            while not done:
                action = agent.get_action(state_idx)
                next_state, reward, done = env.step(index_to_action(action))
                next_state_idx = state_to_index(next_state, env._width)
                
                # Actualización Q-learning 
                best_next_action = np.argmax(agent.Q[next_state_idx])
                agent.Q[state_idx, action] += agent.alpha * (
                    reward + agent.gamma * agent.Q[next_state_idx, best_next_action] - 
                    agent.Q[state_idx, action]
                )
                
                if visualize and episode % 100 == 0:
                    show(env, next_state, reward)
                    time.sleep(0.5)
                
                state_idx = next_state_idx
                episode_return += reward
                
            returns[run, episode] = episode_return
            
    return np.mean(returns, axis=0)

def run_sarsa(env, n_episodes=500, n_runs=100, visualize=False):
    returns = np.zeros((n_runs, n_episodes))
    
    for run in tqdm(range(n_runs), desc="Sarsa runs"):
        agent = CliffWalkingAgent(env._height * env._width, 4)
        
        for episode in range(n_episodes):
            state = env.reset()
            state_idx = state_to_index(state, env._width)
            action = agent.get_action(state_idx)
            done = False
            episode_return = 0
            
            if visualize and episode % 100 == 0:
                print(f"\nEpisode {episode + 1}")
                show(env, state)
                time.sleep(0.5)
            
            while not done:
                next_state, reward, done = env.step(index_to_action(action))
                next_state_idx = state_to_index(next_state, env._width)
                next_action = agent.get_action(next_state_idx)
                
                # Sarsa update
                agent.Q[state_idx, action] += agent.alpha * (
                    reward + agent.gamma * agent.Q[next_state_idx, next_action] - 
                    agent.Q[state_idx, action]
                )
                
                if visualize and episode % 100 == 0:
                    show(env, next_state, reward)
                    time.sleep(0.5)
                
                state_idx = next_state_idx
                action = next_action
                episode_return += reward
                
            returns[run, episode] = episode_return
            
    return np.mean(returns, axis=0)

def run_n_step_sarsa(env, n_steps=4, n_episodes=500, n_runs=100, visualize=False):
    returns = np.zeros((n_runs, n_episodes))
    
    for run in tqdm(range(n_runs), desc="4-step Sarsa runs"):
        agent = CliffWalkingAgent(env._height * env._width, 4)
        
        for episode in range(n_episodes):
            state = env.reset()
            state_idx = state_to_index(state, env._width)
            action = agent.get_action(state_idx)
            done = False
            episode_return = 0
            
            if visualize and episode % 100 == 0:
                print(f"\nEpisode {episode + 1}")
                show(env, state)
                time.sleep(0.5)
            
            # Inicializar almacenamiento de trayectoria
            states = [state_idx]
            actions = [action]
            rewards = [0]
            
            T = float('inf')
            t = 0
            
            while True:
                if t < T:
                    next_state, reward, done = env.step(index_to_action(action))
                    next_state_idx = state_to_index(next_state, env._width)
                    episode_return += reward
                    
                    if visualize and episode % 100 == 0:
                        show(env, next_state, reward)
                        time.sleep(0.5)
                    
                    states.append(next_state_idx)
                    rewards.append(reward)
                    
                    if done:
                        T = t + 1
                    else:
                        next_action = agent.get_action(next_state_idx)
                        actions.append(next_action)
                        action = next_action
                
                tau = t - n_steps + 1
                
                if tau >= 0:
                    # Calculate n-step return
                    G = sum([agent.gamma ** (i - tau - 1) * rewards[i] 
                            for i in range(tau + 1, min(tau + n_steps, T) + 1)])
                    
                    if tau + n_steps < T:
                        G += agent.gamma ** n_steps * agent.Q[states[tau + n_steps], actions[tau + n_steps]]
                    
                    # Update Q-value
                    agent.Q[states[tau], actions[tau]] += agent.alpha * (
                        G - agent.Q[states[tau], actions[tau]]
                    )
                
                if tau == T - 1:
                    break
                    
                t += 1
                
            returns[run, episode] = episode_return
            
    return np.mean(returns, axis=0)

def main():
    try:
        env = CliffEnv()
        n_episodes = 500
        n_runs = 100
        visualize = False  # Set to True to see the environment during training
        
        print("Iniciando experimentos...")
        
        # Run experiments
        print("\nEjecutando Q-learning...")
        q_learning_returns = run_q_learning(env, n_episodes, n_runs, visualize)
        print("\nEjecutando Sarsa...")
        sarsa_returns = run_sarsa(env, n_episodes, n_runs, visualize)
        print("\nEjecutando 4-step Sarsa...")
        n_step_sarsa_returns = run_n_step_sarsa(env, n_steps=4, n_episodes=n_episodes, n_runs=n_runs, visualize=visualize)
        
        print("\nGraficando resultados...")
        # Plot results
        plt.figure(figsize=(10, 6))
        episodes = range(1, n_episodes + 1)
        
        plt.plot(episodes, q_learning_returns, label='Q-learning')
        plt.plot(episodes, sarsa_returns, label='Sarsa')
        plt.plot(episodes, n_step_sarsa_returns, label='4-step Sarsa')
        
        plt.xlabel('Episodios')
        plt.ylabel('Retorno promedio')
        plt.title('Cliff Walking: Q-learning vs Sarsa vs 4-step Sarsa')
        plt.legend()
        plt.grid(True)
        plt.ylim(-200, 0)  # Set y-axis limit as requested
        plt.savefig('cliff_walking_comparison.png')
        plt.close()
        
        print("Done! Results saved in cliff_walking_comparison.png")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 