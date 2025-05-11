import numpy as np
import matplotlib.pyplot as plt
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv
from tqdm import tqdm

def run_centralized_qlearning(env, n_episodes=50000, n_runs=30, gamma=0.95, alpha=0.1, epsilon=0.1):
    avg_lengths = np.zeros((n_runs, n_episodes // 100))
    n_actions = len(env.action_space)
    for run in tqdm(range(n_runs), desc="Centralized Q-learning runs"):
        Q = {}  # Diccionario: estado -> array de Q-values para cada acción
        for ep in range(n_episodes):
            state = env.reset()
            done = False
            steps = 0
            while not done:
                if state not in Q:
                    Q[state] = np.ones(n_actions)
                if np.random.rand() < epsilon:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(Q[state])
                real_action = env.action_space[action]
                next_state, reward, done = env.step(real_action)
                if next_state not in Q:
                    Q[next_state] = np.ones(n_actions)
                best_next = np.max(Q[next_state])
                Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
                state = next_state
                steps += 1
            if (ep + 1) % 100 == 0:
                avg_lengths[run, ep // 100] = steps
    return avg_lengths

def main():
    env = CentralizedHunterEnv()
    avg_lengths = run_centralized_qlearning(env)
    mean_curve = np.mean(avg_lengths, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(mean_curve)) * 100, mean_curve, label='Centralized Q-learning')
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Length (per 100 episodes)')
    plt.title('Centralized Q-learning in CentralizedHunterEnv')
    plt.legend()
    plt.grid(True)
    plt.savefig('centralized_hunter_curve.png')
    plt.close()

if __name__ == "__main__":
    main() 