import numpy as np
import matplotlib.pyplot as plt
from Environments.MultiAgentEnvs.HunterEnv import HunterEnv
from tqdm import tqdm

def run_decentralized_qlearning(env, n_episodes=50000, n_runs=30, gamma=0.95, alpha=0.1, epsilon=0.1):
    avg_lengths = np.zeros((n_runs, n_episodes // 100))
    n_actions = len(env.single_agent_action_space)
    for run in tqdm(range(n_runs), desc="Decentralized Q-learning runs"):
        Q1 = {}  # Q-table for agent 1 (cazador 1)
        Q2 = {}  # Q-table for agent 2 (cazador 2)
        for ep in range(n_episodes):
            state = env.reset()
            done = False
            steps = 0
            max_steps = 200  # Seguridad para evitar bucles infinitos
            while not done and steps < max_steps:
                # Inicialización optimista en 1.0 para cada nuevo estado
                if state not in Q1:
                    Q1[state] = np.ones(n_actions)
                if state not in Q2:
                    Q2[state] = np.ones(n_actions)
                # Agente 1 (cazador 1)
                if np.random.rand() < epsilon:
                    a1 = np.random.randint(n_actions)
                else:
                    a1 = np.argmax(Q1[state])
                # Agente 2 (cazador 2)
                if np.random.rand() < epsilon:
                    a2 = np.random.randint(n_actions)
                else:
                    a2 = np.argmax(Q2[state])
                actions = (a1, a2)
                next_state, rewards, done = env.step(actions)
                r1, r2 = rewards
                # Inicialización optimista para el siguiente estado
                if next_state not in Q1:
                    Q1[next_state] = np.ones(n_actions)
                if next_state not in Q2:
                    Q2[next_state] = np.ones(n_actions)
                best_next1 = np.max(Q1[next_state])
                best_next2 = np.max(Q2[next_state])
                Q1[state][a1] += alpha * (r1 + gamma * best_next1 - Q1[state][a1])
                Q2[state][a2] += alpha * (r2 + gamma * best_next2 - Q2[state][a2])
                state = next_state
                steps += 1
            if (ep + 1) % 100 == 0:
                avg_lengths[run, ep // 100] = steps
    return avg_lengths

def main():
    env = HunterEnv()
    avg_lengths = run_decentralized_qlearning(env)
    mean_curve = np.mean(avg_lengths, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(mean_curve)) * 100, mean_curve, label='Decentralized Q-learning (Coop)')
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Length (per 100 episodes)')
    plt.title('Decentralized Q-learning in HunterEnv (Cooperative)')
    plt.legend()
    plt.grid(True)
    plt.savefig('decentralized_hunter_curve.png')
    plt.close()

if __name__ == "__main__":
    main() 