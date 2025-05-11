import numpy as np
import matplotlib.pyplot as plt
from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv
from tqdm import tqdm

def run_decentralized_competitive_qlearning(env, n_episodes=50000, n_runs=30, gamma=0.95, alpha=0.1, epsilon=0.1):
    avg_lengths = np.zeros((n_runs, n_episodes // 100))
    n_actions = len(env.single_agent_action_space)
    for run in tqdm(range(n_runs), desc="Decentralized Q-learning (Competitive) runs"):
        # Q-tables como diccionarios, inicializados en 1.0 para cada nuevo estado
        Q1 = {}  # Cazador 1
        Q2 = {}  # Cazador 2
        Q3 = {}  # Presa
        for ep in range(n_episodes):
            print(f"Episodio: {ep}")
            state = env.reset()
            done = False
            steps = 0
            max_steps = 200  # Para evitar bucles infinitos
            while not done and steps < max_steps:
                # Inicialización optimista en 1.0
                if state not in Q1:
                    Q1[state] = np.ones(n_actions)
                if state not in Q2:
                    Q2[state] = np.ones(n_actions)
                if state not in Q3:
                    Q3[state] = np.ones(n_actions)
                # Cazador 1
                if np.random.rand() < epsilon:
                    a1 = np.random.randint(n_actions)
                else:
                    a1 = np.argmax(Q1[state])
                # Cazador 2
                if np.random.rand() < epsilon:
                    a2 = np.random.randint(n_actions)
                else:
                    a2 = np.argmax(Q2[state])
                # Presa
                if np.random.rand() < epsilon:
                    a3 = np.random.randint(n_actions)
                else:
                    a3 = np.argmax(Q3[state])
                actions = (a1, a2, a3)
                next_state, rewards, done = env.step(actions)  # Espera solo 3 valores
                r1, r2, r3 = rewards
                # Inicialización optimista para el siguiente estado
                if next_state not in Q1:
                    Q1[next_state] = np.ones(n_actions)
                if next_state not in Q2:
                    Q2[next_state] = np.ones(n_actions)
                if next_state not in Q3:
                    Q3[next_state] = np.ones(n_actions)
                best_next1 = np.max(Q1[next_state])
                best_next2 = np.max(Q2[next_state])
                best_next3 = np.max(Q3[next_state])
                Q1[state][a1] += alpha * (r1 + gamma * best_next1 - Q1[state][a1])
                Q2[state][a2] += alpha * (r2 + gamma * best_next2 - Q2[state][a2])
                Q3[state][a3] += alpha * (r3 + gamma * best_next3 - Q3[state][a3])
                state = next_state
                steps += 1
                # env.show()
            if (ep + 1) % 100 == 0:
                avg_lengths[run, ep // 100] = steps
    return avg_lengths

def main():
    env = HunterAndPreyEnv()
    avg_lengths = run_decentralized_competitive_qlearning(env)
    mean_curve = np.mean(avg_lengths, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(mean_curve)) * 100, mean_curve, label='Decentralized Q-learning (Competitive)')
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Length (per 100 episodes)')
    plt.title('Decentralized Q-learning in HunterAndPreyEnv (Competitive)')
    plt.legend()
    plt.grid(True)
    plt.savefig('decentralized_competitive_hunter_curve.png')
    plt.close()

if __name__ == "__main__":
    main() 