import numpy as np
import matplotlib.pyplot as plt
from Environments.MultiAgentEnvs.HunterEnv import HunterEnv
from tqdm import tqdm
import time

def state_to_tuple(state):
    """Convert state to a hashable tuple format"""
    return tuple(map(tuple, state))

def get_action_index(action, action_space):
    """Get the index of an action in the action space"""
    return action_space.index(action)

def get_action_from_index(index, action_space):
    """Get the action from its index in the action space"""
    return action_space[index]

def state_to_tuple(state):
    """Convert state to a hashable tuple format"""
    return tuple(map(tuple, state))

def run_decentralized_qlearning(env, n_episodes=50000, n_runs=30, gamma=0.95, alpha=0.1, epsilon=0.1):
    avg_lengths = np.zeros((n_runs, n_episodes // 100))
    action_space = env.single_agent_action_space
    n_actions = len(action_space)
    
    for run in tqdm(range(n_runs), desc="Decentralized Q-learning runs"):
        # Inicializar Q-tables fuera del loop de episodios
        Q1 = {}  # Q-table para el cazador 1
        Q2 = {}  # Q-table para el cazador 2
        
        for ep in range(n_episodes):
            state = env.reset()
            state = state_to_tuple(state)
            done = False
            steps = 0
            if run == 1:
                print("Episodio", ep)
            while not done:  # Límite de pasos reducido para evitar bucles infinitos
                # Inicialización optimista para estados nuevos
                if state not in Q1:
                    Q1[state] = np.ones(n_actions)
                if state not in Q2:
                    Q2[state] = np.ones(n_actions)
                
                # Selección de acciones usando epsilon-greedy
                if np.random.rand() < epsilon:
                    a1_idx = np.random.randint(n_actions)
                else:
                    a1_idx = np.argmax(Q1[state])
                
                if np.random.rand() < epsilon:
                    a2_idx = np.random.randint(n_actions)
                else:
                    a2_idx = np.argmax(Q2[state])
                
                # Convertir índices a acciones reales
                a1 = get_action_from_index(a1_idx, action_space)
                a2 = get_action_from_index(a2_idx, action_space)
                
                # Ejecutar acciones
                actions = (a1, a2)
                next_state, rewards, done = env.step(actions)
                next_state = state_to_tuple(next_state)
                r1, r2 = rewards
                
                # Inicialización optimista para el siguiente estado
                if next_state not in Q1:
                    Q1[next_state] = np.ones(n_actions)
                if next_state not in Q2:
                    Q2[next_state] = np.ones(n_actions)
                
                # Actualización Q-learning para cada agente
                best_next1 = np.max(Q1[next_state])
                best_next2 = np.max(Q2[next_state])
                
                Q1[state][a1_idx] += alpha * (r1 + gamma * best_next1 - Q1[state][a1_idx])
                Q2[state][a2_idx] += alpha * (r2 + gamma * best_next2 - Q2[state][a2_idx])
                
                state = next_state
                steps += 1
                # Mostrar el estado actual
                # env.show()
                # time.sleep(0.1)
                # Mostrar el estado actual solo en el último run y cada 10000 episodios
                if run == n_runs-1 and ep % 10000 == 0:
                    # env.show()
                    print(f"\nRun {run+1}, Episodio {ep}, Acciones: {actions}")
                    # time.sleep(0.1)
            
            if (ep + 1) % 100 == 0:
                avg_lengths[run, ep // 100] = steps
                print(f"\nEpisodio {ep + 1}, Pasos: {steps}")
    
    return avg_lengths

def main():
    env = HunterEnv()
    print("Iniciando entrenamiento Q-learning descentralizado (30 runs x 50000 episodios)...")
    avg_lengths = run_decentralized_qlearning(env)
    
    # Graficar resultados
    mean_curve = np.mean(avg_lengths, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(mean_curve)) * 100, mean_curve, label='Decentralized Q-learning (Coop)')
    plt.xlabel('Episodio')
    plt.ylabel('Duración promedio del episodio (por 100 episodios)')
    plt.title('Q-learning Descentralizado en HunterEnv (Cooperativo)')
    plt.legend()
    plt.grid(True)
    plt.savefig('decentralized_hunter_curve.png')
    plt.close()

if __name__ == "__main__":
    main() 