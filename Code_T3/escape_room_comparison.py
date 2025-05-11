import numpy as np
import matplotlib.pyplot as plt
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
from tqdm import tqdm

class DynaAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.5, discount=1.0, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}  # Modelo del ambiente
        self.experiences = []  # Experiencias almacenadas
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update_model(self, state, action, next_state, reward):
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = (next_state, reward)
        self.experiences.append((state, action, next_state, reward))
    
    def planning_step(self):
        if not self.experiences:
            return
        
        # Seleccionar una experiencia aleatoria
        state, action, next_state, reward = self.experiences[np.random.randint(len(self.experiences))]
        
        # Actualizar Q-value usando el modelo
        best_next_action = np.argmax(self.Q[next_state])
        self.Q[state, action] += self.alpha * (
            reward + self.gamma * self.Q[next_state, best_next_action] - 
            self.Q[state, action]
        )

def value_iteration(Np, Nt, Rmax, n_states, n_actions, gamma, m, fict_terminal_idx, theta=1e-5):
    V = np.zeros(n_states + 1)  # +1 para el estado terminal ficticio
    policy = np.zeros(n_states, dtype=int)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            Qs = np.zeros(n_actions)
            for a in range(n_actions):
                if Nt.get((s, a), 0) < m:
                    # No conocido: transición a estado terminal ficticio
                    Qs[a] = Rmax + gamma * V[fict_terminal_idx]
                else:
                    # Conocido: usar modelo empírico
                    total = Nt[(s, a)]
                    Q = 0
                    for (sp, r), count in Np[(s, a)].items():
                        prob = count / total
                        Q += prob * (r + gamma * V[sp])
                    Qs[a] = Q
            V[s] = np.max(Qs)
            policy[s] = np.argmax(Qs)
            delta = max(delta, abs(v - V[s]))
        # El estado terminal ficticio siempre tiene valor 0
        V[fict_terminal_idx] = 0
        if delta < theta:
            break
    return V[:n_states], policy

class RMaxAgentVI:
    """
    Agente R-Max con Value Iteration.
    - En cada paso real:
        1. Ejecuta value iteration sobre el modelo aprendido (planificar).
        2. Elige la acción greedy respecto a la política calculada.
        3. Ejecuta la acción en el entorno y observa la transición.
        4. Actualiza el modelo (contadores N_t y N_p).
    """
    def __init__(self, n_states, n_actions, gamma=1.0, Rmax=1, m=1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Rmax = Rmax
        self.m = m
        self.Nt = {}  # (s, a) -> count
        self.Np = {}  # (s, a) -> {(s', r): count}
        self.fict_terminal_idx = n_states  # índice del estado terminal ficticio
        self.V = np.zeros(n_states)
        self.policy = np.zeros(n_states, dtype=int)
    
    def update_model(self, s, a, sp, r):
        """Actualiza los contadores del modelo con la transición observada."""
        key = (s, a)
        if key not in self.Nt:
            self.Nt[key] = 0
            self.Np[key] = {}
        self.Nt[key] += 1
        spr = (sp, r)
        if spr not in self.Np[key]:
            self.Np[key][spr] = 0
        self.Np[key][spr] += 1
    
    def plan(self):
        """Ejecuta value iteration sobre el modelo aprendido para actualizar V y la política."""
        self.V, self.policy = value_iteration(self.Np, self.Nt, self.Rmax, self.n_states, self.n_actions, self.gamma, self.m, self.fict_terminal_idx)
    
    def get_action(self, s):
        """Devuelve la acción greedy respecto a la política calculada."""
        return self.policy[s]

    def get_policy(self):
        """Devuelve la política actual (array de acciones por estado)."""
        return self.policy.copy()

    def print_policy(self):
        """Imprime la política actual de forma legible."""
        print("Política actual (acción por estado):")
        print(self.policy)

def state_to_index(state, width, height):
    # state es una tupla (row, col, has_key)
    position_idx = state[0] * width + state[1]
    return position_idx + (state[2] * width * height)  # has_key determina el offset

def run_dyna(env, n_episodes=20, n_runs=5, n_planning_steps=0):
    n_states = env._height * env._width * 2  # Estados totales = posiciones * 2 (con/sin llave)
    returns = np.zeros((n_runs, n_episodes))
    
    for run in tqdm(range(n_runs), desc=f"Dyna runs (planning steps: {n_planning_steps})"):
        agent = DynaAgent(n_states, 4, learning_rate=0.5, discount=1.0, epsilon=0.1)
        
        for episode in range(n_episodes):
            state = env.reset()  # El agente comienza en el cuadrado celeste sin la llave
            state_idx = state_to_index(state, env._width, env._height)
            done = False
            episode_return = 0
            
            while not done:
                action = agent.get_action(state_idx)
                next_state, reward, done = env.step(index_to_action(action))
                next_state_idx = state_to_index(next_state, env._width, env._height)
                
                # Actualizar Q-value
                best_next_action = np.argmax(agent.Q[next_state_idx])
                agent.Q[state_idx, action] += agent.alpha * (
                    reward + agent.gamma * agent.Q[next_state_idx, best_next_action] - 
                    agent.Q[state_idx, action]
                )
                
                # Actualizar modelo
                agent.update_model(state_idx, action, next_state_idx, reward)
                
                # Pasos de planning
                for _ in range(n_planning_steps):
                    agent.planning_step()
                
                state_idx = next_state_idx
                episode_return += reward
                
            returns[run, episode] = episode_return
            
    return np.mean(returns, axis=0)

def run_rmax(env, n_episodes=20, n_runs=5):
    n_states = env._height * env._width * 2
    n_actions = 4
    returns = np.zeros((n_runs, n_episodes))
    m = 1  # umbral de visitas
    for run in tqdm(range(n_runs), desc="RMax-VI runs"):
        agent = RMaxAgentVI(n_states, n_actions, gamma=1.0, Rmax=1, m=m)
        for episode in range(n_episodes):
            print(f"Episode number: {episode}")
            state = env.reset()
            state_idx = state_to_index(state, env._width, env._height)
            done = False
            episode_return = 0
            while not done:
                # 1. Planificar (value iteration)
                agent.plan()
                # 2. Elegir acción greedy
                action = agent.get_action(state_idx)
                # 3. Ejecutar acción y observar transición
                next_state, reward, done = env.step(index_to_action(action))
                next_state_idx = state_to_index(next_state, env._width, env._height)
                # 4. Actualizar modelo
                agent.update_model(state_idx, action, next_state_idx, reward)
                # env.show()  # Mostrar el entorno después de cada acción (opcional)
                state_idx = next_state_idx
                episode_return += reward
            returns[run, episode] = episode_return
        # (Opcional) Imprimir la política final aprendida por el agente en este run
        print("Política final del agente en este run:")
        agent.print_policy()
    return np.mean(returns, axis=0)

def index_to_action(action_idx):
    actions = ["up", "right", "down", "left"]
    return actions[action_idx]

def main():
    env = EscapeRoomEnv()
    n_episodes = 20
    n_runs = 5
    planning_steps = [0, 1, 10, 100, 1000, 10000]
    # Almacenar resultados
    dyna_results = []
    print("Starting experiments...")
    print("Environment description:")
    print("- Agent starts in the blue square without the key")
    print("- Goal: Get the key (yellow square) and reach the door (brown square)")
    print("- Rewards: -10 for stepping on nails (gray squares), -1 otherwise")
    print("- Episode ends when the agent reaches the door with the key")
    # Ejecutar experimentos para cada número de pasos de planning SOLO para Dyna
    for n_steps in planning_steps:
        print(f"\nRunning experiments with {n_steps} planning steps (Dyna only)...")
        # Dyna
        dyna_returns = run_dyna(env, n_episodes, n_runs, n_steps)
        dyna_results.append(np.mean(dyna_returns))
    # RMax (solo una vez)
    rmax_result = run_rmax(env, n_episodes, n_runs)
    # Mostrar resultados en tabla
    print("\nResults (Average Return per Episode):")
    print("-" * 50)
    print(f"{'Planning Steps':<15} {'Dyna':<15}")
    print("-" * 50)
    for steps, dyna in zip(planning_steps, dyna_results):
        print(f"{steps:<15} {dyna:<15.2f}")
    print("-" * 50)
    print(f"{'RMax':<15} {np.mean(rmax_result):<15.2f}")
    print("-" * 50)
    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(planning_steps, dyna_results, 'b-o', label='Dyna')
    plt.axhline(y=np.mean(rmax_result), color='r', linestyle='--', label='RMax')
    plt.xlabel('Number of Planning Steps (Dyna)')
    plt.ylabel('Average Return per Episode')
    plt.title('Dyna vs RMax in Escape Room Environment')
    plt.legend()
    plt.grid(True)
    plt.savefig('escape_room_comparison.png')
    plt.close()
    # Justificación teórica
    print("\n¿Dyna se vuelve equivalente a RMax con muchos planning steps?")
    print("No exactamente. Dyna con muchos pasos de planning se acerca a un agente modelo basado en planificación completa, pero sigue usando Q-learning (actualización local y por muestras). RMax realiza value iteration global sobre el modelo aprendido, propagando la información de manera más eficiente y optimista. Si Dyna hiciera infinitos pasos de planning y el modelo fuera perfecto, se acercaría a la política óptima, pero la diferencia clave es que Dyna propaga información localmente, mientras que RMax lo hace globalmente en cada paso. En la práctica, con suficientes pasos de planning, Dyna puede aproximarse mucho al rendimiento de RMax, pero no es estrictamente equivalente.")

if __name__ == "__main__":
    main() 