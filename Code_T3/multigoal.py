import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from tqdm import tqdm
from Environments.MultiGoalEnvs.RoomEnv import RoomEnv

# Estado en RoomEnv es: (posición_del_agente, objetivo_actual)
# Necesitamos adaptar los algoritmos para trabajar con este formato

class StandardQLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount=0.99, epsilon=0.1, init_value=1.0):
        self.action_space = action_space
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(lambda: init_value))
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
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
        
        # Si el episodio terminó, no hay valor futuro
        if done:
            best_next_value = 0
            
        # Actualización Q-learning
        self.Q[state_key][action] += self.alpha * (
            reward + self.gamma * best_next_value - self.Q[state_key][action])
    
    def _get_state_key(self, state):
        # En RoomEnv, state es (posición, objetivo)
        # Convertimos a tupla para usar como clave en diccionario
        return tuple(state[0]), tuple(state[1])


class MultiGoalQLearningAgent(StandardQLearningAgent):
    def __init__(self, action_space, goals, learning_rate=0.1, discount=0.99, epsilon=0.1, init_value=1.0):
        super().__init__(action_space, learning_rate, discount, epsilon, init_value)
        # Guardamos la lista de todas las metas posibles
        self.goals = goals
        # Q[(estado, meta)][acción] para todas las metas
        self.Q = defaultdict(lambda: defaultdict(lambda: init_value))
    
    def get_action(self, state):
        # Usamos solo la meta actual para seleccionar la acción
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        return self.get_greedy_action(state)
    
    def get_greedy_action(self, state):
        # Basamos la decisión solo en la meta actual
        state_key = self._get_state_key(state)
        return max(self.action_space, key=lambda a: self.Q[state_key][a])
    
    def update(self, state, action, reward, next_state, done):
        position, current_goal = state
        next_position, next_goal = next_state
        
        # Actualizar Q-values para TODAS las metas
        for goal in self.goals:
            # Construir claves para estado y siguiente estado con esta meta
            state_key = (tuple(position), tuple(goal))
            next_state_key = (tuple(next_position), tuple(goal))
            
            # Calcular recompensa para esta meta (1 si posición = meta, 0 en otro caso)
            goal_reward = 1.0 if next_position == goal else 0.0
            
            # Q-learning update (off-policy) para esta meta específica
            best_next_action = max(self.action_space, key=lambda a: self.Q[next_state_key][a])
            best_next_value = self.Q[next_state_key][best_next_action]
            
            # Si llegamos a esta meta, no hay valor futuro
            goal_done = next_position == goal
            if goal_done:
                best_next_value = 0
                
            # Actualización Q-learning para esta meta
            self.Q[state_key][action] += self.alpha * (
                goal_reward + self.gamma * best_next_value - self.Q[state_key][action])
    
    def _get_state_key(self, state):
        # Para la selección de acciones, usamos la posición y meta actual
        position, current_goal = state
        return tuple(position), tuple(current_goal)


class StandardSarsaAgent(StandardQLearningAgent):
    def __init__(self, action_space, learning_rate=0.1, discount=0.99, epsilon=0.1, init_value=1.0):
        super().__init__(action_space, learning_rate, discount, epsilon, init_value)
        
    def update(self, state, action, reward, next_state, next_action, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # SARSA update (on-policy)
        next_value = self.Q[next_state_key][next_action] if not done else 0
        
        # Actualización SARSA
        self.Q[state_key][action] += self.alpha * (
            reward + self.gamma * next_value - self.Q[state_key][action])


class MultiGoalSarsaAgent(MultiGoalQLearningAgent):
    def __init__(self, action_space, goals, learning_rate=0.1, discount=0.99, epsilon=0.1, init_value=1.0):
        super().__init__(action_space, goals, learning_rate, discount, epsilon, init_value)
        
    def update(self, state, action, reward, next_state, next_action, done):
        position, current_goal = state
        next_position, next_goal = next_state
        
        # Actualizar Q-values para TODAS las metas
        for goal in self.goals:
            # Construir claves para estado y siguiente estado con esta meta
            state_key = (tuple(position), tuple(goal))
            next_state_key = (tuple(next_position), tuple(goal))
            
            # Calcular recompensa para esta meta (1 si posición = meta, 0 en otro caso)
            goal_reward = 1.0 if next_position == goal else 0.0
            
            # SARSA update (on-policy) para esta meta específica
            goal_done = next_position == goal
            next_value = self.Q[next_state_key][next_action] if not goal_done else 0
                
            # Actualización SARSA para esta meta
            self.Q[state_key][action] += self.alpha * (
                goal_reward + self.gamma * next_value - self.Q[state_key][action])


class StandardNStepSarsaAgent:
    def __init__(self, action_space, n_steps=8, learning_rate=0.1, discount=0.99, epsilon=0.1, init_value=1.0):
        self.action_space = action_space
        self.n = n_steps
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(lambda: init_value))
        self.buffer = []  # Almacenar transiciones (s, a, r)
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        return self.get_greedy_action(state)
    
    def get_greedy_action(self, state):
        state_key = self._get_state_key(state)
        return max(self.action_space, key=lambda a: self.Q[state_key][a])
    
    def start_episode(self):
        self.buffer = []
        
    def add_transition(self, state, action, reward):
        self.buffer.append((state, action, reward))
        
    def update(self, current_state, current_action):
        # Solo actualizamos si tenemos suficientes transiciones en el buffer
        if len(self.buffer) < self.n:
            return
            
        # Calcular el retorno n-step
        G = 0
        for i in range(min(self.n, len(self.buffer))):
            G += self.gamma**i * self.buffer[i][2]  # buffer[i][2] es la recompensa
            
        # Añadir estimación bootstrap si hay más estados
        if len(self.buffer) > self.n:
            bootstrap_state = self.buffer[self.n][0]
            bootstrap_action = self.buffer[self.n][1]
            bootstrap_key = self._get_state_key(bootstrap_state)
            G += self.gamma**self.n * self.Q[bootstrap_key][bootstrap_action]
            
        # Actualizar el estado más antiguo del buffer
        oldest_state = self.buffer[0][0]
        oldest_action = self.buffer[0][1]
        oldest_key = self._get_state_key(oldest_state)
        
        self.Q[oldest_key][oldest_action] += self.alpha * (G - self.Q[oldest_key][oldest_action])
        
        # Eliminar la transición más antigua
        self.buffer.pop(0)
        
    def end_episode(self):
        # Procesar todas las transiciones restantes al final del episodio
        while self.buffer:
            # Calcular retorno para el resto del episodio (sin bootstrap)
            G = 0
            for i, (_, _, r) in enumerate(self.buffer):
                G += self.gamma**i * r
                
            # Actualizar el estado más antiguo
            oldest_state = self.buffer[0][0]
            oldest_action = self.buffer[0][1]
            oldest_key = self._get_state_key(oldest_state)
            
            self.Q[oldest_key][oldest_action] += self.alpha * (G - self.Q[oldest_key][oldest_action])
            
            self.buffer.pop(0)
    
    def _get_state_key(self, state):
        return tuple(state[0]), tuple(state[1])


class MultiGoalNStepSarsaAgent:
    def __init__(self, action_space, goals, n_steps=8, learning_rate=0.1, discount=0.99, epsilon=0.1, init_value=1.0):
        self.action_space = action_space
        self.goals = goals
        self.n = n_steps
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(lambda: init_value))
        # Buffer para cada meta: {goal: [(s, a, r), ...]}
        self.goal_buffers = {goal: [] for goal in goals}
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        return self.get_greedy_action(state)
    
    def get_greedy_action(self, state):
        state_key = self._get_state_key(state)
        return max(self.action_space, key=lambda a: self.Q[state_key][a])
    
    def start_episode(self):
        for goal in self.goals:
            self.goal_buffers[goal] = []
        
    def add_transition(self, state, action, next_state):
        position, _ = state
        next_position, _ = next_state
        
        # Añadir transición para cada meta con su recompensa específica
        for goal in self.goals:
            # Calcular recompensa para esta meta
            goal_reward = 1.0 if next_position == goal else 0.0
            
            # Añadir al buffer de esta meta: (estado actual con meta específica, acción, recompensa)
            state_with_goal = (position, goal)
            self.goal_buffers[goal].append((state_with_goal, action, goal_reward))
            
    def update(self):
        # Actualizar para cada meta si tenemos suficientes transiciones
        for goal in self.goals:
            buffer = self.goal_buffers[goal]
            
            # Solo actualizamos si tenemos suficientes transiciones
            if len(buffer) < self.n:
                continue
                
            # Calcular el retorno n-step para esta meta
            G = 0
            for i in range(min(self.n, len(buffer))):
                G += self.gamma**i * buffer[i][2]  # buffer[i][2] es la recompensa
                
            # Añadir estimación bootstrap si hay más estados
            if len(buffer) > self.n:
                bootstrap_state = buffer[self.n][0]
                bootstrap_action = buffer[self.n][1]
                bootstrap_key = self._get_state_key(bootstrap_state)
                G += self.gamma**self.n * self.Q[bootstrap_key][bootstrap_action]
                
            # Actualizar el estado más antiguo del buffer
            oldest_state = buffer[0][0]
            oldest_action = buffer[0][1]
            oldest_key = self._get_state_key(oldest_state)
            
            self.Q[oldest_key][oldest_action] += self.alpha * (G - self.Q[oldest_key][oldest_action])
            
            # Eliminar la transición más antigua
            self.goal_buffers[goal].pop(0)
        
    def end_episode(self):
        # Procesar todas las transiciones restantes para cada meta
        for goal in self.goals:
            buffer = self.goal_buffers[goal]
            
            while buffer:
                # Calcular retorno para el resto del episodio (sin bootstrap)
                G = 0
                for i, (_, _, r) in enumerate(buffer):
                    G += self.gamma**i * r
                    
                # Actualizar el estado más antiguo
                oldest_state = buffer[0][0]
                oldest_action = buffer[0][1]
                oldest_key = self._get_state_key(oldest_state)
                
                self.Q[oldest_key][oldest_action] += self.alpha * (G - self.Q[oldest_key][oldest_action])
                
                buffer.pop(0)
    
    def _get_state_key(self, state):
        return tuple(state[0]), tuple(state[1])


def run_experiment(n_episodes=500, n_runs=100, alpha=0.1, epsilon=0.1, gamma=0.99, init_value=1.0):
    env = RoomEnv()
    action_space = env.action_space
    goals = env.goals
    
    # Resultados para cada algoritmo
    results = {
        'Q-Learning': np.zeros((n_runs, n_episodes)),
        'SARSA': np.zeros((n_runs, n_episodes)),
        '8-step SARSA': np.zeros((n_runs, n_episodes)),
        'MultiGoal Q-Learning': np.zeros((n_runs, n_episodes)),
        'MultiGoal SARSA': np.zeros((n_runs, n_episodes)),
        'MultiGoal 8-step SARSA': np.zeros((n_runs, n_episodes)),
    }
    
    # Ejecutar experimentos para Q-Learning estándar
    for run in tqdm(range(n_runs), desc="Q-Learning"):
        agent = StandardQLearningAgent(action_space, alpha, gamma, epsilon, init_value)
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
                
            results['Q-Learning'][run, episode] = steps
    
    # Ejecutar experimentos para SARSA estándar
    for run in tqdm(range(n_runs), desc="SARSA"):
        agent = StandardSarsaAgent(action_space, alpha, gamma, epsilon, init_value)
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
                
            results['SARSA'][run, episode] = steps
    
    # Ejecutar experimentos para 8-step SARSA estándar
    for run in tqdm(range(n_runs), desc="8-step SARSA"):
        agent = StandardNStepSarsaAgent(action_space, 8, alpha, gamma, epsilon, init_value)
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            steps = 0
            agent.start_episode()
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                # Añadir transición al buffer
                agent.add_transition(state, action, reward)
                
                # Actualizar valores Q para la transición más antigua (si hay suficientes)
                agent.update(state, action)
                
                state = next_state
                steps += 1
            
            # Procesar transiciones restantes al final del episodio
            agent.end_episode()
                
            results['8-step SARSA'][run, episode] = steps
    
    # Ejecutar experimentos para Q-Learning multi-goal
    for run in tqdm(range(n_runs), desc="MultiGoal Q-Learning"):
        agent = MultiGoalQLearningAgent(action_space, goals, alpha, gamma, epsilon, init_value)
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
                
            results['MultiGoal Q-Learning'][run, episode] = steps
    
    # Ejecutar experimentos para SARSA multi-goal
    for run in tqdm(range(n_runs), desc="MultiGoal SARSA"):
        agent = MultiGoalSarsaAgent(action_space, goals, alpha, gamma, epsilon, init_value)
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
                
            results['MultiGoal SARSA'][run, episode] = steps
    
    # Ejecutar experimentos para 8-step SARSA multi-goal
    for run in tqdm(range(n_runs), desc="MultiGoal 8-step SARSA"):
        agent = MultiGoalNStepSarsaAgent(action_space, goals, 8, alpha, gamma, epsilon, init_value)
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            steps = 0
            agent.start_episode()
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                # Añadir transición al buffer para todas las metas
                agent.add_transition(state, action, next_state)
                
                # Actualizar valores Q para la transición más antigua
                agent.update()
                
                state = next_state
                steps += 1
            
            # Procesar transiciones restantes al final del episodio
            agent.end_episode()
                
            results['MultiGoal 8-step SARSA'][run, episode] = steps
    
    return results


def plot_results(results):
    plt.figure(figsize=(12, 8))
    
    # Calcular promedios sobre todas las ejecuciones
    for algorithm, data in results.items():
        mean_steps = np.mean(data, axis=0)
        # Suavizar datos para visualización
        window_size = 10
        smoothed = np.convolve(mean_steps, np.ones(window_size)/window_size, mode='valid')
        x = np.arange(len(smoothed))
        
        plt.plot(x, smoothed, label=algorithm)
    
    plt.xlabel('Episodio')
    plt.ylabel('Longitud promedio del episodio')
    plt.title('Comparación de algoritmos en RoomEnv')
    plt.legend()
    plt.grid(True)
    plt.savefig('Code_T3/multigoal.png')
    plt.show()


if __name__ == "__main__":
    results = run_experiment()
    plot_results(results)