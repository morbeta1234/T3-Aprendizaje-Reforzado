import numpy as np
import matplotlib.pyplot as plt
from centralized_hunter_qlearning import run_centralized_qlearning
from decentralized_hunter_qlearning import run_decentralized_qlearning
from decentralized_competitive_hunter_qlearning import run_decentralized_competitive_qlearning
from Environments.MultiAgentEnvs.HunterEnv import HunterEnv
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv
from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv

def run_experiments(n_episodes=50000, n_runs=30):
    # Ejecutar experimentos
    print("Ejecutando Q-learning centralizado...")
    env1 = CentralizedHunterEnv()
    centralized_lengths = run_centralized_qlearning(env1, n_episodes=n_episodes, n_runs=n_runs)
    
    print("\nEjecutando Q-learning descentralizado...")
    env2 = HunterEnv()
    decentralized_lengths = run_decentralized_qlearning(env2, n_episodes=n_episodes, n_runs=n_runs)
    
    print("\nEjecutando Q-learning descentralizado competitivo...")
    env3 = HunterAndPreyEnv()
    competitive_lengths = run_decentralized_competitive_qlearning(env3, n_episodes=n_episodes, n_runs=n_runs)
    
    # Calcular medias y desviaciones estándar
    cent_mean = np.mean(centralized_lengths, axis=0)
    cent_std = np.std(centralized_lengths, axis=0)
    
    decent_mean = np.mean(decentralized_lengths, axis=0)
    decent_std = np.std(decentralized_lengths, axis=0)
    
    comp_mean = np.mean(competitive_lengths, axis=0)
    comp_std = np.std(competitive_lengths, axis=0)
    
    # Crear gráfico comparativo
    plt.figure(figsize=(12, 8))
    episodes = np.arange(len(cent_mean)) * 100
    
    plt.plot(episodes, cent_mean, label='Centralizado', color='blue')
    # plt.fill_between(episodes, cent_mean - cent_std, cent_mean + cent_std, 
    #                 alpha=0.2, color='blue')
    
    plt.plot(episodes, decent_mean, label='Descentralizado', color='green')
    # plt.fill_between(episodes, decent_mean - decent_std, decent_mean + decent_std,
    #                 alpha=0.2, color='green')
    
    plt.plot(episodes, comp_mean, label='Descentralizado Competitivo', color='red')
    # plt.fill_between(episodes, comp_mean - comp_std, comp_mean + comp_std,
    #                 alpha=0.2, color='red')
    
    plt.xlabel('Episodio')
    plt.ylabel('Duración promedio del episodio')
    plt.title('Comparación de Algoritmos Q-learning en el Dominio Hunter')
    plt.legend()
    plt.grid(True)
    
    # Guardar gráfico
    plt.savefig('comparison_curves.png')
    plt.close()
    
    # Imprimir análisis estadístico
    print("\nAnálisis Estadístico:")
    print(f"Centralizado - Duración promedio final: {cent_mean[-1]:.2f} ± {cent_std[-1]:.2f}")
    print(f"Descentralizado - Duración promedio final: {decent_mean[-1]:.2f} ± {decent_std[-1]:.2f}")
    print(f"Competitivo - Duración promedio final: {comp_mean[-1]:.2f} ± {comp_std[-1]:.2f}")

if __name__ == "__main__":
    run_experiments(n_episodes=50000, n_runs=30)
