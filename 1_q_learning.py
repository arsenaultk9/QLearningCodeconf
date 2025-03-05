import gymnasium as gym
import numpy as np

# Création de l'environnement FrozenLake (non glissant pour simplifier la démonstration)
game = "FrozenLake-v1"
env = gym.make(game, is_slippery=False, render_mode="human")
num_episodes = 1000  # Nombre d'épisodes réduit pour la démo
show_every = 100

# Récupération du nombre d'états et d'actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialisation de la Q-table
Q_table = np.zeros((n_states, n_actions))

# Paramètres d'apprentissage
learning_rate = 5  # Taux d'apprentissage plus élevé pour des mises à jour plus rapides
discount_factor = (
    0.5  # Facteur d'actualisation abaissé pour privilégier les récompenses immédiates
)
max_steps = 30  # Nombre maximal d'étapes par épisode réduit

# Paramètres d'exploration (epsilon-greedy)
epsilon = 0.7  # Début avec une forte exploration
min_epsilon = 0.0000009
# On utilisera une décroissance multiplicative pour une diminution progressive plus rapide
epsilon_decay = 0.99

def select_action(state):
    # Sélection de l'action avec stratégie epsilon-greedy
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[state, :])
        
    return action

def optimize_q_function(state, action, next_state, reward):
    # Mise à jour de la Q-table selon la formule du Q-learning
    # Belman equation: 𝑄(𝑠,𝑎)←𝑄(𝑠,𝑎)+𝛼( 𝑟+ 𝛾max𝑎′𝑄(𝑠′,𝑎′)−𝑄(𝑠,𝑎))
    
    Q_table[state, action] = Q_table[state, action] + learning_rate * (
        reward
        + discount_factor * np.max(Q_table[new_state, :])
        - Q_table[state, action]
    )

for episode in range(num_episodes):
    show_episode = episode % show_every == 0
    render_mode = "human" if (episode % show_every) == 0 else None

    if show_episode:
        print(f"==== Episode {episode}, epsilon={epsilon}")
        # print("Q Table :")
        # print(Q_table)

    env = gym.make(game, render_mode=render_mode)

    state, _ = env.reset()
    done = False

    for step in range(max_steps):
        action = select_action(state)
        
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Reward shaping : si l'épisode se termine et que la récompense est 0,
        # on considère que l'agent est tombé dans un trou et on pénalise fortement.
        if done and reward == 0:
            reward = -1

        # Indiqué lorsque gagné
        if done and reward == 1:
            print(f"Episode {episode} gagné en {step} étapes")

        optimize_q_function(state, action, new_state, reward)

        state = new_state

        if done:
            break

    # Décroissance d'epsilon pour réduire progressivement l'exploration
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Table Q finale :")
print(Q_table)

# Test de la politique (Strategie) apprise
state, _ = env.reset()
env.render()
done = False
while not done:
    action = np.argmax(Q_table[state, :])
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
