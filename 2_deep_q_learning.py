import gymnasium as gym
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Création de l'environnement FrozenLake (non glissant pour simplifier la démonstration)
game = "FrozenLake-v1"
env = gym.make(game, is_slippery=False, render_mode="human")
num_episodes = 1000  # Nombre d'épisodes réduit pour la démo
show_every = 100

# Récupération du nombre d'états et d'actions
n_states = env.observation_space.n
n_actions = env.action_space.n

max_steps = 30  # Nombre maximal d'étapes par épisode réduit

# TODO: Enlever l'epsilon de l'ancien
# Paramètres d'exploration (epsilon-greedy)
epsilon = 0.7  # Début avec une forte exploration
min_epsilon = 0.09
# On utilisera une décroissance multiplicative pour une diminution progressive plus rapide
epsilon_decay = 0.999


# ==== Deep Q Network implementation
class DQN(nn.Module):

    def __init__(self, observations, actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# ==== Deep Q settings and networks
device = "cpu"

# BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# Action selection with network
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state_encodings = torch.zeros(n_states)
            state_encodings[state] = 1

            # return policy_net(state_encodings).max(1).indices.view(1, 1)[0].item()
            return policy_net(state_encodings).max(0).indices.item()
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)[0].item()


# ==== Model training/optimization function
def optimize_model(state, action, next_state, reward):
    state_encodings = torch.zeros(n_states)
    state_encodings[state] = 1

    next_state_encodings = torch.zeros(n_states)
    next_state_encodings[next_state] = 1

    action_batch = torch.tensor([action])
    reward_batch = torch.tensor([reward])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # state_action_values = policy_net(state_batch).gather(1, action_batch)
    state_action_values = policy_net(state_encodings).gather(0, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(1, device=device)
    with torch.no_grad():
        next_state_values[0] = target_net(next_state_encodings).max(0).values
        
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# Training loop
for episode in range(num_episodes):
    show_episode = episode % show_every == 0
    render_mode = "human" if (episode % show_every) == 0 else None

    if show_episode:
        print(f"==== Episode {episode}, epsilon={math.exp(-1. * steps_done / EPS_DECAY)}")

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

        # ==== Mise à jour des deep networks
        optimize_model(state, action, new_state, reward)

        state = new_state

        if done:
            break

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()

    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        
    target_net.load_state_dict(target_net_state_dict)

# Test de la politique (Strategie) apprise
state, _ = env.reset()
env.render()
done = False
while not done:
    action = select_action(state)
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
