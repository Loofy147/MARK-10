from collections import deque
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree

def compute_conservative_loss(q_nets, states, actions, target_q, alpha=1.0):
    """
    q_nets: List of 3 Q-networks
    target_q: The calculated target (r + gamma * next_q)
    alpha: Scaling factor for conservatism
    """
    total_loss = 0
    for q_net in q_nets:
        # 1. Standard Bellman Error
        current_q = q_net(states).gather(1, actions)
        bellman_loss = F.mse_loss(current_q, target_q)

        # 2. Conservative Penalty (simplified CQL)
        # We penalize high Q-values on random actions to prevent explosion
        random_actions = torch.randint_like(actions, low=0, high=q_net(states).shape[1])
        q_random = q_net(states).gather(1, random_actions)

        # If Q(random) > Q(actual), penalize heavily
        conservative_loss = torch.mean(F.relu(q_random - current_q))

        total_loss += bellman_loss + (alpha * conservative_loss)

    return total_loss

class HERRetrievalBuffer:
    def __init__(self, capacity, obs_dim, goal_dim, her_ratio=0.8, rer_ratio=0.2):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.her_ratio = her_ratio
        self.rer_ratio = rer_ratio
        self.episodes = deque(maxlen=capacity)

        max_buffer_size = capacity * 50  # Approximate max size
        self.flat_buffer = deque(maxlen=max_buffer_size)
        self.state_history = deque(maxlen=max_buffer_size)

        self.tree = None
        self.refresh_rate = 1000
        self.transitions_since_refresh = 0

    def add_episode(self, episode):
        self.episodes.append(episode)

        for obs, action, reward, next_obs, done in episode:
            state = np.concatenate([obs["observation"], obs["desired_goal"]])
            self.flat_buffer.append((obs, action, reward, next_obs, done))
            self.state_history.append(state)

        self.transitions_since_refresh += len(episode)
        if self.transitions_since_refresh >= self.refresh_rate and len(self.state_history) > 1:
            self.tree = cKDTree(list(self.state_history))
            self.transitions_since_refresh = 0


    def sample(self, batch_size, reward_func, k_neighbors=2):
        batch = []

        num_rer_samples = int(batch_size * self.rer_ratio)
        num_her_samples = batch_size - num_rer_samples

        # HER Samples
        if len(self.episodes) > 0:
            ep_indices = np.random.randint(0, len(self.episodes), num_her_samples)
            for ep_idx in ep_indices:
                episode = self.episodes[ep_idx]
                t = np.random.randint(0, len(episode))
                obs, action, _, next_obs, done = episode[t]

                if np.random.rand() < self.her_ratio:
                    future_t = np.random.randint(t, len(episode))
                    desired_goal = episode[future_t][0]["achieved_goal"]
                else:
                    desired_goal = obs["desired_goal"]

                reward = reward_func(next_obs["achieved_goal"], desired_goal)
                state = np.concatenate([obs["observation"], desired_goal])
                next_state = np.concatenate([next_obs["observation"], desired_goal])
                batch.append((state, action, reward, next_state, done))

        # RER Samples
        num_primary_samples = num_rer_samples // (k_neighbors + 1)
        if self.tree and len(self.flat_buffer) > num_primary_samples:
            indices = np.random.choice(len(self.flat_buffer), num_primary_samples, replace=False)
            primary_states = np.array(self.state_history)[indices]

            _, neighbor_idxs = self.tree.query(primary_states, k=k_neighbors)
            neighbor_idxs = neighbor_idxs.flatten()

            final_indices = np.unique(np.concatenate([indices, neighbor_idxs]))
            final_indices = final_indices[final_indices < len(self.flat_buffer)]

            for i in final_indices:
                obs, action, _, next_obs, done = self.flat_buffer[i]
                reward = reward_func(next_obs["achieved_goal"], obs["desired_goal"])
                state = np.concatenate([obs["observation"], obs["desired_goal"]])
                next_state = np.concatenate([next_obs["observation"], obs["desired_goal"]])
                batch.append((state, action, reward, next_state, done))

        # Fill up to batch_size if needed
        while len(batch) < batch_size and len(self.flat_buffer) > 0:
             idx = np.random.randint(0, len(self.flat_buffer))
             obs, action, _, next_obs, done = self.flat_buffer[idx]
             reward = reward_func(next_obs["achieved_goal"], obs["desired_goal"])
             state = np.concatenate([obs["observation"], obs["desired_goal"]])
             next_state = np.concatenate([next_obs["observation"], obs["desired_goal"]])
             batch.append((state, action, reward, next_state, done))

        return batch[:batch_size]


    def __len__(self):
        return len(self.flat_buffer)

class ICM(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ICM, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU()
        )
        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim)
        )
        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(256 + action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256)
        )

    def forward(self, state, next_state, action):
        state_feat = self.encoder(state)
        next_state_feat = self.encoder(next_state)

        # Inverse model
        predicted_action = self.inverse_model(torch.cat([state_feat, next_state_feat], dim=1))

        # Forward model
        action_one_hot = F.one_hot(action.squeeze().long(), num_classes=self.inverse_model[-1].out_features)
        predicted_next_state_feat = self.forward_model(torch.cat([state_feat, action_one_hot], dim=1))

        return predicted_action, predicted_next_state_feat, next_state_feat

class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.l1 = torch.nn.Linear(state_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.l3(x)

class Agent:
    def __init__(self, state_dim, action_dim, obs_dim, goal_dim, gamma=0.99, lr=1e-4, batch_size=64, buffer_capacity=1000, eta=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.eta = eta

        self.q_nets = [QNetwork(state_dim, action_dim) for _ in range(3)]
        self.target_q_nets = [QNetwork(state_dim, action_dim) for _ in range(3)]
        for i in range(3):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())
            self.target_q_nets[i].eval()

        self.icm = ICM(state_dim, action_dim)
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=lr)

        self.optimizer = torch.optim.Adam(
            [param for q_net in self.q_nets for param in q_net.parameters()], lr=lr
        )

        self.replay_buffer = HERRetrievalBuffer(buffer_capacity, obs_dim, goal_dim)

        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = torch.mean(torch.stack([q_net(state) for q_net in self.q_nets]), dim=0)
            return q_values.argmax().item()

    def train(self, reward_func):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size, reward_func)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # ICM Forward Pass
        pred_action, pred_next_state_feat, next_state_feat = self.icm(states, next_states, actions)

        # Intrinsic Reward
        intrinsic_reward = self.eta * F.mse_loss(pred_next_state_feat, next_state_feat, reduction='none').mean(dim=1).unsqueeze(1)
        total_reward = rewards + intrinsic_reward.detach() # Detach to not backprop through Q-nets

        # ICM Loss
        inverse_loss = F.cross_entropy(pred_action, actions.squeeze())
        forward_loss = F.mse_loss(pred_next_state_feat, next_state_feat)
        icm_loss = inverse_loss + forward_loss

        # Update ICM
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

        # Q-Network training
        with torch.no_grad():
            next_q_values = torch.stack([target_q_net(next_states) for target_q_net in self.target_q_nets])
            min_next_q, _ = torch.min(next_q_values, dim=0)
            target_q = total_reward + self.gamma * min_next_q.max(1, keepdim=True)[0] * (1 - dones)

        q_loss = compute_conservative_loss(self.q_nets, states, actions, target_q)

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_networks(self, tau=0.005):
        for i in range(3):
            for target_param, param in zip(self.target_q_nets[i].parameters(), self.q_nets[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
