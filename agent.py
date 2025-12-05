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
    def __init__(self, capacity, obs_dim, goal_dim, latent_goal_module, her_ratio=0.8, rer_ratio=0.2):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.latent_goal_module = latent_goal_module
        self.her_ratio = her_ratio
        self.rer_ratio = rer_ratio
        self.episodes = deque(maxlen=capacity)

        max_buffer_size = capacity * 50  # Approximate max size
        self.flat_buffer = deque(maxlen=max_buffer_size)

        # Use a pre-allocated numpy array for state_history
        self.state_history = np.zeros((max_buffer_size, obs_dim + latent_goal_module.latent_dim))
        self.state_history_ptr = 0
        self.state_history_len = 0

        self.tree = None
        self.refresh_rate = 1000
        self.transitions_since_refresh = 0

    def add_macro_step(self, macro_step):
        self.episodes.append(macro_step) # Still useful for HER

        for obs, action, reward, next_obs, done, latent_goal in macro_step:
            state = np.concatenate([obs["observation"], latent_goal])
            self.flat_buffer.append((obs, action, reward, next_obs, done, latent_goal))

            self.state_history[self.state_history_ptr] = state
            self.state_history_ptr = (self.state_history_ptr + 1) % self.state_history.shape[0]
            if self.state_history_len < self.state_history.shape[0]:
                self.state_history_len += 1

        self.transitions_since_refresh += len(macro_step)
        if self.transitions_since_refresh >= self.refresh_rate and self.state_history_len > 1:
            self.tree = cKDTree(self.state_history[:self.state_history_len])
            self.transitions_since_refresh = 0

    def sample_ll_batch(self, batch_size, k_neighbors=2):
        batch = []
        if len(self.flat_buffer) < batch_size:
            return batch

        # RER Samples
        num_rer_samples = int(batch_size * self.rer_ratio)
        if self.tree and len(self.flat_buffer) > num_rer_samples:
            indices = np.random.choice(len(self.flat_buffer), num_rer_samples // (k_neighbors + 1), replace=False)
            primary_states = np.array([self.state_history[i] for i in indices])

            _, neighbor_idxs = self.tree.query(primary_states, k=k_neighbors)
            neighbor_idxs = neighbor_idxs.flatten()

            final_indices = np.unique(np.concatenate([indices, neighbor_idxs]))
            final_indices = final_indices[final_indices < len(self.flat_buffer)]
        else:
            final_indices = np.random.randint(0, len(self.flat_buffer), num_rer_samples)

        for i in final_indices:
            obs, action, _, next_obs, done, latent_goal = self.flat_buffer[i]
            state = np.concatenate([obs["observation"], latent_goal])
            next_state = np.concatenate([next_obs["observation"], latent_goal])
            achieved_latent = self.latent_goal_module.get_latent(torch.FloatTensor(next_obs["observation"]).unsqueeze(0)).squeeze(0).detach().numpy()
            reward = -np.linalg.norm(achieved_latent - latent_goal)
            batch.append((state, action, reward, next_state, done))

        # HER Samples
        num_her_samples = batch_size - len(batch)
        ep_indices = np.random.randint(0, len(self.episodes), num_her_samples)
        for ep_idx in ep_indices:
            episode = self.episodes[ep_idx]
            t = np.random.randint(0, len(episode))
            obs, action, _, next_obs, done, _ = episode[t]

            future_t = np.random.randint(t, len(episode))
            future_obs = episode[future_t][0]

            # Relabel goal to a future achieved state
            latent_goal = self.latent_goal_module.get_latent(torch.FloatTensor(future_obs["observation"]).unsqueeze(0)).squeeze(0).detach().numpy()

            state = np.concatenate([obs["observation"], latent_goal])
            next_state = np.concatenate([next_obs["observation"], latent_goal])
            achieved_latent = self.latent_goal_module.get_latent(torch.FloatTensor(next_obs["observation"]).unsqueeze(0)).squeeze(0).detach().numpy()
            reward = -np.linalg.norm(achieved_latent - latent_goal)
            batch.append((state, action, reward, next_state, done))

        return batch

    def sample_manager_batch(self, batch_size):
        batch = []
        if len(self.episodes) < batch_size:
            return [], [], [], []

        indices = np.random.randint(0, len(self.episodes), batch_size)
        for i in indices:
            macro_step = self.episodes[i]
            s_start = macro_step[0][0]["observation"]
            s_end = macro_step[-1][3]["observation"]
            latent_goal = macro_step[0][5]
            reward_sum = sum([t[2] for t in macro_step])
            batch.append((s_start, s_end, latent_goal, reward_sum))

        s_starts, s_ends, orig_goals, reward_sums = zip(*batch)
        return np.array(s_starts), np.array(s_ends), np.array(orig_goals), np.array(reward_sums)


    def __len__(self):
        return len(self.flat_buffer)

class LatentGoalModule(torch.nn.Module):
    def __init__(self, state_dim, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim

        # 1. State Encoder (s -> z)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim) # Outputs mean (we skip var for simplicity here)
        )

        # 2. Inverse Model (s_start, s_end) -> z_action/goal
        # Predicts: "What latent goal leads from start to end?"
        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(state_dim * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim)
        )

    def get_latent(self, state):
        return self.encoder(state)

    def predict_goal(self, s_start, s_end):
        inp = torch.cat([s_start, s_end], dim=1)
        return self.inverse_model(inp)

    def train_step(self, s_start, s_end, true_latent_goal, optimizer):
        pred_goal = self.predict_goal(s_start, s_end)
        loss = F.mse_loss(pred_goal, true_latent_goal)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

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

class ActorNetwork(torch.nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(ActorNetwork, self).__init__()
        self.l1 = torch.nn.Linear(state_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, latent_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x)) # tanh to bound the latent goal space

class CriticNetwork(torch.nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(CriticNetwork, self).__init__()
        self.l1 = torch.nn.Linear(state_dim + latent_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, 1)

    def forward(self, state, latent_goal):
        x = torch.cat([state, latent_goal], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

class LowLevelController:
    def __init__(self, obs_dim, action_dim, latent_dim, gamma=0.95, lr=1e-4, batch_size=64):
        self.gamma = gamma
        self.batch_size = batch_size

        self.q_nets = [QNetwork(obs_dim + latent_dim, action_dim) for _ in range(3)]
        self.target_q_nets = [QNetwork(obs_dim + latent_dim, action_dim) for _ in range(3)]
        for i in range(3):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())

        self.optimizer = torch.optim.Adam(
            [param for q_net in self.q_nets for param in q_net.parameters()], lr=lr)

    def select_action(self, obs, latent_goal):
        state = torch.FloatTensor(np.concatenate([obs, latent_goal])).unsqueeze(0)
        with torch.no_grad():
            q_values = torch.mean(torch.stack([q_net(state) for q_net in self.q_nets]), dim=0)
        return q_values.argmax().item()

    def train(self, buffer):
        batch = buffer.sample_ll_batch(self.batch_size)
        if not batch:
            return

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        with torch.no_grad():
            next_q_values = torch.stack([target_q_net(next_states) for target_q_net in self.target_q_nets])
            min_next_q, _ = torch.min(next_q_values, dim=0)
            target_q = rewards + self.gamma * min_next_q.max(1, keepdim=True)[0] * (1 - dones)

        loss = compute_conservative_loss(self.q_nets, states, actions, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Manager:
    def __init__(self, state_dim, latent_dim, gamma=0.99, lr=1e-4):
        self.gamma = gamma
        self.actor = ActorNetwork(state_dim, latent_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critics = [CriticNetwork(state_dim, latent_dim) for _ in range(3)]
        self.target_critics = [CriticNetwork(state_dim, latent_dim) for _ in range(3)]
        for i in range(3):
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        self.critic_optimizer = torch.optim.Adam(
            [param for critic in self.critics for param in critic.parameters()], lr=lr)

    def select_goal(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).squeeze(0).detach().numpy()

    def train(self, s_starts, s_ends, rewards, latent_goal_module):
        s_starts = torch.FloatTensor(s_starts)
        s_ends = torch.FloatTensor(s_ends)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)

        # Critic Update
        with torch.no_grad():
            achieved_goals = latent_goal_module.predict_goal(s_starts, s_ends)
            next_goals = self.actor(s_ends)

            next_values = torch.stack([target_critic(s_ends, next_goals) for target_critic in self.target_critics])
            min_next_q, _ = torch.min(next_values, dim=0)
            target_q = rewards + self.gamma * min_next_q

        critic_loss = 0
        for critic in self.critics:
            current_q = critic(s_starts, achieved_goals)
            critic_loss += F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        new_goals = self.actor(s_starts)
        actor_loss = -self.critics[0](s_starts, new_goals).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
