import numpy as np

class KeyDoorEnv:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = None
        self.key_pos = None
        self.door_pos = None
        self.has_key = False
        self.goal = None

    def reset(self):
        self.agent_pos = np.random.randint(0, self.size, size=2)
        self.key_pos = np.random.randint(0, self.size, size=2)
        self.door_pos = np.random.randint(0, self.size, size=2)
        self.has_key = False

        while np.array_equal(self.agent_pos, self.key_pos) or \
              np.array_equal(self.agent_pos, self.door_pos) or \
              np.array_equal(self.key_pos, self.door_pos):
            self.key_pos = np.random.randint(0, self.size, size=2)
            self.door_pos = np.random.randint(0, self.size, size=2)

        self.goal = np.concatenate([self.door_pos, [1]]) # goal is to be at the door with the key

        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate([self.agent_pos, self.key_pos, [self.has_key]])
        return {
            "observation": obs,
            "achieved_goal": np.concatenate([self.agent_pos, [self.has_key]]),
            "desired_goal": self.goal,
        }

    def compute_reward(self, achieved_goal, desired_goal):
        # Reward is -1 unless the goal is achieved
        if np.array_equal(achieved_goal, desired_goal):
            return 0.0
        return -1.0

    def step(self, action):
        # 0: up, 1: down, 2: left, 3: right
        if action == 0: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3: self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        if not self.has_key and np.array_equal(self.agent_pos, self.key_pos):
            self.has_key = True

        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"])
        done = (reward == 0.0)

        return obs, reward, done, {}
