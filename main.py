import numpy as np
import torch
from agent import Manager, LowLevelController, LatentGoalModule, HERRetrievalBuffer
from env import KeyDoorEnv

def main():
    env = KeyDoorEnv()
    obs = env.reset()
    obs_dim = obs["observation"].shape[0]
    goal_dim = obs["desired_goal"].shape[0]
    action_dim = 4
    latent_dim = 8
    manager_batch_size = 64
    llc_batch_size = 64
    c = 5  # Manager action frequency

    manager = Manager(obs_dim, latent_dim)
    llc = LowLevelController(obs_dim, action_dim, latent_dim)
    latent_module = LatentGoalModule(obs_dim, latent_dim)
    latent_module_optimizer = torch.optim.Adam(latent_module.parameters(), lr=1e-4)
    replay_buffer = HERRetrievalBuffer(1000, obs_dim, latent_module)

    episodes = 200
    max_steps = 50

    for e in range(episodes):
        obs = env.reset()
        done = False
        step = 0
        episode_trajectory = []

        while not done and step < max_steps:
            start_obs_for_manager = obs["observation"]
            latent_goal = manager.select_goal(start_obs_for_manager)

            macro_step_trajectory = []
            for _ in range(c):
                action = llc.select_action(obs["observation"], latent_goal)
                next_obs, reward, done, _ = env.step(action)

                macro_step_trajectory.append((obs, action, reward, next_obs, done, latent_goal))
                obs = next_obs
                step += 1
                if done or step >= max_steps:
                    break

            replay_buffer.add_macro_step(macro_step_trajectory)
            episode_trajectory.extend(macro_step_trajectory)

        # Train Low-Level Controller
        llc.train(replay_buffer)
        llc.update_target_networks()

        # Train Manager and Latent Module
        s_starts, s_ends, orig_goals, rewards = replay_buffer.sample_manager_batch(manager_batch_size)
        if len(s_starts) > 0:
            manager.train(s_starts, s_ends, rewards, latent_module)
            manager.update_target_networks()

            # Train the latent goal module
            latent_module.train_step(torch.FloatTensor(s_starts), torch.FloatTensor(s_ends), torch.FloatTensor(orig_goals), latent_module_optimizer)


        if (e + 1) % 10 == 0:
            # Simple success metric: was the final goal achieved?
            if episode_trajectory:
                final_achieved = episode_trajectory[-1][3]["achieved_goal"]
                final_desired = episode_trajectory[-1][3]["desired_goal"]
                success = np.array_equal(final_achieved, final_desired)
                print(f"Episode {e+1}/{episodes}, Success: {success}")
            else:
                print(f"Episode {e+1}/{episodes}, Success: False (No steps taken)")

if __name__ == "__main__":
    main()
