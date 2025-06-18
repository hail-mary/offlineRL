import minari
# pip install minari==0.5.0
# pip install numpy==1.26.4
# to show available datasets: minari list remote
remote_dataset_id = 'D4RL/kitchen/complete-v2'
dataset = minari.load_dataset(remote_dataset_id)
# if you encounter an error on windows, see https://github.com/Farama-Foundation/Minari/commit/e4e9340449f14195c59a906cee788094d2ab93f4

# env = dataset.recover_environment()
# eval_env = dataset.recover_environment(eval_env=True)
# print("Observation space:", dataset.observation_space)
# print("Action space:", dataset.action_space)
# print("Total episodes:", dataset.total_episodes)
# print("Total steps:", dataset.total_steps)

# assert env.spec == eval_env.spec


import torch
from torch.utils.data import DataLoader, TensorDataset

def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations['observation']) for x in batch],
            batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True
        )
    }
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)


import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation, RecordVideo # pip install moviepy
from minari import DataCollector # pip install "minari[create]""
import gymnasium_robotics # v1.3.0
gym.register_envs(gymnasium_robotics)
env = gym.make('FrankaKitchen-v1', render_mode='rgb_array', width=128, height=128)
# env = TimeLimit(env, max_episode_steps=1000)
env = AddRenderObservation(env, render_only=True)
env = RecordVideo(env, video_folder='expert_demos', episode_trigger=lambda x: True)
# env = DataCollector(env)

for batch in dataloader:
    for episode in range(1):
        obs_traj = []
        obs, _ = env.reset(seed=0)
        done = False
        reward_sum = 0
        actions = batch['actions'][episode]
        for t in range(actions.size(0)):
            obs_traj.append(torch.as_tensor(obs.copy()))
            action = actions[t]
            obs, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated
            reward_sum += reward

        print("Cumulative Reward: ", reward_sum)
        obs_traj = torch.stack(obs_traj)
        torch.save(obs_traj, f'expert_demos/episode-{episode}_128x128.pt')

env.close()



# train_set = TensorDataset(obs_traj)
# train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

# dataset_id = "kitchen/expert-v0"
# # delete the test dataset if it already exists
# local_datasets = minari.list_local_datasets()
# if dataset_id in local_datasets:
#     minari.delete_dataset(dataset_id)

# Create Minari dataset and store locally
# dataset = env.create_dataset(
#     dataset_id=dataset_id,
#     algorithm_name="expert_policy",
#     code_permalink="https://github.com/Farama-Foundation/Minari",
#     author="hail-mary",
#     author_email="s246215w@st.go.tuat.ac.jp"
# )

# print(dataset[0].observations.keys()) 

