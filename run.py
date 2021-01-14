import torch.nn as nn
from preset import ppo
from all.experiments import run_experiment, plot_returns_100
from all.environments import GymEnvironment
from supersuit import dtype_v0
import gym
env = gym.make('InvertedPendulum-v2')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden1 = 64
hidden2 = 64

features = nn.Sequential(nn.Identity())

v = nn.Sequential(
    nn.Linear(state_dim, hidden1),
    nn.ReLU(),
    nn.Linear(hidden1, hidden2),
    nn.ReLU(),
    nn.Linear(hidden2, 1)
)

policy = nn.Sequential(
    nn.Linear(state_dim, hidden1),
    nn.ReLU(),
    nn.Linear(hidden1, hidden2),
    nn.ReLU(),
    nn.Linear(hidden2, action_dim*2)
    )


hyperparameters = {
    # Common settings
    "discount_factor": 0.98,
    # Adam optimizer settings
    "lr": 3e-4,  # Adam learning rate
    "eps": 1e-5,  # Adam stability
    # Loss scaling
    "entropy_loss_scaling": 0.01,
    "value_loss_scaling": 0.5,
    # Training settings
    "clip_grad": 0.5,
    "clip_initial": 0.2,
    "clip_final": 0.01,
    "epochs": 20,
    "minibatches": 4,
    # Batch settings
    "n_envs": 32,
    "n_steps": 128,
    # GAE settings
    "lam": 0.95,
    # Model construction
    'feature_network': features,
    'v_network': v,
    'policy_network': policy}


run_experiment([ppo(hyperparameters=hyperparameters)], [GymEnvironment(dtype_v0(env, 'float32'), device='cuda')], frames=1e6)
#plot_returns_100('runs', timesteps=1e6)
