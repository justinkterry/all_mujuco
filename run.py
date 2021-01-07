from all import nn
import torch
from all.presets.continuous import ppo
from all.experiments import run_experiment, plot_returns_100
from all.environments import GymEnvironment


def modified_fc_actor_critic(env, hidden1=64, hidden2=64):
    features = torch.nn.identity()

    v = nn.Sequential(
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1)
    )

    policy = nn.Sequential(
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2)
    )

    return features, v, policy


ppo = ppo(
    device="cuda",
    discount_factor=0.99,
    last_frame=1e6,
    # Adam optimizer settings
    lr=3e-4,  # Adam learning rate
    eps=1e-5,  # Adam stability
    # Loss scaling
    entropy_loss_scaling=0.0,
    value_loss_scaling=0.5,
    # Training settings
    clip_grad=0.5,
    clip_initial=0.2,
    clip_final=0.2,
    epochs=15,
    minibatches=1,  # not entirely sure if this is correct
    # Batch settings
    n_envs=16,  # probably 32?
    n_steps=2048,
    # GAE settings
    lam=0.95,
    # Model construction
    ac_model_constructor=modified_fc_actor_critic)


run_experiment([ppo], [GymEnvironment('InvertedPendulum-v2', device='cuda')], frames=1e6)
plot_returns_100('runs', timesteps=1e6)
