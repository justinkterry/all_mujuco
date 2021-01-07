from preset import ppo
from all.experiments import run_experiment, plot_returns_100
from all.environments import GymEnvironment

run_experiment([ppo()], [GymEnvironment('InvertedPendulum-v2', device='cuda')], frames=1e6)
