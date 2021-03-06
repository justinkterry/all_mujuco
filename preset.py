import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import PPO, PPOTestAgent
from all.approximation import VNetwork, Identity
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import GaussianPolicy
from all.presets.builder import preset_builder
from all.presets.preset import Preset


hyperparameters = {
    'discount_factor': 0.99,
    'last_frame': 1e6,
    # Adam optimizer settings
    'lr': 3e-4,  # Adam learning rate
    'eps': 1e-5,  # Adam stability
    # Loss scaling
    'entropy_loss_scaling': 0.0,
    'value_loss_scaling': 0.5,
    # Training settings
    'clip_grad': .5,
    'clip_initial': 0.2,
    'clip_final': 0.2,
    'epochs': 15,
    "minibatches": 4,
    # Batch settings
    "n_envs": 1,
    'n_steps': 2048,
    # GAE settings
    'lam': 0.95,
    # Model construction
    'feature_network': 0,
    'v_network': 0,
    'policy_network': 0}


class PPOContinuousPreset(Preset):
    """
    Proximal Policy Optimization (PPO) Continuous Control Preset.

    Args:
        env (all.environments.GymEnvironment): The classic control environment for which to construct the agent.
        device (torch.device, optional): the device on which to load the agent

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        entropy_loss_scaling (float): Coefficient for the entropy term in the total loss.
        value_loss_scaling (float): Coefficient for the value function loss.
        clip_grad (float): The maximum magnitude of the gradient for any given parameter. Set to 0 to disable.
        clip_initial (float): Value for epsilon in the clipped PPO objective function at the beginning of training.
        clip_final (float): Value for epsilon in the clipped PPO objective function at the end of training.
        epochs (int): Number of times to iterature through each batch.
        minibatches (int): The number of minibatches to split each batch into.
        n_envs (int): Number of parallel actors.
        n_steps (int): Length of each rollout.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
        ac_model_constructor (function): The function used to construct the neural feature, value and policy model.
    """

    def __init__(self, env, device="cuda", **hyperparameters):
        hyperparameters = {**hyperparameters}
        super().__init__(hyperparameters["n_envs"])
        feature_model = hyperparameters["feature_network"]
        value_model = hyperparameters["v_network"]
        policy_model = hyperparameters["policy_network"]
        self.feature_model = feature_model.to(device)
        self.value_model = value_model.to(device)
        self.policy_model = policy_model.to(device)
        self.device = device
        self.action_space = env.action_space
        self.hyperparameters = hyperparameters

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = train_steps * self.hyperparameters['epochs'] * self.hyperparameters['minibatches'] / (self.hyperparameters['n_steps'] * self.hyperparameters['n_envs'])

        feature_optimizer = None
        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters['lr'], eps=self.hyperparameters['eps'])
        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters['lr'], eps=self.hyperparameters['eps'])

        features = Identity(
            device=self.device
        )

        v = VNetwork(
            self.value_model,
            value_optimizer,
            loss_scaling=self.hyperparameters['value_loss_scaling'],
            clip_grad=self.hyperparameters['clip_grad'],
            writer=writer,
            scheduler=CosineAnnealingLR(
                value_optimizer,
                n_updates
            ),
        )

        policy = GaussianPolicy(
            self.policy_model,
            policy_optimizer,
            self.action_space,
            clip_grad=self.hyperparameters['clip_grad'],
            writer=writer,
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                n_updates
            ),
        )

        return PPO(
            features,
            v,
            policy,
            epsilon=LinearScheduler(
                self.hyperparameters['clip_initial'],
                self.hyperparameters['clip_final'],
                0,
                n_updates,
                name='clip',
                writer=writer
            ),
            epochs=self.hyperparameters['epochs'],
            minibatches=self.hyperparameters['minibatches'],
            n_envs=self.hyperparameters['n_envs'],
            n_steps=self.hyperparameters['n_steps'],
            discount_factor=self.hyperparameters['discount_factor'],
            lam=self.hyperparameters['lam'],
            entropy_loss_scaling=self.hyperparameters['entropy_loss_scaling'],
            writer=writer,
        )

    def test_agent(self):
        feature = Identity(copy.deepcopy(self.feature_model))
        policy = GaussianPolicy(copy.deepcopy(self.policy_model), space=self.action_space)
        return PPOTestAgent(feature, policy)


ppo = preset_builder('ppo', hyperparameters, PPOContinuousPreset)
