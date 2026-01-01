import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn, multiprocessing
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

# Local noisy layers (optional)
from .noisy import NoisyLazyLinear


def get_device():
    """Determine the appropriate device for training"""
    is_fork = multiprocessing.get_start_method() == "fork"
    if torch.cuda.is_available() and not is_fork:
        return torch.device(0)
    return torch.device("cpu")


class SACPolicy:
    """Soft Actor-Critic policy + value + twin critics built from nn modules.

    Attributes:
        actor: ProbabilisticActor
        value: ValueOperator (state value)
        q1: ValueOperator (Q1)
        q2: ValueOperator (Q2)
    """

    def __init__(
        self,
        env: GymEnv,
        num_cells: int = 256,
        device=None,
        use_noisy: bool = False,
        noise_sigma: float = 0.5,
    ):
        if device is None:
            device = get_device()
        self.device = device
        self.use_noisy = use_noisy
        self.noise_sigma = noise_sigma

        action_dim = int(env.action_spec.shape[-1])

        # helper to pick noisy vs lazy linear layers
        def _lin(out):
            if use_noisy:
                return NoisyLazyLinear(out, sigma=noise_sigma, device=device)
            return nn.LazyLinear(out, device=device)

        # CNN feature extractor for image inputs
        cnn_features = nn.Sequential(
            # Input: [B, C, H, W] - typically [B, 3, 96, 96] for CarRacing
            nn.Conv2d(
                3, 32, kernel_size=8, stride=4, padding=0, device=device
            ),  # → [B, 32, 23, 23]
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=4, stride=2, padding=0, device=device
            ),  # → [B, 64, 10, 10]
            nn.ReLU(),
            nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=0, device=device
            ),  # → [B, 64, 8, 8]
            nn.ReLU(),
            nn.Flatten(start_dim=1),  # → [B, 64*8*8] = [B, 4096]
        )

        actor_net = nn.Sequential(
            cnn_features,
            _lin(num_cells),
            nn.Tanh(),
            _lin(num_cells),
            nn.Tanh(),
            nn.LazyLinear(2 * action_dim, device=device),
            NormalParamExtractor(),
        )

        policy_module = TensorDictModule(
            actor_net, in_keys=["pixels"], out_keys=["loc", "scale"]
        )

        # NOTE: To get the hardcoded action bounds, we would need to create the env here:
        # import gymnasium as gym

        # g = gym.make("CarRacing-v3")
        # print("gym action_bounds:", g.action_space.low, g.action_space.high)

        # TODO: remove hardcoded bounds and use env specs directly
        low = [-1.0, 0.0, 0.0]  # env.action_spec_unbatched.space.low
        high = [1.0, 1.0, 1.0]  # env.action_spec_unbatched.space.high

        try:
            low_t = torch.as_tensor(low, dtype=torch.float32)
            high_t = torch.as_tensor(high, dtype=torch.float32)
            if not torch.all(high_t > low_t):
                print(
                    f"Warning: invalid action bounds detected (low={low_t}, high={high_t}). "
                    "Defaulting to [-1, 1] for each action dimension to satisfy TanhNormal requirements."
                )
                low_t = -torch.ones_like(low_t)
                high_t = torch.ones_like(high_t)
            dist_kwargs = {
                "low": low_t,
                "high": high_t,
            }
        except Exception as _e:
            print(
                f"Warning: could not validate action bounds ({_e}); using raw spec values."
            )
            dist_kwargs = {
                "low": low,
                "high": high,
            }

        self.actor = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs=dist_kwargs,
            return_log_prob=True,
        )

        value_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0, device=device),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=device),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        value_net = nn.Sequential(
            value_cnn,
            _lin(num_cells),
            nn.Tanh(),
            _lin(num_cells),
            nn.Tanh(),
            nn.LazyLinear(1, device=device),
        )

        self.value = ValueOperator(module=value_net, in_keys=["pixels"])

        # CNN feature extractor for target value network
        value_cnn_target = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0, device=device),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=device),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        value_net_target = nn.Sequential(
            value_cnn_target,
            _lin(num_cells),
            nn.Tanh(),
            _lin(num_cells),
            nn.Tanh(),
            nn.LazyLinear(1, device=device),
        )
        self.value_target = ValueOperator(module=value_net_target, in_keys=["pixels"])

        class CriticNet(nn.Module):
            def __init__(self, hidden: int, device):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0, device=device),
                    nn.ReLU(),
                    nn.Conv2d(
                        32, 64, kernel_size=4, stride=2, padding=0, device=device
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        64, 64, kernel_size=3, stride=1, padding=0, device=device
                    ),
                    nn.ReLU(),
                    nn.Flatten(start_dim=1),
                )

                self.fc = nn.Sequential(
                    _lin(hidden),
                    nn.Tanh(),
                    _lin(hidden),
                    nn.Tanh(),
                    nn.LazyLinear(1, device=device),
                )

            def forward(self, pixels, action):
                img_features = self.cnn(pixels)
                act = action.flatten(start_dim=1)
                x = torch.cat([img_features, act], dim=-1)
                return self.fc(x)

        q1_net = CriticNet(num_cells, device)
        q2_net = CriticNet(num_cells, device)

        q1_net_target = CriticNet(num_cells, device)
        q2_net_target = CriticNet(num_cells, device)

        self.q1 = ValueOperator(module=q1_net, in_keys=["pixels", "action"])
        self.q2 = ValueOperator(module=q2_net, in_keys=["pixels", "action"])

        self.q1_target = ValueOperator(
            module=q1_net_target, in_keys=["pixels", "action"]
        )
        self.q2_target = ValueOperator(
            module=q2_net_target, in_keys=["pixels", "action"]
        )

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def modules(self):
        return {
            "actor": self.actor,
            "value": self.value,
            "value_target": self.value_target,
            "q1": self.q1,
            "q2": self.q2,
            "q1_target": self.q1_target,
            "q2_target": self.q2_target,
        }
