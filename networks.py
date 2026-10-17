import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = self.ln1(x)
        out = torch.relu(self.fc1(out))
        out = self.ln2(out)
        out = self.fc2(out)
        return torch.relu(out + identity)


class ResidualMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512, num_blocks=2,
                 final_dims=[256, 128]):
        super().__init__()
        self.input_fc = nn.Linear(obs_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)

        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])

        layers = []
        prev_dim = hidden_dim
        for fd in final_dims:
            layers.append(nn.LayerNorm(prev_dim))
            layers.append(nn.Linear(prev_dim, fd))
            layers.append(nn.ReLU())
            prev_dim = fd
        layers.append(nn.Linear(prev_dim, act_dim))
        self.output_net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.input_ln(x)
        x = torch.relu(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_net(x)


class ResMlpExtractor(nn.Module):
    def __init__(self, obs_dim, act_dim, shared_dim=512, num_blocks=3,
                 final_dims=[256, 128]):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
        )
        self.policy_net = ResidualMLP(
            shared_dim, act_dim,
            hidden_dim=shared_dim, num_blocks=num_blocks,
            final_dims=final_dims,
        )
        self.value_net = ResidualMLP(
            shared_dim, 1,
            hidden_dim=shared_dim, num_blocks=num_blocks,
            final_dims=final_dims,
        )
        self.latent_dim_pi = act_dim
        self.latent_dim_vf = 1

    def forward(self, features):
        shared = self.shared_net(features)
        return self.policy_net(shared), self.value_net(shared)

    def forward_critic(self, features):
        shared = self.shared_net(features)
        return self.value_net(shared)

    def forward_actor(self, features):
        shared = self.shared_net(features)
        return self.policy_net(shared)


class ImprovedPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, squash_output=False)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ResMlpExtractor(
            self.features_dim,
            self.action_space.shape[0],
        )
