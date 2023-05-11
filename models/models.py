import torch
import torch.nn as nn



# Discrete


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.state_dim = args.obs_dim
        self.action_dim = args.action_dim

        self.layers = nn.Sequential(
                        nn.Linear(self.state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, self.action_dim),
                        nn.Softmax(dim=-1)
                    )

    def forward(self, state):
        action_probs = self.layers(state)
        dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()\
    
    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)




class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()
        self.state_dim = args.obs_dim
        self.action_dim = args.action_dim

        self.layers = nn.Sequential(
                        nn.Linear(self.state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, self.action_dim),
                        nn.ReLU()
                    )
        
    def forward(self, old_state, old_action, policy):

        action_probs = policy(old_state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(old_action)
        dist_entropy = dist.entropy()
        state_values = self.layers(old_state)

        return action_logprobs, state_values, dist_entropy

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)
