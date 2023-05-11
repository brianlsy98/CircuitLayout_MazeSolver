# reference : https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

from model.models import Policy, Value
import torch

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class Agent():
    def __init__(self, args):
        #### ===== Metal 1, Metal 2 routing infos ====== ####
        self.x_table, self.y_table = args.x_table, args.y_table
        #### =========================================== ####

        self.K_epochs = args.K_epochs
        self.eps_clip = args.eps_clip
        self.gamma = args.gamma

        # Current Policy & Value
        self.actor, self.critic = Policy(args), Value(args)
        self.optimizer = torch.optim.Adam([
                        {'params': self.actor.parameters(), 'lr': args.lr_actor},
                        {'params': self.critic.parameters(), 'lr': args.lr_critic}
                    ])
        # Old Policy & Value
        self.actor_old, self.critic_old = Policy(args), Value(args)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
        self.buffer = RolloutBuffer()
        


    def get_action(self, state):

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.actor_old(state_tensor)
        
        # actor old results add.
        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()



    def train(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.critic(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * torch.nn.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_dir):
        self.actor_old.save(f"{checkpoint_dir}/actor")
        self.critic_old.save(f"{checkpoint_dir}/critic")
