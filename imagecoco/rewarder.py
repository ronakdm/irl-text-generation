import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class RewardModel(nn.Module):
    def __init__(self, hidden_state_size, mlp_hidden_size, embed_size, vocab_size):
        super(RewardModel, self).__init__()
        # Size of hidden state of generator model.
        self.hidden_state_size = hidden_state_size
        self.mlp_hidden_size = mlp_hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc_i = nn.Linear(hidden_state_size + embed_size, mlp_hidden_size)
        self.fc_h = nn.Linear(mlp_hidden_size, mlp_hidden_size)
        self.fc_o = nn.Linear(mlp_hidden_size, 1)

    def forward(self, x, a):
        """
        x : (batch_size, hidden_state_size) hidden state representing previously generated words.
        a : (batch_size,) words submitted at current timestep.
        """

        a_embed = self.embedding(a)
        z = torch.cat((x, a_embed), dim=len(x.shape) - 1)

        # TODO: Potentially change number of layers.
        z = F.relu(self.fc_i(z))
        z = F.relu(self.fc_h(z))
        output = self.fc_o(z)

        return output


class Rewarder:
    def __init__(
        self,
        seq_length,
        vocab_size,
        hidden_state_size,
        embed_dim,
        mlp_hidden_size,
        learning_rate,
        clip_max_norm,
        momentum,
        baseline=False,
    ):

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_state_size = hidden_state_size  # hidden state of generator
        self.embed_dim = embed_dim  # action embedding
        self.mlp_hidden_size = mlp_hidden_size  # hidden layers of reward model
        self.learning_rate = learning_rate
        self.clip_max_norm = clip_max_norm
        self.momentum = momentum
        self.baseline = baseline

        self.model = RewardModel(
            hidden_state_size, mlp_hidden_size, embed_dim, vocab_size
        ).cuda()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.learning_rate,
            self.momentum,
            weight_decay=1e-5,
        )

    def restore_model(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), self.learning_rate, self.momentum
        )

    def compute_rewards_to_go(self, trajectories, generator, roll_num=4):
        """
        Compute reward for each partitial trajectory t:seq_length
        for all t 1:seq_length

        Parameters:
            trajectories : (batch_size, seq_len) type: int 
            generator : Generator
            roll_num : int number of Monte Carlo estimates to compute roll-out estimate of value.

        Returns:
            rewards_to_go : (num_batches, batch_size, seq_length)
        """

        init_shape = trajectories.shape
        trajectories = trajectories.reshape((-1, self.seq_length))
        trajectories_real = generator.get_hidden_state(trajectories)
        batch_size = init_shape[0]

        rewards_to_go = torch.zeros(batch_size, self.seq_length).cuda()

        for t in range(self.seq_length):
            # Compute reward to go for each trajectory at s_t
            #   using MCMC sampling

            current_traj = trajectories[:, 0 : (t + 1)]
            expected_reward = 0
            for k in range(roll_num):
                rollouts, rollout_hidden_states = generator.generate(
                    batch_size,
                    1,
                    current_traj,
                    inc_hidden_state=True,
                    inc_probs=False,
                    decode=False,
                )

                rewards = self.model(
                    rollout_hidden_states.view(-1, self.hidden_state_size),
                    rollouts[0].view(-1),
                )
                result = rewards.view(batch_size, self.seq_length).sum(axis=1)

                expected_reward += result

            expected_reward /= roll_num

            # Compute the reward actually incurred by the trajectory.
            # (batch_size, remaining_seq_len)
            if self.baseline:
                remaining_trajectory = trajectories[:, t:]
                remaining_trajectory_hidden = trajectories_real[:, t:]
                realized_reward = self.model(
                    remaining_trajectory_hidden.view(-1, self.hidden_state_size),
                    remaining_trajectory.view(-1),
                )
                realized_reward = rewards.view(batch_size, self.seq_length).sum(axis=1)

                rewards_to_go[:, t] = realized_reward - expected_reward
            else:
                rewards_to_go[:, t] = expected_reward

        return rewards_to_go

    def train_step(self, real_batch, generator, generator_batch_size):
        """
        Perform one step of stochastic gradient descent for the Reward objective,
        as per equation (6) in https://arxiv.org/pdf/1804.11258.pdf.
        
        Parameters:
            real_batch : (batch_size, seq_len) Data from training set.
            generator : Generator
            generator_batch_size : int
        """

        # Compute reward for real sequences
        # Obtain batch of trajectories from real data. Each token is an embedding of the
        # state (context) at that index, embedded by GPT2 pre-trained layer.
        # Also store the actions taken at each timestep.
        real_batch_size = real_batch.shape[0]
        hidden_states_real = generator.get_hidden_state(real_batch)

        x_real = hidden_states_real.view(-1, self.hidden_state_size).cuda()
        a_real = real_batch.view(-1).cuda()

        # Compute reward for each state, action pair in the trajectories.
        reward_real = self.model(x_real, a_real).sum() / real_batch_size

        actions_gen, hidden_states_gen, log_probs = generator.generate(
            generator_batch_size,
            1,
            None,
            inc_hidden_state=True,
            inc_probs=True,
            decode=False,  # TODO: not sure about this
        )

        actions_gen = actions_gen[0]

        # Get rewards as a function of example, by summing over the sequence but not the batch size.
        reward_by_example = self.model(
            hidden_states_gen.view(-1, self.hidden_state_size), actions_gen.view(-1),
        )
        reward_by_example = reward_by_example.view(
            generator_batch_size, self.seq_length
        ).sum(axis=1)

        # (batch_size, seq_len, 1)
        indices = actions_gen.unsqueeze(-1)
        log_q = torch.gather(log_probs, 2, indices).sum(axis=1).cpu().data.numpy()
        log_w = reward_by_example.cpu().data.numpy() - log_q

        log_w -= log_w.max()
        w = torch.exp(torch.from_numpy(log_w)).cuda()
        reward_gen = torch.sum((w * reward_by_example) / w.sum())

        loss = -(reward_real - reward_gen)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
        self.optimizer.step()

        return loss.cpu().data.numpy()
