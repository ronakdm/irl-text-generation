# import numpy as np
import torch
from torch.utils.data import DataLoader

from coco_dataset import COCOImageCaptionsDataset
from rewarder import Rewarder, RewardModel
from generator import Generator

import pickle


class Test:
    def __init__(
        self, seq_length, real_batch_size, gen_batch_size, embed_dim, mlp_hidden_size
    ):
        self.seq_length = seq_length
        self.real_batch_size = real_batch_size
        self.gen_batch_size = gen_batch_size
        self.vocab_size = 4839
        self.hidden_state_size = 768
        self.embed_dim = embed_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.learning_rate = 0.01

        str_map = pickle.load(open("rl-project/imagecoco/save/str_map.pkl", "rb"))
        self.generator = Generator(self.seq_length, str_map)

    def test_reward_model(self):
        model = RewardModel(
            hidden_state_size=self.hidden_state_size,
            mlp_hidden_size=self.mlp_hidden_size,
            embed_size=self.embed_dim,
            vocab_size=self.vocab_size,
        )

        x = torch.randn((self.real_batch_size, self.hidden_state_size))
        a = torch.randint(self.vocab_size, (self.real_batch_size,))

        result = model(x, a)

        assert result.shape[0] == self.real_batch_size
        assert result.shape[1] == 1

    def test_rewarder_compute_rewards_to_go(self):
        rewarder = Rewarder(
            self.seq_length,
            self.real_batch_size,
            self.gen_batch_size,
            self.vocab_size,
            self.hidden_state_size,
            self.embed_dim,
            self.mlp_hidden_size,
            self.learning_rate,
        )
        trajectories = torch.randint(
            self.vocab_size, (self.real_batch_size, self.seq_length)
        )
        result = rewarder.compute_rewards_to_go(trajectories, self.generator)

        assert result.shape == self.batch_size

    def test_rewarder_train_step(self):
        rewarder = Rewarder(
            self.seq_length,
            self.real_batch_size,
            self.gen_batch_size,
            self.vocab_size,
            self.hidden_state_size,
            self.embed_dim,
            self.mlp_hidden_size,
            self.learning_rate,
        )
        trajectories = torch.randint(
            self.vocab_size, (self.real_batch_size, self.seq_length)
        )
        rewarder.train_step(trajectories, self.generator)

    def test_generator_rl_train_step(self):
        rewarder = Rewarder(
            self.seq_length,
            self.real_batch_size,
            self.gen_batch_size,
            self.vocab_size,
            self.hidden_state_size,
            self.embed_dim,
            self.mlp_hidden_size,
            self.learning_rate,
        )
        generator_batch_size = 32

        self.generator.rl_train_step(self, rewarder, generator_batch_size)

    def test_dataloader(self):
        train_data = COCOImageCaptionsDataset("save/train_data.pkl")
        train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True)
        for batch_idx, (truth, m_in, mask) in enumerate(train_dataloader):
            print("batch_idx: ", batch_idx)
            print(truth.shape, m_in.shape, mask.shape)
            break

    def runtests(self):
        self.test_reward_model()
        self.test_rewarder_rewards_to_go()
        # self.test_rewarder_train_step()
        # self.test_dataloader()


if __name__ == "__main__":
    test = Test(15, 64, 32, 100, 128)
    test.runtests()

