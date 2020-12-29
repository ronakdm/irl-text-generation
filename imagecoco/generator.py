import numpy as np
import torch.nn as nn
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

from torch.distributions import Categorical
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

def batched_index_select(t, dim, inds):
    res = []

    if len(inds.shape) == 2:
        for i in range(len(inds)):
            res.extend(t[i][inds[i]].cpu())
    elif len(inds.shape) == 1:
        for i in range(len(inds)):
            res.append(t[i][inds[i]].cpu())

    return torch.stack(res).cuda()

class Generator:
    def __init__(self, seq_len, str_map, clip_max_norm, num_decoder_train=2):
        self.seq_len = seq_len
        self.vocab_size = len(str_map)
        self.clip_max_norm = clip_max_norm

        # declare our model, wanting to see hidden states
        self.model = GPT2LMHeadModel.from_pretrained(
            "gpt2", output_hidden_states=True, use_cache=True
        ).cuda()

        # freeze transformer
#        for i in range(self.model.config.n_layer - num_decoder_train):
#            for param in self.model.transformer.h[i].parameters():
#                param.requires_grad = False

        for param in self.model.transformer.parameters():
             param.requires_grad = False

        # mod head for our coco vocab
        self.model.lm_head = nn.Linear(
            self.model.lm_head.in_features, self.vocab_size
        ).cuda()

        # Just making sure the FC layer is not frozen :)
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        self.hidden_state_size = 768

        # we will use AdamW as the optimizer
        self.loss = nn.CrossEntropyLoss()
        self.optim = AdamW(self.model.parameters(), lr=5e-5)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # map to map non-gpt vocab back into strings
        self.str_map = np.array(str_map)

    def restore_model(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint)
        self.optim = AdamW(self.model.parameters(), lr=5e-5)

    def generate(
        self, batch_size, num_batches, start_toks, inc_hidden_state, inc_probs, decode
    ):
        """
        Returns:
            generated : (num_batches, batch_size, seq_len)
            h_states : (batch_size * num_batches, seq_len, hidden_state_size)
            probs : (batch_size * num_batches, seq_len, vocab_size)
        """
        # put into eval mode
        self.model.eval()

        # placeholder for generated words
        generated = torch.empty(
            batch_size * num_batches, self.seq_len, dtype=torch.long
        ).cuda()

        # tensor of probabilities
        probs = torch.empty(
            batch_size * num_batches, self.seq_len, self.vocab_size
        ).cuda()

        # tensor of hidden states
        h_states = torch.empty(
            batch_size * num_batches, self.seq_len, self.model.config.n_embd
        ).cuda()

        # start token

        if start_toks is None:  # begin with <eos>
            tok = 50256 * torch.ones(batch_size * num_batches, dtype=torch.long).cuda()
            attn_mask = torch.ones(batch_size * num_batches, dtype=torch.long).cuda()
        else:
            str_map = self.str_map[start_toks].tolist()
            gpt_map = self.tokenizer(str_map, padding=True, is_split_into_words=True)
            tok = torch.tensor(gpt_map["input_ids"]).cuda()
            attn_mask = torch.tensor(gpt_map["attention_mask"]).cuda()
            tok_mask = attn_mask.argmax(1)

        # generate sequence
        for i in range(self.seq_len):
            # forward pass + extract data
            res = self.model(input_ids=tok, attention_mask=attn_mask)
            prob, _, h_state = res[0], res[1], res[2][-1]

            # pick out most recent token (if inputted > 1 token)
            # TODO: fix this for having other starts than beg token
            if len(prob.shape) == 3:
                prob = batched_index_select(prob, 1, tok_mask)
                h_state = batched_index_select(h_state, 1, tok_mask)

            # Attach hidden state (last layer)
            h_states[:, i, :] = h_state.squeeze(1)

            # concat to probs array
            probs[:, i, :] = F.log_softmax(prob, dim=1)

            # Sample this prob dist for each sentence.
            dist = Categorical(F.softmax(prob, dim=1))

            # Add the new word to all the sentences (in non-gpt vocab)
            generated[:, i] = dist.sample()

            # map to gpt2 vocab
            str_map = self.str_map[generated[:, : i + 1].cpu()].tolist()
            gpt_map = self.tokenizer(str_map, padding=True, is_split_into_words=True)
            tok = torch.tensor(gpt_map["input_ids"]).cuda()
            attn_mask = torch.tensor(gpt_map["attention_mask"]).cuda()
            tok_mask = attn_mask.argmax(1)

        # decode=put back to string
        if decode:
            generated = self.str_map[generated.flatten().cpu()].reshape(
                batch_size * num_batches, self.seq_len
            )
        else:
            generated = torch.split(generated, batch_size, dim=0)

        res = [generated]
        if inc_hidden_state:
            res.append(h_states)
        if inc_probs:
            res.append(probs)

        return res

    # Gets hidden state for inputted data (for rewards).
    def get_hidden_state(self, batch):
        """
        batch: (batch_size, seq_len) int indicating index in <TODO> vocabulary.
        """
        self.model.eval()

        batch_size, seq_len = batch.shape
        res = torch.zeros((batch_size, seq_len, self.hidden_state_size))

        for t in range(seq_len):
            if t == 0:
                data = batch[:, 0].unsqueeze(1)
            else:
                data = batch[:, 0 : (t + 1)]
            # turn into gpt2 vocab
            # str_map = [self.str_map[data[i]].tolist() for i in range(len(data))]
            if t == 0:
                str_map = [[self.str_map[data[i].cpu()].tolist()] for i in range(len(data))]
            else:
                str_map = [self.str_map[data[i].cpu()].tolist() for i in range(len(data))]
            gpt_map = self.tokenizer(str_map, padding=True, is_split_into_words=True)
            tok = torch.tensor(gpt_map["input_ids"]).cuda()
            attn_mask = torch.tensor(gpt_map["attention_mask"]).cuda()
            tok_mask = attn_mask.argmax(1)

            # pass thru transformer
            h_state = self.model(input_ids=tok, attention_mask=attn_mask)[2][-1]
            h_state = batched_index_select(h_state, 1, tok_mask)

            res[:, t, :] = h_state

        return res

    # fine tune new FC layer  using normal transformer opt & train data
    # https://huggingface.co/transformers/custom_datasets.html
    def pretrain_step(self, batch):
         """
         pretrain_step: one step of pretraining

         param: batch 
         """
         self.model.train()

         # Get data from batch
         data, m_in, tok_mask = batch

         # Pass through model
         prob = self.model(input_ids=m_in)[0]
         prob = batched_index_select(prob, 1, tok_mask.bool())

         prob = F.softmax(prob, dim=-1).view(
             -1, self.vocab_size
         )

         # compute loss & backprop
         loss = self.loss(prob, data.flatten())
         loss.backward()
         self.optim.step()

         # ret loss
         return loss

    def rl_train_step(self, rewarder, generator_batch_size, roll_num=4):
        """
        Parameters
            batch_size: int
            rewarder: Rewarder
            roll_num: int
        """

        # Put model in train mode
        self.model.train()

        # Generate batch of data as sample trajectories.
        trajectories, log_probs = self.generate(
            generator_batch_size,
            1,
            None,
            inc_hidden_state=False,
            inc_probs=True,
            decode=False,
        )

        actions = trajectories[0]

        rewards_to_go = rewarder.compute_rewards_to_go(
            actions.cpu(), self, roll_num=roll_num
        )

        # Find the log_probs pi_theta(a_t, s_t) for the actions in the trajectories.
        # Shape: (batch_size, seq_length)

        # Gather values along vocabulary axis.
        indices = actions.unsqueeze(-1)
        log_probs_trajectory = torch.gather(log_probs, 2, indices).squeeze()

        # Pull out the data of this tensor so that gradient doesn't backpropagate through.
        log_probs_static = log_probs_trajectory.cpu().data.numpy()

        a = rewards_to_go.cpu().data.numpy() - log_probs_static - 1
        b = log_probs_trajectory
        reward = torch.sum(torch.from_numpy(a).cuda() * b) / generator_batch_size

        loss = -reward
        self.optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
        self.optim.step()

        return loss.cpu().data.numpy()
