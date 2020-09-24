import torch
import revtok
import numpy as np
from torch import nn
from vocab import Vocab
import matplotlib.pyplot as plt
import pdb
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class CPV(nn.Module):
    def __init__(self, primed_model='models/cpv_model.pth'):
        super().__init__()

        self.pad = 0
        self.seg = 1

        self.device = torch.device('cuda')
        primed_model = torch.load(primed_model, map_location=self.device)
        self.args = primed_model['args']
        self.vocab = primed_model['vocab']

        self.img_shape = 7 * 7 * 3

        self.embed = nn.Embedding(len(self.vocab), self.args.demb)
        self.linear = nn.Linear(self.args.demb, self.img_shape)
        self.enc = nn.LSTM(self.img_shape, self.args.dhid, bidirectional=True, batch_first=True)
        self.to(self.device)

        self.load_state_dict(primed_model['model'], strict=False)

    def encoder(self, batch, batch_size, h_0=None, c_0=None):
        '''
        Encodes a stacked tensor.
        '''

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        out, (h, c) = self.enc(batch, (h_0, c_0)) # -> 2 x B x H

        hid_sum = torch.sum(h, dim=0) # -> B x H

        return hid_sum, h, c

    def forward(self, high, context, target, high_lens, context_lens, target_lens):
        '''

        '''

        B = context.shape[0]

        ### High ###
        high = self.embed(high) # -> B x M x D
        high = self.linear(high) # -> B x M x 147
        high = pack_padded_sequence(high, high_lens, batch_first=True, enforce_sorted=False)
        high, _, _ = self.encoder(high, B) # -> B x H

        ### Context ###
        context = pack_padded_sequence(context, context_lens, batch_first=True, enforce_sorted=False)
        context, h, c = self.encoder(context, B)

        ### Target ###
        packed_target = pack_padded_sequence(target, target_lens, batch_first=True, enforce_sorted=False)
        target, _, _ = self.encoder(packed_target, B)

        ### Full Trajectory ###
        trajectory, _, _ = self.encoder(packed_target, B, h, c)

        ### Combinations ###
        output = {}
        output["H * C"] = torch.matmul(high, torch.transpose(context, 0, 1)) # -> B x B
        output["<H, C>"] = torch.bmm(high.reshape(B, 1, -1), context.reshape(B, -1, 1)).squeeze() # -> B
        output["<H, T>"] = torch.bmm(high.reshape(B, 1, -1), target.reshape(B, -1, 1)).squeeze() # -> B
        output["<H, N>"] = torch.bmm(high.reshape(B, 1, -1), trajectory.reshape(B, -1, 1)).squeeze() # -> B
        output["<H, C + T>"] = torch.bmm(high.reshape(B, 1, -1), (context + target).reshape(B, -1, 1)).squeeze() # -> B
        output["norm(H)"] = torch.norm(high, dim=1) # -> B
        output["norm(C)"] = torch.norm(context, dim=1) # -> B
        output["norm(T)"] = torch.norm(target, dim=1) # -> B
        output["norm(N)"] = torch.norm(trajectory, dim=1) # -> B
        output["cos(H, N)"] = F.cosine_similarity(high, trajectory) # -> B

        return output

    def compute_similarity(self, high, context, high_lens, context_lens): 
        """
        Compute similarity between a high level instruction 
        and a trajectory segment. 
        """

        B = context.shape[0]

        ### High ###
        high = self.embed(high) # -> B x M x D
        high = self.linear(high) # -> B x M x 147
        high = pack_padded_sequence(high, high_lens, batch_first=True, enforce_sorted=False)
        high, _, _ = self.encoder(high, B) # -> B x H

        ### Context ###
        context = pack_padded_sequence(context, context_lens, batch_first=True, enforce_sorted=False)
        context, h, c = self.encoder(context, B)

        ### Combinations ###
        sim = torch.matmul(high, torch.transpose(context, 0, 1)) # -> B x B

        return sim

    def remove_spaces(self, s):
        cs = ' '.join(s.split())
        return cs

    def remove_spaces_and_lower(self, s):
        cs = self.remove_spaces(s)
        cs = cs.lower()
        return cs

    def calculate_reward(self, cpv_buffer, new_obs, idx):

        # Unpack values from buffer. 
        highs = cpv_buffer['mission']
        prev_obs = cpv_buffer['obs']
        prev_rewards = cpv_buffer['prev_reward']

        # If first time step, then tokenize mission using observation. 
        if highs[idx] is None: 

            high = revtok.tokenize(self.remove_spaces_and_lower(new_obs[idx]['mission'])) # -> M
            high = self.vocab.word2index([w.strip().lower() if w.strip().lower() in self.vocab.to_dict()['index2word'] else '<<pad>>' for w in high]) # -> M

            highs[idx] = high

        # Put on device. 
        high = torch.tensor(highs[idx], dtype=torch.long) # -> M
        high = high.reshape(1, -1).to(self.device) # -> 1 x M

        high_len = high.bool().byte().sum().view(1,).to(self.device)

        # Add new observation to buffer. 
        prev_obs[idx].append(new_obs[idx]['image'].reshape(self.img_shape))

        # Full trajectory with new observation. 
        traj = torch.tensor(prev_obs[idx], dtype=torch.float).view(1, len(prev_obs[idx]), -1).to(self.device)
        traj_len = torch.tensor([len(prev_obs[idx])]).to(self.device)

        # Compute CPV reward with new observation incorporated. 
        self.eval()
        new_sim = self.compute_similarity(high, traj, high_len, traj_len).item()

        # Potential-based reward is delta in similarity between previous and current trajectory. 
        reward = new_sim - prev_rewards[idx]

        # Store in cache for use next iteration. 
        prev_rewards[idx] = reward

        return reward
