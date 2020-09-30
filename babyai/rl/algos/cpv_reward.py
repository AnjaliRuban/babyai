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
        context, h, c = self.encoder(context)

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
        context, _ = self.enc(context)
        dir1, dir2 = torch.split(context, context.shape[-1] // 2, dim=-1)
        context = dir2 + dir1

        ### Combinations ###
        dot_prod = torch.bmm(context, high.view(B, high.shape[1], 1)).squeeze() # -> B x M
        norms = torch.norm(high, dim=1).view((64, 1)).expand(-1, dot_prod.shape[1])

        # Similarity between high and current trajectory normalized by the high's norm. 
        sim = dot_prod / norms
        
        return sim

    def remove_spaces(self, s):
        cs = ' '.join(s.split())
        return cs

    def remove_spaces_and_lower(self, s):
        cs = self.remove_spaces(s)
        cs = cs.lower()
        return cs

    def calculate_reward(self, all_obs):

        # Unpack values from input. 
        high = [o['mission'] for o in all_obs[0]]

        obs = []
        for i in range(len(all_obs[0])):
            obs.append([o[i]['image'] for o in all_obs])

        # Tokenize highs. 
        high = [revtok.tokenize(self.remove_spaces_and_lower(h)) for h in high] # -> M
        high = [self.vocab.word2index([w.strip().lower() if w.strip().lower() in self.vocab.to_dict()['index2word'] else '<<pad>>' for w in h]) for h in high] # -> M

        # Put on device. 
        high = torch.tensor(high, dtype=torch.long)
        high = high.reshape(len(high), -1).to(self.device) # -> B x M
        high_len = high.bool().byte().sum(dim=1).view(-1,).to(self.device)

        traj = torch.tensor(obs, dtype=torch.float).view(len(obs), len(obs[0]), self.img_shape).to(self.device) # B X M X 147
        traj_len = torch.full((traj.shape[0],), traj.shape[1]).long().to(self.device)

        # Compute CPV reward with new observation incorporated. 
        with torch.no_grad(): 
            self.eval()
            sims = self.compute_similarity(high, traj, high_len, traj_len)

        # Potential-based reward is delta in similarity between previous and current trajectory. 
        reward = sims[:,1:] - sims[:,:-1]
        reward = torch.cat([torch.zeros((reward.shape[0],1), dtype=torch.float).to(self.device), reward], dim=1)

        return reward.detach()
