import torch
import revtok
from torch import nn
from vocab import Vocab

class Baseline(nn.Module):
    def __init__(self, primed_model):
        super().__init__()

        self.pad = 0
        self.seg = 1

        self.device = torch.device('cuda')
        self.args = primed_model['args']
        self.vocab = primed_model['vocab']

        self.img_shape = 7 * 7 * 3

        self.embed = self.embed = nn.Embedding(len(self.vocab), self.args.demb)
        self.linear = nn.Linear(self.args.demb, self.img_shape)
        self.enc = nn.LSTM(self.img_shape, self.args.dhid, bidirectional=True, batch_first=True)
        self.to(self.device)

        self.load_state_dict(primed_model['model'], strict=False)

    def encoder(self, batch, batch_size, h_0=None, c_0=None):
        '''
        Runs the sequences through an LSTM
        '''

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> L * 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> L * 2 x B x H
        out, (h, c) = self.enc(batch, (h_0, c_0)) # -> L * 2 x B x H

        hid_sum = torch.sum(h, dim=0) # -> B x H

        return hid_sum, h, c

    def forward(self, high, contexts, target):
        '''
        Takes in language mission (string), past observations (list of imgs), and
        the next action observation (img) and returns the dot product of each
        enc(high) - enc(context) with each enc(target)
        '''
        contexts_len = contexts.shape[0]

        ### HIGHS ###
        high = self.embed(high) # -> 1 x M x D (D = embedding size)
        high = self.linear(high) # -> 1 x M x 147
        high, h, c = self.encoder(high, 1) # -> 1 x H

        ### CONTEXTS ###
        contexts = contexts.reshape(1, contexts_len, self.img_shape)
        contexts, _, _ = self.encoder(contexts, 1, h, c) # -> 1 x N x H
        contexts = contexts.squeeze() # -> H

        ### TARGETS ###
        target = target.reshape(1, 1, -1)
        target, _, _ = self.encoder(target, 1) # -> 1 x H
        target = target.squeeze() # -> H

        ### COMB ###
        sim_m = torch.matmul(contexts, target) # -> 1

        return sim_m

    def remove_spaces(self, s):
        cs = ' '.join(s.split())
        return cs

    def remove_spaces_and_lower(self, s):
        cs = self.remove_spaces(s)
        cs = cs.lower()
        return cs

    def calculate_reward(self, high, contexts, target):

        high = revtok.tokenize(self.remove_spaces_and_lower(high)) # -> M
        high = self.vocab.word2index([w.strip().lower() if w.strip().lower() in self.vocab.to_dict()['index2word'] else '<<pad>>' for w in high]) # -> M
        if high == []:
            high = [0]
        high = torch.tensor(high, dtype=torch.long) # -> M
        high = high.reshape(1, -1).to(self.device) # -> 1 x M

        contexts = torch.stack([torch.tensor(img, dtype=torch.float).reshape(self.img_shape) for img in contexts]).to(self.device) # -> N x 147

        target = torch.tensor(target, dtype=torch.float).reshape(self.img_shape).to(self.device) # -> 147

        self.eval()
        reward = self.forward(high, contexts, target).to(torch.device('cpu')).detach().numpy()

        return reward
