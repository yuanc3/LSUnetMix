# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMcell(nn.Module):
    def __init__(self, input_sz, hidden_sz, peephole=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.peephole = peephole
        self.V = nn.Parameter(torch.ones(input_sz, hidden_sz * 4)).to(self.device)
        self.U = nn.Parameter(torch.ones(hidden_sz, hidden_sz * 4)).to(self.device)
        self.bias = nn.Parameter(torch.ones(hidden_sz * 4)).to(self.device)
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        bs, seq_sz, m = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(self.device),
                        torch.zeros(bs, self.hidden_size).to(self.device))
        else:
            h_t, c_t = init_states
        
        HS = self.hidden_size
        for t in range(m):
            x_t = x[:, :, t]

            # batch the computations into a single matrix multiplication
            if self.peephole:
                gates = x_t @ self.U + c_t @ self.V + self.bias
            else:
                gates = x_t @ self.U + h_t @ self.V + self.bias
               
                g_t = torch.tanh(gates[:, HS*2:HS*3])

            i_t, f_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            if self.peephole:
                c_t = f_t * c_t + i_t * torch.sigmoid(x_t @ self.U + self.bias)[:, HS*2:HS*3]
                h_t = torch.tanh(o_t * c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
            # if t==0:
            #     import sys
            #     sys.exit(0)
            hidden_seq.append(h_t.unsqueeze(2))
        
        hidden_seq = torch.cat(hidden_seq, dim=2)

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous().permute(0,2,1)

        # h_t = torch.repeat_interleave(h_t, repeats=2, dim=1)
        # c_t = torch.repeat_interleave(c_t, repeats=2, dim=1)
        
        return hidden_seq, (h_t, c_t)


class LSTM(nn.Module):
    def __init__(self, input_size, channel_num, peephole=False):
        super().__init__()

        self.peephole = peephole
        self.lstm = nn.ModuleList()
        for i in range(4):
            self.lstm.append(LSTMcell(input_size, input_size, peephole))
        # self.lstm.append(cell)
    
    def forward(self,emb,state):
        i = 0
        
        output = []
        for cell in self.lstm:
            o, state = cell(emb[i], init_states = state)
            output.append(o)
            i+=1
        return output, state

        