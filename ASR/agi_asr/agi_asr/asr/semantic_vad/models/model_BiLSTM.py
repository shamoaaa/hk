
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Neural Networks model : Bidirection LSTM
"""


class BiLSTM(nn.Module):
    
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_dim = 300
        self.num_layers = 1
        V = 7370
        D = 300
        C = 1
        self.embed = nn.Embedding(V, D, padding_idx=0)
        self.bilstm = nn.LSTM(D, self.hidden_dim // 2, num_layers=self.num_layers, dropout=0, bidirectional=True, bias=False)

        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)

    def forward(self, x):
        embed = self.embed(x)
        x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        y = self.hidden2label1(bilstm_out)
        y = self.hidden2label2(y)
        logit = torch.sigmoid(y)
        return logit