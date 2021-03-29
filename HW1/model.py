from typing import Dict
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.nn import Embedding

#
# class SeqClassifier(torch.nn.Module):
#     def __init__(
#         self,
#         embeddings: torch.tensor,
#         hidden_size: int,
#         num_layers: int,
#         dropout: float,
#         bidirectional: bool,
#         num_class: int,
#     ) -> None:
#         super(SeqClassifier, self).__init__()
#         self.embed = Embedding.from_pretrained(embeddings, freeze=False)
#         # TODO: model architecture
#
#     @property
#     def encoder_output_size(self) -> int:
#         # TODO: calculate the output dimension of rnn
#         raise NotImplementedError
#
#     def forward(self, batch) -> Dict[str, torch.Tensor]:
#         # TODO: implement model forward
#         raise NotImplementedError

#
# class GRUNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
#         super(GRUNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#
#         self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, x, h):
#         out, h = self.gru(x, h)
#         out = self.fc(self.relu(out[:, -1]))
#         return out, h
#
#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cuda')
#         return hidden

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=300,
            hidden_size=2048,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=False,
            # dropout = 0.1
        )

        self.classifier = nn.Sequential(
                                        nn.Linear(2048, 150)
                                        )


    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # h0 = torch.zeros(2*2, x.size(0), 1024)  # 同样考虑向前层和向后层
        # c0 = torch.zeros(2*2, x.size(0), 1024)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        out=self.classifier(r_out[:, -1, :])
        return out



class SlotRNN(nn.Module):
    def __init__(self):
        super(SlotRNN, self).__init__()
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=300,
            hidden_size=256,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True,
            # dropout = 0.1
        )

        self.classifier = nn.Sequential(
                                        nn.Linear(256*2, 10)
                                        )


    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # h0 = torch.zeros(2*2, x.size(0), 1024)  # 同样考虑向前层和向后层
        # c0 = torch.zeros(2*2, x.size(0), 1024)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        out=self.classifier(r_out.reshape(-1, 256*2))
        return out

'''
class seq2seqRNN(nn.Module):
    def __init__(self):
        super(seq2seqRNN, self).__init__()
        self.rnn_encoder = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=300,
            hidden_size=512,         # rnn hidden unit
            num_layers=3,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # bidirectional=True,
            # dropout = 0.1
        )

        self.rnn_decoder = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=512,
            hidden_size=10,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # bidirectional=True,
            # dropout = 0.1
        )


    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # h0 = torch.zeros(2*2, x.size(0), 1024)  # 同样考虑向前层和向后层
        # c0 = torch.zeros(2*2, x.size(0), 1024)
        r_out1, (h_n, h_c) = self.rnn_encoder(x, None)   # None represents zero initial hidden state
        r_out2, (h_n, h_c) = self.rnn_decoder(r_out1[:,:,:], None)
        # r_out = self.drop(r_out)
        # choose r_out at the last time step

        # fc_1 = F.relu(self.hiddenlayer1(r_out[:, -1, :]))
        # fc_2 = F.relu(self.hiddenlayer2(fc_1))
        out=r_out2[:, : , :]
        return out
        
'''

#
# class SlotRNN(nn.Module):
#     def __init__(self, batch_size, input_size, hidden_size, n_classes, bidirectional=False):
#         super(SlotRNN, self).__init__()
#         # self.vocab_size = vocab_size
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.n_classes = n_classes
#         self.batch_size = batch_size
#
#         # self.embedding = nn.Embedding(vocab_size, hidden_size)
#         self.dropout = nn.Dropout(p=0.25)
#         self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
#         self.linear = nn.Linear(hidden_size, n_classes)
#
#     def forward(self, input):
#         # input_embedded = self.embedding(input.view(self.batch_size, -1))
#         # input embedded size (batch_size, seq_len, input_size)
#         rnn_out, rnn_hidden = self.rnn(input, self.initHidden())
#         affine_out = self.linear(rnn_out.reshape(-1, self.hidden_size))
#         # return F.log_softmax(affine_out)
#         return affine_out
#
#     def initHidden(self):
#         #  (num_layers, batch, hidden_size)
#         init_hidden = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to('cuda'))
#         return init_hidden


