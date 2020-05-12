import torch
from torch import nn, optim
# from torchtext import data, datasets

print('GPU:', torch.cuda.is_available())

class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):

        super(RNN, self).__init__()

        # [0-10001] -> [100]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # [100] -> [256]
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)

        # [256*2] -> [1]
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # [seq, b, q] -> [seq, b, 100]
        embedding = self.dropout(self.embedding(x))

        # output: [seq, b, hid_dim*2]
        # hidden/h: [num_layers*2, b, hid_dim]
        # cell/c: [num_layers*2, b, hid_dim]
        output, (hidden, cell) = self.rnn(embedding)

        # [b, hid_dim*2]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim = 1)

        # [b, hid_dim*2] -> [b, 1]
        hidden = self.dropout(hidden)
        out = self.fc(hidden)

        return out