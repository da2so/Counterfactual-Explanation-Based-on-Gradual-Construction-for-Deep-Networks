import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.conv2d_1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2d_2=nn.Conv2d(32, 32, 3, 1)
        #
        # self.conv2d_3 = nn.Conv2d(32, 64, 3, 1)
        # self.conv2d_4 = nn.Conv2d(64, 64, 3, 1)
        #
        #
        # self.dense_1 = nn.Linear(1024, 200)
        # self.dense_2 = nn.Linear(200, 200)
        # self.dense_3 = nn.Linear(200, 10)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv1_1 = nn.Conv2d(32, 32, 3, 1)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1)

        self.fc3 = nn.Linear(1024, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 10)
        # self.l1 = nn.Linear(784, 520)
        # self.l2 = nn.Linear(520, 320)
        # self.l3 = nn.Linear(320, 240)
        # self.l4 = nn.Linear(240, 120)
        # self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)

        output = F.log_softmax(x, dim=1)
        return output


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

    def extract_em(self):
        return self.embedded

    def expl(self, em):
        embedded = em.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)