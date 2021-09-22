from collections import defaultdict


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)  # 不存在则返回unk

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]


"""
    # 如果在build中用self进行赋值要这么写
    vocab=Vocab()
    text=[["I","am","your","father"],["I","am","your","mother"],["I","am","xinyu"]]
    vocab.build(text=text,min_freq=0)

    # 现在用classmethod只需要这样,不用实例化对象
    text=[["I","am","your","father"],["I","am","your","mother"],["I","am","xinyu"]]
    Vocab.build(text=text,min_freq=0)
"""

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):  # 四个参数：输入、输出、中间层、词表
        super(MLP, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, offsets):
        embedding = self.embedding(inputs, offsets)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        probs = F.log_softmax(outputs, dim=1)
        return probs


from nltk.corpus import sentence_polarity


def load_sentence_polarity():
    vocab = Vocab.build(sentence_polarity.sents())
    train_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories='pos')][
                 :4000] + \
                 [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in
                  sentence_polarity.sents(categories='neg')][:4000]
    print(train_data[:5])

    test_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories='pos')][
                4000:] + \
                [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in
                 sentence_polarity.sents(categories='neg')][4000:]

    return train_data, test_data, vocab


def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    offsets = [0] + [i.shape[0] for i in inputs]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    inputs = torch.cat(inputs)
    return inputs, offsets, targets


from torch.utils.data import DataLoader

from torch.utils.data import Dataset


class BowDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


from tqdm.auto import tqdm
from torch import optim

embedding_dim = 128
hidden_dim = 256
num_class = 2
batch_size = 32
num_epoch = 5

train_data, test_data, vocab = load_sentence_polarity()
train_dataset = BowDataset(train_data)
test_dataset = BowDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device)

nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, offsets, targets = [x.to(device) for x in batch]
        log_probs = model(inputs, offsets)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()  # 梯度清零，防止累加
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss:{total_loss:.2f}")

acc = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, offsets, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, offsets)
        acc += (output.argmax(dim=1) == targets).sum().item()

print(f"Acc:{acc / len(test_data_loader):.2f}")
