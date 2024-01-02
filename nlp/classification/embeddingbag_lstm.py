import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import time

from nlp.datasets import AmazonReviewHierarchical

train_dataset = AmazonReviewHierarchical(
    root='../data/amazon_review_hierarchical', split='train'
)
val_dataset = AmazonReviewHierarchical(
    root='../data/amazon_review_hierarchical', split='val'
)

tokenizer = get_tokenizer('basic_english')
def yield_tokens(dataiter):
    for text, _ in dataiter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(iter(train_dataset)), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

labels = set([label for _, label in iter(train_dataset)])
labels = {label: tuple(i for i, v in enumerate(labels) if v == label)[0]
          for label in labels}

text_pipeline = lambda x: vocab(tokenizer(x))

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _text, _label in batch:
        label_list.append(labels[_label])
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    return label_list, text_list, offsets

batch_size = 32
train_loader = DataLoader(
    dataset=train_dataset, shuffle=True, collate_fn=collate_batch,
    batch_size=batch_size
)
val_loader = DataLoader(
    dataset=val_dataset, shuffle=False, collate_fn=collate_batch,
    batch_size=batch_size
)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        print(embedded.size())
        out, _ = self.lstm(embedded)
        out = self.fc(out)
        return out

vocab_size = len(vocab)
embed_dim = 256
hidden_size = 128
num_layers = 2
num_class = len(labels)
model = TextClassifier(vocab_size, embed_dim, hidden_size, num_layers, num_class)

learning_rate = 5
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
criterion = nn.CrossEntropyLoss()

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 200
    start_time = time.time()
    for i, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted = model(text, offsets)
        print(predicted)
        print(label)
        loss = criterion(predicted, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if i % log_interval == 0 and i > 0:
            elapsed = time.time() - start_time
            print(
                f"| epoch {epoch:2d} | batch {i:4d}/{len(dataloader):4d} "
                f"| loss {loss.item():.4f} | accuracy {total_acc/total_count:.4f} "
                f"| time elapsed {elapsed:.1f}s |"
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for label, text, offsets in dataloader:
            predicted = model(text, offsets)
            total_acc += (predicted.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


total_accuracy = None
epochs = 15
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_loader)
    accuracy_val = evaluate(val_loader)
    if total_accuracy is not None and total_accuracy > accuracy_val:
        scheduler.step()
    else:
        total_accuracy = accuracy_val
    print("-" * 82)
    print(f"| end of epoch {epoch:2d} "
          f"| time elapsed {time.time() - epoch_start_time:.1f}s "
          f"| valid accuracy {accuracy_val:.4f}")
    print("-" * 82)