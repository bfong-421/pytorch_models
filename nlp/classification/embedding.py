import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

from nlp.datasets import AG_NEWS

train_dataset = AG_NEWS(
    root='../data/ag_news', split='train'
)
test_dataset = AG_NEWS(
    root='../data/ag_news', split='test'
)

tokenizer = get_tokenizer('basic_english')
def yield_tokens(dataiter):
    for text, _ in dataiter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(iter(train_dataset)), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

text_pipeline = lambda x: vocab(tokenizer(x))

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _text, _label in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text))
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    padded_text = pad_sequence(text_list, batch_first=True)
    lengths = torch.tensor(lengths)
    return label_list, padded_text, lengths

batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size,
    shuffle=True, collate_fn=collate_batch
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size,
    shuffle=False, collate_fn=collate_batch
)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        embedded = embedded.mean(dim=1)
        out = self.fc(embedded)
        return out


vocab_size = len(vocab)
embed_dim = 128
num_class = len(set([label for _, label in iter(train_dataset)]))
model = TextClassifier(vocab_size, embed_dim, num_class)

learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.01)
criterion = nn.CrossEntropyLoss()

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 200
    for i, (label, text, lengths) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted = model(text, lengths)
        loss = criterion(predicted, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.001)
        optimizer.step()
        total_acc += (predicted.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if i % log_interval == 0 and i > 0:
            print(
                f"| epoch {epoch:2d} | batch {i:4d}/{len(dataloader):4d} "
                f"| loss {loss.item():.4f} | accuracy {total_acc/total_count:.4f} |"
            )
            total_acc, total_count = 0, 0

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for label, text, lengths in dataloader:
            predicted = model(text, lengths)
            total_acc += (predicted.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


total_accuracy = None
epochs = 10
for epoch in range(1, epochs + 1):
    train(train_loader)
    accuracy_val = evaluate(test_loader)
    if total_accuracy is not None and total_accuracy > accuracy_val:
        scheduler.step()
    else:
        total_accuracy = accuracy_val
    print("-" * 62)
    print(f"| end of epoch {epoch:2d} "
          f"| valid accuracy {accuracy_val:.4f}")
    print("-" * 62)
