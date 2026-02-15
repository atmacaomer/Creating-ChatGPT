import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
n_iters = 5000
eval_iters = 200
eval_interval = 500
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
train_val_ratio = 0.9
n_embed = 384
max_new_token = 1000
dropout = 0.2
n_layer = 6
n_head = 6
torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

def encode(in_str : str) -> list[int]:
    """
    Takes string and encodes it to a integer style.
    """
    encoded = [stoi[x] for x in in_str]
    return encoded

def decode(in_int_lst) -> str:
    """
    Takes integer and decodes it to a string
    """
    decoded_Str = "".join([itos[i] for i in in_int_lst])
    return decoded_Str

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(data_type : str):
    data = train_data if data_type == "train" else val_data
    idx = torch.randint(len(data) - block_size, size=(batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in idx])
    yb = torch.stack([data[i+1:i+1+block_size] for i in idx])
    return xb, yb

@torch.no_grad()
def evaluate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            xb = xb.to(device)
            yb = yb.to(device)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape 
        q = self.query(x)  # B,T,C
        k = self.key(x)    #B,T,C
        wei = q @ k.transpose(-2,-1) * C**-0.5  # B,T,T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # B,T,C
        out = wei @ v  # B,T,C
        return out

class MultiHeadSA(nn.Module):
    def __init__(self, head_number, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(head_number)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_lst = [h(x) for h in self.heads]
        x_catted = torch.cat(x_lst, dim=-1)
        return self.dropout(self.projection(x_catted))
    

class FeedForwardNet(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embed),
                                 nn.ReLU(),
                                 nn.Linear(4 * n_embed, n_embed),
                                 nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, head_number):
        super().__init__()
        head_size = n_embed // head_number
        self.multiheadSA = MultiHeadSA(head_number, head_size)
        self.ffn = FeedForwardNet(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.multiheadSA(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BiagramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, head_number=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # B,T,C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #T,C
        inp = tok_emb + pos_emb
        inp = self.blocks(inp)
        logits = self.lm_head(self.ln_f(inp)) #B,T,vocab_size
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_token):
        for i in range(max_new_token):
            idx_truncated = idx[:, -block_size:]
            idx_truncated = idx_truncated.to(device)
            logits, loss = self(idx_truncated)
            logits = logits[:, -1, :]  
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

model = BiagramModel()
model = model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(n_iters):
    if iter % eval_interval == 0:
        out = evaluate_loss()
        print(f"Iteration {iter}. train loss {out['train']:.4f}, val loss {out['val']:.4f}")


    xb, yb = get_batch("train")
    xb = xb.to(device)
    yb = yb.to(device)
    logits, loss = model(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    if iter == n_iters-1:
        out = evaluate_loss()
        print(f"Iteration {iter}. train loss {out['train']:.4f}, val loss {out['val']:.4f}")
        print("Training finished.")



context = torch.zeros((1, 1), dtype=torch.long, device=device)
new_text = decode(model.generate(context, max_new_token=max_new_token)[0].tolist())
with open("new.txt", "w", encoding="utf-8") as f:
    f.write(new_text)
