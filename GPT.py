from modulefinder import Module
from sys import modules

import torch
import torch.nn as nn

from torch.nn import functional as F


torch.manual_seed(123)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# in our implementation: num_heads * atten_emb_dim == emb_dim
batch_size = 4
#maximum context length
seq_len= 8
emb_dim = 128
vocab_size = 56 # computed based on small data
atten_emb_dim = 32
n_layer = 6
num_heads = 4

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, atten_emb_dim):
        super().__init__()
        self.W_Q = nn.Linear(emb_dim, atten_emb_dim, bias=False)
        self.W_K = nn.Linear(emb_dim, atten_emb_dim, bias=False)
        self.W_V = nn.Linear(emb_dim, atten_emb_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        key = self.W_K(x)  # (batch_size, seq_len, atten_emb_dim)
        query = self.W_Q(x)  # (batch_size, seq_len, atten_emb_dim)
        value = self.W_V(x)  # (batch_size, seq_len, atten_emb_dim)
        # The transpose(-2, -1) swaps these last two dimensions.
        # So if x had a shape of (a, b, c), after applying x.transpose(-2, -1),
        # the tensor shape will become (a, c, b)

        # (batch_size, seq_len, atten_emb_dim) @ (batch_size, atten_emb_dim, seq_len ) =
        # (batch_size, seq_len, seq_len)
        wei = query @ key.transpose(-2, -1)  # (batch_size, seq_len, seq_len)
        # [[ 1,  2,  3,  4],
        #  [ 5,  6,  7,  8],
        #  [ 9, 10, 11, 12],
        #  [13, 14, 15, 16]]
        # normalize
        wei = wei * atten_emb_dim ** (-0.5)
        # ### the following part is only for decoder, for encode
        # we do not do masking and tril

        # torch.tril returns the lower triangular part of the matrix
        # upper triangle zero, others one
        tril = torch.tril(torch.ones(seq_len, seq_len))
        # [[ 1,  0,  0,  0],
        #  [ 1,  1,  0,  0],
        # [ 1,  1,  1,  0],
        # [ 1,  1,  1,  1]]

        # The masked_fill function sets elements in wei where the mask is True to the
        # specified value (float('-inf') in this case). so the upper tiangle
        # matrix of wei becomes -inf
        wei = wei.masked_fill(self.tril[:seq_len,:seq_len] == 0, float('-inf'))
        # [[  1, -inf, -inf, -inf],
        #  [  5,    6, -inf, -inf],
        #  [  9,   10,   11, -inf],
        #  [ 13,   14,   15,   16]]

        wei = F.softmax(wei, dim=-1)  # (
        # Note: Elements with -inf become zeros after exponentiation because \( e^{-\infty} = 0 \).
        # do the softmax operation per row
        # e.g., for row 2: input = [5, 6, -inf, -inf]
        # exponenet: [e^5, e^6, 0, 0]
        # compute exponent: [148.41,403.43,0,0]
        # sum of exponent: 148.41+403.43 = 551.84
        # softmax = [0.2689, 0.7311, 0.0000, 0.0000]

        # final softmax
        # [[1.0000, 0.0000, 0.0000, 0.0000],
        #  [0.2689, 0.7311, 0.0000, 0.0000],
        #  [0.0900, 0.2447, 0.6652, 0.0000],
        #  [0.0321, 0.0871, 0.2369, 0.6439]]

        # (batch_size, seq_len, seq_len) x (batch_size, seq_len, atten_emb_dim)=
        # (batch_size, seq_len, atten_emb_dim)
        z = wei @ value  # (batch_size, seq_len, atten_emb_dim)
        return z

class MultiHeadAttention(nn.Module):
    """multiple head of self-attention in parallel"""
    def __init__(self, num_heads, atten_emb_dim):
        super().__init__()
        self.heads = nn.ModuleList([Head(atten_emb_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(atten_emb_dim * num_heads , emb_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out) # (batch_size, seq_len, emb_dim)
        return out


class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        atten_emb_dim = emb_dim//num_heads # head_size = atten_emb_dim = 32
        self.sa = MultiHeadAttention(num_heads, atten_emb_dim)
        self.ffwd = FeedForward(emb_dim)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # per norm, applied before self attention, small deviation from original paper
        # residual connection
        x = x + self.sa(self.ln1(x)) #(batch_size, seq_len, emb_dim)
        x = x + self.ffwd(self.ln2(x)) #(batch_size, seq_len, emb_dim)
        return x


class FeedForward(nn.Module):
    """a simple linear layer followed by multi-head self-attention"""
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            # in implementation, it is 4*emb_dim, but here I used 2
            nn.Linear(emb_dim, 2 * emb_dim), #based on actual paper configuration: Attention is All You Need
            nn.ReLU(),
            nn.Linear(2 * emb_dim,emb_dim))

    def forward(self, x):
        return self.net(x)


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads the logits of the next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        # positional embedding
        self.position_emb = nn.Embedding(seq_len, emb_dim)

        self.blocks = nn.Sequential(*[Block(emb_dim, num_heads) for _ in range(n_layer)])

        self.final_ln = nn.LayerNorm(emb_dim)

        self.final_nn = nn.Linear(emb_dim, vocab_size)


    def forward(self, input, target=None):
        # input and targets are both (batch_size,seq_len) tensor of integers
        token_emb = self.token_embedding_table(input) # (batch_size, seq_len, emb_dim)

        batch_size, seq_len = input.shape
        pos_emb = self.position_emb(torch.arange(seq_len, device=device)) # (seq_len, emb_dim)

        # broadcasting happens within the first dimension of pos_emb to be in shape of
        # (batch_size, seq_len, emb_dim)
        x = token_emb + pos_emb # (batch_size, seq_len, emb_dim)

        # # pass it through blocks of multi-head attentions
        z = self.blocks(x)

        z = self.final_ln(z)

        logits = self.final_nn(z) # (batch_size, seq_len, vocab_size)

        if target is None:
            loss = None
        else:
            # reshape our logits and target to make compatible with pytorch cross_entropy function
            # becase in pytoch cross_entropy, the input should be in 2 dimension
            b, s, v = logits.shape
            logits = logits.view(b * s , v)
            target = target.view(b*s)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, input, max_new_tokens):
        # input is (batch_size, seq_len)
        for i in range(max_new_tokens):
            # get the prediction
            logits, loss = self(input)
            # input size should be the <= seq_len
            # crop input to the last block_size tokens
            input = input[:, -seq_len:]
            # logits[:,-1,:] select only the last element in 2nd dimension.
            # contains only the last vector (of size vocab_size) from the sequence dimension
            # seq_len for each batch element.
            # convert to (batch_size, vocab_size)
            logits = logits[:,-1,:]

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(batch_size, vocab_size)

            next_token = torch.multinomial(probs, num_samples=1) # (batch_size, 1)

            input = torch.cat((input, next_token), dim=1) # (batch_size, seq_len + 1)

        return input



def get_batch(data, seq_len, batch_size,device='cpu'):
    # random integer of size (batch_size), from 0 to the end except the last seq_len
    ix = torch.randint(len(data) -seq_len , (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]) # batch_size x seq_len
    y= torch.stack([data[i+1:i+seq_len+1] for i in ix]) # batch_size x seq_len
    x, y = x.to(device), y.to(device)
    return x,y


# Defining main function
def main():
    # load the data, 8493 lines
    text = open('small_data.txt', 'r', encoding='utf-8').read()

    # character level tokenizer
    # 56 characters
    chars = sorted(set(list(text)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenize the characters
    # {'\n': 0, ' ': 1, '!': 2, "'": 3, ',': 4, '-': 5, '.': 6,...
    s2i = {c:i for i,c in enumerate(chars)}
    i2s = {i:c for i,c in enumerate(chars)}

    encode = lambda arr: [s2i[c] for c in arr]
    decode = lambda arr: [i2s[i] for i in arr]

    data = torch.tensor(encode(text), dtype=torch.long)

    # first 90% as train and the last as val
    n = int(len(data)*0.9)
    train_data = data[:n]
    val_data = data[n:]

    x = train_data[:seq_len]

    for t in range(1,seq_len):
        context = x[:t]
        target = x[t]
        print(f"where input is {context}, the target is {target}")

    torch.manual_seed(123)
    x,y = get_batch(train_data, seq_len, batch_size)
    for b in range(batch_size):
        print(x[b, :])
        print(y[b, :])
        for t in range(seq_len):

            print(f"when input is {x[b,:t+1]}, output is {y[b,t]}")

    model = GPTLanguageModel()
    model.to(device)
    output, loss = model(x,y)
    print(output.shape) #[4, 8, 56] (batch_size, seq_len, vocab_size)
    print(loss)
    # by calling the model, we create a logits (probability distribution)
    # for each token in each seq and each batch individually, that's why the output size
    # is batch_size x seq_len x vocab_size


    # generate
    # create a sample input of size (1,1), with one batch, and seq_len = 1
    # the sample input starts with new line, the code of '\n' is 0
    sample_input = torch.zeros((1,1), dtype= torch.long, device=device)
    generation = model.generate(sample_input, 8)
    generation.shape #(batch_size, seq_len+num_new_tokens)
    #
    generated_tokens = generation[0].tolist()
    print(generated_tokens)
    print("".join(decode(generated_tokens)))
    #
    eval_iters = 50
    @torch.no_grad()
    def estimate_loss():
        """
        It averages loss over multiple batches.
        The estimate loss will be less noisy
        :return:
        """
        out = {}
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, batch_size, seq_len)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out['train'] = losses.mean()

        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(val_data, batch_size, seq_len)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out['val'] = losses.mean()

        model.train()
        return out
    # now train the model
    eval_interval = 100 # after each 100, compute the loss for train and val
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    for step in range(1000):
        # every once in a while evaluate the loss on train and val sets
        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch(train_data, seq_len, batch_size)

        #evaluate the loss
        logits, loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate new tokens after training the model
    sample_input = torch.zeros((1,1), dtype= torch.long, device=device)
    generation = model.generate(sample_input, 8)
    generation.shape #(batch_size, seq_len+num_new_tokens)

    generated_tokens = generation[0].tolist()
    print(generated_tokens)
    print("".join(decode(generated_tokens)))



# Using the special variable
# __name__
if __name__=="__main__":
    main()





