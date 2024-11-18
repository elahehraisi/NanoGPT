from pickletools import optimize

import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads the logits of the next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input, target=None):
        # input and targets are both (batch_size,seq_len) tensor of integers
        logits = self.token_embedding_table(input) # batch_size, seq_len, vocab_size

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

    seq_len = 8 #maximum context length
    x = train_data[:seq_len]
    print(x)
    for t in range(1,seq_len):
        context = x[:t]
        target = x[t]
        print(f"where input is {context}, the target is {target}")

    torch.manual_seed(123)
    batch_size = 4 #how many independent sequences will be processed in parallel?
    x,y = get_batch(train_data, seq_len, batch_size)
    for b in range(batch_size):
        print(x[b, :])
        print(y[b, :])
        for t in range(seq_len):

            print(f"when input is {x[b,:t+1]}, output is {y[b,t]}")

    vocab_size = len(chars)
    model = BigramLanguageModel(vocab_size)
    model.to(device)
    output, loss = model(x,y)
    print(output.shape) #[4, 8, 56]
    print(loss)
    # by calling the model, we create a logits (probability distribution)
    # for each token in each seq and each batch individually, that's why the output size
    # is batch_size x seq_len x vocab_size


    # generate
    # create a sample input of size (1,1), with one batch, and seq_len = 1
    # the sample input starts with new line, the code of '\n' is 0
    sample_input = torch.zeros((1,1), dtype= torch.long, device=device)
    generation = model.generate(sample_input, 10)
    generation.shape #(batch_size, seq_len+num_new_tokens)

    generated_tokens = generation[0].tolist()
    print(generated_tokens)
    print(decode(generated_tokens))

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
    eval_interval = 100
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
    generation = model.generate(sample_input, 10)
    generation.shape #(batch_size, seq_len+num_new_tokens)

    generated_tokens = generation[0].tolist()
    print(generated_tokens)
    print("".join(decode(generated_tokens)))






# Using the special variable
# __name__
if __name__=="__main__":
    main()