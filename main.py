import torch 
import torch.nn as nn
from torch.nn import functional as F

block_size = 256 #size of sampled chunk
batch_size = 64 # How many chunks to process at once
max_iterations = 6000 
eval_interval = 500
learning_rate = 3e-4 #how much to adjust weights
device = 'cuda' if torch.cuda.is_available() else 'cpu' #use GPU, else CPU
eval_iters = 200 #find average of 200 batch losses to find loss
n_embd = 384 #numbers per vector
n_head = 6 #attention heads per block
n_layer = 6 #stacked transformer blocks
dropout = 0.2 

with open('input.txt','r', encoding='utf-8') as f:
    text = f.read() 

chars = sorted(list(set(text))) #all unique characters, then list format, then sort
vocab = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) } #Dictionary, character is the key, index in chars is value
itos = { i:ch for i,ch in enumerate(chars) } #Dictionary, index in chars is the key, character is value

encode = lambda s: [stoi[c] for c in s]  # Convert given characters into ints; s is input
decode = lambda l: ''.join([itos[i] for i in l]) #Convert given ints into characters; l is input

all_data = torch.tensor(encode(text), dtype=torch.long) #Turn text -> ints -> tensor (number holder)

n = int(0.9*len(all_data))
train_data = all_data[:n]
val_data = all_data[n:]
'''
Use the first 90% of the data to train the AI on 
what the response should look like, then compare its output with 
the actual last 10% of the data to see how they compare
'''


'''
train_data[:block_size+1]  
allows for 8 predictions: "1 leads to 5; 1 and 5 leads to 37; 1, 5, and 37 leads to 12" a total of 8 times
Useful for training AI to being used to making predictions with little or a lot of context
'''


def get_batch(split):
    data = train_data if split == 'train' else val_data 
    ix = torch.randint(len(data) - block_size, (batch_size,)) #Create tensor, 1 row of size batch, with ints [0, length of data - block_size], to find random positions
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
'''
x is a tensor 4x8, holds 4 chunks of 8 values
y holds the same, shifted left 1
numbers are random so AI doesn't focus on specific words rather than patterns
'''

