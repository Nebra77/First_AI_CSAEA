import torch 

with open('input.txt','r', encoding='utf-8') as f:
    text = f.read() 

chars = sorted(list(set(text))) #all unique characters, then list format, then sort
vocab = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) } #Dictionary, character is the key, index in chars is value
itos = { i:ch for i,ch in enumerate(chars) } #Dictionary, index in chars is the key, character is value

encode = lambda s: [stoi[c] for c in s]  # Convert given characters into ints; s is input
decode = lambda l: ''.join([itos[i] for i in l]) #Convert given ints into characters; l is input

data = torch.tensor(encode(text), dtype=torch.long) #Turn text -> ints -> tensor (number holder)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
'''Use the first 90% of the data to train the AI on 
what the response should look like, then compare its output with 
the actual last 10% of the data to see how they compare'''

block_size = 8 #size of sampled chunk
train_data[:block_size+1] 
'''allows for 8 predictions: "1 leads to 5; 1 and 5 leads to 37; 1, 5, and 37 leads to 12" a total of 8 times
Useful for training AI to being used to making predictions with little or a lot of context'''

