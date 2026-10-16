import torch 

with open('input.txt','r', encoding='utf-8') as f:
    text = f.read() 
chars = sorted(list(set(text))) #all unique characters, then list format, then sort
vocab = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) } #Dictionary, character is the key, index in chars is value
itos = { i:ch for i,ch in enumerate(chars) } #Dictionary, index in chars is the key, character is value
encode = lambda s: [stoi[c] for c in s]  # Convert given characters into ints; s is input
decode = lambda l: ''.join([itos[i] for i in l]) #Convert given ints into characters; l is input
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])