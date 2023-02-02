---
layout: post
title: Post #1
description: My first post
show_tile: false
image: 
---

# How to build a GPT
In easy terms, a Generatively Pretrained Transformer (GPT) is a language model that can write texts like a human. Probably you have played around with ChatGPT, which uses the GPT-3 model. But how does one build such a model? 

## The overall picture
![image](/assets/images/Transformer_Architecture.png)

The picture shows the model architecture of the transformer. It looks really complicated, so we'll break it up into understandable pieces.
Firstly, let's split the picture into the left and right side. The left side of the model is the part that takes a user input, e.g. in ChatGPT a question that you type in, and helps the model understand your question. The right side of the model is the part that makes the model produce text and that's what we'll be focussing on in this post.

## Output Embedding 
This is the first and easiest step. Since neural networks do not work with characters, but with numbers, we provide an numeric representation of our alphabet. We call this our vocabulary and it is nothing more than a mapping from (sequences of) letters to an integer.
The easiest mapping would just be assigning an integer to each letter in the alphabet. So the sequence of letters ['H', 'e', 'l', 'l', 'o'] might be translated to [0, 1, 2, 2, 3].
There are other embeddings that do not translate single letters to an integer, but something called "Sub-words", which are pieces of words or even entire short words. For example "Hello World." could be split into [Hello][_Wor][ld][.]. An example for this would be Google's [sentencepiece](https://github.com/google/sentencepiece).
Important note: Choosing between a letter-by-letter encoding and a sentence-piece encoding results in a trade-off between the length of translations of the text and a larger vocabulary.

A simple code on how encoding (and decoding) of single letters works:
```python
input_string = 'Hello'

chars = sorted(list(set(input_string)))
vocab_size = len(chars)

str_to_int = {s: i for i,s in enumerate(chars)}
int_to_str = {i: s for i,s in enumerate(chars)}
encode = lambda x: [str_to_int[l] for l in x]
decode = lambda x: ''.join([int_to_str[i] for i in x])

print(encode(input_string))
>>> [0, 1, 2, 2, 3]

print(int_to_str)
>>> {0: 'H', 1: 'a', 2: 'l', 3: 'o'}
```
Of course, in reality, the input and thus the vocabulary includes more letters.

Here's how you can introduce the embedding table for the token identity in a model using PyTorch.
```python
class Model(nn.Module):
  def __init__(self):
    token_embedding_table = nn.Embedding(vocab_size, n_embd)
  
  def forward(self, input, targets):
    tok_emb = self.token_embedding_table(input)
```

Besides the token identitiy, the Transformer uses a second embedding: the tokens positional encoding.

## Positional Encoding
To not only take the token identity into account, but also the position of the token, we introduce the position embedding table.
Therefore, let's look at one more thing: Whenever you feed the input text you want to train on into the model, it won't train on the complete text at once, but take smaller blocks of text out of the text one after the other. These blocks of text are referred to as the _context_ and usually have a fixed length of letters, which we'll call block_size.

```python
class Model(nn.Module):
  def __init__(self):
    token_embedding_table = nn.Embedding(vocab_size, n_embd)
    position_embedding_table = nn.Embedding(block_size, n_embd)
  
  def forward(self, input, targets):
    tok_emb = token_embedding_table(input)
    pos_emb = position_embedding_table(torch.arange(T))
    x = tok_emb + pos_emb
```

So now at this point, x contains both information about the token identity and the position.

## Linear
Let's see how we can get an interpretable output out of this. Therefore, we add a fully connected linear layer to the model with output of size vocab_size, to give us a score for each letter in our vocabulary.

```python
class Model(nn.Module):
  def __init__(self):
    token_embedding_table = nn.Embedding(vocab_size, n_embd)
    position_embedding_table = nn.Embedding(block_size, n_embd)
    fc = nn.Linear(n_embd, vocab_size)
  
  def forward(self, input, targets):
    tok_emb = token_embedding_table(input)
    pos_emb = position_embedding_table(torch.arange(T))
    x = tok_emb + pos_emb
    logits = fc(x)
```


## Attention
Now we get to the part that made the revolutionary ChatGPT possible in the first place: Attention!
Other models such as RNNs or LSTMs have one problem: their recurrent calculations are not parallelizable. Attention solves this issue and offers a high degree of distributed computing - using matrix multiplication.

When the model should produce text, in each step we want to predict the next letter based on the context. Here, context could refer to the complete block of letters, but effectively, we can only access the letters that came before the current one. Let me give you an example:
Imagine we have a block of letters "Hi Tom" (block_size = 6). The model would generate learnings iteratively:

1. Context: "H";     Target "i"
2. Context: "Hi";    Target " "
3. Context: "Hi ";   Target "T"
4. Context: "Hi T";  Target "o"
5. Context: "Hi To"; Target "m"

After the first step, the model has more than one input letter in the context. The goal of attention is to weigh the importance of each preceding letter. Let's look at the last iteration: The context is "Hi To" and consists of 5 letters. The easiest thing would be to give each letter the weight of 0.2 (so that the weights add up to one).
In preparation for attention we can do it like this for all 5 iterations:

```python
block_size = 5

tril = torch.tril(torch.ones(block_size, block_size))   # lower triangular matrix of ones
wei = torch.zeros(block_size, block_size)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)

print(wei)
>>> tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.5000, 0.5000, 0.0000, 0.0000, 0.0000],
            [0.3333, 0.3333, 0.3333, 0.0000, 0.0000],
            [0.2500, 0.2500, 0.2500, 0.2500, 0.0000],
            [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]])
```

For the sake of an example let's say x is the intermediate result from above code snippet.

```python
tril = torch.tril(torch.ones(block_size, block_size))
wei = torch.zeros(block_size, block_size)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
out = wei@x                     # @ is matrix multiplication
```

So now we restricted the influence of the letter to be predicted on the letters that came before the current target. Also, we weighted each of those letters to be equally important for the current target.
In reality, we do not want all letters in the context to be equally important but we want the weights to be data driven.
This is the last step we need to do before we get to attention as it is used in Transformers.

We will introduce 3 more vectors and their functions are usually described as follows: 
Key K: "What kind of information do I offer for other positions?"
Query Q: "What kind of information am I looking for in other positions?"
Value V: "I don't have a fancy interpretation! :("

The values of these vectors are nothing more the result of an linear activation of the intermediate result x. Through the multiplication of key and query, the different positions can communicate with one another and result in higher values (indicating higher importance for one another) or lower values (less importance).
So, let's write a class for an attention head and add it to our model.

```python
class AttentionHead(nn.Module):
  def __init__(self, head_size):
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))
  
  def forward():
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    
    wei = q@k.transpose()
    wei = wei.masked_fill(self.tril == 0, float('-inf')))
    wei = F.softmax(wei, dim=1)
    
    out = wei@v
    
    return out
```

And that's it. We created a single attention head. Now we can insert it into our model.


```python
class Model(nn.Module):
  def __init__(self):
    token_embedding_table = nn.Embedding(vocab_size, n_embd)
    position_embedding_table = nn.Embedding(block_size, n_embd)
    fc = nn.Linear(n_embd, vocab_size)
    attention_head = Head(n_embd)
  
  def forward(self, input, targets):
    tok_emb = token_embedding_table(input)
    pos_emb = position_embedding_table(torch.arange(T))
    x = tok_emb + pos_emb
    x = attention_head(x)
    logits = fc(x)
```

### Multi-Head Attention

## Add + Norm

## Feed Forward

## Softmax

