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



## Attention

### Multi-Head Attention

## Add + Norm

## Feed Forward

## Linear

## Softmax

