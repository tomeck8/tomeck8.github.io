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
![Transformer Model Architecture](/assets/images/Transformer_Architecture.png width="300")
The picture shows the model architecture of the transformer. It looks really complicated, so we'll break it up into understandable pieces.
Firstly, let's split the picture into the left and right side. The left side of the model is the part that takes a user input, e.g. in ChatGPT a question that you type in, and helps the model understand your question. The right side of the model is the part that makes the model produce text and that's what we'll be focussing on in this post.
So let's reduce the image to our scope.
![Transformer Model Architecture](/assets/images/GPT_Arch.png width="300")
