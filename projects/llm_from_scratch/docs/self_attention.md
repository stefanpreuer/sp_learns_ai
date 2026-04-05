---
title: Self-attention
---

The architecture of the book centers around a self-attention based decoder only transformer architecture, like used in LLMs like GPT-3.

Self-attention allows for each input token in the context to take into account all other tokens of the input itself. This is different from the way RNN models work. I enabled a breakthrough in model performance for auto-regressive prediction of natural language.

The mechanism basically is centered about converting an a sequence of word embeddings into a sequence of context vectors. Each context vector takes into account all input tokens, based on a query-key-value approach:

For each of them a corresponding weight matrix for the input embeddings is defined. These are learnable weights, by which the model incorporates the self-referential semantic relationships withing a text sequence.

For each input token the query vector is similarity compared to each input token's key vector yielding the attention weights. These attention weights are arranged in the attention matrix as basis the context vector to be calculated by it as attention weights based weighted sum of the value vectors. Therefore, the attention matrix is adjusted for providing forward-looking only attention. This a context vector only considers the input value of itself or tokens before itself in the sequence. Technically this is done by a proper conversion of the attention matrix into a lower triangular matrix.

So on this high level it actually very simple. The context vector sequence is just the sequence of attention weighted input tokens, with the attention representing learned semantic interrelations within sequences of natural language regarding next token prediction.
