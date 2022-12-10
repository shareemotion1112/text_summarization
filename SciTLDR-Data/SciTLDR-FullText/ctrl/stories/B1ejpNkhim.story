Autoregressive recurrent neural decoders that generate sequences of tokens one-by-one and left-to-right are the workhorse of modern machine translation.

In this work, we propose a new decoder architecture that can generate natural language sequences in an arbitrary order.

Along with generating tokens from a given vocabulary, our model additionally learns to select the optimal position for each produced token.

The proposed decoder architecture is fully compatible with the seq2seq framework and can be used as a drop-in replacement of any classical decoder.

We demonstrate the performance of our new decoder on the IWSLT machine translation task as well as inspect and interpret the learned decoding patterns by analyzing how the model selects new positions for each subsequent token.

<|TLDR|>

@highlight

new out-of-order decoder for neural machine translation