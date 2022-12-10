We present trellis networks, a new architecture for sequence modeling.

On the one hand, a trellis network is a temporal convolutional network with special structure, characterized by weight tying across depth and direct injection of the input into deep layers.

On the other hand, we show that truncated recurrent networks are equivalent to trellis networks with special sparsity structure in their weight matrices.

Thus trellis networks with general weight matrices generalize truncated recurrent networks.

We leverage these connections to design high-performing trellis networks that absorb structural and algorithmic elements from both recurrent and convolutional models.

Experiments demonstrate that trellis networks outperform the current state of the art methods on a variety of challenging benchmarks, including word-level language modeling and character-level language modeling tasks, and stress tests designed to evaluate long-term memory retention.

The code is available at https://github.com/locuslab/trellisnet .

<|TLDR|>

@highlight

Trellis networks are a new sequence modeling architecture that bridges recurrent and convolutional models and sets a new state of the art on word- and character-level language modeling.