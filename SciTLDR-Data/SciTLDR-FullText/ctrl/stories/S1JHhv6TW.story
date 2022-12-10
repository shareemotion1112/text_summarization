The driving force behind deep networks is their ability to compactly represent rich classes of functions.

The primary notion for formally reasoning about this phenomenon is expressive efficiency, which refers to a situation where one network must grow unfeasibly large in order to replicate functions of another.

To date, expressive efficiency analyses focused on the architectural feature of depth, showing that deep networks are representationally superior to shallow ones.

In this paper we study the expressive efficiency brought forth by connectivity, motivated by the observation that modern networks interconnect their layers in elaborate ways.

We focus on dilated convolutional networks, a family of deep models delivering state of the art performance in sequence processing tasks.

By introducing and analyzing the concept of mixed tensor decompositions, we prove that interconnecting dilated convolutional networks can lead to expressive efficiency.

In particular, we show that even a single connection between intermediate layers can already lead to an almost quadratic gap, which in large-scale settings typically makes the difference between a model that is practical and one that is not.

Empirical evaluation demonstrates how the expressive efficiency of connectivity, similarly to that of depth, translates into gains in accuracy.

This leads us to believe that expressive efficiency may serve a key role in developing new tools for deep network design.

<|TLDR|>

@highlight

We introduce the notion of mixed tensor decompositions, and use it to prove that interconnecting dilated convolutional networks boosts their expressive power.

@highlight

This paper theoretically validates that interconnecting networks with different dilations can lead to expressive efficiency using mixed tensor decomposition.

@highlight

The authors study dilated convolutional networks and show that intertwining two dilated convolutional networks A and B at various stages is more expressively efficient than not intertwining.

@highlight

Shows that the WaveNet's structural assumption of a single perfect binary tree is hindering its performance and that WaveNet-like architectures with more complex mixed tree structures perform better.