Large deep neural networks require huge memory to run and their running speed is sometimes too slow for real applications.

Therefore network size reduction with keeping accuracy is crucial for practical applications.

We present a novel neural network operator, chopout, with which neural networks are trained, even in a single training process, so as to truncated sub-networks perform as well as possible.

Chopout is easy to implement and integrate into most type of existing neural networks.

Furthermore it enables to reduce size of networks and latent representations even after training just by truncating layers.

We show its effectiveness through several experiments.

where where grad is a gradient and m is the number drawn in the forward pass.

DISPLAYFORM0

At test time, chopout is defined to behave as a identity function, that is, just pass through the input 45 vector without any modification.

This definition of chopout in prediction mode is contrastive to that 46 of dropout, which, in prediction time, dropout scale inputs to make it consistent with training time.

Training a fully-connected neural network with applying chopout can be interpreted as simultaneous 48 training of randomly sampled sub-networks which are obtained by cuttinng out former parts of the 49 original fully-connected neural network with sharing parameters.

In higher dimensional cases, chopout can be easily extended as a random truncation of channels 51 instead of dimensions.

For example, when applied to a tensor x ∈ R c×h×w , the forward-propagation 52 of chopout is defined as DISPLAYFORM0 where P ({0, 1, · · · , c}) is an arbitrary distribution.

Back-propagation is defined in the same way.

Throughout experiments, we use uniform distributions over {1, · · · , d} for P m ({0, 1, · · · , d}).

We train autoencoders on MNIST (LeCun et al. [1998] , Table 1 , FIG4 ).

We see that by applying 58 chopout on the hidden layer of the autoencoder, the reconstruction is kept well even after the hidden 59 layer is truncated.

We apply chopout for embeddings trained through skip-gram models (Mikolov et al. [2013a,b] ).

We 62 use text8 corpus 1 .

We set the window size to 5 and ignore infrequent words which appear less than 63 20 times in the corpus.

The result TAB2 shows the consistency of embeddings.

ran (512) ran (64) news (512) news (64) good (512)

(1) The distribution P m ({0, 1, · · · , d}) should be explored.

If we put chopouts in every layer of a 79 neural network, then, in training, there could be a layer where drawn m ∼ P m ({0, 1, · · · , d}) is very 80 small and it could be a bottleneck of the prediction accuracy.

(2) Chopout can be used for network pruning.

@highlight

We present a novel simple operator, chopout, with which neural networks are trained, even in a single training process, so as to truncated sub-networks perform as well as possible.