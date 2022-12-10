Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can be prohibitively costly, especially on long sequences.

We introduce two techniques to improve the efficiency of Transformers.

For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from O(L^2) to O(L), where L is the length of the sequence.

Furthermore, we use reversible residual layers instead of the standard residuals, which allows storing activations only once in the training process instead of N times, where N is the number of layers.

The resulting model, the Reformer, performs on par with Transformer models while being much more memory-efficient and much faster on long sequences.

The Transformer architecture (Vaswani et al., 2017 ) is widely used in natural language processing and yields state-of-the-art results on a number of tasks.

To obtain these results, researchers have resorted to training ever larger Transformer models.

The number of parameters exceeds 0.5B per layer in the largest configuration reported in while the number of layers goes up to 64 in (Al-Rfou et al., 2018) .

Transformer models are also used on increasingly long sequences.

Up to 11 thousand tokens of text in a single example were processed in (Liu et al., 2018) and when processing other modalities, like music (Huang et al., 2018) and images , even longer sequences are commonplace.

These large-scale long-sequence models yield great results but strain resources to the point where some argue that this trend is breaking NLP research 1 .

Many large Transformer models can only realistically be trained in large industrial research laboratories and such models trained with model parallelism cannot even be fine-tuned on a single GPU as their memory requirements demand a multi-accelerator hardware setup even for a single training step.

Do large Transformer models fundamentally require such huge resources or are they simply inefficient?

Consider the following calculation: the 0.5B parameters used in the largest reported Transformer layer account for 2GB of memory.

Activations for 64K tokens with embedding size 1024 and batch size 8 account for 64K × 1K × 8 = 0.5B floats, requiring another 2GB of memory.

If our memory use was only per-layer, then we should fairly easily fit a large Transformer even on sequences of length 64K on a single accelerator.

Further, the whole corpus used to train BERT only requires 17GB to store.

Why is it then that we cannot even fine-tune these models on single machines?

The above estimate includes only per-layer memory and input activations cost and does not take into account the following major sources of memory use in the Transformer.

• Memory in a model with N layers is N -times larger than in a single-layer model due to the fact that activations need to be stored for back-propagation.

• Since the depth d f f of intermediate feed-forward layers is often much larger than the depth d model of attention activations, it accounts for a large fraction of memory use.

• Attention on sequences of length L is O(L 2 ) in both computational and memory complexity, so even for a single sequence of 64K tokens can exhaust accelerator memory.

We introduce the Reformer model which solves these problems using the following techniques:

1 https://hackingsemantics.xyz/2019/leaderboards/ • Reversible layers, first introduced in Gomez et al. (2017) , enable storing only a single copy of activations in the whole model, so the N factor disappears.

• Splitting activations inside feed-forward layers and processing them in chunks removes the d f f factor and saves memory inside feed-forward layers.

• Approximate attention computation based on locality-sensitive hashing replaces the O(L 2 ) factor in attention layers with O(L) and so allows operating on long sequences.

We study these techniques and show that they have negligible impact on the training process compared to the standard Transformer.

Splitting activations in fact only affects the implementation; it is numerically identical to the layers used in the Transformer.

Applying reversible residuals instead of the standard ones does change the model but has a negligible effect on training in all configurations we experimented with.

Finally, locality-sensitive hashing in attention is a more major change that can influence the training dynamics, depending on the number of concurrent hashes used.

We study this parameter and find a value which is both efficient to use and yields results very close to full attention.

We experiment on a synthetic task, a text task (enwik8) with sequences of length 64K and an image generation task (imagenet-64 generation) with sequences of length 12K.

In both cases we show that Reformer matches the results obtained with full Transformer but runs much faster, especially on the text task, and with orders of magnitude better memory efficiency.

Dot-product attention.

The standard attention used in the Transformer is the scaled dot-product attention (Vaswani et al., 2017) .

The input consists of queries and keys of dimension d k , and values of dimension d v .

The dot products of the query with all keys are computed, scaled by √ d k , and a softmax function is applied to obtain the weights on the values.

In practice, the attention function on a set of queries is computed simultaneously, packed together into a matrix Q. Assuming the keys and values are also packed together into matrices K and V , the matrix of outputs is defined as:

Multi-head attention.

length, length] .

In the experimental section we train a model on sequences of length 64K -in this case, even at batch-size of 1, this is a 64K × 64K matrix, which in 32-bit floats would take 16GB of memory.

This is impractical and has hindered the use of the Transformer for long sequences.

But it is important to note that the QK T matrix does not need to be fully materialized in memory.

The attention can indeed be computed for each query q i separately, only calculating softmax(

V once in memory, and then re-computing it on the backward pass when needed for gradients.

This way of computing attention may be less efficient but it only uses memory proportional to length.

We use this memory-efficient implementation of attention to run the full-attention baselines presented in the experimental section.

Where do Q, K, V come from?

The multi-head attention described above operates on keys, queries and values, but usually we are only given a single tensor of activations A of the shape [batch size, length, d model ] -e.g., coming from embedding the tokens in a sentence into vectors.

To build Q, K and V from A, the Transformer uses 3 different linear layers projecting A into Q, K and V with different parameters.

For models with LSH attention, we want queries and keys (Q and Figure 1 : An angular locality sensitive hash uses random rotations of spherically projected points to establish buckets by an argmax over signed axes projections.

In this highly simplified 2D depiction, two points x and y are unlikely to share the same hash buckets (above) for the three different angular hashes unless their spherical projections are close to one another (below).

K) to be identical.

This is easily achieved by using the same linear layer to go from A to Q and K, and a separate one for V. We call a model that behaves like this a shared-QK Transformer.

It turns out that sharing QK does not affect the performance of Transformer, even if we additionally normalize the length of the keys K, as we show in the experimental Section 5.

Hashing attention.

For the LSH attention, we start with two tensors, Q=K and V of the shape [batch size, length, d model ].

We keep the multi-head mechanism intact and focus on the attention computation from Equation 1.

As already mentioned, the main issue is the term QK T , which has the shape [batch size, length, length].

But note that we are actually only interested in softmax(QK T ).

Since softmax is dominated by the largest elements, for each query q i we only need to focus on the keys in K that are closest to q i .

For example, if K is of length 64K, for each q i we could only consider a small subset of, say, the 32 or 64 closest keys.

That is much more efficient, but how can we find the nearest neighbors among the keys?

Locality sensitive hashing.

The problem of finding nearest neighbors quickly in high-dimensional spaces can be solved by locality-sensitive hashing (LSH).

A hashing scheme that assigns each vector x to a hash h(x) is called locality-sensitive if nearby vectors get the same hash with high probability and distant ones do not.

In our case, we actually only require that nearby vectors get the same hash with high probability and that hash-buckets are of similar size with high probability.

We achieve this by employing random projections as follows (see Figure 1 ).

To get b hashes, we first fix a random matrix R of size [d k , b/2].

We then define h(x) = arg max([xR; −xR]) where [u; v] denotes the concatenation of two vectors.

This method is a known LSH scheme (Andoni et al., 2015) and is easy to implement and apply to batches of vectors.

LSH attention.

Knowing our LSH scheme and the general idea of hashing attention, we will now formalize the LSH attention we use in this paper.

We first rewrite the equation for normal attention, (1), for a single query position i at a time:

We introduce the notation P i to represent the set that the query at position i attends to, and z to denote the partition function (i.e. the normalizing term in the softmax).

For clarity, we also omit scaling by

For batching purposes we typically perform attention over a larger set P i = {0, 1, . . .

, l} ⊇ P i while masking out elements not in P i : Now we turn to LSH attention, which we can think of in terms of restricting the set P i of target items a query position i can attend to, by only allowing attention within a single hash bucket.

Figure 2 (a-b) shows a schematic comparison of full-attention with a hashed variant.

Part (a) depicts that the attention matrix for full attention is typically sparse, but the computation does not take advantage of this sparsity.

In (b), the queries and keys have been sorted according to their hash bucket.

Since similar items fall in the same bucket with high probability, the full attention pattern can be approximated by only allowing attention within each bucket.

Hash buckets in this formulation tend to be uneven in size, which makes it difficult to batch across buckets.

Moreover, the number of queries and the number of keys within a bucket may be unequalin fact, it is possible for a bucket to contain many queries but no keys.

To alleviate these issues, we first ensure that h(k j ) = h(q j ) by setting k j = qj qj .

Next, we sort the queries by bucket number and, within each bucket, by sequence position; this defines a permutation where i → s i after sorting.

In the sorted attention matrix, pairs from the same bucket will cluster near the diagonal (as depicted in Figure 2c ).

We can follow a batching approach where chunks of m consecutive queries (after sorting) attend to each other, and one chunk back (Figure 2d ).

Following our earlier notation, this corresponds to setting:

If max i |P i | < m, then P i ⊆ P i .

In practice we set m = 2l n buckets (where l is the sequence length).

The average bucket size is l n buckets , and we assume that the probability of a bucket growing to twice that size is sufficiently low.

The overall process of LSH attention is summarized in Figure 2 .

Multi-round LSH attention.

With hashing, there is always a small probability that similar items nevertheless fall in different buckets.

This probability can be reduced by doing multiple rounds of hashing with n rounds distinct hash functions {h

(1) , h (2) , . . .}, such that:

The multi-round case essentially involves performing LSH attention n rounds times in parallel; the details of the procedure are described in in Appendix A.

Causal masking for shared-QK attention.

In a Transformer decoder, masking (denoted by m(j, P i ) in Equation 3 ) is used to prevent positions from attending into the future.

To implement masking in LSH attention, we associate every query/key vector with a position index, re-order the position indices using the same permutations used to sort the query/key vectors, and then use a comparison operation to compute the mask.

While attention to the future is not allowed, typical implementations of the Transformer do allow a position to attend to itself.

Such behavior is undesirable in a shared-QK formulation because the dot-product of a query vector with itself will almost always be greater than the dot product of a query vector with a vector at another position.

We therefore modify the masking to forbid a token from attending to itself, except in situations where a token has no other valid attention targets (e.g. the first token in a sequence).

To verify the performance of LSH attention and study its behavior, we start with the following synthetic task: duplicate a sequence of symbols.

In this task, each training and testing example has the form 0w0w where w ∈ {1, . . .

, N } * is a sequence of symbols ranging from 1 to N (we use N = 127 in our experiments).

An example with the word w of length 3 is given below.

To study LSH attention, we train a language model on examples of the above form where each w is of length 511 (so the whole input 0w0w is of length 1024).

As this is a language modeling task, we always predict the next symbol given all the previous ones, but we mask the loss and accuracy to only consider positions in the second half of the input, i.e., those that can actually be predicted.

The above task can be solved perfectly (to accuracy 100% and loss 0) by a 1-layer Transformer model.

Note though, that it requires non-local attention lookups, so it cannot be solved by any model relying on sparse attention with a limited span.

To make it easy and fast to train but similar to models used in NLP, we use a 1-layer Transformer with d model = d f f = 256 and 4 heads.

We train it for 150K steps in 4 different settings: with full attention, LSH attention with n rounds = 1, n rounds = 2 and n rounds = 4.

From the results summarized in Table 2 we see that a model trained with full attention can be immediately used with LSH attention, but at some loss of accuracy.

When trained from scratch with LSH attention, the model trained with 4 hashes achieves almost perfect accuracy as well.

Interestingly, the accuracy becomes perfect when evaluated with 8 hashes.

It goes down when evaluated with 2 or 1 hashes.

Models trained with less hashes show worse results but even the model trained with just 1 hash performs almost perfectly when evaluated with 8 hashes.

As the above section shows, the complexity of attention can be reduced from square in length to linear, provided an approximation is acceptable.

But it is clear from Table 1 that each field starts with a b · n h · l term: the b · n h · l · d k , or alternatively b · l · d model cost cannot be avoided.

Indeed, the activations before each layer are already of the size b · l · d model , so the memory use of the whole model with n l layers is at least b · l · d model · n l .

Even worse: inside the feed-forward layers of Transformer this goes up to b · l · d f f · n l .

In a big Transformer it is usual to set d f f = 4K and n l = 16 so with l = 64K this again would use an impractical 16GB of memory

In this section, we show how to reduce this cost by first dealing with the n l part of the term using reversible layers and then showing how chunking can allow us to handle the d f f problem.

RevNets.

Reversible residual networks were introduced by Gomez et al. (2017) where it was shown that they can replace ResNets for image classification.

The main idea is to allow the activations at any given layer to be recovered from the activations at the following layer, using only the model parameters.

Rather than having to checkpoint intermediate values for use in the backward pass, layers can be reversed one-by-one as back-propagation proceeds from the output of the network to its input.

Whereas a normal residual layer performs a function x → y that operates on a single input and produces a single output and has the form y = x + F (x), a reversible layer works on pairs of inputs/outputs: (x 1 , x 2 ) → (y 1 , y 2 ), and follows the equations:

A layer can be reversed by subtracting (rather than adding) the residuals:

Reversible Transformer.

We apply the RevNet idea to the Transformer by combining the attention and feed-forward layers inside the revnet block.

In the notation above, F becomes an attention layer while G becomes the feed-forward layer.

Note that Layer Normalization (Ba et al., 2016 ) is moved inside the residual blocks.

The reversible Transformer does not need to store activations in each layer and so gets rid of the n l term.

In Section 5 we show that it performs the same as the normal Transformer when using the same number of parameters; we achieve this by having both x 1 and x 2 have size d model .

Chunking.

While reversibility covers the n l term, the thicker layers can still use a lot of memory.

The feed-forward layer in particular can use intermediate vectors of dimensionality d f f = 4K or higher.

However, computations in feed-forward layers are completely independent across positions in a sequence, so the computation can be split into c chunks:

This layer is typically batched by performing operations for all positions in parallel, but operating on one chunk at a time can reduce memory.

The reverse computation in (8) and the backward pass are also chunked.

In addition to the feed-forward layers, for models with large vocabulary (more than d model word types) we also chunk the log-probabilities at the output and calculate the loss for sections of the sequence at a time.

Chunking, large batches and parameter reuse.

With chunking and reversible layers the memory we use for activations in the whole network is independent of the number of layers.

The same is not true for parameters though as their number grows with the number of layers.

This problem is remedied though because we can swap layer parameters to and from CPU memory when this layer is not computing.

In a standard Transformer this would be inefficient because memory transfer to CPU is slow.

The batch size multiplied by length in Reformer is much larger though and therefore the amount of compute done with the parameters amortizes the cost of their transfer.

The Transformer model introduced in (Vaswani et al., 2017) has been used widely in natural language tasks and further extended to model diverse data such as music scores (Huang et al., 2018) , and images Ramachandran et al., 2019) .

Most notably, this model class has We assume n c = l/32 so 4l/n c = 128 and we write c = 128 2 .

max(bld f f , bn h ln r c)n l (bld f f + bn h n r lc)n l Reformer max(bld model , bn h ln r c) (bld f f + bn h n r lc)n l been applied successfully in the self-supervised training of extremely large language models (Devlin et al., 2018; .

Given the enormous computational requirements of state of the art sequence models, there has been increasing interest in finding methods to reduce the memory footprint and computational requirements of Transformer models.

In addition to standard methods such as precision reduction and gradient checkpointing (Sohoni et al., 2019) , more efficient versions of the Transformer model's self-attention mechanism (Sukhbaatar et al., 2019a; b) have also recently been explored.

In particular, leveraging sparsity in the attention layers has proved fruitful.

OpenAI introduced the sparse Transformer which exploits a factorized sparse representation of attention.

Using product-key attention to increase the key space has also been used to reduce memory requirements in the feed-forward layers with no loss in performance (Lample et al., 2019) .

Locality-sensitive hashing (LSH) has, to our knowledge, not been directly applied to Transformer attention layers before.

But previous work using external memory with neural networks has dealt with memories of large sizes.

The original implementation of memory networks (Weston et al., 2014) and later work on scaling it Chandar et al., 2016) used memory with size in the millions.

The cost of doing so is that the memory must be fixed prior to training.

Moreover, since during the beginning of training the model is unlikely to query the memory correctly, strong supervision is used to encourage the model to query memory locations that are useful.

These hints are either given as additional supervising information by the task or determined heuristically as in Hill et al. (2015) .

The requirement that the memory be fixed before has been removed in Santoro et al. (2016) at the cost of memory size and later alleviated by Rae et al. (2016) .

The last paper considered memory lookups with approximate nearest neighbors including both LSH and random kd-trees, but only for lookups in external memory.

In this section we present experimental results demonstrating the techniques described above.

We analyze the techniques one-by-one to make clear which combinations have impact on performance.

We start by showing that reversible layers and shared query-key spaces do not impact performance, then proceed to analyze hashing attention and finally the full Reformer model.

We ran our experiments on the imagenet64 and enwik8-64K tasks, where the latter is a variant of enwik8 that is chunked into subsequences of 2 16 = 64K tokens.

We use 3-layer models for our ablations so as to make it tractable to compare with the regular Transformer, which has high memory usage and performs full O(l 2 ) attention.

All experiments have d model = 1024, d f f = 4096, and n heads = 8.

They were trained with batch sizes of one sequence per GPU, with a total of 8 GPUs operating in parallel.

We used the Adafactor optimizer (Shazeer & Stern, 2018) for training our models.

Code for training our models will be made publicly available.

Effect of sharing QK.

We first consider the effect of shared-QK attention on a regular Transformer model.

Shared-QK attention sets k j = qj qj and prevents tokens from attending to themselves (except when no other context is available).

In the left part of Figure 3 , we plot perplexity curves for both regular and shared-QK attention.

A shared query-key space does not perform worse than Figure 3 : Effect of shared query-key space (left) and reversibility (right) on performance on enwik8 and imagenet64 training.

The curves show bits per dim on held-out data.

regular attention; in fact, for enwik8 it appears to train slightly faster.

In other words, we are not sacrificing accuracy by switching to shared-QK attention.

Effect of reversible layers.

In the two plots on the right in Figure 3 , we compare a regular Transformer per Vaswani et al. (2017) with the reversible one describe in Section 3.

The two models have identical parameter counts, and the learning curves likewise appear to be nearly the same.

These results show that the memory savings in the reversible Transformer do not come at the expense of accuracy.

LSH attention in Transformer.

LSH attention is an approximation for full attention that, as evidenced in Figure 4 , becomes more accurate as the number of hashes increases.

At n rounds = 8, it already almost matches full attention.

The computational cost of a model grows with the number of hashes, so this hyperparameter can be adjusted depending on the available compute budget.

Additionally, as in Table 2 , the number of hashes can be increased at evaluation time to produce more accurate results.

On the right half of Figure 5 , we plot the speed of different attention types vs. the sequence length, while holding the total number of tokens fixed.

We see that while regular attention becomes slower at longer sequence lengeth, LSH attention speed remains flat.

Large Reformer models.

To verify that the Reformer can indeed fit large models on a single core and train fast on long sequences, we train up to 20-layer big Reformers on enwik8 and imagenet64.

As can be seen in Figure 5 , these models fit into memory and train.

We were not able to train Transformer baselines in this case as they are too slow and memory-hungry, but we see clear improvement with the number of layers.

A 12-layer model on enwik8 trained for 20K steps with a dropout rate of 0.1 achieves 1.19 bits/dim on the test set.

We also trained a 12-layer Reformer model for longer with further tuning and improvements and we reached 1.05 bits/dim on the enwiki8 test set 3 .

Reformer combines the modeling capacity of a Transformer with an architecture that can be executed efficiently on long sequences and with small memory use even for models with a large number of layers.

We believe that this will help large, richly-parameterized Transformer models become more widespread and accessible.

Also, the ability to handle long sequences opens the way for the use of the Reformer on many generative tasks.

In addition to generating very long coherent text, the Reformer can bring the power of Transformer models to other domains like time-series forecasting, music, image and video generation.

In this section we describe in more detail the multi-hash version of our LSH attention mechanism.

We first repeat Equation (3) from the main text, which describes a general formulation of attention with sparsity:

exp (q i · k i − m(j, P i ) − z(i, P i )) v j where m(j, P i ) = ∞ if j / ∈ P i 0 otherwise

In the multi-round case, a query position i can attend to key positions P i as defined in (6), which we also repeat here:

For batching purposes, attention is performed on chunks of sorted queries/keys:

Combining (3) and (6) gives: i that can be computed independently from other rounds, except for the inclusion of a term N i,j to avoid double-counting elements when constructing the union of P (r) i sets.

In our implementation we fold the N i,j factor into the masking term m We also modify m (r) i,j to introduce a special case for i = j. This case is added because causal masking in a standard Transformer allows position i to attend to itself, which is not desirable in a shared-QK formulation.

We set the mask to a large but finite value to disallow attention-in-place, except in the situation where a token has no other valid attention targets.

For example, the first token in a sequence attends only to itself, because no prior context is available.

<|TLDR|>

@highlight

Efficient Transformer with locality-sensitive hashing and reversible layers