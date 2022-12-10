Computations for the softmax function in neural network models are expensive when the number of output classes is large.

This can become a significant issue in both training and inference for such models.

In this paper, we present Doubly Sparse Softmax (DS-Softmax), Sparse Mixture of Sparse of Sparse Experts, to improve the efficiency for softmax inference.

During training, our method learns a two-level class hierarchy by dividing entire output class space into several partially overlapping experts.

Each expert is responsible for a learned subset of the output class space and each output class only belongs to a small number of those experts.

During inference, our method quickly locates the most probable expert to compute small-scale softmax.

Our method is learning-based and requires no knowledge of the output class partition space a priori.

We empirically evaluate our method on several real-world tasks and demonstrate that we can achieve significant computation reductions without loss of

can be as an extension of their method to allow overlapping hierarchy.

This is because, in language 42 modeling, it is often difficult to exactly assign a word to a single cluster.

For example, if we want to 43 predict next word of "I want to eat " and one possible correct answer is "cookie", we can quickly 44 notice that possible answer belongs to something eatable.

So if we only search right answer inside 45 words with the eatable property, we can dramatically increase the efficiency.

On the other, though

"cookie" is one of the correct answers, it might also like appear under some non-eatable context, such as "a piece of data" in computer science.

Thus, a two-level overlapping hierarchy can naturally 48 accommodate word homonyms like this by allowing each word to belong to more than one cluster.

We believe this observation is likely to be true in other applications besides language modeling.

The result is illustrated in FIG1 and FIG1 .

We found our DS-Softmax can perfectly capture the 75 hierarchy.

For sanity check and visualization purposes, the ground-truth two hierachy in the synthetic 76 data does not have overlappings.

g k = max i G i (h) and set all other gates to be zero.

Also, corresponding k-th expert is chosen.

DISPLAYFORM0 This allows gradient to be back-propagated to whole DISPLAYFORM1 Shazeer et al. FORMULA0 , normalization is done after top-K experts are selected.

We can not do that since

we only choose top-1 expert since it will carry no gradient information since it becomes constant 1.

Given the sparse gate, we compute the probability of class c as, DISPLAYFORM0 where W training with γ is a lasso threshold according to Eq. 4.

DISPLAYFORM1 DISPLAYFORM2 Loading Balance.

We denote the sparsity percentage out of full softmax in k-th expert as sparsity k 136 and proportion of k-th expert activated as utilization k .

Then, the overall speedup compared to the full 137 softmax can be calculated as as 1/ k (utilization k * sparsity k ).

Thus, better utilization is essential for 138 speedup as well.

For example, there is no speedup if the expert with full output space is always chosen.

We borrow a similar loading balance function from Shazeer et al. (2017) Figure 3 : The mitosis training scheme: the sparsity is inherited when parent experts produce offspring, reducing the memory requirements for training with more experts.

DISPLAYFORM0 DISPLAYFORM1

1: Initialization: Let x be the input, y be the corresponding label, H be the pretrained function, V be the output dimension and D(y , y) be an arbitrarily distance function.

Set W e ← parameters for experts and W g ← parameters for the gating network.

The hyper-parameter t denotes target performance.

2: while epoch < Max do 3: DISPLAYFORM0 for all W and then gradually breed to a bigger one after noisy cloning shown in Fig. 3 .

For each cloning, the 147 sparsity is inherited so that less memory is required.

For example, in one of our experiments, we only 148 need 3.25x memory with 64 experts compared to a full softmax implementation.

The final training algorithm.

Our final training objective, L all , consists of a combination of the 150 related contributions discussed above.

We describe our training procedure in Algorithm 1.

@highlight

We present doubly sparse softmax, the sparse mixture of sparse of sparse experts, to improve the efficiency for softmax inference through exploiting the two-level overlapping hierarchy. 

@highlight

The paper proposes the new Softmax algorithm implementation with two hierarchical levels of sparsity which speeds up the operation in language modeling.