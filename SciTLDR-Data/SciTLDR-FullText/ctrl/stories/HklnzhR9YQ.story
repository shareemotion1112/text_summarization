We develop new approximation and statistical learning theories of convolutional neural networks (CNNs) via the ResNet-type structure where the channel size, filter size, and width are fixed.

It is shown that a ResNet-type CNN is a universal approximator and its expression ability is no worse than fully-connected neural networks (FNNs) with a \textit{block-sparse} structure even if the size of each layer in the CNN is fixed.

Our result is general in the sense that we can automatically translate any approximation rate achieved by block-sparse FNNs into that by CNNs.

Thanks to the general theory, it is shown that learning on CNNs satisfies optimality in approximation and estimation of several important function classes.



As applications, we consider two types of function classes to be estimated: the Barron class and H\"older class.

We prove the clipped empirical risk minimization (ERM) estimator can achieve the same rate as FNNs even the channel size, filter size, and width of CNNs are constant with respect to the sample size.

This is minimax optimal (up to logarithmic factors) for the H\"older class.

Our proof is based on sophisticated evaluations of the covering number of CNNs and the non-trivial parameter rescaling technique to control the Lipschitz constant of CNNs to be constructed.

Convolutional Neural Network (CNN) is one of the most popular architectures in deep learning research, with various applications such as computer vision (Krizhevsky et al. (2012) ), natural language processing (Wu et al. (2016) ), and sequence analysis in bioinformatics (Alipanahi et al. (2015) , Zhou & Troyanskaya (2015) ).

Despite practical popularity, theoretical justification for the power of CNNs is still scarce from the viewpoint of statistical learning theory.

For fully-connected neural networks (FNNs), there is a lot of existing work, dating back to the 80's, for theoretical explanation regarding their approximation ability (Cybenko (1989) , Barron (1993) , Lu et al. (2017) , Yarotsky (2017), and Petersen & Voigtlaender (2017) ) and generalization power (Barron (1994) , Arora et al. (2018), and Suzuki (2018) ).

See also Pinkus (2005) and Kainen et al. (2013) for surveys of earlier works.

Although less common compared to FNNs, recently, statistical learning theory for CNNs has been studied, both about approximation ability (Zhou (2018) , Yarotsky (2018) , Petersen & Voigtlaender (2018) ) and about generalization power (Zhou & Feng (2018) ).

One of the standard approaches is to relate the approximation ability of CNNs with that of FNNs, either deep or shallow.

For example, Zhou (2018) proved that CNNs are a universal approximator of the Barron class (Barron (1993) , Klusowski & Barron (2016) ), which is a historically important function class in the approximation theory.

Their approach is to approximate the function using a 2-layered FNN (i.e., an FNN with a single hidden layer) with the ReLU activation function (Krizhevsky et al. (2012) ) and transform the FNN into a CNN.

Very recently independent of ours, Petersen & Voigtlaender (2018) showed any function realizable with an FNN can extend to an equivariant function realizable by a CNN that has the same order of parameters.

However, to the best of our knowledge, no CNNs that achieves the minimax optimal rate (Tsybakov (2008) , Giné & Nickl (2015) ) in important function classes, including the Hölder class, can keep the number of units in each layer constant with respect to the sample size.

Architectures that have extremely large depth, while moderate channel size and width have become feasible, thanks to recent methods such as identity mappings (He et al. (2016) , Huang et al. (2018) ), sophisticated initialization schemes (He et al. (2015) , Chen et al. (2018) ), and normalization techniques (Ioffe & Szegedy (2015) , Miyato et al. (2018) ).

Therefore, we would argue that there are growing demands for theories which can accommodate such constant-size architectures.

In this paper, we analyze the learning ability of ResNet-type ReLU CNNs which have identity mappings and constant-width residual blocks with fixed-size filters.

There are mainly two reasons that motivate us to study this type of CNNs.

First, although ResNet is the de facto architecture in various practical applications, the approximation theory for ResNet has not been explored extensively, especially from the viewpoint of the relationship between FNNs and CNNs.

Second, constant-width CNNs are critical building blocks not only in ResNet but also in various modern CNNs such as Inception (Szegedy et al. (2015) ), DenseNet (Huang et al. (2017) ), and U-Net (Ronneberger et al. (2015) ), to name a few.

Our strategy is to replicate the learning ability of FNNs by constructing tailored ResNet-type CNNs.

To do so, we pay attention to the block-sparse structure of an FNN, which roughly means that it consists of a linear combination of multiple (possibly dense) FNNs (we define it rigorously in the subsequent sections).

Block-sparseness decreases the model complexity coming from the combinatorial sparsity patterns and promotes better bounds.

Therefore, it is often utilized, both implicitly or explicitly, in the approximation and learning theory of FNNs (e.g., Bölcskei et al. (2017) , Yarotsky (2018) ).

We first prove that if an FNN is block-sparse with M blocks (M -way block-sparse FNN), we can realize the FNN with a ResNet-type CNN with O(M ) additional parameters, which are often negligible since the original FNN already has Ω(M ) parameters.

Using this approximation, we give the upper bound of the estimation error of CNNs in terms of the approximation errors of block sparse FNNs and the model complexity of CNNs.

Our result is general in the sense that it is not restricted to a specific function class, as long as we can approximate it using block-sparse FNNs.

To demonstrate the wide applicability of our methods, we derive the approximation and estimation errors for two types of function classes with the same strategy: the Barron class (of parameter s = 2) and Hölder class.

We prove, as corollaries, that our CNNs can achieve the approximation error of orderÕ(M ) for the β-Hölder class, where M is the number of parameters (we used M here, same as the number of blocks because it will turn out that CNNs have O(M ) blocks for these cases), N is the sample size, and D is the input dimension.

These rates are same as the ones for FNNs ever known in the existing literature.

An important consequence of our theory is that the ResNet-type CNN can achieve the minimax optimal estimation error (up to logarithmic factors) for β-Hölder class even if its filter size, channel size and width are constant with respect to the sample size, as opposed to existing works such as Yarotsky (2017) and Petersen & Voigtlaender (2018) , where optimal FNNs or CNNs could have a width or a channel size goes to infinity as N → ∞.In summary, the contributions of our work are as follows:• We develop the approximation theory for CNNs via ResNet-type architectures with constant-width residual blocks.

We prove any M -way block-sparse FNN is realizable such a CNN with O(M ) additional parameters.

That means if FNNs can approximate a function with O(M ) parameters, we can approximate the function with CNNs at the same rate (Theorem 1).• We derive the upper bound of the estimation error in terms of the approximation error of FNNs and the model complexity of CNNs (Theorem 2).

This result gives the sufficient conditions to derive the same estimation error as that of FNNs (Corollary 1).• We apply our general theory to the Barron class and Hölder class and derive the approximation (Corollary 2 and 4) and estimation (Corollary 3 and 5) error rates, which are identical to those for FNNs, even if the CNNs have constant channel and filter size with respect to the sample size.

In particular, this is minimax optimal for the Hölder case.

We summarize in Table 1 the differences in the CNN architectures between our work and Zhou (2018) and Petersen & Voigtlaender (2018) , which established the approximation theory of CNNs via FNNs.

First and foremost, Zhou (2018) only considered a specific function class -the Barron class -as a target function class, although their method is applicable to any function class that can be realized by a 2-layered ReLU FNN.

Regarding the architecture, they considered CNNs with a single channel and whose width is "linearly increasing" (Zhou (2018)) layer by layer.

For regression or classification problems, it is rare to use such an architecture.

In addition, since they did not bound the norm of parameters in the approximating CNNs, we cannot derive the estimation error from this method.

Petersen & Voigtlaender (2018) fully utilized the group invariance structure of underlying input spaces to construct CNNs.

Such a structure makes theoretical analysis easier, especially for investigating the equivariance properties of CNNs since it enables us to incorporate mathematical tools such as group theory, Fourier analysis, and representation theory.

Although their results are quite strong in that it is applicable to any function that can be approximated by FNNs, their assumption on the group structure excludes the padding convolution layer, an important and popular type of convolution operations.

Another point is that if we simply apply their construction method to derive the estimation error for (equivariant) Hölder functions, combined with the approximation result of Yarotsky (2017), the resulting CNN that achieves the minimax optimal rate hasÕ(ε − D β ) channels where ε is the approximation error threshold.

It is partly because their construction is not aware of the internal sparse structure of approximating FNNs.

Finally, the filter size of their CNN is as large as the input dimension.

As opposed to these two works, we employ padding-and ResNettype CNNs which have multiple channels, fixed-size filters, and constant widths.

Like Petersen & Voigtlaender (2018) , our result is applicable to any function, as long as the FNNs to be approximated are block sparse, including the Barron and Hölder cases.

If we apply our theorem to these classes, we can show that the optimal CNNs can achieve the same approximation and estimation rate as FNNs, while the number of channels is independent of the sample size.

Further, this is minimax optimal up to the logarithmic factors for the Hölder class.

Due to its practical success, theoretical analysis for ResNet has been explored recently (e.g., Lin & Jegelka (2018) , Lu et al. (2018) , Nitanda & Suzuki (2018), and Huang et al. (2018) ).

From the viewpoint of statistical learning theory, Nitanda & Suzuki (2018) and Huang et al. (2018) investigated the generalization power of ResNet from the perspective of the boosting interpretation.

However, they did not discuss the function approximation ability of ResNet.

To the best of our knowledge, our theory is the first work to provide the approximation ability of the CNN class that can accommodate the ResNet-type ones.

We import the approximation theories for FNNs, especially ones for the Barron class and Hölder class.

The approximation theory for the Barron class has been investigated in e.g., Barron (1993) , Klusowski & Barron (2016), and Lee et al. (2017) .

Originally Barron (1993) considered the parameter s = 1 (see Definition 3) and the activation function σ satisfying σ(z) → 1 as z → ∞ and σ(z) → 0 as z → −∞. Later, Klusowski & Barron (2016) Using this bound, Schmidt-Hieber (2017) proved that the estimation error of the ERM estimator is O(N − 2β 2β+D ), which is minimax optimal up to logarithmic factors (see, e.g., Tsybakov (2008) ).

We consider a regression task in this paper.

Let X be a [−1, 1] D -valued random variable with unknown probability distribution P X and ξ be an independent random noise drawn from the Gaussian distribution with an unknown variance DISPLAYFORM0 • rigorously in the theorems later).

We define a random variable Y by Y := f• (X) + ξ.

We denote the joint distribution of (X, Y ) by P. Suppose we are given a dataset D = ((x 1 , y 1 ), . . .

, (x N , y N )) independently and identically sampled from the distribution P, we want to estimate the true function f• from the finite dataset D.We evaluate the performance of an estimator by the squared error.

For a measurable function f : DISPLAYFORM1 Here, clip is the clipping operator defined DISPLAYFORM2 we define the L 2 -norm (weighted by P X ) and the sup norm of f by DISPLAYFORM3 The task is to estimate the approximation error min f ∈F f − f • ∞ and the estimation error of the clipped ERM estimator: R(f ) − R(f • ).

Note that the estimation error is a random variable with respect the choice of the training dataset D. By the definition of R and the independence of X and ξ, the estimation error equals to f − f DISPLAYFORM4

In this section, we define CNNs used in this paper.

For this purpose, it is convenient to introduce 0 , the set of real-valued sequences whose finitely many elements are non-zero: 0 := {w = (w n ) n∈N>0 | ∃N ∈ N >0 s.t.

w n = 0, ∀n ≥ N }.

w = (w 1 , . . . , w K ) ∈ R K can be regarded as an element of 0 by setting w n = 0 for all n > K. Likewise, for C, C ∈ N >0 , which will be the input and output channel sizes, respectively, we can think of (w k,j,i DISPLAYFORM0 , we define the one-sided padding and stride-one convolution by w as an order-4 tensor DISPLAYFORM1 Here, i (resp.

j) runs through 1 to C (resp.

C ) and α and β runs through 1 to D. Since we fix the input dimension D throughout the paper, we will omit the subscript D and write as L w if it is obvious from context.

DISPLAYFORM2 Using this equality, we can expand a size-K filter to size-K .We can interpret L w as a linear mapping from DISPLAYFORM3 Next, we define the building block of CNNs: convolutional layers and fully-connected layers.

Let C, C , K ∈ N >0 be the input channel size, output channel size, and filter size, respectively.

For a weight tensor w ∈ R K×C ×C , a bias vector b ∈ R C , and an activation function σ : R → R, we define the convolutional layer Conv DISPLAYFORM4 where, ⊗ is the outer product of vectors and σ is applied in element-wise manner.

Similarly, let W ∈ R D×C , b ∈ R, and σ : R → R, we define the fully-connected layer FC DISPLAYFORM5 is the vectorization operator that flattens a matrix into a vector.

Finally, we define the ResNet-type CNN as a sequential concatenation of one convolution block, M residual blocks, and one fully-connected layer.

FIG2 is the schematic view of the CNN we adopt in this paper.

Definition 1 (Convolutional Neural Networks (CNNs)).

Let M ∈ N >0 and L m ∈ N >0 , which will be the number of residual blocks and the depth of m-th block, respectively.

Let C DISPLAYFORM6 m ∈ R be the weight tensors and biases of l-th layer of the m-th block in the convolution part, respectively.

Finally, let W ∈ R D×C (L 0 ) 0 and b ∈ R be the weight matrix and the bias for the fully-connected layer part, respectively.

For θ := ((w DISPLAYFORM7 is the identity function.

Although CNN σ θ in this definition has a fully-connected layer, we refer to the stack of convolutional layers both with or without the final fully-connect layer as a CNN in this paper.

We say a linear convolutional layer or a linear CNN when the activation function σ is the identity function and a ReLU convolution layer or a ReLU CNN when σ is ReLU defined by ReLU(x) = x ∨ 0.

We borrow the term from ResNet and call Conv For architecture parameters C = (C DISPLAYFORM8 , and norm parameters for convolution layers B (conv) > 0 and for fully-connected layers DISPLAYFORM9 , the hypothesis class consisting of ReLU CNNs, as follows: DISPLAYFORM10 Here, the domain of CNNs is restricted to [−1, 1] D .

Note that we impose norm constraints to the convolution part and fully-connected part separately.

We emphasize that we do not impose any sparse constraints (e.g., restricting the number of non-zero parameters in a CNN to some fixed value) to F (CNN) , as opposed to previous literature such as Yarotsky FORMULA24 , Schmidt-Hieber (2017), and Imaizumi & Fukumizu (2018) .

Since the notation is cluttered, we sometimes omit the subscripts as we do in the above.

Remark 2.

In this paper, we adopted one-sided padding, which is not often used practically, in order to make proofs simple.

However, with slight modifications, all statements are true for equallypadded convolutions, the widely employed padding style which adds (approximately) same numbers of zeros to both ends of an input signal, with the exception that the filter size K is restricted to DISPLAYFORM11 We also discuss our design choice, especially the comparison with the original ResNet proposed in He et al. (2016) in Section G of the appendix.

In this section, we mathematically define FNNs we consider in this paper, in parallel with the CNN case.

Our FNN, which we coin a block-sparse FNN, consists of M possibly dense FNNs (blocks) concatenated in parallel, followed by a single fully-connected layer.

We sketch the architecture of a block-sparse FNN in Figure 2 .

Let DISPLAYFORM0 m ∈ R be the weight matrix and the bias of the l-th layer of mth block (with the convention DISPLAYFORM1 be the weight (sub)vector of the final fully-connected layer corresponding to the m-th block and b ∈ R be the bias for the last layer.

DISPLAYFORM2 We call a block-sparse FNN with M blocks a M -way block-sparse FNN.

We say θ is compatible with (D [Lm] and norm parameters for the block part B (bs) > 0 and for the final layer DISPLAYFORM3 DISPLAYFORM4 , the set of function realizable by FNNs: DISPLAYFORM5 Again, the domain is restricted to [−1, 1] D .

Similar to the CNN case, we sometimes remove subscripts of the function class for simplicity.

With the preparation in the previous sections, we state our main results of this paper.

We only describe statements of theorems and corollaries and key ideas in the main article.

All complete proofs are deferred to the appendix.

Our first main theorem claims that any M -way block-sparse FNN is realizable by a ResNet-type CNN with fixed-sized channels and filters by adding O(M ) parameters, if we treat the widths D (l) m of the FNN as constants with respect to M .

DISPLAYFORM0 m , and DISPLAYFORM1 that is, any FNN in DISPLAYFORM2 An immediate consequence of this theorem is that if we can approximate a function f • with a blocksparse FNN, we can also approximate f• with a CNN.

Our second main theorem bounds the estimation error of the clipped ERM estimatorf .

DISPLAYFORM0 (conv) and B (fc) as in Theorem 1.

Suppose L m , C, K satisfies the equation FORMULA24 DISPLAYFORM1 Here, DISPLAYFORM2 .

M 1 and M 2 are defined by DISPLAYFORM3 The first term of FORMULA27 is the approximation error achieved by F (FNN) .

On the other hand, M 1 and M 2 are determined by the architectural parameters of F (CNN) -M 1 corresponds to the Lipschitz constant of a function realized by a CNN and M 2 is the number of parameters, including zeros, of a CNN.

Therefore, the second term of (2) represents the model complexity of F (CNN) .

There is a trade-off between the two terms.

Using appropriately chosen M to balance them, we can evaluate the order of estimation error with respect to the sample size N .

Corollary 1.

Under the same assumptions as Theorem 2, suppose further log DISPLAYFORM4

The Barron class is an example of the function class that can be approximated by block-sparse FNNs.

We employ the definition of Barron functions used in Klusowski & Barron (2016) .Definition 3 (Barron class).

We say a measurable function f DISPLAYFORM0 Here, F andF are the Fourier transformation and the inverse Fourier transformation, respectively.

Klusowski & Barron (2016) studied the approximation of the Barron function f• with the parameter s = 2 by a linear combination of M ridge functions (i.e., a 2-layered ReLU FNN).

Specifically, they showed that there exists a function f M of the form DISPLAYFORM1 with |b m | ≤ 1, a m 1 = 1 and |t m | ≤ 1, such that f DISPLAYFORM2 .

Using this approximator f M , we can derive the same approximation order using CNNs by applying Theorem 1 with DISPLAYFORM3 Barron function with the parameter s = 2 such that f• (0) = 0 and ∇f DISPLAYFORM4 with M residual blocks, each of which has depth O(1) and at most 4 channels, and whose filter size is at DISPLAYFORM5 We have one design choice when we apply Corollary 1 to derive the estimation error: how to set B (bs) and B (fin) .

Looking at (3), the naive choice would be B (1 + ρ m ) whose logarithm is O(M ).

We want its logarithm to beÕ(1).

In order to do that, we change the relative scale between parameters in the block-sparse part and the fully-connected part using the homogeneous property of the ReLU function: ReLU(ax) = aReLU(x) for a > 0.

The rescaling operation enables us to choose B , depth of each residual block L = O (1), channel size C = O(1), filter size K ∈ {2, . . .

, D}, and norm bounds for the con- DISPLAYFORM6 , and for the fully-connected part DISPLAYFORM7 such that for sufficiently large N , the clipped ERM estimatorf of F : DISPLAYFORM8 .

Here, DISPLAYFORM9

We next consider the approximation and error rates of CNNs when the true function is a β-Hölder function.

Definition 4 (Hölder class).

Let β > 0, f DISPLAYFORM0 Yarotsky FORMULA24 showed that FNNs with O(S) non-zero parameters can approximate any D variate β-Hölder function (β > 0) with the order ofÕ(S − β D ).

Schmidt-Hieber (2017) also proved a similar statement using a different construction method.

They only specified their width (Schmidt-Hieber (2017) only), depth, and non-zero parameter counts of the approximating FNN and did not write in detail how non-zero parameters are distributed explicitly in the statements (see Theorem 1 of Yarotsky (2017) and Theorem 5 of Schmidt-Hieber FORMULA24 ).

However, if we carefully look at their proofs, we find that we can transform the FNNs they constructed into the block-sparse ones.

Therefore, we can utilize these FNNs and apply Theorem 1.

To meet the assumption of Corollary 1, we again rescale the parameters of the FNNs, as we did in the Barron class case, so that log M 1 =Õ(1).

We can derive the approximation and estimation errors by setting γ 1 = β D and γ 2 = 1. and O(1) channels, and whose filter size is at most K, such that f DISPLAYFORM1 Corollary 5.

There exist the number of residual blocks DISPLAYFORM2 , depth of each residual block L =Õ (1), channel size C = O(1), filter size K ∈ {2, . . .

, D}, norm bounds for the convolution part B (conv) = O(1), and for the fully-connected part B (fc) > 0 (log B (fc) = O(log N )) such that for sufficiently large N , the clipped ERM estimatorf of F : DISPLAYFORM3 .

Here, DISPLAYFORM4 Since the estimation error rate of the β-Hölder class is DISPLAYFORM5 (see, e.g., Tsybakov (2008)), Corollary 5 implies that our CNN can achieve the minimax optimal rate up to logarithmic factors even the width D, the channel size C, and the filter size K are constant with respect to the sample size N .

In this paper, we established new approximation and statistical learning theories for CNNs by utilizing the ResNet-type architecture of CNNs and the block-sparse structure of FNNs.

We proved that any M -way block-sparse FNN is realizable using CNNs with O(M ) additional parameters, when the width of the FNN is fixed.

Using this result, we derived the approximation and estimation errors for CNNs from those for block-sparse FNNs.

Our theory is general because it does not depend on a specific function class, as long as we can approximate it with block-sparse FNNs.

To demonstrate the wide applicability of our results, we derived the approximation and error rates for the Barron class and Hölder class in almost same manner and showed that the estimation error of CNNs is same as that of FNNs, even if the CNNs have a constant channel size, filter size, and width with respect to the sample size.

The key techniques were careful evaluations of the Lipschitz constant of CNNs and non-trivial weight parameter rescaling of FNNs.

One of the interesting open questions is the role of the weight rescaling.

We critically use the homogeneous property of the ReLU activation function to change the relative scale between the block-sparse part and the fully-connected part, if it were not for this property, the estimation error rate would be worse.

The general theory for rescaling, not restricted to the Barron nor Hölder class would be beneficial for deeper understanding of the relationship between the approximation and estimation capabilities of FNNs and CNNs.

Another question is when the approximation and estimation error rates of CNNs can exceed that of FNNs.

We can derive the same rates as FNNs essentially because we can realize block-sparse FNNs using CNNs that have the same order of parameters (see Theorem 1).

Therefore, if we dig into the internal structure of FNNs, like repetition, more carefully, the CNNs might need fewer parameters and can achieve better estimation error rate.

Note that there is no hope to enhance this rate for the Hölder case (up to logarithmic factors) because the estimation rate using FNNs is already minimax optimal.

It is left for future research which function classes and constraints of FNNs, like block-sparseness, we should choose.

For tensor a, a + := a∨0 where maximum operation is performed in element-wise manner.

Similarly a − := −(−a ∨ 0).

Note that a = a + − a − holds for any tensor a. For normed spaces (V, · V ), (W, · W ) and linear operator T : V → W we denote the operator norm of T by T op := sup v V =1 T v W .

For a sequence w = (w (1) , . . .

, w (L) ) and l ≤ l , we denote its subsequence from the l-th to l -th elements by w[l : l ] := (w (l) , . . .

, w (l ) ).

1 P equals to 1 if the statement P is true, equals to 0 otherwise.

DISPLAYFORM0 , we realize a CNN f (CNN) using M residual blocks by "serializing" blocks in the FNN and converting them into convolution layers.

First, we double the channel size using the m = 0 part of CNN (i.e., D (L0) 0 = 2).

We will use the first channel for storing the original input signal for feeding to downstream (i.e., m ≥ 1) blocks and the second one for accumulating the output of each blocks, that is, m size-1 filters made from the weight parameters of the corresponding layer of the FNN.

Observing that the convolution operation with size-1 filter is equivalent to a dimension-wise affine transformation, the first coordinate of the output of l-th layer of the CNN is inductively same as that of the m-th block of the FNN.

After computing the m-th block FNN using convolutions, we add its output to the accumulating channel in the identity mapping.

Finally, we pick the first coordinate of the accumulating channel and subtract the bias term using the final affine transformation.

We relate the approximation error of Theorem 2 with the estimation error using the covering number of the hypothesis class F (CNN) .

Although there are several theorems of this type, we employ the one in Schmidt-Hieber (2017) due to its convenient form (Lemma 5).

We can prove that the logarithm of the covering number is upper bounded by M 2 log((B (conv) ∨ B (fc) )M 1 /ε) (Lemma 4) using the similar techniques to the one in Schmidt-Hieber (2017).

Theorem 2 is the immediate consequence of these two lemmas.

To prove Cororraly 1, we set M = O(N α ) for some α ≥ 0.

Then, under the assumption of the corolarry, we have DISPLAYFORM0 from Theorem 2.

The order of the right hand side with respect to N is minimized when α = DISPLAYFORM1 and b ∈ R such that 1.

DISPLAYFORM2 Proof.

First, observe that the convolutional layer constructed from u = [u 1 . . .

u K ] ∈ R K×1×1 takes the inner product with the first K elements of the input signal: DISPLAYFORM3 K×1×1 works as the "left-translation" by K − 1.

Therefore, we should define w so that it takes the inner product with the K left-most elements in the first channel and shift the input signal by K − 1 with the second channel.

Specifically, we define DISPLAYFORM4 We set b := ( 0, . . .

, 0 L0 − 1 times , t).

Then w and b satisfy the condition of the lemma.

The following lemma shows that we can convert any linear CNN to a ReLU CNN that has approximately 4 times larger parameters.

This type of lemma is also found in Petersen & Voigtlaender (2017) (Lemma 2.3).

DISPLAYFORM0 be filter sizes.

Let (l) .

Consider the linear convolution layers constructed from w and b: DISPLAYFORM1 DISPLAYFORM2 ∞ , and DISPLAYFORM3 Proof.

We definew andb as follows: DISPLAYFORM4 By definition, a pair (w,b) satisfies the conditions FORMULA24 and FORMULA27 .

For any x ∈ R D , we set y DISPLAYFORM5 for any α, β ∈ [D].

Summing them up and using the definition ofb DISPLAYFORM6 (y DISPLAYFORM7 (y DISPLAYFORM8 (y DISPLAYFORM9 −(w (l+1) ) α,:,: (y DISPLAYFORM10 for any α, β ∈ [D].

Again, by taking the summation and using the definition ofb (l+1) , we get DISPLAYFORM11 .By applying ReLU, we get DISPLAYFORM12 By using the induction hypothesis, we get DISPLAYFORM13 Therefore, the claim holds for l + 1.

By induction, the claim holds for L, which is what we want to prove.

We can concatenate two CNNs with the same depths and filter sizes in parallel.

Although it is almost trivial, we state it formally as a proposition.

In the following proposition, C (0) and C (0) is not necessarily 1.

DISPLAYFORM0 We define w and b in the same way, with the exception that C (l) is replaced with C (l) .

We definew = (w (1) , . . .

,w (L) ) and DISPLAYFORM1 .

Then, we have, DISPLAYFORM2 for any x, x ∈ R D×C (0) and any σ : R → R.Note that by the definition of · 0 and · ∞ , we have DISPLAYFORM3 and DISPLAYFORM4 .

We will construct the desired CNN consisting of M residual blocks, whose m-th residual block is made from the ingredients of the corresponding m-th block in f Lm] , and w m ).

DISPLAYFORM5 [The m = 0 Block]: We prepare a single convolutional layer with 2 output channels and 2 size-1 filters suth that the first filter works as the identity function and the second filter inserts zeros to the second channel.

Weight parameters of this convolutional layer are all zeros except single one.

We denote this block by Conv 0 .

DISPLAYFORM6 1×D is the d-th row of the matrix DISPLAYFORM7 m ×D .

We apply Lemma 1 and Lemma 2 and obtain ReLU CNNs realizing the hinge functions.

By combining them in parallel using Proposition 1, we have a learnable parameter θ DISPLAYFORM8 Since we double the channel size in the m = 0 part, the identity mapping has 2 channels.

Therefore, we made Conv ReLU θ(1) m so that it has 2 input channels and neglects the input signals coming from the second one.

This is possible by adding filters consisting of zeros appropriately.

Next, for l-th layer (l = 2, . . .

, L m ), we prepare size-1 filters w DISPLAYFORM9 where ⊗ is the Kronecker product of matrices.

Intuitively, the l = 2 layer will pick all odd indices of the output of Conv DISPLAYFORM10 .

By the inductive calculation, we have DISPLAYFORM11 By definition, Conv m has the depth of L 0 + L m − 1, at most 4D DISPLAYFORM12 DISPLAYFORM13 Then, we have DISPLAYFORM14 (the subscript 1 represents the first coordinate).[Final Fully-connected Layer] Finally, we set w := B

We assume w, w ∈ R K×J×I , b, b ∈ R, and x ∈ R D×I unless specified.

We have in mind that the activation function σ is either the ReLU function or the identity function id.

But the following proposition holds for any 1-Lipschitz function such that σ(0) = 0.

Remember that we can treat L w as a linear operator from R D×I to R D×J .

We endow R D×I and R D×J with the sup norm and denote the operator norm DISPLAYFORM0 is evaluated as follows: DISPLAYFORM1 Note that the first inequality holds because the ReLU function is 1-Lipschitz.

DISPLAYFORM2

In the following propositions in this subsection, we assume W, W ∈ R D×C , b, b ∈ R, and x ∈ R D×C .

Again, these propositions hold for any 1-Lipschitz function σ : R → R such that σ(0) = 0.

But σ = ReLU or id is enough for us.

DISPLAYFORM0 The number of non-zero summand in the summation is at most W 0 and each summand is bounded by W ∞ x ∞ Therefore, we have |FC DISPLAYFORM1 In this section, we denote the architecture of CNNs by DISPLAYFORM2 and the norm constraint on the convolution part by DISPLAYFORM3 and (conv) .

Then, for any DISPLAYFORM4 DISPLAYFORM5 Proof.

We write in shorthand as C DISPLAYFORM6 By Proposition 2 and assumptions w DISPLAYFORM7 , it is further bounded by DISPLAYFORM8 by Proposition 2 and 4) DISPLAYFORM9 by Proposition 2 and 5) DISPLAYFORM10 We denote the l-th convolution layer of the m-th block by C (l) m and the m-th residual block of by C m : Proof.

By using Proposition 9 inductively, we have DISPLAYFORM11 DISPLAYFORM12 Lemma 3.

Let ε > 0.

Suppose θ and θ are within distance ε, that is, max m,l w DISPLAYFORM13 We will bound each term of (7).

By Proposition 8 and Proposition 11, DISPLAYFORM14 On the other hand, for m = 0, . . .

, M , DISPLAYFORM15 DISPLAYFORM16 By applying (8) and FORMULA37 to FORMULA37 , we have DISPLAYFORM17

For a metric space (M 0 , d) and ε > 0, we denote the (external) covering number of DISPLAYFORM0 Proof.

The idea of the proof is same as that of Lemma 12 of Schmidt-Hieber (2017 (i.e., 2B DISPLAYFORM1 can be realized by parameters such that every pair of corresponding parameters are in a same bin, then, f − f ∞ ≤ ε by Lemma 3.

We make a subset F 0 of F (CNN) by picking up every combination of bins for M 2 parameters.

Then, for each f ∈ F (CNN) , there exists f 0 ∈ F 0 such that f − f 0 ∞ ≤ ε.

There are at most 2BM 1 ε −1 choices of bins for each parameter.

Therefore, the cardinality of F 0 is at most DISPLAYFORM2 D.2 PROOF OF THEOREM 2 AND COROLLARY 1We use the lemma in Schmidt-Hieber (2017) to bound the estimation error of the clipped ERM estimatorf .

Since our problem setting is slightly different from one in the paper, we restate the statement.

Lemma 5 (cf.

Schmidt-Hieber (2017) Lemma 10).

Let F be a family of measurable functions from [−1, 1] D to R. Letf be the clipped ERM estimator of the regression problem described in Section 3.1.

Suppose the covering number of F satisfies N (ε, F, · ∞ ) ≥ 3.

Then, Proof.

Basically, we convert our problem setting so that it fits to the assumptions of Lemma 10 of Schmidt-Hieber (2017) • (x n )).

Then, the probability that D is drawn from P ⊗N is same as the probability that D is drawn from P ⊗N where P is the joint distribution of (X , Y ).

Also, we can show thatf is the ERM estimator of the regression problem Y = f DISPLAYFORM3 • +ξ using the dataset D :f 1 ∈ arg min f ∈F R D (f ).

We apply the Lemma 10 of Schmidt-Hieber (2017) with n ← N , d ← D, ε ← 1, δ ← 1 N , ∆ n ← 0, F ← F, F ← 2F ,f ←f 1 and use the fact that the estimation error of the clipped ERM estimator is no worse than that of the ERM estimator, that is, Next, we prove the existence of a block-sparse FNN with constant-width blocks that optimally approximates a given β-Hölder function.

It is almost same as the proof of Theorem 5 of SchmidtHieber (2017) .

However, we need to construct the FNN so that it has a block-sparse structure.

FORMULA24 ) layers.

It is left for future research whether our result can extend to the ResNet-type CNNs with pooling or Batch Normalization layers.

Second, our CNN does not have ReLU activation after the junction points and the final layer of the 0-th block, while they have in the original ResNet.

We choose this design to make proofs simpler.

We can easily extend our results to the architecture that adds the ReLU activations to those points with slight modifications using similar techniques appeared in Lemma 2 of the appendix.

DISPLAYFORM4 DISPLAYFORM5

<|TLDR|>

@highlight

It is shown that ResNet-type CNNs are a universal approximator and its expression ability is not worse than fully connected neural networks (FNNs) with a \textit{block-sparse} structure even if the size of each layer in the CNN is fixed.