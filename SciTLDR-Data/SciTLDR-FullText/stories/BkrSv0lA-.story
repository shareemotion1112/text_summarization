The huge size of deep networks hinders their use in small computing devices.

In this paper, we consider compressing the network by weight quantization.

We extend a recently proposed loss-aware weight binarization scheme to ternarization, with possibly different scaling parameters for the positive and negative weights, and m-bit (where m > 2) quantization.

Experiments on feedforward and recurrent neural networks show that the proposed scheme outperforms state-of-the-art weight quantization algorithms, and is as accurate (or even more accurate) than the full-precision network.

The last decade has witnessed huge success of deep neural networks in various domains.

Examples include computer vision, speech recognition, and natural language processing BID16 .

However, their huge size often hinders deployment to small computing devices such as cell phones and the internet of things.

Many attempts have been recently made to reduce the model size.

One common approach is to prune a trained dense network BID5 BID29 .

However, most of the pruned weights may come from the fully-connected layers where computations are cheap, and the resultant time reduction is insignificant.

BID21 and Molchanov et al. (2017) proposed to prune filters in the convolutional neural networks based on their magnitudes or significance to the loss.

However, the pruned network has to be retrained, which is again expensive.

Another direction is to use more compact models.

GoogleNet (Szegedy et al., 2015) and ResNet BID7 replace the fully-connected layers with simpler global average pooling.

However, they are also deeper.

SqueezeNet BID11 reduces the model size by replacing most of the 3 × 3 filters with 1 × 1 filters.

This is less efficient on smaller networks because the dense 1 × 1 convolutions are costly.

MobileNet BID10 compresses the model using separable depth-wise convolution.

ShuffleNet (Zhang et al., 2017) utilizes pointwise group convolution and channel shuffle to reduce the computation cost while maintaining accuracy.

However, highly optimized group convolution and depth-wise convolution implementations are required.

Alternatively, Novikov et al. (2015) compressed the model by using a compact multilinear format to represent the dense weight matrix.

The CP and Tucker decompositions have also been used on the kernel tensor in CNNs BID15 BID13 .

However, they often need expensive fine-tuning.

Another effective approach to compress the network and accelerate training is by quantizing each full-precision weight to a small number of bits.

This can be further divided to two sub-categories, depending on whether pre-trained models are used BID22 BID24 or the quantized model is trained from scratch BID1 BID20 .

Some of these also directly learn with low-precision weights, but they usually suffer from severe accuracy deterioration BID20 BID26 .

By keeping the full-precision weights during learning, BID1 pioneered the BinaryConnect algorithm, which uses only one bit for each weight while still achieving state-of-the-art classification results.

Rastegari et al. (2016) further incorporated weight scaling, and obtained better results.

Instead of simply finding the closest binary approximation of the full-precision weights, a loss-aware scheme is proposed in .

Beyond binarization, TernaryConnect BID23 quantizes each weight to {−1, 0, 1}. BID19 and added scaling to the ternarized weights, and DoReFa-Net (Zhou et al., 2016) further extended quantization to more than three levels.

However, these methods do not consider the effect of quantization on the loss, and rely on heuristics in their procedures (Zhou et al., 2016; .

Recently, a loss-aware low-bit quantized neural network is proposed in BID18 .

However, it uses full-precision weights in the forward pass and the extra-gradient method (Vasilyev et al., 2010) for update, both of which are expensive.

In this paper, we propose an efficient and disciplined ternarization scheme for network compression.

Inspired by , we explicitly consider the effect of ternarization on the loss.

This is formulated as an optimization problem which is then solved efficiently by the proximal Newton algorithm.

When the loss surface's curvature is ignored, the proposed method reduces to that of BID19 , and is also related to the projection step of BID18 .

Next, we extend it to (i) allow the use of different scaling parameters for the positive and negative weights; and (ii) the use of m bits (where m > 2) for weight quantization.

Experiments on both feedforward and recurrent neural networks show that the proposed quantization scheme outperforms state-of-the-art algorithms.

Notations: For a vector x, √ x denotes the element-wise square root (i.e., [ DISPLAYFORM0 p is its p-norm, and Diag(x) returns a diagonal matrix with x on the diagonal.

For two vectors x and y, x y denotes the element-wise multiplication and x y the element-wise division.

Given a threshold ∆, I ∆ (x) returns a vector such DISPLAYFORM1

Let the full-precision weights from all L layers be w = [w 1 , w 2 , . . .

, w L ] , where w l = vec(W l ), and W l is the weight matrix at layer l. The corresponding quantized weights will be denotedŵ = [ŵ 1 ,ŵ 2 , . . .

,ŵ L ] .

In BinaryConnect BID1 , each element of w l is binarized to −1 or +1 by using the sign function: Binarize(w l ) = sign(w l ).

In the Binary-Weight-Network (BWN) (Rastegari et al., 2016) , a scaling parameter is also included, i.e., Binarize(w l ) = α l b l , where α l > 0, b l ∈ {−1, +1} n l and n l is the number of weights in w l .

By minimizing the difference between w l and α l b l , the optimal α l , b l have the simple form: α l = w l 1 /n l , and b l = sign(w l ).Instead of simply finding the best binary approximation for the full-precision weight w t l at iteration t, the loss-aware binarized network (LAB) directly minimizes the loss w.r.t.

the binarized weight α

In a weight ternarized network, zero is used as an additional quantized value.

In TernaryConnect BID23 BID19 DISPLAYFORM0 DISPLAYFORM1 However, ∆ t l in (1)

In a weight quantized network, m bits (where m ≥ 2) are used to represent each weight.

Let Q be a set of (2k + 1) quantized values, where k = 2 DISPLAYFORM0 DISPLAYFORM1 .

Similar to loss-aware binarization , BID18 proposed a loss-aware quantized network called low-bit neural network (LBNN).

The alternating direction method of multipliers (ADMM) BID0 ) is used for optimization.

At the tth iteration, the full-precision weight w t l is first updated by the method of extra-gradient (Vasilyev et al., 2010) : DISPLAYFORM2 In weight ternarization, TWN simply finds the closest ternary approximation of the full precision weight at each iteration, while TTQ sets the ternarization threshold heuristically.

Inspired by LAB (for binarization), we consider the loss explicitly during quantization and obtain the quantization thresholds and scaling parameter by solving an optimization problem.

As in TWN, the weight w l is ternarized asŵ l = α l b l , where α l > 0 and b l ∈ {−1, 0, 1} n l .

Given a loss function , we formulate weight ternarization as the following optimization problem: DISPLAYFORM3 where Q is the set of desired quantized values.

As in LAB, we will solve this using the proximal Newton method BID17 Rakotomamonjy et al., 2016) .

At iteration t, the objective is replaced by the second-order expansion DISPLAYFORM4 where H t−1 is an estimate of the Hessian of atŵ t−1 .

We use the diagonal equilibration preconditioner BID2 , which is robust in the presence of saddle points and also readily available in popular stochastic deep network optimizers such as Adam BID14 .

Let D l be the approximate diagonal Hessian at layer l. We use DISPLAYFORM5 as an estimate of H. Substituting (4) into FORMULA7 , we solve the following subproblem at the tth iteration: DISPLAYFORM6 DISPLAYFORM7 Proposition 3.1 The objective in (5) can be rewritten as DISPLAYFORM8 DISPLAYFORM9 ), and DISPLAYFORM10 Obviously, this objective can be minimized layer by layer.

Each proximal Newton iteration thus consists of two steps: (i) Obtain w t l in FORMULA14 by gradient descent along ∇ l (ŵ t−1 ), which is preconditioned by the adaptive learning rate 1 d ]

i is small), the loss is not sensitive to the weight and ternarization error can be less penalized.

When the loss surface is steep, ternarization has to be more accurate.

Though the constraint in (6) is more complicated than that in LAB, interestingly the following simple relationship can still be obtained for weight ternarization. ; whereas when α is fixed, DISPLAYFORM11 Equivalently, b can be written as Π Q (w t l /α), where Π Q (·) projects each entry of the input argument to the nearest element in Q. Further discussions on how to solve for α t l will be presented in Sections 3.1.1 and 3.1.2.

When the curvature is the same for all dimensions at layer l, the following Corollary shows that the solution above reduces that of TWN.

DISPLAYFORM12 In other words, TWN corresponds to using the proximal gradient algorithm, while the proposed method corresponds to using the proximal Newton algorithm with diagonal Hessian.

In composite optimization, it is known that the proximal Newton algorithm is more efficient than the proximal gradient algorithm BID17 Rakotomamonjy et al., 2016) .

Moreover, note that the interesting relationship ∆ t l = α t l /2 is not observed in TWN, while TTQ completely neglects this relationship.

In LBNN BID18 , its projection step uses an objective which is similar to (6), but without using the curvature information.

Besides, their w t l is updated with the extra-gradient in BID29 , which doubles the number of forward, backward and update steps, and can be costly.

Moreover, LBNN uses full-precision weights in the forward pass, while all other quantization methods including ours use quantized weights (which eliminates most of the multiplications and thus faster training).When (i) is continuously differentiable with Lipschitz-continuous gradient (i.e., there exists β > 0 such that ∇ (u) − ∇ (v) 2 ≤ β u − v 2 for any u, v); (ii) is bounded from below; and (iii) [d t l ] k > β ∀l, k, t, it can be shown that the objective of (3) produced by the proximal Newton algorithm (with solution in Proposition 3.2) converges .

In practice, it is important to keep the full-precision weights during update BID1 .

Hence, we replace (7) by w DISPLAYFORM13 .

The whole procedure, which is called Loss-Aware Ternarization (LAT), is shown in Algorithm 3 of Appendix B. It is similar to Algorithm 1 of LAB , except that α t l and b t l are computed differently.

In step 4, following BID19 , we first rescale input x t−1 l with α l , so that multiplications in dot products and convolutions become additions.

Algorithm 3 can also be easily extended to ternarize weights in recurrent networks.

Interested readers are referred to for details.

To simplify notations, we drop the superscripts and subscripts.

From Proposition 3.2, DISPLAYFORM0 We now consider how to solve for α.

First, we introduce some notations.

Given a vector x = [x 1 , x 2 , . . .

, x n ], and an indexing vector s ∈ R n whose entries are a permutation of {1, . . .

, n}, perm s (x) returns the vector [x s1 , x s2 , . . .

We sort elements of |w| in descending order, and let the vector containing the sorted indices be s. DISPLAYFORM1 where c = cum(perm s (|d w|)) cum(perm s (d)) 2, and j is the index such that DISPLAYFORM2 For simplicity of notations, let the dimensionality of w (and thus also of c) be n, and the operation find(condition(x)) returns all indices in x that satisfies the condition.

It is easy to see that any j satisfying FORMULA20 DISPLAYFORM3 , where c [1:(n−1)] is the subvector of c with elements in the index range 1 to n − 1.

The optimal α (= 2c j ) is then the one which yields the smallest objective in (6), which can be simplified by Proposition 3.3 below.

The procedure is shown in Algorithm 1.

The optimal α t l of (6) equals 2 arg max cj :j∈S c DISPLAYFORM0 Algorithm 1 Exact solver of (6) DISPLAYFORM1

In case the sorting operation in step 2 is expensive, α

As in TTQ (Zhu et al., 2017), we can use different scaling parameters for the positive and negative weights in each layer.

The optimization subproblem at the tth iteration then becomes: DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 , and q DISPLAYFORM3 The exact and approximate solutions for α t l and β t l can be obtained in a similar way as in Sections 3.1.1 and 3.1.2.

Details are in Appendix C.

For m-bit quantization, we simply change the set Q of desired quantized values in FORMULA7 to one with k = 2 m−1 − 1 quantized values.

The optimization still contains a gradient descent step with adaptive learning rates like LAT, and a quantization step which can be solved efficiently by alternating minimization of (α, b) (similar to the procedure in Algorithm 2) using the following Proposition.

In this section, we perform experiments on both feedforward and recurrent neural networks.

The following methods are compared: (i) the original full-precision network; (ii) weight-binarized networks, including BinaryConnect BID1 , Binary-Weight-Network (BWN) (Rastegari et al., 2016) , and Loss-Aware Binarized network (LAB) ; (iii) weightternarized networks, including Ternary Weight Networks (TWN) BID19 , Trained Ternary Quantization (TTQ) 2 (Zhu et al., 2017) , the proposed Loss-Aware Ternarized network with exact solution (LATe), approximate solution (LATa), and with two scaling parameters (LAT2e and LAT2a); (iv) m-bit-quantized networks (where m > 2), including DoReFa-Netm (Zhou et al., 2016) , the proposed loss-aware quantized network with linear quantization (LAQm(linear)), and logarithmic quantization (LAQm(log)).

Since weight quantization can be viewed as a form of regularization BID1 , we do not use other regularizers such as dropout and weight decay.

In this section, we perform experiments with the multilayer perceptron (on the MNIST data set) and convolutional neural networks (on CIFAR-10, CIFAR-100 and SVHN).

For MNIST, CIFAR-10, and SVHN, the setup is similar to that in BID1 BID30 .

Details can be found in Appendix D. For CIFAR-100, we use 45, 000 images for training, another 5, 000 for validation, and the remaining 10, 000 for testing.

The testing errors are shown in TAB1 .Ternarization: On MNIST, CIFAR100 and SVHN, the weight-ternarized networks perform better than weight-binarized networks, and are comparable to the full-precision networks.

Among the weight-ternarized networks, the proposed LAT and its variants have the lowest errors.

On CIFAR-10, LATa has similar performance as the full-precision network, but is outperformed by BinaryConnect.

FIG4 shows convergence of the training loss for LATa on CIFAR-10, and FIG4 shows the scaling parameter obtained at each CNN layer.

As can be seen, the scaling parameters for the first and last layers (conv1 and linear3, respectively) are larger than the others.

This agrees with the finding that, to maintain the activation variance and back-propagated gradients variance during the forward and backward propagations, the variance of the weights between the lth and (l + 1)th layers should roughly follow 2/(n l +n l+1 ) BID3 .

Hence, as the input and output layers are small, larger scaling parameters are needed for their high-variance weights.

Using Two Scaling Parameters: Compared to TTQ, the proposed LAT2 always has better performance.

However, the extra flexibility of using two scaling parameters does not always translate to lower testing error.

As can be seen, it outperforms algorithms with one scaling parameter only on CIFAR-100.

We speculate this is because the capacities of deep networks are often larger than needed, and so the limited expressiveness of quantized weights may not significantly deteriorate performance.

Indeed, as pointed out in BID1 , weight quantization is a form of regularization, and can contribute positively to the performance.

Using More Bits:

Among the 3-bit quantization algorithms, the proposed scheme with logarithmic quantization has the best performance.

It also outperforms the other quantization algorithms on CIFAR-100 and SVHN.

However, as discussed above, more quantization flexibility is useful only when the weight-quantized network does not have enough capacity.

In this section, we follow and perform character-level language modeling experiments on the long short-term memory (LSTM) BID8 .

The training objective is the cross-entropy loss over all target sequences.

Experiments are performed on three data sets: (i) Leo Tolstoy's War and Peace; (ii) source code of the Linux Kernel; and (iii) Penn Treebank Corpus BID31 .

For the first two, we follow the setting in BID30 .

For Penn Treebank, we follow the setting in BID25 .

In the experiment, we tried different initializations for TTQ and then report the best.

Cross-entropy values on the test set are shown in TAB3 .

Ternarization: As in Section 4.1, the proposed LATe and LATa outperform the other weight ternarization schemes, and are even better than the full-precision network on all three data sets.

FIG6 shows convergence of the training and validation losses on War and Peace.

Among the ternarization methods, LAT and its variants converge faster than both TWN and TTQ.

Using Two Scaling Parameters: LAT2e and LAT2a outperform TTQ on all three data sets.

They also perform better than using one scaling parameter on War and Peace and Penn Treebank.

Using More Bits: The proposed LAQ always outperforms DoReFa-Net when 3 or 4 bits are used.

As noted in Section 4.1, using more bits does not necessarily yield better generalization performance, and ternarization (using 2 bits) yields the lowest validation loss on War and Peace and Linux Kernel.

Moreover, logarithmic quantization is better than linear quantization.

FIG18 shows distributions of the input-to-hidden (full-precision and quantized) weights of the input gate trained after 20 epochs using LAQ3(linear) and LAQ3(log) (results on the other weights are similar).

As can be seen, distributions of the full-precision weights are bell-shaped.

Hence, logarithmic quantization can give finer resolutions to many of the weights which have small magnitudes.

Quantized vs Full-precision Networks: The quantized networks often perform better than the fullprecision networks.

We speculate that this is because deep networks often have larger-than-needed capacities, and so are less affected by the limited expressiveness of quantized weights.

Moreover, low-bit quantization acts as regularization, and so contributes positively to the performance.

In this paper, we proposed a loss-aware weight quantization algorithm that directly considers the effect of quantization on the loss.

The problem is solved using the proximal Newton algorithm.

Each iteration consists of a preconditioned gradient descent step and a quantization step that projects fullprecision weights onto a set of quantized values.

For ternarization, an exact solution and an efficient approximate solution are provided.

The procedure is also extended to the use of different scaling parameters for the positive and negative weights, and to m-bit (where m > 2) quantization.

Experiments on both feedforward and recurrent networks show that the proposed quantization scheme outperforms the current state-of-the-art.

@highlight

A loss-aware weight quantization algorithm that directly considers its effect on the loss is proposed.

@highlight

Proposes a method of compressing network by means of weight ternarization. 

@highlight

The paper proposes a new method to train DNNs with quantized weights, by including the quantization as a constraint in a proximal quasi-Newton algorithm, which simultaneously learns a scaling for the quantized values.

@highlight

The paper extends the loss-aware weight binarization scheme to terarization and arbitrary m-bit quantization and demonstrate its promising performance.