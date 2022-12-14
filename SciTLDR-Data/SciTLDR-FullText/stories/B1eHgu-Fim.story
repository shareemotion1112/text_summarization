Modern deep neural networks have a large amount of weights, which make them difficult to deploy on computation constrained devices such as mobile phones.

One common approach to reduce the model size and computational cost is to use low-rank factorization to approximate a weight matrix.

However, performing standard low-rank factorization with a small rank can hurt the model expressiveness and significantly decrease the performance.

In this work, we propose to use a mixture of multiple low-rank factorizations to model a large weight matrix, and the mixture coefficients are computed dynamically depending on its input.

We demonstrate the effectiveness of the proposed approach on both language modeling and image classification tasks.

Experiments show that our method not only improves the computation efficiency but also maintains (sometimes outperforms) its accuracy compared with the full-rank counterparts.

Modern neural networks usually contain millions of parameters BID4 BID8 , and they are difficult to be deployed on mobile devices with limited computation resources.

To solve this problem, model compression techniques are proposed in recent years.

Low-rank factorization is a popular way of reducing the matrix size.

It has been extensively explored in the literature BID5 BID6 BID3 BID10 .

Mathematically, a large weight matrix W ∈ R m×n is factorized to two small rank-d matrices U ∈ R m×d , V ∈ R n×d with W = U V T .

Since both U and V are dense, no sparsity support is required from specialized hardware.

It naturally fits the general-purpose, off-the-shelf CPUs and GPUs.

To significantly reduce the model size and computation, the rank d in the low-rank factorization needs to be small.

However, a small rank can limit the expressiveness of the model BID9 and lead to worse performance.

To understand the limitations, given a n-dim feature vector h, we observe that DISPLAYFORM0 , is a linear projection from a high-dimensional space (n dims) to a low-dimensional space (d dims).

This can lead to a significant loss of information.

The conflict between the rank d and the model expressiveness prevents us from obtaining a both compact and accurate model.

To address the dilemma, we propose to increase the expressiveness by learning an adaptive, inputdependent factorization, rather than performing a fixed factorization of a weight matrix.

To do so, we use a mixture of multiple low-rank factorizations.

The mixing weights are computed based on the input.

This creates an adaptive linear projection from a high-dimensional space to a low-dimensional space.

Compared to the conventional low-rank factorization, the proposed approach can significantly improve its performance while only introducing a small additional cost.

DISPLAYFORM1 where z can be treated as the middle layer.

Techniques like pooling can be applied to compute π to make it efficient.

We propose to use an unnormalized learned mixture of low-rank factorizations whose mixing weights are computed adaptively based on the input.

More specifically, denoting the input by h and the number of mixture components by K, we decompose a large weight matrix by DISPLAYFORM0 where π(·) : R n → R K is the function which maps each input to its mixture coefficients.

For example, π can be a small neural network.

This introduces a small amount of extra parameters and computation.

We will later discuss the details of efficient ways to implement the mixture function π.

If π k , k = 1, ..., K, is chosen to be constant (input independent), it can be absorbed into either DISPLAYFORM1 .

Thus, the proposed method reduces to the low-rank factorization.

This is evidenced by rewriting DISPLAYFORM2 .

In other words, the conventional low-rank factorization can be considered as a special case of our method.

FIG0 depicts the proposed framework.

Adaptive mixing weights π(h).

The mixing weights can encode important information that we can use to increase the expressiveness of the projected low-dimensional space.

Under our framework, the generation of the mixing weights π(h) is flexible.

A straight-forward approach is to use a non-linear transformation of the input to the weight matrix.

For example, π(h) = σ(P h), where σ is a non-linear transformation, such as sigmoid or hyperbolic tangent function, and P ∈ R K×n is an extra weight matrix.

This adds some extra parameters and computation to the model since the linear projection that we construct is R n → R K .

To further reduce the parameter and computation in the mixing weights π, we propose the following strategies.

Pooling before projection.

We do not require the whole input to compute the mixture function π.

Instead, we can apply pooling to the input h before projection.

For example, a global average pooling can be applied if the input is a 3D tensor (for images); for a 1D vector, we can segment the vector and average each segmentations.

By applying pooling, we can both save the computation and better capture the global information.

Random projection.

To reduce the number of parameters in the linear projection of h, we can use a random matrix P random in place of a fully adjustable P , i.e. π(h) = σ(P random h).

Note that we can simply save a seed to recover the random matrix, but it still requires the same amount of memory and computation as the fully adjustable linear projection of h.

Increased expressiveness.

The adaptive mixing weights introduce a non-linear transformation into the high-to-low-dimensional projection that can be more expressive.

Since each W (h) is a data-dependent low-rank matrix, there is no constant linear weight independent to the input (even a full-rank matrix) that can mimic the transformation W (h) .

Generating the whole weight matrices can be very expensive.

Our method can be seen as a swift approach to generate the weights by adaptively adjusting mixing weights for the linear bottleneck.

It assigns weights into groups and dynamically controls them at the group level.

Recurrent neural networks for language modeling.

Recurrent neural networks (RNNs) are widely used in language modeling, machine translation and sequence modeling in general.

We adopt the same Long Short Term Memory (LSTM) models and follow the settings from a previous state-ofthe-art model BID11 for language modeling, and use Penn Tree Bank (PTB) as well as Text8 datasets.

More specifically, we use the medium sized model introduced in BID11 .We test three variants of the proposed model against regular low-rank factorization, each with different ways of computing mixing weights, namely (1) MIX-ALL-PJ: direct linear projection of the input vector h, (2) MIX-POOL-PJ: linear projection after segment-based mean pooling of the input vector h, and (3) MIX-RND-PJ: use a random projection for the input vector h. Among these adaptive projection methods, MIX-ALL-PJ has a large amount of extra parameters, MIX-POOL-PJ has a small amount of extra parameters, and MIX-RND-PJ has no extra parameters.

We compute the FLOPs of a single time-step of applying LSTM, and the perplexity associated to different settings.

The results are shown in FIG1 .

Firstly, with adaptive mixtures, the low-rank factorization model achieved 40% reduction in FLOPs, and even surpassed the performance of the full matrix baseline by decreasing the perplexity by 1.7 points.

Secondly, the use of adaptive mixtures can significantly improve the performance compared with regular, non-adaptive low-rank factorization.

Thirdly, using pooling before projection can be a good choice for computing the mixing weights π.

It not only reduces the computation and parameter size, but can better capture the global information and achieve better accuracy.

CNN for image recognition.

We further demonstrate the effectiveness of the proposed approach on compressing CNN models on ImageNet BID1 .

We chose to use modern compact CNN models as the baseline (which are harder to compress), rather than using the bulky CNN models (which is easier to compress).

Specifically, we choose to compress the point-wise convolution in depth-wise separable convolutions BID0 , MobileNet BID2 BID7 in particular.

TAB0 shows the comparison of different state-of-art compact convolutional models.

We observed that compared to the regular low-rank factorization of MobileNet model (a.k.a.

MobileNet V2), the proposed method achieves significantly better results (2.5% and 2% for two different Low-rank MobileNet settings, respectively), while only adding negligible extra FLOPs (less than 1%).

@highlight

A simple modification to low-rank factorization that improves performances (in both image and language tasks) while still being compact.