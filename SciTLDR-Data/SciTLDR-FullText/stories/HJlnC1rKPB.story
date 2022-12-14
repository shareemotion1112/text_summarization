Recent trends of incorporating attention mechanisms in vision have led researchers to reconsider the supremacy of convolutional layers as a primary building block.

Beyond helping CNNs to handle long-range dependencies, Ramachandran et al. (2019) showed that attention can completely replace convolution and achieve state-of-the-art performance on vision tasks.

This raises the question: do learned attention layers operate similarly to convolutional layers?

This work provides evidence that attention layers can perform convolution and, indeed, they often learn to do so in practice.

Specifically, we prove that a multi-head self-attention layer with sufficient number of heads is at least as expressive as any convolutional layer.

Our numerical experiments then show that self-attention layers attend to pixel-grid patterns similarly to CNN layers, corroborating our analysis.

Our code is publicly available.

Recent advances in Natural Language Processing (NLP) are largely attributed to the rise of the transformer (Vaswani et al., 2017) .

Pre-trained to solve an unsupervised task on large corpora of text, transformer-based architectures, such as GPT-2 (Radford et al., 2018) , BERT (Devlin et al., 2018) and Transformer-XL , seem to possess the capacity to learn the underlying structure of text and, as a consequence, to learn representations that generalize across tasks.

The key difference between transformers and previous methods, such as recurrent neural networks (Hochreiter & Schmidhuber, 1997) and convolutional neural networks (CNN), is that the former can simultaneously attend to every word of their input sequence.

This is made possible thanks to the attention mechanism-originally introduced in Neural Machine Translation to better handle long-range dependencies (Bahdanau et al., 2015) .

With self-attention in particular, the similarity of two words in a sequence is captured by an attention score measuring the distance of their representations.

The representation of each word is then updated based on those words whose attention score is highest.

Inspired by its capacity to learn meaningful inter-dependencies between words, researchers have recently considered utilizing self-attention in vision tasks.

Self-attention was first added to CNN by either using channel-based attention (Hu et al., 2018) or non-local relationships across the image (Wang et al., 2018) .

More recently, augmented CNNs by replacing some convolutional layers with self-attention layers, leading to improvements on image classification and object detection tasks.

Interestingly, Ramachandran et al. (2019) noticed that, even though state-of-the art results are reached when attention and convolutional features are combined, under same computation and model size constraints, self-attention-only architectures also reach competitive image classification accuracy.

These findings raise the question, do self-attention layers process images in a similar manner to convolutional layers?

From a theoretical perspective, one could argue that transfomers have the capacity to simulate any function-including a CNN.

Indeed, P??rez et al. (2019) showed that a multilayer attention-based architecture with additive positional encodings is Turing complete under some strong theoretical assumptions, such as unbounded precision arithmetic.

Unfortunately, universality results do not reveal how a machine solves a task, only that it has the capacity to do so.

Thus, the question of how self-attention layers actually process images remains open.

We here recall the mathematical formulation of self-attention layers and emphasize the role of positional encodings.

Let X ??? R T ??Din be an input matrix consisting of T tokens in of D in dimensions each.

While in NLP each token corresponds to a word in a sentence, the same formalism can be applied to any sequence of T discrete objects, e.g. pixels.

A self-attention layer maps any query token t ??? [T ] from D in to D out dimensions as follows:

Self-Attention(X) t,: := softmax (A t,: ) XW val ,

where we refer to the elements of the T ?? T matrix A := XW qry W key X

as attention scores and the softmax output 3 as attention probabilities.

The layer is parametrized by a query matrix W qry ??? R Din??D k , a key matrix W key ??? R Din??D k and a value matrix W val ??? R Din??Dout .For simplicity, we exclude any residual connections, batch normalization and constant factors.

A key property of the self-attention model described above is that it is equivariant to reordering, that is, it gives the same output independently of how the T input tokens are shuffled.

This is problematic for cases we expect the order of things to matter.

To alleviate the limitation, a positional encoding is learned for each token in the sequence (or pixel in an image), and added to the representation of the token itself before applying self-attention

where P ??? R T ??Din contains the embedding vectors for each position.

More generally, P may be substituted by any function that returns a vector representation of the position.

It has been found beneficial in practice to replicate this self-attention mechanism into multiple heads, each being able to focus on different parts of the input by using different query, key and value matrices.

In multi-head self-attention, the output of the N h heads of output dimension D h are concatenated and projected to dimension D out as follows:

and two new parameters are introduced: the projection matrix W out ??? R N h D h ??Dout and a bias term b out ??? R Dout .

Convolutional layers are the de facto choice for building neural networks that operate on images.

We recall that, given an image tensor X ??? R W ??H??Din of width W , height H and D in channels, the output of a convolutional layer for pixel (i, j) is given by

where W is the K ?? K ?? D in ?? D out weight tensor 4 , b ??? R Dout is the bias vector and the set

contains all possible shifts appearing when convolving the image with a K ?? K kernel.

In the following, we review how self-attention can be adapted from 1D sequences to images.

With images, rather than tokens, we have query and key pixels q, k

.

Accordingly, the input is a tensor X of dimension W ?? H ?? D in and each attention score associates a query and a key pixel.

To keep the formulas consistent with the 1D case, we abuse notation and slice tensors by using a 2D index vector: if p = (i, j), we write X p,: and A p,: to mean X i,j,: and A i,j,:,: , respectively.

With this notation in place, the multi-head self attention layer output at pixel q can be expressed as follows:

and accordingly for the multi-head case.

There are two types of positional encoding that has been used in transformer-based architectures: the absolute and relative encoding (see also Table 3 in the Appendix).

With absolute encodings, a (fixed or learned) vector P p,: is assigned to each pixel p.

The computation of the attention scores we saw in eq. (2) can then be decomposed as follows:

A abs q,k = (X q,: + P q,: )W qry W key (X k,: + P k,: ) = X q,: W qry W key X k,: + X q,: W qry W key P k,: + P q,: W qry W key X k,: + P q,: W qry W key P k,:

where q and k correspond to the query and key pixels, respectively.

The relative positional encoding was introduced by .

The main idea is to only consider the position difference between the query pixel (pixel we compute the representation of) and the key pixel (pixel we attend) instead of the absolute position of the key pixel:

In this manner, the attention scores only depend on the shift ?? := k ??? q. Above, the learnable vectors u and v are unique for each head, whereas for every shift ?? the relative positional encoding r ?? ??? R Dp is shared by all layers and heads.

Moreover, now the key weights are split into two types: W key pertain to the input and W key to the relative position of pixels.

This section derives sufficient conditions such that a multi-head self-attention layer can simulate a convolutional layer.

Our main result is the following: Theorem 1.

A multi-head self-attention layer with N h heads of dimension D h , output dimension D out and a relative positional encoding of dimension D p ??? 3 can express any convolutional layer of kernel size

The theorem is proven constructively by selecting the parameters of the multi-head self-attention layer so that the latter acts like a convolutional layer.

In the proposed construction, the attention scores of each self-attention head should attend to a different relative shift within the set ??? ??? K = {??? K/2 , . . .

, K/2 } 2 of all pixel shifts in a K ?? K kernel.

The exact condition can be found in the statement of Lemma 1.

Then, Lemma 2 shows that the aforementioned condition is satisfied for the relative positional encoding that we refer to as the quadratic encoding:

The learned parameters

2 ) and ?? (h) determine the center and width of attention of each head, respectively.

On the other hand, ?? = (?? 1 , ?? 2 ) is fixed and expresses the relative shift between query and key pixels.

It is important to stress that the above encoding is not the only one for which the conditions of Lemma 1 are satisfied.

In fact, in our experiments, the relative encoding learned by the neural network also matched the conditions of the lemma (despite being different from the quadratic encoding).

Nevertheless, the encoding defined above is very efficient in terms of size, as only D p = 3 dimensions suffice to encode the relative position of pixels, while also reaching similar or better empirical performance (than the learned one).

The theorem covers the general convolution operator as defined in eq. (17).

However, machine learning practitioners using differential programming frameworks (Paszke et al., 2017; Abadi et al., 2015) might question if the theorem holds for all hyper-parameters of 2D convolutional layers:

??? Padding: a multi-head self-attention layer uses by default the "SAME" padding while a convolutional layer would decrease the image size by K ??? 1 pixels.

The correct way to alleviate these boundary effects is to pad the input image with K/2 zeros on each side.

In this case, the cropped output of a MHSA and a convolutional layer are the same.

??? Stride: a strided convolution can be seen as a convolution followed by a fixed pooling operation-with computational optimizations.

Theorem 1 is defined for stride 1, but a fixed pooling layer could be appended to the Self-Attention layer to simulate any stride.

??? Dilation: a multi-head self-attention layer can express any dilated convolution as each head can attend a value at any pixel shift and form a (dilated) grid pattern.

Remark for the 1D case.

Convolutional layers acting on sequences are commonly used in the literature for text (Kim, 2014) , as well as audio (van den Oord et al., 2016) and time series (Franceschi et al., 2019) .

Theorem 1 can be straightforwardly extended to show that multi-head self-attention with N h heads can also simulate a 1D convolutional layer with a kernel of size K = N h with min(D h , D out ) output channels using a positional encoding of dimension D p ??? 2.

Since we have not tested empirically if the preceding construction matches the behavior of 1D self-attention in practice, we cannot claim that it actually learns to convolve an input sequence-only that it has the capacity to do so.

The proof follows directly from Lemmas 1 and 2 stated below:

be a bijective mapping of heads onto shifts.

Further, suppose that for every head the following holds:

Then, for any convolutional layer with a K ?? K kernel and D out output channels, there exists {W

Attention maps for pixel val .

We show attention maps computed for a query pixel at position q.

Proof.

Our first step will be to rework the expression of the Multi-Head Self-Attention operator from equation (1) and equation (4) such that the effect of the multiple heads becomes more transparent:

Note that each head's value matrix W (h) val ??? R Din??D h and each block of the projection matrix W out of dimension D h ?? D out are learned.

Assuming that D h ??? D out , we can replace each pair of matrices by a learned matrix W (h) for each head.

We consider one output pixel of the multi-head self-attention:

Due to the conditions of the Lemma, for the h-th attention head the attention probability is one when k = q ??? f (h) and zero otherwise.

The layer's output at pixel q is thus equal to

For K = ??? N h , the above can be seen to be equivalent to a convolutional layer expressed in eq. 17: there is a one to one mapping (implied by map f ) between the matrices

2 .

Remark about D h and D out .

It is frequent in transformer-based architectures to set

In that case, W (h) can be seen to be of rank D out ??? D h , which does not suffice to express every convolutional layer with D out channels.

Nevertheless, it can be seen that any D h out of D out outputs of MHSA(X) can express the output of any convolutional layer with D h output channels.

To cover both cases, in the statement of the main theorem we assert that the output channels of the convolutional layer should be min(D h , D out ).

In practice, we advise to concatenate heads of dimension D h = D out instead of splitting the D out dimensions among heads to have exact re-parametrization and no "unused" channels.

Lemma 2.

There exists a relative encoding scheme {r ?? ??? R Dp } ?????Z 2 with D p ??? 3 and parame-

Proof.

We show by construction the existence of a D p = 3 dimensional relative encoding scheme yielding the required attention probabilities.

As the attention probabilities are independent of the input tensor X, we set W key = W qry = 0 which leaves only the last term of eq. (8).

Setting W key ??? R D k ??Dp to the identity matrix (with appropriate row padding), yields A q,k = v r ?? where ?? := k ??? q. Above, we have assumed that D p ??? D k such that no information from r ?? is lost.

Now, suppose that we could write:

for some constant c. In the above expression, the maximum attention score over A q,: is ?????c and it is reached for A q,k with ?? = ???. On the other hand, the ?? coefficient can be used to scale arbitrarily the difference between A q,??? and the other attention scores.

In this way, for ?? = ???, we have

and for ?? = ???, the equation becomes lim ???????? softmax(A q,: ) k = 0, exactly as needed to satisfy the lemma statement.

What remains is to prove that there exist v and {r ?? } ?????Z 2 for which eq. (14) holds.

Expanding the RHS of the equation, we have

which matches eq. (14) with c = ??? ??? 2 and the proof is concluded.

Remark on the magnitude of ??.

The exact representation of one pixel requires ?? (or the matrices W qry and W key ) to be arbitrary large, despite the fact that the attention probabilities of all other pixels converge exponentially to 0 as ?? grows.

Nevertheless, practical implementations always rely on finite precision arithmetic for which a constant ?? suffices to satisfy our construction.

For instance, since the smallest positive float32 scalar is approximately 10 ???45 , setting ?? = 46 would suffice to obtain hard attention.

The aim of this section is to validate the applicability of our theoretical results-which state that self-attention can perform convolution-and to examine whether self-attention layers in practice do actually learn to operate like convolutional layers when trained on standard image classification tasks.

In particular, we study the relationship between self-attention and convolution with quadratic and learned relative positional encodings.

We find that, for both cases, the attention probabilities learned tend to respect the conditions of Lemma 1, supporting our hypothesis.

We study a fully attentional model consisting of six multi-head self-attention layers.

As it has already been shown by that combining attention features with convolutional features improves performance on Cifar-100 and ImageNet, we do not focus on attaining state-of-the-art performance.

Nevertheless, to validate that our model learns a meaningful classifier, we compare it to the standard ResNet18 (He et al., 2015) on the CIFAR-10 dataset (Krizhevsky et al.) .

In all experiments, we use a 2 ?? 2 invertible down-sampling (Jacobsen et al., 2018) on the input to reduce the size of the image.

As the size of the attention coefficient tensors (stored during forward) scales quadratically with the size of the input image, full attention cannot be applied to bigger images.

The fixed size representation of the input image is computed as the average pooling of the last layer representations and given to a linear classifier.

We used the PyTorch library (Paszke et al., 2017) and based our implementation on PyTorch Transformers 5 .

We release our code on Github 6 and hyper-parameters are listed in Table 2 (Appendix).

Remark on accuracy.

To verify that our self-attention models perform reasonably well, we display in Figure 6 the evolution of the test accuracy on CIFAR-10 over the 300 epochs of training for our self-attention models against a small ResNet (Table 1 ).

The ResNet is faster to converge, but we cannot ascertain whether this corresponds to an inherent property of the architecture or an artifact of the adopted optimization procedures.

Our implementation could be optimized to exploit the locality of Gaussian attention probabilities and reduce significantly the number of FLOPS.

We observed that learned embeddings with content-based attention were harder to train probably due to their increased number of parameters.

We believe that the performance gap can be bridged to match the ResNet performance, but this is not the focus of this work.

As a first step, we aim to verify that, with the relative position encoding introduced in equation (9), attention layers learn to behave like convolutional layers.

We train nine attention heads at each layer to be on par with the 3 ?? 3 kernels used predominantly by the ResNet architecture.

The center of attention of each head h is initialized to ??? (h) ??? N (0, 2I 2 ).

Figure 3 shows how the initial positions of the heads (different colors) at layer 4 changed during training.

We can see that after optimization, the heads attend on specific pixel of the image forming a grid around the query pixel.

Our intuition that Self-Attention applied to images learns convolutional filters around the queried pixel is confirmed.

Figure 4 displays all attention head at each layer of the model at the end of the training.

It can be seen that in the first few layers the heads tend to focus on local patterns (layers 1 and 2), while deeper layers (layers 3-6) also attend to larger patterns by positioning the center of attention further from the queried pixel position.

We also include in the Appendix a plot of the attention positions for a higher number of heads (N h = 16).

Figure 14 displays both local patterns similar to CNN and long range dependencies.

Interestingly, attention heads do not overlap and seem to take an arrangement maximizing the coverage of the input space.

Figure 4 : Centers of attention of each attention head (different colors) for the 6 self-attention layers using quadratic positional encoding.

The central black square is the query pixel, whereas solid and dotted circles represent the 50% and 90% percentiles of each Gaussian, respectively.

We move on to study the positional encoding used in practice by fully-attentional models on images.

We implemented the 2D relative positional encoding scheme used by (Ramachandran et al., 2019; : we learn a D p /2 position encoding vector for each row and each column pixel shift.

Hence, the relative positional encoding of a key pixel at position k with a query pixel at position q is the concatenation of the row shift embedding ?? 1 and the column shift embedding ?? 2 (where ?? = k ??? q).

We chose D p = D out = 400 in the experiment.

We differ from their (unpublished) implementation in the following points: (i) we do not use convolution stem and ResNet bottlenecks for downsampling, but only a 2 ?? 2 invertible downsampling layer (Jacobsen et al., 2018)

At first, we discard the input data and compute the attention scores solely as the last term of eq. (8).

The attention probabilities of each head at each layer are displayed on Figure 5 .

The figure confirms our hypothesis for the first two layers and partially for the third: even when left to learn the positional encoding scheme from randomly initialized vectors, certain self-attention heads (depicted on the left) learn to attend to individual pixels, closely matching the condition of Lemma 1 and thus Theorem 1.

At the same time, other heads pay attention to horizontally-symmetric but non-localized patterns, as well as to long-range pixel inter-dependencies.

We move on to a more realistic setting where the attention scores are computed using both positional and content-based attention (i.e., q k + q r in (Ramachandran et al., 2019) ) which corresponds to a full-blown standalone self-attention model.

The attention probabilities of each head at each layer are displayed in Figure 6 .

We average the attention probabilities over a batch of 100 test images to outline the focus of each head and remove the dependency on the input image.

Our hypothesis is confirmed for some heads of layer 2 and 3: even when left to learn the encoding from the data, certain self-attention heads only exploit positionbased attention to attend to distinct pixels at a fixed shift from the query pixel reproducing the receptive field of a convolutional kernel.

Other heads use more content-based attention (see Figures 8 to 10 in Appendix for non-averaged probabilities) leveraging the advantage of Self-Attention over CNN which does not contradict our theory.

In practice, it was shown by that combining CNN and self-attention features outperforms each taken separately.

Our experiments shows that such combination is learned when optimizing an unconstrained fully-attentional model.

The similarity between convolution and multi-head self-attention is striking when the query pixel is slid over the image: the localized attention patterns visible in Figure 6 follow the query pixel.

This characteristic behavior materializes when comparing Figure 6 with the attention probabilities at a different query pixel (see Figure 7 in Appendix).

Attention patterns in layers 2 and 3 are not only localized but stand at a constant shift from the query pixel, similarly to convolving the receptive field of a convolutional kernel over an image.

This phenomenon is made evident on our interactive website 7 .

This tool is designed to explore different components of attention for diverse images with or without content-based attention.

We believe that it is a useful instrument to further understand how MHSA learns to process images.

Figure 6: Attention probabilities for a model with 6 layers (rows) and 9 heads (columns) using learned relative positional encoding and content-content based attention.

Attention maps are averaged over 100 test images to display head behavior and remove the dependence on the input content.

The black square is the query pixel.

More examples are presented in Appendix A.

In this section, we review the known differences and similarities between CNNs and transformers.

The use of CNN networks for text-at word level (Gehring et al., 2017) or character level (Kim, 2014) -is more seldom than transformers (or RNN).

Transformers and convolutional models have been extensively compared empirically on tasks of Natural Language Processing and Neural Machine Translation.

It was observed that transformers have a competitive advantage over convolutional model applied to text (Vaswani et al., 2017) .

It is only recently that ; Ramachandran et al. (2019) used transformers on images and showed that they achieve similar accuracy as ResNets.

However, their comparison only covers performance and number of parameters and FLOPS but not expressive power.

Beyond performance and computational-cost comparisons of transformers and CNN, the study of expressiveness of these architectures has focused on their ability to capture long-term dependencies .

Another interesting line of research has demonstrated that transformers are Turingcomplete (Dehghani et al., 2018; P??rez et al., 2019) , which is an important theoretical result but is not informative for practitioners.

To the best of our knowledge, we are the first to show that the class of functions expressed by a layer of self-attention encloses all convolutional filters.

The closest work in bridging the gap between attention and convolution is due to Andreoli (2019) .

They cast attention and convolution into a unified framework leveraging tensor outerproduct.

In this framework, the receptive field of a convolution is represented by a "basis" tensor A ??? R K??K??H??W ??H??W .

For instance, the receptive field of a classical K ?? K convolutional kernel would be encoded by A ???,q,k = 1{k ??? q = ???} for ??? ??? ??? ??? K .

The author distinguishes this index-based convolution with content-based convolution where A is computed from the value of the input, e.g., using a key/query dot-product attention.

Our work moves further and presents sufficient conditions for relative positional encoding injected into the input content (as done in practice) to allow content-based convolution to express any index-based convolution.

We further show experimentally that such behavior is learned in practice.

We showed that self-attention layers applied to images can express any convolutional layer (given sufficiently many heads) and that fully-attentional models learn to combine local behavior (similar to convolution) and global attention based on input content.

More generally, fully-attentional models seem to learn a generalization of CNNs where the kernel pattern is learned at the same time as the filters-similar to deformable convolutions (Dai et al., 2017; Zampieri, 2019) .

Interesting directions for future work include translating existing insights from the rich CNNs literature back to transformers on various data modalities, including images, text and time series.

Jean-Baptiste Cordonnier is thankful to the Swiss Data Science Center (SDSC) for funding this work.

Andreas Loukas was supported by the Swiss National Science Foundation (project "Deep Learning for Graph Structured Data", grant number PZ00P2 179981).

We present more examples of attention probabilities computed by self-attention model.

Figure 7 shows average attention at a different query pixel than Figure 6 .

Figures 8 to 10 display attention for single images.

Figure 7: Attention probabilities for a model with 6 layers (rows) and 9 heads (columns) using learned relative positional encoding and content-content attention.

We present the average of 100 test images.

The black square is the query pixel.

Figure 8: Attention probabilities for a model with 6 layers (rows) and 9 heads (columns) using learned relative positional encoding and content-content based attention.

The query pixel (black square) is on the frog head.

Figure 9: Attention probabilities for a model with 6 layers (rows) and 9 heads (columns) using learned relative positional encoding and content-content based attention.

The query pixel (black square) is on the horse head.

Figure 10: Attention probabilities for a model with 6 layers (rows) and 9 heads (columns) using learned relative positional encoding and content-content based attention.

The query pixel (black square) is on the building in the background.

Proof.

Our first step will be to rework the expression of the Multi-Head Self-Attention operator from equation (1) and equation (4) such that the effect of the multiple heads becomes more transparent:

Note that each head's value matrix W (h) val ??? R Din??D h and each block of the projection matrix W out of dimension D h ?? D out are learned.

Assuming that D h ??? D out , we can replace each pair of matrices by a learned matrix W (h) for each head.

We consider one output pixel of the multi-head self-attention and drop the bias term for simplicity:

with a

q,: ) k .

We rewrite the output of a convolution at pixel q in the same manner:

Equality between equations (16) and (17) has a restricted support: only the columns associated with a pixel shift ??? ??? ??? ??? K in the receptive field of pixel q can be non-zero.

This leads to the factorization Figure 11 where W conv ??? R Necessary.

Assume there exists x ??? R HW such that x ??? row(E q ) and x ??? row(A q ) and set x to be a row of V

We noticed the similarity of the attention probabilities in the quadratic positional encoding (Section 3) to isotropic bivariate Gaussian distributions with bounded support:

Building on this observation, we further extended our attention mechanism to non-isotropic Gaussian distribution over pixel positions.

Each head is parametrized by a center of attention ??? and a covariance matrix ?? to obtain the following attention scores,

where, once more, ?? = k ??? q. The last term can be discarded because the softmax is shift invariant and we rewrite the attention coefficient as a dot product between the head target vector v and the relative position encoding r ?? (consisting of the first and second order combinations of the shift in pixels ??):

Evaluation.

We trained our model using this generalized quadratic relative position encoding.

We were curious to see if, using the above encoding the self-attention model would learn to attend to non-isotropic groups of pixels-thus forming unseen patterns in CNNs.

Each head was parametrized by ??? ??? R 2 and ?? ???1/2 ??? R 2??2 to ensure that the covariance matrix remained positive semi-definite.

We initialized the center of attention to ??? (h) ??? N (0, 2I 2 ) and ?? ???1/2 = I 2 + N (0, 0.01I 2 ) so that initial attention probabilities were close to an isotropic Gaussian.

Figure 12 shows that the network did learn non-isotropic attention probability patterns, especially in high layers.

Nevertheless, the fact that we do not obtain any performance improvement seems to suggest that attention non-isotropy is not particularly helpful in practice-the quadratic positional encoding suffices.

Figure 12 : Centers of attention of each attention head (different colors) for the 6 self-attention layers using non-isotropic Gaussian parametrization.

The central black square is the query pixel, whereas solid and dotted circles represent the 50% and 90% percentiles of each Gaussian, respectively.

Pruning degenerated heads.

Some non-isotropic attention heads attend on "non-intuitive" patches of pixels: either attending a very thin stripe of pixels, when ?? ???1 was almost singular, or attending all pixels uniformly, when ?? ???1 was close to 0 (i.e. constant attention scores).

We asked ourselves, are such attention patterns indeed useful for the model or are these heads degenerated and unused?

To find out, we pruned all heads having largest eigen-values smaller than 10 ???5 or condition number (ratio of the biggest and smallest eigen-values) greater than 10 5 .

Specifically in our model with 6-layer and 9-heads each, we pruned [2, 4, 1, 2, 6, 0] heads from the first to the last layer.

This means that these layers cannot express a 3 ?? 3 kernel anymore.

As shown in yellow on fig. 2 , this ablation initially hurts a bit the performance, probably due to off biases, but after a few epochs of continued training with a smaller learning rate (divided by 10) the accuracy recovers its unpruned value.

Hence, without sacrificing performance, we reduce the size of the parameters and the number of FLOPS by a fourth.

For completeness, we also tested increasing the number of heads of our architecture from 9 to 16.

Similar to Figure 4 , we see that the network distinguishes two main types of attention patterns.

Localized heads (i.e., those that attend to nearly individual pixels) appear more frequently in the first few layers.

The self-attention layer uses these heads to act in a manner similar to how convolutional layers do.

Heads with less-localized attention become more common at higher layers.

@highlight

A self-attention layer can perform convolution and often learns to do so in practice.