Learning rich and compact representations is an open topic in many fields such as word embedding, visual question-answering, object recognition or image retrieval.

Although deep neural networks (convolutional or not) have made a major breakthrough during the last few years by providing hierarchical, semantic and abstract representations for all of these tasks, these representations are not necessary as rich as needed nor as compact as expected.

Models using higher order statistics, such as bilinear pooling, provide richer representations at the cost of higher dimensional features.

Factorization schemes have been proposed but without being able to reach the original compactness of first order models, or at a heavy loss in performances.

This paper addresses these two points by extending factorization schemes to codebook strategies, allowing compact representations with the same dimensionality as first order representations, but with second order performances.

Moreover, we extend this framework with a joint codebook and factorization scheme, granting a reduction both in terms of parameters and computation cost.

This formulation leads to state-of-the-art results and compact second-order models with few additional parameters and intermediate representations with a dimension similar to that of first-order statistics.

Learning rich and compact representations is an open topic in many fields such as word embedding BID16 ), visual question-answering ), object recognition BID23 ) or image retrieval BID19 ).

The standard approach extracts features from the input data (text, image, etc.) and builds a representation that will be next processed for a given task (classification, retrieval, etc.) .

These features are usually extracted with deep neural networks and the representation is trained in an end-to-end manner.

Recently, representations that compute first order statistics over input data have been outperformed by improved models that compute higher order statistics such as bilinear models.

This embedding strategy generates richer representations and has been applied in a wide range of tasks : word embedding BID2 ), VQA BID9 ), fine grained classification BID28 ), etc. and gets state-of-the-art results.

For instance, Bilinear models perform the best for fine grained visual classification tasks by producing efficient representations that model more details within an image than classical first order statistics BID14 ).However, even if the increase in performances is unquestionable, second order models suffer from a collection of drawbacks: Their intermediate dimension increases quadratically with respect to input features dimension, they require a projection to lower dimension that is costly both in number of parameters and in computation, they are harder to train than first order models due to the increased dimension, they lack a proper adapted pooling scheme which leads to sub-optimal representations.

The two main downsides, namely the high dimensional output representations and the sub-efficient pooling scheme, have been widely studied over the last decade.

On one hand, the dimensionality issue has been studied through factorization scheme, either representation oriented such as Compact Bilinear Pooling ) and Hadamard Product for Low Rank Bilinear Pooling BID9 ), or task oriented as Low-rank Bilinear Pooling BID10 ).

While these factorization schemes are efficient in term of computation cost and number of parameters, the intermediate representation is still too large (typically 10k dimension) to ease the training process and using lower dimension greatly deteriorate performances.

On the other hand, it is well-known that global average pooling schemes aggregate unrelated features.

This problem has been tackled by the use of codebooks such as VLAD BID0 ) or, in the case of second-order information, Fisher Vectors BID20 ).

These strategies have been enhanced to be trainable in an end-to-end manner BID1 ; BID24 ).

However, using a codebook on end-to-end trainable second order features leads to an unreasonably large model, since the already large second order model has to be duplicated for each entry of the codebook.

This is for example the case in MFAFVNet BID13 ) for which the second order layer alone (i.e., without the CNN part) already costs over 25M parameters and 40 GFLOP, or about as much as an entire ResNet50.In this paper, we tackle both of these shortcomings (intermediate representation cost and lack of proper pooling) by exploring joint factorization and codebook strategies.

Our main results are the following:-We first show that state-of-the-art factorization schemes can already be improved by the use of a codebook pooling, albeit at a prohibitive cost.

-We then propose our main contribution, a joint codebook and factorization scheme that achieves similar results at a much reduced cost.

Since our approach focuses on representation learning and is task agnostic, we validate it in a retrieval context on several image datasets to show the relevance of the learned representations.

We show our model achieves competitive results on these datasets at a very reasonable cost.

The remaining of this paper is organized as follows: in the next section, we present the related work on second order pooling, factorization schemes and codebook strategies.

In section 3, we present our factorization with the codebook strategy and how we improve its integration.

In section 4, we show an ablation study on the Stanford Online Products dataset BID18 ).

Finally, we compare our approach to the state-of-the-art methods on three image retrieval datasets (Stanford Online Products, CUB-200-2001, Cars-196) .

In this section, we focus on methods that use representations based on second-order information and we provide a comparison in terms of computational efficiency and number of parameters for the second-order layer.

These second-order methods exploit either bilinear pooling (section 2.1) and factorization schemes (section 2.2) or codebook strategies (section 2.3).

In this section, we briefly review end-to-end trainable Bilinear pooling BID14 ).

This method extracts representations from the same image with two CNNs and computes the crosscovariance as representation.

This representation outperforms its first-order version and other second-order representations such as Fisher Vectors BID20 ) once the global architecture is fine-tuned.

However, bilinear pooling leads to a small improvement compared to secondorder pooling (i.e., the covariance of the CNN features) at the cost of a higher computation.

Most of recent works on bilinear pooling only focus on computing covariance of the extracted features, that is : DISPLAYFORM0 where X = {x i ??? R d |i ??? S} ??? R d??hw is the matrix of concatenated CNN features, h and w are the height and the width of the extracted feature map.

Another formulation is the vectorized version of y obtained by computing the Kronecker product (???) of x i with itself: DISPLAYFORM1 Due to the very high dimension of this representation, most recent works on bilinear pooling tend to improve this representation by providing new factorization scheme to reduce the computation.

Recent works on bilinear pooling proposed factorization schemes with two objectives: avoiding the direct computation of second order features and reducing the high dimensionality output representation.

proposed Compact Bilinear Pooling that tackles the high dimensionality of second-order features by analyzing two low-rank approximations of the equivalent polynomial kernel (equation FORMULA1 from ): DISPLAYFORM0 with ?? the mapping function that approximates the kernel.

This compact formulation allows to keep less than 4% of the components with nearly no loss in performances compared to the uncompressed model.

BID9 Instead, it projects the features using a hyperplan and computes a quadratic form.

The matrix of this quadratic form is supposed rank deficient to reduce the number of parameters and the computation cost with negligible loss in performances.

Thus, the output representation (or directly the number of classes for classification tasks) is generated by concatenating these scalars for all projections.

Wei et al. FORMULA0 presented Grassmann Bilinear Pooling as a new factorization scheme.

The objective is to take advantage of the rank deficient covariance matrix using Singular Value Decomposition (SVD) and then compute the classifier over these Grassmann manifolds.

This factorization is efficient in the sense that it never directly computes the second-order representation and contrary to LR-BP, this formulation allows the construction of a representation, by replacing the number of classes by the representation dimension.

In practice, however, they need to greatly reduce the input feature dimension due to the SVD complexity which is cubic in the feature dimension.

In this work, we start from a similar factorization as BID9 detailed in section 3.1.

However, this factorization is improved by the introduction of a codebook strategy that allows smaller representation dimension and improves performances.

An acknowledged drawback of pooling methods is that they pool unrelated features that may decrease performances.

To cope with this observation, codebook strategies have been proposed and greatly improved performances by pooling only features that belong to the same codeword.

The first representations that take advantage of codebook strategies are Bag of Words (BoW) and in the case of second order information Fisher Vectors BID20 ).

Fisher Vectors (FVs) extend the BoW framework by replacing the hard assignment of BoW by a Gaussian Mixture Model (GMM) and then compute the representation as an extension of the Fisher Kernel.

In practice, covariance matrices are supposed to be diagonal which leads to representations of size N (2d + 1) where d is the dimension of the features and N is the codebook size.

BID24 proposed FisherNet, an architecture that integrates FVs as differentiable layer.

The proposed layer outperforms non-trainable FVs approach but nonetheless has the high output dimension of the original FV.

BID13 introduced MFA-FV network, a deep architecture which extends the MFA-FV of BID4 by producing a second order information embedding trainable in an endto-end manner.

The proposed formulation takes advantage of both worlds: MFA-FV generates an efficient representation of non-linear manifolds with a small latent space and it can be trained in an end-to-end manner.

The main drawbacks of their method is the direct computation of second-order DISPLAYFORM0 Table 1: Summary of second order methods.

C. and F. columns are respectively for "Codebook" and "Factorization".

Numbers in brackets are typical values.

Methods marked ??? used the original paper values.

Other methods use the following parameters: h = w = 28 are the height and width of the feature map, d = 512 is the feature dimension, D is the output representation, and is set to 512 if possible.

Our proposed method uses a codebook N = 32 and a projection set of size R = 8.features for each codeword (computation cost), the raw projection of this covariance matrix into the latent space for each codeword (computation cost and number of parameters), and finally the representation dimension.

In the original paper, the proposed representation reaches 500k dimension, twice the already high dimension of Bilinear Pooling.

For a more compact review, computation cost, number of parameters, use of codebook and/or factorization are sumed-up in table 1.

This table shows that, to our knowledge, no efficient factorization combined with codebook strategy has been proposed to exploit the richer representation due to codebook but at a small increase in terms of number of parameters or computation cost.

As is shown in this table, our proposition combine the best of both worlds by providing a joint codebook and factorization optimization scheme with a similar number of parameters and computation cost to that of methods without codebook strategies.

In section 3.1, we detail the initial factorization scheme and the properties of the Kronecker Product and the dot product that are used in the two next sections.

In section 3.2, we extend this factorization to a codebook strategy and show the limitations of this architecture in terms of computation cost, low-rank approximation, number of parameters, etc.

In section 3.3 we enhance this representation by sharing projectors to all codewords into the codebook, leading to a joint codebook and factorization optimization.

In this section, we present the factorization used and highlight the advantages and limitations of this scheme.

For a given input feature x ??? R d , we compute its second-order representation DISPLAYFORM0 and project it into a smaller subspace with W ??? R d 2 ??D to build the output feature z(x) ??? R D .

These output features are then pooled to build the output representation z: DISPLAYFORM1 In the rest of the paper, we use the notation z i that refers to the i-th dimension of the output representation z and z i (x) the i-th dimension of the output feature z(x), that is: DISPLAYFORM2 with DISPLAYFORM3 a column of W and ?? ; ?? the dot product.

Then we enforce a factorization of w i to take advantage of the properties of dot product and Kronecker product, that is DISPLAYFORM4 Thus, we use the following rank one decomposition of FORMULA6 becomes: DISPLAYFORM5 DISPLAYFORM6 This factorization is efficient in term of parameters as it needs only 2dD parameters instead of d 2 D for the full projection matrix.

However, even if this rank one decomposition allows interesting dimension reduction ; BID9 ) it is not enough to keep rich representation with smaller dimension.

Consequently, we extend the second-order feature to a codebook strategy.

To extend second-order pooling, we want to pool only similar features, that is which belong to the same codeword.

This codebook pooling is interesting because each projection to a sub-space should have only similar features, and they should be encoded with fewer dimension.

For a codebook size of N , we compute an assignment function h(??) ??? R N .

This function could be a hard assignment function (e.g., arg min over distance to each cluster) or a soft assignment (e.g., the softmax function).

Thus, our output feature z i (x) becomes: DISPLAYFORM0 Remark that now W ??? R DISPLAYFORM1 Here, we duplicate h(x) for generalization purpose: In the case of the original bilinear pooling this formulation becomes h 1 (x 1 ) ??? x 1 ??? h 2 (x 2 ) ??? x 2 and we can use two different codebooks, one for each network.

Moreover, this formulation allows more degrees of freedom for the next factorization.

As in equation 6, we enforce the rank one decomposition of DISPLAYFORM2 2 .

This first factorization leads to the following output feature z i (x): DISPLAYFORM3 This intermediate representation is too large to be computed directly, e.g. using N = 100 and the same parameters as in and q i = j e (j) ??? v i,j where e (j) ??? R N is the j-th vector from the natural basis of R N and DISPLAYFORM4 The decompositions of p i and q i play similar roles as intra-projection in VLAD representation BID3 ).

Indeed, if we consider h(??) as a hard assignment function, the projection that will be computed is the only one assigned to the corresponding codewords.

Thus, this model learns a projection matrix for each codebook entry.

Furthermore, equation FORMULA15 can be factorized using h j (x), the j-th component of h(x): DISPLAYFORM5 where U i ??? R d??N is the matrix concatenating the projections of all entries of the codebook for the i-th output dimension.

We call this approach C-CBP as it corresponds to the extension of CBP ) to a Codebook strategy.

This representation has multiple advantages: First, it computes second order features that leads to better performances compared to its first order counterpart.

Second, the first factorization provides an efficient alternative in terms of number of parameters and computation despite the decreasing performances when it reaches small representation dimension.

This downside is addressed by the third advantage which is the codebook strategy.

It allows the pooling of only related features while their projections to a sub-space is more compressible.

However, even if this codebook strategy improves the performances, the number of parameters is in O(dDN ) As such, using large codebook may become intractable.

In the next section, we extend this scheme by sharing a set of projectors and enhance the decompositions of p i and q i .

In the previous model, for a given codebook entry, there is one dedicated projector that is learned to map to a smaller vector space all features that belong to this codebook entry.

The proposed idea is, instead of using a a one-to-one correspondence, we learn a set of projectors that is shared across the codebook.

The reasoning behind is that projectors from different codebook entries are unlikely to be all orthogonal.

By doing such hypothesis, that is, the vector space spaned by the projection matrices has a lower dimension than the codebook itself, we can have smaller models with nearly no loss in performances.

To check this hypothesis, we extend the proposed factorization from section 3.2.

We want to generate U i from { U i } i???{1,...,R} and V i from { V i } i???{1,...,R} where R is the number of projections in the set.

Then the two new enforced factorization of p i and q i are: DISPLAYFORM0 where f p and f q are two functions from R N to R R that transform the codebook assignment into a set of coefficient which generate their respective projection matrices.

Then, using these factorizations lead to the following equation: DISPLAYFORM1 In this paper, we only study the case of a linear projection to the sub-space DISPLAYFORM2 Finally, the fully factorized z transform is computed using the following equation: DISPLAYFORM3 Equation FORMULA0 is more efficient in terms of parameters than equation FORMULA0 as it requires only 2(RdD + N R) parameters instead of 2N dD. We call this approach JCF for Joint Codebook and Factorization.

This shared projection is both efficient in terms of number of parameters and in computation by a factor R / N .

In the next section, we provide an ablation study of the proposed method, comparing equation FORMULA0 and equation FORMULA0 , demonstrating that learning recombination is both efficient and performing.

In this section, we give some details about our implementation.

All our experiments are performed on image retrieval datasets to assess the quality of the representations independantly of any classification scheme.

We build our model over pre-trained network such as VGG16 BID21 ) or ResNet50 ).

In both case we reduce the features dimension to 256 dimensions and we l 2 -normalize them.

For the assignment function h, we use the softmax over cosine similarity between the features and the codebook.

Once the second-order representations are computed we pull them using global average pooling and we l 2 -normalize the output representation.

Similarities between images are computed using the cosine similarity.

In metric, we use Recall@K which takes the value 1 if there is at least one element from the same instance in the top-K results else 0 and averages these scores over the test set.

The network is trained in 3 steps using a standard triplet loss function.

In the first step, we freeze the ResNet50 and only train our added layers with 100 images per batch, we sample the negative within the batch and we use a learning rate of 10 ???4 for 40 epochs.

In the second one, we unfreeze ResNet50 and fine-tune the whole architecture for 40 epochs more with a learning rate of 10 ???5 and a batch of 64 images with the negative sampled within the batch.

In the last one, we fine-tune the network with a batch size of 64 images sampled by hard mining the training set with a learning rate of 10 ???5 .

The margin of the triplet loss is set to 0.1.

Images are resized to 224x224 pixels for both train and test sets.

Baseline In this section, we demonstrate both the relevance of second-order information for retrieval tasks and the influence of the codebook on our method.

We report recall@1 on Stanford Online Products in TAB3 for the different configuration detailed below with the training procedure from Section 3.4 without the hard mining step.

DISPLAYFORM0 First, as a reference, we train a ResNet50 with a global average pooling and a fully connected layer to project the representation dimension from 2048 to 512.

We denote it Baseline.

Then we reimplement Bilinear Pooling (BP) and Compact Bilinear Pooling (CBP) and extend them naively to a codebook strategy (C-BP and C-CBP).The objective is to demonstrate that such strategy performs well, but a an intractable cost.

Results are reported in TAB3 .

Note that, for each bilinear pooling method, we first add a 1 ?? 1 convolution to project the ResNet50 features from 2048 to 256 dimensions.

This experiment confirm the interest of bilinear pooling in image retrieval with a improvement of 2% over the baseline, while using a 512 dimension representation.

Furthermore, even using a codebook strategy with few codewords enhance bilinear pooling by 1% more, however, the number of parameters become intractable for codebook of size greater than 4: this naive strategy requires 270M parameters to extend this model to a codebook with a size of 8.Using the factorization from equation FORMULA0 greatly reduces the required number of parameters and allows the exploration of larger codebook.

However, this factorization without codebook leads to lower scores than the non factorized bilinear pooling, but adding a codebook strategy increases performances by more than 4% over bilinear pooling without codebook, with nearly 4 times less parameters.

In this part, we study the impact of the sharing projection.

We use the same training procedure as in the previous section.

For each codebook size, we train architecture with a different number of projections, allowing to compare architectures without the sharing process to architectures with greater codebook size but with the same number of parameters by sharing projectors.

Results are reported in Table 3 .

Sharing projectors leads to smaller models with few loss in performances, and using richer codebooks allows more compression with superior results.

In this section, we report performances of our factorization on 3 fine-grained visual classification (FGVC) datasets: CUB (Wah et al. (2011) ), CARS BID11 ) and AIRCRAFT BID15 ).

We use VGG16 as backbone network.

Furthermore, to demonstrate the effectiveness of our codebook based factorization scheme to produce compact but effective second-order representations we compare JCF to closely-related formulations on FGVC tasks, that are: DISPLAYFORM0 .

Non-linear multi-rank with shared U , V .Method CUB CARS AIRCRAFT Feature dim.

Parameters Full BP - BID14 84.1 90.6 86.9 256k 200MB CBP-RM - 83.9 90.5 84.3 8192 38MB CBP-TS - 84 Table 4 : Evaluation of our proposed factorization scheme.

We compare our method to the state-ofthe-art on Bilinear factorization and similar methods.

We evaluate them with small representation dimension to attest our dimensionality reduction efficiency.

* denotes our re-implementation.

DISPLAYFORM1 Multi-rank extension of Eq.(6) which allows to compare the benefit of codebook against direct rank increase.

This approach is related to FBP which uses a higher rank decomposition (R = 20) than in our tests (R = 8).

DISPLAYFORM2 Multi-rank with the same non-linearity as HPBP.

DISPLAYFORM3 which adds weights to the multi-rank combination.

For a fair comparison, we fix the number of parameters for all of these methods to the same number as JCF (N = 32, R = 8).

Thus, all methods use d = 256 and D = 512 and R = 8 except HPBP which uses R = 2048 to compensate for its shared matrices U , V .We report classification accuracy on the three aforementioned datasets in Table 4 .

As we can see, our method consistently outperforms the multi-rank variants.

This confirms our intuition about the importance of grouping features by similarity before projection and aggregation.

Indeed, multirank variants do not have a selection mechanism preceding the projection into the subspace that would allow to selectively choose the projectors based on the input features.

Instead, all features are projected using the same projectors and then aggregated.

We argue that non-linear multi-rank variants bring only marginal improvements, since the non-linearity happens after the projection is made.

Although it is still possible to learn a projection coupled with the non-linearity that would lead to a similarity driven aggregation, it is not enforced by design.

Since JCF does the similarity driven aggregation by design, it is easier to train, which we believe explains the results.

In this section, we compare our method to the state-of-the-art on 3 retrieval datasets: Stanford Online Products BID18 ), CUB-200-2011 (Wah et al. (2011 ) and Cars-196 (Krause et al. (2013) ).

For Stanford Online Products and CUB-200-2011, we use the same train/test split as Oh BID18 .

For Cars-196, we use the same as BID19 .

We report the standard recall@K with K ??? {1, 10, 100, 1000} for Stanford Online Products and with K ??? {1, 2, 4, 8, 16, 32} for the other two.

On CUB-200-2011 and Cars-196 (see TAB6 ) we re-implement Bilinear Pooling (BP) and Compact Bilinear Pooling (CBP) on a VGG16.

Even if the results are interesting, the constraint over intermediate representation is too strong to achieve relevant results.

We then implement the codebook factorization from equation 10 a codebook size of 32 (denoted C-CBP).

This formulation outperforms both classical second-order models by a large margin with few parameters.

Moreover, our model that shares projections over the codebook (JCF , computed following equation 13) with R = 8 has 4 times less parameters for a 2% loss on CUB-200-2011 dataset.

On Cars-196, the sharing induces a higher loss, but may be improved with more projections to share.

On Stanford Online Products, we report the Baseline, implementations from equations 10 (C-CBP) and 13 (JCF ) with R = 8.

We achieve state-of-the-art results using both methods and more than 10% improvement over the Baseline.

Remark that JCF costs 4 times less than C-CBP at a 1% loss.

In this paper, we propose a new pooling scheme based which is both efficient in performances (rich representation) and in representation dimension (compact representation).

This is thanks to the second-order information that allows richer representation than first-order statistics and thanks to a codebook strategy which pools only related features.

To control the computational cost, we extend this pooling scheme with a factorization that shares sets of projections between each entry of the codebook, trading fewer parameters and fewer computation for a small loss in performance.

We achieve state-of-the-art results on Stanford Online Products and Cars-196, two image retrieval datasets.

Even if our tests are performed on image retrieval datasets, we believe our method can readily be used in place of global average pooling for any task.

<|TLDR|>

@highlight

We propose a joint codebook and factorization scheme to improve second order pooling.

@highlight

This paper presents a way to combine existing factorized second order representations with a codebook style hard assignment.

@highlight

Proposal for a novel bilinear representation based on a codebook model, and an efficient formulation in which codebook-based projections are factorized via shared projection to further reduce parameter size.