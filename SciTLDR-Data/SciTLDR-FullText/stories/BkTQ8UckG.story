We present a new technique for learning visual-semantic embeddings for cross-modal retrieval.

Inspired by the use of hard negatives in structured prediction, and ranking loss functions used in retrieval, we introduce a simple change to common loss functions used to learn multi-modal embeddings.

That, combined with fine-tuning and the use of augmented data, yields significant gains in retrieval performance.

We showcase our approach, dubbed VSE++, on the MS-COCO and Flickr30K datasets, using ablation studies and comparisons with existing methods.

On MS-COCO our approach outperforms state-of-the-art methods by 8.8% in caption retrieval, and 11.3% in image retrieval (based on R@1).

Joint embeddings enable a wide range of tasks in image, video and language understanding.

Examples include shape-image embeddings BID18 ) for shape inference, bilingual word embeddings BID33 ), human pose-image embeddings for 3D pose inference BID17 ), fine-grained recognition BID22 ), zero-shot learning BID7 ), and modality conversion via synthesis BID23 a) ).

Such embeddings entail mappings from two (or more) domains into a common vector space in which semantically associated inputs (e.g., text and images) are mapped to similar locations.

The embedding space thus represents the underlying structure of the domains, where locations and often direction are semantically meaningful.

In this paper we focus on learning visual-semantic embeddings, central to tasks such as imagecaption retrieval and generation BID13 ; BID11 , and visual questionanswering BID20 .

One approach to visual question-answering, for example, is to first describe an image by a set of captions, and then to find the nearest caption in response to a question BID0 ; BID32 ).

In the case of image synthesis from text, one approach is to invert the mapping from a joint visual-semantic embedding to the image space BID23 a) ).Here we focus on visual-semantic embeddings for the generic task of cross-modal retrieval; i.e. the retrieval of images given captions, or of captions from a query image.

As is common in information retrieval, we measure performance by R@K, i.e., recall at K -the fraction of queries for which the correct item is retrieved in the closest K points to the query in the embedding space (K is usually a small integer, often 1).

More generally, retrieval is a natural way to assess the quality of joint embeddings for image and language data for use in subsequent tasks BID9 ).To this end, the problem is one of ranking, for which the correct target(s) should be closer to the query than other items in the corpus, not unlike learning to rank problems (e.g., BID16 ), and max-margin structured prediction BID2 ; .

The formulation and model architecture in this paper are most closely related to those of BID13 , learned with a triplet ranking loss.

In contrast to that work, we advocate a novel loss, the use of augmented data, and fine-tuning, that together produce a significant increase in caption retrieval performance over the baseline ranking loss on well-known benchmark datasets.

We outperform the best reported result on MS-COCO by almost 9%.

We also demonstrate that the benefit from a more powerful image encoder, and fine-tuning the image encoder, is amplified with the use of our stronger loss function.

To ensure reproducibility, our code will be made publicly available.

We refer to our model as VSE++.Finally, we note that our formulation complements other recent articles that propose new model architectures or similarity functions for this problem.

BID28 propose an embedding network to fully replace the similarity function used for the ranking loss.

An attention mechanism on both image and caption is used by BID21 , where the authors sequentially and selectively focus on a subset of words and image regions to compute the similarity.

In BID10 , the authors use a multi-modal context-modulated attention mechanism to compute the similarity between an image and a caption.

Our proposed loss function and triplet sampling could be extended and applied to other such approaches.

For image-caption retrieval the query is a caption and the task is to retrieve the most relevant image(s) from a database.

Or the query may be an image and one retrieves relevant captions.

The goal is to maximize recall at K (R@K), the fraction of queries for which the most relevant item is ranked among the top K items returned.

DISPLAYFORM0 be a training set of image-caption pairs.

We refer to (i n , c n ) as positive pairs and (i n , c m =n ) as negative pairs; i.e., the most relevant caption to the image i n is c n and for caption c n , it is the image i n .

We define a similarity function s(i, c) ∈ R that should, ideally, give higher similarity scores to positive pairs than negatives.

In caption retrieval, the query is an image and we rank a database of captions based on the similarity function; i.e., R@K is the percentage of queries for which the positive caption is ranked among the top K captions using s(i, c).

And likewise for image retrieval.

In what follows the similarity function is defined on the joint embedding space.

This approach differs from others, such as BID28 , which use a similarity network to directly classify an image-caption pair as matching or non-matching.

Let φ(i; θ φ ) ∈ R D φ be a feature-based representation computed from the image (e.g. the representation before logits in VGG19 BID24 ) or ResNet152 BID8 )).

Similarly, let ψ(c; θ ψ ) ∈ R D ψ be a representation of a caption c in a caption embedding space (e.g. a GRU-based text encoder).

Here, θ φ and θ ψ denote the model parameters used for the respective mappings to obtain the initial image and caption representations.

The mappings into the joint embedding space are then defined in terms of linear projections; i.e., DISPLAYFORM0 where DISPLAYFORM1 , and g(c; W g , θ ψ ), to lie on the unit hypersphere.

The similarity function in the embedding space is then defined as an inner product: DISPLAYFORM2 Let θ = {W f , W g , θ ψ } be the model parameters.

If we also fine-tune the image encoder, then we would also include θ φ in θ.

Training entails the minimization of empirical loss with respect to θ, i.e., the cumulative loss over training data S = {(i n , c n )} N n=1 : DISPLAYFORM3 where (i n , c n ) is a suitable loss function for a single training exemplar.

Recent approaches to joint visual-semantic embeddings have used a form of triplet ranking loss BID13 ; BID11 ; BID31 ; BID25 ), inspired Figure 1: An illustration of typical positive pairs and the nearest negative samples.

Here assume similarity score is the negative distance.

Filled circles show a positive pair (i, c), while empty circles are negative samples for the query i.

The dashed circles on the two sides are drawn at the same radii.

Notice that the hardest negative sample c is closer to i in (a).

Assuming a zero margin, (b) has a higher loss with the SH loss compared to (a).

The MH loss assigns a higher loss to (a).

its use in image retrieval BID6 ; BID3 ).

Prior work has employed a hinge-based, triplet ranking loss with margin α: DISPLAYFORM4 where DISPLAYFORM5 .

This hinge loss comprises two symmetric terms, with i and c being queries.

The first sum is taken over all negative captionsĉ given query i.

The second negative imagesî given caption c.

Each term is proportional to the expected loss (or violation) over sets of negative samples.

If i and c are closer to one another in the joint embedding space than to any negatives pairs, by the margin α, the hinge loss is zero.

In practice, for computational efficiency, rather than summing over all possible negatives in the training set, it is common to only sum over (or randomly sample) the negatives within a mini-batch of stochastic gradient descent (e.g., see BID13 ; BID25 ; BID11 )

.Of course there are other loss functions that one might consider.

One approach is a pairwise hinge loss in which elements of positive pairs are encouraged to be within a radius ρ 1 in the joint embedding space, while negative pairs should be no closer than ρ 2 > ρ 1 .

This is problematic as it constrains the structure of the latent space more than does the ranking loss, and it entails the use of two hyper-parameters which can be very difficult to set.

Another possible approach is to use Canonical Correlation Analysis to learn W f and W g , thereby trying to preserve correlation between the text and images in the joint embedding (e.g., BID14 ; BID4 ).

By comparison, when measuring performance as R@K, for small K, a correlation-based loss will not give sufficient influence to the embedding of negative items in the local vicinity of positive pairs, which is critical for R@K.

Inspired by common loss functions used in structured prediction BID26 ; BID30 BID5 ), we focus on hard negatives for training, i.e., the negatives closest to each training query.

This is particularly relevant for retrieval since it is the hardest negative that determines success or failure as measured by R@1.Given a positive pair (i, c), the hardest negatives are given by i = arg max j =i s(j, c) and c = arg max d =c s(i, d).

To emphasize hard negatives we therefore define our loss as DISPLAYFORM0 Like Eq. 5, the loss comprises two terms, one with i and one with c as queries.

Unlike Eq. 5, this loss is specified in terms of the hardest negatives, c and i .

Hereafter, we refer to the loss in Eq. 6 as Max of Hinges (MH) loss, and the loss function in Eq. 5 as Sum of Hinges (SH) loss.

An example of where the MH loss is superior to SH is when multiple negatives with relatively small violations combine to dominate the SH loss.

For example, in Fig. 1 , a positive pair is depicted together with two sets of negative samples.

In Fig. 1(a) , there exists a single negative sample that is too close to the query.

Essentially, moving such a hard negative, might require a significant change to the mapping.

However, any training step that pushes the hard negative away, can bring back many small violating negative samples, as in Fig. 1(b) .

Using the SH loss, these 'new' negative samples may dominate the loss, so the model is pushed back to the first example in Fig. 1(a) .

This may create local minima in the SH loss that may not be as problematic for the MH loss as it focuses solely on the hardest negative.

For computational efficiency, instead of finding the hardest negatives in the whole training set, we find them in a mini-batch.

With random sampling of the mini-batches, this approximate yields other advantages.

One is that there is a high probability of getting hard negatives that are harder than at least 90% of the entire training set.

Moreover, the loss is potentially robust to label errors in the training data because the probability of sampling the hardest negative over the entire training set is somewhat low.

In Appendix A, we analyze the probability of sampling hard negatives further.

We first perform experiments with our approach, VSE++, and compare it to a baseline formulation with SH loss, referred to as VSE0, and other state-of-the-art approaches.

Essentially, the baseline formulation, VSE0, is the same used by BID13 , here referred to as UVS.We experiment with two image encoders: VGG19 by BID24 and ResNet152 by BID8 .

In what follows below we use VGG19 unless specified otherwise.

As in previous work we extract image features directly from FC7, the penultimate fully connected layer.

The dimensionality of the image embedding, D φ , is 4096 for VGG19 and 2048 for ResNet152.In somewhat more detail, we first resize the image to 256 × 256, and then use either a single center crop of size 224 × 224 or the mean of feature vectors for 10 crops of similar size, as done by BID14 and BID27 .

We refer to training with one center crop as 1C and training with 10 crops as 10C.

We also consider using random crops, denoted by RC.

For RC, we have the full VGG19 model and extract features over a single randomly chosen cropped patch on the fly as opposed to pre-computing the image features once and reusing them.

For the caption encoder, we use a GRU similar to the one used in BID13 .

We set the dimensionality of the GRU, D ψ , and the joint embedding space, D, to 1024.

The dimensionality of the word embeddings that are input to the GRU is set to 300.We further note that in BID13 , the caption embedding is normalized, while the image embedding is not.

Normalization of both vectors means that the similarity function is cosine similarity.

In VSE++ we normalize both vectors.

Not normalizing the image embedding changes the importance of samples.

In our experiments, not normalizing the image embedding helped the baseline, VSE0, to find a better solution.

However, VSE++ is not significantly affected by this normalization.

We evaluate our method on the Microsoft COCO dataset BID19 ) and the Flickr30K dataset BID29 ).

Flickr30K has a standard 30, 000 images for training.

Following BID11 , we use 1000 images for validation and 1000 images for testing.

We also use the splits of BID11 for MS-COCO.

In this split, the training set contains 82, 783 images, 5000 validation and 5000 test images.

However, there are also 30, 504 images that were originally in the validation set of MS-COCO but have been left out in this split.

We refer to this set as rV. Some papers use rV for training (113, 287 training images in total) to further improve accuracy.

We report results using both training sets.

Each image comes with 5 captions.

The results are reported by either averaging over 5 folds of 1K test images or testing on the full 5K test images.

We use the Adam optimizer BID12 to train the models.

We train models for at most 30 epochs.

Except for fine-tuned models, we start training with learning rate 0.0002 for 15 epochs and then lower the learning rate to 0.00002 for another 15 epochs.

The fine-tuned models are trained by taking a model that is trained for 30 epochs with a fixed image encoder and then training it for 15 epochs with a learning rate of 0.00002.

We set the margin to 0.2 for most of the experiments.

We use a mini-batch size of 128 in all our experiments.

Notice that since the size of the training set for different models is different, the actual number of iterations in each epoch can vary.

For evaluation on the test set, we tackle over-fitting by choosing the snapshot of the model that performs best on the validation set.

The best snapshot is selected based on the sum of the recalls on the validation set.

The results on the MS-COCO dataset are presented in Table 1 .

To understand the effect of training and algorithmic variations we report ablation studies for the baseline VSE0 (see TAB1 ).

Our best result with VSE++ is achieved by using ResNet152 and fine-tuning the image encoder (row 1.11), where we see 21.2% improvement in R@1 for caption retrieval and 21% improvement in R@1 for image retrieval compared to UVS (rows 1.1 and 1.11).

Notice that using ResNet152 and fine-tuning can only lead to 12.6% improvement using the VSE0 formulation (rows 2.6 and 1.1), while our MH loss function brings a significant gain of 8.6% (rows 1.11 and 2.6).Comparing VSE++ (ResNet152, fine-tuned) to the current state-of-the-art on MS-COCO, 2WayNet (row 1.11 and row 1.5), we see 8.8% improvement in R@1 for caption retrieval and compared to sm-LSTM (row 1.11 and row 1.4), 11.3% improvement in image retrieval.

We also report results on the full 5K test set of MS-COCO in rows 1.13 and 1.14.Effect of the training set.

We compare VSE0 and VSE++ by incrementally improving the training data.

Comparing the models trained on 1C (rows 1.1 and 1.6), we only see 2.7% improvement in R@1 for image retrieval but no improvement in caption retrieval performance.

However, when we train using RC (rows 1.7 and 2.2) or RC+rV (rows 1.8 and 2.3), we see that VSE++ gains an improvement of 5.9% and 5.1%, respectively, in R@1 for caption retrieval compared to VSE0.

This shows that VSE++ can better exploit the additional data.

Effect of a better image encoding.

We also investigate the effect of a better image encoder on the models.

Row 1.9 and row 2.4 show the effect of fine-tuning the VGG19 image encoder.

We see that the gap between VSE0 and VSE++ increases to 6.1%.

If we use ResNet152 instead of VGG19 (row 1.10 and row 2.5), the gap is 5.6%.

As for our best result, if we use ResNet152 and also fine-tune the image encoder (row 1.11 and row 2.6) the gap becomes 8.6%.

The increase in the performance gap shows that the improved loss of VSE++ can better guide the optimization when a more powerful image encoder is used.

Tables 3 summarizes the performance on Flickr30K.

We obtain 23.1% improvement in R@1 for caption retrieval and 17.6% improvement in R@1 for image retrieval (rows 3.1 and 3.14).

We observed that VSE++ over-fits when trained with the pre-computed features of 1C.

The reason is potentially the limited size of the Flickr30K training set.

As explained in Sec. 3.2, we select a snapshot of the model before over-fitting occurs, based on performance with the validation set.

Over-fitting does not occur when the model is trained using the RC training data.

Our results show the improvements incurred by our MH loss persist across datasets, as well as across models.

We have observed that the MH loss can take a few epochs to 'warm-up' during training.

Fig. 2(a) depicts such behavior on the Flickr30K dataset using RC.

One can see that the SH loss starts off faster, but after approximately 5 epochs MH loss surpasses SH loss.

To explain this, the MH loss depends on a smaller set of triplets compared to the SH loss.

At the beginning of the training, there is so much that the model has to learn.

However, the gradient of the MH loss, may only be influenced by a small set of triples.

As such, it can take longer to train a model with the MH loss.

We explored a simple form of curriculum learning BID1 ) to speed-up the training.

We start training with the SH loss for a few epochs, then switch to the MH loss for the rest of the training.

However, it did not perform better than training solely with the MH loss.

In practice, our MH loss searches for the hardest negative only within each mini-batch at each iteration.

To explore the impact of this approximation we examined how performance depends on the effective sample size over which we searched for negatives (while keeping the mini-batch size fixed at 128).

In the extreme case, when the negative set is the training set, we get the hardest negatives in the entire training set.

As discussed in Sec. 2.3, sampling a negative set smaller than the training set can potentially be more robust to label errors.(a) (b) Figure 2 : Analysis of the behavior of the MH loss on the Flickr30K dataset training with RC.

Fig. (a) compares the SH loss to the MH loss TAB3 , row 3.9 and row 3.11).

Notice that, in the first 5 epochs the SH loss achieves a better performance, however, from there-on the MH loss leads to much higher recall rates.

Fig. (b) shows the effect of the negative set size on the R@1 performance.

Table 4 : Comparison on MS-COCO.

Training set for all the rows is 10C+rV. Fig. 2(b) shows the effect of the negative sample size on the MH Loss function.

We compare the caption retrieval performance for different negative set sizes varied from 2 to 512.

In practice, for negative set sizes smaller than the mini-batch size, 128, we randomly sample the negative set from the mini-batch.

In other cases where the mini-batch size is smaller than the negative set, we randomly sample the mini-batch from the negative set.

We observe that on this dataset, the optimal negative set size is around 128.

Interestingly, for negative sets as small as 2, R@1 is slightly below VSE0.

To understand this, note that the SH loss is still over a large sample size which has a relatively high probability of containing hard negatives.

For large negative sets, the model takes longer to train for the first epochs.

Using the negative set size 512, the performance dropped.

This can be due to the small size of the dataset and the increase in the probability of sampling the hardest negative and outliers.

Even though the performance drops with larger mini-batch sizes, it still performs better than the SH loss.

Given the simplicity of our approach, our proposed loss function can complement the recent approaches that use more sophisticated model architectures or similarity functions.

Here we demonstrate the benefits of the MH loss by applying it to another approach to joint embeddings called order-embeddings BID27 .

The main difference with the formulation above is the use of an asymmetric similarity function, i.e., s(i, c) = − max(0, g(c; W g , θ ψ ) − f (i; W f , θ φ )) 2 .

Again, we simply replace their use of the SH loss by our MH loss.

Like their experimental setting, we use the training set 10C+rV. For our Order++, we use the same learning schedule and margin as our other experiments.

However, we use their training settings to train Order0.

We start training with a learning rate of 0.001 for 15 epochs and lower the learning rate to 0.0001 for another 15 epochs.

Like BID27 we use a margin of 0.05.

Additionally, BID27 takes the absolute value of embeddings before computing the similarity function which we replicate only for Order0.

GT: A woman in a short pink skirt holding a tennis racquet.

[6] A man playing tennis and holding back his racket to hit the ball.

A woman is standing while holding a tennis racket.

Figure 3: Examples of test images and the top 1 retrieved captions for VSE0 and VSE++ (ResNet)-finetune.

The value in brackets is the rank of the highest ranked ground-truth caption.

GT is a sample from the ground-truth captions.

Table 4 reports the results when the SH loss is replaced by the MH loss.

We replicate their results using our Order0 formulation and get slightly better results (row 4.1 and row 4.3).

We observe 4.5% improvement from Order0 to Order++ in R@1 for caption retrieval (row 4.3 and row 4.5).

Compared to the improvement from VSE0 to VSE++, where the improvement on the 10C+rV training set is 1.8%, we gain an even higher improvement here.

This shows that the MH loss can potentially improve numerous similar loss functions used in retrieval and ranking tasks.

This paper focused on learning visual-semantic embeddings for cross-modal, image-caption retrieval.

Inspired by structured prediction, we proposed a new loss based on violations incurred by relatively hard negatives compared to current methods that used expected errors BID13 BID27 ).

We performed experiments on the MS-COCO and Flickr30K datasets and showed that our proposed loss significntly improves performance on these datasets.

We observed that the improved loss can better guide a more powerful image encoder, ResNet152, and also guide better when fine-tuning an image encoder.

With all modifications, our VSE++ model achieves state-of-the-art performance on the MS-COCO dataset, and is slightly below the best recent model on the Flickr30K dataset.

Our proposed loss function can be used to train more sophisticated models that have been using a similar ranking loss for training.

@highlight

A new loss based on relatively hard negatives that achieves state-of-the-art performance in image-caption retrieval.

@highlight

Learning joint embedding of sentences and images using triplet loss that is applied to hardest negatives instead of averaging over all triplets