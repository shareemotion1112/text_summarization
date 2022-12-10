Language style transfer is the problem of migrating the content of a source sentence to a target style.

In many applications, parallel training data are not available and source sentences to be transferred may have arbitrary and unknown styles.

In this paper, we present an encoder-decoder framework under this problem setting.

Each sentence is encoded into its content and style latent representations.

By recombining the content with the target style, we can decode a sentence aligned in the target domain.

To adequately constrain the encoding and decoding functions, we couple them with two loss functions.

The first is a style discrepancy loss, enforcing that the style representation accurately encodes the style information guided by the discrepancy between the sentence style and the target style.

The second is a cycle consistency loss, which ensures that the transferred sentence should preserve the content of the original sentence disentangled from its style.

We validate the effectiveness of our proposed model on two tasks: sentiment modification of restaurant reviews, and dialog response revision with a romantic style.

Style transfer is a long-standing research problem that aims at migrating the content of a sample from a source style to a target style.

Recently, great progress has been achieved by applying deep neural networks to redraw an image in a particular style BID7 BID10 BID2 BID20 BID12 .

However, until now very few approaches have been proposed for style transfer of natural language sentences, i.e., changing the style or genre of a sentence while preserving its semantic content.

For example, we would like a system that can convert a given text piece in the language of Shakespeare BID14 ; or rewrite product reviews with a favored sentiment BID17 .One important issue on language style transfer is that parallel data are unavailable.

For instance, considering the task of rewriting a negative review of a product to its counterpart with a positive sentiment, we can hardly find paired data that describe the same content.

Yet, many text generation frameworks require parallel data, such as the popular sequence-to-sequence model in machine translation and document summarization BID16 , and thus are not applicable under this scenario.

A few recent approaches have been proposed for style transfer with non-parallel data BID4 BID17 .

Their key idea is to learn a latent representation of the content disentangled from the source style, and then recombine it with the target style to generate the corresponding sentence.

All the above approaches assume that data have only two styles, and their task is to transfer sentences from one style to the other.

However, in many practical settings, we may deal with sentences in more than two styles.

Taking the review sentiment modification as an example again, some reviews may be neither positive nor negative, but in a neutral style.

Moreover, even reviews considered negative can be categorized into more fine-grained sentiments, such as anger, sadness, boredom and other negative styles.

It may be beneficial if such styles are treated differently.

As another example, consider a chatbot with a coherent persona, which has a consistent language behavior and interaction style BID9 .

A simple framework for this task is to first use human dialog data to train a chatbot system, such as a retrieval-based dialog model BID11 , and then transfer the output responses with a language style transfer model so that multi-round responses always have a consistent style.

Note that the human dialog sentences are collected from different users, and users' expressions of the content and tones may be in different personalized characteristics.

Thus the output responses retrieved from the dialog model may have the language style of any user.

Simply treating the responses with a single style and employing the existing style transfer models would lead to unsatisfactory results.

Hence, in this paper, we study the setting of language style transfer in which the source data to be transferred can have various (and possibly unknown) styles.

Another challenging problem in language style transfer is that the transferred sentence should preserve the content of the original sentence disentangled from its style.

To tackle this problem, BID17 assumed the source domain and the target domain share the same latent content space, and trained their model by aligning these two latent spaces.

BID4 constrained that the latent content representation of the original sentence could be inferred from the transferred sentence.

However, these attempts considered content modification in the latent content space but not the sentence space.

In this work, we develop an encoder-decoder framework that can transfer a sentence from a source domain to its counterpart in a target domain.

The training data in the two domains are non-parallel, and sentences in the source domain can have arbitrary language styles but those in the target domain are with a consensus style.

We encode each sentence into two latent representations, one for the content disentangled from the style, and the other for the style.

Intuitively, if a source sentence is considered having the target style with a high probability, its style representation should be close to the target style representation.

To make use of this idea, we enforce that the discrepancy between an arbitrary style representation and the target style representation should be consistent with the closeness of its sentence style to the target style.

A cycle consistency loss is further introduced to avoid content change by directly considering the transferred sentence.

Its idea is that the generated sentence, when put back into the encoder and recombined with its original style representation, can recover the original sentence.

We evaluate the performance of our proposed model on two tasks.

The first is the sentiment modification task with its source domain containing more than one sentiments, and the second is to transfer general dialog responses to a romantic style.

Most style transfer approaches in the literatures focus on vision data, and some of them are also designed for the non-parallel data setting.

BID7 proposed to disentangle the content representations from image attributes, and control the image generation by manipulating the graphics code that encodes the attribute information.

BID2 used the Convolutional Neural Networks (CNNs) to learn separated representations of the image content and style, and then created the new image from their combination.

Some approaches have been proposed to align the two data domains with the idea of the generative adversarial networks (GAN) BID3 .

BID10 proposed the coupled GAN framework to learn a joint distribution of multidomain data by the weight-sharing constraint.

BID20 introduced a cycle consistency loss, which minimizes the gap between the transferred images and the original ones.

However, due to the discreteness of the natural language, this loss function cannot be directly applied on text data.

In our work, we show how the idea of cycle consistency can be used on text data.

Only a small number of approaches have been proposed for language style transfer.

To handle the non-parallel data problem, BID14 revised the latent representation of a sentence in a certain direction guided by a classifier, so that the decoded sentence imitates those favored by the classifier.

BID1 encoded textual property values with embedding vectors, and adopted a conditioned language model to generate sentences satisfying the specified content and style properties.

BID4 used the variational auto-encoder (VAE) to encode the sentence into a latent content representation disentangled from the source style, and then recombine it with the target style to generate its counterpart, An additional distribution is added to enforce that the generated sentence and the original sentence share the same latent content representation.

BID17 considered transferring between two styles simultaneously.

Specifically, they utilized adversarial training in the Professor-Forcing framework BID8 , to align the generated sentences from one style to the data domain of the other style.

We also adopt similar adversarial training in our model.

However, since we assume the source domain contains data with various and possibly unknown styles, we cannot align data from the target domain to the source domain as in BID17 .

We now formally present our problem formulation.

Suppose there are two data domains, one source domain X 1 in which each sentence may have its own language style, and one target domain X 2 consisting of data with the same language style.

During training, we observe n samples from X 1 and m samples from X 2 , denoted as X 1 = {x DISPLAYFORM0 Note that we can hardly find a sentence pair (x DISPLAYFORM1 2 ) that describes the same content.

Our task is to design a model to learn from these non-parallel training data such that for an unseen testing sentence x ∈ X 1 , we can transfer it into its counterpartx ∈ X 2 , wherex should preserve the content of x but with the language style in X 2 .

Similar to BID17 ; BID4 , we assume each sentence x can be decomposed into two representations: one is the style representation y ∈ Y, and the other is the content representation z ∈ Z, which is disentangled from its style.

Each sentence x (i) 1 ∈ X 1 has its individual style y DISPLAYFORM0 1 , while all the sentences x (i) 2 ∈ X 2 share the same style, denoted as y * .

Our model is built upon the encoder-decoder framework.

In the encoding module, we assume that z and y of a sentence x can be obtained through two encoding functions E z (x) and E y (x) respectively: DISPLAYFORM1 DISPLAYFORM2 where E y (x) = 1 {x∈X1} · g(x) + 1 {x∈X2} · y , and 1 {·} is an indicator function.

When a sentencex comes from source domain, we use a function g(x) to encode its style representation.

For x from target domain, a shared style representation y is used.

Both y * and parameters in g(x) are learnt jointly together with other parameters in our model.

For the decoding module, we first employ a reconstruction loss to encourage that the sentence from the decoding function given z and y of a sentence x can well reconstruct x itself.

Here, we use a probabilistic generator G as the decoding function and the reconstruction loss is: DISPLAYFORM3 where θ denotes the parameter of the corresponding module.

To enable style transfer using non-parallel training data, we enforce that for a sample x 1 ∈ X 1 , its decoded sequence using G given its content representation z and the target style y * should be in the target domain X 2 .

We use the idea of GAN BID3 ) and introduce an adversarial loss to be minimized in decoding.

The goal of the discriminator D is to distinguish between G(z 1 , y * ) and G(z 2 , y * ), while the generator tries to bewilder the discriminator: DISPLAYFORM4 As discussed in Section 2, since our source domain X 1 contains sentences with various unknown language styles but not a consistent style, it is impossible for us to apply a discriminator to determine whether a sentence transferred from X 2 is aligned in the domain X 1 as in BID17 .During optimization, we adopt the continuous approximation in BID4 for gradients propagation in adversarial training over discrete sentences.

That is, instead of feeding a single word as the input to the generator, we use the approximation averaging word embeddings by a multinomial distribution.

This distribution is computed as softmax(o t /γ), where o t is the logit vector output by the generator at time step t, γ > 0 is a temperature parameter.

Next, we follow the framework of Professor-Forcing BID8 , which matches two sequences of output words using a discriminator D. Specifically, we have one kind of sequences G(z 2 , y * ) teacher-forced by the ground-truth sample x 2 ∈ X 2 , and the other one G(z 1 , y * ) with z 1 obtained from samples in X 1 , in which the input at each time step is self-generated by the previous continuous approximation.

However, the above encoder-decoder framework is under-constrained.

First, for a sample x 1 ∈ X 1 , y 1 can have an arbitrary value that minimizes the above losses in Equation 3 and 4, which may not DISPLAYFORM5 Figure 1: Basic model with the style discrepancy loss.

Solid lines: encode and decode the sample itself; dash lines: transfer DISPLAYFORM6 Figure 2: Proposed cycle consistency loss (can be applied for samples in X 2 similarly).necessarily capture the sentence style.

This will affect the other decomposed part z, making it not fully represent the content which should be invariant with the style.

Second, the discriminator can only encourage the generated sentence to be aligned with the target domain X 2 , but cannot guarantee to keep the content of the source sentence intact.

To address the first problem, we propose a style discrepancy loss, to constrain that the learnt y should have its distance from y * guided by another discriminator which evaluates the closeness of the sentence style to the target style.

For the second problem, we get inspired by the idea in BID20 and introduce a cycle consistency loss applicable to word sequence, which requires that the generated sentencex can be transferred back to the original sentence x.

By using a portion of the training data, we can first train a discriminator D s to predict whether a given sentence x has the target language style with an output probability, denoted as p Ds (x ∈ X 2 ).

When learning the decomposed style representation y 1 for a sample x 1 ∈ X 1 , we enforce that the discrepancy between this style representation and the target style representation y * , should be consistent with the output probability from D s .

Specifically, since the styles are represented with embedding vectors, we measure the style discrepancy using the 2 norm: DISPLAYFORM0 Intuitively, if a sentence has a larger probability to be considered having the target style, its style representation should be closer to the target style representation y * .

Thus, we would like to have d(y 1 , y * ) positively correlated with 1 − p Ds (x 1 ∈ X 2 ).

To incorporate this idea in our model, we use a probability density function q(y 1 , y * ), and define the style discrepancy loss as: DISPLAYFORM1 where f (·) is a valid probability density function.

p Ds (x 1 ∈ X 2 ) is pre-trained and then fixed.

If a sentence x 1 has a large p Ds (x 1 ∈ X 2 ), incorporating the above loss into the encoder-decoder framework will encourage a large q(y 1 , y * ) and hence a small d(y 1 , y * ), which means y 1 would be close to y * .

In our experiment, we instantiate q(y 1 , y * ) with the standard normal distribution for simplicity: DISPLAYFORM2 However, better probability density functions can be used if we have some prior knowledge of the style distribution.

With Equation 8, the style discrepancy loss can be equivalently minimized by: DISPLAYFORM3 3.4 CYCLE CONSISTENCY LOSS Inspired by BID20 , we require that a sentence transferred by the generator G should preserve the content of its original sentence, and thus it should have the capacity to recover the original sentence in a cyclic manner.

For a sample x 1 ∈ X 1 with its transferred sentencex 1 having the target style y * , we encodex 1 and combine its contentz 1 with its original style y 1 for decoding.

We should expect that with a high probability, the original sentence x 1 is generated.

For a sample x 2 ∈ X 2 , though we do not aim to change its language style in our task, we can still compute its cycle consistency loss for the purpose of additional regularization.

We first choose an arbitrary style y 1 obtained from a sentence in X 1 , and transfer x 2 into this y 1 style.

Next, we put this generated sentence into the encoder-decoder model with the style y * , and the original sentence x 2 should be generated.

Formally, the cycle consistency is: DISPLAYFORM4 3.5 FULL OBJECTIVE An illustration of our basic model with the style discrepancy loss is shown in Figure 1 and the full model is combined with the cycle consistency loss shown in Figure 2 .

To summarize, the full loss function of our model is: DISPLAYFORM5 where λ 1 , λ 2 , λ 3 are parameters balancing the relative importance of the different loss parts.

The overall training objective is a minmax game played among the encoder E z , E y , generator G and discriminator D: DISPLAYFORM6 We implement the encoder E z using an RNN with the last hidden state as the content representation, and the style encoder g(x) using a CNN with the output representation of the last layer as the style representation.

The generator G is an RNN that takes the concatenation of the content and style representation as the initial hidden state.

The discriminator D and the pre-trained discriminator D s used in the style discrepancy loss are CNNs with the similar network structure in E y followed by a sigmoid output layer.

Yelp: Raw data are from the Yelp Dataset Challenge Round 10, which are restaurant reviews on Yelp.

Generally, reviews rated with 4 or 5 stars are considered positive, 1 or 2 stars are negative, and 3 stars are neutral.

For positive and negative reviews, we use the processed data released by BID17 .

For neutral reviews, we follow similar steps in BID17 to process and select the data.

We first filter out neutral reviews (rated with 3 stars and categorized with the keyword 'restaurant') with the length exceeding 15 or less than 3.

Then, data selection in Moore & Lewis (2010) is used to ensure a large enough vocabulary overlap between neutral data and data in BID17 .

Afterwards, we sample 500k sentences from the resulting dataset as the neutral data.

We use the positive data as the target style domain.

Based on the three classes of data, we construct two datasets with multiple styles:• Positive+Negative (Pos+Neg): we add different numbers of positive data (50k, 100k, 150k) into the negative data, so that the source domain contains data with two sentiments.• Neutral+Negative (Neu+Neg): we combine neutral (50k, 100k, 150k) and negative data together.

We consider these datasets are harder to learn from.

For the Pos+Neg dataset, we can make use of a pre-trained classifier to possibly filter out some positive data so that most of the source data have the same style and the model in BID17 can work.

However, the neutral data cannot be removed in this way.

Also, most of the real data may be in the neutral sentiment, and we want to see if such sentences can be transferred well.

Details about the data statistics can be found in TAB6 in the Appendix.

Chat: We use sentences from a real Chinese dialog dataset as the source domain.

Users can chat with various personalized language styles, which are not easy to be classified into one of the three sentiments as in Yelp.

Romantic sentences are collected from several online novel websites and filtered by human annotators.

Our task is to transfer the dialog sentences with a romantic style, characterized by the selected romantic sentences.

TAB7 in the Appendix shows detailed statistics about this dataset.

We implement our model using Tensorflow BID0 .

We use GRU as the encoder and generation cells in our encoder-decoder framework.

Dropout BID18 ) is applied in GRUs and the dropout probability is set to 0.5.

Throughout our experiments, we set the dimension of the word embedding, content representation and style representation as 200, 1000 and 500 respectively.

For the style encoder g(x), we follow the CNN architecture in BID5 , and use filter sizes of 200 × {1, 2, 3, 4, 5} with 100 feature maps each, so that the resulting output layer is of size 500, i.e., the dimension of the style representation.

The pre-trained discriminator D s is implemented similar to g(x) but using filter sizes 200 × {2, 3, 4, 5} with 250 feature maps each.

Statistics of data used to pre-train D s are shown in TAB8 and TAB0 in the Appendix.

The testing accuracy of the pre-trained D s is 95.82% for Yelp and 87.72% for Chat respectively.

We further set the balancing parameters λ 1 = λ 2 = 1, λ 3 = 5, and train the model using the Adam optimizer BID6 with the learning rate 10 −4 .

All input sentences are padded so that they have the same length 20 for Yelp and 35 for Chat.

Furthermore, we use the pre-trained word embeddings Glove (Pennington et al., 2014) for Yelp and the Chinese word embeddings trained on a large amount of Chinese news data for Chat when training the classifier.

We compare our method with BID17 which is the state-of-the-art language style transfer model with non-parallel data, and we name as Style Transfer Baseline (STB).

As described in Section 2 and 3, STB is built upon an auto-encoder framework.

It focuses on transferring sentences from one style to the other.

The text styles are represented by two embedding vectors.

It assumes source domain and target domain share a content space, and relies on adversarial training methods to align content spaces of two domains.

We keep the configurations of the modules in STB, such as the encoder, decoder and discriminator, the same as ours for a fair comparison.

Following BID17 , we use a model-based evaluation metric.

Specifically, we use a pretrained evaluation classifier to classify whether the transferred sentence has the correct style.

The classifier is implemented same as the discriminator D s .

Statistics of the data used for the evaluation classifier are shown in TAB0 in the Appendix.

The testing accuracy of evaluation classifiers is 95.36% for Yelp and 87.05% for Chat.

We repeat the training three times for each experiment setting and report the mean accuracy on the testing data with their standard deviation.

We first perform experiments on the source data containing both positive and negative reviews.

In this setting, we specifically compare two versions of both STB and our model, one with the cycle consistency loss and one without, to validate the effectiveness of the cycle consistency loss 1 .

Results are shown in TAB0 .

It can be seen that incorporating the cycle consistency loss improves the performance for both STB and our proposed model consistently.

We further manually examine the generated sentences for a detailed study of the various methods.

TAB1 shows a few samples for the above setting with 150k positive samples used.

Overall, our full model can generate grammatically correct positive reviews without changing the original content in more cases than the other methods.

For some simple sentences such as the first example, all models perform well.

For the second example in which the input sentence is more complex, both versions of STB and our basic model without the cycle consistency loss cannot generate fluent sentences, but our full model still succeeds.

However, our model also suffers some mistakes as shown in the third example.

Though it successfully makes the sentence positive, some additional information about the food is added, which is not discussed in the original sentence.

Original Sentence service has gotten worse and worse at this location .STB service is great for the family and family .

STB (with Cyc) service has always great and at this location .

Ours (without Cyc) service has been better than the best experience .

Ours service was super friendly and food was great .Next, we compare the results of STB and our proposed method in TAB0 .

As the number of positive sentences in the source data increases, the average performance of both versions of STB decreases drastically.

This is reasonable because STB introduces a discriminator to align the sentences from the target domain back to the source domain, and when the source domain contains more positive samples, it is hard to find a good alignment to the source domain.

Meanwhile the performance of our model, even the basic one without the cycle consistency loss, does not fluctuate much with the increase of the number of positive samples, showing that our model is not that sensitive to the source data containing more than one sentiments.

Overall, our model with the cycle consistency loss performs the best.

The above setting is not challenging enough, because we can use a pre-trained discriminator similar to D s in our model, to remove those samples classified as positive with high probabilities, so that only sentences with a less positive sentiment remain in the source domain.

Thus, we test our second dataset which combines neutral reviews and negative reviews as the source domain.

In this setting, in case that some positive sentences exist in those neutral reviews, when STB is trained, we use the same pre-trained discriminator in our model to filter out samples classified as positive with probabilities larger than 0.9.

In comparison, our model utilizes all the data, since it naturally allows for those data with styles similar to the target style.

In the following, we report and analyze both STB and our model with the cycle consistency loss added.

The experimental results in TAB2 show that STB (with Cyc) suffers a large performance drop with 150k neutral data mixed in the source domain, while our model remains stable.

In real applications, there may be only a small amount of data in the target domain.

To simulate this scenario, we limit the amount of the target data (randomly sampled from the positive data) used for training, and evaluate the robustness of the compared methods.

TAB3 shows the experimental results.

It is surprising to see that both methods obtain relatively steady accuracies with different numbers of target samples.

Yet, our model surpasses STB (with Cyc) in all the cases.

As in the Yelp experiment, we vary the number of target sentences to test the robustness of the compared methods.

The experimental results are shown in TAB4 .

As can be seen, STB (with Cyc) obtains a relatively low performance with only 10k target samples, and as more target samples are used, its performance increases.

However, the accuracy of our model is relatively high even with 10k target samples used, and remains stable in all the cases.

Thus, our model achieves better performance as well as stronger robustness on Chat.

A few examples are shown in Table 6 .

We can see that our model generally successfully transfers the sentence into a romantic style with some romantic phrases used.

Table 6 : Example sentences on Chat transferred into a romantic style.

English translations are provided (* denotes that the sentence has grammar mistakes in Chinese).Original Sentence 回眸一笑 就 好 It is enough to look back and smile STB (with Cyc) 回眸一笑 就 好 了 It would be just fine to look back and smile Ours 回眸一笑 , 勿念 。 Look back and smile, please do not miss me.

Original Sentence 得过且过 吧 !

Just live with it!

STB (with Cyc) 想不开 吧 , 我 的 吧 。

I just take things too hard.

* Ours 爱到深处 , 随遇而安 。

Love to the depths, enjoy myself wherever I am.

Original Sentence 自己 的 幸福 给 别人 了 Give up your happiness to others STB (with Cyc) 自己 的 幸福 给 别人 , 你 的 。 Give up your happiness to others.

* Ours 自己 的 幸福 是 自己 , 自己 的 。

Leave some happiness to yourself, yourself.

In this paper, we present an encoder-decoder framework for language style transfer, which allows for the use of non-parallel data and source data with various unknown language styles.

Each sentence is encoded into two latent representations, one corresponding to its content disentangled from the style and and the other representing the style only.

By recombining the content with the target style, we can decode a sentence aligned in the target domain.

Specifically, we propose two loss functions, i.e., the style discrepancy loss and the cycle consistency loss, to adequately constrain the encoding and decoding functions.

The style discrepancy loss is to enforce a properly encoded style representation while the cycle consistency loss is used to ensure that the style-transferred sentences can be transferred back to their original sentences.

Experimental results on two tasks demonstrate that our proposed method outperforms the state-of-the-art style transfer method BID17 We randomly select 200 test samples from Yelp and perform human evaluations on four aspects of the results: (1) content: estimates if the content of an input sentence is preserved in a transferred sentence; content rating has 0 (changed), 1 (synonym substitution or partially changed), and 2 (unchanged); (2) sentiment: estimates if the sentiment of a transferred sentence is consistent with the target sentiment; sentiment rating has 0 (unchanged and wrong), 1 (changed but wrong), 2 (correct); (3) fluency: estimates the fluency of transferred sentences; fluency is rated from 1 (unreadable) to 4 (perfect); (4) overall: estimates the overall quality of transferred sentences; overall rating ranges from 1 (failed) to 4 (perfect).We hired five annotators and averaged their evaluations.

TAB0 shows results on Yelp when the source domain contains not only negative sentences but also 150k positive sentences (row 3 in TAB0 ), and TAB0 shows results on Yelp when the target domain contains only 100k positive sentences ( row 1 in TAB3 ).

As can be seen, our model is better in terms of sentiment accuracy and overall quality, which is consistent with the automatic evaluation results.

We experiment on revising modern text in the language of Shakespeare at the sentence-level as in BID14 .

Following their experimental setup, we collect 29388 sentences authored by Shakespeare and 54800 sentences from non-Shakespeare-authored works.

The length of all the sentences ranges from 3 to 15.

Statistics of data for training and evaluating the style transfer model are shown in TAB0 , 14, and 15 in Section 6.1.

Since the dataset is small, we train the discriminator D s using a subset of the data for training the style transfer model.

The testing accuracy of D s is 87.6%.

The evaluation classifier has a testing accuracy 88.7%.Our model achieves a classification accuracy of 95.1% and STB with cycle consistency loss achieves 94.1%.

Following are some examples.

Compared with STB, our model can generate sentences which are more fluent and have a higher probability to have a correct target style.

However, we find that both STB and our model tend to generate short sentences and change the content of source sentences in more cases in this set of experiment than in the Yelp and Chat datasets.

We conjecture this is caused by the scarcity of training data.

Sentences in the Shakespeare's style form a vocabulary of 8559 words, but almost 60% of them appear less than 10 times.

On the other hand, the source domain contains 19962 words, but there are only 5211 common words in these two vocabularies.

Thus aligned words/phrases may not exist in the dataset.

<|TLDR|>

@highlight

We present an encoder-decoder framework for language style transfer, which allows for the use of non-parallel data and source data with various unknown language styles.