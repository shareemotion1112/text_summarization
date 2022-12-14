Standard image captioning tasks such as COCO and Flickr30k are factual, neutral in tone and (to a human) state the obvious (e.g., “a man playing a guitar”).

While such tasks are useful to verify that a machine understands the content of an image,  they are not engaging to humans as captions.

With this in mind we define a new task, Personality-Captions, where the goal is to be as engaging to humans as possible by incorporating controllable style and personality traits.

We collect and release a large dataset of 201,858 of such captions conditioned over 215 possible traits.

We build models that combine existing work from (i) sentence representations (Mazaré et al., 2018) with Transformers trained on 1.7 billion dialogue examples; and (ii) image representations (Mahajan et al., 2018) with ResNets trained on 3.5 billion social media images.

We obtain state-of-the-art performance on Flickr30k and COCO, and strong performance on our new task.

Finally, online evaluations validate that our task and models are engaging to humans, with our best model close to human performance.

If we want machines to communicate with humans, they must be able to capture our interest, which means spanning both the ability to understand and the ability to be engaging, in particular to display emotion and personality as well as conversational function BID17 BID18 BID41 BID19 .Communication grounded in images is naturally engaging to humans BID15 , and yet the majority of studies in the machine learning community have so far focused on function only: standard image captioning BID36 requires the machine to generate a sentence which factually describes the elements of the scene in a neutral tone.

Similarly, visual question answering BID2 and visual dialogue BID6 require the machine to answer factual questions about the contents of the image, either in single turn or dialogue form.

They assess whether the machine can perform basic perception over the image which humans take for granted.

Hence, they are useful for developing models that understand content, but are not useful as an end application unless the human cannot see the image, e.g. due to visual impairment BID13 .Standard image captioning tasks simply state the obvious, and are not considered engaging captions by humans.

For example, in the COCO BID5 and Flickr30k BID52 tasks, some examples of captions include "a large bus sitting next to a very tall building" and "a butcher cutting an animal to sell", which describe the contents of those images in a personality-free, factual manner.

However, humans consider engaging and effective captions ones that "avoid stating the obvious", as shown by advice to human captioners outside of machine learning.1 For example, "If the bride and groom are smiling at each other, don't write that they are smiling at each other.

The photo already visually shows what the subject is doing.

Rephrase the caption to reflect the story behind the image".

Moreover, it is considered that "conversational language works best.

Write the caption as though you are talking to a family member or friend".2 These instructions for human captioners to engage human readers seem to be in direct opposition to standard captioning datasets.

In this work we focus on image captioning that is engaging for humans by incorporating personality.

As no large dataset exists that covers the range of human personalities, we build and release a new dataset, PERSONALITY-CAPTIONS, with 201,858 captions, each conditioned on one of 215 Standard captioning output: A plate with a sandwich and salad on it.

Our model with different personality traits: Sweet That is a lovely sandwich.

This sandwich looks so delicious!

My goodness!

Anxious I'm afraid this might make me sick if I eat it.

Sympathetic I feel so bad for that carrot, about to be consumed.

Arrogant I make better food than this Optimistic It will taste positively wonderful!

Money-minded I would totally pay $100 for this plate.

Figure 1: Comparison of a standard captioning model compared to our TransResNet model's predictions on the same image conditioned on various personality traits.

Our model is trained on the new PERSONALITY-CAPTIONS dataset which covers 215 different personality traits.

The standard captioning system used for comparison is the best COCO UPDOWN model described in Section 4.2.

different possible personality traits.

We show that such captions are far more engaging to humans than traditional ones.

We then develop model architectures that can simultaneously understand image content and provide engaging captions for humans.

To build strong models, we consider both retrieval and generative variants, and leverage state-of-the-art modules from both the vision and language domains.

For image representations, we employ the work of BID28 that uses a ResNeXt architecture trained on 3.5 billion social media images which we apply to both.

For text, we use a Transformer sentence representation following BID32 ) trained on 1.7 billion dialogue examples.

Our generative model gives a new state-of-the-art on caption generation on COCO, and our retrieval architecture, TransResNet, yields the highest known hits@1 score on the Flickr30k dataset.

To make the models more engaging to humans, we then adapt those same architectures to the PERSONALITY-CAPTIONS task by conditioning the input image on the given personality traits, giving strong performance on our new task.

In particular, when compared to human captions, annotators preferred our retrieval model's captions over human ones 49.5% of the time, where the difference is not statistically significant.

A large body of work has focused on developing image captioning datasets and models that work on them.

In this paper we also perform experiments on the COCO BID5 and Flickr30k BID52 datasets, comparing to a range of models, including both generative models such as in BID45 BID49 BID1 and retrieval based such as in BID12 BID10 BID34 .

These setups measure the ability of models to understand the content of an image, but do not address more natural human communication.

A number of works have tried to induce more engaging captions for human readers.

One area of study is to make the caption personalized to the reader, e.g. by using user level features such as location and age BID7 or knowledge of the reader's active vocabulary BID38 .

Our work does not address this issue.

Another research direction is to attempt to produce amusing captions either through wordplay (puns) BID4 or training on data from humour websites BID50 .

Our work focuses on a general set of personality traits, not on humour.

Finally, closer to our work are approaches that attempt to model the style of the caption.

Some methods have tried to learn style in an unsupervised fashion, as a supervised dataset like we have built in this work was not available.

As a result, evaluation was more challenging in those works, see e.g. BID30 .

Others such as BID51 have used small datasets like SentiCap BID31 with ∼800 images to inject sentiment into captions.

BID11 collect a somewhat bigger dataset with 10,000 examples, FlickrStyle10K, but only covers two types of style (romantic and humorous).

In contrast, our models are trained on the PERSONALITY-CAPTIONS dataset that has 215 traits and ∼200,000 images.

Our work can also be linked to the more general area of human communication, separate from just factual captioning, in particular image grounded conversations between humans (Mostafazadeh .

In those tasks, simple word overlap based automatic metrics are shown to perform weakly BID24 due to the intrinsically more diverse outputs in the tasks.

As in those domains, we thus also perform human evaluations in this work to measure the engagingness of our setup and models.

In terms of modeling, image captioning performance is clearly boosted with any advancements in image or text encoders, particularly the former.

In this work we make use of the latest advancements in image encoding by using the work of BID28 which provides state-of-the-art performance on Imagenet image classification, but has so far not been applied to captioning.

For text encoding we use the latest advances in attention-based representations using Transformers BID42 ; in particular, their use in retrieval models for dialogue by large-scale pretraining (?) is adapted here for our captioning tasks.

The PERSONALITY-CAPTIONS dataset is a large collection of (image, personality trait, caption) triples that we collected using crowd-workers, and will be made publicly available upon acceptance.

We considered 215 possible personality traits which were constructed by selecting a subset from a curated list of 638 traits 3 that we deemed suitable for our captioning task.

The traits are categorized into three classes: positive (e.g., sweet, happy, eloquent, humble, perceptive, witty), neutral (e.g., old-fashioned, skeptical, solemn, questioning) and negative (e.g., anxious, childish, critical, fickle, frivolous).

Examples of traits that we did not use are allocentric, insouciant, flexible, earthy and invisible, due to the difficulty of their interpretation with respect to captioning an image.

We use a randomly selected set of the images from the YFFC100M Dataset 4 to build our training, validation and test sets, selecting for each chosen image a random personality trait from our list.

In each annotation round, an annotator is shown an image along with a trait.

The annotators are then asked to write an engaging caption for the image in the context of the personality trait.

It was emphasized that the personality trait describes a trait of the author of the caption, not properties of the content of the image.

See Section D in the appendix for the exact instructions given to annotators.

4 MODELS et al., 2016 ) trained on 3.5 billion Instagram pictures following the procedure described by BID28 , which we refer to in the rest of the paper as ResNeXt-IG-3.5B.

The authors provided the weights of their trained model to us.

Both networks embed images in a 2048-dimensional vector which is the input for most of our models.

In some of the caption generation models that make use of attention, we keep the spatial extent of the features by adapting activation before the last average pooling layer, and thus extract features with 7 × 7 × 2048 dimensions.

We re-implemented three widely used previous/current state-of-the-art methods BID45 BID49 BID1 for image captioning as representatives of caption generation models.

We refer them as SHOWTELL, SHOWATTTELL and UPDOWN respectively.

We extract the image representation r I using the aforementioned image encoders.

The SHOWTELL model uses image features with 2048 dimensions and the other models use image features with 7 × 7 × 2048 dimensions.

In the case where we augment our models with personality traits, we learn an embedding for each trait, which is concatenated with each input of the decoder.

Caption Decoders The SHOWTELL model first applies a linear projection to reduce image features into a feature vector with 512 dimensions.

Similar to BID45 , this embedding is the input for a LSTM model that generates the output sequence.

In SHOWATTTELL, while the overall architecture is similar to BID49 , we adopt the modification suggested by BID39 and input the attention-derived image features to the cell node of the LSTM.

Finally, we use the UPDOWN model exactly as described in BID1 .

We perform a two-stage training strategy to train such caption generation models as proposed by BID39 .

In the first stage, we train the model to optimize the standard cross-entropy loss.

In the second stage, we perform policy gradient with REINFORCE to optimize the non-differentiable reward function (CIDEr score in our case).

During inference, we apply beam search (beam size=2) to decode the caption.

We define a simple yet powerful retrieval architecture, named TransResNet.

It works by projecting the image, personality, and caption in the same space S using image, personality, and text encoders.

Image and Personality Encoders The representation r I of an image I is obtained by using the 2048-dimensional output of the image encoder described in Sec. 4.1 as input to a multi-layer perceptron with ReLU activation units and a final layer of 500 dimensions.

To take advantage of personality traits in the PERSONALITY-CAPTIONS task, we embed each trait to a 500-dimensional vector to obtain its representation r P .

Image and personality representations are then summed.

Caption Encoders Each caption is encoded into a vector r C of the same size using a Transformer architecture BID42 , followed by a two layer perceptron.

We try two sizes of Transformer: a larger architecture (4 layers, 300 hidden units, 6 attention heads) and a smaller one (2 layers, 300 hidden units, 4 attention heads).

We consider either training from scratch or pretraining our models.

We either pretrain only the word embeddings, i.e. where we initialize word vectors trained using fastText BID3 ) trained on Wikipedia, or pretrain the entire encoder.

For the latter, we follow the setup described in BID32 : we train two encoders on a next-utterance retrieval task on a dataset of dialogs containing 1.7 billion pairs of utterances, where one encodes the context and another the candidates for the next utterance, their dot product indicates the degree of match, and they are trained with negative log-likelihood and k-negative sampling.

We then initialize our system using the weights of the candidate encoder only, and then train on our task.

For comparison, we also consider a simple bag-of-words encoder (pretrained or not).

In this case, r C is the sum of the 300-dimensional word embeddings of the caption.

In each case, given an input image and personality trait (I, P ) and a candidate caption C, the score of the final combination is then computed as s(I, P, C) = (r I + r P ) · r C .

Figure 2: Our architecture TransResNet, used for our retrieval models.

Training and Inference Given a pair I, P , and a set of candidates (c 1 , .., c N ), at inference time the predicted caption is the candidate c i that maximizes the score s(I, P, c i ).

At training time we pass a set of scores through a softmax and train to maximize the log-likelihood of the correct responses.

We use mini-batches of 500 training examples; for each example, we use the captions of the other elements of the batch as negatives.

Our overall TransResNet architecture is detailed in Figure 2 .

We first test our architectures on traditional caption datasets to assess their ability to factually describe the contents of images in a neutral tone.

We then apply the same architectures to PERSONALITY-CAPTIONS to assess their ability to produce engaging captions conditioned on personality.

The latter is tested with both automatic metrics and human evaluation of engagingness.

Generative Models For our generative models, we test the quality of our implementations of existing models (SHOWTELL, SHOWATTTELL and UPDOWN) as well as the quality of our image encoders, where we compare ResNet152 and ResNeXt-IG-3.5B.

We report performance on the COCO caption dataset BID23 .

We evaluate BLEU BID37 , ROUGE-L BID22 , CIDEr and SPICE BID0 and compare model's performances to state-of-the-art models under BID20 's setting.

The results are shown in TAB3 .

Models trained with ResNeXt-IG-3.5B features consistently outperform their counterparts with ResNet152 features, demonstrating the effectiveness of ResNeXt-IG-3.5B beyond the original image classification and detection results in BID28 .

More importantly, our best model (UPDOWN) either outperforms or is competitive with state-ofthe-art single model performance BID1

Retrieval Models We compare our retrieval architecture, TransResNet, to existing models reported in the literature on the COCO caption and Flickr30k tasks.

We evaluate retrieval metrics R@1, R@5, R@10, and compare our model performance to state-of-the-art models under the setting of BID20 ).

The results are given in Table 4 (for more details, see TAB0 in the appendix for COCO and Flickr30k, respectively).

For our model, we see large improvements using ResNeXt-IG-3.5B compared to Resnet152, and stronger performance with a Transformer-based text encoding compared to a bag-of-words encoding.

Pretraining the text encoder also helps substantially (see Appendix A for more analysis of pretraining of our systems).

Our best models are competitive on COCO and are state-of-the-art on Flickr30k by a large margin (68.4 R@1 for our model vs. 56.8 R@1 for the previous state-of-the-art).

Generative models We first train the aforementioned caption generation models without using the personality traits.

This setting is similar to standard image captioning, and TAB5 shows that the three caption generation models that we considered are ranked in the same order, with the UPDOWN model being the most effective.

The best results are again obtained using the ResNeXt-IG-3.5B features.

Adding the embedding of the personality trait allows our best model to reach a CIDEr score of 22.0, showing the importance of modeling personality in our new task.

Note that all scores are lower than for the COCO captioning task.

Indeed standard image captioning tries to produce text descriptions that are semantically equivalent to the image, whereas PERSONALITY-CAPTIONS captures how a human responds to a given image when speaking to another human when both can see the image -which is rarely to simply state its contents.

Hence, PERSONALITY-CAPTIONS has intrinsically more diverse outputs, similar to results found in other human communication tasks BID24 .

For that reason we perform human evaluation in Section 5.3 in addition to automatic evaluations.

Retrieval models Similarly we compare the effect of various configurations of our retrieval model, TransResNet.

The models are evaluated in terms of R@1, where for each sample there are 100 candidates to rank: 99 randomly chosen candidates from the test set plus the true label.

TAB6 shows the scores obtained on the test set of PERSONALITY-CAPTIONS.

Again, the impact of using the image encoder trained on billions of images is considerable, we obtain 53.5% for our best ResNeXt-IG-3.5B model, and 34.4% for our best Resnet152 model.

Conditioning on the personality traits is also very important (53.5% vs. 38.5% R@1 for the best variants with and without conditioning).

Transformer text encoders also outperform bag-of-word embeddings encoders, Table 4 : Retrieval model performance on Flickr30k and COCO caption using the splits of BID20 where pretraining for either type of encoder helps.

For Transformers pretraining the whole network performed better than just pretraining the word embeddings, see Appendix A.Example predictions of our best model, TransResNet (ResNeXt-IG-3.5B), are given in TAB2 .

The goal of PERSONALITY-CAPTIONS is to be engaging to human readers by emulating human personality traits.

We thus test our task and models in a set of human evaluation studies.

Evaluation Setup Using 500 random images from the YFCC-100M dataset that are not present in PERSONALITY-CAPTIONS, we obtain captions for them using a variety of methods, as outlined in the sections below, including both human authored captions and model predicted captions.

Using a separate set of human annotators, comparisons are then done pairwise: we show each image, with two captions to compare, to five separate annotators and ask them to choose the "more engaging" caption.

For experiments where both captions are conditioned on a personality, we show the annotator the personality; otherwise, the personality is hidden.

We then report the percentage of the time one method is chosen over the other.

The results are summarized in FIG0 .

We compare human authored PERSONALITY-CAPTIONS captions to human authored traditional neutral (COCO-like) captions.

Captions conditioned on a personality were found to be significantly more engaging than those that were neutral captions of the image, with a win rate of 64.5%, which is statistically significant using a binomial two-tailed test.

We compare the best-performing models from Section 5.2 to human authored PERSONALITY-CAPTIONS captions.

For each test image we condition both human and model on the same (randomly-chosen) personality trait.

Our best TransResNet model from Sec.

We also compare our models in a pairwise fashion directly, as measured by human annotators.

The results given in FIG0 (all statistically significant) show the same trends as we observed before: TransResNet with ResNext-IG-3.5B outperforms the same model with ResNet152 features with a win rate of 55.2%, showing the importance of image features.

Additionally, TransResNetwith ResNext-IG-3.5B image features (with no pretraining) also substantially outperforms the UPDOWN model using ResNext-IG-3.5B with a winrate of 80.1%.

In this work we consider models that can simultaneously understand image content and provide engaging captions for humans.

To build strong models, we first leverage the latest advances in image and sentence encoding to create generative and retrieval models that perform well on standard image captioning tasks.

In particular, we attain a new state-of-the-art on caption generation on COCO, and introduce a new retrieval architecture, TransResNet, that yields the highest known hits@1 score on the Flickr30k dataset.

To make the models more engaging to humans, we then condition them on a set of controllable personality traits.

To that end, we collect a large dataset, PERSONALITY-CAPTIONS to train such models.

Using automatic metrics and human evaluations, we show that our best system is able to produce captions that are close to matching human performance in terms of engagement.

Our benchmark will be made publicly available to encourage further model development, leaving the possibility of superhuman performance coming soon in this domain.

A IMPACT OF PRETRAINED WORD EMBEDDINGS AND TEXT ENCODERS Table 7 : More detailed results for retrieval model performance on COCO Captions using the splits of BID20 .

For our TransResNet models, we compare two types of pretraining: Full indicates a model with a pretrained text encoder, while Word indicates a model with pretrained word embeddings only.

Caption retrieval Pretraining R@1 R@5 R@10 Med Rank 1k Images m-CNN BID27 42.8 -84.1 2.0 UVS BID21 43.4 75.7 85.8 2.0 HM-LSTM 43.9 -87.8 2.0 Order Embeddings BID44 46.7 -88.9 2.0 Embedding Net BID46 50.4 79.3 69.4 -DSPE+Fisher Vector 50.1 -89.2 -sm-LSTM BID16 53.2 83.1 91.5 1.0 VSE++ (ResNet, FT) BID10 64.6 90.0 95.7 1.0 GXN (i2t+t2i) BID12 68.

BID44 23.3 -65.0 5.0 VSE++ (ResNet, FT) BID10 41.3 71.1 81.2 2.0 GXN (i2t+t2i) BID12 42.

Table 8 : Retrieval model performance on Flickr30k using the splits of BID20 .

For our models, we compare two types of pretraining: Full indicates a model with a pretrained text encoder, while Word indicates a model with pretrained word embeddings only.

Caption retrieval Pretraining R@1 R@5 R@10 Med Rank UVS BID21 23.0 50.7 62.9 5.0 UVS (Github)

29.8 58.4 70.5 4.0 Embedding Net BID46 40.7 69.7 79.2 -DAN BID34 41.4 73.5 82.5 2.0 sm-LSTM BID16 42.5 71.9 81.5 2.0 2WayNet BID8 49.8 67.5 --VSE++ (ResNet, FT) BID10 52.9 80.5 87.2 1.0 DAN (ResNet) BID34 55.0 81.8 89.0 1.0 GXN (i2t+t2i) BID12 56.

Engaging-only Captions Instead of asking to author a caption based on a personality trait, we can ask humans to simply write an "engaging" caption instead, providing them with no personality cue.

We found that human annotators overall preferred captions written by those unconditioned on a personality by a slight margin (∼ 54%).

To further understand this difference, we split the images into three subsets based on the personality on which the PERSONALITY-CAPTIONS annotator conditioned their caption, i.e. whether the personality was positive, negative, or neutral.

We then examined the engagingness rates of images for each of these subsets.

In the set where PERSONALITY-CAPTIONS annotators were provided with positive personalities, which totaled 185 out of the 500 images, we found that human annotators preferred the captions conditioned on the personality to those that were not.

However, in the other two sets, we found that the unconditioned captions were preferred to the negative or neutral ones.

For these two subsets, we believe that, without the context of any personality, annotators may have preferred the inherently more positive caption provided by someone who was asked to be engaging but was not conditioned on a personality.

Diversity of captions We found that the captions written via our method were not only more engaging for positive personality traits, but also resulted in more diversity in terms of personality traits.

To measure this diversity, we constructed a model that predicted the personality of a given comment.

The classifier consists in the same Transformer as described in 4.3, pre-trained on the same large dialog corpus, followed by a softmax over 215 units.

We then compare the total number of personality types as predicted by the classifier among each type of human-labeled data: "engaging" captions conditioned on personalities, "engaging" captions not conditioned on personalities, and traditional image captions.

That is, we look at each caption given by the human annotators, assign it a personality via the classifier, and then look at the total set of personalities we have at the end for each set of human-labeled data.

For example, out of the 500 human-generated traditional captions, the classifier found 63% of all possible positive personalities in this set of captions.

As indicated in TAB0 , the human annotators who were assigned a personality produce more diverse captions, particularly negatively and neutrally conditioned ones, as compared to human annotators who are just told to be "engaging" or those who are told to write an image caption.

The ultimate test of our generative and retrieval models on PERSONALITY-CAPTIONS is performed using human evaluations.

Comparing them using automatic metrics is typically difficult because retrieval methods perform well with ranking metrics they are optimized for and generative models perform well with word overlap metrics they are optimized for, but neither of these necessarily correlate with human judgements, see e.g. .Nevertheless, here we compare our generative and retrieval models directly with automatic metrics on COCO.

We computed the BLEU, CIDEr, SPICE, and ROUGE-L scores for our best TransResNet model.

The comparison is given in TAB0 .

TAB0 : Generative and retrieval model performance on COCO caption using the test split of BID20 That is so cool!

I I love street art!

OptimisticThe future is bright for people who can dream in artistic ways.

Critical I do believe this taggers verbage is a tad junvenile Charming What a charming wall.

Adventurous I think I could create art like that, I will go learn and take action.

The color of this flower is absolutely astounding.

I can't believe it.

Wishful I always wish I could grow these types of flowers.

Sweet Beautiful flowers!

I would give them to you.

RomanticThe pink flowers would make a beautiful bouquet for my wife.

Oh my, what a lovely purple color of nature's new sprouts!

TAB0 : More example predictions from our best TRANSRESNET model on the PERSONALITY-CAPTIONS validation set.

<|TLDR|>

@highlight

We develop engaging image captioning models conditioned on personality that are also state of the art on regular captioning tasks.