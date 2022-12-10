In this paper, we present a method for adversarial decomposition of text representation.

This method can be used to decompose a representation of an input sentence into several independent vectors, where each vector is responsible for a specific aspect of the input sentence.

We evaluate the proposed method on two case studies: the conversion between different social registers and diachronic language change.

We show that the proposed method is capable of fine-grained con- trolled change of these aspects of the input sentence.

For example, our model is capable of learning a continuous (rather than categorical) representation of the style of the sentence, in line with the reality of language use.

The model uses adversarial-motivational training and includes a special motivational loss, which acts opposite to the discriminator and encourages a better decomposition.

Finally, we evaluate the obtained meaning embeddings on a downstream task of para- phrase detection and show that they are significantly better than embeddings of a regular autoencoder.

Despite the recent successes in using neural models for representation learning for natural language text, learning a meaningful representation of input sentences remains an open research problem.

A variety of approaches, from sequence-to-sequence models that followed the work of BID37 to the more recent proposals BID2 BID29 BID8 BID25 BID36 BID5 share one common drawback.

Namely, all of them encode the input sentence into just one single vector of a fixed size.

One way to bypass the limitations of a single vector representation is to use an attention mechanism BID3 BID40 .

We propose to approach this problem differently and design a method for adversarial decomposition of the learned input representation into multiple components.

Our method encodes the input sentence into several vectors, where each vector is responsible for a specific aspect of the sentence.

In terms of learning different separable components of input representation, our work most closely relates to the style transfer work, which has been applied to a variety of different aspects of language, from diachronic language differences BID42 to authors' personalities BID24 and even sentiment BID17 BID13 .

The style transfer work effectively relies on the more classical distinction between meaning and form BID9 , which accounts for the fact that multiple surface realizations are possible for the same meaning.

For simplicity, we will use this terminology throughout the rest of the paper.

Consider the case when we encode an input sentence into a meaning vector and a form vector.

We are then able to perform a controllable change of meaning or form by a simple change applied to these vectors.

For example, we can encode two sentences written in two different styles, then swap the form vectors while leaving the meaning vectors intact.

We can then generate new unique sentences with the original meaning, but written in a different style.

In the present work, we propose a novel model for this type of decomposition based on adversarialmotivational training and design an architecture inspired by the GANs BID14 and adversarial autoencoders BID26 .

In addition to the adversarial loss, we use a special motivator BID0 , which, in contrast to the discriminator, is used to provide a motivational loss to encourage the model to better decomposition of the meaning and the form, as well as specific aspects of meaning.

We make all the code publicly available on GitHub 1 .We evaluate the proposed methods for learning separate aspects of input representation on the following case studies:1.

Learning to separate out a representation of the specific diachronic slice of language.

One may express the same meaning using the Early Modern English (e.g. What would she have?) and the contemporary English ( What does she want?)2.

Learning a representation for a social register BID16 -that is, subsets of language appropriate in a given context or characteristic of a certain group of speakers.

These include formal and informal language, the language used in different genres (e.g., fiction vs. newspapers vs. academic texts), different dialects, and even literary idiostyles.

We experiment with the registers corresponding to the titles of scientific papers vs. newspaper articles.

As mentioned above, the most relevant previous work comes from the style transfer research, and it can be divided into two groups:1.

Approaches that aim to generate text in a given form.

For example, the task may be to produce just any verse as long as it is in the "style" of the target poet.2.

Approaches that aim to induce a change in either the "form" or the "meaning" of an existing utterance.

For example, "Good bye, Mr. Anderson." can be transformed to "Fare you well, good Master Anderson" BID42 ).An example of the first group is the work by BID31 , who trained several separate networks on verses by different hip-hip artists.

An LSTM network successfully generated verses that were stylistically similar to the verses of the target artist (as measured by cosine distance on TfIdf vectors).

More complicated approaches use language models that are conditioned in some way.

For example, BID24 produced product reviews with a target rating by passing the rating as an additional input at each timestep of an LSTM model.

BID38 generated reviews not only with a given rating but also for a specific product.

At each timestep a special context vector was provided as input, gated so as to enable the model to decide how much attention to pay to that vector and the current hidden state.

BID23 used "speaker" vectors as an additional input to a conversational model, improving consistency of dialog responses.

Finally, BID12 performed an extensive evaluation of conditioned language models based on "content" (theme and sentiment) and "style" (professional, personal, length, descriptiveness) .

Importantly, they showed that it is possible to control both "content" and "style" simultaneously.

Work from the second group can further be divided into two clusters by the nature of the training data: parallel aligned corpora, or non-aligned datasets.

The aligned corpora enable approaching the problem of form shift as a paraphrasing or machine translation problem.

BID42 used statistical and dictionary-based systems on a dataset of original plays by Shakespeare and their contemporary translations.

BID4 trained an LSTM network on 33 versions of the Bible.

BID19 used a Pointer Network BID41 , an architecture that was successfully applied to a wide variety of tasks BID27 BID15 BID32 , to enable direct copying of the input tokens to the output.

Note that these works use BLEU BID30 as the main, or even the only evaluation measure.

This is only possible in cases where a parallel corpus is available.

Recently, new approaches that do not require a parallel corpora were developed in both CV and NLP.

BID17 succeeded in changing tense and sentiment of sentences with a two steps procedure based on a variational auto-encoder (VAE) BID21 .

After training a VAE, a discriminator and a generator are trained in an alternate manner, where the discriminator tries to correctly classify the target sentence attributes.

A special loss component forces the hidden representation of the encoded sentence to not have any information about the target sentence attributes.

BID28 used a VAE to produce a hidden representation of a sentence, and then modify it to match the desired form.

Unlike BID17 , they do not separate the form and meaning embeddings.

BID34 applied a GAN to align the hidden representation of sentences from two corpora and force them to do not have any information about the form via adversarial loss.

During the decoding, similarly the work by BID24 , special "style" vectors are passed to the decoder at every timestep to produce a sentence with the desired properties.

The model is trained using the Professor-Forcing algorithm BID22 .

BID20 worked directly on hidden space vectors that are constrained with the same adversarial loss instead of outputs of the generator, and use two different generators for two different "styles".

Finally, BID13 proposed two models for generating sentences with the target properties using an adversarial loss, similarly to BID34 and BID20 .Comparison with previous work In contrast to the proposals of BID42 , BID4 , BID19 , our solution does not require a parallel corpus.

Furthermore, unlike the model by BID34 , our model works directly on representation of sentences in the hidden space.

Most importantly, in contrast to the proposals by BID28 , BID17 , BID20 , BID13 , our model produces a representation for both meaning and form and does not treat the form as a categorical (in the vast majority of works, binary) variable.

Although the form was represented as dense vectors in previous work, it is still just a binary feature, as they use a single pre-defined vector for each form, with all sentences of the same form assigned the same form vector.

In contrast, our work treats form as a truly continuous variable, where each sentence has its own, unique, form vector.

Treating meaning and form not as binary/categorical, but as continuous is more consistent with the reality of language use, since there are different degrees of overlap between the language used by different registers or in different diachronic slices.

Indeed, language change is gradual, and the acceptability of expressions in a given register also forms a continuum, so one expects a substantial overlap between the grammar and vocabulary used, for example, on Twitter and by New York Times.

To the best of our knowledge, this is the first model that considers linguistic form in the task of text generation as a continuous variable.

One significant consequence of learning a continuous representation for form is that it allows the model to work with a large, and potentially infinite, number of forms.

Note that in this case the locations of areas of specific forms in the vector style space would reflect the similarity between these forms.

For example, the proposed model could be directly applied to the authorship attribution problem.

In this case, each author would have their own area in the form space, and the more similar the authors are in terms of writing style, the closer these areas would be to each other.

We performed preliminary experiments on this and report the results in Appendix A.

Let us formulate the problem of decomposition of text representation on an example of controlled change of linguistic form and conversion of Shakespeare plays in the original Early Modern to contemporary English.

Let X a be a corpus of texts DISPLAYFORM0 and X b be a corpus of texts DISPLAYFORM1 We assume that the texts in both X a and X b has the same distribution of meaning m ∈ M. The form f , however, is different and generated from a mixture of two distributions: DISPLAYFORM2 where f a and f b are two different languages (Early Modern and contemporary English).

Intuitively,we say that a sample x i has the form f a if α .

The goal of dissociation meaning and form is to learn two encoders E m : X → M and E f : X → F for the meaning and form correspondingly, and the generator G : M, F → X such that ∀j ∈ {a, b}, ∀k ∈ {a, b} : DISPLAYFORM3 That is, the form of a generated sample depends exclusively on the provided f j and can be the in the same domain for two different m u and m v from two samples from different domains X a and X b .Note that, in contrast to the previously proposals, the form f is not a categorical variable but a continuous vector.

This enables fine-grained controllable change of form: the original form f i is changed to reflect the form of the specific target sentence f j with its own unique α a and α b while preserving the original meaning m i .An important caveat concerns the core assumption of the similar meaning distribution in the two corpora, which is also made in all other works reviewed in Section 2.

It limits the possible use of this approach to cases where the distributions are in fact similar (i.e. parallel or at least comparable corpora are available).

It does not apply to many cases that could be analyzed in terms of meaning and form.

For example, books for children and scholarly papers are both registers, they have their own form (i.e. specific subsets of linguistic means and structure conventions) -but there is little overlap in the content.

This would make it hard even for a professional writer to turn a research paper into a fairy tale.

Encoder encodes the inputs sentences into two latent vectors m and f .

The Generator takes them as the input and produces the output sentence.

During the training, the Discriminator is used for an adversarial loss that forces m to do not carry any information about the form, and the M otivator is used for a motivational loss that encourages f to carry the needed information about the form.

Our solution is based on a widely used sequence-to-sequence framework BID37 and consists of four main parts.

The encoder E encodes the inputs sequence x into two latent vectors m and f which capture the meaning and the form of the sentence correspondingly.

The generator G then takes these two vectors as the input and produces a reconstruction of the original input sequencex.

The encoder and generator by themselves will likely not achieve the dissociation of the meaning and form.

We encourage this behavior in a way similar to Generative Adversarial Networks (GANs) BID14 , which had an overwhelming success the past few years and have been proven to be a good way of enforcing a specific distribution and characteristics on the output of a model.

Inspired by the work of BID0 and the principle of "carrot and stick" BID33 , in contrast to the majority of work that promotes pure adversarial approach BID14 BID34 BID13 , we propose two additional components, the discriminator D and the motivator M to force and motivate the model to learn the dissociation of the meaning and the form.

Similarly to a regular GAN model, the adversarial discriminator D tries to classify the form f based on the latent meaning vector m, and the encoder E is penalized to make this task as hard as possible.

Opposed to such vicious behaviour, the motivator M tries to classify the form based on the latent form vector f , as it should be done, and encourages the encoder E to make this task as simple as possible.

We could apply the adversarial approach here as well and force the distribution of the form vectors to fit a mixture of Gaussians (in this particular case, a mixture of two Guassians) with another discriminator, as it is done by BID26 , but we opted for the "dualistic" path of two complimentary forces.

Both the encoder E and the generator G are modeled with a neural network.

Gated Recurrent Unit (GRU) BID6 ) is used for E to encode the input sentence x into a hidden vector h = GRU(x).The vector h is then passed through two different fully connected layers to produce the latent vectors of the form and the meaning of the input sentence: DISPLAYFORM0 We use θ E to denote the parameters of the encoder E: W m , b m , W f , b f , and the parameters of the GRU unit.

The generator G is also modelled with a GRU unit.

The generator takes as input the meaning vector m and the form vector f , concatenates them, and passes trough a fully-connected layer to obtain a hidden vector z that represents both meaning and form of the original input sentence: DISPLAYFORM1 After that, we use a GRU unit to generate the output sentence as a probability distribution over the vocabulary tokens: DISPLAYFORM2 We use θ G to denote the parameters of the generator G: W z , b m , and the parameters of the used GRU.

The encoder and generator are trained using the standard reconstruction loss: DISPLAYFORM3

The representation of the meaning m produced by the encoder E should not contain any information about the form f .

We achieve this by using an adversarial approach.

First, we train a discriminator D, consisting of several fully connected layers with ELU activation function BID7 between them, to predict the form f of a sentence by its meaning vector:f D = D(m), wheref is the score (logit) reflecting the probability of the sentence x to belong to one of the form domains.

Motivated by the Wasserstein GAN BID1 , we use the following loss function instead of the standard cross-entropy: DISPLAYFORM0 Thus, a successful discriminator will produce negative scoresf for sentences from X a and positive scores for sentences from X b .

This discriminator is then used in an adversarial manner to provide a learning signal for the encoder and force dissociation of the meaning and form by maximizing L D : L adv (θ E ) = −λ adv L D , where λ adv is a hyperparameter reflecting the strength of the adversarial loss.

Note that this loss applies to the parameters of the encoder.

Our experiments showed that it is enough to have just the discriminator D and the adversarial loss L adv to force the model to dissociate the form and the meaning.

However, in order to achieve a better dissociation, we propose to use a motivator M (Albanie et al., 2017) and the corresponding motivational loss.

Conceptually, this is the opposite of the adversarial loss, hence the name.

As the discriminator D, the motivator M learns to classify the form f of the input sentence.

However, its input is not not the meaning vector but the form vector: DISPLAYFORM0 The motivator has the same architecture as the discriminator, and the same loss function.

While the adversarial loss forces the encoder E to produce a meaning vector m with no information about the form f , the motivational loss encourages E to encode this information in the form vector by minimizing DISPLAYFORM1

The overall training procedure follows the methods for training GANs BID14 BID1 and consists of two stages: training the discriminator D and the motivator M , and training the encoder E and the generator G.In contrast to BID1 , we do not train the D and M more than the E and the G. In our experiments we found that simple training in two stages is enough to achieve dissociation of the meaning and the form.

Encoder and generator are trained with the following loss function that combines reconstruction loss with the losses from the discriminator and the motivator: DISPLAYFORM0 Similarly to the evaluation of style transfer in CV ), evaluation of this task is difficult.

We follow the approach of ; BID34 and recently proposed by BID13 methods of evaluation of "transfer strength" and "content preservation".

The authors showed the proposed automatic metrics to a large degree correlate with human judgment and can serve as a proxy.

Below we give an overview of these metrics.

Transfer Strength.

The goal of this metric is to capture whether the form has been changed successfully.

To do that, a classifier C is trained on the two corpora, X a and X b to recognize the linguistic "form" typical of each of them.

After that, a sentence the form/meaning of which was changed is passed to the classifier.

The overall accuracy reflects the degree of success of changing the form/meaning.

This approach is widely used in CV , and was applied in NLP as well BID34 .In our experiments we used a GRU unit followed by four fully-connected layers with ELU activation functions between them as the classifier.

Content preservation Note that transfer strength by itself does not capture the overall quality of a changed sentence.

A extremely overfitted model that produces the same, the most characteristic sentence of one corpus all the time would have a high score according to this metric.

Thus, we need to measure how much of the meaning was preserved while changing the form.

To do that, BID13 proposed to use a cosine similarity based metric using pretrained word embeddings.

First, a sentence embedding is computed by concatenation of max, mean, and average pooling over the timesteps: DISPLAYFORM1 Next, the cosine similarity score s i between the embedding v s i of the original source sentence and the target sentence with the changed form v t i is computed, and the scores across the dataset are averaged to obtain the total score: DISPLAYFORM2 The metrics described above treat the form as a categorical (in most cases, even binary) variable.

This was not a problem in previous work since the change of form could be done by just inverting the form vector.

Our work, in contrast, treats the form as a continuous variable, and, therefore, we cannot just use the proposed metrics directly.

To enable a fair comparison, we propose the following procedure.

For each sentence s a s in the test set from the corpus X a we sample k = 10 random sentence from the corpus X b of the opposite form.

After that, we encode them into the meaning m i and form DISPLAYFORM3 We then generate a new sentence with its original meaning vector m s and the resulting form vector f avg , and use it for evalation.

This process enables a fair comparison with the previous works that treat form as a binary variable.

We performed an extensive evaluation of the proposed method on several dataset that reflect different changes of meaning, form, or specific aspects of meaning, such as sentiment polarity.

Changing form: register This experiment is conducted with a dataset of titles of scientific papers and news articles published by BID13 .

This dataset (referred to as "Headlines") contains titles of scientific articles crawled from online digital libraries, such as "ACM Digital Library" and "arXiv".

The titles of the news articles are taken from the "News Aggregator Data Set" from UCI Machine Learning Repository BID10 Changing form: language diachrony Diachronic language change is explored with the dataset composed by BID42 .

It includes the texts of 17 plays by William Shakespeare in the original Early Modern English, and their translations into contemporary English.

We randomly permuted all sentences from all plays and sampled the training, validation, and test sets.

Note that this is the smallest dataset in our experiments.

Previous work on style transfer for text also included the experiments with changing sentiment polarity BID34 BID13 .

We do not report the experiments with sentiment data, since the change in sentiment polarity corresponds to a change in a specific aspect of meaning, rather than form.

We therefore believe the comparison with these data would not be instructive.

Probably, the most recent and similar to our work is the model proposed by BID13 , in particular the "style-embedding" model.

We implemented this model to provide a baseline for comparison.

The classifier used in the transfer strength metric achieves very high accuracy (0.832 and 0.99 for the Shakespeare and Headlines datasets correspondingly).

These results concur with the results of BID34 and BID13 , and show that the two forms in the corpora are significantly different.

Following BID13 , we show the result of different configuration of the size of the form and meaning vectors on FIG2 .

Namely, we report combinations of 64 and 256-dimensional vectors.

Note that the sizes of the form vector are important.

The larger is the form vector, the higher is the transfer strength, but smaller is content preservation.

This is consistent with BID13 , where they observed a similar behaviour.

It is clear that the proposed method achieves significantly better transfer strength then the previously proposed model.

It also has a lower content preservation score, which means that it repeats fewer exact words from the source sentence.

Note that a low transfer strength and very high (0.9) content preservation score means that the model was not able to successfully learn to transfer the form and the target sentence is almost identical to the source sentence.

The Shakespeare dataset is the hardest for the model in terms of transfer strength, probably because it is the smallest dataset, but the proposed method performs consistently well in transfer of both form and meaning and, in contrast to the baseline.

Fluency of generated sentences Note that there is no guarantee that the generated sentences would be coherent after switching the form vector.

In order to estimate how this switch affects the fluency of generated sentences, we trained a language model on the Shakespeare dataset and calculated the perplexity of the generated sentences using the original form vector and the average of form vectors of k random sentences from the opposite style (see subsubsection 5.1.1).

While the perplexity of such sentences does go up, this change is not big (6.89 vs 9.74).

To investigate the impact of the motivator, we visualized form and meaning embeddings of 1000 random samples from the Headlines dataset using t-SNE algorithm BID39 with the Multicore-TSNE library (Ulyanov, 2016) .

The result is presented in FIG3 .There are three important observations.

First, there is no clear separation in the meaning embeddings, which means that any accurate form transfer is due to the form embeddings, and the dissociation of form and meaning was successful.

Second, even without the motivator the model is able to produce the form embeddings that are clustered into two group.

Recall from section 4 that without the motivational loss there are no forces that influence the form embeddings, but nevertheless the model learns to separate them.

However, the separation effect is much more pronounced in the presence of motivator.

This explains why the motivator consistently improved transfer strength of ADNet, as shown in FIG2 .

6.2 QUALITATIVE EVALUATION Table 1 and Table 2 show several examples of the successful form/meaning transfer achieved by ADNet.

Table 1 presentes the results of an experiment that to some extent replicates the approach taken by the authors who treat linguistic form as a binary variable BID34 BID13 .

The sentences the original Shakespeare plays were averaged to get the "typical" Early Modern English form vector.

This averaged vector was used to decode a sentence from the modern English translation back into the original.

The same was done in the opposite direction.

→

This man will tell us everything. (EME) I've done no more to caesar than you will do to me. (CE)

→

I have done no more to caesar than, you shall do to me. (EME) Table 1 : Decoding of the source sentence from Early Modern English (EME) into contemporary English (CE), and vice versa.

Table 2 illustrates the possibilities of ADNet on fine-grained transfer applied to the change of register.

We encoded two sentences in different registers from the Headlines dataset to produce form and meaning embeddings, and then we decoded the first sentence with the meaning embedding of the second, and vice versa.

As can be seen from Table 2 , the model correctly captures the meaning of sentences and decodes them using the form of the source sentences.

Note how the model preserves specific words and the structure of the source sentence.

In particular, note how in the first example, the model decided to put the colon after the "crisis management", as the source form sentence has this syntactic structure ("A review:").

This is not possible in the previously proposed models, as they treat form as just a binary variable.

A review: detection techniques for LTE system Crisis management: media practices in telecommunication management Situation management knowledge from social media A review study against intelligence internet Security flaw could not affect digital devices, experts say Semantic approach approach: current multimedia networks as modeling processes Semantic approach to event processing Security flaw to verify leaks Table 2 : Flipping the meaning and the form embeddings of two sentence from different registers.

Note the use of the colon in the first example, and the use of the "to"-constructions in the second example, consistent with the form of the source sentences.

We conducted some experiments to test the assumption that the derived meaning embeddings should improve performance on downstream tasks that require understanding of the meaning of the sentences regardless of their form.

We evaluated embeddings produced by the ADNet, trained in the Headlines dataset, on a task of paraphrase detection.

We used the SentEval toolkit BID8 and the Microsoft Research Paraphrase Corpus BID11 .

The F1 scores on this task for different models are presented in Table 3 .

Note that all models, except InferSent, are unsupervised.

The InferSent model was trained on a big SNLI dataset, consisting of more than 500,000 manually annotated pairs.

ADNet achieves the the highest score among the unsupervised systems and outperforms the regular sequence-to-sequence autoencoder with a large gap.

Table 3 : F1 scores on the task of paraphrase detection using the SentEval toolkit BID8 7 CONCLUSIONIn this paper, we presented ADNet, a new model that performs adversarial decomposition of text representation.

In contrast to previous work, it does not require a parallel training corpus and works directly on hidden representations of sentences.

Most importantly, is does not treat the form as a binary variable (as done in most previously proposed models), enabling a fine-grained change of the form of sentences or specific aspects of meaning.

We evaluate ADNet on two tasks: the shift of language register and diachronic language change.

Our solution achieves superior results, and t-SNE visualizations of the learned meaning and style embeddings illustrate that the proposed motivational loss leads to significantly better separation of the form embeddings.

@highlight

A method which learns separate representations for the meaning and the form of a sentence