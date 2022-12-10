In this paper, we extend the persona-based sequence-to-sequence (Seq2Seq) neural network conversation model to a multi-turn dialogue scenario by modifying the state-of-the-art hredGAN architecture to simultaneously capture utterance attributes such as speaker identity, dialogue topic, speaker sentiments and so on.

The proposed system, phredGAN has a persona-based HRED generator (PHRED) and a conditional discriminator.

We also explore two approaches to accomplish the conditional discriminator: (1) $phredGAN_a$, a system that passes the attribute representation as an additional input into a traditional adversarial discriminator, and (2) $phredGAN_d$, a dual discriminator system which in addition to the adversarial discriminator, collaboratively predicts the attribute(s) that generated the input utterance.

To demonstrate the superior performance of phredGAN over the persona SeqSeq model, we experiment with two conversational datasets, the Ubuntu Dialogue Corpus (UDC) and TV series transcripts from the Big Bang Theory and Friends.

Performance comparison is made with respect to a variety of quantitative measures as well as crowd-sourced human evaluation.

We also explore the trade-offs from using either variant of $phredGAN$ on datasets with many but weak attribute modalities (such as with Big Bang Theory and Friends) and ones with few but strong attribute modalities (customer-agent interactions in Ubuntu dataset).

Recent advances in machine learning especially with deep neural networks has lead to tremendous progress in natural language processing and dialogue modeling research BID13 BID14 BID10 .

Nevertheless, developing a good conversation model capable of fluent interaction between a human and a machine is still in its infancy stage.

Most existing work relies on limited dialogue history to produce response with the assumption that the model parameters will capture all the modalities within a dataset.

However, this is not true as dialogue corpora tend to be strongly multi-modal and practical neural network models find it difficult to disambiguate characteristics such as speaker personality, location and sub-topic in the data.

Most work in this domain has primarily focused on optimizing dialogue consistency.

For example, Serban et al. BID10 BID12 a) and BID15 introduced a Hierarchical Recurrent Encoder-Decoder (HRED) network architecture that combines a series of recurrent neural networks to capture long-term context state within a dialogue.

However, the HRED system suffers from lack of diversity and does not have any guarantee on the generator output since the output conditional probability is not calibrated.

BID8 tackles these problems by training a modified HRED generator alongside an adversarial discriminator in order to increase diversity and provide a strong and calibrated guarantee to the generator's output.

While the hredGAN system improves upon response quality, it does not capture speaker and other attributes modality within a dataset and fails to generate persona specific responses in datasets with multiple modalities.

On the other hand, there has been some recent work on introducing persona into dialogue models.

For example, BID5 integrates attribute embeddings into a single turn (Seq2Seq) generative dialogue model.

In this work, Li et al. consider persona models one with Speaker-only representation and the other with Speaker and Addressee representations (Speaker-Addressee model), both of which capture certain speaker identity and interactions.

BID7 continue along the Figure 1 : The PHRED generator with local attention -The attributes C, allows the generator to condition its response on the utterance attributes such as speaker identity, subtopics and so on.

same line of thought by considering a Seq2Seq dialogue model with Responder-only representation.

In both of these cases, the attribute representation is learned during the system training.

BID16 proposed a slightly different approach.

Here, the attributes are a set of sentences describing the profile of the speaker.

In this case, the attributes representation is not learned.

The system however learns how to attend to different parts of the attributes during training.

Still, the above persona-based models have limited dialogue history (single turn); suffer from exposure bias worsening the trade-off between personalization and conversation quality and cannot generate multiple responses given a dialogue context.

This is evident in the relatively short and generic responses produced by these systems, even though they generally capture the persona of the speaker.

In order to overcome these limitations, we propose two variants of an adversarially trained persona conversational generative system, phredGAN , namely phredGAN a and phredGAN d .

Both systems aim to maintain the response quality of hredGAN and still capture speaker and other attribute modalities within the conversation.

In fact, both systems use the same generator architecture (PHRED generator), i.e., an hredGAN generator BID8 with additional utterance attribute representation at its encoder and decoder inputs as depicted in Figure 1 .

Conditioning on external attributes can be seen as another input modality as is the utterance into the underlying system.

The attribute representation is an embedding that is learned together with the rest of model parameters similar to BID5 .

Injecting attributes into a multi-turn dialogue system allows the model to generate responses conditioned on particular attribute(s) across conversation turns.

Since the attributes are discrete, it also allows for exploring different what-if scenarios of model responses.

The difference between the two systems is in the discriminator architecture based on how the attribute is treated.

We train and sample both variants of phredGAN similar to the procedure for hredGAN BID8 .

To demonstrate model capability, we train on a customer service related data such as the Ubuntu Dialogue Corpus (UDC) that is strongly bimodal between question poser and answerer, and transcripts from a multi-modal TV series The Big Bang Theory and Friends with quantitative and qualitative analysis.

We examine the trade-offs between using either system in bi-modal or multi-modal datasets, and demonstrate system superiority over state-of-the-art persona conversational models in terms of dialogue response quality and quantitatively with perplexity, BLEU, ROUGE and distinct n-gram scores.

In this section, we briefly introduce the state-of-the-art hredGAN model and subsequently show how we derive the two persona versions by combining it with the distributed representation of the dialogue speaker and utterance attributes, or with an attribute discrimination layer at the end of the model pipeline.

The phredGAN d dual discriminator -Left: D adv is a word-level discriminator used by both phredGAN a and phredGAN d to judge normal dialogue coherency as in hredGAN .

Right: D att , an utterance-level attribute discriminator is used only in phredGAN d to predict the likelihood a given utterance was generated from a particular attribute.

Problem Formulation: The hredGAN BID8 formulates multi-turn dialogue response generation as: given a dialogue history of sequence of utterances, DISPLAYFORM0 contains a variable-length sequence of M i word tokens such that X i j ∈ V for vocabulary V , the dialogue model produces an output DISPLAYFORM1 , where T i is the number of generated tokens.

The framework uses conditional GAN structure to learn a mapping from an observed dialogue history to a sequence of output tokens.

The generator, G, is trained to produce sequences that cannot be distinguished from the ground truth by an adversarially trained discriminator, D akin to a two-player min-max optimization problem.

The generator is also trained to minimize the cross-entropy loss L M LE (G) between the ground truth X i+1 , and the generator output Y i .

The following objective summarizes both goals: DISPLAYFORM2 where λ G and λ M are training hyperparamters and L cGAN (G, D) and L M LE (G) are defined in Eqs.(5) and (7) of BID8 respectively.

Please note that the generator G and discriminator D share the same encoder and embedding representation of the word tokens.

The proposed architecture of phredGAN is very similar to that of hredGAN BID8 .

The only difference is that the dialogue history is now DISPLAYFORM0 where C i is additional input that represents the speaker and/or utterance attributes.

Please note that C i can either be a sequence of tokens or single token such that C i j ∈ V c for vocabulary V c. Also, at the ith turn, C i and C i+1 are the source/input attribute and target/output attribute to the generator respectively.

The embedding for attribute tokens is also learned similar to that of word tokens.

Both versions of phredGAN shares the same generator architecture (PHRED) but different discriminators.

Below is the highlight of how they are derived from the hredGAN architecture.

Encoder: The context RNN, cRN N takes the source attribute C i as an additional input by concatenating its representation with the output of eRN N as in Figure 1 .

If the attribute C i is a sequence of tokens, then an attention (using the output of eRN N ) over the source attribute representations is concatenated with the output of eRN N .

This output is used by the generator to create a context state for a turn i.

Generator: The generator decoder RNN, dRN N takes the target attribute C i+1 as an additional input as in Fig. 1 .

If the attribute C i+1 is a sequence of tokens, then an attention (using the output of dRN N ) over the attribute representations is concatenated with the rest of the decoder inputs.

This forces the generator to draw a connection between the generated responses and the utterance attributes such as speaker identity.

Noise Injection: As in BID8 , we also explore different noise injection methods.

Objective: For phredGAN , the optimization objective in eq. (1) can be updated as: DISPLAYFORM1 where DISPLAYFORM2 are the traditional adversarial and attribute prediction loss respectively and dependent on the architectural variation.

It is worth to point out that while the former is adversarial, the later is collaborative in nature.

The MLE loss is common and can be expressed as: DISPLAYFORM3 where Z i the noise sample and depends on the choice of either utterance-level or word-level noise input into the generator BID8 .2.3 phredGAN a : ATTRIBUTES AS A DISCRIMINATOR INPUT phredGAN a shares the same discriminator architecture as the hredGAN but with additional input, C i+1 .

Since it does not use attribute prediction, λ Gatt = 0.

DISPLAYFORM4 The addition of speaker or utterance attributes allows the dialogue model to exhibit personality traits given consistent responses across style, gender, location, and so on.

phredGAN d does not take the attribute representation at its input but rather use the attributes as the target of an additional discriminator D att .

The adversarial and the attribute prediction losses can be respectively expressed as: DISPLAYFORM0 Attribute Discriminator: In addition to the existing word-level adversarial discriminator D adv from hredGAN , we add an attribute discriminator, D att , that discriminates on an utterance level to capture attribute modalities since attributes are assigned at utterance level.

The discriminator uses a unidirectional RNN (D attRN N ) that maps the input utterance to the particular attribute(s) that generated it.

The attributes can be seen as hidden states that inform or shape the generator outputs.

The attribute discriminator can be expressed as: DISPLAYFORM1 where E(.) is the word embedding lookup BID8 , χ = X i+1 for groundtruth and χ = Y i for the generator output.3 MODEL TRAINING AND INFERENCE

We train both the generator and the discriminator (with shared encoder) of both variants of phredGAN using the training procedure in Algorithm 1 BID8 .

For both variants, λ G adv = λ M = 1, and for phredGAN a and phredGAN d , λ Gatt = 0 and λ Gatt = 1 respectively.

Since the encoder, word embedding and attribute embedding are shared, we are able to train the system end-to-end with back-propagation.

Encoder: The encoder RNN, eRN N , is bidirectional while cRRN is unidirectional.

All RNN units are 3-layer GRU cell with hidden state size of 512.

We use word vocabulary size, V = 50, 000 with word embedding size of 512.

The number of attributes, V c is dataset dependent but we use an attribute embedding size of 512.

In this study, we only use one attribute per utterance so that is no need to use attention to combine the attribute embeddings.

Generator: The generator decoder RNN, dRN N is also a 3-layer GRU cell with hidden state size of 512.

The aRN N outputs are connected to the dRN N input using an additive attention mechanism BID0 .Adversarial Discriminator: The word-level discriminator RNN, D RN N is a bidirectional RNN, each 3-layer GRU cell with hidden state size of 512.

The output of both the forward and the backward cells for each word are concatenated and passed to a fully-connected layer with binary output.

The output is the probability that the word is from the ground truth given the past and future words of the sequence, and in the case of phredGAN a , the responding speaker's embedding.

Attribute Discriminator: The attribute discriminator RNN, D attRN N is a unidirectional RNN with a 3-layer GRU cell, each of hidden state size 512.

A softmax layer is then applied to project the final hidden state to a prespecified number of attributes, V c .

The output is the probability distribution over the attributes.

Others: All parameters are initialized with Xavier uniform random initialization BID2 .

Due to the large word vocabulary size, we use sampled softmax loss BID3 for MLE loss to expedite the training process.

However, we use full softmax for model evaluation.

For both systems, parameters updates are conditioned on the word-level discriminator accuracy performance as in BID8 with acc D th adv = 0.99 and acc G th = 0.75.

The model is trained end-to-end using the stochastic gradient descent algorithm.

Finally, the model is implemented, trained, and evaluated using the TensorFlow deep learning framework.

We use an inference strategy similar to the approach in BID8 .For the modified noise sample, we perform a linear search for α with sample size L = 1 based on the average word-level discriminator loss, −logD adv (G(.)) BID8 ) using trained models run in autoregressive mode to reflect performance in actual deployment.

The optimum α value is then used for all inferences and evaluations.

During inference, we condition the dialogue response generation on the encoder outputs, noise samples, word embedding and the attribute embedding of the intended responder.

With multiple noise samples, L = 64, we rank the generator outputs by the discriminator which is also conditioned on encoder outputs, and the intended responder's attribute embedding.

The final response is the response ranked highest by the discriminator.

For phredGAN d , we average the confidences produced by D adv and D att .

In this section, we explore the performance of PHRED, phredGAN a and phredGAN d on two conversational datasets and compare its performance to non-adversarial persona Seq2seq models BID5 as well as to the adversarial hredGAN BID8 with no explicit persona.

TV Series Transcripts dataset BID10 .

We train all models on transcripts from the two popular TV drama series, Big Bang Theory and Friends.

Following a similar preprocessing setup in BID5 , we collect utterances from the top 12 speakers from both series to construct a corpus of 5,008 lines of multi-turn dialogue.

We split the corpus into training, development, and test set with a 94%, 3%, and 3% proportions, respectively, and pair each set with a corresponding attribute file that maps speaker IDs to utterances in the combined dataset.

Due to the small size of the combined transcripts dataset, we first train our model on the larger Movie Triplets Corpus (MTC) by BID1 which consists of 240,000 dialogue triples.

We pre-train our model on this dataset to initialize our model parameters to avoid overfitting on a relatively small persona TV series dataset.

After pre-training on MTC, we reinitialize the attribute embeddings in the generator from a uniform distribution following a Xavier initialization BID2 for training on the combined person TV series dataset.

Ubuntu Dialogue Corpus (UDC) dataset BID12 .

We train our model on 1.85 million conversations of multi-turn dialogue from the Ubuntu community hub, with an average of 5 utterances per conversation.

We assign two types of speaker IDs to utterances in this dataset: questioner and helper.

We follow a similar training, development, and test split as the UDC dataset in BID8 , with 90%, 5%, and 5% proportions, respectively, and pair each set with a corresponding attribute file that maps speaker IDs to utterances in the combined datasetWhile the overwhelming majority of utterances in UDC follow two speaker types, the dataset does include utterances that do not classify under either a questioner or helper speaker type.

In order to remain consistent, we assume that there are only two speaker types within this dataset and that the first utterance of every dialogue is from a questioner.

This simplifying assumption does introduce a degree of noise into each persona model's ability to construct attribute embeddings.

However, our experiment results demonstrate that both phredGAN a and phredGAN d is still able to differentiate between the larger two speaker types in the dataset.

We use similar evaluation metrics as in BID8 including perplexity, BLEU BID9 , ROUGE BID6 , distinct n-gram BID4 and normalized average sequence length (NASL) scores.

For human evaluation, we follow a similar setup as BID4 , employing crowd-sourced judges to evaluate a random selection of 200 samples.

We present both the multi-turn context and the generated responses from the models to 3 judges and asked them to rank the general response quality in terms of relevance, informativeness, and persona.

For N models, the model with the lowest quality is assigned a score 0 and the highest is assigned a score N-1.

Ties are not allowed.

The scores are normalized between 0 and 1 and averaged over the total number of samples and judges.

We compare the non-adversarial persona HRED model, PHRED with the adversarially trained ones, i.e. hredGAN , phredGAN a and phredGAN d , to demonstrate the impact of adversarial training.

Please note that no noise was added to the PHRED model.

We also compare the persona models to Li et al.'s work BID5 which uses a Seq2Seq framework in conjunction with learnable persona embeddings.

Their work explores two persona models in order to incorporate vector representations of speaker interaction and speaker attributes into the decoder of their Seq2Seq model i.e., Speaker model (SM) and Speaker-Addressee model (SAM).

All reported results are based on our implementation of their models in BID5 .

For both phredGAN a and phredGAN d , we determine the noise injection method and the optimum noise variance α that allows for the best performance on both datasets.

We find that phredGAN d performs optimally with word-level noise injection on both Ubuntu and TV transcripts, while phredGAN a performs the best with utterance-level noise injection on TV transcripts and word-level injection on UDC.

For all phredGAN models, we perform a linear search for optimal noise variance values between 1 and 30 at an increment of 1, with a sample size of L = 1.

For phredGAN d , we obtain an optimal α of 4 and 6 for the UDC and TV Transcripts respectively.

For phredGAN a , we obtain an optimal value of 2 and 5 for the combined TV series dataset and the much larger UDC respectively.

We will now present our assessment of performance comparisons of phredGAN against the baselines, PHRED, hredGAN and Li et al.'s persona Seq2Seq models.

We first report the performance on TV series transcripts in table 1.

The performance of both SM and SAM models in BID5 compared to the hredGAN shows a strong baseline and indicates that the effect of persona is more important than that of multi-turn and adversarial training for datasets with weak multiple persona.

However, once the persona information is added to the hredGAN , the resulting phredGAN shows a significant improvement over the SM and SAM baselines with phredGAN a performing best.

We also observe that PHRED performs worse than the baseline S(A)M models on a number of metrics but we attribute this to the effect of persona on a limited dataset that results into less informative responses.

This behavior was also reported in BID5 where the persona models produce less informative responses than the non-personal Seq2seq models but it seems to be even worse in multi-turn context.

However, unlike the SpeakerAddressee and PHRED models that suffer from lower response quality due to persona conditioning, we note that conditioning the generator and discriminator of phredGAN on speaker embeddings does not compromise the systems ability to produce diverse responses.

This problem might have been alleviated by the adversarial training that encourages the generator model to produce longer, more informative, and diverse responses that have high persona relevance even with a limited dataset.

We also compare the models performances on the UDC.

The evaluation result is summarized in table 2.

While the deleterious effect of persona conditioning on response diversity is still worse with PHRED than with S(A)M models, we note that hredGAN performs much better than the S(A)M models.

This is because, the external persona only provides just a little more information than is already available from the UDC utterances.

We also note an improvement of phredGAN variants over the hredGAN in a variety of evaluation metrics including perplexity, ROUGE with the exception of distinct n-grams.

This is expected as phredGAN should be generally less diverse than hredGAN since the number of distinct data distribution modes is more for phredGAN dataset due to the persona attributes.

However, this leads to better response quality with persona, something not achievable with hredGAN .

Also, the much better ROUGE(F1) score indicates that phredGAN is able to strike a better balance between diversity and precision while still capturing the characteristics of the speaker attribute modality in the UDC dataset.

Within the phredGAN variants, phredGAN d seems to perform better.

This is not surprising as speaker classification is much easier on UDC than on TV series.

The attribute discriminator, D att is able to provide more informative feedback on UDC than on TV series where it is more difficult to accurately predict the speaker.

Therefore, we recommend phredGAN a for datasets with weak attribute distinction and phredGAN d for strong attribute distinction.

In addition to the quantitative analysis above, we report the results of the human evaluation in the last column of tables 1 and 2 for the TV Series and UDC datasets respectively.

The human evaluation scores largely agrees with the automatic evaluations on the TV Series with phredGAN a clearly giving the best performance.

However, on the UDC, both hredGAN and phredGAN d performs similarly which indicates that there is a trade off between diversity and persona by each model.

We believe this is due to the strong persona information that already exists in the UDC utterances.

An additional qualitative assessment of these results are in TAB2 with responses from several characters in the TV series dataset and the two characters in UDC.We see that for TV drama series, phredGAN responses are comparatively more informative than that of the Speaker-Addressee model of BID5 .

For example, all the characters in the TV series respond the same to the dialogue context.

Similar behavior is reported in BID5 where for the Speaker-Addressee model, nearly all the characters in the TV series respond with "Of course I love you." to the dialogue context, "Do you love me?" despite the fact that some of the responders sometimes have unfriendly relationship with the addressee.

Many of the novel situations explored by phredGAN are unachievable with the Speaker-Addressee model due to lack of informative responses.

For example, by conditioning as Sheldon from The Big Bang Theory and asking "Do you like me?", our model responds with annoyance if conditioned as Penny ("No, you don't understand.

You're an idiot"), brevity with Leonard ("Yes?") and sarcasm with Raj ("Well , you know , we could be a little more than my friend's friends.") The wide range of responses indicate our model's ability to construct distinct attribute embeddings for each character even from a limited dataset.

The other interesting responses in table 3 indicate phredGAN 's ability to infer not only the context of the conversation but important character information about the addressee.

We also see similar results with our model's output on UDC in table 4.

We demonstrate that by conditioning as either a helper or questioner from the UDC dataset, phredGAN models are able to respond differently to input utterances as well as stay close to the context of the conversation.

In this paper, we improve upon state-of-the-art persona-based response generation models by exploring two persona conversational models: phredGAN a which passes the attribute representation as an additional input into a traditional adversarial discriminator, and phredGAN d a dual discriminator system which in addition to the adversarial discriminator from hredGAN , collaboratively predicts the attribute(s) that are intrinsic to the input utterance.

Both systems demonstrate quantitative improvements upon state-of-the-art persona conversational systems such as the work from BID5 with respect to both quantitative automatic and qualitative human measures.

Our analysis also demonstrates how both variants of phredGAN perform differently on datasets with weak and strong modality.

One of our future direction is to take advantage of phredGAN d 's ability to predict utterance attribute such as speaker identity from just the utterance.

We believe its performance can be improved even with weak modality by further conditioning adversarial updates on both the attribute and adversarial discriminator accuracies.

Overall, this paper demonstrates clear benefits from adversarial training of persona generative dialogue system and leaves the door open for more interesting work to be accomplished in this domain. (Xi, Ci) with N utterances.

Each utterance mini batch i contains Mi word tokens.

Update the context state.

DISPLAYFORM0 Compute the generator output similar to Eq. FORMULA2 in BID8 .

DISPLAYFORM1 Sample a corresponding mini batch of utterance Yi.

Yi ∼ P θ G Yi|, Zi, X i , Ci+1 end for Compute the adversarial discriminator accuracy D acc adv over N − 1 utterances {Yi} DISPLAYFORM2 else Update phredGANa's θ D adv with gradient of the discriminator loss.

DISPLAYFORM3 end if end if if D adv acc < acc G th then Update θ G with the generator's MLE loss only.

DISPLAYFORM4 Update θ G with attribute, adversarial and MLE losses.

DISPLAYFORM5 6 RESULTS -DISCRIMINATOR After training both phredGAN models on the TV series and UDC datasets, we ran inference on some example dialogue contexts.

The responses and their discriminator scores from phredGAN s are listed in Tables 6, and 7.

The tables shows that phredGAN (i) can handle multi-turn dialogue context with utterances and corresponding persona attributes; (ii) generates responses conditioned on a persona attribute; (iii) generates multiple responses per dialogue context and score their human likelihood by the discriminator; and (iv) in case of phredGAN d , can predict the attribute such as speaker identity that might have produced the utterance.

We observe that the discriminator score(s) is/are generally reasonable with longer, more informative and more persona-related responses receiving higher scores.

It worth to note that this behavior, although similar to the behavior of a human judge is learned without supervision.

More so, we observe that phredGAN responses retain contextual consistency sometimes referencing background information that is inherent in the conversation between two speakers.

For example, in the second sample of the TV series in TAB6 , phredGAN a generator, conditioned on Leonard refers to Sheldon by name who is the second interlocutor.

Also, in the third sample, phredGAN a , conditioned on Raj refers to Penny when responding to Leonard who happens to be Penny's boy friend.

We see similar persona-based response generation for the UDC dataset with distinct communication style between the asker and the helper.

For example, in TAB7 , when the asker could not hear some music, phredGAN d , conditioned on helper suggested the asker might not be using the right driver.

For the purpose of completion, we also show some samples from PHRED generator on both UDC and TV series dataset in TAB5 .

<|TLDR|>

@highlight

This paper develops an adversarial learning framework for neural conversation models with persona

@highlight

This paper proposes an extension to hredGAN to simultaneously learn a set of attribute embeddings that represent the persona of each speaker and generate persona-based responses