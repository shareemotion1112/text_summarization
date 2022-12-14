We propose a generative adversarial training approach for the problem of clarification question generation.

Our approach generates clarification questions with the goal of eliciting new information that would make the given context more complete.

We develop a Generative Adversarial Network (GAN) where the generator is a sequence-to-sequence model and the discriminator is a utility function that models the value of updating the context with the answer to the clarification question.

We evaluate on two datasets, using both automatic metrics and human judgments of usefulness, specificity and relevance, showing that our approach outperforms both a retrieval-based model and ablations that exclude the utility model and the adversarial training.

A goal of natural language processing is to develop techniques that enable machines to process naturally occurring language.

However, not all language is clear and, as humans, we may not always understand each other BID10 ; in cases of gaps or mismatches in knowledge, we tend to ask questions BID9 .

In this work, we focus on the task of automatically generating clarification questions: questions that ask for information that is missing from a given linguistic context.

Our clarification question generation model builds on the sequence-to-sequence approach that has proven effective for several language generation tasks BID37 BID39 BID5 .

Unfortunately, training a sequence-to-sequence model directly on context/question pairs yields generated questions that are highly generic 1 , corroborating a common finding in dialog systems BID17 .

Our goal is to be able to generate questions that are useful and specific.

To achieve this, we begin with a recent observation of BID30 , who considered the task of question reranking: the system should learn to generate clarification questions whose answers have high utility, which they defined as the likelihood that this question would lead to an answer that will make the context more complete ( §2.3).

Inspired by this, we construct a question generation model that first generates a question given a context, and then generates a hypothetical answer to that question.

Given this (context, question, answer) tuple, we train a utility calculator to estimate the usefulness of this question.

We then show that this utility calculator can be generalized using ideas for generative adversarial networks BID8 for text BID40 , wherein the utility predictor plays the role of the "discriminator" and the question generator is the "generator" ( §2.2), which we train using the MIXER algorithm BID29 .We evaluate our approach on two question generation datasets: for posts on Stack Exchange and for Amazon product descriptions (Figure 1 ).

Using both automatic metrics and human evaluation, we demonstrate that our adversarially trained model generates a more diverse set of questions than all the baseline models.

Furthermore, we find that although all models generate questions that are relevant to the context at hand, our adversarially-trained model generates questions that are more specific to the context.

Our goal is to build a model that, given a context, can generate an appropriate clarification question.

As a running example, we will use the Amazon setting: where the dataset consists of (context, question, answer) triples where the context is the product description, question is clarification question about that product that (preferably) is not already answered in the description and answer is the seller's (or other users') reply to the question.

Representationally, our question generator is a standard sequence-to-sequence model with attention ( §2.1).

The learning problem is: how to train the sequence-to-sequence model to produce good question.

An overview of our training setup is shown in FIG1 .

Given a context, our question generator outputs a question.

In order to evaluate the usefulness of this question, we then have a second sequence-to-sequence model called the "answer generator" that generates a hypothetical answer based on the context and the question ( §2.5).

This (context, question and answer) triple is fed into a UTILITY calculator, whose initial goal is to estimate the probability that this question/answer pair is useful in this context ( §2.3).

This UTILITY is treated as a reward, which is used to update the question generator using the MIXER BID29 algorithm ( §2.2).

Finally, we reinterpret the answer-generator-plus-utility-calculator component as a discriminator for differentiating between true (context, question, answer) triples and synthetic triples ( § 2.4), and optimize this adversarial objective using MIXER.

We use a standard attention based sequence-to-sequence model BID21 for our question generator.

Given an input sequence (context) c = (c 1 , c 2 , ..., c N ), this model generates an output sequence (question) q = (q 1 , q 2 , ..., q T ).

The architecture of this model is an encoder-decoder with attention.

The encoder is a recurrent neural network (RNN) operating over the input word embeddings to compute a source context representationc.

The decoder uses this source representation to generate the target sequence one word at a time: DISPLAYFORM0 (1) In Eq 1,h t is the attentional hidden state of the RNN at time t and W s and W c are parameters of the model (details in Appendix A).

The predicted token q t is the token in the vocabulary that is assigned the highest probability using the softmax function.

The standard training objective for sequence-tosequence model is to maximize the log-likelihood of all (c, q) pairs in the training data D which is equivalent to minimizing the loss, DISPLAYFORM1 2.2 TRAINING THE GENERATOR TO OPTIMIZE QUESTION UTILITYTraining sequence-to-sequence models for the task of clarification question generation (with context as input and question as output) using maximum likelihood objective unfortunately leads to the generation of highly generic questions, such as "What are the dimensions?" when asking questions about home appliances.

This issue has been observed in dialog generation as well BID17 .

Recently BID30 observed that usefulness of a question can be better measured as the utility that would be obtained if the context were updated with the answer to the proposed question.

We use this observation to define a UTILITY based reward function and train the question generator to optimize this reward.

We train the UTILITY reward to predict the likelihood that a question would generate an answer that would increase the utility of the context by adding useful information to it (see §2.3 for details).Similar to optimizing metrics like BLEU and ROUGE, this UTILITY function also operates on discrete text outputs, which makes optimization difficult due to non-differentiability.

A successful recent approach dealing with the non-differentiability while also retaining some advantages of maximum likelihood training is the Mixed Incremental Cross-Entropy Reinforce BID29 algorithm (MIXER).

In MIXER, the overall loss L is differentiated as in REINFORCE BID38 : DISPLAYFORM2 where y s is a random output sample according to the model p θ , where θ are the parameters of the network.

We then approximate the expected gradient using a single sample q s = (q DISPLAYFORM3 In REINFORCE, the policy is initialized random, which can cause long convergence times.

To solve this, MIXER starts by optimizing maximum likelihood and slowly shifts to optimizing the expected reward from Eq 3.

For the initial ∆ time steps, MIXER optimizes L mle and for the remaining (T − ∆) time steps, it optimizes the external reward.

In our model, we minimize the UTILITY-based loss L max-utility defined as: DISPLAYFORM4 where r(q p ) is the UTILITY based reward on the predicted question and r(q b ) is a baseline reward introduced to reduce the high variance otherwise observed when using REINFORCE.In MIXER, the baseline is estimated using a linear regressor that takes in the current hidden states of the model as input and is trained to minimize the mean squared error (||r( DISPLAYFORM5 Instead we use a self-critical training approach BID32 where the baseline is estimated using the reward obtained by the current model under greedy decoding during test time.

Given a (context, question, answer) triple, BID30 introduce a utility function UTILITY(c, q, a) to calculate the value of updating a context c with the answer a to a clarification question q. The inspiration for thier utility function is to estimate the probability that an answer would be a meaningful addition to a context, and treat this as a binary classification problem where the positive instances are the true (context, question, answer) triples in the dataset whereas the negative instances are contexts paired with a random (question, answer) from the dataset.

The model we use is to first embed of the words in the context c, then use an LSTM (long-short term memory) BID12 to generate a neural representationc of the context by averaging the output of each of the hidden states.

Similarly, we obtain a neural representationq and a of q and a respectively using question and answer LSTM models.

Finally, a feed forward neural network F UTILITY (c,q,ā) predicts the usefulness of the question.

The UTILITY function trained on true vs random samples from real data (as described in the previous section) can be a weak reward signal for questions generated by a model due to the large discrepancy between the true data and the model's outputs.

In order to strengthen the reward signal, we reinterpret the UTILITY function (coupled with the answer generator) as a discriminator in an adversarial learning setting.

That is, instead of taking the UTILITY calculator to be a fixed model that outputs the expected quality of a question/answer pair, we additionally optimize it to distinguish between true question/answer pairs and model-generated ones.

This reinterpretation turns our model into a form of a generative adversarial network (GAN) BID8 .A GAN is a training procedure for "generative" models that can be interpreted as a game between a generator and a discriminator.

The generator is an arbitrary model g ∈ G that produces outputs (in our case, questions).

The discriminator is another model d ∈ D that attempts to classify between true outputs and model-generated outputs.

The goal of the generator is to generate data such that it can fool the discriminator; the goal of the discriminator is to be able to successfully distinguish between real and generated data.

In the process of trying to fool the discriminator, the generator produces data that is as close as possible to the real data distribution.

Generically, the GAN objective is: DISPLAYFORM0 where x is sampled from the true data distributionp, and z is sampled from a prior defined on input noise variables p z .Although GANs have been successfully used for image tasks, training GANs for text generation is challenging due to the discrete nature of outputs in text.

The discrete outputs from the generator make it difficult to pass the gradient update from the discriminator to the generator.

Recently, BID40 proposed a sequence GAN model for text generation to overcome this issue.

They treat their generator as an agent and use the discriminator as a reward function to update the generative model using reinforcement learning techniques.

Our GAN-based approach is inspired by this sequence GAN model with two main modifications: a) We use the MIXER algorithm as our generator ( §2.2) instead of policy gradient approach; and b) We use the UTILITY function ( §2.3) as our discriminator instead of a convolutional neural network (CNN).In our model, the answer is an latent variable: we do not actually use it anywhere except to train the discriminator.

Because of this, we train our discriminator using (context, true question, generated answer) triples as positive instances and (context, generated question, generated answer) triples as the negative instances.

Formally, our objective function is:LGAN DISPLAYFORM1 where U is the UTILITY discriminator, M is the MIXER generator,p is our data of (context, question, answer) triples and A is our answer generator.

Question Generator.

We pretrain our question generator using the sequence-to-sequence model §2.1 where we define the input sequence as the context and the output sequence as the question.

This answer generator is trained to maximize the log-likelihood of all ([context+question], answer) pairs in the training data.

Parameters of this model are updated during adversarial training.

Answer Generator.

We pretrain our answer generator using the sequence-to-sequence model §2.1 where we define the input sequence as the concatenation of the context and the question and the output sequence as the answer.

This answer generator is trained to maximize the log-likelihood of all (context, question) pairs in the training data.

Unlike the question generator, the parameters of the answer generator are kept fixed during the adversarial training.

Discriminator.

We pretrain the discriminator using (context, question, answer) triples from the training data.

For positive instances, we use a context and its true question, answer and for negative instances, we use the same context but randomly sample a question from the training data (and use the answer paired with that random question).

We base our experimental design on the following research questions:1.

Do generation models outperform simpler retrieval baselines?

2.

Does optimizing the UTILITY reward improve over maximum likelihood training?

3.

Does using adversarial training improve over optimizing the pretrained UTILITY?

4.

How do the models perform when evaluated for nuances such as specificity and usefulness?

We evaluate our model on two datasets.

The first is from StackExchange and was curated by BID30 ; the second is from Amazon, curated by BID22 , and has not previously been used for the task of question generation.

StackExchange.

This dataset consists of posts, questions asked to that post on stackexchange.com (and answers) collected from three related subdomains on stackexchage.com (askubuntu, unix and superuser).

Additionally, for 500 instances each from the tune and the test set, the dataset includes 1 to 5 other questions identified as valid questions by expert human annotators from a pool of candidate questions.

This dataset consists of 61, 681 training, 7710 validation and 7709 test examples.

Amazon.

Each instance consists of a question asked about a product on amazon.com combined with other information (product ID, question type "Yes/No", answer type, answer and answer time).To obtain the description of the product, we use the metadata information contained in the amazon reviews dataset BID23 .

We consider at most 10 questions for each product.

This dataset includes several different product categories.

We choose the Home and Kitchen category since it contains a high number of questions and is relatively easy category for human based evaluation.

This dataset consists of 19, 119 training, 2435 validation and 2305 test examples, and each product description contains between 3 and 10 questions (average: 7).

We compare three variants (ablations) of our proposed approach, together with an information retrieval baseline: GAN-Utility is our full model which is a UTILITY function based GAN training ( § 2.4) including the UTILITY discriminator, a MIXER question generator and a sequence-tosequence based answer generator.

Max-Utility is our reinforcement learning baseline with a pretrained question generator described model ( § 2.2) without the adversarial training.

MLE is the question generator model pretrained on context, question pairs using maximum likelihood objective ( §2.1).

Lucene 3 is a TF-IDF (term frequency-inverse document frequency) based document ranking system which given a document, retrieves N other documents that are most similar to the given document.

Given a context, we use Lucene to retrieve top 10 contexts that are most similar to the given context.

We randomly choose a question from the 10 questions paired with these contexts to construct our Lucene baseline BID45 .

Experimental details of all our models are described in Appendix B.

We evaluate initially with several automated evaluation metrics, and then more substantially based on crowdsourced human judgments.

Automatic metrics include: Diversity, which calculates the proportion of unique trigrams 5 in the output to measure the diversity as commonly used to evaluate dialogue generation BID17 ; BLEU BID27 , which evaluate n-gram precision between a predicted sentence and reference sentences; and METEOR BID1 , which is similar to BLEU but includes stemmed and synonym matches when measuring the similarity between the predicted sequence and the reference sequences.

Table 1 : DIVERSITY as measured by the proportion of unique trigrams in model outputs.

BLEU and METEOR scores using up to 10 references for the Amazon dataset and up to six references for the StackExchange dataset.

Numbers in bold are the highest among the models.

All results for Amazon are on the entire test set whereas for StackExchange they are on the 500 instances of the test set that have multiple references.

Human judgments involve showing contexts and generated questions to crowdworkers 6 and asking them to evaluate the questions along several axes.

Roughly, we ask for the following five judgments for each question (exact wordings in Appendix C): Is it relevant (yes/no); Is it grammatical (yes/comprehensible/incomprehensible); How specific is it to this product (four options from "specific to only this product" to "generic to any product"); Does this question ask for new information not contained in the discription (completely/somewhat/no); and How useful is this question to a potential buyer (four options from "should be included in the description" to "useful only to the person asking").

For the last three questions, we also allowed a "not applicable" response in the case that the question was either ungrammatical or irrelevant.

Table 1 shows the results on the two datasets when evaluated according to automatic metrics.

In the Amazon dataset, GAN-Utility outperforms all ablations on DIVERSITY, suggesting that it produces more diverse outputs.

Lucene, on the other hand, has the highest DIVERSITY since it consists of human generated questions, which tend to be more diverse because they are much longer compared to model generated questions.

This comes at the cost of lower match with the reference as visible in the BLEU and METEOR scores.

In terms of BLEU and METEOR, there is inconsistency.

Although GAN-Utility outperforms all baselines according to METEOR, the fully ablated MLE model has a higher BLEU score.

This is because BLEU score looks for exact n-gram matches and since MLE produces more generic outputs, it is much more likely that it will match one of 10 references compared to the specific/diverse outputs of GAN-Utility, since one of those ten is highly likely to itself be generic.

In the StackExchange dataset GAN-Utility outperforms all ablations on both BLEU and METEOR.

Unlike in the Amazon dataset, MLE does not outperform GAN-Utility in BLEU.

This is because the MLE outputs in this dataset are not as generic as in the amazon dataset due to the highly technical nature of contexts in StackExchange.

As in the Amazon dataset, GAN-Utility outperforms MLE on DIVERSITY.

Interestingly, the Max-Utility ablation achieves a higher DIVERSITY score than GAN-Utility.

On manual analysis we find that Max-Utility produces longer outputs compared to GAN-Utility but at the cost of being less grammatical.

Table 2 shows the numeric results of human-based evaluation performed on the reference and the system outputs on 500 random samples from the test set of the Amazon dataset.

7 These results overall show that the GAN-Utility model successfully generates the most specific questions, while being equally good at seeking new information and being useful to potential buyers.

All approaches produce relevant, grammatical questions.

All our models are all equally good at seeking new information, but are weaker than Lucene, which performs better according to new information but at Table 2 : Results of human judgments on model generated questions on 500 sample Home & Kitchen product descriptions.

The options described in §3.3 are converted to corresponding numeric range (as described in Appendix C).

The difference between the bold and the non-bold numbers is statistically insignificant with p <0.001.

Reference is excluded in the significance calculation.

the cost of much lower specificity and slightly lower relevance.

Our models are all equally good also at generating useful questions: their usefulness score is significantly better than both Lucene and Reference, largely because Lucene and Reference tend to ask questions that are more often useful only to the person asking the question, making them less useful for potential other buyers (see FIG3 ).

Our full model, GAN-Utility, performs significantly better when measured by specificity to the product, which aligns with the higher DIVERSITY score obtained by GAN-Utility under automatic metric evaluation.

Question Generation.

Most previous work on question generation has been on generating reading comprehension style questions i.e. questions that ask about information present in a given text BID11 BID33 BID11 BID6 .

Outside reading comprehension questions, BID15 use crowdsourcing to generate question templates, BID19 use templated questions to help authors write better related work sections, BID24 introduced visual question answer tasking that focuses on generating natural and engaging questions about an image.

BID25 introduced an extension of this task called the Image Grounded Conversation task where they use both the image and some initial textual context to generate a natural follow-up question and a response to that question.

BID3 propose an active question answering model where they build an agent that learns to reformulate the question to be asked to a question-answering system so as to elicit the best possible answers.

BID6 extract large number of question-answer pairs from community question answering forums and use them to train a model that can generate a natural question given a passage.

Neural Models and Adversarial Training for Text Generation.

Neural network based models have had significant success at a variety of text generation tasks, including machine translation BID0 BID21 , summarization BID26 ), dialog (Serban et al., 2016 BID2 BID16 BID36 , textual style transfer BID13 BID14 BID31 and question answering BID39 .Our task is most similar to dialog, in which a wide variety of possible outputs are acceptable, and where lack of specificity in generated outputs is common.

We addresses Table 3 : Example outputs from each of the systems for a single product description this challenge using an adversarial network approach BID8 , a training procedure that can generate natural-looking outputs, which have been effective for natural image generation BID4 .

Due to the challenges in optimizing over discrete output spaces like text, BID40 introduced a Seq(uence)GAN approach where they overcome this issue by using RE-INFORCE to optimize.

BID18 train an adversarial model similar to SeqGAN for generating next utterance in a dialog given a context.

However, unlike our work, their discriminator is a binary classifier trained to distinguish between human and machine generated utterances.

Finally, BID7 introduce an actor-critic conditional GAN for filling in missing text conditioned on the surrounding context.

In this work, we describe a novel approach to the problem of clarification question generation.

Given a context, we use the observation of BID30 that the usefulness of a clarification question can be measured by the value of updating the context with an answer to the question.

We use a sequence-to-sequence model to generate a question given a context and a second sequenceto-sequence model to generate an answer given the context and the question.

Given the (context, predicted question, predicted answer) triple we calculator the utility of this triple and use it as a reward to retrain the question generator using reinforcement learning based MIXER model.

Further, to improve upon the utility function, we reinterpret it as a discriminator in an adversarial setting and train both the utility function and the MIXER model in a minimax fashion.

We find that our adversarial training approach produces more diverse questions compared to both a model trained using maximum likelihood objective and a model trained using utility reward based reinforcement learning.

There are several avenues of future work in this area.

Following BID24 , we could combine text input with image input to generate more relevant questions.

Because some questions can be answered by looking at the product image in the Amazon dataset BID22 , this could help generate more relevant and useful questions.

As in most One significant research challenge in the space of free text generation problems when the set of possible outputs is large, is that of automatic evaluation BID20 : in our results we saw some correlation between human judgments and automatic metrics, but not enough to trust the automatic metrics completely.

Lastly, integrating such a question generation model into a real world platform like StackExchange or Amazon to understand the real utility of such models and to unearth additional research questions.

A DETAILS OF SEQUENCE-TO-SEQUENCE MODELIn this section, we describe the attention based sequence-to-sequence model introduced in §2.1 of the main paper.

In Eq 1,h t is the attentional hidden state of the RNN at time t obtained by concatenating the target hidden state h t and the source-side context vectorc t , and W s is a linear transformation that maps h t to an output vocabulary-sized vector.

The predicted token q t is the token in the vocabulary that is assigned the highest probability using the softmax function.

Each attentional hidden stateh t depends on a distinct input context vectorc t computed using a global attention mechanism over the input hidden states as: DISPLAYFORM0 a nt h n (7) DISPLAYFORM1 The attention weights a nt is calculated based on the alignment score between the source hidden state h n and the current target hidden state h t .

In this section, we describe the details of our experimental setup.

We preprocess all inputs (context, question and answers) using tokenization and lowercasing.

We set the max length of context to be 100, question to be 20 and answer to be 20.

Our sequence-to-sequence model ( § 2.1) operates on word embeddings which are pretrained on in domain data using Glove BID28 .

We use embeddings of size 200 and a vocabulary with cut off frequency set to 10.

During train time, we use teacher forcing.

During test time, we use beam search decoding with beam size 5.

We use a hidden layer of size two for both the encoder and decoder recurrent neural network models with size of hidden unit set to 100.

We use a dropout of 0.5 and learning ratio of 0.0001 In the MIXER model, we start with ∆ = T and decrease it by 2 for every epoch (we found decreasing ∆ to 0 is ineffective for our task, hence we stop at 2).

In this section, we describe in detail the human based evaluation methodology introduced in §3.3 of the main paper.

<|TLDR|>

@highlight

We propose an adversarial training approach to the problem of clarification question generation which uses the answer to the question to model the reward. 