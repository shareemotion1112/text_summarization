Knowledge-grounded dialogue is a task of generating an informative response based on both discourse context and external knowledge.

As we focus on better modeling the knowledge selection in the multi-turn knowledge-grounded dialogue, we propose a sequential latent variable model as the first approach to this matter.

The model named sequential knowledge transformer (SKT) can keep track of the prior and posterior distribution over knowledge; as a result, it can not only reduce the ambiguity caused from the diversity in knowledge selection of conversation but also better leverage the response information for proper choice of knowledge.

Our experimental results show that the proposed model improves the knowledge selection accuracy and subsequently the performance of utterance generation.

We achieve the new state-of-the-art performance on Wizard of Wikipedia (Dinan et al., 2019) as one of the most large-scale and challenging benchmarks.

We further validate the effectiveness of our model over existing conversation methods in another knowledge-based dialogue Holl-E dataset (Moghe et al., 2018).

Knowledge-grounded dialogue is a task of generating an informative response based on both discourse context and selected external knowledge (Ghazvininejad et al., 2018) .

For example, it is more descriptive and engaging to respond "I've always been more of a fan of the American football team from Pittsburgh, the Steelers!" than "Nice, I like football too.".

As it has been one of the key milestone tasks in conversational research (Zhang et al., 2018) , a majority of previous works have studied how to effectively combine given knowledge and dialogue context to generate an utterance (Zhang et al., 2018; Li et al., 2019b; Parthasarathi & Pineau, 2018; Madotto et al., 2018; Gopalakrishnan et al., 2019) .

Recently, Dinan et al. (2019) proposed to tackle the knowledge-grounded dialogue by decomposing it into two sub-problems: first selecting knowledge from a large pool of candidates and generating a response based on the selected knowledge and context.

In this work, we investigate the issue of knowledge selection in the multi-turn knowledge-grounded dialogue, since practically the selection of pertinent topics is critical to better engage humans in conversation, and technically the utterance generation becomes easier with a more powerful and consistent knowledge selector in the system.

Especially, we focus on developing a sequential latent variable model for knowledge selection, which has not been discussed in previous research.

We believe it brings several advantages for more engaging and accurate knowledge-based chit-chat.

First, it can correctly deal with the diversity in knowledge selection of conversation.

Since one can choose any knowledge to carry on the conversation, there can be one-to-many relations between dialogue context and knowledge selection.

Such multimodality by nature makes the training of a dialogue system much more difficult in a data-driven way.

However, if we can sequentially model the history of knowledge selection in previous turns, we can reduce the scope of probable knowledge candidates at current turn.

Second, the sequential latent model can better leverage the response information, which makes knowledge selection even more accurate.

It is naturally easy to select the knowledge in the pool once the response is known, because the response is generated based on the selected knowledge.

Our sequential model can keep track of prior and posterior distribution over knowledge, which are sequentially updated considering the responses in previous turns, and thus we can better predict the knowledge by sampling from the posterior.

Third, the latent model works even when the knowledge selection labels for previous dialogue are not available, which is common (Dinan et al., 2019) .

Table 1 : Accuracy of knowledge selection with and without knowing the response.

We test with GRU (Cho et al., 2014) , Transformer (Vaswani et al., 2017) and BERT (Devlin et al., 2019) as the sentence encoder.

For human evaluation, we randomly sample 20 dialogues and ask human annotators to select the most likely knowledge sentence from the pool.

Finally, the contributions of this work are as follows.

1.

We propose a novel model named sequential knowledge transformer (SKT).

To the best of our knowledge, our model is the first attempt to leverage a sequential latent variable model for knowledge selection, which subsequently improves knowledge-grounded chit-chat.

2.

Our experimental results show that the proposed model improves not only the knowledge selection accuracy but also the performance of utterance generation.

As a result, we achieve the new state-of-the-art performance on Wizard of Wikipedia (Dinan et al., 2019 ) and a knowledge-annotated version of Holl-E (Moghe et al., 2018) dataset.

As a main testbed of our research, we choose the Wizard of Wikipedia (WoW) benchmark (Dinan et al., 2019) , since it is one of the most large-scale and challenging datasets for open-domain multi-turn knowledge-based dialogue.

Moreover, the dataset can evaluate the algorithm's ability for solving the two subproblems of knowledge selection and response generation.

That is, it provides ground-truth labels of knowledge selection and clear grounding between the pairs of selected knowledge and response.

In our experiments, we also evaluate on Holl-E (Moghe et al., 2018) as another dataset for knowledge-grounded dialogue, after collecting clearer labels of knowledge sentences.

The Flow of Conversation.

The WoW (Dinan et al., 2019) deals with a chit-chat dialogue task where two speakers discuss in depth about a given topic.

One speaker (coined as Wizard) is to be both engaging and knowledgeable on the topic with access to an information retrieval (IR) system over Wikipedia to supplement its knowledge.

The other speaker (Apprentice) is curious and eager to learn about the topic.

With an example in Figure 1 , the conversation flow takes place as follows.

1.

One topic is chosen among 1,431 topics and shared between the two speakers.

2.

Given an apprentice's utterance and a wizard's previous utterance, the IR system retrieves relevant knowledge, which includes the first paragraph of top 7 articles each for wizard and apprentice and the first 10 sentences of the original Wikipedia page of the topic (e.g. the lifeguard wikipage).

The knowledge pool contains 67.57 sentences on average.

Then.

the wizard must choose a single relevant sentence from them (knowledge selection) and construct an utterance (response generation).

3.

The conversation repeats until a minimum number of turns (5 each) reaches.

The Motivation of Sequential Latent Models.

The goal of the task is to model the wizard that solves the two subproblems of knowledge selection and response generation (Dinan et al., 2019) .

In the knowledge selection step, a single relevant knowledge sentence is chosen from a pool of candidates, and in the response generation step, a final utterance is generated with the chosen knowledge and dialogue context.

This pipeline is originally proposed to tackle open-domain TextQA (Chen et al., 2017) ; for example, Min et al. (2018) show its effective for single-document TextQA, to which the key is to locate the sentences that contain the information about the answer to a question.

For knowledge-grounded dialogue, however, there can be one-to-many relations between the dialogue context and the knowledge to be selected unlike TextQA.

Except a direct question about context, one can choose any diverse knowledge to carry on the conversation.

Therefore, the knowledge selection in dialogue is diverse (i.e. multimodal) by nature, which should be correctly considered in the model.

It is our main motivation to propose a sequential latent variable model for knowledge selection, which has not been studied yet.

The latent variable not only models such diversity of knowledge but also sequentially track the topic flow of knowledge in the multi-turn dialogue.

Another practical advantage of the sequential latent model lies in that it is easy to find which knowledge is chosen once the response is known, since the response is written based on the selected knowledge.

Table 1 clearly validates this relation between knowledge and response.

In the WoW dataset, knowing a response boosts the accuracy of knowledge sentence selection for both human and different models.

These results hint that knowledge selection may need to be jointly modeled with response generation in a sequence of multi-turn chit-chats, which can be done by the sequential latent models.

We propose a novel model for knowledge-grounded conversation named sequential knowledge transformer (SKT), whose graphical model is illustrated in Figure 2 .

It is a sequential latent model that sequentially conditions on previously selected knowledge to generate a response.

We will use 1 ≤ t ≤ T to iterate over dialogue turns, 1 ≤ m ≤ M and 1 ≤ n ≤ N to respectively iterate over words in the utterance of apprentice and wizard, and 1 ≤ l ≤ L to denote knowledge sentences in the pool.

Thus, T is the dialogue length, M and N are the length of each utterance of apprentice and wizard, and L is the size of the knowledge pool.

The input to our model at turn t is previous turns of conversation, which consists of utterances from apprentice x 1 , ..., x t , utterances from wizard y 1 , ..., y t−1 and the knowledge pool

The output of the model is selected knowledge k t s and the wizard's response y t .

Below, we discuss sentence embedding, knowledge selection and utterance decoding in our approach.

Note that our technical novelty lies in the knowledge selection model, while exploiting existing techniques for text encoding and utterance decoding.

Sentence Encoding.

We represent an apprentice utterance x t to an embedding h t x using BERT (Devlin et al., 2019) and average pooling over time steps (Cer et al., 2018) :

Likewise, the utterance of Wizard y t−1 is embedded as h t−1 y and knowledge sentences are as {h

at dialog turn t is jointly represented through a GRU (Cho et al., 2014)

Sequential Knowledge Selection.

Compared to previous works, we make two significant modifications.

First, we regard the knowledge selection as a sequential decision process instead of a single-step decision process.

Second, due to the diversity of knowledge selection in dialogue, we model it as latent variables.

As a result, we can carry out the joint inference of multi-turns of knowledge selection and response generation rather than separate inference turn by turn.

There have been much research on sequential latent variable models (Chung et al., 2015; Fraccaro et al., 2016; Goyal et al., 2017; Aneja et al., 2019; Shankar & Sarawagi, 2019) .

For example, Shankar & Sarawagi (2019) propose a posterior attention model that represents the attention of seq2seq models as sequential latent variables.

Inspired by them, we factorize the response generation with Figure 2: A graphical representation of the proposed sequential knowledge transformer (SKT) model.

At the third turn, the goal is to generate wizard's response (y 3 ) given dialogue context (x ≤3 , y <3 ).

Our model sequentially infer which knowledge is likely to be used (k ≤3 ), from which the utterance y 3 is generated.

latent knowledge selection and derive the variational lower bound as follows:

where

is a categorical conditional distribution of knowledge given dialogue context and previously selected knowledge, and

The conditional probability of generating wizard's response y t given dialogue context x ≤t and y <t , can be re-written from Eq. (2) as follows: Finally, we sample knowledge k t s over attention distribution in Eq.(5) and pass it to the decoder.

At test time, we select the knowledge with the highest probability.

Decoding with Copy Mechanism.

We generate the wizard's response at turn t, given current context x t and selected knowledge sentence k

where α copy t,n = σ(W copy p copy t,n (w) · V t ) and σ is a sigmoid.

Finally, we select the word with the highest probability y t n+1 = arg max w∈V p t,n (w) where V is the dictionary.

Unless the word y t n+1

is an EOS token, we repeat generating the next word by feeding y t n+1 to the decoder.

Obviously, there is a large gap in knowledge selection accuracy between training with or without true labels (e.g. 23.2 of E2E Transformer MemNet with labels vs 4.8 of PostKS without labels in Table 2 ).

As one way to take advantage of true labels for training of latent models, prior research has employed auxiliary losses over latent variables (Wen et al., 2017; Zhao et al., 2017) .

Similarly, we use the knowledge loss from Dinan et al. (2019) (i.e. the cross-entropy loss between predicted and true knowledge sentences) as an auxiliary loss for the latent variable.

Thus, the training objective is a combination of the variational lower-bound from Eq. (3) and the auxiliary knowledge loss as

where k t s is a sampled knowledge from q φ (k t |x ≤t , y ≤t , k <t ), k t a is a true knowledge, and λ is a hyperparameter.

Note that knowledge is sequentially sampled from attention distribution as in Eq.(5).

We train our model by mini-batch gradient descent.

We approximate the expectation of one sample from the posterior by using Gumbel-Softmax function (Jang et al., 2017; Maddison et al., 2017b) .

Further details of optimization can be found in Appendix.

We evaluate our model mainly on the Wizard of Wikipedia (Dinan et al., 2019) and additionally Holl-E (Moghe et al., 2018) as another knowledge-grounded chit-chat dataset.

We qualitatively and quantitatively compare our approach with other state-of-the-art models.

Wizard of Wikipedia.

It contains 18,430 dialogues for training, 1,948 dialogues for validation and 1,933 dialogues for test.

The test set is split into two subsets, Test Seen and Test Unseen.

Test Seen contains 965 dialogues on the topics overlapped with the training set, while Test Unseen contains 968 dialogues on the topics never seen before in training and validation set.

Holl-E. It contains 7,228 dialogues for training, 930 dialogues for validation and 913 dialogues for test.

A single document is given per dialogue; the documents include about 58 and 63 sentences on average for training/validation and test set, respectively.

The dataset provides spans in the document as additional information to provide which parts of a document is used to generate a response.

However, the span labels are highly inconsistent; for example, they are often shorter than a single sentence or contain multiple consecutive sentences.

Thus, it is undesirable to use them without modifications because it is different from WoW setting where all of the ground-truth (GT) knowledge are in the form of sentence.

Hence, we collect a set of ground-truth (GT) knowledge as follows.

We select the sentence that includes the span as the GT knowledge sentence.

If the span is given over multiple sentences, we select the minimum number of consecutive sentences containing the span and use them as GT.

If all sentences have zero F1 scores to the span and the response, we tag no passages used as the GT, which amounts to 5% of GT labels.

We will make our set of GT annotations public.

Evaluation Metrics.

We follow the evaluation protocol of WoW (Dinan et al., 2019) .

We measure unigram F1 (R-1), bigram F1 (R-2) and perplexity (PPL) for response generation, and the accuracy for knowledge selection.

For n-gram metrics, we remove all the punctuations and (a, an, the) before computing the score.

We remind that lower perplexity and higher n-gram (R-1, R-2) scores indicate better performance.

The test set for Holl-E is split into two subsets, single reference and multiple references.

The dataset basically provides a single response per context (denoted as single reference).

However, for some conversations, more responses (e.g. 2-13) are collected from multiple annotators per context (multiple references).

For evaluation of multiple references, we take the best score over multiple GTs by following Moghe et al. (2018) .

For knowledge accuracy, we regard the model's prediction is correct if it matches at least one of the correct answers.

Baselines.

We closely compare with two state-of-the-art knowledge-grounded dialogue models.

The first one is E2E Transformer MemNet (Dinan et al., 2019) , which uses a Transformer memory network for knowledge selection and a Transformer decoder for utterance prediction.

The second one is PostKS (Lian et al., 2019) , which uses the posterior knowledge distribution as a pseudo-label for knowledge selection.

For fair comparison, we replace all GRU layers in PostKS with Transformers 1 .

We also compare with four variants of these models as an ablation study: (i) E2E Transformer MemNet + BERT, where we replace the Transformer memory network with pre-trained BERT, (ii) PostKS + Knowledge loss, where we additionally use the knowledge loss, (iii) E2E Transformer MemNet + BERT + PostKS, which combines all the components of baselines, and (iv) E2E Transformer MemNet + BERT + PostKS + Copy, where we additionally use copy mechanism with Transformer decoder.

We use official BERT tokenizer to tokenize the words and use pre-defined BERT vocabulary (V = 30522) to convert token to index 2 .

All the baselines use the exactly same inputs with our model except PostKS, which does not make use of knowledge labels as proposed in the original paper.

Table 2 compares the performance of different methods on the Wizard of Wikipedia dataset.

Our model outperforms the state-of-the-art knowledge-grounded dialogue models in all metrics for knowledge selection (accuracy) and utterance generation (unigram F1, bigram F1) .

The PostKS that is trained with no knowledge label shows low accuracy on knowledge selection, which is slightly better than random guess.

However, it attains better performance than E2E Transformer MemNet with the knowledge loss in the WoW Test Seen, which shows that leveraging prior and posterior knowledge distribution is effective for knowledge-grounded dialogue, although using sequential latent variable improves further.

BERT improves knowledge selection accuracy, but not much as in TextQA because of diversity in knowledge selection of conversation.

The E2E Transformer MemNet + BERT + PostKS + Copy performs the best among baselines, but not as good as ours, which validates that sequential latent modeling is critical for improving the accuracy of knowledge selection and subsequently utterance generation.

Additionally, the performance gaps between ours and baselines are larger in Test Unseen.

It can be understood that the sequential latent variable can generalize better.

Adding copy mechanism to the baseline substantially improves the accuracy of utterance generation, but barely improves the knowledge selection accuracy, which also justifies the effectiveness of the sequential latent variable.

Transformer (no knowledge) shows the lowest perplexity in the WoW Test Seen, and it is mainly due to that it may generate only general and simple utterances since no knowledge is grounded.

This behavior can be advantageous for the perplexity, while the other knowledge-based models take a risk of predicting wrong knowledge, which is unfavorable for perplexity.

Table 3 compares the performance of our model on Holl-E dataset.

Similarly, our model outperforms all the baselines in all metrics.

One notable trend is that BERT considerably reduces the perplexity in all models, which may be due to that the dataset size of Holl-E is much smaller than WoW and BERT prevents overfitting (Hao et al., 2019) .

User Studies.

We perform human evaluation to complement the limitation of automatic language metrics.

We evaluate several aspects of utterance generation using the similar setting in Guu et al. (2018) .

We randomly sample 100 test examples, and each sample is evaluated by three unique human annotators on Amazon Mechanical Turk (AMT).

At test, we show dialogue context and generated utterance by our method or baselines.

We ask turkers to rate the quality of each utterance in two aspects, which are referred to Li et al. (2019a) : (i) Engagingness:

how much do you like the response?

and (ii) Knowledgeability: how much is the response informative?

Each item is scored from 1 to 4 to avoid "catch-all" category in the answer (Dalal et al., 2014) , where 1 means not at all, 2 is a little, 3 is somewhat, and 4 is a lot.

To mitigate annotator bias and inter-annotator variability, we adjusted human evaluation results with Bayesian calibration (Kulikov et al., 2019) .

Note that human evaluation on knowledge selection is not possible, since any knowledge can be fine for a given context, which is key motivation for our sequential latent model -diversity of knowledge selection.

Table 4 summarizes the results of the human evaluation, which validates that annotators prefer our results to those of baselines.

Again, the performance gaps between ours and baselines are larger in Test Unseen, thank to better generality of our sequential latent model.

However, the gaps in Test Seen are not large, since the evaluation is not done in a multi-turn setting, for which our sequential model's merit would be more salient, due to the difficulty of multi-turn tasks for AMT turkers.

Dialogue Examples.

Figure 3 shows selected examples of utterance prediction.

In each set, we show dialogue context, human response, and utterances generated by our method and baselines.

Thanks to the use of latent variables, our model can better capture the changes in dialogue topics and thus generate more appropriate responses.

Knowledge-based conversations have been studied much including collecting new datasets (Qin et al., 2019; Zhang et al., 2018; Ghazvininejad et al., 2018; Zhou et al., 2018; Dinan et al., 2019; Moghe et al., 2018) or developing new models (Lian et al., 2019; Li et al., 2019b; Yavuz et al., 2019; Zhao et al., 2019b; Dinan et al., 2019; Liu et al., 2019) .

Most works on the models have less investigated the knowledge selection issue but instead focused on how to effectively combine given knowledge and dialogue context to improve response informativeness.

For example, Ghazvininejad et al. (2018) aid a Seq2Seq model with an external knowledge memory network, and Li et al. (2019b) propose an Incremental Transformer to encode multi-turn utterances along with knowledge in related documents.

Recently, Dinan et al. (2019) propose both a dataset of Wizard of Wikipedia and a model to leverage the two-step procedure of selecting knowledge from the pool and generating a response based on chosen knowledge and given context.

One of the most related models to ours may be Lian et al. (2019) , who also focus on the knowledge selection issue in the two-stage knowledge-grounded dialogue.

However, our work is novel in that we model it as a sequence decision process with latent variables and introduce the knowledge loss.

Thanks to these updates, our model achieves significantly better performance as shown in the experiments.

Sequential Latent Variable Models.

There have been many studies about sequential latent variable models.

Chung et al. (2015) propose one of the earliest latent models for sequential data, named VRNN.

Later, this architecture is extended to SRNN (Fraccaro et al., 2016) and Z-Forcing (Goyal et al., 2017) .

There have been some notable applications of sequential latent models, including document summarization (Li et al., 2017) , image captioning (Aneja et al., 2019 ) and text generation (Shao et al., 2019) .

Another related class of sequential latent models may be latent attention models (Deng et al., 2018; Wang et al., 2018; Yang et al., 2017) , which exploit the latent variables to model the attention mapping between input and output sequences.

Although our method is partly influenced by such recent models, it is novel to propose a sequential latent model for the knowledgegrounded chit-chat problem.

This work investigated the issue of knowledge selection in multi-turn knowledge-grounded dialogue, and proposed a sequential latent variable model, for the first time, named sequential knowledge transformer (SKT).

Our method achieved the new state-of-the-art performance on the Wizard of Wikipedia benchmark (Dinan et al., 2019) and a knowledge-annotated version of Holl-E dataset (Moghe et al., 2018) .

There are several promising future directions beyond this work.

First, we can explore other inference models such as sequential Monte Carlo methods using filtering variational objectives (Maddison et al., 2017a) .

Second, we can study the interpretability of knowledge selection such as measuring the uncertainty of attention (Heo et al., 2018) .

A:

That is true but we always have to watch out for excessive hunting.

It has caused some species to be endangered.

Yes I agree.

I don't believe in the useless hunting that poachers do.

Its so cruel.

Table 6 : Quantitative results of our proposed model with partial knowledge labels on the Wizard of Wikipedia dataset (Dinan et al., 2019) . (Kingma & Ba, 2015) with β 1 = 0.9, β 2 = 0.999, = 1e − 07.

For the models without BERT, we set the learning rate to 0.001 and initialize the embedding matrix with fastText (Bojanowski et al., 2016) trained on the Common Crawl corpus.

For the models with BERT, we set the learning rate to 0.00002 and initialize encoder weights with BERT-Base, Uncased pretrained weights.

We apply label smoothing (Pereyra et al., 2017; Edunov et al., 2017; Vaswani et al., 2017) for both knowledge selection and utterance generation, and set 0.1 and 0.05 for each.

We set the temperature of Gumbel-Softmax to τ = 0.1 and the hyperparameter for the knowledge loss to λ = 1.0.

For efficiency, we batch the dialogues rather than individual turns.

We train our model up to 5 epochs on a single NVIDIA TITAN Xp GPU.

C KNOWLEDGE SELECTION ACCURACY OVER TURNS Table 5 compares the knowledge selection accuracy of different methods for each turn on the Wizard of Wikipedia.

Thanks to sequential latent variable, our model consistently outperforms other models for all turns in knowledge selection accuracy.

Notably, in all models, the accuracy significantly drops after the first turn (which is often easily predictable topic definition sentence), which shows the diversity nature in knowledge selection, as discussed in Section 2.

D QUANTITATIVE RESULTS ON SEMI-SUPERVISED SETTING Table 6 shows results of our model with partial knowledge labels on the Wizard of Wikipedia.

Results show that the better performance is attained with more labeled knowledge data for training as expected.

Furthermore, our model achieves competitive performance with less label.

For instance, our model using only 1/4 labeled training data is comparable to E2E Transformer MemNet and is even better in Test Unseen.

As a result, our sequential latent knowledge selection model can be utilized in a semi-supervised method without severe drop in its performance. (Vaswani et al., 2017) and GRU (Cho et al., 2014)

We add human evaluation results in a multi-turn setting using the evaluation toolkit from Wizard of Wikipedia (Dinan et al., 2019) .

Following their setting, humans are paired with one of the models and chat about a specific topic (given a choice of 2-3 topics) for 3-5 dialogue turns.

After conversation, they rate their dialogue partner on a scale of 1-5, with the rating indicating how much they liked the conversation.

We collect the votes for 110 randomly sampled conversations from 11 different turkers.

Figure 4 and 5 show selected examples of knowledge selection and utterance prediction.

In each set, we show dialogue context, human response, and selected knowledge and generated utterances by our method and baselines.

<|TLDR|>

@highlight

Our approach is the first attempt to leverage a sequential latent variable model for knowledge selection in the multi-turn knowledge-grounded dialogue. It achieves the new state-of-the-art performance on Wizard of Wikipedia benchmark.

@highlight

A sequential latent variable model for knowledge selection in dialogue generation that extends the posterior attention model to the latent knowledge selection problem and achieves higher performances than previous state-of-the-art models.

@highlight

A novel architecture for selecting knowledge-grounded multi-turn dialogue that yields state of the art on relevant benchmarks datasets, and scores higher in human evaluations.