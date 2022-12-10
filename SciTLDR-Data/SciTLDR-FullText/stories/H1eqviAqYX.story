Recent advances in neural Sequence-to-Sequence (Seq2Seq) models reveal a purely data-driven approach to the response generation task.

Despite its diverse variants and applications, the existing Seq2Seq models are prone to producing short and generic replies, which blocks such neural network architectures from being utilized in practical open-domain response generation tasks.

In this research, we analyze this critical issue from the perspective of the optimization goal of models and the specific characteristics of human-to-human conversational corpora.

Our analysis is conducted by decomposing the goal of Neural Response Generation (NRG) into the optimizations of word selection and ordering.

It can be derived from the decomposing that Seq2Seq based NRG models naturally tend to select common words to compose responses, and ignore the semantic of queries in word ordering.

On the basis of the analysis, we propose a max-marginal ranking regularization term to avoid Seq2Seq models from producing the generic and uninformative responses.

The empirical experiments on benchmarks with several metrics have validated our analysis and proposed methodology.

Past years have witnessed the dramatic progress on the application of generative sequential models (also noted as seq2seq learning (Sutskever et Despite these promising results, current Sequence-to-Sequence (Seq2Seq) architectures for response generation are still far from steadily generating relevant and coherent replies.

The essential issue identified by many studies is the Universal Replies: the model tends to generate short and general replies which contain limited information, such as "That's great!", "I don't know", etc.

Nevertheless, most previous analysis over the issue are empirical and lack of statistical evidence.

Therefore, in this paper, we conduct an in-depth investigation on the performance of seq2seq models on the NRG task.

In our inspections on the existing dialog corpora, it is shown that those repeatedly appeared replies have two essential traits: 1) Most of them are composed of highly frequent words; 2) They cover a large portion of the dialog corpora that each universal reply stands for the response of various queries.

Above characteristics of universal replies deviate the NRG from other successful applications of sea2seq model such as translation, and lead current generative NRG models to prefer common replies.

To discuss the influences from the specific distributed corpus, we decompose the target sequence's probability into two parts and analyze the probability respectively.

To break down the mentioned characteristics of dialog corpora in the model training step, we propose a ranking-oriented regularization term to prune the scores of those irrelevant replies.

Experimental results reveal that the model with such regularization can produce better results and avoid generating ambiguous responses.

Also, case studies show that the issue of generic response is alleviated that these common responses are ranked relatively lower than more appropriate answers.

The main contributions of this paper are concluded as follows: 1) We analyze the loss function of Seq2seq models on NRG task and conclude several critical reasons that the NRG models prefer universal replies; 2) Based on the analysis, a max-marginal ranking regularization is presented to help the model converge to informative responses.

Different from significant advances in machine translation BID1 and abstractive summarization (Rush et al., 2015; Nallapati et al., 2016), it remains challenging to apply Seq2Seq models in practical response generation.

One widely accepted issue within current models is that Seq2Seq architectures are inclined to produce common and unrelated replies, even when the quality of training data is significantly improved and different Seq2Seq variants are proposed.

The primary reason for this phenomenon lies in the fact that the semantic constraint from query to the possible responses is naturally weak, since the responses to a given query are not required to be semantically equivalent.

In contrast, the references in machine translation or summarization are usually restricted to be equivalent to each other semantically or even lexically.

Especially, for machine translation, words that appear in the target language should satisfy word level mapping from the source sentence, so the learned word alignment function could ensure the model to generate suitable translated words.

Different from learning the semantic alignments between languages in NMT, in NRG the replies can be diversified as they only need to satisfy the causality with the given queries.

Moreover, given a query, the sequential model is optimized to learn the shared information among all replies, thus the model is more likely to choose those high-frequent common replies, which is also mentioned in BID13 .Taking the case in TAB0 for example, the topic of this query is about movie.

It can be observed that the replies shown in the table are semantically diversified: the first two replies are related to the opinion of the respondent toward the movie, while the rest of the replies are about the director, content, and origin of the movie.

By contrast, the two valid translations in French are very similar regarding their semantics, which can be attributed to the fixed word-level mapping between query and targets.

The sequence-mapping problem in NRG can be decomposed into two independent sub-learning problems: 1) Target word selection, in which a query is summarized and translated into the semantic space of responses, and then a set of target words is selected to represent the meaning; 2) Word ordering, in which a grammatical coherent reply is generated based on the candidate word set BID25 .

The word selection and ordering of the target sequence are jointly learned which can TAB0 .

also be reflected in the model's loss function by two possible factored phases: DISPLAYFORM0 where x stands for the given query and y is the corresponding response with n words.

Besides, S(y) = {w 1 , · · · , w n |w i ∈ y, i ∈ [1, n]} represents all predicted words without sequential order, so p(S(y)|x) is referred as the probability of the target word selection.

Meanwhile, p(y|S(y), x) indicates the probability of word ordering given this group of possible words.

Thus, the objective can be redescribed from maximizing the probability of the ground truth response y under query x to maximizing these two joint probabilities simultaneously.

After the above interpretation, we will further discuss the impact of the implicative constriction from two separated probabilities in Eq. 1, which results in the potential failure of models in learning conversational patterns.

Assuming that we have a set of K ground-truth replies: {y 1 , · · · , y K } to a given query x, the upper bound of the target word selection probability can be derived via Jensen's Inequality BID2 : DISPLAYFORM0 where ∪ K k S(y k ) denotes all the words appearing in the entire response set, and DISPLAYFORM1 Thus, optimizing the first segment is proportional to maximizing the last conditional probabilities, and the optimal strategy is to assign probabilities according to the frequency of words in these K responses.

Such strategy adopted by Seq2Seq can be verified by the long-tailed distribution of words in FIG2 , in which only few common words are assigned with preferred high probabilities.

Given that, during the inference, the best strategy is to employ more frequently occurring words rather than rare ones such as "background," "art," and "director" in TAB0 .Furthermore, assuming that each response contains a fixed number of T words (so that 1 ≤ L S ≤ K × T ), we can find that the probability of each response for x is inversely proportional to K: DISPLAYFORM2 where E(w|x) denotes the mean frequency of words appeared in these K replies, which is 1.32 for the cases in TAB0 .

In general, the mean frequency is around 1 owing to the long-tailed Unigram distribution which satisfies Zipf's law (Zipf, 1935) .

In other words, the target word selection probability is limited by K, so queries with more diverse answers are more challenging to learn.

Meanwhile, it is difficult to obtain good predictions for lower-informational queries, as they contain more possible responses which are somewhat equivalent to a larger K (Li et al., 2016a).Nonetheless, the translation task requires word-level mappings as they are well-aligned in the semantic space, therefore source and target sentences are semantically equivalent.

So that, translated candidates are confined to K ≈ 1.

Thus the upper bound can be approximated as the full probability.

Before discussing the word ordering probability, we present four lemmas and corresponding proofs.

Moreover, all these lemmas are only available for the response generation task except Lemma 1.According to the Zipf's law (Zipf, 1935) , the frequency of any word is inversely proportional to its rank in the frequency table, such that the probability p(w i ) = Z/i α , where Z ≈ 0.1, α ≈ 1, and i is the frequency rank of the word w i .

Then, denoting the vocabulary size as V and the total number of query-response pairs as N , we can formulate two characteristics of a universal reply y as follows:1) A response is universal if it consists of only top-t ranked words.

For any word w in such response, p(w) ≥ 1/(10t) according to the Zipf's law.2) The amount of possible queries M of y is directly proportional to the size of query-response pairs DISPLAYFORM0 To simplify, we suppose that t > 1000 to cover most universal replies, and the frequency of the response not belonging to the universal replies is a constant c (1 ≤ c M ).

Accordingly, we can derive the following lemmas.

DISPLAYFORM1 Proof.

Lemma 1 describes the obvious fact that the event "the word set of the response equals to S(y)" must happen when the event "y stands for the response" is established.

Lemma 2 p(x|y ur ) = 1 , where 1 > 0 and is sufficiently small, and y ur is a universal reply.

Proof.

Based on the second character of the universal reply and the fact that N is a very large number for any large scaled datasets, Lemma 2 is established as: x dx = ln(t + 1), we can get the conclusion that the probability of a chosen word belonging to the most frequent t words is large than 0.1 * ln(t + 1) > 0.69.

Since y contains T words, there is at least T ln(t + 1) words belonging to the top-t ranked on average according to the binomial distribution.

DISPLAYFORM2 We suppose m responses are universal replies among the n possible responses when their words are constrained by S(y).

Besides, the proportion of m can be computed as: DISPLAYFORM3 where C donates the combination.

Since n/m is not a very large number, the total probability of these m replies can be deducted as: DISPLAYFORM4 where f (y) donates the frequency of a response y in the corpus.

According to the Eq. 5 and the fact that M ∝ N is a very large number for any practical large-scale datasets, i p(y ur i |S(y)) → 1 can be established.

Apparently, for any other candidate response y o j , its probability satisfies DISPLAYFORM5 Lemma 4 Assuming each informative query has K ground-truth replies and the query-response pairs are extracted from a multi-turn conversational corpus, a reply y not belonging to universal replies has K unique queries, noted as p(x|y) = 1 K .

Proof.

Most query-response pairs are extracted from a practical large-scale multi-turn conversational corpus, so that any response always works as the post in another pair.

That is, y also appears K times as it also has K replies.

Therefore, there also exist K unique posts for y.

On the basis of Lemma 1, the word ordering probability could be deducted as: DISPLAYFORM0 All the possible y i satisfying S(y i ) ⊆ S(y) can be divided into three categories: ground-truth reply y, universal replies y ur and other replies y o .

From above, we can get the following direct proportion according to the Lemma 2 and Lemma 3, On the basis of Eq. 7 and Lemma 4, for any reply y not belonging to universal replies, the Eq. 6 can be further deducted as: DISPLAYFORM1 where = 1 + 2 > 0, which is also a sufficiently small positive value.

Thus, optimizing the word ordering probability for the non-universal replies is partially equivalent to maximizing p(y|S(y)).In fact the term p(y|S(y)) is the language model probability and it is irrelevant with the query x FIG6 ).

In the sequential models, it is performed as t p(y t |y 1:t−1 , S(y)), in other words the sequences are generated based only on previously outputted words.

This equation indicates that optimizing the mainly seeks the grammatical competence based on the selected words.

In conclusion, the insufficient constraint of the target words' cross-entropy loss in NRG is the primary reason that hinders seq2seq models from exploring presumable parameters.

This situation is mainly caused by the particular distribution of NRG corpus, since there exist many universal replies composed of high-frequent words in corpus.

Consequently, the model tends to promotes such universal replies, regardless of the given query.

As discussed above, various responses corresponding to the same query appearing in the training data leads to the undesired preference of NRG on universal replies, so an intuitive solution is removing the multiple replies and just keeping one-to-one pairs.

However, filtering the training dataset in large scale raises the difficulty of model training.

Besides, naively removing the multiple replies is detrimental to the reply diversity, which is important in NRG task.

As shown in TAB0 , an ideal chatbot agent is prospected to provide all listed replies and build a connection with some keywords such as 'film', 'background', 'director' and 'book', rather than other commonly appeared words like 'I', 'him', 'a' and 'really'.Thus, under this assumption, we propose a max-marginal ranking loss to emphasize the queries' impact on these less common but relevant words.

During training, as it becomes a necessity to constrain the learned feature space and reinforce related replies with more discriminative information, we classify the candidate responses into two categories: positive (i.e., highly related) and negative (i.e., irrelevant) answers.

A training instance is re-constructed as a triplet (x, y, y − ), where a tuple (x, y) is the original query-response pair and noise y − is uniformly sampled from all of the responses in the training data.

Given that, the model's loss function is reconstructed as: DISPLAYFORM0 where γ > 0, log p(y|x) denotes the cross-entropy loss between the model's prediction and ground truth sequences, and the second part encourages the separation between the irrelevant responses and related replies.

Moreover, the hyper-parameter λ defines the penalty for the seq2seq loss, it offers a degree of freedom to control the importance of the max-marginal between the positive and negative instances.

The model is trained in the same setting as the conventional model when λ = 0.The gradient of θ is computed using the sub-gradient method, as the second term is nondifferentiable but convex BID0 .

Supposing log p(y|x) − log p(y − |x) ≤ γ, the gradient of the composed loss function can be formalized as: DISPLAYFORM1 If log p(y|x) − log p(y − |x) > γ, then the gradient should be written as: DISPLAYFORM2 The underlying motivation of our proposed loss function is based on three considerations: 1) Universal replies are more likely to be sampled from a statistical perspective, so adding a negative term would directly ease the weight of these generic responses, and the ranking regularization can penalize those irrelevant responses; 2) Positive and negative sentences overall share a same set of generic words, which suggests that the loss optimization should pay more attention on those different words rather than generic ones; 3) Only differentiable loss can solely be served as the model's optimization goal for the sequence generation model.

Furthermore, the newly proposed loss aims to penalize frequent words and irrelevant candidates, rather than repudiating the literal expression included in negative samples.

Consequently, based on these considerations, we propose this term as a regularization to constrain the search space of parameters instead of the stand-alone loss function.

The dataset used in this study contained almost ten million query and response pairs collected from a popular Chinese social media site: Douban Group Chat 1 .

All case studies used in this paper were extracted from this dataset and translated into English.

For easier training and better efficiency, the maximal lengths of queries and replies were set to 30 and 50 respectively.

In all of our experiments, our dataset was split into the training, validation and test sets, with detailed statistical characterization given in TAB1 .

Thirty percent of queries had more than one responses, and each answer appeared about 1.33 times in the training dataset, which is consistent with our hypothesis in the analysis section.

To validate the performance of the proposed model, the following baselines were considered:• S2SA: The basic seq2seq model with attention mechanism BID1 at the target output side.• S2SA + MMI: The best performing model in BID6 with the length norm based on the same S2SA.• Ranking-Reg: The seq2seq model with proposed ranking regularization and attention.

In this model, negative samples were uniformly sampled from the corpus, and the process was repeated 4 times for every positive case.

The averaged negative loss was calculated as the probability of universal replies.• Ranking-Reg + MMI: Ranking-Reg with MMI during inference procedure.

The quality of response was measured using both numeric metrics and human annotators.

Firstly, Word Perplexity (PPL) is used to measure the model's ability to account for the syntactic structure for each utterance (Serban et al., 2016).

Secondly, ROGUE score BID9 , which evaluates the extent of overlapping words between the ground-truth and predicted replies, was also adopted in experiments.

Thirdly, we employed the widely used diversity measurements Distinct-1 and Distinct-2 to evaluate the number of distinct Unigrams and Bigrams of generated responses BID6 ).Furthermore, we recruited three highly educated human annotators to cross verify the quality of generated responses.

We randomly sampled 100 queries and generated 10 replies for each query DISPLAYFORM0 The response cannot be used as a reply to the message.

It is either semantically irrelevant or not fluent (e.g., with grammatical errors or UNK).

The response can be used as a reply to the message, which includes the universal replies such as "Yes, I see" , "Me too" and "I dont know".

The response is not only relevant and natural, but also informative and interesting. , and the initial learning rate was 1e-4.

All of the models were implemented in Theano (Theano Development Team, 2016), and each ran on a standalone K40m GPU device for 7 epochs, which took 7 days; twice longer time was required for training models with rank regularization.

The last two models with the rank regularization share the related hyper-parameters.

We set λ to 0.1 and γ to 0.18, according to the model's performance on the validation set.

FIG6 shows cross-entropy loss flows vs. training epoch numbers.

The model with max-marginal ranking regularization converges faster than S2SA throughout the training.

This shows that the additional regularization term helps to speed up the fitting by removing these sub-optimal paths.

The performance of four models on existing metrics is summarized in TAB2 .

The model with the max-marginal ranking regularization outperforms the model with primary loss function on the target loss PPL.

As the MMI method is performing during inference, losses of models with MMI are identical to those without revision.

However, the results are opposite regarding the ROGUE scores.

The generated responses by the S2SA model contain more words appearing in the ground truth answers.

These experimental results can be attributed to mainly two factors.

a) The very low ROUGE scores reflect few words shared by any predictions and the ground truth.

Most n-gram overlaps belonging to the common words, such as "I", "are", "that".

b) A certain proportion of replies in the test set are universal themselves.

Therefore, S2SA has achieved higher ROUGE score as its' results are more consistent with those common ground truth responses.

University are far away, and the city's most famous commercial street are near to me.

Query:Replies from S2S+Attention:Replies from S2S+Attention: Replies from Ranking Loss :Replies from Ranking Loss : Figure 3 : Response re-rank capability.

Responses generated by the basic model and model with rank loss are linked by arrows, and same topics are typeset using the same color.

Some ungrammatical and incomprehensible sentences exist due to the translating try to keep the word order.

The human evaluation is the most important metric, and it is clear from TAB2 that the models with rank regularization beat S2SA with a large margin.

It increases the number of meaningful responses by around 10% and reduces the number of irrelevant cases by around 4%.

Meanwhile, most the acceptable replies (labeled as "1" or "2") of S2SA is labeled as "1", which indicates the model prefer the safe responses.

We attribute the gaps to the promotion of highly related words and reducing of the universal replies.

Same trend can be also spotted on Distinct-1 and Distinct-2, it reveals the model's ability to generate diverse responses BID6 BID15 .

The seq2seq model yields lower levels of unigram and bigram diversity than the rank loss model.

As another comparison, we note that the improvement introduced by MMI is much smaller than that introduced by the ranking regularization, whereas MMI is a widely used mechanism for promoting diverse responses during inference.

Besides, performing it upon the regularization reduces the rate of informative and interesting responses.

This observation indicates that the fundamental reason behind generating tasteless or inappropriate replies is that Seq2Seq model learned from conversational corpora prefers universal replies.

Moreover, the revision during the greedy search is less effective on solving the underlying problems than the proposed ranking regularization.

From the generated results, it is found that the seq2seq model with the ranking regularization term prefers meaningful content when the query contains sufficient amount of information.

We present top responses for two queries generated by different models in Fig. 3 .

As shown in the first case, user posts a query which initiates a complicated discussion about locations.

It is observed that S2SA converges to a typical "where is your" pattern of replies when discussing locations, which is an example of universal replies.

As the greedy beam search strategy is utilized during inference, many location-related constraints further promote these relevant universal replies instead of more varied results from different beams.

In contrast, some of the responses in the right column captured the "commercial street" clues and inferred a possible location "Joy City shopping mall" demoting the generic beams results.

We attributed this to the boosting ability associated with semantically relevant words, as mentioned in Section 3.The second case is quite different.

In this case, the seq2seq model did not perform satisfactorily.

Even though the subject "bank" was extracted into the generated candidates, we cannot perceive the results aligned with the same "not reliable" topic, and most of them were just chosen from two beams.

Inspecting the replies generated by the rank loss model, we found that more complicated and diverse sentences that discuss "unreliable" can be generated, and irrelevant answers about "bank" are lower-ranked.

To further investigate the difference brought by the max-marginal ranking regularization, we randomly sampled more cases shown in the Fig. 4 as appendix.

Even though some of them were bad cases and contained some grammatical errors, overall the model with rank regularization tends to generate more informative and interesting sentences compared with baselines.

In conclusion, the seq2seq model with rank regularization can not only formulate the conditional language model but also boost related answers to higher ranks than the rest of universal or inappropriate replies.

Recent years have witnessed the rapid development of data-driven dialog models with the help of accumulated conversational data from online communities.

Query During inference, these models generate responses by first sampling an assignment of latent variables, so that models can generate more diverse responses.

Such methods attempt to improve the diversity of responses by modifying the Seq2Seq architecture, and our analysis may be also helpful to design more effective latent variable based models to restrain current problems.

Besides, the ranking penalty has also been used by BID27 , they employ a word-level margin to promote ground-truth sequences appearing in the beam search results.

Different from our method, they directly optimize the beam search procedure to fine-tune the trained model.

Eliminating generic responses is the essence for the widely practical utilization of the Seq2Seq based neural response generation architectures, and thus, this paper has conducted a thorough investigation on the cause of such uninformative responses and proposed the solution from the statistical perspective.

The main contributions of this work can be summarized as follows: a) The theoretical analysis is performed to capture the root reason of NRG models producing generic responses through the optimization goal of models and the statistical characteristics of human-to-human conversational corpora, which has been little studied currently.

In detail, we have decomposed the goal of NRG into the optimizations of word selection and word ordering, and finally derived that NRG models tend to select common words as responses and order words from the language model perspective which ignores queries.

b) According to the analysis, a max-marginal ranking regularization term is proposed to cooperate with the learning target of Seq2Seq, so as to help NRG models converge to the status of producing informative responses, rather than merely manipulating the decoding procedure to constrain the generation of universal replies.

Furthermore, the empirical experiments on the conversation dataset indicate that the models utilizing this strategy notably outperform the current baseline models.

@highlight

Analyze the reason for neural response generative models preferring universal replies; Propose a method to avoid it.

@highlight

Investigates the problem of universal replies plaguing the Seq2Seq neural generation models

@highlight

The paper looks into improving the neural response generation task by deemphasizing the common responses using modification of the loss function and presentation the common/universal responses during the training phase.