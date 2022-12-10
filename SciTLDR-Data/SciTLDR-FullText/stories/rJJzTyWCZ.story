Cloze test is widely adopted in language exams to evaluate students' language proficiency.

In this paper, we propose the first large-scale human-designed cloze test dataset CLOTH in which the questions were used in middle-school and high-school language exams.

With the missing blanks carefully created by teachers and candidate choices purposely designed to be confusing, CLOTH requires a deeper language understanding and a wider attention span than previous automatically generated cloze datasets.

We show humans outperform dedicated designed baseline models by a significant margin, even when the model is trained on sufficiently large external data.

We investigate the source of the performance gap, trace model deficiencies to some distinct properties of CLOTH, and identify the limited ability of comprehending a long-term context to be the key bottleneck.

In addition, we find that human-designed data leads to a larger gap between the model's performance and human performance when compared to automatically generated data.

Being a classic language exercise, the cloze test BID26 is an accurate assessment of language proficiency BID7 BID11 BID27 and has been widely employed in language examinations.

Under standard setting, a cloze test requires examinees to fill in the missing word (or sentence) that best fits the surrounding context.

To facilitate natural language understanding, automatically generated cloze datasets were introduced to measure the ability of machines in reading comprehension BID8 BID9 BID17 .

In these datasets, each cloze question typically consists of a context paragraph and a question sentence.

By randomly replacing a particular word in the question sentence with a blank symbol, a single test case is created.

For instance, the CNN/Daily Mail BID8 take news articles as the context and the summary bullet points as the question sentence.

Only named entities are considered when creating the blanks.

Similarly, in Children's Books test (CBT) BID9 , the cloze question is obtained by removing a word in the last sentence of every consecutive 21 sentences, with the first 20 sentences being the context.

Different from the CNN/Daily Mail datasets, CBT also provides each question with a candidate answer set, consisting of randomly sampled words with the same part-of-speech tag from the context as that of the ground truth.

Thanks to the automatic generation process, these datasets can be very large in size, leading to significant research progress.

However, compared to how humans would create cloze questions, the automatic generation process bears some inevitable issues.

Firstly, the blanks are chosen uniformly without considering which aspect of the language phenomenon the question will test.

Hence, quite a portion of automatically generated questions can be purposeless or even trivial to answer.

Another issue involves the ambiguity of the answer.

Given a context and a blanked sentence, there can be multiple words that fit almost equally well into the blank.

A possible solution is to include a candidate option set, as done by CBT, to get rid of the ambiguity.

However, automatically generating the candidate option set can be problematic since it cannot guarantee the ambiguity is removed.

More importantly, automatically generated candidates can be totally irrelevant or simply grammatically unsuitable for the blank, resulting in again trivial questions.

Probably due to these unsatisfactory issues, it has been shown neural models have achieved comparable performance with human within very short time BID3 BID6 BID23 .

While there has been work trying to incorporate human design into cloze question generation BID30 , the MSR Sentence Completion Challenge created by this effort is quite small in size, limiting the possibility of developing powerful neural models on it.

Motivated by the aforementioned drawbacks, we propose CLOTH, a large-scale cloze test dataset collected from English exams.

Questions in the dataset are designed by middle-school and highschool teachers to prepare Chinese students for entrance exams.

To design a cloze test, teachers firstly determine the words that can test students' knowledge of vocabulary, reasoning or grammar; then replace those words with blanks and provide three candidate options for each blank.

If a question does not specifically test grammar usage, all of the candidate options would complete the sentence with correct grammar, leading to highly confusing questions.

As a result, human-designed questions are usually harder and are a better assessment of language proficiency.

Note that, different from the reading comprehension task, a general cloze test does not focus on testing reasoning abilities but evaluates several aspects of language proficiency including vocabulary, reasoning and grammar.

To verify if human-designed cloze questions are difficult for current models, we train dedicated models as well as the state-of-the-art language model and evaluate their performance on this dataset.

We find that the state-of-the-art model lags behind human performance even if the model is trained on a large external corpus.

We analyze where the model fails compared to human.

After conducting error analysis, we assume the performance gap results from the model's inability to use long-term context.

To verify this assumption, we evaluate humans' performance when they are only allowed to see one sentence as the context.

Our assumption is confirmed by the matched performances of the model and human when given only one sentence.

In addition, we demonstrate that human-designed data is more informative and more difficult than automatically generated data.

Specifically, when the same amount of training data is given, human-designed training data leads to better performance.

Additionally, it is much easier for the same model to perform well on automatically generated data.

In this section, we introduce the CLOTH dataset that is collected from English examinations, and study the assessed abilities of this dataset.

We collected the raw data from three free websites 2 in China that gather exams designed by English teachers.

These exams are used to prepare students for college/high school entrance exams.

Before cleaning, there are 20, 605 passages and 332, 755 questions.

We perform the following processes to ensure the validity of the data: 1.

We remove questions with an inconsistent format such as questions with more than four options; 2.

We filter all questions whose validity relies on external information such as pictures or tables; 3.

Further, we delete duplicated passages; 4.

On one of the websites, the answers are stored as images.

We use two OCR software, tesseract 3 and ABBYY FineReader 4 , to extract the answers from images.

We discard the question when results from the two software are different.

After the cleaning process, we obtain a dataset of 7, 131 passages and 99, 433 questions.

Since high school questions are more difficult than middle school questions, we divided the datasets into CLOTH-M and CLOTH-H, which stand for the middle school part and the high school part.

We split 11% of the data for both the test set and the dev set.

The detailed statistics of the whole dataset and two subsets are presented in TAB1 .

In order to evaluate students' mastery of a language, teachers usually design tests so that questions cover different aspects of a language.

Specifically, they first identity words in the passage that can Table 2 .Passage: Nancy had just got a job as a secretary in a company.

Monday was the first day she went to work, so she was very 1 and arrived early.

She 2 the door open and found nobody there.

"

I am the 3 to arrive."

She thought and came to her desk.

She was surprised to find a bunch of 4 on it.

They were fresh.

She 5 them and they were sweet.

She looked around for a 6 to put them in.

"

Somebody has sent me flowers the very first day!" she thought 7 .

"

But who could it be?" she began to 8 .

The day passed quickly and Nancy did everything with 9 interest.

For the following days of the 10 , the first thing Nancy did was to change water for the followers and then set about her work.

Then came another Monday.

11 she came near her desk she was overjoyed to see a(n) 12 bunch of flowers there.

She quickly put them in the vase, 13 the old ones.

The same thing happened again the next Monday.

Nancy began to think of ways to find out the 14 .

On Tuesday afternoon, she was sent to hand in a plan to the 15 .

She waited for his directives at his secretary's 16 .

She happened to see on the desk a half-opened notebook, which 17 : "In order to keep the secretaries in high spirits, the company has decided that every Monday morning a bunch of fresh flowers should be put on each secretarys desk."

Later, she was told that their general manager was a business management psychologist.

Questions: DISPLAYFORM0 A Table 2 : A Sample passage from our dataset.

The correct answers are highlighted.

To understand the assessed abilities on this dataset, we divide questions into several types and label the proportion of each type of questions.

We find that the questions can be divided into the following types:• Grammar: The question is about grammar usage, involving tense, preposition usage, active/passive voices, subjunctive mood and so on.• Short-term-reasoning: The question is about content words and can be answered based on the information within the same sentence.• Matching/paraphrasing: The question is answered by copying/paraphrasing a word.• Long-term-reasoning: The answer must be inferred from synthesizing information distributed across multiple sentences.

We sample 100 passages in the high school category and the middle school category respectively.

Each passage in the high school category has 20 questions and each passage in the middle school category has 10 questions.

The types of the 3000 question are labeled on Amazon Turk.

We pay $1 and $0.5 for high school passage and middle school passage respectively.

The proportion of different questions is shown in TAB4 .

We find that the majority of questions are short-term-reasoning questions, in which the examinee needs to utilize grammar knowledge, vocabulary knowledge and simple reasoning to answer the questions.

Note that questions in middle school are easier since they have more grammar questions.

Finally, only approximately 22.4% of data needs long-term information, in which the long-term-reasoning questions constitute a large proportion.

In this section, we study if human-designed cloze test is a challenging problem for state-of-the-art models.

We find that the language model trained on large enough external corpus could not solve the cloze test.

After conducting error analysis, we hypothesize that the model is not able to deal with long-term dependencies.

We verify the hypothesis by evaluating human's performance when human only see one sentence as the context.

LSTM To test the performance of RNN based supervised models, we train a bidirectional LSTM BID10 to predict the missing word given the context, with only labeled data.

The implementation details are in Appendix A.1.Attention Readers To enable the model to gather information from a longer context, we augment the supervised LSTM model with the attention mechanism BID1 , so that the representation at the blank is used as a query to find the relevant context in the document and a blank-specific representation of the document is used to score each candidate answer.

Specifically, we adapt the Stanford Attention Reader BID3 and the position-aware attention model BID29 to the cloze test problem.

With the position-aware attention model, the attention scores are based on both the context match and the distances of two words.

Both attention models are trained only with the human-designed blanks just as the LSTM model.

Language model Language modeling and cloze test are similar since, in both tasks, a word is predicted based on the context.

In cloze test, the context on both sides may determine the correct answer.

Suppose x i is the missing word and x 1 , · · · , x i−1 , x i+1 , · · · , x n are the context.

Although language model is trained to predict the next word only using the left context, to utilize the surrounding context, we could choose x i that maximizes the joint probability p(x 1 , · · · , x n ), which essentially maximizes the conditional likelihood p( DISPLAYFORM0 .

Therefore, language model can be naturally adapted to cloze test.

In essence, language model treats each word as a possible blank and learns to predict it.

As a result, it receives more supervision than the supervised model trained on human-labeled questions.

Additionally, it can be trained on a very large unlabeled corpus.

Interested in whether the state-ofthe-art language model can solve cloze test, we first train a neural language model on the training set of our corpus, then we test the language model trained on One Billion Word Benchmark BID2 ) (referred as 1-billion-language-model) that achieves a perplexity of 30.0 BID13 5 .

To make the evaluation time tractable, we limit the context length to one sentence or three sentences.

Human performance We measure the performance of Amazon Turkers on 3, 000 sampled questions when the whole passage is given.

The comparison is shown in Table 4 .

Both attention models achieve a similar accuracy to the LSTM.

We hypothesize the attention model's unsatisfactory performance is due to the difficulty to learn to comprehend longer context when the majority of the training data only requires understanding short-term information.

The language model trained on our dataset achieves an accuracy of 0.548 while the supervised model's accuracy is 0.484, indicating that more training data results in better generalization.

When only one sentence is given as context, the accuracy of 1-billion-languagemodel is 0.695, which shows that the amount of data is an essential factor affecting the model's performance.

It also indicates that the language model can learn sophisticated language regularities when given enough data.

The same conclusion can also be drawn from state-of-the-art results on six language tasks resulted from applying language model representations as word vectors BID0 .

However, if we increase the context length to three sentences, the accuracy of 1-billionlanguage-model only improves to 0.707.

In contrast, human outperforms 1-billion-language-model by a significant margin, which demonstrates that deliberately designed questions in CLOTH are not completely solved even for state-of-the-art models.

Table 4 : Model and human's performance on CLOTH.

Attention model does not leads to performance improvement compared to vanilla LSTM.

Language model outperforms LSTM since it receives more supervisions in learning to predict each word.

Training on large external corpus further significantly enhances the accuracy.

In this section, we would like to understand why the state-of-the-art model lags behind human performance.

We find that most of the errors made by the large language model involve long-term reasoning.

Additionally, in a lot of cases, the dependency is within the context of three sentences.

Several errors made by the large language model are shown in Table 5 .

In the first example, the model does not know that Nancy found nobody in the company means that Nancy was the first one to arrive at the company.

In the second and third example, the model fails probably because of the coreference from "they" to "flowers".

The dependency in the last case is longer.

It depends on the fact that "Nancy" was alone in the company.

Based on the case study, we hypothesize that the language model is not able to take long-term information into account, although it achieves a surprisingly good overall performance.

Moreover, the 1-billion-language-model is trained on the sentence level, which might also result in paying more attention to short-term information.

However, we do not have enough computational resources to train a large model on 1 Billion Word Benchmark to investigate the differences of training on sentence level or on paragraph level.

She smelled them and they were sweet.

She looked around for a to put them in.

A. vase B. room C. glass D. bottle "Somebody has sent me flowers the very first day!" "

But who could it be?" she began to .

The day passed quickly and Nancy did A. seek B. wonder C. work D. ask everything with great interest.

Table 5 : Error analysis of 1-billion-language-model with three sentences as the context.

The questions are sampled from the sample passage shown in Table 2 .

The correct answer is in bold text.

The incorrectly selected options are in italics.

An available comparison is to test the model's performance on different types of questions.

We find that the model's accuracy is 0.591 on long-term-reasoning questions of CLOTH-H while achieving 0.693 on short-term-reasoning, which partially confirms that long-term-reasoning is harder.

However, we could not completely rely on the performance on specific questions types, partly due to the small sample size.

A more fundamental reason is that the question type labels are subjective and their reliability depends on whether turkers are careful enough.

For example, in the error analysis shown in Table 5 , a careless turker would label the second example as short-term-reasoning without noticing that the meaning of "they" relies on a long context span.

To objectively verify if the language model's strengths are in dealing with short-term information, we obtain the ceiling performance of only utilizing short-term information.

Showing only one sentence as the context, we ask the turkers to label all possible options that they deem to be correct given the insufficient information.

We also ask them to select a single option based on their best guesses.

By limiting the context span manually, the ceiling performance with only the access to short context is estimated accurately.

The performances of turkers and 1-billion-language-model are shown in TAB8 .

The performance of 1-billion-language-model using one sentence as the context can almost match the ceiling performance of only using short-term information.

Hence we conclude that the language model can almost perfectly solve all short-term cloze questions.

However, the performance of language model is not improved significantly when the needed long-term context is given, indicating that the performance gap is due to the inability of long-term reasoning.

Assuming the majority of question type labels is reliable, we verify the strengths and weaknesses of models and human by studying the performance of models and human on different question categories.

The comparison is shown in Figure 1 .The human study on short-term ceiling performance also reveals that the options are carefully picked.

Specifically, when a Turker thinks that a question has multiple answers, 3.41 out of 4 options are deemed to be possibly correct, which means that teachers design the options so that three or four options all make sense if we only look at the local context.4 COMPARING HUMAN-DESIGNED DATA AND AUTOMATICALLY GENERATED DATAIn this section, we demonstrate that human-designed data is a better test bed than automatically generated data for general cloze test since it results in a larger gap between the model's performance and human performance.

However, the distributional mismatch between two types of data makes the human-designed data an unsuitable training source for solving automatically generated questions.

In addition, we improve the model's performance by finding generated data that resembles humandesigned data.

At a casual observation, a cloze test can be created by randomly deleting words and randomly sampling candidate options.

In fact, to generate large-scale data, similar generation processes have been introduced and widely used in machine comprehension BID8 BID9 BID17 .

However, research on cloze test design BID22 shows that tests created by deliberately deleting words are more reliable than tests created by randomly or periodically deleting words.

To design accurate language proficiency assessment, teachers usually select words in order to examine students' proficiency in grammar, vocabulary and reasoning.

Moreover, in order to make the question non-trivial, the three incorrect options provided by teachers are usually grammatically correct and relevant to the context.

For instance, in the fourth problem of the sample passage shown in Table 2 , "grapes", "flowers" and "bananas" all fit the description of being fresh.

We know "flowers" is the correct answer after seeing the sentence "Somebody has sent me flowers the very first day!".Naturally, we hypothesize that the distribution of human-generated data is different from automatically generated data.

To verify this assumption, we compare the LSTM model's performance when given different proportion of the two types of data.

Specifically, to train a model with α percent of automatically generated data, we randomly replace a percent blanks with blanks at random positions, while keeping the remaining 100 − α percent questions the same.

The candidate options for the generated blanks are random words sampled from the unigram distribution.

We test the trained model on human-designed data and automatically generated data respectively.

Table 7 : We train a model on α percent of automatically generated data and 100 − α percent of human-designed data and test it on human-designed data and automatically generated data respectively.

The performance is shown in Table 7 .

We have the following observations: (1) human-designed data leads to a larger gap between the model's performance and the human performance, when given the same model.

The model's performance and human's performance on the human-designed data are 0.484 and 0.860 respectively, leading to a gap of 0.376.

In comparison, the performance gap on the automatically generated data is at most 0.185 since the model's performance reaches 0.815 when trained on generated data.

It shows that the distributions of human-designed data and automatically generated data are quite different.

(2) the distributional mismatch between two types of data makes it difficult to transfer a model trained on human-designed data to automatically generated data.

Specifically, the model's performance on automatically generated data monotonously increases when given a higher ratio of automatically generated training data.

To conclude, human-designed data is a good test base because of the larger gap between performances of the model and the human, although the distributional mismatch problem makes it difficult to be the best training source for out-of-domain cloze test such as automatically generated cloze test.4.2 COMBINING HUMAN-DESIGNED DATA WITH AUTOMATICALLY GENERATED DATA In Section 3.1, we show that language model is able to take advantage of more supervisions since it predicts each word based on the context.

In essence, each word can provide an automatically generated question.

At the same time, we also show that human-designed data and the automatically generated data are quite different in Section 4.1.

In this section, we propose to combine humandesigned data with automatically generated data to achieve better performance.

Note that discriminative models can also treat all words in a passage as automatically generated questions, just like a language model (Please see the Appendix A.3 for details).

We study two methods of leveraging automatically generated data and human-designed data:Equally averaging Let J h be the average loss for all human-designed questions and J u be the average loss for all automatically generated questions in the passage.

A straightforward method is to optimize J h + λJ u so that the model learns to predict words deleted by human and all other words in the passage.

We set λ to 1 in our experiments.

This model treats each automatically generated questions as equally important.

Representativeness-based weighted averaging A possible avenue towards having large-scale indomain data is to automatically pick out questions which are representative of in-domain data among a large number of out-of-domain samples.

Hence, we mimick the design behavior of language teachers by training a network to predict the representativeness of each automatically generated question.

Note that the candidate option set for a automatically generated question is the whole vocabulary.

We leave the candidate set prediction for future work.

The performance of the representativeness prediction network and an example are shown in Appendix A.4.Let J i denotes the negative log likelihood loss for the i−th question and let l i be the outputted representativeness of the i-th question (The definition of l i is in Appendix A.2).

We define the representativeness weighted loss function as DISPLAYFORM0 where H is the set of all human-generated questions and α is the temperature of the Softmax function.

When the temperature is +∞, the model degenerate into equally averaging objective function without using the representativeness.

When the temperature is 0, only the most representative question is used.

We set α to 2 based on the performance on the dev set.

We present the results in Table 8 .

When all other words are treated as equally important, the accuracy is 0.543, similar to the performance of language model.

Representativeness-based weighted averaging leads to an accuracy of 0.565.

When combined with human-designed data, the performance can be improved to 0.583 6 .

Large-scale automatically generated cloze test BID8 BID9 BID17 leaded to significant research advancement.

However, the generated questions do not consider Table 8 : Overall results on CLOTH.

The "representativeness" means weighted averaging the loss of each question using the predicted representativeness.

"equal-average" means to equally average losses of questions.the language phenomenon to be tested and are relatively easy to solve.

Recently proposed reading comprehension datasets are all labeled by human to ensure their qualities BID20 BID12 BID28 BID16 .

Aiming to evaluate machines under the same conditions human is evaluated, there are a growing interests in obtaining data from examinations.

NTCIR QA Lab BID24 contains a set of real-world university entrance exam questions.

The Entrance Exams task at CLEF QA Track BID18 BID21 evaluates machine's reading comprehension ability.

The AI2 Elementary School Science Questions dataset 7 provides 5, 060 scientific questions used in elementary and middle schools.

BID15 proposes the first large-scale machine comprehension dataset obtained from exams.

They show that questions designed by teachers have a significant larger proportion of reasoning questions.

Our dataset focuses on evaluating language proficiency while the focus of reading comprehension is reasoning.

In Section 4.2, we employ a simple supervised approach that predicts how likely a word is selected by teachers as a cloze question.

It has been shown that features such as morphology information and readability are beneficial in cloze test prediction BID25 BID5 .

We leave investigating the advanced approaches of automatically designing cloze test to future work.

In this paper, we propose a large-scale cloze test dataset CLOTH that is designed by teachers.

With the missing blanks and candidate options carefully created by teachers to test different aspects of language phenomenon, CLOTH requires a deep language understanding and better captures the complexity of human language.

We find that human outperforms state-of-the-art models by a significant margin, even if the model is trained on a large corpus.

After detailed analysis, we find that the performance gap is due to model's inability to understanding a long context.

We also show that, compared to automatically-generated questions, human-designed questions are more difficult and leads to a larger margin between human performance and the model's performance.

A predicted sample is shown in FIG2 .

Clearly, words that are too obvious have low scores, such as punctuation marks, simple words "a" and "the".

In contrast, content words whose semantics are directly related to the context have a higher score, e.g., "same", "similar", "difference" have a high score when the difference between two objects is discussed and "secrets" has a high score since it is related to the subsequent sentence "does not want to share with others".Our prediction model achieves an F1 score of 36.5 on the test set, which is understandable since there are many plausible questions within a passage.

@highlight

A cloze test dataset designed by teachers to assess language proficiency