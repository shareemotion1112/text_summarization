In this paper, we explore \textit{summary-to-article generation}: the task of generating long articles given a short summary, which provides finer-grained content control for the generated text.

To prevent sequence-to-sequence (seq2seq) models from degenerating into language models and better controlling the long text to be generated, we propose a hierarchical generation approach which first generates a sketch of intermediate length based on the summary and then completes the article by enriching the generated sketch.

To mitigate the discrepancy between the ``oracle'' sketch used during training and the noisy sketch generated during inference, we propose an end-to-end joint training framework based on multi-agent reinforcement learning.

For evaluation, we use text summarization corpora by reversing their inputs and outputs, and introduce a novel evaluation method that employs a summarization system to summarize the generated article and test its match with the original input summary.

Experiments show that our proposed hierarchical generation approach can generate a coherent and relevant article based on the given summary, yielding significant improvements upon conventional seq2seq models.

In contrast to the well-explored text generation tasks like machine translation (Bahdanau et al., 2014) and text summarization (See et al., 2017) , open-ended long text generation is much less explored.

The existing studies on long text generation either generate long text unconditionally, such as GPT-2 (Radford et al.), or generate long text conditioning on a single sentence prompt (Fan et al., 2018; Keskar et al., 2019; Zellers et al., 2019) .

Although they can generate seemingly fluent text in a general domain/topic, they suffer from a lack of fine-grained control of content to be generated, which may result in generating much undesirable text and make them difficult to use in practice.

In this paper, we study long text generation with fine-grained content control.

We explore summary-toarticle generation: the task of generating a coherent and relevant long article based on a short summary of 3 to 5 sentences which summarizes the main content of the article to be generated.

Compared to the previously studied unconditional or prompt-based long text generation, summary-to-article generation specifies the content to be generated more clearly, leading to finer-grained control of text generation.

As prior work (Fan et al., 2018) points out, however, it remains challenging to generate a long, coherent and relevant article based on a summary because complex and underspecified dependencies between the summary and the article are much harder to model than the closer dependencies required for language modeling, which makes standard seq2seq models prone to degenerating into language models, neglecting salient information provided in the summary and resulting in undesirable outputs.

To address this challenge, inspired by previous work that attempts to generate text with multiple steps (Dalianis & Hovy, 1993; Reiter & Dale, 2000) , we propose a hierarchical summary-to-article generation approach which decomposes the task into two subtasks: summary-to-sketch generation and sketch-to-article generation.

As illustrated in Figure 1 , the sketch is of an intermediate length and serves as a draft of the output article to be generated and outlines its main content, which resembles how people plan to write long articles in their mind.

This hierarchical generation approach avoids the need for seq2seq text generation models to extend the length of source text too much, thus alleviating the aforementioned degenerating problem and enhancing the coherence and relevance of the generated text.

To bridge the gap between training and inference, which arises from the discrepancy between the extracted "oracle sketch" used during training and the noisy sketch generated during inference, we propose a gated model fusion mechanism Figure 1: Illustration of the proposed hierarchical generation model for summary-to-article generation: (a) The conventional seq2seq model generates an article directly based on the summary; (b) Our proposed hierarchical summary-to-article generation approach first generates a sketch based on the summary and then completes the article based on the generated sketch.

which establishes a skip-connection from the input summary to the sketch-to-article generation model, and jointly train the summary-to-article and the sketch-to-article generation model to communicate and cooperate with each other in an end-to-end fashion with multi-agent reinforcement learning.

For evaluation, we use the text summarization corpora by reversing their inputs and outputs as our dataset.

We also introduce a novel evaluation metric -ROUGE-rec which calculates how much the original summary can be reconstructed from the generated article.

Experiments on the CNN/DM and BIGPATENT datasets demonstrate our proposed hierarchical generation approach can generate fluent, coherent and relevant articles based on a given summary, yielding better results than conventional seq2seq models.

Our contributions are threefold:

??? We explore the task of summary-to-article generation which provides finer-grained control of generated long text and propose a hierarchical summary-to-article generation approach.

??? We propose a gated model fusion mechanism and a multi-agent reinforcement learning with denoising seq2seq pretraining objective to help bridge the gap between training and inference of the hierarchical generation model.

??? We propose a novel evaluation method for summary-to-article generation.

Experimental results demonstrate that the proposed evaluation metric correlates better with human evaluation than the traditional metrics like perplexity for this task.

The task of summary-to-article generation aims to generate long articles (??? 500 words) based on a short summary (??? 50 words) containing several sentences which specifies the main content of the article.

Given an input summary S, the output article A is expected to be fluent, coherent, relevant and faithful with respect to S. Compared to the previously studied unconditional (Radford et al.) or prompt-based (Fan et al., 2018; Zellers et al., 2019; Keskar et al., 2019 ) long text generation, the summary used in our task specifies the content to be generated more clearly, leading to finer-grained control of text generation and improves the coherence of the generated article by providing richer information.

The setting is more practical in real-world application scenarios, such as generating a news article from several highlights, generating a patent claim from a short description, or writing a story from an outline.

The main challenge of generating a long article based on a summary is the information gap between the input and output as the output is much longer than the input and contains additional information.

That requires the model to capture underspecified mapping from the summary to the article, which is much more difficult than the closer dependencies required for language modeling.

As a consequence, the larger expansion ratio between the input and the output, the more likely the standard seq2seq Figure 2: Overview of the proposed hierarchical summary-to-article generation model.

models are to degenerate into language models, failing to focus on the salient information provided in the summary while generating text.

Motivated by this observation, we propose a hierarchical summary-to-article generation approach which decomposes the task into two subtasks: summary-to-sketch generation and sketch-to-article generation.

The sketch (we denote as K in the following parts) is a draft of the final article to be generated, which outlines the main structure and the content of the article with an intermediate length.

With the help of the sketch as a bridge between the short summary and the long article, hierarchical generation divides the challenge into two simpler sub-tasks, thus alleviating the issue of degeneration, facilitating the generation and enhancing the coherence and relevance of the generated article.

The overview of our proposed model is shown in Figure 2 .

The proposed hierarchical summary-toarticle generation model mainly includes two seq2seq components: the summary-to-sketch generation model (G 1 ), which generates a sketch based on the summary, and the sketch-to-article generation model (G 2 ) which completes the article based on the generated sketch.

The input summary will be first fed into the G 1 to obtain the sketch which will be then taken as input by G 2 .

To avoid the cases where the generated sketch is not clean and not good enough for generating articles, we add a skip-connection that is implemented as a summary-to-article generation model (G 3 ) to the hidden outputs of G 2 's decoder, as shown in Figure 2 .

In this way, the hierarchical outputs of G 2 are fused with the outputs of the skip-connection from the original summary, allowing the model to adaptively learn to generate the article based on both the information from the original summary which is limited but clean and that from the generated sketch which is the more adequate but potentially noisy.

Following previous work (Fan et al., 2018) , we formulate the gated model fusion mechanism as:

where ??? denotes element-wise multiplication and ??(??) denotes the sigmoid function.

For model fusion, the t-th decoder hidden state of the summary-to-article generation model and the sketch-to-article generation model (represented by h t ) are concatenated to learn gates g t .

The gated hidden layers are then combined by concatenation and followed by fully connected layers to generate the output token.

The hierarchical generation approach with the proposed fusion mechanism is illustrated in Figure 2 .

The core part of the proposed decomposition approach is constructing appropriate sketches for training.

Inspired by the approaches for constructing supervision labels for training extractive summarization models (Nallapati et al., 2017) , we propose a heuristic approach to extract important sentences in the article which are the most relevant to the summary to construct the sketch.

We compute the relevance score of each sentence a i in the article by computing the maximum of its cosine embedding similarity with sentences s j in the summary S under a pretrained language model 1 , which can be formulated as:

Figure 3: Illustration of the proposed training strategy: (a) MARL: An end-to-end joint training framework for hierarchical summary-to-article generation, G 1 and G 2 are updated during MARL while G 3 is fixed.; (b) denoising seq2seq pre-training for the sketch-to-article model G 2 to improve the robustness of G 2 against the noise in the generated sketch.

Afterward, we iteratively extract the sentence of maximum relevance score from each paragraph in the article (without putting back) until the length of extracted sentences exceeds the threshold, which is empirically set to be the geometric mean of the length of the summary and the article measured by the number of tokens.

This ensures the ratio of sequence length between the input and the output of the two components of our model to be roughly the same.

With extracted sketches as weakly supervision, we can train the two components in our model separately with MLE.

The major limitation of training the two generation models in the proposed hierarchical generation approach separately with MLE is the discrepancy between the sketches used for training and inference.

During training, the sketch used to generate the article is the "oracle" sketch that consists of sentences extracted from the article; while during inference, the sketch is generated from the summary by the summary-to-sketch model.

In other words, the sketch-to-article generation model is trained to generate articles based on extracted sketches which are clean and of high quality, but receives generated sketches which are generally noisy and less adequate during inference.

As a result, the gap between training and inference makes it difficult for the sketch-to-article model to work well and generate good articles in practice where no extracted sketch is available.

To address this problem and help the sketch-to-article generation model be better adapted to the generated noisy sketches during inference, we propose an end-to-end joint training framework with multi-agent reinforcement learning (MARL) and denoising pretraining to train our model, as illustrated in Figure 3 .

Inspired by the previous work on multi-agent communication tasks (Lowe et al., 2017; Lee et al., 2019) , we model the summary-to-article generation task as a two-agent cooperation task and jointly train the agents to communicate and cooperate with each other in an end-to-end fashion with reinforcement learning.

The first agent G 1 (i.e. summary-to-sketch generation model) receives a summary S as input and generates a sketch K as output message.

The second agent G 2 is then trained to maximize the log-likelihood of the ground truth article A given the sketch message, i.e. log p(A|K).

Agent G 1 is trained using REINFORCE (Williams, 1992) with reward R = log p G2,3 (A|K, S).

Following Lee et al. (2019) , we formulate the learning objective of Agent G 1 as:

where ?? pg , ?? entr , ?? b are hyperparameters, H and MSE denote entropy and mean squared error losses, R t is a state-dependent baseline for reducing variance.

The first term is the reward we aim to maximize, the second term is an entropy regularization on Agent G 1 's decoder to encourage exploration, the last term is the training objective of the baseline R t .

The summary-to-article generation model G 3 is pretrained with MLE and fixed during reinforcement learning.

The training objective encourages Agent G 1 to develop helpful communication policies for Agent G 2 and to generate better sketch in terms of its usefulness for Agent G 2 , while also allows Agent G 2 to be adapted to noisy sketches generated by Agent G 1 .

To provide a good initialization for reinforcement learning based joint training, we pretrain the summary-to-sketch generation model G 1 and the summary-to-article generation model G 3 with MLE.

For the sketch-to-article generation model G 2 which suffers from the aforementioned problem of input discrepancy, we employ a denoising seq2seq pretraining objective to improve the robustness of the model with respect to the noise in the input.

Specifically, we corrupt the input sketches with both word-level and sentence-level perturbation 2 and train the model to generate the same articles based on the perturbed sketches.

The perturbation is expected to resemble the noise in generated sketches, thus helping the model be better adapted to the generated sketches during training.

The noises injected in the sketches also help prevent the sketch-to-article generation model from learning to directly copy sentences in the sketch when generating articles, which is undesirable as generated sketches are noisy during inference.

During inference, the input summary S is fed into the summary-to-sketch generation model G 1 to generate a sketch K. The generated sketch K and the original summary S is then fed to the fused sketch-to-article generation model G 2 to generate the output article.

We employ the top-p (p = 0.95 in our experiments) sampling approach (Holtzman et al., 2019) , which samples from the top p portion of the probability mass, expanding and contracting the candidate pool dynamically, instead of standard beam search, to avoid repetition and encourage generating diverse articles.

Text generation models are usually evaluated using the word-overlap based metrics (e.g., BLEU and ROUGE) and perplexity.

As suggested by Liu et al. (2016) , however, these metrics tend to perform poorly when evaluating open-domain text generation systems.

The long and diverse nature of articles makes them even worse for evaluating the quality of summary-to-article generation models.

Moreover, they are incapable of evaluating the relevance between the generated articles and the summary, which is important for evaluating the model's ability for fine-grained content control.

For better evaluating the summary-to-article text generation task, we introduce a novel evaluation metric ROUGE-rec.

ROUGE-rec evaluates how much the original summary can be reconstructed from the generated article.

Intuitively, if the generated article can be summarized back to the original summary (ROUGE-rec is high), that indicates that the article well focuses on the summary; on the contrary, if the generated article's summary is dissimilar to the original summary (ROUGE-rec is low), the article probably deviates from the ideas of the original summary, which is undesirable.

Formally, we define ROUGE-rec to be the ROUGE-L (Lin, 2004) score of the reconstructed summary from the generated article against the original summary.

To compute ROUGE-rec, we first use a state-of-the-art abstractive summarization model (Wu et al., 2019) to summarize the generated article to obtain the reconstructed summary, and then derive the score ROUGE-rec by computing the ROUGE-L of the reconstructed summary.

We conduct experiments on the summary-to-article generation task by using text summarization datasets in the reverse direction, i.e. taking the summary as inputs and the corresponding articles as outputs.

Specifically, we evaluate the proposed approaches and several baseline models on the CNN / Daily Mail (Hermann et al., 2015) dataset and the BIGPATENT (Sharma et al., 2019) dataset.

We follow the default train-dev-test split of the original datasets.

The statistics of the datasets are shown in Table 1 .

In our experiments, we employ the following automated metrics for evaluating the performance of compared models.

??? PPL(gpt-2): The perplexity of generated articles under GPT-2 (Radford et al.), one of the most powerful pretrained language model with 340M parameters, which is able to measure long range dependency.

It can measure the fluency and the coherence of generated articles well.

??? ROUGE-rec: Our proposed new metric for summary-to-article generation, which reflects how well the generated article expands the input summary, as introduced in Section 2.5.

Following Fan et al. (2018) , the basic structure of all the summary-to-sketch (G 1 ), sketch-to-article (G 2 ) and summary-to-article (G 3 ) models is a seq2seq convolutional model with 3-layer encoder blocks with hidden unit sizes 128 ?? 2, 512 and conlutional kernel widths 3 ?? 3, and 8-layer decoder blocks with hidden unit sizes 512 ?? 4, 768 ?? 2, 1024 with convolutional kernel widths 4 ?? 8.

The size of both input embedding and output embedding is 256, and the number of heads for self-attention in the decoder is 4.

We keep the words which appear more than 10 times in the corpora, resulting in a vocabulary 3 of 142,971 in CNN/Daily, and 195,401 in BIGPATENT datasets.

We tune the hyperparameters in Eq (3) on the validation set: ?? pg = 1, ?? entr = 0.001, ?? b = 0.01.

We train the model on 4 GPUs with learning rate 0.25, dropout 0.3 and a batch of 4,000 tokens per GPU, and the pre-training of G 1 , G 2 and G 3 uses the same learning configuration.

We pretrain the model and choose the best checkpoint based on the perplexity on the validation set as initialization for the reinforcement learning.

We then train our models with 10,000 updates by reinforcement learning.

In our experiments, we compare the proposed hierarchical summary-to-article generation model with conventional LSTM seq2seq models and the convolutional seq2seq model (Fan et al., 2018) which is in the same structure as our seq2seq models.

The average length of generated articles is 712.3 and 3278.3 in the two datasets, which is slightly shorter than the training data.

We find that the output length of different models is very similar, which ensures that they are comparable with employed metrics.

As Table 2 shows, the perplexity of the articles generated by our proposed hierarchical model is better than that generated by the conventional seq2seq models, indicating that the hierarchical models can generate more fluent and coherent articles.

Moreover, we observe the hierarchical models largely outperform the conventional seq2seq models in terms of ROUGE-rec, demonstrating their powerful capability for content control to generate articles that are more relevant and faithful to the input summary.

Within the hierarchical models, the proposed gated model fusion mechanism plays an important role in improving the performance, because it allows the model to adaptively focus on the information from the original summary or that from the generated sketch, and also facilitates the training of the sketch-to-summary generation model by encouraging it to focus on what the summary-to-article generation model fails to learn.

To better analyze the performance of the hierarchical model, we compare the perplexity of groundtruth articles given an input summary by the conventional seq2seq convolutional model, to the perplexity of ground-truth articles given the generated sketch by the hierarchical model.

According to Table 3 , the perplexity of generating ground-truth articles by the hierarchical model is much smaller than that with the seq2seq baseline, demonstrating that the generated sketch by the hierarchical model is helpful for generating the long article.

With the help of the sketch, the hierarchical model can learn the content to be generated more easily, accounting for its advantage over the seq2seq baseline.

To make the evaluation more convincing, we further conduct human evaluation 4 .

Specifically, we follow Fan et al. (2018) and invite 20 graduate students with good English proficiency as human annotators and ask them to : 1) pair shuffled summaries and generated articles, and 2) choose the better article from two articles generated by the compared model and the seq2seq-conv baseline respectively.

The pairing accuracy measures the relevance between the generated article and the given input summary, while human preference measures the overall quality of generated articles.

The results of human evaluation are shown in Table 4 .

Our approach yields consistently better results upon the seq2seq baselines in terms of both pairing accuracy and human preference.

Also, the effectiveness of the fusion mechanism is verified by human evaluation.

We calculate the sample-level Pearson correlation of our proposed automated evaluation metrics (i.e., ROUGE-rec score) as well as the conventional metrics including PPL(gpt-2), BLEU, ROUGE, and the prompt relevance test (Fan et al., 2018) with human pairing accuracy and human preference.

According to Table 5 , most conventional automated metrics like BLEU and ROUGE do not correlate well with human evaluation.

The automated metrics that correlate with human score best are ROUGErec and PPL(GPT-2).

The former one evaluates the relevance of generated article while the latter one measures the fluency and coherence of the generated article.

Between these two metrics, human annotators tend to prefer articles with higher ROUGE-rec.

In other words, people prefer generated articles that have better control of the content, which suggests that the proposed ROUGE-rec may play an important role in evaluating conditional long text generation models for future work.

To investigate the effects of the proposed training strategies and different choices of the length of extracted sketches, we conduct an ablation study on the CNN/DM dataset and report the result Table 6 : Ablation study of training strategies and the influence of the length of sketch.

PPL(K * ???A) is the perplexity of ground-truth articles generated based on extracted "oracle" sketches.

0.5?? and 2?? denote the length of the sketch for training, compared with that of the geometric mean of the summary length and the article length, which is used in our model.

Table 6 .

We find that the proposed training strategies substantially reduce the gap between the perplexity of the articles based on extracted sketches and generated sketches and improve the fluency and coherence of generated articles measured by GPT-2, confirming their effects in bridging the gap between the training and inference stages and improving the model's robustness.

As for the influence of sketch length used for training, we find that both shorter sketches and longer sketches result in sub-optimal performance, suggesting that the geometric mean of summary length and article length may be a good default choice.

We hypothesize that it is because this choice makes the expansion ratio of the summary-to-sketch and sketch-to-article model identical, avoiding too much uncertainty which arises from a too large expansion ratio in either component.

For qualitative comparison, we present several samples of the article and patent claim generated by our approach and baselines in Appendix B.

Decomposed text generation: Our work is inspired by previous research that studies decomposing text generation into several steps.

In general, the previous studies focus on either statistical templatebased approaches (Wahlster et al., 1993; Dalianis & Hovy, 1993) or neural text generation models (Fan et al., 2018; Xu et al., 2018) .

Among the neural text generation models, most of them decompose text generation by either constructing intermediary output of roughly the same length of the final output (Fan et al., 2019; Xu et al., 2018) , or generating a very short "plan" in a higher level (Yao et al., 2019) .

They do not address the major challenge of long text generation.

In contrast, we construct sketches of intermediate length, thus providing more adequate information for generation final output and reducing the difficulty of long text generation.

Existing studies on long text generation either generate long text by unconditionally sampling from a pretrained language model, such as GPT-2 (Radford et al.), or generate long text conditioning on a single sentence prompt (Fan et al., 2018; Keskar et al., 2019; Zellers et al., 2019) .

While they can generate fluent text in a general domain/topic, they suffer from a lack of fine-grained content control of the article to be generated, which may result in generating much undesirable text and make them difficult to use in practice.

We explore the task of summary-to-article generation and propose a novel hierarchical summary-toarticle generation approach.

The approach first drafts a sketch that outlines the article to be generated based on the summary, then generates the article based on information in the summary and the sketch.

We propose an end-to-end joint training framework through the multi-agent reinforcement learning to train the hierarchical model and evaluate its performance in multiple datasets.

The experimental results show that our approach can generate a coherent and relevant article based on a given summary, outperforming the conventional seq2seq models for summary-to-article generation.

However, since the input summary only contains a subset of the information of the article, a summary-to-article generation model will tend to generate fabricated content by filling in the rest of the narrative, which may arise ethical concerns.

We will investigate the characteristic of additional information included in the generated articles and seek approaches to control them in our future work. (2019) pretrains a conditional language model to generate fake news based on given one-sentence headline which specifies the topic of the generated news.

Keskar et al. (2019) pretrains a conditional langauge model to generate contents in the domain specified by a "domain code".

Although they can generate seemingly fluent text in a general domain/topic, they suffer from a lack of fine-grained control of content to be generated, which may result in generating much undesirable text and make them difficult to use in practice.

Decomposing text generation Decomposing text generation into several steps has been explored in both statistical template-based text generation approaches (Wahlster et al., 1993; Dalianis & Hovy, 1993) and neural text generation models (Fan et al., 2018; Xu et al., 2018) .

Strategies for decomposing long text generation have been explored by transferring sentence compression, text summarization, and keyword extraction models to build an outline for guiding long text generation models.

However, these approaches are built for generating a short "pseudo-summary" unconditionally or based on a single sentence.

The intermediary training data for these approaches is thus generated and noisy.

In addition, previous work on decomposing text generation generally either constructs intermediary output of roughly the same length of the final output (Fan et al., 2019; Xu et al., 2018) , or generates a very short "plan" in a higher level (Yao et al., 2019) .

As a result, these approaches do not address the major difficulty of long text generation, which is the large difference of the length of input and output text in the seq2seq model.

Indeed,the hierarchical model of Xu et al. (2018) and Yao et al. (2019) is designed for generate sentences and short stories within 50 words, and the hierarchical model of Fan et al. (2018) only generate a single sentence prompt.

More recently, Fan et al. (2019) propose to first generate an action plan, then generate a anonymized story and fill in the entities in the last step.

Their deconposition method extend the length of sequence in the first step, thus is orthogonal to our proposed method.

Our decomposition approach is different from the aforementioned approaches in two perspectives: 1) the sketches used in our work are extracted with the guidance from both the article and the summary, thus of much better quality, and 2) we construct sketches of intermediate length, thus providing more adequate information for generation final output and reducing the difficulty of expanding the length of input by a large ratio in one pass.

Denoising pretraining for seq2seq models Pretraining a denoising autoencoder for text generation is explored in recent works (Edunov et al., 2018; Lample et al., 2017; Wang et al., 2019; Zhao et al., 2019) .

The motivation of their approaches is to tackle the data sparsity problem while we employ the denoising objective for training the sketch-to-article generation model to be better adapted to generated sketches.

As a result, the corruption methods in our work are different and our model is trained to directly output the target sequence instead of reconstructing the original input.

Learning to communicate and cooperate between multiple agents The idea of training multiple agents to communicate and cooperate with each other for accomplishing a common goal is well explored in multi-agent reinforcement learning literatures (Lowe et al., 2017; Foerster et al., 2018; Das et al., 2017) .

The most similar work to ours is that of Lee et al. (2019) , which pretrains two translation models of Fr-En and En-De respectively, and train them to perform Fr-De translation cooperatively with reinforcement learning.

Model fusion Previous work has investigated the integration of language models with seq2seq models and the fusion of two identical seq2seq models.

Gulcehre et al. (2015) combined a trained language model with a trained seq2seq model to learn a gating function that joins them.

Sriram et al. (2017) propose training the seq2seq model given the fixed language model then learning a gate to filter the information from the language model.

Fan et al. (2018) propose to train another identical seq2seq model based on a pretrained seq2seq model.

To our knowledge, our work is the first to investigate the fusion of two seq2seq model with different input and the same output to combine different source of information and prevent error-propagation.

We present samples in both datasets generated by both baseline convolutional seq2seq architecture and the proposed model in this section to facilitate qualitative analysis.

We highlight repetitive parts of generated article in red.

Female-named hurricanes cause `` significantly more deaths , '' a study finds.

Researchers analyzed over six decades of death rates from U.S. hurricanes.

Experiment participants considered hypothetical male storms deadlier than female storms.

One critic says earlier hurricanes, which were deadlier, were only given female names.

Scientists are starting to analyse a new relation of hurricanes and their names.

`` female hurricanes cause significantly more deaths , which is because they lead to more significant damages, '' a team of researchers wrote in the Proceedings of the National Academy of Sciences .

`

` The study was presented to the University of California and the <unk> government of the British Civil Liberties Union .

They were based on the true extent of the latest census and the number of hurricanes identified , '' said Marine Alfred Piccard , the curator of the university 's environmental journal .

The scientists also examined the evidence of <unk> and <unk> , and that the <unk> ratio of the species in the current temperature and age of about 1,000 people .

They studied the researchers 's findings from 2012 , when the <unk> numbers were associated with an outbreak of the last tropical storm .

Female-named hurricanes cause `` significantly more deaths , '' a study finds.

Researchers analyzed over six decades of death rates from U.S. hurricanes.

Experiment participants considered hypothetical male storms deadlier than female storms.

One critic says earlier hurricanes, which were deadlier, were only given female names.

Scientists at the University of Texas at Boulder had seen the `` mass '' of blood coming from the body of a rock star .

A study in the Proceedings of the National Academy of Sciences shows.

The results of the study suggest the planet 's dead bodies may be more than a year old and is less likely to be found in the area , the team 's researchers write .

`

` This is the the`` mass '' of blood coming from the body of a rock star is due to the high level of mass injury , '' the team wrote The study is expected to be published on the journal Nature on Friday .

`` This has a little impact on our understanding of the <unk> disease , '' <unk> said .

`

` It 's one of the highest rates in the world . ''

The researchers said.

The study 's authors took the time to study the findings , but they were n't surprised .

`

` We did n't know what it was because they were so hot , '' said researcher Dr. Paul <unk> , the professor of research at the University of Texas in Houston .

`

` We have a lot of differences in the age, but there is a specific side , '' he added .

<newline> '' `` This has a little impact on our understanding of the <unk> disease , '' <unk> said .

`

` It 's one of the highest rates in the world . ''

The researchers said that they were n't surprised by the results and said they had been surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they were n't surprised by the results and added that they ????????? Figure 5 :

Sample generated by baseline model on the CNN / Daily Mail dataset.

We highlight repetitive parts of generated article in red.

The famous rapper Iggy Azalea is dominating this year 's American Music Awards nominees.

It is the big event of the field of music and it is really exciting for the rappers.

The event is very important for music and it will be the most exciting reward for the year.

The famous rapper Iggy Azalea is very successful this year and she will dominate this year 's nominees.

The famous rapper Iggy Azalea is the artist of the year and is this year.

The artist leads six nominations including artist of the year, best artist, best rapper and what is exciting.

The competition is very hard and many other well knowed people will come to the party.

This year, their will be 10 contenders in the category feature.

It will be very hard for the artist to win the title at the final, which will be very hard.

There will be many other contenders for the competition, including Azalea and Beyonce.

The reward is very meaningful to the rapper and it is very meaningful.

It 's the biggest chance of a single contest , as they are well known .

There is a large majority of participants coming out and winning , as they would be more prone to the election .

<unk> , who played from the same stage , is the first to win the top five , the first to win the top five .

`

` This is a

A team find the Awards of music and Ms <unk> will win the game.

Ms <unk> said she was planning to take the same action again , but said she was planning to take the same action again.

The company is very successful.

The company was also popular for the first time and this is the first time an art artist has been shown in a mysterious area in the local state of <unk> .

` We have a number of different things from <unk> of the reward and we have a number of different things from <unk> of the reward The singer said : ` It is very cool and very hard to show .

At the time , she said she had been told that she was alive .

`

` I was not there , '' she said .

`

` I do n't know what happened .

I just did n't know what happened .

It was really scary . ''

'

I think it 's just something I am really proud of .

` It 's been a long journey . '

I think I 've been great for a long time .

` I 've been very happy with the way I performed at the bar , but it 's not the first time that I 've taken the pictures .

` This is not the first time I have done the dance routine , so it 's a big boost for me to be part of that .

` My time was just doing the show because I 'm really working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy who is working for a guy??????..

In use , the first resistance means 14 , the second resistance means 10 , the third resistance means 11 , the third resistance means 11 and the fourth resistance means 12 and the third resistance means 11 are connected to each other by means of the first resistance means 14 , the second resistance means 12 and the fourth resistance means 16 .

The first and second resistance means 14 and 16 are connected to each other by means of the first resistance means 14 , the second resistance means 16 and the third resistance means 11 .

as a result of the construction of the springs , a switch ( not shown ) is made for the control unit a , which has its own first resistance means 14 , the first resistance means 14 and 16 are connected to each other by means of the first resistance means 14 , the second resistance means 16 and the third resistance means 11 .

As a result of the construction of the springs , a switch ( not shown ) is made for the control unit a , which has its own first resistance means 14 , the first resistance means 14 ?????? Figure 9 : Sample generated by baseline model on the BIGPATENT dataset.

We highlight repetitive parts of generated article in red.

Sentence-level Perturbations We consider the following operations 1) Shuf that shuffles the sequence of sentences in the extracted sketches, 2) Drop that completely drops certain sentences, and 4) Repl that randomly replace one sentence in the extracted sketches by another sentence in the dataset.

Word-level perturbations We consider similar operations but at the word level within every sentence 1) word-shuffle that randomly shuffles the words within a sentence 2) reverse that reverses the ordering of words, 3) word-drop that drops 30% of the words in one sentence uniformly 4) noun-drop that drops all nouns, 5) verb-drop that drops all verbs, and 6) word-repl that replace 30% of words with a random word in the vocabulary uniformly.

We explain the role of different perturbation and their potential effects briefly.

The Shuf and reverse perturbations change the chronological order of sentences and denoising seq2seq pretraining with these kinds of perturbation may help the model to be robust when the summary-to-sketch generation model fails to generate sentences in chronological order.

The Drop and Repl perturbations may help the model to be robust when some information is lost during summary-to-sketch generation.

For news articles in the CNN/DM dataset, we truncate them to be at most 1000 tokens, for patent claims, we truncate them to 3000 tokens considering the limit of GPU memory.

We tokenize training data with moses tokenizer and did not use byte-pair encoding following Fan et al. (2018) .

For human evaluation, we invite 20 graduate students with good English proficiency as human annotators.

For each dataset, we randomly sample 100 summaries from the test set and generate articles with them using each compared models and distribute them randomly to human annotators.

Human annotators are required to perform two tasks: 1) the triple pairing task, where groups of three articles are presented to the human judges.

The articles and their corresponding prompts are shuffled, and human annotators are asked to select the correct pairing for all three prompts.

The accuracy is used to measure the relevance between generated articles and corresponding summaries.

2) human preference task, where human annotators are shown two different articles generated by a compared model and the seq2seq-conv model respectively, together with the same summary based on which the articles are generated.

Annotators are then asked to mark which article they prefer.

Each generated news article is paired 3 times and appears in the preference test 2 times, so that each model get 300/200 results in the triple pairing task and the human preference task respectively.

Patent claims are strimmed to 200 and 400 words respectively to ease human evaluation.

@highlight

we explore the task of summary-to-article generation and propose a hierarchical generation scheme together with a jointly end-to-end reinforcement learning framework to train the hierarchical model.

@highlight

To address the issue of degeneration in summary-to-article generation, this paper proposes a hierarchical generation approach which first generates an intermediate sketch of the article and then the full article.