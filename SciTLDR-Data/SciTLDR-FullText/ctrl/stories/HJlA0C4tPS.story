We present a deep generative model for unsupervised text style transfer that unifies previously proposed non-generative techniques.

Our probabilistic approach models non-parallel data from two domains as a partially observed parallel corpus.

By hypothesizing a parallel latent sequence that generates each observed sequence, our model learns to transform sequences from one domain to another in a completely unsupervised fashion.

In contrast with traditional generative sequence models (e.g. the HMM), our model makes few assumptions about the data it generates: it uses a recurrent language model as a prior and an encoder-decoder as a transduction distribution.

While computation of marginal data likelihood is intractable in this model class, we show that amortized variational inference admits a practical surrogate.

Further, by drawing connections between our variational objective and other recent unsupervised style transfer and machine translation techniques, we show how our probabilistic view can unify some known non-generative objectives such as backtranslation and adversarial loss.

Finally, we demonstrate the effectiveness of our method on a wide range of unsupervised style transfer tasks, including sentiment transfer, formality transfer, word decipherment, author imitation, and related language translation.

Across all style transfer tasks, our approach yields substantial gains over state-of-the-art non-generative baselines, including the state-of-the-art unsupervised machine translation techniques that our approach generalizes.

Further, we conduct experiments on a standard unsupervised machine translation task and find that our unified approach matches the current state-of-the-art.

Text sequence transduction systems convert a given text sequence from one domain to another.

These techniques can be applied to a wide range of natural language processing applications such as machine translation (Bahdanau et al., 2015) , summarization (Rush et al., 2015) , and dialogue response generation (Zhao et al., 2017) .

In many cases, however, parallel corpora for the task at hand are scarce.

Therefore, unsupervised sequence transduction methods that require only non-parallel data are appealing and have been receiving growing attention (Bannard & Callison-Burch, 2005; Ravi & Knight, 2011; Mizukami et al., 2015; Shen et al., 2017; Lample et al., 2018; .

This trend is most pronounced in the space of text style transfer tasks where parallel data is particularly challenging to obtain (Hu et al., 2017; Shen et al., 2017; Yang et al., 2018) .

Style transfer has historically referred to sequence transduction problems that modify superficial properties of texti.e.

style rather than content.

We focus on a standard suite of style transfer tasks, including formality transfer (Rao & Tetreault, 2018) , author imitation (Xu et al., 2012) , word decipherment (Shen et al., 2017) , sentiment transfer (Shen et al., 2017) , and related language translation (Pourdamghani & Knight, 2017) .

General unsupervised translation has not typically been considered style transfer, but for the purpose of comparison we also conduct evaluation on this task (Lample et al., 2017) .

Recent work on unsupervised text style transfer mostly employs non-generative or non-probabilistic modeling approaches.

For example, Shen et al. (2017) and Yang et al. (2018) design adversarial discriminators to shape their unsupervised objective -an approach that can be effective, but often introduces training instability.

Other work focuses on directly designing unsupervised training objectives by incorporating intuitive loss terms (e.g. backtranslation loss), and demonstrates state-ofthe-art performance on unsupervised machine translation (Lample et al., 2018; Artetxe et al., 2019) and style transfer (Lample et al., 2019) .

However, the space of possible unsupervised objectives is extremely large and the underlying modeling assumptions defined by each objective can only be reasoned about indirectly.

As a result, the process of designing such systems is often heuristic.

In contrast, probabilistic models (e.g. the noisy channel model (Shannon, 1948) ) define assumptions about data more explicitly and allow us to reason about these assumptions during system design.

Further, the corresponding objectives are determined naturally by principles of probabilistic inference, reducing the need for empirical search directly in the space of possible objectives.

That said, classical probabilistic models for unsupervised sequence transduction (e.g. the HMM or semi-HMM) typically enforce overly strong independence assumptions about data to make exact inference tractable (Knight et al., 2006; Ravi & Knight, 2011; Pourdamghani & Knight, 2017) .

This has restricted their development and caused their performance to lag behind unsupervised neural objectives on complex tasks.

Luckily, in recent years, powerful variational approximation techniques have made it more practical to train probabilistic models without strong independence assumptions (Miao & Blunsom, 2016; Yin et al., 2018) .

Inspired by this, we take a new approach to unsupervised style transfer.

We directly define a generative probabilistic model that treats a non-parallel corpus in two domains as a partially observed parallel corpus.

Our model makes few independence assumptions and its true posterior is intractable.

However, we show that by using amortized variational inference (Kingma & Welling, 2013) , a principled probabilistic technique, a natural unsupervised objective falls out of our modeling approach that has many connections with past work, yet is different from all past work in specific ways.

In experiments across a suite of unsupervised text style transfer tasks, we find that the natural objective of our model actually outperforms all manually defined unsupervised objectives from past work, supporting the notion that probabilistic principles can be a useful guide even in deep neural systems.

Further, in the case of unsupervised machine translation, our model matches the current state-of-the-art non-generative approach.

We first overview text style transfer, which aims to transfer a text (typically a single sentence or a short paragraph -for simplicity we refer to simply "sentences" below) from one domain to another while preserving underlying content.

For example, formality transfer (Rao & Tetreault, 2018) is the task of transforming the tone of text from informal to formal without changing its content.

Other examples include sentiment transfer (Shen et al., 2017) , word decipherment (Knight et al., 2006) , and author imitation (Xu et al., 2012) .

If parallel examples were available from each domain (i.e. the training data is a bitext consisting of pairs of sentences from each domain) supervised techniques could be used to perform style transfer (e.g. attentional Seq2Seq (Bahdanau et al., 2015) and Transformer (Vaswani et al., 2017) ).

However, for most style transfer problems, only non-parallel corpora (one corpus from each domain) can be easily collected.

Thus, work on style transfer typically focuses on the more difficult unsupervised setting where systems must learn from non-parallel data alone.

The model we propose treats an observed non-parallel text corpus as a partially observed parallel corpus.

Thus, we introduce notation for both observed text inputs and those that we will treat as latent variables.

Specifically, we let X = {x

Corresponding indices represent parallel sentences.

Thus, none of the observed sentences share indices.

In our model, we introduce latent sentences to complete the parallel corpus.

Specifically,X = {x (m+1) ,x (m+2) , ?? ?? ?? ,x (n) } represents the set of latent parallel sentences in D 1 , whil??

} represents the set of latent parallel sentences in D 2 .

Then the goal of unsupervised text transduction is to infer these latent variables conditioned the observed non-parallel corpora; that is, to learn p(??|x) and p(x|y).

First we present our generative model of bitext, which we refer to as a deep latent sequence model.

We then describe unsupervised learning and inference techniques for this model class.

Directly modeling p(??|x) and p(x|y) in the unsupervised setting is difficult because we never directly observe parallel data.

Instead, we propose a generative model of the complete data that defines a joint likelihood, p(X,X, Y,?? ).

In order to perform text transduction, the unobserved halves can be treated as latent variables: they will be marginalized out during learning and inferred via posterior inference at test time.

Our model assumes that each observed sentence is generated from an unobserved parallel sentence in the opposite domain, as depicted in Figure 1 .

Specifically, each sentence

, and prior, p D1 (x (j) ).

We let ?? x|?? and ?? y|x represent the parameters of the two transduction distributions respectively.

We assume the prior distributions are pretrained on the observed data in their respective domains and therefore omit their parameters for simplicity of notation.

Together, this gives the following joint likelihood:

The log marginal likelihood of the data, which we will approximate during training, is:

Note that if the two transduction models share no parameters, the training problems for each observed domain are independent.

Critically, we introduce parameter sharing through our variational inference procedure, which we describe in more detail in Section 3.2.

Architecture: Since we would like to be able to model a variety of transfer tasks, we choose a parameterization for our transduction distributions that makes no independence assumptions.

Specifically, we employ an encoder-decoder architecture based on the standard attentional Seq2Seq model which has been shown to be successful across various tasks (Bahdanau et al., 2015; Rush et al., 2015) .

Similarly, our prior distributions for each domain are parameterized as recurrent language models which, again, make no independence assumptions.

In contrast, traditional unsupervised generative sequence models typically make strong independence assumptions to enable exact inference (e.g. the HMM makes a Markov assumption on the latent sequence and emissions are one-to-one).

Our model is more flexible, but exact inference via dynamic programming will be intractable.

We address this problem in the next section.

Ideally, learning should directly optimize the log data likelihood, which is the marginal of our model shown in Eq. 2.

However, due to our model's neural parameterization which does not factorize, computing the data likelihood cannot be accomplished using dynamic programming as can be done with simpler models like the HMM.

To overcome the intractability of computing the true data likelihood, we adopt amortized variational inference (Kingma & Welling, 2013) in order to derive a surrogate objective for learning, the evidence lower bound (ELBO) on log marginal likelihood 2 :

The surrogate objective introduces q(??|x (i) ; ???? |x ) and q(x|y (j) ; ??x |y ), which represent two separate inference network distributions that approximate the model's true posteriors, p(??|x (i) ; ?? x|?? ) and p(x|y (j) ; ?? y|x ), respectively.

Learning operates by jointly optimizing the lower bound over both variational and model parameters.

Once trained, the variational posterior distributions can be used directly for style transfer.

The KL terms in Eq. 3, that appear naturally in the ELBO objective, can be intuitively viewed as regularizers that use the language model priors to bias the induced sentences towards the desired domains.

Amortized variational techniques have been most commonly applied to continuous latent variables, as in the case of the variational autoencoder (VAE) (Kingma & Welling, 2013) .

Here, we use this approach for inference over discrete sequences, which has been shown to be effective in related work on a semi-supervised task (Miao & Blunsom, 2016) .

Inference Network and Parameter Sharing: Note that the approximate posterior on one domain aims to learn the reverse style transfer distribution, which is exactly the goal of the generative distribution in the opposite domain.

For example, the inference network q(??|x (i) ; ???? |x ) and the generative distribution p(y|x (i) ; ?? y|x ) both aim to transform D 1 to D 2 .

Therefore, we use the same architecture for each inference network as used in the transduction models, and tie their parameters: ??x |y = ?? x|?? , ???? |x = ?? y|x .

This means we learn only two encoder-decoders overall -which are parameterized by ?? x|?? and ?? y|x respectively -to represent two directions of transfer.

In addition to reducing number of learnable parameters, this parameter tying couples the learning problems for both domains and allows us to jointly learn from the full data.

Moreover, inspired by recent work that builds a universal Seq2Seq model to translate between different language pairs (Johnson et al., 2017) , we introduce further parameter tying between the two directions of transduction: the same encoder is employed for both x and y, and a domain embedding c is provided to the same decoder to specify the transfer direction, as shown in Figure 2 .

Ablation analysis in Section 5.3 suggests that parameter sharing is important to achieve good performance.

The reconstruction terms and the KL terms in Eq. 3 still involve intractable expectations due to the marginalization of the latent sequence, thus we need to approximate their gradients.

We can approximate the expectations by sampling latent sequences, but gradients are difficult to back-propagate directly through discrete samples.

Gumbel-softmax (Jang et al., 2017) and REINFORCE (Sutton et al., 2000) are often used as stochastic gradient estimators in the discrete case.

Since the latent text variables have an extremely large domain, we find that REINFORCE-based gradient estimates result in high variance.

Thus, we use the Gumbel-softmax straight-through estimator to backpropagate gradients from the KL terms.

However, we find that approximating gradients of the reconstruction loss is much more challenging -both the Gumbelsoftmax estimator and REINFORCE are unable to outperform a simple stop-gradient baseline, which confirms a similar observation in previous work on unsupervised machine translation (Lample et al., 2018) .

Therefore, we simply stop computing the gradients for the inference network that would result from the reconstruction term, and perform greedy sampling of the latent sequences during training.

Note that the inference networks still receive gradients from the prior through the KL term, and their parameters are shared with the decoders which do receive gradients from reconstruction.

We consider this to be the best empirical compromise at the moment.

Initialization.

Good initialization is often necessary for successful optimization of unsupervised learning objectives.

In preliminary experiments, we find that the encoder-decoder structure has difficulty generating realistic sentences during the initial stages of training, which usually results in a disastrous local optimum.

This is mainly because the encoder-decoder is initialized randomly and there is no direct training signal to specify the desired latent sequence in the unsupervised setting.

Therefore, we apply a self-reconstruction loss L rec at the initial epochs of training.

We denote the output the encoder as e(??) and the decoder distribution as p dec , then

?? decays from 1.0 to 0.0 linearly in the first k epochs.

k is a tunable parameter and usually less than 3 in all our experiments.

Our probabilistic formulation can be connected with recent advances in unsupervised text transduction methods.

For example, back translation loss (Sennrich et al., 2016) plays an important role in recent unsupervised machine translation (Artetxe et al., 2018; Lample et al., 2018; Artetxe et al., 2019) and unsupervised style transfer systems Lample et al. (2019) .

In back translation loss, the source language x is translated to the target language y to form a pseudo-parallel corpus, then a translation model from y to x can be learned on this pseudo bitext just as in supervised setting.

While back translation was often explained as a data augmentation technique, in our probabilistic formulation it appears naturally with the ELBO objective as the reconstruction loss term.

Some previous work has incorporated a pretrained language models into neural semi-supervised or unsupervised objectives.

He et al. (2016) uses the log likelihood of a pretrained language model as the reward to update a supervised machine translation system with policy gradient.

Artetxe et al. (2019) utilize a similar idea for unsupervised machine translation.

Yang et al. (2018) employed a similar approach, but interpret the LM as an adversary, with the generator is trained to fool the LM.

We show how our ELBO objective is connected with these more heuristic LM regularizers by expanding the KL loss term (assume x is observed):

Note that the loss used in previous work does not include the negative entropy term, ???H q .

Our objective results in this additional "regularizer", the negative entropy of the transduction distribution, ???H q .

Intuitively, ???H q is helps avoid a peaked transduction distribution, preventing the transduction from constantly generating similar sentences to satisfy the language model.

In experiments we will show that this additional regularization is important and helps bypass bad local optimum and improve performance.

These important differences with past work suggest that a probabilistic view of the unsupervised sequence transduction may provide helpful guidance in determining effective training objectives.

We test our model on five style transfer tasks: sentiment transfer, word substitution decipherment, formality transfer, author imitation, and related language translation.

For completeness, we also evaluate on the task of general unsupervised machine translation using standard benchmarks.

We compare with the unsupervised machine translation model (UNMT) which recently demonstrated state-of-the-art performance on transfer tasks such as sentiment and gender transfer (Lample et al., 2019) .

4 To validate the effect of the negative entropy term in the KL loss term Eq. 5, we remove it and train the model with a back-translation loss plus a language model negative log likelihood loss (which we denote as BT+NLL) as an ablation baseline.

For each task, we also include strong baseline numbers from related work if available.

For our method we select the model with the best validation ELBO, and for UNMT or BT+NLL we select the model with the best back-translation loss.

Complete model configurations and hyperparameters can be found in Appendix A.1.

Word Substitution Decipherment.

Word decipherment aims to uncover the plain text behind a corpus that was enciphered via word substitution where word in the vocabulary is mapped to a unique type in a cipher dictionary (Dou & Knight, 2012; Shen et al., 2017; Yang et al., 2018) .

In our formulation, the model is presented with a non-parallel corpus of English plaintext and the ciphertext.

We use the data in (Yang et al., 2018 ) which provides 200K sentences from each domain.

While previous work (Shen et al., 2017; Yang et al., 2018) controls the difficulty of this task by varying the percentage of words that are ciphered, we directly evaluate on the most difficult version of this task -100% of the words are enciphered (i.e. no vocabulary sharing in the two domains).

We select the model with the best unsupervised reconstruction loss, and evaluate with BLEU score on the test set which contains 100K parallel sentences.

Sentiment Transfer.

Sentiment transfer is a task of paraphrasing a sentence with a different sentiment while preserving the original content.

Evaluation of sentiment transfer is difficult and is still an open research problem (Mir et al., 2019) .

Evaluation focuses on three aspects: attribute control, content preservation, and fluency.

A successful system needs to perform well with respect to all three aspects.

We follow prior work by using three automatic metrics (Yang et al., 2018; Lample et al., 2019) : classification accuracy, self-BLEU (BLEU of the output with the original sentence as the reference), and the perplexity (PPL) of each system's output under an external language model.

We pretrain a convolutional classifier (Kim, 2014) to assess classification accuracy, and use an LSTM language model pretrained on each domain to compute the PPL of system outputs.

We use the Yelp reviews dataset collected by Shen et al. (2017) which contains 250K negative sentences and 380K positive sentences.

We also use a small test set that has 1000 human-annotated parallel sentences introduced in Li et al. (2018) .

We denote the positive sentiment as domain D 1 and the negative sentiment as domain D 2 .

We denote self-BLEU score as BLEU s and reference BLEU score on the 1000 sentences as BLEU r .

Formality Transfer.

Next, we consider a harder task of modifying the formality of a sequence.

We use the GYAFC dataset (Rao & Tetreault, 2018) , which contains formal and informal sentences from two different domains.

In this paper, we use the Entertainment and Music domain, which has about 52K training sentences, 5K development sentences, and 2.5K test sentences.

This dataset actually contains parallel data between formal and informal sentences, which we use only for evaluation.

We follow the evaluation of sentiment transfer task and test models on three axes.

Since the test set is a parallel corpus, we only compute reference BLEU and ignore self-BLEU.

We use D 1 to denote formal text, and D 2 to denote informal text.

Author Imitation.

Author imitation is the task of paraphrasing a sentence to match another author's style.

The dataset we use is a collection of Shakespeare's plays translated line by line into modern English.

It was collected by Xu et al. (2012) 5 and used in prior work on supervised style trans-4 The model they used is slightly different from the original model of Lample et al. (2018) in certain details -e.g.

the addition of a pooling layer after attention.

We re-implement their model in our codebase for fair comparison and verify that our re-implementation achieves performance competitive with the original paper.

5 https://github.com/tokestermw/tensorflow-shakespeare fer (Jhamtani et al., 2017) .

This is a parallel corpus and thus we follow the setting in the formality transfer task.

We use D 1 to denote modern English, and D 2 to denote Shakespeare-style English.

Related Language Translation.

Next, we test our method on a challenging related language translation task (Pourdamghani & Knight, 2017; Yang et al., 2018) .

This task is a natural test bed for unsupervised sequence transduction, because the goal is to preserve the meaning of the source sentence while rewriting it into the target language.

For our experiments, we choose Bosnian (bs) and Serbian (sr) as the related language pairs.

We follow Yang et al. (2018) to report BLEU-1 score on this task since BLEU-4 score is close to zero.

Unsupervised MT.

In order to draw connections with a related work on general unsupervised machine translation, we also evaluate on the WMT'16 German English translation task.

This task is substantially more difficult than the style transfer tasks considered so far.

We compare with the state-of-the-art UNMT system using the existing implementation from the XLM codebase, 6 and implement our approach in the same framework with XLM initialization for fair comparison.

We train both systems on 5M non-parallel sentences from each language.

Results for the full suite of tasks are collected in Tables 1 and 2 .

We list the PPL of the test set under the external LM for both the source and target domain in Table  1 .

PPL of system outputs should be compared to PPL of the test set itself because extremely low PPL often indicates that the generated sentences are short or trivial.

Tables 1 and 2 demonstrate some general trends.

First, UNMT is able to outperform other prior methods in unsupervised text style transfer, such as (Yang et al., 2018; Hu et al., 2017; Shen et al., 2017) .

The performance improvements of UNMT indicate that flexible and powerful architectures are crucial (prior methods generally do not have attention mechanism).

Second, our model achieves comparable classification accuracy to UNMT but outperforms it in all style transfer tasks in terms of the reference-BLEU, which is probably the most important metric since it directly measures the quality of the final generations against gold parallel data.

This indicates that our method is both effective and consistent across many different tasks.

Finally, the BT+NLL baseline is sometimes quite competitive, which indicates that the addition of a language model alone can be beneficial.

However, our method consistently outperforms the simple BT+NLL method, which indicates the effectiveness of the additional entropy regularizer in Eq. 5 that is the byproduct of our probabilistic formulation.

Next, we examine the PPL of the system outputs under pretrained domain LMs, which should be evaluated in comparison with the PPL of the test set itself.

For both the sentiment transfer and the formality transfer tasks in Table 1 , BT+NLL achieves extremely low PPL, lower than the PPL of the test corpus in the target domain.

After a close examination of the output, we find that it contains many repeated and overly simple outputs.

For example, the system generates many examples of "I love this place" when transferring negative to positive sentiment (see Appendix A.3 for examples).

It is not surprising that such a trivial output has low perplexity, high accuracy, and low BLEU score.

On the other hand, our system obtains reasonably competitive PPL, and our approach achieves the highest accuracy and higher BLEU score than the UNMT baseline.

Parameter Sharing.

We also conducted an experiment on the word substitution decipherment task, where we remove parameter sharing (as explained in Section 3.2) between two directions of transduction distributions, and optimize two encoder-decoder instead.

We found that the model only obtained an extremely low BLEU score and failed to generate any meaningful outputs.

Performance vs. Domain Divergence.

Figure 3 plots the relative improvement of our method over UNMT with respect to accuracy of a naive Bayes' classifier trained to predict the domain of test sentences.

Tasks with high classification accuracy likely have more divergent domains.

We can see that for decipherment and en-de translation, where the domains have different vocabularies and thus are easily distinguished, our method yields a smaller gain over UNMT.

This likely indicates that the (discrimination) regularization effect of the LM priors is less important or necessary when the two domains are very different.

Why do we do better than UNMT?

Finally, we examine in detail the output of our model and UNMT for the author imitation task.

We pick this task because the reference outputs for the test set are provided, aiding analysis.

Examples shown in Table 3 demonstrate that UNMT tends to make overly large changes to the source so that the original meaning is lost, while our method is better at preserving the content of the source sentence.

Next, we quantitatively examine the outputs from UNMT and our method by comparing the F1 measure of words bucketed by their syntactic tags.

We use the open-sourced compare-mt tool (Neubig et al., 2019) , and the results are shown in Figure 4 .

Our system has an advantage over UNMT in all word categories.

In particular, our system is much better at generating nouns, which contribute to preserving the content of the sentences.

Greedy vs. Sample-based Gradient Approximation.

In our experiments, we use greedy decoding from the inference network to approximate the expectation required by ELBO, which is a biased estimator.

The main purpose of this approach is to reduce the variance of the gradient estimator during training, especially in the early stages when the variance of sample-based approaches is quite high.

As an ablation experiment on the sentiment transfer task we compare greedy and sample-based gradient approximations in terms of both train and test ELBO, as well as task performance corresponding to best test ELBO.

After the model is fully trained, we find that the sample-based approximation has Table 4 , where the sampled-based training underperforms on both ELBO and task evaluations.

As noted above, to stabilize the training process, we stop gradients from propagating to the inference network from the reconstruction loss.

Does this approach indeed better optimize the actual probabilistic objective (i.e. ELBO) or only indirectly lead to improved task evaluations?

In this section we use sentiment transfer as an example task to compare different methods for propagating gradients and evaluate both ELBO and task evaluations.

Specifically, we compare three different methods:

??? Stop Gradient: The gradients from reconstruction loss are not propagated to the inference network.

This is the method we use in all previous experiments.

??? Gumbel Softmax (Jang et al., 2017) : Gradients from the reconstruction loss are propagated to the inference network with the straight-through Gumbel estimator.

??? REINFORCE (Sutton et al., 2000) : Gradients from reconstruction loss are propagated to the inference network with ELBO as a reward function.

This method has been used in previous work for semi-supervised sequence generation (Miao & Blunsom, 2016; Yin et al., 2018) , but often suffers from instability issues.

We report the train and test ELBO along with task evaluations in Table 5 , and plot the learning curves on validation set in Figure 5 .

7 While being much simpler, we show that the stop-gradient trick produces superior ELBO over Gumbel Softmax and REINFORCE.

This result suggests that stopping gradient helps better optimize the likelihood objective under our probabilistic formulation in comparison with other optimization techniques that propagate gradients, which is counter-intuitive.

A likely explanation is that as a gradient estimator, while clearly biased, stop-gradient has substantially reduced variance.

In comparison with other techniques that offer reduced bias but extremely high variance when applied to our model class (which involves discrete sequences as latent variables), stop-gradient actually leads to better optimization of our objective because it achieves better balance of bias and variance overall.

A.1 MODEL CONFIGURATIONS.

We adopt the following attentional encoder-decoder architecture for UNMT, BT+NLL, and our method across all the experiments:

??? We use word embeddings of size 128.

??? We use 1 layer LSTM with hidden size of 512 as both the encoder and decoder.

??? We apply dropout to the readout states before softmax with a rate of 0.3.

??? Following Lample et al. (2019) , we add a max pooling operation over the encoder hidden states before feeding it to the decoder.

Intuitively the pooling window size would control how much information is preserved during transduction.

A window size of 1 is equivalent to standard attention mechanism, and a large window size corresponds to no attention.

See Appendix A.2 for how to select the window size.

??? There is a noise function for UNMT baseline in its denoising autoencoder loss (Lample et al., 2017; .

We use the default noise function and noise hyperparameters in Lample et al. (2017) .

For BT+NLL and our method we found that adding the extra noise into the self-reconstruction loss (Eq. 4) often hurts the performance, because we already have a language model to avoid the local optimum of direct-copy generation.

A.2 HYPERPARAMETER TUNING.

We vary pooling windows size as {1, 5}, the decaying patience hyperparameter k for selfreconstruction loss (Eq. 4) as {1, 2, 3}. For the baseliens UNMT and BT+NLL, we also try the option of not annealing the self-reconstruction loss at all as in the unsupervised machine translation task (Lample et al., 2018) .

We vary the weight ?? for the NLL term (BT+NLL) or the KL term (our method) as {0.001, 0.01, 0.03, 0.05, 0.1}.

We list some examples of the sentiment transfer task in Table 6 .

Notably, the BT+NLL method tends to produce extremely short and simple sentences.

In Section 5 we mentioned that the baseline BT+NLL has a low perplexity for some tasks because it tends to generate overly simple and repetitive sentences.

From Table 1 we see that two representative tasks are sentiment transfer and formatliy transfer.

In Appendix A.3 we have demonstrated some examples for sentiment transfer, next we show some repetitive samples of BT+NLL in Table 7 .

<|TLDR|>

@highlight

We formulate a probabilistic latent sequence model to tackle unsupervised text style transfer, and show its effectiveness across a suite of unsupervised text style transfer tasks. 