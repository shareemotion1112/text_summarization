In this paper, we propose to extend the recently introduced model-agnostic meta-learning algorithm (MAML, Finn et al., 2017) for low resource neural machine translation (NMT).

We frame low-resource translation as a meta-learning problem, and we learn to adapt to low-resource languages based on multilingual high-resource language tasks.

We use the universal lexical representation (Gu et al., 2018b) to overcome the input-output mismatch across different languages.

We evaluate the proposed meta-learning strategy using eighteen European languages (Bg, Cs, Da, De, El, Es, Et, Fr, Hu, It, Lt, Nl, Pl, Pt, Sk, Sl, Sv and Ru) as source tasks and five diverse languages (Ro, Lv, Fi, Tr, and Ko) as target tasks.

We show that the proposed approach significantly outperforms the multilingual, transfer learning based approach (Zoph et al., 2016) and enables us to train a competitive NMT system with only a fraction of training examples.

For instance, the proposed approach can achieve as high as 22.04 BLEU on Romanian-English WMT’16 by seeing only 16,000 translated words (~600 parallel sentences).

Despite the massive success brought by neural machine translation (NMT, BID36 BID4 BID37 , it has been noticed that the vanilla NMT often lags behind conventional machine translation systems, such as statistical phrase-based translation systems (PBMT, BID24 , for low-resource language pairs (see, e.g., BID23 .

In the past few years, various approaches have been proposed to address this issue.

The first attempts at tackling this problem exploited the availability of monolingual corpora BID17 BID32 BID40 .

It was later followed by approaches based on multilingual translation, in which the goal was to exploit knowledge from high-resource language pairs by training a single NMT system on a mix of high-resource and low-resource language pairs (Firat et al., 2016a,b; BID27 BID21 BID19 .

Its variant, transfer learning, was also proposed by BID42 , in which an NMT system is pretrained on a high-resource language pair before being finetuned on a target low-resource language pair.

In this paper, we follow up on these latest approaches based on multilingual NMT and propose a meta-learning algorithm for low-resource neural machine translation.

We start by arguing that the recently proposed model-agnostic meta-learning algorithm (MAML, Finn et al., 2017) could be applied to low-resource machine translation by viewing language pairs as separate tasks.

This view enables us to use MAML to find the initialization of model parameters that facilitate fast adaptation for a new language pair with a minimal amount of training examples ( §3).

Furthermore, the vanilla MAML however cannot handle tasks with mismatched input and output.

We overcome this limitation by incorporating the universal lexical representation BID15 and adapting it for the meta-learning scenario ( §3.3).We extensively evaluate the effectiveness and generalizing ability of the proposed meta-learning algorithm on low-resource neural machine translation.

We utilize 17 languages from Europarl and Russian from WMT as the source tasks and test the meta-learned parameter initialization against five target languages (Ro, Lv, Fi, Tr and Ko), in all cases translating to English.

Our experiments using only up to 160k tokens in each of the target task reveal that the proposed meta-learning approach outperforms the multilingual translation approach across all the target language pairs, and the gap grows as the number of training examples 2 Background Neural Machine Translation (NMT) Given a source sentence X = {x 1 , ..., x T }, a neural machine translation model factors the distribution over possible output sentences Y = {y 1 , ..., y T } into a chain of conditional probabilities with a leftto-right causal structure: DISPLAYFORM0 where special tokens y 0 ( bos ) and y T +1 ( eos ) are used to represent the beginning and the end of a target sentence.

These conditional probabilities are parameterized using a neural network.

Typically, an encoder-decoder architecture BID36 BID9 BID4 with a RNN-based decoder is used.

More recently, architectures without any recurrent structures BID13 BID37 have been proposed and shown to speedup training while achieving state-of-the-art performance.

Low Resource Translation NMT is known to easily over-fit and result in an inferior performance when the training data is limited BID23 .

In general, there are two ways for handling the problem of low resource translation:(1) utilizing the resource of unlabeled monolingual data, and (2) sharing the knowledge between low-and high-resource language pairs.

Many research efforts have been spent on incorporating the monolingual corpora into machine translation, such as multi-task learning BID17 Zong, 2016), back-translation (Sennrich et al., 2015) , dual learning BID20 and unsupervised machine translation with monolingual corpora only for both sides BID3 BID26 .

For the second approach, prior researches have worked on methods to exploit the knowledge of auxiliary translations, or even auxiliary tasks.

For instance, BID8 ; BID28 investigate the use of a pivot to build a translation path between two languages even without any directed resource.

The pivot can be a third language or even an image in multimodal domains.

When pivots are not easy to obtain, BID11 ; BID27 ; BID21 have shown that the structure of NMT is suitable for multilingual machine translation.

BID15 also showed that such a multilingual NMT system could improve the performance of low resource translation by using a universal lexical representation to share embedding information across languages.

All the previous work for multilingual NMT assume the joint training of multiple high-resource languages naturally results in a universal space (for both the input representation and the model) which, however, is not necessarily true, especially for very low resource cases.

Meta Learning In the machine learning community, meta-learning, or learning-to-learn, has recently received interests.

Meta-learning tries to solve the problem of "fast adaptation on new training data." One of the most successful applications of meta-learning has been on few-shot (or oneshot) learning BID25 , where a neural network is trained to readily learn to classify inputs based on only one or a few training examples.

There are two categories of meta-learning:1.

learning a meta-policy for updating model parameters (see, e.g., BID1 BID18 BID30 2.

learning a good parameter initialization for fast adaptation (see, e.g., BID10 BID38 BID35 .In this paper, we propose to use a meta-learning algorithm for low-resource neural machine translation based on the second category.

More specifically, we extend the idea of model-agnostic metalearning (MAML, Finn et al., 2017) in the multilingual scenario.

The underlying idea of MAML is to use a set of source tasks T 1 , . . .

, T K to find the initialization of parameters θ 0 from which learning a target task T 0 would require only a small number of training examples.

In the context of machine translation, this amounts to using many high-resource language pairs to find good initial parameters and training a new translation model on a low-resource language starting from the found initial parameters.

This process can be understood as That is, we meta-learn the initialization from auxiliary tasks and continue to learn the target task.

DISPLAYFORM0 We refer the proposed meta-learning method for NMT to MetaNMT.

See FIG0 for the overall illustration.

Given any initial parameters θ 0 (which can be either random or meta-learned), the prior distribution of the parameters of a desired NMT model can be defined as an isotropic Guassian: DISPLAYFORM0 where 1/β is a variance.

With this prior distribution, we formulate the language-specific learning process Learn(D T ; θ 0 ) as maximizing the logposterior of the model parameters given data D T : DISPLAYFORM1 where we assume p(X|θ) to be uniform.

The first term above corresponds to the maximum likelihood criterion often used for training a usual NMT system.

The second term discourages the newly learned model from deviating too much from the initial parameters, alleviating the issue of overfitting when there is not enough training data.

In practice, we solve the problem above by maximizing the first term with gradient-based optimization and early-stopping after only a few update steps.

Thus, in the low-resource scenario, finding a good initialization θ 0 strongly correlates the final performance of the resulting model.

We find the initialization θ 0 by repeatedly simulating low-resource translation scenarios using auxiliary, high-resource language pairs.

Following Finn et al. FORMULA0 we achieve this goal by defining the meta-objective function as DISPLAYFORM0 where k ∼ U({1, . . .

, K}) refers to one metalearning episode, and D T , D T follow the uniform distribution over T 's data.

We maximize the meta-objective function using stochastic approximation BID31 with gradient descent.

For each episode, we uniformly sample one source task at random, T k .

We then sample two subsets of training examples independently from the chosen task, D T k and D T k .

We use the former to simulate languagespecific learning and the latter to evaluate its outcome.

Assuming a single gradient step is taken only the with learning rate η, the simulation is: DISPLAYFORM1 Once the simulation of learning is done, we evaluate the updated parameters θ k on D T k , The gradient computed from this evaluation, which we refer to as meta-gradient, is used to update the meta model θ.

It is possible to aggregate multiple episodes of source tasks before updating θ: where η is the meta learning rate.

Unlike a usual learning scenario, the resulting model θ 0 from this meta-learning procedure is not necessarily a good model on its own.

It is however a good starting point for training a good model using only a few steps of learning.

In the context of machine translation, this procedure can be understood as finding the initialization of a neural machine translation system that could quickly adapt to a new language pair by simulating such a fast adaptation scenario using many high-resource language pairs.

DISPLAYFORM2

We use the following approximation property DISPLAYFORM0 where ν is a small constant and DISPLAYFORM1 In practice, we find that it is also possible to ignore the second-order term, ending up with the following simplified update rule: DISPLAYFORM2 Related Work: Multilingual Transfer Learning The proposed MetaNMT differs from the existing framework of multilingual translation BID27 BID21 BID15 or transfer learning BID42 .

The latter can be thought of as solving the following problem: DISPLAYFORM3 We omit the subscript k for simplicity.where D k is the training set of the k-th task, or language pair.

The target low-resource language pair could either be a part of joint training or be trained separately starting from the solution θ 0 found from solving the above problem.

The major difference between the proposed MetaNMT and these multilingual transfer approaches is that the latter do not consider how learning happens with the target, low-resource language pair.

The former explicitly incorporates the learning process within the framework by simulating it repeatedly in Eq. (2).

As we will see later in the experiments, this results in a substantial gap in the final performance on the low-resource task.

Illustration In Fig. 2 , we contrast transfer learning, multilingual learning and meta-learning using three source language pairs (Fr-En, Es-En and Pt-En) and two target pairs (Ro-En and Lv-En).

Transfer learning trains an NMT system specifically for a source language pair (Es-En) and finetunes the system for each target language pair (RoEn, Lv-En).

Multilingual learning often trains a single NMT system that can handle many different language pairs (Fr-En, Pt-En, Es-En), which may or may not include the target pairs (Ro-En, LvEn).

If not, it finetunes the system for each target pair, similarly to transfer learning.

Both of these however aim at directly solving the source tasks.

On the other hand, meta-learning trains the NMT system to be useful for fine-tuning on various tasks including the source and target tasks.

This is done by repeatedly simulating the learning process on low-resource languages using many high-resource language pairs (Fr-En, Pt-En, Es-En).

I/O mismatch across language pairs One major challenge that limits applying meta-learning for low resource machine translation is that the approach outlined above assumes the input and output spaces are shared across all the source and target tasks.

This, however, does not apply to ma-chine translation in general due to the vocabulary mismatch across different languages.

In multilingual translation, this issue has been tackled by using a vocabulary of sub-words BID32 or characters BID27 shared across multiple languages.

This surface-level sharing is however limited, as it cannot be applied to languages exhibiting distinct orthography (e.g., IndoEuroepan languages vs. Korean.)Universal Lexical Representation (ULR) We tackle this issue by dynamically building a vocabulary specific to each language using a keyvalue memory network (Miller et al., 2016; BID16 , as was done successfully for low-resource machine translation recently by BID15 .

We start with multilingual word embedding matrices k query ∈ R |V k |×d pretrained on large monolingual corpora, where V k is the vocabulary of the k-th language.

These embedding vectors can be obtained with small dictionaries of seed word pairs BID2 BID34 or in a fully unsupervised manner BID41 BID0 .

We take one of these languages k to build universal lexical representation consisting of a universal embedding matrix u ∈ R M ×d and a corresponding key matrix key ∈ R M ×d , where M < |V k |.

Both k query and key are fixed during meta-learning.

We then compute the language-specific embedding of token x from the language k as the convex sum of the universal embedding vectors by DISPLAYFORM0 DISPLAYFORM1 and τ is set to 0.05.

This approach allows us to handle languages with different vocabularies using a fixed number of shared parameters ( u , key and A.)Learning of ULR It is not desirable to update the universal embedding matrix u when finetuning on a small corpus which contains a limited set of unique tokens in the target language, as it could adversely influence the other tokens' embedding vectors.

We thus estimate the change to each embedding vector induced by languagespecific learning by a separate parameter ∆ k [x]: DISPLAYFORM2 During language-specific learning, the ULR Preprocessing and ULR Initialization As described in §3.3, we initialize the query embedding vectors k query of all the languages.

For each language, we use the monolingual corpora built from Wikipedia 7 and the parallel corpus.

The concatenated corpus is first tokenized and segmented using byte-pair encoding (BPE, BID33 , resulting in 40, 000 subwords for each language.

We then estimate word vectors using fastText BID5 and align them across all the languages in an unsupervised way using MUSE BID0 to get multilingual word vectors.

We use the multilingual word vectors of the 20,000 most frequent words in English to form the universal embedding matrix u .

Model We utilize the recently proposed Transformer BID37 as an underlying NMT system.

We implement Transformer in this paper based on BID14 8 and mod-ify it to use the universal lexical representation from §3.3.

We use the default set of hyperparameters (d model = d hidden = 512, n layer = 6, n head = 8, n batch = 4000, t warmup = 16000) for all the language pairs and across all the experimental settings.

We refer the readers to BID37 BID14 for the details of the model.

However, since the proposed metalearning method is model-agnostic, it can be easily extended to any other NMT architectures, e.g. RNN-based sequence-to-sequence models with attention BID4 .Learning We meta-learn using various sets of source languages to investigate the effect of source task choice.

For each episode, by default, we use a single gradient step of language-specific learning with Adam BID22 per computing the meta-gradient, which is computed by the first-order approximation in Eq. (3).For each target task, we sample training examples to form a low-resource task.

We build tasks of 4k, 16k, 40k and 160k English tokens for each language.

We randomly sample the training set five times for each experiment and report the average score and its standard deviation.

Each fine-tuning

Ro is done on a training set, early-stopped on a validation set and evaluated on a test set.-En Lv-En Fi-En Tr-En Ko-EnFine-tuning Strategies The transformer consists of three modules; embedding, encoder and decoder.

We update all three modules during metalearning, but during fine-tuning, we can selectively tune only a subset of these modules.

Following BID42 , we consider three fine-tuning strategies; (1) fine-tuning all the modules (all), (2) fine-tuning the embedding and encoder, but freezing the parameters of the decoder (emb+enc) and (3) fine-tuning the embedding only (emb).

We metalearn the initial models on all the source tasks using either Ro-En or Lv-En as a validation task.

We also train the initial models to be multilingual translation systems.

We fine-tune them using the four target tasks (Ro-En, Lv-En, Fi-En and Tr-En; 16k tokens each) and compare the proposed meta-learning strategy and the multilingual, transfer learning strategy.

As presented in FIG2 , the proposed learning approach significantly outperforms the multilingual, transfer learning strategy across all the target tasks regardless of which target task was used for early stopping.

We also notice that the emb+enc strategy is most effective for both meta-learning and transfer learning approaches.

With the proposed meta-learning and emb+enc fine-tuning, the final NMT systems trained using only a fraction of all available training examples achieve 2/3 (Ro-En) and 1/2 (Lv-En, Fi-En and Tr-En) of the BLEU score achieved by the models trained with full training sets.

Similarly to training any other neural network, meta-learning still requires early-stopping to avoid overfitting to a specific set of source tasks.

In doing so, we observe that the choice of a validation task has nonnegligible impact on the final performance.

For instance, as shown in FIG2 , Fi-En benefits more when Ro-En is used for validation, while the opposite happens with Tr-En.

The relationship between the task similarity and the impact of a validation task must be investigated further in the future.

Training Set Size We vary the size of the target task's training set and compare the proposed meta-learning strategy and multilingual, transfer learning strategy.

We use the emb+enc fine-tuning on Ro-En and Fi-En.

FIG3 demonstrates that the meta-learning approach is more robust to the drop in the size of the target task's training set.

The gap between the meta-learning and transfer learning grows as the size shrinks, confirming the effectiveness of the proposed approach on extremely lowresource language pairs.

TAB2 , we present the results on all five target tasks obtained while varying the source task set.

We first see that it is Source (Tr) google mülteciler için 11 milyon dolar toplamaküzere bagış eşleştirme kampanyasını başlattı .

Target google launches donation-matching campaign to raise $ 11 million for refugees .

Meta-0 google refugee fund for usd 11 million has launched a campaign for donation .

Meta-16k google has launched a campaign to collect $ 11 million for refugees .

always beneficial to use more source tasks.

Although the impact of adding more source tasks varies from one language to another, there is up to 2× improvement going from one source task to 18 source tasks (Lv-En, Fi-En, Tr-En and Ko-En).

The same trend can be observed even without any fine-tuning (i.e., unsupervised translation, BID26 BID3 ).

In addition, the choice of source languages has different implications for different target languages.

For instance, Ro-En benefits more from {Es, Fr, It, Pt} than from {De, Ru}, while the opposite effect is observed with all the other target tasks.

Training Curves The benefit of meta-learning over multilingual translation is clearly demonstrated when we look at the training curves in FIG4 .

With the multilingual, transfer learning approach, we observe that training rapidly saturates and eventually degrades, as the model overfits to the source tasks.

MetaNMT on the other hand continues to improve and never degrades, as the metaobjective ensures that the model is adequate for fine-tuning on target tasks rather than for solving the source tasks.

Sample Translations We present some sample translations from the tested models in TAB4 .Inspecting these examples provides the insight into the proposed meta-learning algorithm.

For instance, we observe that the meta-learned model without any fine-tuning produces a word-by-word translation in the first example (Tr-En), which is due to the successful use of the universal lexcial representation and the meta-learned initialization.

The system however cannot reorder tokens from Turkish to English, as it has not seen any training example of Tr-En.

After seeing around 600 sentence pairs (16K English tokens), the model rapidly learns to correctly reorder tokens to form a better translation.

A similar phenomenon is observed in the Ko-En example.

These cases could be found across different language pairs.

In this paper, we proposed a meta-learning algorithm for low-resource neural machine translation that exploits the availability of high-resource languages pairs.

We based the proposed algorithm on the recently proposed model-agnostic metalearning and adapted it to work with multiple languages that do not share a common vocabulary using the technique of universal lexcal representation, resulting in MetaNMT.

Our extensive evaluation, using 18 high-resource source tasks and 5 low-resource target tasks, has shown that the proposed MetaNMT significantly outperforms the existing approach of multilingual, transfer learning in low-resource neural machine translation across all the language pairs considered.

The proposed approach opens new opportunities for neural machine translation.

First, it is a principled framework for incorporating various extra sources of data, such as source-and targetside monolingual corpora.

Second, it is a generic framework that can easily accommodate existing and future neural machine translation systems.

@highlight

we propose a meta-learning approach for low-resource neural machine translation that can rapidly learn to translate on a new language