Despite much success in many large-scale language tasks, sequence-to-sequence (seq2seq) models have not been an ideal choice for conversational modeling as they tend to generate generic and repetitive responses.

In this paper, we propose a Latent Topic Conversational Model (LTCM) that augments the seq2seq model with a neural topic component to better model human-human conversations.

The neural topic component encodes information from the source sentence to build a global “topic” distribution over words, which is then consulted by the seq2seq model to improve generation at each time step.

The experimental results show that the proposed LTCM can generate more diverse and interesting responses by sampling from its learnt latent representations.

In a subjective human evaluation, the judges also confirm that LTCM is the preferred option comparing to competitive baseline models.

Sequence-to-Sequence model (seq2seq) BID35 , as a data-driven approach to mapping between two arbitrary length sequences, has attracted much attention and been widely applied to many natural language processing tasks such as machine translation BID20 , syntactic parsing , and summarisation BID28 .

Neural conversational models BID32 BID30 are the latest development in open-domain conversational modelling, where seq2seq-based models are employed for learning dialogue decisions in an end-to-end fashion.

Despite promising results, the lack of explicit knowledge representations (or the inability to learn them from data) impedes the model from generating causal or even rational responses.

This leads to many problems discussed in previous literature such as generic responses BID17 , inconsistency BID18 , and redundancy and contradiction BID33 .On the other hand, goal-oriented dialogues BID45 use the notion of dialogue ontology to constrain the scope of conversation and facilitate rational system behaviour within the domain.

Neural network-based task-oriented dialogue systems usually retrieve knowledge from a pre-defined database either by discrete accessing BID41 BID3 or through an attention mechanism BID8 .

The provision of this database offers a proxy for language grounding, which is crucial to guide the generation or selection of the system responses.

As shown in BID40 , a stochastic neural dialogue model can generate diverse yet rational responses mainly because they are heavily driven by the knowledge the model is conditioned on.

Despite the need for explicit knowledge representations, building a general-purpose knowledge base and actually making use of it have been proven difficult BID22 BID25 .

Therefore, progress has been made in conditioning the seq2seq model on coarse-grained knowledge representations, such as a fuzzily-matched retrieval result via attention BID10 or a set of pre-organised topic or scenario labels .

In this work, we propose a hybrid of a seq2seq conversational model and a neural topic model -Latent Topic Conversational Model (LTCM) -to jointly learn the useful latent representations and the way to make use of them in a conversation.

LTCM uses its underlying seq2seq model to capture the local dynamics of a sentence while extracts and represents its global semantics by a mixture of topic components like topic models BID2 .

This separation of global semantics and local dynamics turns out to be crucial to the success of LTCM.Recent advances in neural variational inference BID27 BID23 have sparked a series of latent variable models applied to conversational modeling BID31 BID5 .

The majority of the work passes a Gaussian random variable to the hidden state of the LSTM decoder and employs the reparameterisation trick BID15 to build an unbiased and low-variance gradient estimator for updating the model parameters.

However, studies have shown that training this type of models for language generation tasks is tough because the effect of the latent variable tends to vanish and the language model would take over the entire generation process over time BID4 .

This results in several workarounds such as KL annealing BID4 BID5 , word dropout and historyless decoding BID4 , as well as auxiliary bag-of-word signals .

Unlike previous approaches, LTCM is similar to TopicRNN (Dieng et al., 2017) where it passes the latent variable to the output layer of the decoder and only back-propagates the gradient of the topic words to the latent variable.

In summary, the contribution of this paper is two-fold: first and most importantly, we show that LTCM can learn to generate more diverse and interesting responses by sampling from the learnt topic representations.

The results were confirmed by a corpus-based evaluation and a human assessment; secondly, we conducted a series of experiments to understand the properties of seq2seq-based latent variables models better, which may serve as rules of thumb for future model development.

We present the necessary building blocks of the LTCM model.

We first introduce the seq2seq-based conversational model and its latent variable variant, followed by an introduction of the neural topic models.

In general, a seq2seq model BID35 generates a target sequence given a source sequence using two Recurrent Network Networks (RNNs), one for encoding the source, another for decoding the target.

Given a user input u = {x 1 , x 2 , ...x U } in the conversational setting, the goal is to produce a machine response m = {y 1 , y 2 , ...y M } that maximises the conditional probability m * = argmax m p(m|u).

The decoder of the seq2seq model is effectively an RNN language model which measures the likelihood of a sequence through a joint probability distribution, DISPLAYFORM0 The conditional probability is then modeled by an RNN, DISPLAYFORM1 where h t is the hidden state at step t and function f W h (·) is the hidden state update that can either be a vanilla RNN cell or a more complex cell like Long Short-term Memory (LSTM) BID12 .

The initial state of the decoder h 0 is initialised by a vector representation of the source sentence, which is taken from the last hidden state of the encoder h 0 =ĥ U .

The encoder state update also follows Equation 3.While theoretically, RNN-based models can memorise arbitrarily long sequences if provided with sufficient capacity, in practice even the improved version such as LSTM or GRU BID7 encounter difficulties during optimisation BID1 .

This inability to memorising long-term dependencies prevents the model from extracting useful sentence-level semantics.

As a consequence, the model tends to focus on the low-hanging fruit (language modelling) during optimisation and yields a suboptimal result.

Latent variable conversational model BID31 BID5 ) is a derivative of the seq2seq model in which it incorporates a latent variable ν at the sentence-level to inject stochasticity and diversity.

The objective function of the latent variable model is DISPLAYFORM0 where ν is usually chosen to be Gaussian distributed and passed to the decoder at every time step where we rewrite Equation 3 as h t = f W h (y t−1 , h t−1 , ν).

Since the optimisation against Equation 4 is intractable, we apply variational inference and alternatively optimise the variational lowerbound, DISPLAYFORM1 where we introduce the inference network q(ν|u, m), a surrogate of p(ν|u), to approximate the true posterior during training.

Based on Equation 5, we can then sample ν ∼ q(ν|u, m) and apply the Gaussian reparameterisation trick BID15 to calculate the gradients and update the parameters.

Although latent variable conversational models were able to generate diverse responses, its optimisation has been proven difficult, and several tricks are needed to obtain a good result.

Among these tricks, KL loss annealing is the most general and effective approach BID4 .

The main idea of KL annealing is, instead of optimising the full KL term during training, we gradually increase using a linear schedule.

This way, the model is encouraged to encode information cheaply in ν without paying huge KL penalty in the early stage of training.

Probabilistic topic models are a family of models that are used to capture the global semantics of a document set BID34 .

They can be used as a tool to organise, summarise, and navigate document collections.

As an unsupervised approach, topic models rely on counting word co-occurrence in the same document to group words into topics.

Therefore, each topic represents a word cluster which puts most of its mass (weight) on this subset of the vocabulary.

Despite there are many probabilistic graphical topic models BID2 , we focus on neural topic models BID16 BID23 because they can be directly integrated into seq2seq model as a submodule of LTCM.One neural topic model that is similar to LDA is the Gaussian-softmax neural topic model introduced by BID24 .

The generation process works as following: DISPLAYFORM0

where β = {β 1 , β 2 , ...β K }, β k is the word distribution of topic k, and µ 0 and σ 0 are the mean and variance of an isotropic Gaussian.

The likelihood of a document d = {y 1 , y 2 , ...y D } is therefore, DISPLAYFORM0 Note that in the original LDA, both the θ and β are drawn from a Dirichlet prior.

Gaussian-softmax model, on the other hand, constrcuts θ from a draw of an isotropic Gaussian with parameters µ 0 and σ 0 , where as β is random initialised as a parameter of the network.

Like most of the topic models, Gaussian-softmax model makes the bag-of-words assumption where the word order is ignored.

This simple assumption sacrifices the ability to model local transitions between words and phrases in exchange for the capability to capture global semantics.

Therefore, although topic model could not be used as a conversational model itself, it is nevertheless a perfect fit to a sentence-level semantic extractor alongside a seq2seq model to improve the global coherence of the generated responses.

Model The proposed Latent Topic Conversational Model (LTCM) is a hybrid of the seq2seq conversational model and the neural topic model, as shown in FIG0 .

The neural topic sub-component is responsible for extracting and mapping between the input and output global semantics so that the seq2seq submodule can focus on perfecting local dynamics of the sentence such as syntax and word order.

Given a user input u and a machine response m, the generative process of LTCM can be described as the following, DISPLAYFORM0 2.

Draw a sentence-level latent vector ν ∼ p Λ (ν|u).

4.

Initialise the decoder hidden state h 0 =ĥ U , whereĥ U is the last encoder state.5.

Given y 1:t−1 , for the t-th word y t in the response, DISPLAYFORM0 , where DISPLAYFORM1 where DISPLAYFORM2 ) is a parametric isotropic Gaussian with a mean and variance both condition on the input prompt µ(u) = MLP(u), σ(u) = MLP(u).

To combine the seq2seq model with the neural topic module, we adopt the hard-decision style from TopicRNN (Dieng et al., 2017) by introducing an additional random variable l t .

The topic indicator l t is to decide whether or not to take the logits of the neural topic module into account.

If l t = 0, which indicates that y t is a stop-word, the topic vector θ would have no contribution to the final output.

However, if l t = 1, then the topic contribution term β i θ is added to the output of the seq2seq model, where β i is the word-topic vector for the i-th vocabulary word.

Although the topic word indicator l t is sampled during inference, during training it is treated as observed and can be produced by either a stop-word list or ranking words in the vocabulary by their inverse document frequencies.

This hard decision of l t is crucial for LTCM because it explicitly sets two gradient routes for the model: when l t = 1 the gradients are back-propagated to the entire network; otherwise, they only flow through the seq2seq model.

This is important because topic models are known to be bad at dealing with stop-words BID26 .

Therefore, preventing the topic model to learn from stop-words can help the extraction of global semantics.

Finally, the logits of the seq2seq and neural topic model are combined through an additive procedure.

This makes the gradient flow more straightforward and the training of LTCM becomes easier 1 .The parameters of LTCM can be denoted as DISPLAYFORM3 and L is the vocabulary size.

During training, the observed variables are input u, output m, and the topic word indicators l 1:M .

The parametric form of LTCM is therefore, DISPLAYFORM4 Inference As a direct optimisation of Equation 7 is intractable because it involves an integral over the continuous latent space, variational inference BID13 ) is applied to approximate the log-likelihood objective.

The variational lowerbound of Equation 7 can therefore be derived as DISPLAYFORM5 1 For example, LTCM does not need to be trained with KL annealing to achieve a good performance.

where q(θ|u, m) is the inference network introduced during training to approximate the true posterior.

The neural variational inference framework BID27 BID23 and the Gaussian reparameterisation trick BID15 BID29 are then followed to construct q(θ|u, m), DISPLAYFORM6 where Φ = {W a , Ω 1 , Ω 2 } is the new set of parameters introduced for the inference network, u b and m b are the bag-of-words representations for u and m, respectively.

Although q(θ|u, m) and p(θ|u) are both parameterised as an isotropic Gaussian distribution, the approximation q(θ|u, m) only functions during training by producing samples to compute the stochastic gradients, while p(θ|u) is the generative distribution that generates the required topic proportion vectors for composing the machine response.

Dataset We assessed the performance of the LTCM using both a corpus-based evaluation and a human assessment.

The dataset used in the experiments is a subset of the data collected by BID33 , which includes mainly the Reddit 2 data which contains about 1.7 billion messages (221 million conversations).

Given the large volume of the data, a random subset of 15 million singleturn conversations was selected for this experiment.

To process the Reddit data, messages belonging to the same post are organized as a tree, a single-turn conversation is extracted merely by treating each parent node as a prompt and its corresponding child nodes as responses.

A length of 50 words was set for both the source and target sequences during preprocessing.

Sentences with any nonRoman alphabet were also removed.

This filters out around 40% to 50% of the examples.

A few standardizations were made via regular expressions such as mapping all valid numbers to <number> and web URLs to <url>. A vocabulary size of 30K was set for encoder, decoder, and the neural topic component.

Model The LTCM model was implemented on the publicly available NMT 3 code base BID21 .

Three model types were compared in the experiments, the vanilla seq2seq conversational model (S2S) BID5 , and the Latent Topic Conversational Model (LTCM).

For all the seq2seq components, a 4-layer LSTM with 500 hidden units was used for both the encoder and decoder.

We used the GNMT style encoder where the first layer is a bidirectional LSTM, while the last three layers are unidirectional.

Residual connections were used BID11 to ease the optimisation of deep networks.

Layer Normalisation (Ba et al.) was applied to all the LSTM cells to facilitate learning.

The batch size was 128, and a dropout rate of 0.2 was used.

The Adam optimiser (Kingma & Ba, 2014) with a fixed annealing schedule was used to update the parameters.

For the latent variable conversational model, we explored the KL annealing strategy as suggested in BID4 where the KL loss is linearly increased and reaches to the full term after one training epoch.

In LTCM, the 300 words with the highest inverse document frequency are used as stop-words and the rest are treated as topic words.

Both the mutual angular regularisation BID43 and the l2 regularisation were applied to the β matrix during training.

To build the development and testing sets, additional 20K sentence pairs were extracted and divided evenly.

For evaluation, five metrics were reported: the approximated perplexity, the variational lowerbound, the KL loss, the sentence uniqueness and the Zipf coefficient BID5 of the generated responses.

Because the exact perplexity of the latent variable models is hard to assess due to sampling, an approximated perplexity is reported as suggested in BID9 .

For latent variable conversational models, the approximate distribution for computing perplexity is p(y t |y 1:t−1 , u) = t p(y t |h t ,ν), whereν is the mean estimate of ν.

While for LTCM it is p(y t |y 1: DISPLAYFORM0 where againθ is the mean estimate of θ.

Both latent variable model and LTCM used greedy decoding to make sure the diversity they produce comes from the latent variable.

For seq2seq model, however, we explored both the greedy and random sampling strategies.

Given a prompt, each model was requested to generate five responses.

This leads to 50K generated responses for the testing set.

The sentence uniqueness score and Zipf coefficient 4 , which were introduced both by Cao & Clark 2017 as proxies to evaluate sentence and lexicon diversity respectively, were computed.

The result of the corpus-based evaluation is presented in Table 1 .

The first block shows the performance of the baseline seq2seq model, either by greedy decoding or random sampling.

Unsurprisingly, S2S-sample can generate much more diverse responses than S2S-greedy.

However, these responses are not of high quality as can be seen in the human assessment in the next section.

One interesting observation is that the sentence uniqueness score of S2S-greedy is much lower than the ex- Table 3 : Pairwise preference assessment.

Note the numbers are the percentage of wins when comparing models in the first column with the ones in first row.

*p < 0.05 pected (2.65%< 20% 5 ).

This echoes the generic response problem mentioned in previous works BID17 BID31 .

The second block demonstrates the result of the latent variable conversational models.

As can be seen, neither sampling from a prior (LV-S2S, p(ν)) nor a conditional (LV-S2S, p(ν|u)) helps to beat the performance of the seq2seq model.

Although both models perform equally well in terms of perplexity and lowerbound, the likewise low uniqueness scores as seq2seq indicate that both of their latent variables collapse into a single mode and do not encode much information.

This was also observed in when training seq2seq-based latent variable models.

The KL annealed model LV-S2S, p(ν|u), +A, as suggested by BID4 , can help to mitigate this problem and achieve a much higher uniqueness score (42.6%).The third block shows the result of the LTCM models.

As can be seen, LTCM trades in its KL loss and variational lowerbound in exchange for a higher response diversity (higher uniqueness score and lower Zipf).

Interestingly, although the lowerbound was substantially worse than the baselines, the conditional LTCM models (LTCM, p(θ|u) and LTCM, p(θ|u), +V) can still reach comparable perplexities.

This indicates that most of the additional loss incurred by LTCM was to encode the discourse-level diversity into the latent variable and therefore may not be a bad idea.

Given that the latent variable of LTCM can encode more useful information, sampling from a conditional can therefore better tailor the neural topic component to the user prompt and produce more relevant responses (LTCM, p(θ) v.s. LTCM, p(θ|u)).

Overall speaking, LTCM can generate more diverse responses comparing to baselines by encoding more information into the latent space.

However, the slightly higher lowerbound and KL loss do not necessarily mean that the quality of the responses is worse.

More discussions follow in the next section.

Due to the difficulty in evaluating conversational agents BID39 , a human evaluation is usually necessary to assess the performance of the models.

To do a less biased evaluation, a set of judges (∼ 250) were recruited on AMT.

For each task (a prompt), two randomly selected models were paired and each of them was asked to generate five responses given the prompt.

There is a total of 5000 comparisons randomly split between all pairs.

This results in approximately 90 experiments per pair of comparison.

The number of tasks that each judge can do is capped to 20.

To consider the response diversity, each judge was asked to rate each of the five generated responses from 1 to 5 based on the interestingness and appropriateness scores.

The quality assessment is shown in TAB2 .

The numbers inside the brackets are calculated by averaging the mean of the generated responses across prompts, while the ones outside the brackets are the average of the maximum scores across prompts.

Moreover, at the end of the task, the judge was also asked to state a preference between the two systems.

The result is shown in Table 3 .

Prompt: what do you think about messi ?

S2S * i think he 's a good player .

LV-S2S+A *

he 's a fantastic player , but he 's not a good player .

* he 's a great player , but he 's not a good player .

* he 's a great player , but he needs to be more consistent .

LTCM * i love him .

* i think he 's a good player , but i feel like he 's a bit overrated .

* i think he 's a great player , but i do not think messi deserves to play for the rest of the season .

* i think messi is the best .Prompt: what is the purpose of existence ?

S2S * to create a universe that is not a universe .

LV-S2S+A * to be able to understand what you are saying .

LTCM * to be a <unk> .

* to be able to see the world .

* to be able to see things .

* to make it better .

Table 4 : Example comparisons of the three models: S2S-greedy, LV-S2S, p(ν|u), +A, and LTCM, p(θ|u).

The result is produced by removing duplicated sentences from the five generated responses.

Each * indicates the beginning of a response.

Figure 2: Analysis of the learned topic gate l t shown in percentage.

TAB2 shows that the average scores (numbers inside the brackets) of S2S-greedy, LV-S2S, p(ν|u), +A, and LTCM, p(θ|u) are pretty much the same (with the appropriateness of S2S-greedy slightly better).

However, the maximum scores (numbers outside the brackets) show that LTCM is the best among the four (interestingness: 3.97 and appropriateness: 4.04).

This indicates that although LTCM can generate pretty good responses, it could also produce sentences that are quite bad.

This variance in response quality could be beneficial if reinforcement learning is introduced to fine-tune the latent variable BID40 .

Table 3 shows the result of pairwise preference test between four models.

As can be seen, LTCM is the preferred option for most of the judges when compared to other approaches.

Table 4 shows a few examples for qualitative analysis of the models.

As shown in the table, LTCM can generate more diverse and interesting responses comparing to the baseline methods.

The diversity can be found at both the semantic and the syntactic level.

Figure 2 shows the analysis of the topic gate.

As can be seen, the learned gate corresponds to the human intuition and helps to coordinate the contribution of the two models while generating.

For more examples of the generated responses, please refer to Appendix A.

In this paper, we have proposed the Latent Topic Conversational Model (LTCM) for general-purpose conversational modeling.

We have shown that LTCM can generate more interesting and diverse responses by combining the seq2seq model and neural topic model so that global semantic representations and local word transitions can be modeled separately but learned jointly.

Both a corpus-based evaluation and a human assessment confirm this finding.

Future work would be to study the learned representations and use them to control the meaning of the generated responses.

Prompt: What is the purpose of dying?

S2S * Dying LV-S2S+A * The fact that you are not a doctor .

* The fact that you are not alone is not a good thing .

LTCM * To die .

* To make it a better life .

* To make it a reality .Prompt: aerial view of colpitt lake in halifax S2S * i 'm going to go out on a limb and say that the lake is a lake .

LV-S2S+A * i bet it 's a lot of fun to drive through there .

* Prompt: ( spoilers ) what are your season finale predictions ?

S2S * i 'm going to be a <unk> fan .

LV-S2S+A * i think the <unk> will be the last one . . .

* i think the <unk> will be the next <unk> .

* i think the <unk> will be the one to go .

LTCM * i think it 's a pretty good idea .

* i think it 's going to be a tie between the <unk> and <unk> .

* jon <unk> dies .

* jon snow will be dead .

* jon snow will die .

Prompt: i 've been waiting all year and it 's finally happening S2S * i 'm still waiting for the <number> th .

LV-S2S+A * i 'm so excited !

LTCM * congratulations ! ! !

* You 're a little late to the party , but you 're still in the process of making a purchase .

* you 're a little late to the party , but you 're still young .

* you 're a lucky man .

congrats !

* you 're lucky you got it .

congrats !

Prompt: what kind of operating system you are using ?

S2S * <unk> LV-S2S+A * <unk> .

* ubuntu .

LTCM * i 'm a <unk> .

* i 'm a linux user .

* i use <unk> <smile> * linux .

* ubuntu linux .Prompt: i created the universe ! give me the gift certificate !

S2S * i 'm not sure if i should be happy or sad .

LV-S2S+A * i always get really excited when i see this .

*

i hope he 's okay .

* i need to get a job .

* i really like this .

thanks for sharing .

* i want to see this .

LTCM * i love this .

*

oh god , i love this guy .

*

oh my god , i love this .

*

oh my god , i love this guy .

*

oh my god , yes !

Table 6 : Extended examples of Table 4 .

The result is produced by removing duplicated sentences from the five generated responses.

Each * indicates the beginning of a response.

@highlight

Latent Topic Conversational Model, a hybrid of seq2seq and neural topic model to generate more diverse and interesting responses.

@highlight

This paper proposed the combination of topic model and seq2seq conversational model

@highlight

Proposes a conversational model with topical information by combining seq2seq model with neural topic models and shows the proposed model outperforms some the baseline model seq2seq and other latent variable model variant of seq2seq.

@highlight

The paper addresses the issue of enduring topicality in conversation models and proposes a model which is a combination of a neural topic model and a seq2seq-based dialog system. 