When communicating, humans rely on internally-consistent language representations.

That is, as speakers, we expect listeners to behave the same way we do when we listen.

This work proposes several methods for encouraging such internal consistency in dialog agents in an emergent communication setting.

We consider two hypotheses about the effect of internal-consistency constraints: 1) that they improve agents’ ability to refer to unseen referents, and 2) that they improve agents’ ability to generalize across communicative roles (e.g. performing as a speaker de- spite only being trained as a listener).

While we do not find evidence in favor of the former, our results show significant support for the latter.

Emergent communication is the study of how linguistic protocols evolve when agents are tasked to cooperate.

For example, agents engaged in a simple object retrieval task learn to communicate with one another in order to get the items they want .

To date, work of this type has each agent assume a conversational role.

Thus, agents are often trained only to speak or only to listen , or similarily trained to speak using a vocabulary disjoint from the vocabulary it is understands as a listener-e.g.

speaking only to ask questions ("what color?") and listening only to comprehend the answer ("blue") Das et al., 2017) .

These assumptions are misaligned with how we think about human communication, and with the way we'd like computational models to work in practice.

As humans, not only can we easily shift between roles, we also know that there is inherent symmetry between these roles: we expect others to speak (or listen) similarly to the way we do, and we know that others expect the same of us.

We test if dialog agents that incorporate the symmetry between themselves and their communicative partners learn more generalizable representations than those which do not.

We introduce three modifications to the agents to encourage that they abide by the "golden rule": speak/listen as you would want to be spoken/listened to.

Specifically, these modifications include self-play training objectives, shared embedding spaces, and symmetric decoding and encoding mechanisms that share parameters.

We test two hypotheses about the effect of the proposed modifications on emergent communication:

1.

Internal-consistency constraints improve agents' ability to generalize to unseen items-e.g.

training on "red square" and "blue circle" and then testing on "blue square".

2.

Internal-consistency constraints improve agents' ability to generalize across communicative roles-e.g.

training on "blue" as a listener, and using "blue" as a speaker when testing.

We evaluate the effect of each of the proposed modifications with two reference game datasets and two model architectures, an RNN model used by and a Transformer model.

We find no evidence to support that internal-consistency improves generalization to unseen items (Hypothesis 1), but significant evidence that these proposed constraints enable models to generalize learned representations across communicative roles (Hypothesis 2), even in the case of where the agent receives no direct training in the target (test) role.

All of our code and data are available at bit.ly/internal-consistency-emergent-communication.

Notation.

The space of possible references is parameterized by the number of attributes n f that describe each item (e.g. color) and the number of values n v each attribute can take (e.g.{red, blue}).

Each item o is a bag-of-features vector o P t0, 1u N where N " n f¨nv .

Each index o i is 1 if o expresses the ith feature value.

The speaker produces a message with symbols from a vocabulary V with length L. For comparison, we use the best-performing setting |V| " 100 and L " 10 from previous work .

Symbols in V are represented as 1-hot vectors.

In each round of the reference game, we construct xC, r, ry where C is the context (set of item column vectors stacked into a matrix), r is a vector representing the referent, and r is the index of the referent in C. We uniformly sample k´1 items as distractors to form C " to 1 , . . .

o k´1 uYtru.

The distractors are is sampled randomly each round (in every epoch).

We begin with a general architecture and training objective to underly all of our models (Sections 3.1 and 3.2).

We then introduce three modifications which can be used to encourage internallyconsistent representations: a self-play training objective, a shared embedding space, and a symmetric decoding and encoding mechanism with shared parameters (Section 3.3)

1 .

Agents contain four modules.

Embedding modules 1) E item P R NˆD , E message P R |V|ˆD .

E˚pxq embed items and messages.

When speaking, the decoder module 2) consumes the embedded referent E item prq and produces a discrete message M P V L .

Next, when listening, the encoder module 3) consumes embedded messages E message pM q P R WˆD and then produces a representation of the referentr P R D .

Finally, a non-parametric pointing module 4) produces a distribution P pCq over the context by matrix multiplyingr with the embedded context E item pCq.

The decoders emit one symbol at a time, auto-regressively producing fixed-length messages.

The messages are discretized with the straight-through Gumbel Softmax (Jang et al., 2016) as in Mordatch & Abbeel (2018) .

This converts a distribution to a one-hot vector while permitting gradients to flow through the operation and enables the agents to be optimized without reinforcement methods.

The Recurrent model uses a LSTM (Hochreiter & Schmidhuber, 1997) decoder when speaking and a LSTM encoder when listening, as in .

The Transformer model (Vaswani et al., 2017) uses a Transformer Decoder when speaking and a Transformer Encoder to encode when listening.

See Appendix A for implementation details and hyperparameters.

Speaking S : r, θ Ñ M and listening L : C, M , θ Ñ P pCq are both functions where

Lˆ|V| is a discrete-valued message with length L, P pCq is a distribution over the items in context, and θ are optimizable parameters.

We optimize the parameters θ A , θ B of agents A, B over a dataset D to select each referent r from among the distractors in its context C by minimizing the negative log likelihood of selecting the correct referent in context (Eq 1).

In our experiments, the speaker modules and the listener modules instantiate the function S and L respectively.

We investigate three internal-consistency constraints, that encourage internally-consistent representations.

Baseline agents consist of two separate sets of parameters, one for listening and one for speaking.

For example, the baseline recurrent model consists of two recurrent models.

This corresponds to the scenario where agents either speak or listen, but not both .

We introduce a 1) self-play loss for both agents of the same form as Eq. 1, except the given agent fulfills both roles, encouraging it to speak/listen to others the same way it speaks/listens to itself.

When we use the self-play training objective, we use it for both agents.

Next, 2) shared embedding agents use the same item embeddings and the same message embeddings when listening and speaking.

Finally, 3) symmetric encoding and decoding agents use the same parameters (but different mechanisms) to decode a message when speaking as it does to encode a message when listening.

Parameters are only ever shared between roles within an agent, and never between agents.

Our evaluation is based on the simple reference game 2 as described in Section 2, played across two datasets.

The datasets, summarized in Table 1 , target different aspects of lexical reference.

The first, Visual Attributes for Concepts (CONCEPTS) Silberer et al. (2013) , is derived from annotated images of actual items (animals, tools, etc) .

Thus, the data contains realistic co-occurance patterns: groups of attributes like has-head, has-mouth, and has-snout appears together several times, whereas has-seeds, has-mouth, made-of-metal never co-occur.

The intuition is that a good lexicon will exploit the structure by developing words which refer to groups of frequently co-occurring attributes (e.g. "mammal") and will describe unseen referents in terms of these primitive concepts.

The second dataset, SHAPES, is one we create ourselves in contrast with the CONCEPTS data.

In SHAPES, items correspond to abstract shapes which are exactly describable by their attributes (e.g. blue, shaded, hexagon).

All the attributes are independent and there is no co-occurence structure to exploit, so a good lexicon should ideally provide a means for uniquely specifying each attribute's value independently of the others.

We present experimental results aimed at testing the hypotheses stated in Section 1.

To provide intuition, we frame experiments in terms of two agents, "Alice" (Agent A) and "Bob" (Agent B), who are taking turns asking for and fetching toys.

We first test whether any of the proposed internal-consistency constraints improve the agents' ability to generalize to novel items-i.e.

items which consist of unseen combinations of features.

Here, we focus on the performance when models are trained and tested within the same communicative role.

This corresponds to the setting that has typically been used in prior work on emergent communication: Alice always speaks in order to ask for toys, Bob always responds by fetching them, and the pair's success is evaluated in terms of Alice's ability to describe new toys such that Bob correctly gets them.

For evaluation, we hold out a subset of the value combinations from each dataset to use for testing.

For example, the agents might be trained to refer to trshape:circle, color:reds, rshape:square, color:bluesu and then tested on its ability to refer to rshape:circle, color:blues.

We compute validation and test accuracies.

Table 2 : Results when agents are trained and tested in a single role, before any internal-consistency constraints.

These scores are mean accuracy with 95% confidence range averaged over 5 runs over different test set samplings (the distractors change).

one role, they can still impose internally consistent behavior across both roles.

It is conceivable that doing so might improve performance even though each agent remains in a fixed role, either by providing the model with additional information about the task structure, or simply by acting as a regularizer.

Thus, for completeness, we assess whether the internal-consistency constraints provide any advantage to the models in the vanilla emergent communication setting.

Table 3 shows the effect of adding the self-play objective in the fixed-role setting, across architectures and datasets.

The trends are mixed: it appears the additional signal only noises the baseline and symmetric models, whereas the shared embeddings models are able to leverage it effectively.

Thus, the effect is not clear enough to establish conclusively that the internal-consistency constraints help the agents generalize in this fixed-role setting, and in fact it may hurt.

Table 3 : Performance on task of referring to/fetching unseen items for baseline model compared against models with the internal-consistency constraints.

To highlight the difference of each constraint compared to the baseline performance, each delta compares the performance of the modified model to the baseline model.

In this setting, we see no clear advantage to enforcing internalconsistency via self-play.

These scores are mean accuracy with 95% confidence interval averaged over 5 runs over different test set samplings (the distractors change).

We now look at whether internal-consistency improves the agents' ability to generalize linguistic knowledge across roles.

For example, we can picture the following scenario: Alice is speaking to Bob, and asks for the "truck".

Bob hands her the doll, and Alice replies negatively, indicating that what she actually wanted was the truck.

Now, without additional direct supervision, when Bob wants the truck, will he know do use the word "truck"?

Such a setting is particularly relevant in practical settings, for example when robotic agents must reach high accuracy despite only limited access to human interaction for training.

We consider two versions of this setting, involving different levels of direct supervision (i.e. interaction with the other agent) as described below.

Training in one role.

Our first experimental setting assumes that Alice and Bob each only receive direct training in one role, e.g. Alice only ever speaks to Bob, so Alice only receives feedback on how she is performing as a speaker, and Bob on how he is performing as a listener.

However, both Alice and Bob are able to practice in the opposite role via self-play.

This setup is analogous to the experiment just discussed in Section 5.1.3.

However, unlike before, Alice and Bob will be tested in the roles opposite of those in which they were trained.

That is, if Alice was trained as a speaker, then she will be tested as a listener (on her ability to correctly identify items to which Bob refers).

Training in both roles.

In our second experimental setting, we assume Alice and Bob enjoy a healthy friendship, in which both take turns speaking and listening to each other, and thus both receive direct supervision in both roles.

However, they do not necessarily receive equal training on every vocabulary item.

Rather, there are some contexts in which Alice only speaks and other contexts in which she only listens.

Intuitively, this corresponds to a scenario in which Alice speaks exclusively about food (while Bob listens), while Bob speaks exclusively about toys (while Alice listens).

We are interested in testing how well Alice is able to speak about toys at test time.

We use the SHAPES dataset 4 to create two training splits, each having the same attributes but covering disjoint sets of values.

For example, the first training split (train-1) might have colorP{blue, red, yellow} whereas the second training split (train-2) has colorP{green, orange, purple}. We use train-1 to train Alice as speaker and Bob as listener and train-2 to train them in the reverse roles.

We then report performance with Alice as listener and Bob as speaker using a test set that uses the same attribute values as train-1.

Our results for both training conditions are shown in Table 4 .

The baseline model (which includes no internal-consistency constraints) performs, unsurprisingly, at chance.

Adding the self-play objective gives improvements across the board.

Again, while seemingly straight forward, this result has promising practical interpretations in settings in which a model has access to only a small amount of interaction.

For example, a human may be willing to train a robot via speaking (pointing and naming items), but not patient enough to train it via listening (responding to the robot's noisy commands).

In such a setting, the ability to massively augment performance via self-play is significant.

In addition to the self-play objective, we see that enforcing shared-embedding spaces yields further significant performance gains (in the range of 30 percentage points in some cases).

The symmetric constraints on top of self-play and shared embeddings seem to hurt performance in general.

Baseline`Self-play`Shared Emb.`Symmetric Table 4 :

Performance for tasks that requires agents to generalize across roles-e.g.

training on the word "blue" as a listener, but then having to produce "blue" as a speaker.

"

One Role" refers to when agents receive direct feedback in a single role (i.e. their training on the other roles is only via self-play).

"

Both Roles" refers to when agents receive direct feedback in both roles, but only see the test vocabulary in the role opposite that in which they are tested.

To inspect the additive differences between the internal-consistency constraints, each delta compares the performance of the current column to the previous column.

These scores are mean accuracy with 95% confidence range averaged over 5 with different test set samplings (the distractors change).

Overall, when agents can be trained directly in the role in which they are tested, there is no clear evidence that adding internal-consistency constraints improves the ability of agents to generalize to new items.

However, internal-consistency constraints improve performance significantly when agents have limited ability to train in a given role.

Specifically, models which are equipped with selfplay training objectives and shared embedding spaces show superior ability to generalize learned representations across roles, and perform about as well as if they had been trained on the target role.

In this section we provide additional analyses to highlight the effects of internal-consistency (in particular, self-play) on training efficiency and on the emerged protocol.

Here, we use a smaller SHAPES dataset (see B.1), and reduce the vocabulary size and message length (|V| " 10, L " 3).

Here we inspect if self-play supplants direct supervision between agents.

We consider the setting in which Alice trains with full data as a speaker, but vary the amount of data she has access to as a listener.

We then test Alice's performance as a listener (and vice-versa for Bob as a speaker).

Fig. 2 shows the results, with fraction of the full training data that Alice (Bob) sees as a speaker (listener) shown along the x-axis.

The self-play models without direct supervision perform well: it appears their protocol transfers across roles with out drifting.

This sheds some light on the performance drop between the "one role" and "two role" settings in Section 5.2.2.

where the additional experience in the "two role" setting did not help.

Fig. 2 shows that additional training in the primary role is unnecessary, so it is not surprising that training on disjoint features (train-2) is helpful.

We measure whether self-play leads to better communication protocols in general.

First, we measure the agents' speaking and listening capacities separately, using measures proposed by Lowe et al. (2019) ; Eccles et al. (2019) .

Specifically, positive signaling (S`) measures if the speaker's messages depend on the features of the referent, and positive listening (L`) measures if the listener's actions depend on the message 6 .

Table 5 shows that self-play improves the agents' communication as measured by accuracy as well as these orthogonal metrics.

We also find that the model architectures and self-play impact the agents' lexicons.

The recurrent models produces fewer unique messages than the transformer models (on average 110 versus 300), and often neglect to use all the vocabulary.

Fig.  3 shows that self-play helps the recurrent model use more of the vocabulary, and leads to both the recurrent and transformer models to develop sparser mappings from symbols onto features.

Work in emergent communication (Das et al., 2017; analyzes agents that develop a shared protocol by playing reference games (Lewis, 2008) .

presented results showing that computational models do not learn compositional protocols by default.

Instead, the agents tend to develop brittle protocols that have trouble generalizing to novel items.

Several approaches have been proposed which could encourage models to learn more generalizable representations of language, including compression (Kirby et al., 2015) , efficiency (Gibson et al., 2019) , memory constraints , pragmatic constraints (Tomlin & Pavlick, 2018) , and positive biases to respond to other agents (Jaques et al., 2018; Eccles et al., 2019) .

Some work, like ours, assumes access to symbolic representations of referents and their attributes, whereas others' are set in pixel-based multi-agent games (Jaques et al., 2018; Eccles et al., 2019; Das et al., 2018) or multi-agent grid-worlds (Sukhbaatar et al., 2016; Lowe et al., 2019) .

Our work also relates to a broader body of work on speaker-listener models, specifically pragmatically-informed models in which speakers reason recursively about listeners' beliefs (and vice-versa) (Frank & Goodman, 2012; Goodman & Frank, 2016) .

Such models have been used in applications such image captioning (Andreas & Klein, 2016; Yu et al., 2017; Monroe & Potts, 2015) , and robotics (Vogel & Jurafsky, 2010; Vogel et al., 2013; Fried et al., 2018) , as well as in linguistics and psychology in order to explain complex linguistic inferences (Tessler & Goodman, 2016; Monroe et al., 2017) .

Conceptually, our proposed internal-consistency constraints share something in common with these neural speaker-listener models developed outside of emergent communication.

However, again, past work has tended to assume that a speaker's mental model of their listener is not necessarily consistent-in fact, it is often assumed explicitly to be inconsistent (Frank & Goodman, 2012 )-with the way the speaker themself would behave as a listener.

We note, however, that our proposed model architecture (because it lacks the recursion typical in other pragmatics models) is likely unable to handle the types of higher-level inferences (e.g. implicatures) targeted by the mentioned prior work on computational pragmatics, though this is an interesting avenue to explore.

We propose three methods for encouraging dialog agents to follow "the golden rule": speak/listen to others as you would expect to be spoken/listened to.

In the emergent communication setting, we find that the internal-consistency constraints do not systematically improve models' generalization to novel items, but both the self-play objective and shared embeddings significantly improve performance when agents are tested on roles they were not directly trained for.

In fact, when trained in one role and tested on another, these internal-consistency constraints allow the agents to perform about as well as if they had been trained in the target role.

We use the deep learning framework Pytorch 7 (v1.2.0) to implement our models (and Python 3.7.3).

For reproducibility, all random seeds (random, numpy, torch) are arbitrarily set to 42.

The general architecture, the four modules that comprise each agent are shown in Figure 4 .

The recurrent model decodes and encodes message as follows: to generate a message, the first input is the embedding of a SOS start-of-sentence symbol and the initial hidden state is set as the embedded referent (and the cell memory is all zeroes).

From here, at each step, the outputted hidden state (P R D ) is projected by the transposed word embeddings (E ⊺ message P R Dˆ|V | ), and the next word is sampled from this resulting distribution across the vocabulary.

Moving forward, the next input is the embedding of the sampled word, and the hidden state and cell memory are set those emitted at the previous step.

We produce words until the maximum length is reached.

When encoding a message, a learned embedding is set to the first hidden state, and the input at each time step is the corresponding embedded word.

The last hidden state is set as the encoding.

In the symmetric variant of this model, the LSTM cell used for encoding and decoding is the same.

This architecture underlies each model we use; only the implementation of the Decoder and Encoder modules vary between models.

In the baseline models no parameters shared within an agent.

In shared embedding models, the embeddings (purple) are shared across roles.

In symmetric models, the encoder and decoder (pink) are shared across both roles.

The blue modules are non-parametric.

The transformer model decodes and encodes message as follows: to generate a message describing the referent auto-regressively, all the embeddings of the words produced so far M and the embedding of a NEXT symbol are concatenated together into a matrix X (Eq. 2).

Next, the transformer decoder consumes this matrix and the referent embedding and produces a contextualized representation of the input matrix (Eq. 3).

The last column vectorX :,W , which corresponds to the NEXT embedding, is the internal representation of the next word.

The next word m is sampled from the projection of this representation with the transposed word embedding.

More words are produced in this way until the maximum length message is formed.

Producing the i`1th word (so M consists of the first i words), works as follows:

To encode incoming messages when listening, all the embeddings of the words in the message plus the embedding of a ITEM symbol are concatenated together into a matrix X (Eq. 5).

Then, the transformer encoder is used to produce a contextualized embeddingX (Eq. 6).

The last column vectorX :,W , which corresponds to the ITEM embedding, is set as the message encoding.

Note, W is the number of words in each message (and the length of M ).

X " TransformerEncoderpXq P R pW`1qˆD .

r "X :,W

A.3 SYMMETRIC AGENTS For the Symmetric Recurrent Model, a LSTM cell is shared between the encoder and decoder.

Otherwise, the recurrent model is unchanged.

For the Symmetric Transformer Model, Transformer Encoders and Transformer Decoders have different structures, so to share parameters between them, we have change either how the transformer agent speaks or how it listens.

We opt to replace the Transformer Decoder with Transformer Encoder, and use it to decode messages in-place when speaking.

The Symmetric Transformer uses the same mechanism for encoding messages when listening as the default transformer model (described directly above).

However, it uses a Transformer Encoder when speaking instead of a Transformer Decode.

When speaking, to produce the next symbol, the embeddings of the i words produced so far, the referent embedding, and the embedding of a NEXT symbol are concatenated together into a matrix X (Eq. 8).

The Transformer Encoder then maps X to a contextualized representationX (Eq. 9).

Finally, the column vector in X that corresponds to NEXT, is used to sample the next symbol:

X " rE item prq; E message pM q; E ITEM s,

A.4 HYPER PARAMETERS We uniformly sampled 25 hyperparameter configurations for each model architecture, experiment, and dataset split.

In every case, we fixed the hidden size dimensionality and embedding dimensionality to be the same.

We searched over three different learning rate schedulers: (1) None (or no scheduler, (2) ReduceLROnPlateau with a patience of 25, reduction factor of 0.1, and uses validation accuracy as its measure of progress, and (3) CyclicLR rising from 0.00001 to the given learning rate over 500 batches and then declines towards 0.00001 for the rest of training (10,000 batches).

This is similar to the Noam Update in Vaswani et al. (2017) .

To save space, we relegate the hyper-parameter selections in our code bit.ly/internal-consistency-emergent-communicationsee /lib/hyperparamters.py.

TRE takes two important hyperparameters, an error function and a composition function.

We select the same choices as the author, for what amounts to the same task (producing a discrete message) as detailed in the original paper.

This method requires structured feature representations, so we assume that the features in each item are entirely right-branching.

The composition function is learned, and set the number of update steps to 1000.

The original implementation is at https://github.com/jacobandreas/tre, and our modification is at bit.ly/internal-consistency-emergent-communication in the file ./lib/compositionality.py.

We simplify the SHAPES dataset in order to be able to empirically compute the positive listening and signaling scores, which requires iterating overall possible messages.

In the original version of SHAPES this is impractical as there would be 50 10 possible messages.

The details of this smaller version of SHAPES detailed in Table 9 .

We fix the settings for the recurrent and transformer models as we found that a majority of models across experiments used the same parameters.

See Tables 10, 11 .

Furthermore, all results in Sec. 6 are averaged over 5 arbitrary random seeds (both trained and tested) (43, 44, 45, 46, 46) .

Positive listening was introduced by Lowe et al. (2019) as Causal Influence of Communication, and the precursor to positive signaling was introduced by Jaques et al. (2018) .

We use the definitions Eccles et al. (2019) modified for a one step game for both metrics:

Positive Listening (L`)

.

" D KL pP rpa|mq P r l paqq, Positive Speaking (S`) . " Ipm, xq " Hpmq´Hpm|xq, where m is a given message, a are the actions the agent can take, x is state, D KL is the Kuller-bach divergence, I is the mutual information, H is the entropy.

We can compute these quantities without sampling messages because the number of messages is tractable.

In our setting, the game is a single step, and the average policy over actions independent of the message converges to the uniform distribution over actions, as the order of referents is randomized.

Thus we have: L`" D KL pP rpa|mq U q.

The positive speaking metric is computed over a sampling of the dataset (the distractors are random) as follows as in (Eccles et al., 2019) :

S`" Hpmq´Hpm|xq, "´ÿ m πpmq log πpmq`E x r ÿ m πpm|xq log πpm|xqs, where the summations are over all possible messages, x is the given referent, and πpmq is empirically likelihood of the message being produced irrespective of x, and πpm|xq is the likelihood of the given message being produced given the referent x.

We report empirical averages of L`and Sò ver all items in the dataset, also averaged over 5 arbitrary random seeds.

Figs. 5, 6, 7, 8 show lexicons across random seeds.

See Table 8 for additional results in the vanilla setting for SHAPES-small that were elided for space.

@highlight

Internal-consistency constraints improve agents ability to develop emergent protocols that generalize across communicative roles.