We present a neural framework for learning associations between interrelated groups of words such as the ones found in Subject-Verb-Object (SVO) structures.

Our model induces a joint function-specific word vector space, where vectors of e.g. plausible SVO compositions lie close together.

The model retains information about word group membership even in the joint space, and can thereby effectively be applied to a number of tasks reasoning over the SVO structure.

We show the robustness and versatility of the proposed framework by reporting state-of-the-art results on the tasks of estimating selectional preference (i.e., thematic fit) and event similarity.

The results indicate that the combinations of representations learned with our task-independent model outperform task-specific architectures from prior work, while reducing the number of parameters by up to 95%.

The proposed framework is versatile and holds promise to support learning function-specific representations beyond the SVO structures.

Word representations are in ubiquitous usage across all areas of natural language processing (NLP) (Collobert et al., 2011; Chen & Manning, 2014; Melamud et al., 2016) .

Standard approaches rely on the distributional hypothesis (Harris, 1954; Schütze, 1993) and learn a single word vector space based on word co-occurrences in large text corpora (Mikolov et al., 2013b; Pennington et al., 2014; Bojanowski et al., 2017) .

This purely context-based training produces general word representations that capture the broad notion of semantic relatedness and conflate a variety of possible semantic relations into a single space (Hill et al., 2015; Schwartz et al., 2015) .

However, this mono-faceted view of meaning is a well-known deficiency in NLP applications (Faruqui, 2016; Mrkšić et al., 2017) as it fails to distinguish between fine-grained word associations.

In this work we present a novel approach to word representation learning that moves beyond the restrictive single-space assumption.

We propose to learn a joint function-specific word vector space grouped by the different roles/functions a word can take in text.

The space is trained specifically for one structure (such as SVO), 1 and the space topology is governed by the associations between the groups.

In other words, vectors for plausible combinations from two or more groups will lie close, as illustrated by Figure 1 .

For example, the verb vector study will be close to plausible subject vectors researcher or scientist and object vectors subject or art.

For words that can occur as either subject or object such as chicken, this effectively means we may obtain several vectors, one per group, e.g., one for chicken as subject and another one for chicken as object.

We achieve this through a novel multidirectional neural representation learning approach, which takes a list of N groups of words (G 1 , . . .

, G N ), factorises it into all possible "group-to-group" sub-models, and trains them jointly with an objective similar to skip-gram negative sampling used in WORD2VEC (Mikolov et al., 2013a; b, §3) .

In other words, we learn the joint function-specific word vector space by relying on sub-networks which consume one group G i on the input side and predict words from a second group G j on the output side, i, j = 1, . . .

, N ; i = j.

At the same time, all sub-network losses are tied into a single joint loss, and all groups G 1 , . . .

, G n are shared between all sub-networks.

2 1 We choose the SVO structure as there is a number of well defined tasks reasoning over it.

Future work could look at different phenomena and how to combine vectors from several function-specific spaces.

2 This can be seen as a form of multi-task learning on shared parameters (Ruder, 2017 Figure 1 : Left: Nearest neighbours in a function-specific space trained for the SVO structure.

In the Joint SVO space (bottom) we show nearest neighbors for verbs (V) from the two other subspaces (O and S).

Right: Illustration of three neighbourhoods in a function-specific space trained for the SVO structure.

The space is structured by group (i.e. S, V, and O) and optimised such that vectors for plausible SVO compositions will be close.

Note that one word can have several vectors, for example chicken can occur as subject or object (e.g., it can eat something or someone/something can eat it).

To validate the effectiveness of our multidirectional model in language applications, we focus on modeling a prominent linguistic phenomenon: a general model of who does what to whom (Gell-Mann & Ruhlen, 2011) .

In language, this event understanding information is typically uttered by the SVO structures and, according to the cognitive science literature, is well aligned with how humans process sentences (McRae et al., 1997; 1998; Grefenstette & Sadrzadeh, 2011a; ; it reflects the likely distinct storage and processing of objects (typically nouns) and actions (typically verbs) in the brain (Caramazza & Hillis, 1991; Damasio & Tranel, 1993) .

When focusing on the SVO structures, the model will produce one joint space for the three groups (S, V and O) by tying 6 sub-networks (S→V ; V →S; S→O, . . .) with shared parameters and a joint loss (i.e. there are no duplicate parameters, the model has one set of parameters for each group).

The vectors from the induced function-specific space can then be composed by standard composition functions (Milajevs et al., 2014) to yield the so-called event representations (Weber et al., 2018) , that is, representations for the full SVO structure.

The quantitative results are reported on two established test sets for the compositional event similarity task (Grefenstette & Sadrzadeh, 2011a; which requires reasoning over SVO structures: it quantifies the plausibility of the SVO combinations by scoring them against human judgments.

We report consistent gains over standard single vector spaces as well as over two recent tensor-based architectures (Tilk et al., 2016; Weber et al., 2018) which were tailored to solve the event similarity task in particular.

Furthermore, we show that our method is general and not tied to the 3-group condition.

We conduct additional experiments in a 4-group setting where indirect objects are also modeled, along with a selectional preference 3 evaluation of 2-group SV and VO relationships (Chambers & Jurafsky, 2010; Van de Cruys, 2014) , yielding the highest scores on several established benchmarks.

Representation Learning.

Standard word representation models such as skip-gram negative sampling (SGNS) (Mikolov et al., 2013b; a) , Glove (Pennington et al., 2014) , or FastText (Bojanowski et al., 2017) induce a single word embedding space capturing broad semantic relatedness (Hill et al., 2015) .

For instance, SGNS makes use of two vector spaces for this purpose, which are referred to as A w and A c .

SGNS has been shown to approximately correspond to factorising a matrix M = A w A T c , where elements in M represent the co-occurrence strengths between words and their context words (Levy & Goldberg, 2014a) .

Both matrices represent the same vocabulary: therefore, only one of them is needed in practice to represent each word.

Typically only A w is used while A c is discarded, or the two vector spaces are averaged to produce the final space.

Levy & Goldberg (2014b) used depdendency-based contexts, resulting in two separate vector spaces; however, the relation types were embedded into the vocabulary and the model was trained only in one direction.

Rei et al. (2018) described a related task-dependent neural network for mapping word embeddings into relation-specific spaces for scoring lexical entailment.

In this work, we propose a task-independent approach and extend it to work with a variable number of relations.

Neuroscience.

Theories from cognitive linguistics and neuroscience reveal that single-space representation models fail to adequately reflect the organisation of semantic concepts in the human brain (i.e., semantic memory): there seems to be no single semantic system indifferent to modalities or categories in the brain (Riddoch et al., 1988) .

Recent fMRI studies strongly support this proposition and suggest that semantic memory is in fact a widely distributed neural network (Davies et al., 2009; Huth et al., 2012; Pascual et al., 2015; Rice et al., 2015; de Heer et al., 2017) , where sub-networks might activate selectively or more strongly for a particular function such as modality-specific or category-specific semantics (such as objects/actions, abstract/concrete, animate/inanimate, animals, fruits/vegetables, colours, body parts, countries, flowers, etc.) (Warrington, 1975; Warrington & McCarthy, 1987; McCarthy & Warrington, 1988) .

This indicates a function-specific division of lower-level semantic processing.

Single-space distributional word models have been found to partially correlate to these distributed brain activity patterns Huth et al., 2012; Anderson et al., 2017) , but fail to explain the full spectrum of fine-grained word associations humans are able to make.

Our work has been partly inspired by this literature.

Compositional Distributional Semantics.

Partially motivated by similar observations, prior work frequently employs tensor-based methods for composing separate tensor spaces (Coecke et al., 2010) : there, syntactic categories are often represented by tensors of different orders based on assumptions on their relations.

One fundamental difference is made between atomic types (e.g., nouns) versus compositional types (e.g., verbs).

Atomic types are seen as standalone: their meaning is independent from other types.

On the other hand, verbs are compositional as they rely on their subjects and objects for their exact meaning.

4 The goal is then to compose constituents into a semantic representation which is independent of the underlying grammatical structure.

Therefore, a large body of prior work is concerned with finding appropriate composition functions (Grefenstette & Sadrzadeh, 2011a; b; Kartsaklis et al., 2012; Milajevs et al., 2014) to be applied on top of word representations.

Since this approach represents different syntactic structures with tensors of varying dimensions, comparing syntactic constructs is not straightforward.

This compositional approach thus struggles with transferring the learned knowledge to downstream tasks.

State-of-the-art compositional models (Tilk et al., 2016; Weber et al., 2018) combine similar tensor-based approaches with neural training, leading to task-specific compositional solutions.

While effective for a task at hand, the resulting model relies on a large number of parameters and is not robust: we observe deteriorated performance on other related compositional tasks, as shown in §5.

Modeling SVO-s is important for tasks such as compositional event similarity using all three variables, and thematic fit modeling based on SV and VO associations separately.

Traditional solutions are typically based on clustering of word co-occurrence counts from a large corpus (Baroni & Lenci, 2010; Greenberg et al., 2015a; b; Emerson & Copestake, 2016) .

More recent solutions combine neural networks with tensor-based methods.

Van de Cruys (2014) present a feedforward neural net trained to score compositions of both two and three groups with a max-margin loss.

Grefenstette & Sadrzadeh (2011a; b) ; Milajevs et al. (2014) ; Edelstein & Reichart (2016) employ tensor compositions on standard single-space word vectors.

Hashimoto & Tsuruoka (2016) discern compositional and non-compositional phrase embeddings starting from HPSG-parsed data.

Objectives.

We propose to induce function-specific vector spaces which enable a better model of associations between concepts and consequently improved event representations by encoding the relevant information directly into the parameters for each word during training.

Word vectors offer several advantages over tensors: a large reduction in parameters and fixed dimensionality across concepts.

This facilitates their reuse and transfer across different tasks.

For this reason, we find our multidirectional training to deliver good performance: the same function-specific vector space achieves state-of-the-art scores across multiple related tasks, previously held by task-specific models.

The directionality of prediction in neural models is important.

Representations can be of varying quality depending on whether they are induced at the input or output side of the model.

Our multidirectional approach resolves this problem by training on shared representations in all directions.

We require a flexible model that has a high capacity for learning associations between all groups of words and their representations.

Formally, our goal is to model the mutual associations (cooccurrences) between N groups of words, where the vocabularies of each group can partially overlap.

We induce an embedding matrix R |Vi|×d for each group i = 1, . . .

, N , where |V i | corresponds to the vocabulary size of the i-th group.

For consistency, the vector dimensionality d is kept equal across all variables.

Multiple Groups.

Without loss of generality we present a model which creates a function-specific vector space for N = 3 groups, referring to those groups as A, B, and C. Note that the model is not limited to this setup, as we show later in §5.

A, B and C might be interrelated phenomena, and we aim for a model which can reliably score the plausibility of combining three vectors ( A, B, C) taken from this space.

5 In addition to the full joint prediction, we aim for any two vector combinations ( A B, B C, C A) to have plausible scores of their own.

Observing relations between words inside single-group subspaces (A, B, or C) is another desirable feature.

Directionality.

To design a solution with the necessary properties, we first need to consider the influence of prediction directionality in representation learning.

A representation model such as SGNS (Mikolov et al., 2013a; b) learns two vectors for each word in one large vocabulary, one vector on the input side (word vector), another on the output side (context vector).

6 Here, we require several distinct vocabularies (i.e., three, one each for group A, B, and C).

Instead of context vectors, we train the model to predict words from another group, hence directionality is an important consideration.

We find that prediction directionality has a strong impact on the quality of the induced representations, and illustrate this effect on an example that is skewed extremely to one side: an n:1 assignment case.

Let us assume data of two groups, where each word of group A 1 is assigned to exactly one of three clusters in group B 3 .

We expect a function-specific word vector space customised for this purpose to show three clearly separated clusters.

Figure 2 visualises obtained representations.

7 Figure 2a plots the vector spaces when we use words on the input side of the model and predict the cluster: A 1 → B 3 ; this can be seen as n:1 assignment.

In the opposite direction (B 3 → A 1 , 1:n assignment) we do not observe the same trends (Figure 2b ).

Representations for other and more complex phenomena suffer from the same issue.

For example, the verb eat can take many arguments corresponding to various food items such as pizza, beans, or kimchi.

A more specific verb such as embark might take only a few arguments such as journey, whereas journey might be fairly general and can co-occur with many other verbs themselves.

We thus effectively deal with an n:m assignment case, which might be inclined towards 1:n or n:1 entirely depending on the words in question.

Therefore, it is unclear whether one should rather construct a model predicting verb → object or object → verb.

We resolve this fundamental design question by training representations in a multidirectional way with a joint loss function.

For all directions the model operates on the same shared parameters.

As shown in Figure 2c , we learn sensibly clustered representations without explicitly injecting directionality assumptions.

By relying on shared parameters the model also becomes more memory-efficient.

We now describe details of the architecture.

Sub-Network Architecture.

We factorise groups into sub-networks, representing all possible directions of prediction.

8 Similar to Mikolov et al. (2013a; b) , we calculate the dot-product between two word vectors to quantify their association.

For instance, the sub-network A → B computes its prediction P A→B = σ( a · B T e + b ab ), where a is a word vector from the input group A, B e is the word embedding matrix for the target group B, b ab is a bias vector, and σ is the sigmoid function.

The loss of each sub-network is computed using cross-entropy between this prediction and the correct labels L A→B = cross entropy(P A→B , L A→B ), where L A→B are one-hot vectors corresponding to the correct predictions.

We leave experiments with more sophisticated sub-network designs for future work.

Synchronous Joint Training.

We integrate all sub-networks into one joint model via two mechanisms: (1) Shared Parameters.

The three embedding matrices referring to groups A, B and C are shared across all sub-networks.

That is, we train one matrix per group, regardless of whether it is being employed at the input or the output side of any sub-network.

This leads to a substantial reduction in the model size.

9 (2) Joint Loss.

We also train all sub-networks with a single joint loss and a single backward pass.

We refer to this manner of joining the losses as synchronous: it synchronises the backward pass of all sub-networks.

It could also be seen as a form of multi-task learning, where each sub-network optimises the shared parameters for a different task.

In practice, for the synchronous loss we do a forward pass in each direction separately, then join all sub-network cross-entropy losses and backpropagate this joint loss through all sub-networks.

We rely on addition to compute the joint loss: L = µ L µ , where µ represents one of the six sub-networks, L µ is the corresponding loss, and L the overall joint loss.

Preliminary Task: Pseudo-Disambiguation.

In the first evaluation, we adopt a standard pseudodisambiguation task from the selectional preference literature (Rooth et al., 1999; Bergsma et al., 2008; Erk et al., 2010; Chambers & Jurafsky, 2010; Van de Cruys, 2014) .

For the three-group (S-V-O) case, the task is to score a true triplet (i.e., the (S-V-O) structure attested in the corpus) above all corrupted triplets (S-V'-O), (S'-V-O), (S-V-O'), where S', V' and O' denote subjects and objects randomly drawn from their respective vocabularies.

Similarly, for the two-group setting, the task is to express a higher preference towards the attested pairs (V-O) or (S-V) over corrupted pairs (V-O') or (S'-V).

We report accuracy scores, i.e., we count all items where score(true) > score(corrupted).

This simple pseudo-disambiguation task serves as a preliminary sanity check: it can be easily applied to a variety of training conditions with different variables.

However, as pointed out by Chambers & Jurafsky (2010) , the performance on this task is strongly influenced by a number of factors such as vocabulary size and the procedure for constructing corrupted examples.

Therefore, we additionally evaluate our models on a number of other established datasets .

A standard task to measure the plausibility of SVO structures (i.e., events) is event similarity (Grefenstette & Sadrzadeh, 2011a; Weber et al., 2018) : the goal is 8 Two groups will lead to two sub-networks A → B and B → A. Three groups lead to six sub-networks.

9 For example, with a vocabulary of 50, 000 words and 25-dimensional vectors (our experimental setup, see §4), we work only with 1.35M parameters.

Comparable models for the same tasks are trained with much larger sets of parameters: 26M or even up to 179M when not factorised (Tilk et al., 2016) .

Our modeling approach thus can achieve a large reduction in the number of parameters, > 95%.

Table 2 : Pseudo-disambiguation: accuracy scores.

to score similarity between SVO triplet pairs and correlate the similarity scores to human-elicited similarity judgements.

Robust and flexible event representations are important to many core areas in language understanding such as script learning, narrative generation, and discourse understanding (Chambers & Jurafsky, 2009; Pichotta & Mooney, 2016; Modi, 2016; Weber et al., 2018) .

We evaluate event similarity on two benchmarking data sets: GS199 (Grefenstette & Sadrzadeh, 2011a) and KS108 .

GS199 contains 199 pairs of SV O triplets/events.

In the GS199 data set only the V is varied, while S and O are fixed in the pair: this evaluation prevents the model from relying only on simple lexical overlap for similarity computation.

10 KS108 contains 108 event pairs for the same task, but is specifically constructed without any lexical overlap between the events in each pair.

For this task function-specific representations are composed into a single event representation/vector.

Following prior work, we compare cosine similarity of event vectors to averaged human scores and report Spearman's ρ correlation with human scores.

We compose the function-specific vectors into event vectors using simple addition and multiplication, as well as more sophisticated compositions from prior work (Milajevs et al., 2014, inter alia) .

The summary is provided in Table 3 .

Thematic-Fit Evaluation (2 Variables: SV and VO).

Similarly to the 3-group setup, we also evaluate the plausibility of SV and V O pairs separately in the 2-group setup.

The thematic-fit evaluation quantifies the extent to which a noun fulfils the selectional preference of a verb given a role (i.e., agent:S, or patient:O) (McRae et al., 1997) .

We evaluate our 2-group function-specific spaces on two standard benchmarks: 1) MST1444 (McRae et al., 1998) contains 1,444 word pairs where humans provided thematic fit ratings on a scale from 1 to 7 for each noun to score the plausibility of the noun taking the agent role, and also taking the patient role.

11 2) PADO414 (Padó, 2007 ) is similar to MST1444, containing 414 pairs with human thematic fit ratings, where role-filling nouns were selected to reflect a wide distribution of scores for each verb.

We compute plausibility by simply taking the cosine similarity between the verb vector (from the V space) and the noun vector from the appropriate function-specific space (S space for agents; O space for patients).

We again report Spearman's ρ correlation scores.

Training Data.

We parse the ukWaC corpus (Baroni et al., 2009 ) and the British National Corpus (BNC) (Leech, 1992) using the Stanford Parser with Universal Dependencies v1.4 (Chen & Manning, 2014; Nivre et al., 2016) and extract co-occurring subjects, verbs and objects.

All words are lowercased and lemmatised, and tuples containing non-alphanumeric characters are excluded.

We also remove tuples with (highly frequent) pronouns as subjects, and filter out training examples containing words with frequency lower than 50.

After preprocessing, the final training corpus comprises 22M SVO triplets in total.

Table 1 additionally shows training data statistics when training in the 2-group setup (SV and VO) and in the 4-group setup (when adding indirect objects: SVO+iO).

Reference Formula

Copy Object W2V Milajevs et al. (2014) Results on the event similarity task.

Best baseline score is underlined, and the best overall result is provided in bold.

We report the number of examples in training and test sets, as well as vocabulary sizes and most frequent words across different categories.

Hyperparameters.

We train with batch size 128, and use Adam for optimisation (Kingma & Ba, 2015) with a learning rate 0.001.

All gradients are clipped to a maximum norm of 5.0.

All models were trained with the same fixed random seed.

We train 25-dimensional vectors for all setups (2/3/4 groups), and we additionally train 100-dimensional vectors for the 3-group (SVO) setup.

Pseudo-Disambiguation.

Accuracy scores on the pseudo-disambiguation task in the 2/3/4-group setups are summarised in Table 2 .

12 We find consistently high pseudo-disambiguation scores (>0.94) across all setups.

As mentioned in §4, this initial evaluation already suggests that our model is able to capture associations between interrelated groups which are instrumental to modeling SVO structures and composing event representations.

Event Similarity.

We now test correlations of SVO-based event representations composed from a function-specific vector space (see Table 3 ) to human scores in the event similarity task.

A summary of the main results is provided in Table 3 .

We also report best baseline scores from prior work.

The main finding is that our model based on function-specific word vectors outperforms previous state-of-the-art scores on both datasets.

It is crucial to note that different modeling approaches and configurations from prior work held previous peak scores on the two evaluation sets.

13 Interestingly, by relying only on the representations from the V subspace (i.e., by completely discarding the knowledge stored in S and O vectors), we can already obtain reasonable correlation scores.

This is an indicator that the verb vectors indeed stores some selectional preference information as designed, i.e., the information is successfully encoded into the verb vectors themselves.

Thematic-Fit Evaluation.

Correlation scores on two thematic-fit evaluation data sets are summarised in Table 4 .

We also report results with representative baseline models for the task: 1) a TypeDM-based model (Baroni & Lenci, 2010) , further improved by Greenberg et al. (2015a; b) (G15) , and 2) current state-of-the-art tensor-based neural model by Tilk et al. (2016) (TK16) .

We find that vectors taken from the model trained in the joint 3-group SVO setup perform on a par with state-of-the-art models also in the 2-group evaluation on SV and VO subsets.

Vectors trained explicitly in the 2-group setup using three times more data lead to substantial improvements on PADO414.

As a general finding, our function-specific approach leads to peak performance on both data sets.

The results are similar with 25-dim SVO vectors.

Our model is also more light-weight than the baselines: we do not require a full (tensor-based) neural model, but simply function-specific word vectors to reason over thematic fit.

To further verify the importance of joint multidirectional training, we have also compared our function-specific vectors against standard single-space word vectors (Mikolov et al., 2013b) .

The results indicate the superiority of function-specific spaces: respective correlation scores on MST1444 and PADO414 are 0.28 and 0.41 (vs 0.34 and 0.58 with our model).

It is interesting to note that we obtain state-of-the-art scores calculating cosine similarity of vectors taken from two groups found in the joint space.

This finding verifies that the model does indeed learn a joint space where co-occurring words from different groups lie close to each other.

Qualitative Analysis.

We retrieve nearest neighbours from the function-specific (S, V , O) space, shown in Figure 1 .

We find that the nearest neighbours indeed reflect the relations required to model the SVO structure.

For instance, the closest subjects/agents to the verb eat are cat and dog.

The closest objects to need are three plausible nouns: help, support, and assistance.

As the model has information about group membership, we can also filter and compare nearest neighbours in single-group subspaces.

For example, we find subjects similar to the subject memory are dream and feeling, and objects similar to beer are ale and pint.

Model Variants.

We also conduct an ablation study that compares different model variants.

The variants are constructed by varying 1) the training regime: asynchronous (async) vs synchronous (sync) 14 and 2) the type of parameter sharing: training on separate parameters for each sub-network (sep) 15 or training on shared variables (shared).

Table 5 shows the results with the model variants, demonstrating that both aspects (i.e., shared parameters and synchronous training) are important to reach improved overall performance.

We reach the peak scores on all evaluation sets using the sync+shared variant.

We suspect that asynchronous training deteriorates performance because each sub-network overwrites the updates of other sub-networks as their training is not tied through a joint loss function.

On the other hand, the synchronous training regime guides the model towards making updates that can benefit all sub-networks.

We presented a novel multidirectional neural framework for learning function-specific word representations, which can be easily composed into multi-word representations to reason over event similarity and thematic fit.

We induce a joint vector space in which several groups of words (e.g., S, V, and O words forming the SVO structures) are represented while taking into account the mutual associations between the groups.

We found that resulting function-specific vectors yield state-of-the-art results on established benchmarks for the tasks of estimating event similarity and evaluating thematic fit, previously held by task-specific methods.

In future work we will investigate more sophisticated neural (sub-)networks within the proposed framework.

We will also apply the idea of function-specific training to other interrelated linguistic phenomena and other languages, probe the usefulness of function-specific vectors in other language tasks, and explore how to integrate the methodology with sequential models.

The pre-trained word vectors used in this work are available online at: [URL] .

14 In the asynchronous setup we update the shared parameters per sub-network directly based on their own loss, instead of relying on the joint synchronous loss as in §3.

15 With separate parameters we merge vectors from "duplicate" vector spaces by non-weighted averaging.

<|TLDR|>

@highlight

Task-independent neural model for learning associations between interrelated groups of words.

@highlight

The paper proposed a method for training function-specific word vectors, in which each word is represented with three vectors each in a different category (Subject-Verb-Object).

@highlight

This paper proposes a neural network to learn function-specific work representations and demonstrates the advantage over alternatives.