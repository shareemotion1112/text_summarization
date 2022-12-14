Many machine learning algorithms represent input data with vector embeddings or discrete codes.

When inputs exhibit compositional structure (e.g. objects built from parts or procedures from subroutines), it is natural to ask whether this compositional structure is reflected in the the inputs’ learned representations.

While the assessment of compositionality in languages has received significant attention in linguistics and adjacent fields, the machine learning literature lacks general-purpose tools for producing graded measurements of compositional structure in more general (e.g. vector-valued) representation spaces.

We describe a procedure for evaluating compositionality by measuring how well the true representation-producing model can be approximated by a model that explicitly composes a collection of inferred representational primitives.

We use the procedure to provide formal and empirical characterizations of compositional structure in a variety of settings, exploring the relationship between compositionality and learning dynamics, human judgments, representational similarity, and generalization.

Figure 1: Representations arising from a communication game.

In this game, an observation (b) is presented to a learned speaker model (c), which encodes it as a discrete character sequence (d) to be consumed by a listener model for some downstream task.

The space of inputs has known compositional structure (a).

We want to measure the extent to which this structure is reflected (perhaps imperfectly) in the structure of the learned codes.

The success of modern representation learning techniques has been accompanied by an interest in understanding the structure of learned representations.

One feature shared by many humandesigned representation systems is compositionality: the capacity to represent complex concepts (from objects to procedures to beliefs) by combining simple parts BID18 .

While many machine learning approaches make use of human-designed compositional analyses for representation and prediction BID44 BID16 , it is also natural to ask whether (and how) compositionality arises in learning problems where compositional structure has not been built in from the start.

Consider the example in Figure 1 , which shows a hypothetical character-based encoding scheme learned for a simple communication task (similar to the one studied by Lazaridou et al., 2016) .

Is this encoding scheme compositional?

That is, to what extent can we analyze the agents' messages as being built from smaller pieces (e.g. pieces xx meaning blue and bb meaning triangle)?A large body of work, from early experiments on language evolution to recent deep learning models BID24 Lazaridou et al., 2017) , aims to answer questions like this one.

But existing solutions rely on manual (and often subjective) analysis of model outputs BID32 , or at best automated procedures tailored to the specifics of individual problem domains BID7 .

They are difficult to compare and difficult to apply systematically.

We are left with a need for a standard, formal, automatable and quantitative technique for evaluating claims about compositional structure in learned representations.

The present work aims at first steps toward meeting that need.

We focus on an oracle setting where the compositional structure of model inputs is known, and where the only question is whether this structure is reflected in model outputs.

This oracle evaluation paradigm covers most of the existing representation learning problems in which compositionality has been studied.

The first contribution of this paper is a simple formal framework for measuring how well a collection of representations (discrete-or continuous-valued) reflects an oracle compositional analysis of model inputs.

We propose an evaluation metric called TRE, which provides graded judgments of compositionality for a given set of (input, representation) pairs.

The core of our proposal is to treat a set of primitive meaning representations as hidden, and optimize over them to find an explicitly compositional model that approximates the true model as well as possible.

For example, if the compositional structure that describes an object is a simple conjunction of attributes, we can search for a collection of "attribute vectors" that sum together to produce the observed object representations; if it is a sparse combination of (attribute, value) pairs we can additionally search for "value vectors" and parameters of a binding operation; and so on for more complex compositions.

Having developed a tool for assessing the compositionality of representations, the second contribution of this paper is a survey of applications.

We present experiments and analyses aimed at answering four questions about the relationship between compositionality and learning:• How does compositionality of representations evolve in relation to other measurable model properties over the course of the learning process? (Section 4) •

How well does compositionality of representations track human judgments about the compositionality of model inputs? (Section 5) •

How does compositionality constrain distances between representations, and how does TRE relate to other methods that analyze representations based on similarity? (Section 6) • Are compositional representations necessary for generalization to out-of-distribution inputs?(Section 7)We conclude with a discussion of possible applications and generalizations of TRE-based analysis.

Arguments about whether distributed (and other non-symbolic) representations could model compositional phenomena were a staple of 1980s-era connectionist-classicist debates.

BID42 provides an overview of this discussion and its relation to learnability, as well as a concrete implementation of a compositional encoding scheme with distributed representations.

Since then, numerous other approaches for compositional representation learning have been proposed, with BID30 BID43 and without BID15 BID22 ) the scaffolding of explicit composition operations built into the model.

The main experimental question is thus when and how compositionality arises "from scratch" in the latter class of models.

In order to answer this question it is first necessary to determine whether compositional structure is present at all.

Most existing proposals come from linguistics and and philosophy, and offer evaluations of compositionality targeted at analysis of formal and natural languages BID8 BID28 .

Techniques from this literature are specialized to the details of linguistic representations-particularly the algebraic structure of grammars BID31 .

It is not straightforward to apply these techniques in more general settings, particularly those featuring non-string-valued representation spaces.

We are not aware of existing work that describes a procedure suitable for answering questions about compositionality in the general case.

Machine learning research has responded to this absence in several ways.

One class of evaluations BID32 BID11 derives judgments from ad-hoc manual analyses of representation spaces.

These analyses provide insight into the organization of representations but are time-consuming and non-reproducible.

Another class of evaluations BID6 BID0 BID4 ) exploits task-specific structure (e.g. the ability to elicit pairs of representations known to feature particular relationships) to give evidence of compositionality.

Our work aims to provide a standard and scalable alternative to these model-and task-specific evaluations.

Other authors refrain from measuring compositionality directly, and instead base analysis on measurement of related phenomena, for which more standardized evaluations exist.

Examples include correlation between representation similarity and similarity of oracle compositional analyses BID7 and generalization to structurally novel inputs BID26 .

Our approach makes it possible to examine the circumstances under which these surrogate measures in fact track stricter notions of compositionality; similarity is discussed in Sec. 6 and generalization in Sec. 7.A long line of work in natural language processing BID13 BID2 BID12 BID19 focuses on learning composition functions to produce distributed representations of phrases and sentences-that is, for purposes of modeling rather than evaluation.

We use one experiment from this literature to validate our own approach (Section 5).

On the whole, we view work on compositional representation learning in NLP as complementary to the framework presented here: our approach is agnostic to the particular choice of composition function, and the aforementioned references provide well-motivated choices suitable for evaluating data from language and other sources.

Indeed, one view of the present work is simply as a demonstration that we can take existing NLP techniques for compositional representation learning, fit them to representations produced by other models (even in non-linguistic settings), and view the resulting training loss as a measure of the compositionality of the representation system in question.

Consider again the communication task depicted in Figure 1 .

Here, a speaker model observes a target object described by a feature vector.

The speaker sends a message to a listener model, which uses the message to complete a downstream task-for example, identifying the referent from a collection of distractors based on the content of the message BID17 Lazaridou et al., 2017) .

Messages produced by the speaker model serve as representations of input objects; we want to know if these representations are compositional.

Crucially, we may already know something about the structure of the inputs themselves.

In this example, inputs can be identified via composition of categorical shape and color attributes.

How might we determine whether this oracle analysis of input structure is reflected in the structure of representations?

This section proposes an automated procedure for answering the question.

Representations A representation learning problem is defined by a dataset X of observations x ( Figure 1b) ; a space Θ of representations θ ( Figure 1d ); and a model f : X → Θ ( Figure 1c ).

We assume that the representations produced by f are used in a larger system to accomplish some concrete task, the details of which are not important for our analysis.

Derivations The technique we propose additionally assumes we have prior knowledge about the compositional structure of inputs.

In particular, we assume that inputs can be labeled with treestructured derivations d (Figure 1a ), defined by a finite set D 0 of primitives and a binary bracketing operation ·, · , such that if DISPLAYFORM0 Compositionality In intuitive terms, the representations computed by f are compositional if each f (x) is determined by the structure of D(x).

Most discussions of compositionality, following BID31 , make this precise by defining a composition operation θ a * θ b → θ in the space of representations.

Then the model f is compositional if it is a homomorphism from inputs to representations: we require that for any x with DISPLAYFORM1 In the linguistic contexts for which this definition was originally proposed, it is straightforward to apply.

Inputs x are natural language strings.

Their associated derivations D(x) are syntax trees, and composition of derivations is syntactic composition.

Representations θ are logical representations of meaning (for an overview see BID47 .

To argue that a particular fragment of language is compositional, it is sufficient to exhibit a lexicon D 0 mapping words to their associated meaning representations, and a grammar for composing meanings where licensed by derivations.

Algorithms for learning grammars and lexicons from data are a mainstay of semantic parsing approaches to language understanding problems like question answering and instruction following BID49 BID9 BID1 .But for questions of compositionality involving more general representation spaces and more general analyses, the above definition presents two difficulties: (1) In the absence of a clearly-defined syntax of the kind available in natural language, how do we identify lexicon entries: the primitive parts from which representations are constructed?

(2) What do we do with languages like the one in Figure 1d , which seem to exhibit some kind of regular structure, but for which the homomorphism condition given in Equation 1 cannot be made to hold exactly?Consider again the example in Figure 1 .

The oracle derivations tell us to identify primitive representations for dark, blue, green, square, and triangle.

The derivations then suggest a process for composing these primitives (e.g. via string concatenation) to produce full representations.

The speaker model is compositional (in the sense of Equation 1) as long as there is some assignment of representations to primitives such that for each model input, composing primitive representations according to the oracle derivation reproduces the speaker's prediction.

In Figure 1 there is no assignment of strings to primitives that reproduces model predictions exactly.

But predictions can be reproduced approximately-by taking xx to mean blue, aa to mean square, etc.

The quality of the approximation serves as a measure of the compositionality of the true predictor: predictors that are mostly compositional but for a few exceptions, or compositional but for the addition of some noise, will be well-approximated on average, while arbitrary mappings from inputs to representations will not.

This suggests that we should measure compositionality by searching for representations that allow an explicitly compositional model to approximate the true f as closely as possible.

We define our evaluation procedure as follows:Tree Reconstruction Error (TRE)First choose : DISPLAYFORM2 , a compositional approximation to f with parameters η, as: DISPLAYFORM3 f η has one parameter vector η i for every d i in D 0 ; these vectors are members of the representation space Θ.Given a dataset X of inputs x i with derivations d i = D(x i ), compute: DISPLAYFORM4 Then we can define datum-and dataset-level evaluation metrics: DISPLAYFORM5 DISPLAYFORM6 TRE and compositionality How well does the evaluation metric TRE(X ) capture the intuition behind Equation 1?

The definition above uses parameters η i to witness the constructability of representations from parts, in this case by explicitly optimizing over those parts rather than taking them to be given by f .

Each term in Equation 2 is analogous to an instance of Equation 1, measuring how wellf η * (x i ), the best compositional prediction, matches the true model prediction f (x i ).

In the case of models that are homomorphisms in the sense of Equation 1, TRE reduces to the familiar case: DISPLAYFORM7 Learnable composition operators The definition of TRE leaves the choice of δ and * up to the evaluator.

Indeed, if the exact form of the composition function is not known a priori, it is natural to define * with free parameters (as in e.g. BID2 , treat these as another learned part off , and optimize them jointly with the η i .

However, some care must be taken when choosing * (especially when learning it) to avoid trivial solutions:Remark 2.

Suppose D is injective; that is, every x ∈ X is assigned a unique derivation.

Then there is always some * that achieves TRE(X ) = 0: DISPLAYFORM8 as in the preceding definition, and setf = f .In other words, some pre-commitment to a restricted composition function is essentially inevitable: if we allow the evaluation procedure to select an arbitrary composition function, the result will be trivial.

This paper features experiments with * in both a fixed functional form and a learned parametric one.

Implementation details For models with continuous Θ and differentiable δ and * , TRE(X ) is also differentiable.

Equation 2 can be solved using gradient descent.

We use this strategy in Sections 4 and 5.

For discrete Θ, it may be possible to find a continuous relaxation with respect to which δ(θ, ·) and * are differentiable, and gradient descent again employed.

We use this strategy in Section 7 (discussed further there).

An implementation of an SGD-based TRE solver is provided in the accompanying software release.

For other problems, task-specific optimizers (e.g. machine translation alignment models; BID4 or general-purpose discrete optimization toolkits can be applied to Equation 2.The remainder of the paper highlights ways of using TRE to answer questions about compositionality that arise in machine learning problems of various kinds.

We begin by studying the relationship between compositionality and learning dynamics, focusing on the information bottleneck theory of representation learning proposed by BID45 .

This framework proposes that learning in deep models consists of an error minimization phase followed by a compression phase, and that compression is characterized by a decrease in the mutual information between inputs and their computed representations.

We investigate the hypothesis that the compression phase finds a compositional representation of the input distribution, isolating decision-relevant attributes and discarding irrelevant information.

FIG1 ).

We predict classifiers in a meta-learning framework BID39 BID37 : for each sub-task, the learner is presented with two images corresponding to some compositional visual concept (e.g. "digit 8 on a black background" or "green with heavy stroke") and must determine whether a held-out image is an example of the same visual concept.

Given example images x 1 and x 2 , a test image x * , and label y * , the model computes: DISPLAYFORM0 We use θ as the representation of a classifier for analysis.

The model is trained to minimize the logistic loss between logitsŷ and ground-truth labels y * .

More details are given in Appendix A.Compositional structure Visual concepts used in this task are all single attributes or conjunctions of attributes; i.e. their associated derivations are of the form attr or attr 1 , attr 2 .

Attributes include background color, digit color, digit identity and stroke type.

The composition function * is addition and the distance δ(θ, θ ) is cosine similarity 1 − θ θ /( θ θ ).Evaluation The training dataset consists of 9000 image triplets, evenly balanced between positive and negative classes, with a validation set of 500 examples.

At convergence, the model achieves validation accuracy of 75.2% on average over ten training runs. (Perfect accuracy is not possible because the true classifier is not fully determined by two training examples).

We explore the relationship between the information bottleneck and compositionality by comparing TRE(X ) to the mutual information I(θ; x) between representations and inputs over the course of training.

Both quantities are computed on the validation set, calculating TRE(X ) as described in Section 3 and I(θ; X) as described in BID41 .

(For discussion of limitations of this approach to computing mutual information between inputs and representations, see Saxe et al., 2018.)

FIG2 shows the relationship between TRE(X ) and I(θ; X).

Recall that small TRE is indicative of a high degree of compositionality.

It can be seen that both mutual information and reconstruction error are initially low (because representations initially encode little about distinctions between inputs).

Both increase over the course of training, and decrease together after mutual information reaches a maximum FIG2 .

This pattern holds if we plot values from multiple training runs at the same time FIG2 ), or if we consider only the postulated compression phase FIG2 ).

These results are consistent with the hypothesis that compression in the information bottleneck framework is associated with the discovery of compositional representations.

Next we investigate a more conventional representation learning task.

High-dimensional embeddings of words and phrases are useful for many natural language processing applications BID46 , and many techniques exist to learn them from unlabeled text BID14 BID29 .

The question we wish to explore is not whether phrase vectors are compositional in aggregate, but rather how compositional individual phrase representations are.

Our hypothesis is that bigrams whose representations have low TRE are those whose meaning is essentially compositional, and well-explained by the constituent words, while bigrams with large reconstruction error will correspond to non-compositional multi-word expressions BID33 ).This task is already well-studied in the natural language processing literature BID36 , and the analysis we present differs only in the use of TRE to search for atomic representations rather than taking them to be given by pre-trained word representations.

Our goal is to validate our approach in a language processing context, and show how existing work on compositionality (and representations of natural language in particular) fit into the more general framework proposed in the current paper.

We train embeddings for words and bigrams using the CBOW objective of BID29 using the implementation provided in FastText BID5 with 100-dimensional vectors and a context size of 5.

Vectors are estimated from a 250M-word subset of the Gigaword dataset BID34 .

More details are provided in Appendix A.Compositional structure We want to know how close phrase embeddings are to the composition of their constituent word embeddings.

We define derivations for words and phrases in the natural way: single words w have primitive derivations d = w; bigrams w 1 w 2 have derivations of the form w 1 , w 2 .

The composition function is again vector addition and distance is cosine distance.

(Future work might explore learned composition functions as in e.g. BID21 , for future work.)

We compare bigram-level judgments of compositionality computed by TRE with a dataset of human judgments about noun-noun compounds BID35 .

In this dataset, humans rate bigrams as compositional on a scale from 0 to 5, with highly conventionalized phrases like gravy train assigned low scores and graduate student assigned high ones.

Results We reproduce the results of BID36 within the tree reconstruction error framework: for a given x, TRE(x) is anticorrelated with human judgments of compositionality (ρ = −0.34, p < 0.01).

Collocations rated "most compositional" by our approach (i.e. with lowest TRE) are: application form, polo shirt, research project; words rated "least compositional" are fine line, lip service, and nest egg.

The next section aims at providing a formal, rather than experimental, characterization of the relationship between TRE and another perspective on the analysis of representations with help from oracle derivations.

BID7 introduce a notion of topographic similarity, arguing that a learned representation captures relevant domain structure if distances between learned representations are correlated with distances between their associated derivations.

This can be viewed as providing a weak form of evidence for compositionality-if the distance function rewards pairs of representations that share overlapping substructure (as might be the case with e.g. string edit distance), edit distance will be expected to correlate with some notion of derivational similarity .In this section we aim to clarify the relationship between the two evaluations.

To do this we first need to equip the space of derivations described in Section 3 with a distance function.

As the derivations considered in this paper are all tree-structured, it is natural to use a simple tree edit distance BID3 for this purpose.

We claim the following: Proposition 1.

Letf =f η * be an approximation to f estimated as in Equation 2, with all TRE(x) ≤ for some .

Let ∆ be the tree edit distance (defined formally in Appendix B, Definition 2), and let δ be any distance on Θ satisfying the following properties: DISPLAYFORM0 , where 0 is the identity element for * .

DISPLAYFORM1 (This condition is satisfied by any translation-invariant metric.)Then ∆ is an approximate upper bound on δ: for any DISPLAYFORM2 In other words, representations cannot be much farther apart than the derivations that produce them.

Proof is provided in Appendix B.We emphasize that small TRE is not a sufficient condition for topographic similarity as defined by BID7 : very different derivations might be associated with the same representation (e.g. when representing arithmetic expressions by their results).

But this result does demonstrate that compositionality imposes some constraints on the inferences that can be drawn from similarity judgments between representations.

In our final set of experiments, we investigate the relationship between compositionality and generalization.

Here we focus on communication games like the one depicted in Figure 1 and in more detail in FIG3 .

As in the previous section, existing work argues for a relationship between compositionality and generalization, claiming that agents need compositional communication protocols to generalize to unseen referents BID26 BID11 .

Here we are able to evaluate this claim empirically by training a large number of agents from random initial conditions, measuring the compositional structure of the language that emerges, and seeing how this relates to their performance on both familiar and novel objects.

A speaker model observes a pair of target objects, and sends a description of the objects (as a discrete code) to a listener model.

The listener attempts to reconstruct the targets, receiving fractional reward for partially-correct predictions.

Our experiment focuses on a reference game BID20 .

Two policies are trained: a speaker and a listener.

The speaker observes pair of target objects represented with a feature vector.

The speaker then sends a message (coded as a discrete character sequence) to the listener model.

The listener observes this message and attempts to reconstruct the target objects by predicting a sequence of attribute sets.

If all objects are predicted correctly, both the speaker and the listener receive a reward of 1 (partial credit is awarded for partly-correct objects; FIG3 ).Because the communication protocol is discrete, policies are jointly trained using a policy gradient objective BID48 .

The speaker and listener are implemented with RNNs; details are provided in Appendix A.Compositional structure Every target referent consists of two objects; each object has two attributes.

The derivation associated with each communicative task thus has the tree structure attr 1a , attr 1b , attr 2a , attr 2b .

We hold out a subset of these object pairs at training time to evaluate generalization: in each training run, 1/3 of possible reference candidates are never presented to the agent at training time.

Where the previous examples involved a representation space of real embeddings, here representations are fixed-length discrete codes.

Moreover, the derivations themselves have a more complicated semantics than in Sections 4 and 5: order matters, and a commutative operation like addition cannot capture the distinction between green, square , blue, triangle and green, triangle , blue, square .

We thus need a different class of composition and distance operations.

We represent each agent message as a sequence of one-hot vectors, and take the error function δ to be the 1 distance between vectors.

The composition function has the form: DISPLAYFORM0 with free composition parameters η * = {A, B} in Equation 2.

These matrices can redistribute the tokens in θ and θ across different positions of the input string, but cannot affect the choice of the tokens themselves; this makes it possible to model non-commutative aspects of string production.

To (b) However, compositional languages also exhibit lower absolute performance (r = 0.57, p < 1e−9).

Both facts remain true even if we restrict analysis to "successful" training runs in which agents achieve a reward > 0.5 on held-out referents (r = 0.6, p < 1e−3 and r = 0.38, p < 0.05 respectively).

Figure 6: Fragment of languages resulting from two multiagent training runs.

In the first section, the left column shows the target referent, while the remaining columns show the message generated by speaker in the given training run after observing the referent.

The two languages have substantially different TRE, but induce similar listener performance (Train and Test reward).compute TRE via gradient descent, we allow the elements of D 0 to be arbitrary vectors (intuitively assigning fractional token counts to string indices) rather than restricting them to one-hot indicators.

With this change, both δ and * have subgradients and can be optimized using the same procedure as in preceding sections.

Results We train 100 speaker-listener pairs with random initial parameters and measure their performance on both training and test sets.

Our results suggest a more nuanced view of the relationship between compositionality and generalization than has been argued in the existing literature.

TRE is significantly correlated with generalization error (measured as the difference between training accuracies, FIG4 ).

However, TRE is also significantly correlated with absolute model reward ( FIG4 )-"compositional" languages more often result from poor communication strategies than successful ones.

This is largely a consequence of the fact that many languages with low TRE correspond to trivial strategies (for example, one in which the speaker sends the same message regardless of its observation) that result in poor overall performance.

Moreover, despite the correlation between TRE and generalization error, low TRE is by no means a necessary condition for good generalization.

We can use our technique to automatically mine a collection of training runs for languages that achieve good generalization performance at both low and high levels of compositionality.

Examples of such languages are shown in Figure 6 .

We have introduced a new evaluation method called TRE for generating graded judgments about compositional structure in representation learning problems where the structure of the observations is understood.

TRE infers a set of primitive meaning representations that, when composed, approximate the observed representations, then measures the quality of this approximation.

We have applied TRE-based analysis to four different problems in representation learning, relating compositionality to learning dynamics, linguistic compositionality, similarity and generalization.

Many interesting questions regarding compositionality and representation learning remain open.

The most immediate is how to generalize TRE to the setting where oracle derivations are not available; in this case Equation 2 must be solved jointly with an unsupervised grammar induction problem BID25 .

Beyond this, it is our hope that this line of research opens up two different kinds of new work: better understanding of existing machine learning models, by providing a new set of tools for understanding their representational capacity; and better understanding of problems, by better understanding the kinds of data distributions and loss functions that give rise to compositionalor non-compositional representations of observations.

Code and data for all experiments in this paper are provided at https://github.com/jacobandreas/tre.

Thanks to Daniel Fried and David Gaddy for feedback on an early draft of this paper.

The author was supported by a Facebook Graduate Fellowship at the time of writing.

Few-shot classification The CNN has the following form:Conv(out=6, kernel=5) ReLU MaxPool(kernel=2) Conv(out=16, kernel=5) ReLU MaxPool(kernel=2) Linear(out=128) ReLU Linear(out=64) ReLUThe model is trained using ADAM BID23 ) with a learning rate of .001 and a batch size of 128.

Training is ended when the model stops improving on a held-out set.

Word embeddings We train FastText BID5 on the first 250 million words of the NYT section of Gigaword BID34 .

To acquire bigram representations, we pre-process this dataset so that each occurrence of a bigram from the BID35 dataset is treated as a single word for purposes of estimating word vectors.

Communication The encoder and decoder RNNs both use gated recurrent units BID10 with embeddings and hidden states of size 256.

The size of the discrete vocabulary is set to 16 and the maximum message length to 4.

Training uses a policy gradient objective with a scalar baseline set to the running average reward; this is optimized using ADAM BID23 ) with a learning rate of .001 and a batch size of 256.

Each model is trained for 500 steps.

Models are trained by sampling from the decoder's output distribution, but greedy decoding is used to evaluate performance and produce Figure 6 .

First, some definitions:Definition 1.

The size of a derivation is given by: DISPLAYFORM0 Definition 2.

The tree edit distance between derivations is defined by: Proof.

For d ∈ D 0 this follows immediately from Condition 2 in the proposition.

For composed derivations it follows from Condition 3 taking θ k = θ = 0 and induction on |d|.

DISPLAYFORM1

@highlight

This paper proposes a simple procedure for evaluating compositional structure in learned representations, and uses the procedure to explore the role of compositionality in four learning problems.