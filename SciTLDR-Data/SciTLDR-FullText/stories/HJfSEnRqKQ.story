While many active learning papers assume that the learner can simply ask for a label and receive it, real annotation often presents a mismatch between the form of a label (say, one among many classes), and the form of an annotation (typically yes/no binary feedback).

To annotate examples corpora for multiclass classification, we might need to ask multiple yes/no questions, exploiting a label hierarchy if one is available.

To address this more realistic setting, we propose active learning with partial feedback (ALPF), where the learner must actively choose both which example to label and which binary question to ask.

At each step, the learner selects an example, asking if it belongs to a chosen (possibly composite) class.

Each answer eliminates some classes, leaving the learner with a partial label.

The learner may then either ask more questions about the same example (until an exact label is uncovered) or move on immediately, leaving the first example partially labeled.

Active learning with partial labels requires (i) a sampling strategy to choose (example, class) pairs, and (ii) learning from partial labels between rounds.

Experiments on Tiny ImageNet demonstrate that our most effective method improves 26% (relative) in top-1 classification accuracy compared to i.i.d.

baselines and standard active learners given 30% of the annotation budget that would be required (naively) to annotate the dataset.

Moreover, ALPF-learners fully annotate TinyImageNet at 42% lower cost.

Surprisingly, we observe that accounting for per-example annotation costs can alter the conventional wisdom that active learners should solicit labels for hard examples.

Given a large set of unlabeled images, and a budget to collect annotations, how can we learn an accurate image classifier most economically?

Active Learning (AL) seeks to increase data efficiency by strategically choosing which examples to annotate.

Typically, AL treats the labeling process as atomic: every annotation costs the same and produces a correct label.

However, large-scale multi-class annotation is seldom atomic; we can't simply ask a crowd-worker to select one among 1000 classes if they aren't familiar with our ontology.

Instead, annotation pipelines typically solicit feedback through simpler mechanisms such as yes/no questions.

For example, to construct the 1000-class ImageNet dataset, researchers first filtered candidates for each class via Google Image Search, then asking crowd-workers questions like "Is there a Burmese cat in this image?" BID5 .

For tasks where the Google trick won't work, we might exploit class hierarchies to drill down to the exact label.

Costs scale with the number of questions asked.

Thus, real-world annotation costs can vary per example BID24 .We propose Active Learning with Partial Feedback (ALPF), asking, can we cut costs by actively choosing both which examples to annotate, and which questions to ask?

Say that for a new image, our current classifier places 99% of the predicted probability mass on various dog breeds.

Why start at the top of the tree -"is this an artificial object?

" -when we can cut costs by jumping straight to dog breeds ( FIG0 )?

ALPF proceeds as follows: In addition to the class labels, the learner possesses a pre-defined collection of composite classes, e.g. dog ⊃ bulldog, mastiff, ....

At each round, the learner selects an (example, class) pair.

The annotator responds with binary feedback, leaving the learner with a partial label.

If only the atomic class label remains, the learner has obtained an exact label.

For simplicity, we focus on hierarchically-organized collections-trees with atomic classes as leaves and composite classes as internal nodes.

For this to work, we need a hierarchy of concepts familiar to the annotator.

Imagine asking an annotator "is this a foo?

" where foo represents a category comprised of 500 random ImageNet classes.

Determining class membership would be onerous for the same reason that providing an exact label is: It requires the annotator be familiar with an enormous list of seemingly-unrelated options before answering.

On the other hand, answering "is this an animal?

" is easy despite animal being an extremely coarse-grained category -because most people already know what an animal is.

We use active questions in a few ways.

To start, in the simplest setup, we can select samples at random but then once each sample is selected, choose questions actively until finding the label:ML: "Is it a dog?

" Human: Yes!

ML: "Is it a poodle?

" Human: No!

ML: "Is it a hound?

" Human: Yes!

ML: "Is it a Rhodesian ?

" Human: No!

ML: "Is it a Dachsund?" Human: Yes!In ALPF, we go one step further.

Since our goal is to produce accurate classifiers on tight budget, should we necessarily label each example to completion?

After each question, ALPF learners have the option of choosing a different example for the next binary query.

Efficient learning under ALPF requires (i) good strategies for choosing (example, class) pairs, and (ii) techniques for learning from the partially-labeled data that results when labeling examples to completion isn't required.

We first demonstrate an effective scheme for learning from partial labels.

The predictive distribution is parameterized by a softmax over all classes.

On a per-example basis, we convert the multiclass problem to a binary classification problem, where the two classes correspond to the subsets of potential and eliminated classes.

We determine the total probability assigned to potential classes by summing over their softmax probabilities.

For active learning with partial feedback, we introduce several acquisition functions for soliciting partial labels, selecting questions among all (example, class) pairs.

One natural method, expected information gain (EIG) generalizes the classic maximum entropy heuristic to the ALPF setting.

Our two other heuristics, EDC and ERC, select based on the number of labels that we expect to see eliminated from and remaining in a given partial label, respectively.

We evaluate ALPF learners on CIFAR10, CIFAR100, and Tiny ImageNet datasets.

In all cases, we use WordNet to impose a hierarchy on our labels.

Each of our experiments simulates rounds of active learning, starting with a small amount of i. 2 ACTIVE LEARNING WITH PARTIAL FEEDBACK By x ∈ R d and y ∈ Y for Y = {{1}, ..., {k}}, we denote feature vectors and labels.

Here d is the feature dimension and k is the number of atomic classes.

By atomic class, we mean that they are indivisible.

As in conventional AL, the agent starts off with an unlabeled training set D = {x 1 , ..., x n }.Composite classes We also consider a pre-specified collection of composite classes C = {c 1 , ..., c m }, where each composite class c i ⊂ {1, ..., k} is a subset of labels such that |c i | ≥ 1.

Note that C includes both the atomic and composite classes.

In this paper's empirical section, we generate composite classes by imposing an existing lexical hierarchy on the class labels BID19 .

For an example i, we use partial label to describe any elementỹ i ⊂ {1, ..., k} such that y i ⊃ y i .

We callỹ i a partial label because it may rule out some classes, but doesn't fully indicate underlying atomic class.

For example, dog = {akita, beagle, bulldog, ...} is a valid partial label when the true label is {bulldog}. An ALPF learner eliminates classes, obtaining successively smaller partial labels, until only one (the exact label) remains.

To simplify notation, in this paper, by an example's partial label, we refer to the smallest partial label available based on the already-eliminated classes.

At any step t and for any example i, we useỹ (t) i to denote the current partial label.

The initial partial label for every example isỹ 0 = {1, ..., k} An exact label is achieved when the partial labelỹ i = y i .

The set of possible questions Q = X × C includes all pairs of examples and composite classes.

An ALPF learner interacts with annotators by choosing questions q ∈ Q. Informally, we pick a question q = (x i , c j ) and ask the annotator, does x i contain a c j ?

If the queried example's label belongs to the queried composite class (y i ⊂ c j ), the answer is 1, else 0.Let α q denote the binary answer to question q ∈ Q. Based on the partial feedback, we can compute the new partial labelỹ (t+1) according to Eq. equation 1, DISPLAYFORM0 Note that hereỹ (t) and c are sets, α is a bit, c is a set complement, and thatỹ (t) \ c andỹ (t) \ c are set subtractions to eliminate classes from the partial label based on the answer.

Learning Process The learning process is simple: At each round t, the learner selects a pair (x, c) for labeling.

Note that a rational agent will never select either (i) an example for which the exact label is known, or (ii) a pair (x, c) for which the answer is already known, e.g., if c ⊃ỹ (t) or c ∩ỹ (t) = ∅. After receiving binary feedback, the agent updates the corresponding partial labelỹ (t) →ỹ (t+1) , using Equation 1.

The agent then re-estimates its model, using all available non-trivial partial labels and selects another question q. In batch-mode, the ALPF learner re-estimates its model once per T queries which is necessary when training is expensive (e.g. deep learning).

We summarize the workflow of a ALPF learner in Algorithm 1.Objectives We state two goals for ALPF learners.

First, we want to learn predictors with low error (on exactly labeled i.i.d.

holdout data), given a fixed annotation budget.

Second, we want to fully annotate datasets at the lowest cost.

In our experiments (Section 3), a ALPF strategy dominates on both tasks.

Table 1 : Learning from partial labels on Tiny ImageNet.

These results demonstrate the usefulness of our training scheme absent the additional complications due to ALPF.

In each row, γ% of examples are assigned labels at the atomic class (Level 0).

Levels 1, 2, and 4 denote progressively coarser composite labels tracing through the WordNet hierarchy.

We now address the task of learning a multiclass classifier from partial labels, a fundamental requirement of ALPF, regardless of the choice of sampling strategy.

At time t, our modelŷ(y, x, θ (t) ) parameterised by parameters θ (t) estimates the conditional probability of an atomic class y. For simplicity, when the context is clear, we will useŷ to designate the full vector of predicted probabilities over all classes.

The probability assigned to a partial labelỹ can be expressed by marginalizing over the atomic classes that it contains:

.

We optimize our model by minimizing the log loss: DISPLAYFORM0 Note that when every example is exactly labeled, our loss function simplifies to the standard cross entropy loss often used for multi-class classification.

Also note that when every partial label contains the full set of classes, all partial labels have probability 1 and the update is a no-op.

Finally, if the partial label indicates a composite class such as dog, and the predictive probability mass is exclusively allocated among various breeds of dog, our loss will be 0.

Models are only updated when their predictions disagree (to some degree) with the current partial label.

Expected Information Gain (EIG):

Per classic uncertainty sampling, we can quantify a classifer's uncertainty via the entropy of the predictive distribution.

In AL, each query returns an exact label, and thus the post-query entropy is always 0.

In our case, each answer to the query yields a different partial label.

We use the notationŷ 0 , andŷ 1 to denote consequent predictive distributions for each answer (no or yes).

We generalize maximum entropy to ALPF by selecting questions with greatest expected reduction in entropy.

DISPLAYFORM0 where S(·) is the entropy function.

It's easy to prove that EIG is maximized whenp(c, x, θ) = 0.5.Expected Remaining Classes (ERC): Next, we propose ERC, a heuristic that suggests arriving as quickly as possible at exactly-labeled examples.

At each round, ERC selects those examples for which the expected number of remaining classes is fewest: DISPLAYFORM1 where ||ŷ α || is the size of the partial label following given answer α.

ERC is minimized when the result of the feedback will produce an exact label with probability 1.

For a given example x i , if ||ŷ i || 0 = 2 containing only the potential classes (e.g.) dog and cat, then with certainty, ERC will produce an exact label by querying the class {dog} (or equivalently {cat}).

This heuristic is inspired by BID2 , which shows that the partial classification loss (what we optimize with partial labels) is an upper bound of the true classification loss (as if true labels are available) with a linear factor of 1 1−ε , where ε is ambiguity degree and ε ∝ |ỹ|.

By selecting q ∈ Q that leads to the smallest |ỹ|, we can tighten the bound to make optimization with partial labels more effective.

Expected Decrease in Classes (EDC): More in keeping with the traditional goal of minimizing uncertainty, we might choose EDC, the sampling strategy which we expect to result in the greatest reduction in the number of potential classes.

We can express EDC as the difference between the number of potential labels (known) and the expected number of potential labels remaining: DISPLAYFORM2

We evaluate ALPF algorithms on the CIFAR10, CIFAR100, and Tiny ImageNet datasets, with training sets of 50k, 50k, and 100k examples, and 10, 100, and 200 classes respectively.

After imposing the Wordnet hierarchy on the label names, the size of the set of possible binary questions |C| for each dataset are 27, 261, and 304, respectively.

The number of binary questions between re-trainings are 5k, 15k, and 30k, respectively.

By default, we warm-start each learner with the same 5% of training examples selected i.i.d. and exactly labeled.

Warm-starting has proven essential in other papers combining deep and active learning BID26 .

Our own analysis (Section 3.3) confirms the importance of warm-starting although the affect appears variable across acquisition strategies.

Model For each experiment, we adopt the widely-popular ResNet-18 architecture BID11 .

Because we are focused on active learning and thus seek fundamental understanding of this new problem formulation, we do not complicate the picture with any fine-tuning techniques.

Note that some leaderboard scores circulating on the Internet appear to have far superior numbers.

This owes to pre-training on the full ImageNet dataset (from which Tiny-ImageNet was subsampled and downsampled), constituting a target leak.

We initialize weights with the Xavier technique BID9 ) and minimize our loss using the Adam BID16 optimizer, finding that it outperforms SGD significantly when learning from partial labels.

We use the same learning rate of 0.001 for all experiments, first-order momentum decay (β 1 ) of 0.9, and second-order momentum decay (β 2 ) of 0.999.

Finally, we train with mini-batches of 200 examples and perform standard data augmentation techniques including random cropping, resizing, and mirror-flipping.

We implement all models in MXNet and have posted our code publicly 1 .Re-training Ideally, we might update models after each query, but this is too costly.

Instead, following BID26 and others, we alternately query labels and update our models in rounds.

We warm-start all experiments with 5% labeled data and iterate until every example is exactly labeled.

At each round, we re-train our classifier from scratch with random initialization.

While we could initialize the new classifier with the previous best one (as in BID26 ), preliminary experiments showed that this faster convergence comes at the cost of worse performance, perhaps owing to severe over-fitting to labels acquired early in training.

In all experiments, for simplicity, we terminate the optimization after 75 epochs.

Since 30k questions per re-training (for TinyImagenet) seems infrequent, we compared against 10x more frequent re-training More frequent training conferred no benefit (Appendix B).

Since the success of ALPF depends in part on learning from partial labels, we first demonstrate the efficacy of learning from partial labels with our loss function when the partial labels are given a priori.

In these experiments we simulate a partially labeled dataset and show that the learner achieves significantly better accuracy when learning from partial labels than if it excluded the partial labels and focused only on exactly annotated examples.

Using our WordNet-derived hierarchy, we conduct experiments with partial labels at different levels of granularity.

Using partial labels from one level above the leaf, German shepherd becomes dog.

Going up two levels, it becomes animal.

We first train a standard multi-class classifier with γ (%) exactly labeled training data and then another classifier with the remaining (1 − γ)% partially labeled at a different granularity (level of hierarchy).

We compare the classifier performance on holdout data both with and without adding partial labels in Table 1 .

We make two key observations: (i) additional coarse-grained partial labels improve model accuracy (ii) as expected, the improvement diminishes as partial label gets coarser.

These observations suggest we can learn effectively given a mix of exact and partial labels.

Baseline This learner samples examples at random.

Once an example is sampled, the learner applies topdown binary splitting-choosing the question that most evenly splits the probability mass, see Related Work for details-with a uniform prior over the classes until that example is exactly labeled.

AL To disentangle the effect of active sampling of questions and samples, we compare to conventional AL approaches selecting examples with uncertainty sampling but selecting questions as baseline.

AQ Active questions learners, choose examples at random but use partial feedback strategies to efficiently label those examples, moving on to the next example after finding an example's exact label.

ALPF ALPF learners are free to choose any (example, question) pair at each turn, Thus, unlike AL and AQ, ALPF learners commonly encounter partial labels during training.

Results We run all experiments until fully annotating the training set.

We then evaluate each method from two perspectives: classification and annotation.

We measure each classifiers' top-1 accuracy at each annotation budget.

To quantify annotation performance, we count the number questions required to exactly label all training examples.

We compile our results in Table 2 , rounding costs to 10%, 20% etc.

The budget includes the (5%) i.i.d.

data for warm-starting.

Some key results: (i) vanilla active learning does not improve over i.i.d.

baselines, confirming similar observations on image classification by BID22 ; (ii) AQ provides a dramatic improvement over baseline.

The advantage persists throughout training.

These learners sample examples randomly and label to completion (until an exact label is produced) before moving on, differing only in how efficiently they annotate data. (iii) On Tiny ImageNet, at 30% of budget, ALPF-ERC outperforms AQ methods by 4.5% and outperforms the i.i.d.

baseline by 8.1%.

First, we study how different amounts of warm-starting affects ALPF learners' performance with a small set of i.i.d.

labels.

Second, we compare the selections due to ERC and EDC to those produced through uncertainty sampling.

Third, we note that while EDC and ERC appear to perform best on our problems, they may be vulnerable to excessively focusing on classes that are trivial to recognize.

We examine this setting via an adversarial dataset intended to break the heuristics.

We compare the performance of each strategy under different percentages (0%, 5%, and 10%) of pre-labeled i.i.d.

data ( FIG4 , Appendix A).

Results show that ERC works properly even without warm-starting, while EIG benefits from a 5% warm-start and EDC suffers badly without warm-starting.

We observe that 10% warm-starting yields no further improvement.

Sample uncertainty Classic uncertainty sampling chooses data of high uncertainty.

This question is worth re-examining in the context of ALPF.

To analyze the behavior of ALPF learners vis-a-vis uncertainty we plot average prediction entropy of sampled data for ALPF learners with different sampling strategies (Figure 3) .

Note that ALPF learners using EIG pick high-entropy data, while ALPF learners with EDC and ERC choose examples with lower entropy predictions.

The (perhaps) surprising performance of EDC and ERC may owe to the cost structure of ALPF.

While labels for examples with low-entropy predictions confer less information, they also come at lower cost.

Adversarial setting Because ERC goes after "easy" examples, we test its behavior on a simulated dataset where 2 of the CIFAR10 classes (randomly chosen) are trivially easy.

We set all pixels white for one class all pixels black for the other.

We plot the label distribution among the selected data over rounds of selection in against that on the unperturbed CIFAR10 in Figure 4 .

As we can see, in the normal case, EIG splits its budget among all classes roughly evenly while EDC and ERC focus more on different classes at different stages.

In the adversarial case, EIG quickly learns the easy classes, thereafter focusing on the others until they are exhausted, while EDC and ERC concentrate on exhausting the easy ones first.

Although EDC and ERC still manage to label all data with less total cost than EIG, this behavior might cost us when we have trivial classes, especially when our unlabeled dataset is enormous relative to our budget.

Binary identification: Efficiently finding answers with yes/no questions is a classic problem BID7 dubbed binary identification.

BID12 proved that finding the optimal strategy given an arbitrary set of binary tests is NP-complete.

A well-known greedy algorithm called binary splitting BID8 BID17 , chooses questions that most evenly split the probability mass.

Active learning: Our work builds upon the AL framework BID0 BID1 Settles, 2010) (vs. i.i.d labeling) .

Classical AL methods select examples for which the current predictor is most uncertain, according to various notions of uncertainty: BID4 selects examples with maximum entropy (ME) predictive distributions, while BID3 uses the least confidence (LC) heuristic, sorting examples in ascending order by the probability assigned to the argmax.

BID25 notes that annotation costs may vary across data points suggesting cost-aware sampling heuristics but doesn't address the setting when costs change dynamically during training as a classifier grows stronger.

BID18 incorporates structure among outputs into an active learning scheme in the context of structured prediction.

BID20 addresses hierarchical label structure in active learning interestingly in a setting where subclasses are easier to learn.

Thus they query classes more fine-grained than the targets, while we solicit feedback on more general categories.

Deep Active Learning Deep Active Learning (DAL) has recently emerged as an active research area.

BID27 explores a scheme that combines traditional heuristics with pseudo-labeling.

BID6 notes that the softmax outputs of neural networks do not capture epistemic uncertainty BID15 , proposing instead to use Monte Carlo samples from a dropout-regularized neural network to produce uncertainty estimates.

DAL has demonstrated success on NLP tasks.

BID28 explores AL for sentiment classification, proposing a new sampling heuristic, choosing examples for which the expected update to the word embeddings is largest.

Recently, BID26 matched state of the art performance on named entity recognition, using just 25% of the training data.

BID13 and BID14 explore other measures of uncertainty over neural network predictions.

Learning from partial labels Many papers on learning from partial labels BID10 BID21 BID2 assume that partial labels are given a priori and fixed.

BID10 formalizes the partial labeling problem in the probabilistic framework and proposes a minimum entropy based solution.

BID21 proposes an efficient algorithm to learn classifiers from partial labels within the max-margin framework.

BID2 addresses desirable properties of partial labels that allow learning from them effectively.

While these papers assume a fixed set of partial labels, we actively solicit partial feedback.

This presents new algorithmic challenges: (i) the partial labels for each data point changes across training rounds; (ii) the partial labels result from active selection, which introduces bias; and (iii) our problem setup requires a sampling strategy to choose questions.

Our experiments validate the active learning with partial feedback framework on large-scale classification benchmarks.

The best among our proposed ALPF learners fully labels the data with 42% fewer binary questions as compared to traditional active learners.

Our diagnostic analysis suggests that in ALPF, it's sometimes more efficient to start with "easier" examples that can be cheaply annotated rather than with "harder" data as often suggested by traditional active learning.

A WARM-STARTING PLOT ALPF -ERC -0% ALPF -ERC -5% ALPF -ERC -10% FIG4 : This plot compares our models under various amounts of warm-starting with pre-labeled i.i.d.

data.

We find that on the investigated datasets, ERC does benefit from warm-starting.

However, absent warm-starting, EIG performs significantly worse and EDC suffers even more.

We find that 5% warmstarting helps these two models and that for both, increasing warm-starting from 5% up to 10% does not lead to further improvements.

On Tiny ImageNet, we normally re-initialize and train models from scratch for 75 epochs after every 30K questions.

Since we found re-initialization is crucial for good performance, to ensure a fair comparison, we keep the same re-initialization frequency (i.e. every 30K questions) while updating the model by fine-tuning 5 epochs after every 3K questions.

This results in 10X faster model updating frequency.

As in Figure 6 and Table 3 , results show only ALPF-EDC and ALPF-ERC seem to benefit from updating 10 times more frequently

@highlight

We provide a new perspective on training a machine learning model from scratch in hierarchical label setting, i.e. thinking of it as two-way communication between human and algorithms, and study how we can both measure and improve the efficiency. 

@highlight

Introduces a new Active Learning setting where the oracle offers a partial or weak label instead of querying for a particular example's label, leading to a simpler retrieval of information.

@highlight

This paper proposes a method of active learning with partial feedback that outperforms existing baselines under a limited budget.

@highlight

The paper considers a multiclass classification problem in which labels are grouped in a given number M of subsets, which contain all individual labels as singletons.