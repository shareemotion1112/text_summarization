In many applications labeled data is not readily available, and needs to be collected via pain-staking human supervision.

We propose a rule-exemplar model for collecting human supervision to combine the scalability of rules with the quality of instance labels.

The supervision is coupled such that it is both natural for humans and synergistic for learning.

We propose a training algorithm that jointly denoises rules via latent coverage variables, and trains the model through a soft implication loss over the coverage and label variables.

Empirical evaluation on five different tasks shows that (1) our algorithm is more accurate than several existing methods of learning from a mix of clean and noisy supervision, and (2)   the coupled rule-exemplar supervision is effective in denoising rules.

With the ever-increasing reach of machine learning, a common hurdle to new adoptions is the lack of labeled data and the pain-staking process involved in collecting it via human supervision.

Over the years, several strategies have evolved for reducing the tedium of collecting human supervision.

On the one hand are methods like active learning and crowd-consensus learning that seek to reduce the cost of supervision in the form of per-instance labels.

On the other hand is the rich history of rule-based methods (Appelt et al., 1993; Cunningham, 2002) where humans code-up their supervision as labeling rules.

There is growing interest in learning from such scalable, albiet noisy, supervision (Ratner et al., 2016; Pal & Balasubramanian, 2018; Bach et al., 2019; Sun et al., 2018; Kang et al., 2018 ).

However, clean task-specific instance labels continue to be critical for reliable results (Goh et al., 2018; Bach et al., 2019) even when fine-tuning models pre-trained on indirect supervision (Sun et al., 2017; Devlin et al., 2018) .

In this paper we propose a unique blend of cheap coarse-grained supervision in the form of rules and expensive fine-grained supervision in the form of labeled instances.

Instead of supervising rules and instance labels independently, we propose that each labeling rule be attached with exemplars of where the rule correctly 'fires'.

Thus, the rule can be treated as a noisy generalization of those exemplars.

Often rules are coded up only after inspecting data.

As a human inspects instances, he labels them, and then generalizes them to rules.

Thus, humans provide paired supervision of rules and exemplars demonstrating correct deployment of that rule.

We explain further with two illustrative applications.

Our examples below are from the text domain because rules have been traditionally used in many NLP tasks, but our learning algorithm is agnostic to how rules are expressed.

Sentiment Classification Consider an instance I highly recommend this modest priced cellular phone that a human inspects for a sentiment labeling task.

After labeling it as positive, he can easily generalize it to a rule Contains 'highly recommend' → positive label.

This rule generalizes to several more instances, thereby eliminating the need of per-instance labeling on those.

However, the label assigned by this rule on unseen instances may not be as reliable as the explicit label on this specific exemplar it generalized.

For example, it misfires on I would highly recommend this phone if it weren't for their poor service.

Slot-filling Consider a slot-filling task on restaurant reviews over labels like cuisine, location, and time.

When an annotator sees an instance like: what chinese restaurants in this city have good reviews?

, after labeling token chinese as cuisine, he generalizes it to a rule: (.

* ese|.

* ian|mexican) restaurants → (cuisine) restaurants.

This rule matches hundreds of instances in the unlabeled set, but could wrongly label a phrase like these restaurants.

We present in Section 3 other applications where such supervision is natural.

Our focus in this paper is developing algorithms for training models under such coupled rule-exemplar supervision.

Our main challenge is that the labels induced by the rules are more noisy than instance-level supervised labels because humans tend to over generalize (Tessler & Goodman, 2019) as we saw in the illustrations above.

Learning with noisy labels with or without additional clean data has been a problem of long-standing interest in ML (Khetan et al., 2018; Zhang & Sabuncu, 2018; Ren et al., 2018b; Veit et al., 2017; Shen & Sanghavi, 2019) .

However, we seek to design algorithms that better capture rule-specific noise with the help of exemplars around which we have supervision that the rule fired correctly.

We associate a latent random variable on whether a rule correctly 'covers' an instance, and jointly learn the distribution among the label and all cover variables.

This way we simultaneously train the classifier with corrected rule-label examples, and restrict over-generalized rules.

In summary our contributions in this paper are as follows:

Our contributions (1) We propose the paradigm of supervision in the form of rules generalizing labeled exemplars that is natural in several applications.

(2) We design a training method that simultaneously denoises over-generalized rules via latent coverage variables, and trains a classification model with a soft implication loss that we introduce.

(3) Through experiments on five tasks spanning question classification, spam detection, sequence labeling, and record classification we show that our proposed paradigm of supervision enables an effective synergy between rule-level and instance-level supervision.

(4) We compare our algorithm to several recent frameworks for learning with noisy supervision and constraints, and show much better results with our method.

We first formally describe the problem of learning from rules generalizing examplars on a classification task.

Let X denote the space of instances and Y = {1, . . .

, K} denote the space of class labels.

Let the set of labeled examples be L = {(x 1 , 1 , e 1 ), . . . , (x n , n , e n )} where x i ∈ X is an instance, i ∈ Y is its user-provided label, and e i ∈ {R 1 , . . .

, R m , ∅} denotes that x i is an exemplar for rule e i .

Some labeled instances may not be generalized to rules and for them e i = ∅. Also, a rule can have more than one exemplar associated with it.

Each rule R j could be a blackbox function R j : x → { j , ∅} that takes as input an instance x ∈ X and assigns it either label j or no-label.

When the ith labeled instance is an exemplar for rule R j (that is, e i = R j ), the label of the instance i should be j .

Additionally, we have a different set of unlabeled instances U = {x n+1 , . . .

, x N }.

The cover set H j of rule R j is the set of all instances in U ∪ L for which R j assigns a noisy label j .

An instance may be covered by more than one rule or no rule at all, and the labels provided by these rules may be conflicting.

Our goal is to train a classification model P θ (y|x) using L and U to maximize accuracy on unseen test instances.

A baseline solution is to use R j to noisily label the covered U instances using majority or other consensus method of resolving conflicts.

We then train P θ (y|x) on the noisy labels using existing algorithms for learning from noisy and clean labels (Veit et al., 2017; Ren et al., 2018b) .

However, we expect to be able to do better by learning the systematic pattern of noise in rules along with the classifier P θ (y|x).

Our noise model on R j A basic premise of our learning paradigm is that the noise induced by a rule R j is due to over-generalizing the exemplar(s) seen when creating the rule.

And, there exists a smaller neighborhood closer to the exemplar(s) where the noise is zero.

We model this phenomenon by associating a latent Bernoulli random variable r ji for each instance x i in the stated cover set H j of each rule R j .

When r ji = 1, rule R j has not over-generalized on x i , and there is no noise in the label j that R j assigns to x i .

When r ji = 0 we flag an over-generalization, and abstain from labeling x i as j suspecting it to be too noisy.

We call r ji s as the latent coverage variables.

We propose to learn the distribution of r j using another network with parameters φ that outputs the probability P jφ (r j |x) that r j = 1.

We then seek to jointly learn P θ (y|x) and P jφ (r j |x) to model the distribution over the true label y and true coverage r j for each rule j and each x in H j .

Thus P jφ plays the role of restricting a rule R j so that r j is not necessarily 1 for all instances in its cover set H j An example We make our discussion concrete with an example.

Figure 1 shows a two-dimensional X space with labeled points L denoted as red crosses and blue circles and unlabeled points as dots, and the true labels as background color of the region.

We show two rule-exemplar pairs: (x 1 , y 1 = red, R 1 ), (x 2 , y 2 = blue, R 2 ).

Clearly, both rules R 1 , R 2 have over-generalized to the wrong region.

If we train a classifier with many examples in H 1 ∪ H 2 wrongly labeled by rules, then even with a noise tolerant loss function like Zhang & Sabuncu (2018) , the classifier P θ (y|x) might be misled.

In contrast, what we hope to achieve is to learn the P jφ (r j |x) distribution using the limited labeled data and the overlap among the rules such that Pr(r j |x) predicts a value of 0 for examples wrongly covered.

Such examples are then excluded from training P θ .

The dashed boundaries indicate the revised boundaries of R j s that we can hope to learn based on consensus on the labeled data and the set of rules.

Even after such restriction, R j s are useful for training the classifier because of the unlabeled points inside the dashed regions that get added to the labeled set.

2.1 HOW WE JOINTLY LEARN P θ AND P jφ In general we will be provided with several rules with arbitrary overlap in the set of labeled L and unlabeled examples U that they cover.

Intuitively, we want the label distribution P θ (y|x) to correctly restrict the coverage distribution P jφ (r j |x), which in turn can provide clean labels to instances in U that can be used to train P θ (y|x).

We have two types of supervision in our setting.

First, individually for each of the networks we have ground truth values of y and r j for some instances.

For the P θ (y|x) distribution, supervision on y is provided by the human labeled data L, and we use these to define the usual log-likelihood as one term in our training objective:

For learning the distribution P jφ (r j |x) over the coverage variables, the only sure-shot labeled data is that r ji = 1 for any x i that is an exemplar of rule R j and r ji = 0 for any x i ∈ H j whose label i is different from j .

For other labeled instances x i covered with rules R j with agreeing labels, that is i = j we do not strictly require that r ji = 1.

In the example above the corrected dashed red boundary excludes a red labeled point to reduce its noise on other points.

However, if the number of labeled exemplars are too few, we regularize the networks towards more rule firings, by adding a noise tolerant r ji = 1 loss on the instances with agreeing labels.

We use the generalized cross entropy loss of Zhang & Sabuncu (2018) .

Note for other instances x i in R j 's cover H j , value of r ji is unknown and latent.

The second type of supervision is on the relationship between r ji and y i for each

Figure 2: Negative implication loss A rule R j imposes a causal constraint that when r ji = 1, the label y i has to be j .

r ji = 1 =⇒ y i = j ∀x i ∈ H j (3) We convert this hard constraint into a (log) probability of the constraint being satisfied under the P θ (y|x) and P jφ (r j |x) distributions as: Figure 2 shows a surface plot of the above log probability as a function of P θ ( j |x) (shown as axis P(y) in figure) and P jφ (r j = 1|x) (shown Table 1 : Statistics of datasets and their rules.

%Cover is fraction of instances in U covered by at least one rule.

Precision refers to micro precision of rules.

Conflict denotes the fraction of instances covered by conflicting rules among all the covered instances.

Avg |Hj| is average cover size of a rule in U .

Rules Per Instance is average number of rules covering an instance in U .

as axis P(r) in figure) for a single rule.

Observe that likelihood drops sharply as P (r j |x) is close to 1 but P (y = j |x) is close to zero.

For all other values of these probabilities the log-likelihood is flat and close to zero.

Specifically, when P jφ predicts low values of r j for a x, the P θ (y|x) surface is flat, effectively withdrawing the (x, j ) supervision from training the classifier P θ .

Thus maximizing this likelihood provides a soft enforcement of the constraint without any other unwanted biases.

We call this the negative implication loss.

We do not need to explicitly model the conflict among rules, that is when an x i is covered by two rules R j and R k of differing labels ( j = k ), then both r ji and r ki cannot be 1.

This is because the constraint among pairs (y i , r ji ) and (y i , r ki ) as stated in Equation 3 subsumes this one.

During training we then seek to maximize the log of the above probability along with normal data likelihood terms.

Putting the terms in Equations 1, 2 and 4 together our final training objective is:

We refer to our training loss as a denoised rule-label implication loss or ImplyLoss for short.

The LL(φ) term seeks to denoise rule coverage which then influence the y distribution via the implication loss.

We explored several other methods of enforcing the constraint among y and r j in the training of the P θ and P jφ networks.

Our method ImplyLoss consistently performed the best among several methods we tried including the recent posterior regularization (Ganchev et al., 2010; Hu et al., 2016) method of enforcing soft constraints and co-training (Blum & Mitchell, 1998) .

Network Architecture Our network has three modules.

(1) A shared embedding layer that provides the feature representation of the input.

When labeled data is scarce, this will typically be a pre-trained layer from a related task.

We describe the embedding module for each task individually in the experiments section.

(2) A classification network that models P θ (y|x) with parameters θ.

The embedding of an input x is passed through multiple non-linear layers with ReLU activation, a last linear layer followed by Softmax to output a distribution over the class labels.

(3) A rule network that models P jφ (r j = 1|x) whose parameters φ are shared across all rules.

The input to the network is rule-specific and concatenates the embedding of the input instance x, and a a one-hot encoding of the rule id 'j'.

The inputs are transformed through multiple layers of ReLU before passing through a Sigmoid activation which outputs the probability P jφ (r j = 1|x).

We compare our training algorithms against simple baselines, existing error-tolerant learning algorithms, and existing constraint-based learning in deep networks.

We evaluate across five datasets spanning three task types: text classification, sequence labeling, and record classification.

We augment the datasets with rules, that we obtained manually in three cases, from pre-existing public sources in one case, and automatically in another.

Table 1 presents statistics summarizing the datasets and rules.

A brief description of each appears below.

Question Classification (Li & Roth, 2002 ): This is a TREC-6 dataset to classify a question to one of six categories: {Abbreviation, Entity, Description, Human, Location, Numeric-value}. The training set has 5452 instances which are split as 68 for L, 500 for validation, and the remaining as U .

Each example in L is generalized as a rule represented by a regular expression.

E.g. After labeling How do you throw a housewarming party ?

as Description we define a rule (how|How|what|What)(does|do|to|can).

* → Description.

More rules in Table 4 of supplementary.

Although, creating such 68 generalised rules required 90 minutes, the generalizations cover 4637 instances in U , almost two orders of magnitude more instances than in L!

On an average each of our rule covered 124 instances (|H j | column in Table 1 ).

But the precision of labels assigned by rules was only 63.8%, and 22.5% of covered instances had an inter-rule conflict.

This clearly demonstrates the noise in the rule labelings.

Accuracy is used as the performance metric.

MIT-R 1 (Liu et al., 2013) : This is a slot-filling task on sentences about restaurant search and the task is to label each token as one of {Location, Hours, Amenity, Price, Cuisine, Dish, Restaurant Name, Rating, Other}. The training data is randomly split into 200 sentences (1690 tokens) as L, 500 sentences (4k tokens) as validation and remaining 6.9k sentences (64.9k tokens) as U .

We manually generalize 15 examples in L. E.g. After inspecting the sentence where can i get the highest rated burger within ten miles and labeling highest rated as Rating, we provide the rule:

.

* (highly|high|good|top|highest)(rate|rating|rated).

* → Rating to the matched positions.

More examples in Table 7 of supplementary.

Although, creating 15 generalizing rules took 45 minutes of annotator effort, the rules covered roughly 9k tokens in U .

F1 metric is used for evaluation on the default test set of 14.2k tokens over 1.5k sentences. (Almeida et al., 2011) : This dataset contains 5.5k text messages labeled as spam/not-spam, out of which 500 were held out for validation and 500 for testing.

We manually generalized 69 exemplars to rules.

Remaining examples go in the U set.

The rules here check for presence of keywords or phrases in the SMS .

* guaranteed gift .

* → spam.

A rule covers 31 examples on an average and has a precision of 97.3%.

However, in this case only 40% of the unlabeled set is covered by a rule.

We report F1 here since class is skewed.

More examples in Table 5 of supplementary.

Youtube Spam Classification (Alberto et al., 2015) :

Here the task is to classify comments on YouTube videos as Spam or Not-Spam.

We obtain this from Snorkel's Github page 2 , which provides 10 labeling functions which we use as rules, an unlabeled train set which we use as U , a labeled dev set to guide the creation of their labeling functions which we use as L, and labeled test and validation sets which we use in the same roles.

Their labeling functions have a large coverage (258 on average), and a precision of 78.6%.

Census Income (Dua & Graff, 2019) : This UCI dataset is extracted from the 1994 U.S. census.

It lists a total of 13 features of an individual such as age, education level, marital status, country of origin etc.

The primary task on it is binary classification -whether a person earns more than $50K or not.

The train data consists of 32563 records.

We choose 83 random data points as L, 10k points as U and 5561 points as validation data.

For this case we created the rules synthetically as follows: We hold out disjoint 16k random points from the training dataset as a proxy for human knowledge and extract a PART decision list (Frank & Witten, 1998) from it as our set of rules.

We retain only those rules which fire on L.

Network Architecture Since our labeled data is small we depend on pre-trained resources.

As the embedding layer we use a pretrained ELMO network where 1024 dimensional contextual token embeddings serve as representations of tokens in the MIT-R sentences, and their average serve as representation for sentences in Question and SMS dataset.

Parameters of the embedding network are held fixed during training.

For sentences in the YouTube dataset, we use Snorkel's 2 architecture of a simple bag-of-words feature representation marking the frequent uni-grams and bi-grams present in a sentence using a few-hot vector.

For the Census dataset categorical features are represented as one hot vectors, while real valued features are simply normalized.

For MIT-R, Question and SMS both classification and rule-weight network contain two 512 dimensional hidden layers with ReLU activation.

For Census, both the networks contain two 256 dimensional hidden layers with ReLU activation.

For YouTube, the classifier network is a simple logistic regression like in Snorkel's code.

The rule network has one 32-dimensional hidden layer with ReLU activation.

Each reported number is obtained by averaging over five random initializations.

Whenever a method involved hyper-parameters to weigh the relative contribution of various terms in the objective, we used a validation dataset to tune the value of the hyper-parameter.

Hyperparameters used are provided in Section C of supplementary.

In Table 2 we compare our method with the following alternatives on each of the five datasets:

Majority: that predicts via majority vote among the rules that cover an instance.

This baseline indicates the stand-alone quality of rules, no network is learned here.

Ties are broken arbitrarily for class-balanced datasets or by using a default class.

Table 2 , shows that the accuracy of majority is quite poor indicating either poor precision or poor coverage of the rule sets 3 .

Only-L : Here we train the classifier P θ (y|x) only on the labeled data L using the standard crossentropy loss (Equation 1).

Rule generalisations are not utilized at all in this case.

We observe in Table 2 that even with the really small labeled set we used for each dataset, the accuracy of a classifier learned with clean labeled data is much higher than noisy majority labels of rules.

We consider this method as our baseline and report the gains on remaining methods.

L+Umaj:

Next we train the classifier on L along with U maj obtained by labeling instances in U with the majority label among the rules applicable to the instance.

The row corresponding to L+Umaj in Table 2 provides the gains of this method over Only-L. We observe gains with the noisily labeled U in four out of the five cases.

Noise-tolerant: Since labels in U maj are noisy, we next use Zhang & Sabuncu (2018)'s noise tolerant generalized cross entropy loss on them with regular cross-entropy loss on the clean L as follows:

Parameter q ∈ [0, 1] controls the noise tolerance which we tune as a hyper-parameter.

We observe that in all cases the above objective improves beyond Only-L validating that noise-tolerant loss functions are useful for learning from noisy labels on U maj .

Learning to Reweight (L2R) (Ren et al., 2018b) : is a recent method for training with a mix of clean and noisy labeled data.

They train the classifier by meta-learning to re-weight the loss on the noisily labelled instances (U maj ) with the help of the clean examples (L).

This method shows huge variance in its accuracy gains over Only-L across datasets and is worse in two of the cases.

All the above methods employ no extra parameters to denoise or weight individual rules.

We next compare with a number of methods that do.

L+Usnorkel: This method replaces Majority-based consensus with Snorkel's generative model (Ratner et al., 2016 ) that assigns weights to rules and labels examples in U .

Thereafter we use the same approach as in L+Umaj with just Snorkel's soft-labels instead of Majority on U .

The results are mixed and we do not get any consistent gains over Only-L and over L+Umaj.

We also compare with using noise-tolerant loss on U labeled by Snorkel (Eqn:6) which we call SnorkelNoise-Tolerant.

We observe more consistent improvements then, but these results are not much better than Noise-Tolerant on U maj .

We next compared with a method that simultaneously learns two sets of networks P θ and P jφ like in ours but with different loss function and training schedule.

Posterior Regularization (PR): This method proposed in Hu et al. (2016) also treats rules as softconstraints and has been used for training neural networks for structured outputs.

They use Ganchev et al. (2010) 's posterior regularization framework to train the two networks in a teacher-student setup.

We adapt the same framework and get a procedure as follows: The student proposes a distribution over y and r j s using current P θ and P jφ , the teacher uses the constraint in Eq 3 to revise the distributions so as to minimize the probability of violations, the student updates parameters θ and φ to minimize KL distance with the revised distribution.

The detailed formulation appear in the Section A of supplementary.

We find that this method is worse than Only-L in two cases and worse than the noise-tolerant method that does not train extra φ parameters.

Overall our approach of training with denoised rule-label implication loss provides much better accuracy than all the above eight methods and we get consistent gains over Only-L on all datasets.

On the question dataset we get 11.9 points gains over Only-L whereas the best gains by existing method was 0.5.

A useful property of our method compared to the PR method above is that the training process is simple and fits into the batch stochastic gradient training template.

In contrast, PR requires special alternating computations.

We next perform a number of diagnostics experiments to explain the reasons for the superior performance of our method.

Diagnostics: Effectiveness of learning true coverage via P jφ An important part of our method is the rule-specific denoising learned via the P jφ network.

In the chart alongside we plot the original precision of each rule on the test data, and the precision after suppressing those rule labelings where P jφ (r j |x) predicts 0 instead of 1.

Observe now that the precision is more than 91% on all datasets.

For the Question dataset, the precision jumped from 64% to 98%.

The percentage of labelings suppressed (shown by the dashed line) is higher on datasets with noisier rules (e.g. compare Question and SMS).

This shows that P jφ is able to denoise rules by capturing the distribution of the latent true coverage variables with the limited LL(φ) loss and indirectly via the implication loss.

We next evaluate the importance of the exemplar-rule pairs in learning the P jφ and P θ networks.

The exemplars of a rule give an interesting new form of supervision about an instance where a labeling rule must fire.

To evaluate the importance of this supervision, we exclude the r j = 1 likelihood on rule-exemplar pairs from LL(φ), that is, the first term in Equation 2 is dropped.

In the table below we see that performance of ImplyLoss drops when the exemplar-rule supervision is removed.

Interestingly, even after this drop, the performance of ImplyLoss surpasses most of the methods in Table 2 indicating that even without exemplar-rule pairs our training objective is effective at learning from rules and labeled instances.

Effect of increasing labeled data L We increase L while keeping the number of rules fixed on the Question dataset.

In the attached plot we see the accuracy of our method (ImplyLoss) against Only-L and Posterior Reg.

We observe the expected trend that the gap between the method narrows as labeled data increases.

Learning from noisily labeled data has been extensively studied in settings like crowdsourcing.

One category of these algorithms upper-bound the loss function to make it robust to noise.

These include methods like MAE (Ghosh et al., 2017) , Generalized Cross Entropy (CE) (Zhang & Sabuncu, 2018) , and Ramp loss (Collobert et al., 2006) .

Most of these assume that noise is independent of the input given the true label.

In our model noise is systematic and instance-dependent.

A second category assume that a small clean dataset is available along with noisily labeled data.

This is also true in our case, and we compared with a state of the art method in that category Ren et al. (2018b) that chooses a descent direction that aligns with a clean validation set using meta-learning.

Others in this category include: Shen & Sanghavi (2019)'s method of iteratively selecting examples with smallest loss, and Veit et al. (2017)'s method of learning a separate network to transform noisy labels to cleaned ones which are used to impose a cross-entropy loss on P θ (y|x).

In contrast, we perform rule-specific cleaning via latent coverage variables and a flexible implication loss which withdraws y supervision when P jφ (r ji |x) assumes low values.

Another way of relating clean and noisy labels is via an instance-independent confusion matrix learned jointly with the classifier (Khetan et al., 2018; Goldberger & Ben-Reuven, 2016; Han et al., 2018b; a) .

These works assume that the confusion matrix is instance independent, which does not hold for our case.

Tanaka et al. (2018) uses confidence from the classifier to eliminate noise but they need to ensure that the network does not memorize noise.

Our learning setup also has the advantage of extracting confidence from a different network.

There is growing interest in integrating logical rules with labeled examples for training networks, specifically for structured outputs (Manhaeve et al., 2018; Xu et al., 2018; Fischer et al., 2019; Sun et al., 2018; Ren et al., 2018a) .

Xu et al. (2018); Fischer et al. (2019) convert rules on output nodes of network, to (almost differentiable) loss functions during training.

The primary difference of these methods from ours is that they assume that rules are correct whereas we assume them to be noisy.

Accordingly, we simultaneously correct the rules and use them to improve the classifier, whereas they use the rules as-is to train the network outputs.

A well-known framework for working with soft rules is posterior regularization (Ganchev et al., 2010) which is used in Hu et al. (2016) to train deep structured output networks while harnessing logic rules.

Ratner et al. (2016) works only with noisy rules treating them as black-box labeling functions and assigns a linear weight to each rule based on an agreement objective.

Our learning model is more powerful that attempts to learn a non-linear network to restrict rule boundaries rather than just weight their outputs.

We presented a comparison with both these approaches in the experimental section, and showed superior performance.

To the best of our knowledge, our proposed paradigm of coupled rule-exemplar supervision is novel, and our proposed training algorithm is able to harness them in ways not possible by existing frameworks for learning from rules or noisy supervision.

We proposed a new rule-exemplar model for collecting human supervision to combine the scalability of top-level rules with the quality of instance-level labels.

We show that such supervision is natural since humans typically inspect examples to code rules.

Furthermore, such coupled examples provide supervision on correct firing of rules which help to denoise rules.

We propose to train the classifier while jointly denoising rules via latent coverage variables imposing a soft-implication constraint on the true label.

Empirically on five datasets we show that our training algorithm that performs rule-specific denoising is better than generic noise-tolerant learning.

In future we plan to deploy this framework on other applications where human supervision is a scarce resource.

We model a joint distribution Q(y, r 1 , . . .

, r n |x) to capture the interaction among the label random variable y and coverage random variables r 1 , . . .

, r n of any instance x. We use r to compactly represent r 1 , . . .

, r n .

Strictly speaking, when a rule R j does not cover x, the r j is not a random variable and its value is pinned to 0 but we use this fixed-tuple notation for clarity.

The random variables r j and y impose a constraint on the joint distribution Q: for a x ∈ H j when r j = 1, the label y cannot be anything other than j .

r j = 1 =⇒ y = j ∀x ∈ H j (7) We can convert this into a soft constraint on the marginals of the distribution Q by stating the probability of y = j Q(y, r j = 1|x) should be small.

The singleton marginals of Q along the y and r j variables are tied to the P θ and P jφ (r j |x) we seek to learn.

A network with parameters θ models the classifier P θ (y|x), and a separate network with φ variables (shared across all rules) learns the P jφ (r j |x) distribution.

The marginals of joint Q should match these trained marginals and we use a KL term for that:

We call the combined KL term succinctly as KL(Q, P θ ) + KL(Q, P φ ).

Further the P θ and P jφ distributions should maximize the log-likelihood on their respective labeled data as provided in Equation 1 and Equation 2 respectively.

Putting all the above objectives together with hyper-parameters α > 0, λ > 0 we get our final objective as:

We show in Section A.1 that this gives rise to the solution for Q in terms of P θ , P jφ and alternately for P θ , P jφ in terms of Q as follows.

where δ(y = j ∧ r j = 1) is an indicator function that is 1 when the constraint inside holds, else it is 0.

Computing marginals of the above using straight-forward message passing techniques we get:

(13) Thereafter, we solve for θ and φ in terms of a given Q as

Here, γ = 1 α .

This gives rise to an alternating optimization algorithm as in the posterior regularization framework of Ganchev et al. (2010) .

We initialize θ and φ randomly.

Then in a loop, we perform the following two steps alternatively much like the EM algorithm (Dempster et al., 1977) .

Here we compute marginals Q(y|x) and Q(r j |x) from current P θ and P jφ using Equations 12 and 13 respectively for each x in a batch.

This computation is straight-forward and does not require any neural optimization.

We can interpret the Q(y|x) as a small correction of the P θ (y|x) so as to align better with the constraints imposed by the rules in Equation 3.

Likewise Q(r j |x) is an improvement of current P jφ s in the constraint preserving direction.

For example, the expected r j values might be reduced for an instance if its probability of y being j is small.

Parameter update step: We next reoptimize the θ and φ parameters to match the corrected Q distribution as shown in Equation 14.

This is solved using standard stochastic gradient techniques.

The Q terms can just be viewed as weights at this stage which multiply the loss or label likelihood.

A pseudocode of our overall training algorithm is described in Algorithm 1.

Input: L, U Initialize parameters θ, φ randomly for a random training batch from U ∪ L do Obtain P θ (y|x) from the classification network.

Obtain P jφ (r j |x) j∈[n] from the rule-weight network.

Calculate Q(y|x) using Eqn 12 and Q(r j |x) j∈[n] using Eqn 13.

Update θ and φ by taking a step in the direction to minimize the loss in Eqn 14. end for Output: θ , φ

Treat each Q(y, r) as an optimization variable with the constraint that y,r Q(y, r) = 1.

We express this constraint with a Langrangian multiplier η in the objective.

Also, define a distribution

It is easy to verify that the KL terms in our objective 10 can be collapsed as KL(Q; P θ,φ ).

The rewritten objective (call it F (Q, θ, φ) ) is now:

Next we solve for ∂F ∂Q(y,r) = 0 after expressing the marginals in their expanded forms: e.g. Q(y, r j |x) = r1,...

,rj−1,rj+1,...,rn Q(y, r 1 , . . . , r n |x).

This gives us ∂F ∂Q(y, r) = log Q(y, r) − log P θ,φ (y, r|x)

Equating it to zero and substituting for P θ,φ we get the solution for Q(y, r) in Equation 11.

The proof for the optimal P θ and P jφ while keeping Q fixed in Equation 15 is easy and we skip here.

We provide a list of rules for each task type.

Great News!

Call FREEFONE 08006344447 to claim your guaranteedå£1000 CASH orå£2000 gift.

cuisine1a= ['italian','american', 'japanese','spanish','mexican', 'chinese','vietnamese','vegan']

cuisine1b= ['bistro','delis'] cuisine2= ['barbecue','halal', 'vegetarian','bakery'] can you find me some chinese food

For all the experiments we use a learning rate of 1e-4, batch-size of 32 and a dropout of 0.8 (keep probability).

All the models were trained for a maximum of 100 epochs.

We use early stopping using a validation set.

We provide a list of hyperparameters used in our experiments.

Table 9 : Meta-learning rate of Learning to Reweight method (L2R) for various datasets

@highlight

Coupled rule-exemplar supervision and a implication loss helps to jointly learn to denoise rules and imply labels.