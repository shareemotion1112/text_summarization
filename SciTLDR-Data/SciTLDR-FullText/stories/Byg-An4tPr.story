In this paper, we aim to develop a novel mechanism to preserve differential privacy (DP) in adversarial learning for deep neural networks, with provable robustness to adversarial examples.

We leverage the sequential composition theory in DP, to establish a new connection between DP preservation and provable robustness.

To address the trade-off among model utility, privacy loss, and robustness, we design an original, differentially private, adversarial objective function, based on the post-processing property in DP, to tighten the sensitivity of our model.

An end-to-end theoretical analysis and thorough evaluations show that our mechanism notably improves the robustness of DP deep neural networks.

The pervasiveness of machine learning exposes new vulnerabilities in software systems, in which deployed machine learning models can be used (a) to reveal sensitive information in private training data (Fredrikson et al., 2015) , and/or (b) to make the models misclassify, such as adversarial examples (Carlini & Wagner, 2017) .

Efforts to prevent such attacks typically seek one of three solutions:

(1) Models which preserve differential privacy (DP) (Dwork et al., 2006) , a rigorous formulation of privacy in probabilistic terms; (2) Adversarial training algorithms, which augment training data to consist of benign examples and adversarial examples crafted during the training process, thereby empirically increasing the classification accuracy given adversarial examples (Kardan & Stanley, 2017; Matyasko & Chau, 2017) ; and (3) Provable robustness, in which the model classification given adversarial examples is theoretically guaranteed to be consistent, i.e., a small perturbation in the input does not change the predicted label (Cisse et al., 2017; Kolter & Wong, 2017) .

On the one hand, private models, trained with existing privacy-preserving mechanisms (Abadi et al., 2016; Shokri & Shmatikov, 2015; Phan et al., 2016; 2017b; a; Yu et al., 2019; Lee & Kifer, 2018) , are unshielded under adversarial examples.

On the other hand, robust models, trained with adversarial learning algorithms (with or without provable robustness to adversarial examples), do not offer privacy protections to the training data.

That one-sided approach poses serious risks to machine learning-based systems; since adversaries can attack a deployed model by using both privacy inference attacks and adversarial examples.

To be safe, a model must be i) private to protect the training data, and ii) robust to adversarial examples.

Unfortunately, there has not yet been research on how to develop such a model, which thus remains a largely open challenge.

Simply combining existing DP-preserving mechanisms and provable robustness conditions (Cisse et al., 2017; Kolter & Wong, 2017; Raghunathan et al., 2018) cannot solve the problem, for many reasons.

(a) Existing sensitivity bounds (Phan et al., 2016; 2017b; a) and designs (Yu et al., 2019; Lee & Kifer, 2018) have not been developed to protect the training data in adversarial training.

It is obvious that using adversarial examples crafted from the private training data to train our models introduces a previously unknown privacy risk, disclosing the participation of the benign examples (Song et al., 2019) .

(b) There is an unrevealed interplay among DP preservation, adversarial learning, and robustness bounds.

(c) Existing algorithms cannot be readily applied to address the trade-off among model utility, privacy loss, and robustness.

Therefore, theoretically bounding the robustness of a model (which both protects the privacy and is robust against adversarial examples) is nontrivial.

Motivated by this open problem, we propose to develop a novel differentially private adversarial learning (DPAL) mechanism to: 1) preserve DP of the training data, 2) be provably and practically robust to adversarial examples, and 3) retain high model utility.

In our mech-anism, privacy-preserving noise is injected into inputs and hidden layers to achieve DP in learning private model parameters (Theorem 1).

Then, we incorporate ensemble adversarial learning into our mechanism to improve the decision boundary under DP protections.

To do this, we introduce a concept of DP adversarial examples crafted using benign examples in the private training data under DP guarantees (Eq. 9).

To address the trade-off between model utility and privacy loss, we propose a new DP adversarial objective function to tighten the model's global sensitivity (Theorem 3); thus, we significantly reduce the amount of noise injected into our function, compared with existing works (Phan et al., 2016; 2017b; a) .

In addition, ensemble DP adversarial examples with a dynamic perturbation size µ a are introduced into the training process to further improve the robustness of our mechanism under different attack algorithms.

An end-to-end privacy analysis shows that, by slitting the private training data into disjoint and fixed batches across epochs, the privacy budget in our DPAL is not accumulated across training steps (Theorem 4).

After preserving DP in learning model parameters, we establish a solid connection among privacy preservation, adversarial learning, and provable robustness.

Noise injected into different layers is considered as a sequence of randomizing mechanisms, providing different levels of robustness.

By leveraging the sequential composition theory in DP (Dwork & Roth, 2014) , we derive a novel generalized robustness bound, which essentially is a composition of these levels of robustness (Theorem 5 and Proposition 1).

To our knowledge, our mechanism establishes the first connection between DP preservation and provable robustness against adversarial examples in adversarial learning.

Rigorous experiments conducted on MNIST and CIFAR-10 datasets (Lecun et al., 1998; Krizhevsky & Hinton, 2009) show that our mechanism notably enhances the robustness of DP deep neural networks, compared with existing mechanisms.

In this section, we revisit adversarial learning, DP, and our problem definition.

Let D be a database that contains N tuples, each of which contains data x ∈ [−1, 1] d and a ground-truth label y ∈ Z K , with K possible categorical outcomes.

Each y is a one-hot vector of K categories y = {y 1 , . . . , y K }.

A single true class label y x ∈

y given x ∈ D is assigned to only one of the K categories.

On input x and parameters θ, a model outputs class scores f : R d → R K that maps d-dimensional inputs x to a vector of scores f (x) = {f 1 (x), . . .

, f K (x)} s.t.

∀k ∈ [1, K] : f k (x) ∈ [0, 1] and K k=1 f k (x) = 1.

The class with the highest score value is selected as the predicted label for the data tuple, denoted as y(x) = max k∈K f k (x).

A loss function L(f (x), y) presents the penalty for mismatching between the predicted values f (x) and original values y.

For the sake of clarity, the notations and terminologies frequently used in this paper are summarized in Table 1 (Appendix A).

Let us briefly revisit DP-preserving techniques in deep learning, starting with the definition of DP.

Definition 1 ( , δ)-DP (Dwork et al., 2006) .

A randomized algorithm A fulfills ( , δ)-DP, if for any two databases D and D differing at most one tuple, and for all O ⊆ Range(A), we have:

A smaller enforces a stronger privacy guarantee.

Here, controls the amount by which the distributions induced by D and D may differ, δ is a broken probability.

DP also applies to general metrics ρ(D, D ) ≤ 1, where ρ can be l p -norms (Chatzikokolakis et al., 2013) .

DP-preserving algorithms in deep learning can be categorized into two lines: 1) introducing noise into gradients of parameters (Abadi et al., 2016; Shokri & Shmatikov, 2015; Abadi et al., 2017; Yu et al., 2019; Lee & Kifer, 2018; Phan et al., 2019) , 2) injecting noise into objective functions (Phan et al., 2016; 2017b; a) , and 3) injecting noise into labels (Papernot et al., 2018) .

In Lemmas 2 and 4, we will show that our mechanism achieves better sensitivity bounds compared with existing works (Phan et al., 2016; 2017b; a) .

Adversarial Learning.

For some target model f and inputs (x, y x ), the adversary's goal is to find an adversarial example x adv = x + α, where α is the perturbation introduced by the attacker, such that: (1) x adv and x are close, and (2) the model misclassifies x adv , i.e., y(x adv ) = y(x).

In this paper, we consider well-known l p∈{1,2,∞} -norm bounded attacks (Goodfellow et al., 2014b) .

Let l p (µ) = {α ∈ R d : α p ≤ µ} be the l p -norm ball of radius µ. One of the goals in adversarial learning is to minimize the risk over adversarial examples: θ * = arg min θ E (x,ytrue)∼D max α p ≤µ L f (x + α, θ), y x , where an attack is used to approximate solutions to the inner maximization problem, and the outer minimization problem corresponds to training the model f with parameters θ over these adversarial examples x adv = x + α.

There are two basic adversarial example attacks.

The first one is a single-step algorithm, in which only a single gradient computation is required.

For instance, FGSM algorithm (Goodfellow et al., 2014b) finds adversarial examples by solving the inner maximization max α p ≤µ L f (x + α, θ), y x .

The second one is an iterative algorithm, in which multiple gradients are computed and updated.

For instance, in (Kurakin et al., 2016a) , FGSM is applied multiple times with T µ small steps, each of which has a size of µ/T µ .

To improve the robustness of models, prior work focused on two directions: 1) Producing correct predictions on adversarial examples, while not compromising the accuracy on legitimate inputs (Kardan & Stanley, 2017; Matyasko & Chau, 2017; Wang et al., 2016; Papernot et al., 2016b; a; Gu & Rigazio, 2014; Papernot & McDaniel, 2017; Hosseini et al., 2017) ; and 2) Detecting adversarial examples (Metzen et al., 2017; Grosse et al., 2017; Xu et al., 2017; Abbasi & Gagné, 2017; Gao et al., 2017) .

Among existing solutions, adversarial training appears to hold the greatest promise for learning robust models (Tramèr et al., 2017) .

One of the well-known algorithms was proposed in (Kurakin et al., 2016b) .

At every training step, new adversarial examples are generated and injected into batches containing both benign and adversarial examples.

The typical adversarial learning in (Kurakin et al., 2016b ) is presented in Alg.

2 (Appendix B).

DP and Provable Robustness.

Recently, some algorithms (Cisse et al., 2017; Kolter & Wong, 2017; Raghunathan et al., 2018; Cohen et al., 2019; Li et al., 2018) have been proposed to derive provable robustness, in which each prediction is guaranteed to be consistent under the perturbation α, if a robustness condition is held.

Given a benign example x, we focus on achieving a robustness condition to attacks of l p (µ)-norm, as follows:

where k = y(x), indicating that a small perturbation α in the input does not change the predicted label y(x).

To achieve the robustness condition in Eq. 2, Lecuyer et al. (Lecuyer et al., 2018) introduce an algorithm, called PixelDP.

By considering an input x (e.g., images) as databases in DP parlance, and individual features (e.g., pixels) as tuples, PixelDP shows that randomizing the scoring function f (x) to enforce DP on a small number of pixels in an image guarantees robustness of predictions against adversarial examples.

To randomize f (x), random noise σ r is injected into either input x or an arbitrary hidden layer, resulting in the following ( r , δ r )-PixelDP condition:

Lemma 1 ( r , δ r )-PixelDP (Lecuyer et al., 2018) .

Given a randomized scoring function f (x) satisfying ( r , δ r )-PixelDP w.r.t.

a l p -norm metric, we have:

is the expected value of f k (x), r is a predefined budget, δ r is a broken probability.

At the prediction time, a certified robustness check is implemented for each prediction.

A generalized robustness condition is proposed as follows:

whereÊ lb andÊ ub are the lower and upper bounds of the expected valueÊf (x) = 1 n n f (x) n , derived from the Monte Carlo estimation with an η-confidence, given n is the number of invocations of f (x) with independent draws in the noise σ r .

Passing the check for a given input guarantees that no perturbation up to l p (1)-norm can change the model's prediction.

PixelDP does not preserve DP in learning private parameters θ to protect the training data.

That is different from our goal.

Our new DPAL mechanism is presented in Alg.

1.

Our network (Figure 1 ) can be represented as: f (x) = g(a(x, θ 1 ), θ 2 ), where a(x, θ 1 ) is a feature representation learning model with x as an input, and g will take the output of a(x, θ 1 ) and return the class scores f (x).

At a high level, DPAL has three key components: (1) DP a(x, θ 1 ), which is to preserve DP in learning the feature representation model a(x, θ 1 ); (2) DP Adversarial Learning, which focuses on preserving DP in adversarial learning, given DP a(x, θ 1 ); and (3) Provable Robustness and Verified Inferring, which are to compute robustness bounds given an input at the inference time.

In particular, given a deep neural network f with model parameters θ (Lines 2-3), the network is trained over T training steps.

In each step, a batch of m perturbed training examples and a batch of m DP adversarial examples derived from D are used to train our network Take a batch Bi ∈ B where i = t%(N/m), Assign Bt ← Bi 6:

Ensemble DP Adversarial Examples:

Take a batch Bi+1 ∈ B, Assign B adv t ← ∅ 9:

for l ∈ A do 10:

Take the next batch Ba ⊂ Bi+1 with the size m/|A| 11: ∀xj ∈ Ba: Craft x adv j by using attack algorithm

(θ2) with the noise

Output:

( 1 + 1/γx + 1/γ + 2)-DP parameters θ = {θ1, θ2}, robust model with an r budget 13: Verified Inferring: (an input x, attack size µa) 14: Compute robustness size (κ + ϕ)max in Eq. 15 of x 15: if (κ + ϕ)max ≥ µa then 16:

Return isRobust(x) = T rue, label k, (κ + ϕ)max 17: else 18:

Return isRobust(x) = F alse, label k, (κ + ϕ)max 3.1 DP FEATURE REPRESENTATION LEARNING Our idea is to use auto-encoder to simultaneously learn DP parameters θ 1 and ensure that the output of a(x, θ 1 ) is DP.

The reasons we choose an auto-encoder are: (1) It is easier to train, given its small size; and (2) It can be reused for different predictive models.

A typical data reconstruction function (cross-entropy), given a batch B t at the training step t of the input x i , is as follows:

where the transformation of x i is h i = θ T 1 x i , the hidden layer h 1 of a(x, θ 1 ) given the batch B t is denoted as h 1Bt = {θ T 1 x i } xi∈Bt , and x i = θ 1 h i is the reconstruction of x i .

To preserve 1 -DP in learning θ 1 where 1 is a privacy budget, we first derive the 1st-order polynomial approximation of R Bt (θ 1 ) by applying Taylor Expansion (Arfken, 1985) , denoted as R Bt (θ 1 ).

Then, Functional Mechanism (Zhang et al., 2012 ) is employed to inject noise into coefficients of the approximated function

where

, parameters θ 1j derived from the function optimization need to be 1 -DP.

To achieve that, Laplace noise

) is injected into coefficients 1 2 − x ij h i , where ∆ R is the sensitivity of R Bt (θ 1 ), as follows:

To ensure that the computation of x i does not access the original data, we further inject Laplace noise

) into x i .

This can be done as a preprocessing step for all the benign examples in D to construct a set of disjoint batches B of perturbed benign examples (Lines 2 and 5).

The perturbed function now becomes:

where

, and x i = θ 1 h i .

Let us denote β as the number of neurons in h 1 , and h i is bounded in [−1, 1], the global sensitivity ∆ R is as follows:

Lemma 2 The global sensitivity of R over any two neighboring batches, B t and B t , is as follows:

All the proofs are in our Appendix.

By setting ∆ R = d(β + 2), we show that the output of a(·), which is the perturbed affine transformation h 1Bt = {θ

and θ 1 1,1 is the maximum 1-norm of θ 1 's columns (Operator norm, 2018) .

This is important to tighten the privacy budget consumption in computing the remaining hidden layers g(a(x, θ 1 ), θ 2 ).

In fact, without using additional information from the original data, the computation of g(a(x, θ 1 ), θ 2 ) is also ( 1 /γ)-DP (the post-processing property of DP).

Similarly, we observe that the perturbation of a batch

Note that we do not use the post-processing property of DP to estimate the DP guarantee of h 1Bt based upon the DP guarantee of B t , since 1 /γ < 1 /γ x in practice.

As a result, the ( 1 /γ)-DP h 1Bt provides a more rigorous DP protection to the computation of g(·) and to the output layer.

Lemma 3 The computation of the affine transformation h 1Bt is ( 1 /γ)-DP and the computation of the batch B t as the input layer is ( 1 /γ x )-DP.

The following Theorem shows that optimizing R Bt (θ 1 ) is ( 1 /γ x + 1 )-DP in learning θ 1 given an ( 1 /γ x )-DP B t batch.

The optimization of R Bt (θ 1 ) preserves ( 1 /γ x + 1 )-DP in learning θ 1 .

To integrate adversarial learning, we first draft DP adversarial examples x adv j using perturbed benign examples x j , with an ensemble of attack algorithms A and a random perturbation budget µ t ∈ (0, 1], at each step t (Lines 6-11).

This will significantly enhances the robustness of our models under different types of adversarial examples with an unknown adversarial attack size µ.

with y(x j ) is the class prediction result of f (x j ) to avoid label leaking of the benign examples x j during the adversarial example crafting.

Given a set of DP adversarial examples B adv t , training the auto-encoder with B adv t preserves ( 1 /γ x + 1 )-DP.

The proof of Theorem 2 is in Appendix H, Result 4.

It can be extended to iterative attacks as

where y(x

.

Second, we propose a novel DP adversarial objective function L Bt (θ 2 ), in which the loss function L for benign examples is combined with an additional loss function Υ for DP adversarial examples, to optimize the parameters θ 2 .

The objective function L Bt (θ 2 ) is defined as follows:

where ξ is a hyper-parameter.

For the sake of clarity, in Eq. 10, we denote y i and y j as the true class labels y xi and y xj of examples x i and x j .

Note that x adv j and x j share the same label y xj .

Now we are ready to preserve DP in objective functions L f (x i , θ 2 ), y i and Υ f (x adv j , θ 2 ), y j in order to achieve DP in learning θ 2 .

Since the objective functions use the true class labels y i and y j , we need to protect the labels at the output layer.

Let us first present our approach to preserve DP in the objective function L for benign examples.

Given h πi computed from the x i through the network with W π is the parameter at the last hidden layer h π .

Cross-entropy function is approximated as

Based on the post-processing property of DP (Dwork & Roth, 2014) ,

.

As a result, the optimization of the function L 1Bt θ 2 does not disclose any information from the training data, and

1/γ , given neighboring batches B t and B t .

Thus, we only need to preserve 2 -DP in the function L 2Bt (θ 2 ), which accesses the ground-truth label y ik .

Given coefficients h πi y ik , the sensitivity ∆ L2 of L 2Bt (θ 2 ) is computed as:

Lemma 4 Let B t and B t be neighboring batches of benign examples, we have the following inequality: ∆ L2 ≤ 2|h π |, where |h π | is the number of hidden neurons in h π .

The sensitivity of our objective function is notably smaller than the state-of-the-art bound (Phan et al., 2017a) , which is crucial to improve our model utility.

The perturbed functions are as follows:

We apply the same technique to preserve

As the perturbed functions L and Υ are always optimized given two disjoint batches B t and B adv t , the privacy budget used to preserve DP in the adversarial objective function L Bt (θ 2 ) is ( 1 /γ + 2 ), following the parallel composition property of DP (Dwork & Roth, 2014) .

The total budget to learn private parameters

We have shown that our mechanism achieves DP at the batch level B t ∪B adv t given a specific training step t. By constructing disjoint and fixed batches from the training data D, we leverage both parallel composition and post-processing properties of DP (Dwork & Roth, 2014) to extend the result to ( 1 + 1 /γ x + 1 /γ + 2 )-DP in learning θ = {θ 1 , θ 2 } on D across T training steps.

There are three key properties in our approach: (1) It only reads perturbed inputs B t and perturbed coefficients h 1 , which are DP across T training steps; (2) Given N/m disjoint batches in each epoch, for any example x, x is included in one and only one batch, denoted B x ∈ B. As a result, the DP guarantee to x in D is equivalent to the DP guarantee to x in B x ; since the optimization using any other batches does not affect the DP guarantee of x; and (3) All the batches are fixed across T training steps to prevent additional privacy leakage, caused by generating new and overlapping batches (which are considered overlapping datasets in the parlance of DP) in the typical training approach.

Theorem 4 Algorithm 1 achieves ( 1 + 1 /γ x + 1 /γ + 2 )-DP parameters θ = {θ 1 , θ 2 } on the private training data D across T training steps.

Now, we establish the correlation between our mechanism and provable robustness.

In the inference time, to derive the provable robustness condition against adversarial examples x+α, i.e., ∀α ∈ l p (1), PixelDP mechanism randomizes the scoring function f (x) by injecting robustness noise σ r into either input x or a hidden layer, i.e., x = x + Lap(

, where ∆ x r and ∆ h r are the sensitivities of x and h, measuring how much x and h can be changed given the perturbation α ∈ l p (1) in the input x. Monte Carlo estimation of the expected valuesÊf (x),Ê lb f k (x), and E ub f k (x) are used to derive the robustness condition in Eq. 4.

On the other hand, in our mechanism, the privacy noise σ p includes Laplace noise injected into both input x, i.e.,

This helps us to avoid injecting the noise directly into the coefficients h πi y ik .

The correlation between our DP preservation and provable robustness lies in the correlation between the privacy noise σ p and the robustness noise σ r .

We can derive a robustness bound by projecting the privacy noise σ p on the scale of the robustness noise σ r .

Given the input x, let κ =

, in our mechanism we have that:

By applying a group privacy size κ (Dwork & Roth, 2014; Lecuyer et al., 2018) , the scoring function f (x) satisfies r -PixelDP given α ∈ l p (κ), or equivalently is κ r -PixelDP given α ∈ l p (1), δ r = 0.

By applying Lemma 1, we have ∀k, ∀α ∈ l p (κ) :

With that, we can achieve a robustness condition against l p (κ)-norm attacks, as follows:

with the probability ≥ η x -confidence, derived from the Monte Carlo estimation ofÊf (x).

Our mechanism also perturbs h (Eq. 7).

Given ϕ =

).

Therefore, the scoring function f (x) also satisfies r -PixelDP given the perturbation α ∈ l p (ϕ).

In addition to the robustness to the l p (κ)-norm attacks, we achieve an additional robustness bound in Eq. 12 against l p (ϕ)-norm attacks.

Similar to PixelDP, these robustness conditions can be achieved as randomization processes in the inference time.

They can be considered as two independent and provable defensive mechanisms applied against two l p -norm attacks, i.e., l p (κ) and l p (ϕ).

One challenging question here is: "What is the general robustness bound, given κ and ϕ?" Intuitively, our model is robust to attacks with α ∈ l p (κ + ϕ).

We leverage the theory of sequential composition in DP (Dwork & Roth, 2014) to theoretically answer this question.

Given S independent mechanisms M 1 , . . .

, M S , whose privacy guarantees are 1 , . . .

, S -DP with α ∈ l p (1).

Each mechanism M s , which takes the input x and outputs the value of f (x) with the Laplace noise only injected to randomize the layer s (i.e., no randomization at any other layers), denoted as f s (x), is defined as:

We aim to derive a generalized robustness of any composition scoring function f (M 1 , . . .

, M s |x) bounded in [0, 1], defined as follows:

Our setting follows the sequential composition in DP (Dwork & Roth, 2014) .

Thus, we can prove that the expected value Ef (M 1 , . . .

, M S |x) is insensitive to small perturbations α ∈ l p (1) in Lemma 5, and we derive our composition of robustness in Theorem 5, as follows:

Lemma 5 Given S independent mechanisms M 1 , . . .

, M S , which are 1 , . . .

, S -DP w.r.t a l p -norm metric, then the expected output value of any sequential function f of them, i.e., f (M 1 , . . .

, M S |x) ∈ [0, 1], meets the following property:

Theorem 5 (Composition of Robustness) Given S independent mechanisms M 1 , . . .

, M S .

Given any sequential function f (M 1 , . . . , M S |x), and letÊ lb andÊ ub are lower and upper bounds with an η-confidence, for the Monte Carlo estimation ofÊf

then the predicted label k = arg max kÊ f k (M 1 , . . .

, M S |x), is robust to adversarial examples x + α, ∀α ∈ l p (1), with probability ≥ η, by satisfying:

, which is the targeted robustness condition in Eq. 2.

It is worth noting that there is no η s -confidence for each mechanism s, since we do not estimate the expected valueÊf s (x) independently.

To apply the composition of robustness in our mechanism, the noise injections into the input x and its affine transformation h can be considered as two mechanisms M x and M h , sequentially applied as

with independent draws in the noise χ 2 , the noise χ 1 injected into x is fixed; and vice-versa.

By applying group privacy (Dwork & Roth, 2014) with sizes κ and ϕ, the scoring functions f

x (x) and f h (x), given M x and M h , are κ r -DP and ϕ r -DP given α ∈ l p (1).

With Theorem 5, we have a generalized bound as follows:

e., Eq. 14), then the predicted label k of our function f (M h , M x |x) is robust to perturbations α ∈ l p (κ + ϕ) with the probability ≥ η, by satisfying

Our model is trained similarly to training typical deep neural networks.

Parameters θ 1 and θ 2 are independently updated by applying gradient descent (Line 12).

Regarding the inference time, we implement a verified inference procedure as a post-processing step (Lines 13-18).

Our verified inference returns a robustness size guarantee for each example x, which is the maximal value of κ + ϕ, for which the robustness condition in Proposition 1 holds.

Maximizing κ + ϕ is equivalent to maximizing the robustness epsilon r , which is the only parameter controlling the size of κ + ϕ; since, all the other hyper-parameters, i.e., ∆ R , m, 1 , 2 , θ 1 , θ 2 , ∆ x r , and ∆ h r are fixed given a well-trained model f (x):

e., Eq. 14) (15) The prediction on an example x is robust to attacks up to (κ + ϕ) max .

The failure probability 1-η can be made arbitrarily small by increasing the number of invocations of f (x), with independent draws in the noise.

Similar to (Lecuyer et al., 2018 ), Hoeffding's inequality is applied to bound the approximation error inÊf k (x) and to search for the robustness bound (κ + ϕ) max .

We use the following sensitivity bounds ∆ h r = β θ 1 ∞ where θ 1 ∞ is the maximum 1-norm of θ 1 's rows, and ∆ x r = µd for l ∞ attacks.

We also propose a new way to draw independent noise following the distribution of χ 1 + /ψ) for the transformation h, where χ 1 and χ 2 are the fixed noise used to train the network, and ψ is a parameter to control the distribution shifts between training and inferring.

This new Monte Carlo Estimation of Ef (x) works better without affecting the DP bounds and the robustness (Appendix L).

We have conducted an extensive experiment on the MNIST and CIFAR-10 datasets.

We consider the class of l ∞ -bounded adversaries to see whether our mechanism could retain high model utility, while providing strong DP guarantees and protections against adversarial examples.

Baseline Approaches.

Our DPAL mechanism is evaluated in comparison with state-of-the-art mechanisms in: (1) DP-preserving algorithms in deep learning, i.e., DP-SGD (Abadi et al., 2016) , AdLM (Phan et al., 2017a); in (2) Provable robustness, i.e., PixelDP (Lecuyer et al., 2018) ; and in (3) DP-preserving algorithms with provable robustness, i.e., SecureSGD given heterogeneous noise (Phan et al., 2019) , and SecureSGD-AGM (Phan et al., 2019) given the Analytic Gaussian Mechanism (AGM) (Balle & Wang, 2018) .

To preserve DP, DP-SGD injects random noise into gradients of parameters, while AdLM is a Functional Mechanism-based approach.

PixelDP is one of the state-ofthe-art mechanisms providing provable robustness using DP bounds.

SecureSGD is a combination of PixelDP and DP-SGD with an advanced heterogeneous noise distribution; i.e., "more noise" is injected into "more vulnerable" latent features, to improve the robustness.

The baseline models share the same design in our experiment.

Four white-box attacks were used, including FGSM, I-FGSM, Momentum Iterative Method (MIM) (Dong et al., 2017) , and MadryEtAl (Madry et al., 2018) .

is equivalent to an attack size 2µ a = 0.6 in our setting.

The reason for using x ∈ [−1, 1] d is to achieve better model utility, while retaining the same global sensitivities to preserve DP, compared with x ∈ [0, 1] d .

Our model configurations are in Appendix M and our approximation error bound analysis is presented in Appendix N. As in (Lecuyer et al., 2018) , we apply two accuracy metrics:

|test| where |test| is the number of test cases, isCorrect(·) returns 1 if the model makes a correct prediction (else, returns 0), and isRobust(·) returns 1 if the robustness size is larger than a given attack size µ a (else, returns 0).

Our task of validation focuses on shedding light into the interplay among model utility, privacy loss, and robustness bounds, by learning 1) the impact of the privacy budget t = ( 1 + 1 /γ x + 1 /γ + 2 ), and 2) the impact of attack sizes µ a .

All statistical tests are 2-tail t-tests.

All experimental Figures are in Appendix O.

Results on the MNIST Dataset.

Figure 2 illustrates the conventional accuracy of each model as a function of the privacy budget t on the MNIST dataset under l ∞ (µ a )-norm attacks, with µ a = 0.2 (a pretty strong attack).

It is clear that our DPAL outperforms AdLM, DP-SGD, SecureSGD, and SecureSGD-AGM, in all cases, with p < 1.32e − 4.

On average, we register a 22.36% improvement over SecureSGD (p < 1.32e − 4), a 46.84% improvement over SecureSGD-AGM (p < 1.83e − 6), a 56.21% improvement over AdLM (p < 2.05e − 10), and a 77.26% improvement over DP-SGD (p < 5.20e − 14), given our DPAL mechanism.

AdLM and DP-SGD achieve the worst conventional accuracies.

There is no guarantee provided in AdLM and DP-SGD.

Thus, the accuracy of the AdLM and DPSGD algorithms seem to show no effect against adversarial examples, when the privacy budget is varied.

This is in contrast to our DPAL model, the SecureSGD model, and the SecureSGD-AGM model, whose accuracies are proportional to the privacy budget.

When the privacy budget t = 0.2 (a tight DP protection), there are significant drops, in terms of conventional accuracy, given the baseline approaches.

By contrast, our DPAL mechanism only shows a small degradation in the conventional accuracy (6.89%, from 89.59% to 82.7%), compared with a 37% drop in SecureSGD (from 78.64% to 41.64%), and a 32.89% drop in SecureSGD-AGM (from 44.1% to 11.2%) on average, when the privacy budget t goes from 2.0 to 0.2.

At t = 0.2, our DPAL mechanism achieves 82.7%, compared with 11.2% and 41.64% correspondingly for SecureSGD-AGM and SecureSGD.

This is an important result, showing the ability to offer tight DP protections under adversarial example attacks in our model, compared with existing algorithms.

• Figure 4 presents the conventional accuracy of each model as a function of the attack size µ a on the MNIST dataset, under a strong DP guarantee, t = 0.2.

It is clear that our DPAL mechanism outperforms the baseline approaches in all cases.

On average, our DPAL model improves 44.91% over SecureSGD (p < 7.43e − 31), a 61.13% over SecureSGD-AGM (p < 2.56e − 22), a 52.21% over AdLM (p < 2.81e − 23), and a 62.20% over DP-SGD (p < 2.57e − 22).

More importantly, our DPAL model is resistant to different adversarial example algorithms with different attack sizes.

When µ a ≥ 0.2, AdLM, DP-SGD, SecureSGD, and SecureSGD-AGM become defenseless.

We further register significantly drops in terms of accuracy, when µ a is increased from 0.05 (a weak attack) to 0.6 (a strong attack), i.e., 19.87% on average given our DPAL, across all attacks, compared with 27.76% (AdLM), 29.79% (DP-SGD), 34.14% (SecureSGD-AGM), and 17.07% (SecureSGD).

• Figure 6 demonstrates the certified accuracy as a function of µ a .

The privacy budget is set to 1.0, offering a reasonable privacy protection.

In PixelDP, the construction attack bound r is set to 0.1, which is a pretty reasonable defense.

With (small perturbation) µ a ≤ 0.2, PixelDP achieves better certified accuracies under all attacks; since PixelDP does not preserve DP to protect the training data, compared with other models.

Meanwhile, our DPAL model outperforms all the other models when µ a ≥ 0.3, indicating a stronger defense to more aggressive attacks.

More importantly, our DPAL has a consistent certified accuracy to different attacks given different attack sizes, compared with baseline approaches.

In fact, when µ a is increased from 0.05 to 0.6, our DPAL shows a small drop (11.88% on average, from 84.29%(µ a = 0.05) to 72.41%(µ a = 0.6)), compared with a huge drop of the PixelDP, i.e., from 94.19%(µ a = 0.05) to 9.08%(µ a = 0.6) on average under I-FGSM, MIM, and MadryEtAl attacks, and to 77.47%(µ a = 0.6) under FGSM attack.

Similarly, we also register significant drops in terms of certified accuracy for SecureSGD (78.74%, from 86.74% to 7.99%) and SecureSGD-AGM (81.97%, from 87.23% to 5.26%) on average.

This is promising.

Our key observations are as follows.

(1) Incorporating ensemble adversarial learning into DP preservation, with tightened sensitivity bounds and a random perturbation size µ t ∈ [0, 1] at each training step, does enhance the consistency, robustness, and accuracy of our model against different attack algorithms with different levels of perturbations.

(2) Our DPAL model outperforms baseline algorithms, including both DP-preserving and non-private approaches, in terms of conventional accuracy and certified accuracy in most of the cases.

It is clear that existing DP-preserving approaches have not been designed to withstand against adversarial examples.

Results on the CIFAR-10 Dataset further strengthen our observations.

In Figure 3 , our DPAL clearly outperforms baseline models in all cases (p < 6.17e−9), especially when the privacy budget is small ( t < 4), yielding strong privacy protections.

On average conventional accuracy, our DPAL mechanism has an improvement of 10.42% over SecureSGD (p < 2.59e − 7), an improvement of 14.08% over SecureSGD-AGM (p < 5.03e − 9), an improvement of 29.22% over AdLM (p < 5.28e − 26), and a 14.62% improvement over DP-SGD (p < 4.31e − 9).

When the privacy budget is increased from 2 to 10, the conventional accuracy of our DPAL model increases from 42.02% to 46.76%, showing a 4.74% improvement on average.

However, the conventional accuracy of our model under adversarial example attacks is still low, i.e., 44.22% on average given the privacy budget at 2.0.

This opens a long-term research avenue to achieve better robustness under strong privacy guarantees in adversarial learning.

• The accuracy of our model is consistent given different attacks with different adversarial perturbations µ a under a rigorous DP protection ( t = 2.0), compared with baseline approaches (Figure 5 ).

In fact, when the attack size µ a increases from 0.05 to 0.5, the conventional accuracies of the baseline approaches are remarkably reduced, i.e., a drop of 25.26% on average given the most effective baseline approach, SecureSGD.

Meanwhile, there is a much smaller degradation (4.79% on average) in terms of the conventional accuracy observed in our DPAL model.

Our model also achieves better accuracies compared with baseline approaches in all cases (p < 8.2e − 10).

Figure 7 further shows that our DPAL model is more accurate than baseline approaches (i.e., r is set to 0.1 in PixelDP) in terms of certified accuracy in all cases, with a tight privacy budget set to 2.0 (p < 2.04e − 18).

We register an improvement of 21.01% in our DPAL model given the certified accuracy over SecureSGD model, which is the most effective baseline approach (p < 2.04e − 18).

In this paper, we established a connection among DP preservation to protect the training data, adversarial learning, and provable robustness.

A sequential composition robustness theory was introduced to generalize robustness given any sequential and bounded function of independent defensive mechanisms.

An original DP-preserving mechanism was designed to address the trade-off among model utility, privacy loss, and robustness by tightening the global sensitivity bounds.

A new Monte Carlo Estimation was proposed to improve and stabilize the estimation of the robustness bounds; thus improving the certified accuracy under adversarial example attacks.

However, there are several limitations.

First, the accuracy of our model under adversarial example attacks is still very low.

Second, the mechanism scalability is dependent on the model structures.

Third, further study is needed to address the threats from adversarial examples crafted by unseen attack algorithms.

Fourth, in this study, our goal is to illustrate the difficulties in providing DP protections to the training data in adversarial learning with robustness bounds.

The problem is more challenging when working with complex and large networks, such as ResNet (He et al., 2015) , VGG16 (Zhang et al., 2015) , LSTM (Hochreiter & Schmidhuber, 1997) , and GAN (Goodfellow et al., 2014a) .

Fifth, there can be alternative approaches to draft and to use DP adversarial examples.

Addressing these limitations needs significant efforts from both research and practice communities.

A NOTATIONS AND TERMINOLOGIES

Function/model f that maps inputs x to a vector of scores f (x) = {f1(x), . . .

, fK (x)} yx ∈ y A single true class label of example x y(x) = max k∈K f k (x)

Predicted label for the example x given the function f x adv = x + α Adversarial example where α is the perturbation lp(µ) = {α ∈ R d : α p ≤ µ} The lp-norm ball of attack radius µ ( r , δr)

Robustness budget r and broken probability δr

The expected value of f k (x) E lb andÊ ub Lower and upper bounds of the expected valueÊf (x) =

Feature representation learning model with x and parameters θ1 Bt A batch of benign examples xi

Data reconstruction function given Bt in a(x, θ1)

The values of all hidden neurons in the hidden layer h1 of a(x, θ1) given the batch Bt RB t (θ1) and R B t (θ1)

Approximated and perturbed functions of RB t (θ1) xi and xi Perturbed and reconstructed inputs xi

Sensitivity of the approximated function RB t (θ1) h1B Sensitivities of x and h, given the perturbation α ∈ lp(1)

Privacy budget to protect the training data D (κ + ϕ)max Robustness size guarantee given an input x at the inference time B PSEUDO-CODE OF ADVERSARIAL TRAINING (KURAKIN ET AL., 2016B)

Given a loss function:

where m 1 and m 2 correspondingly are the numbers of examples in B t and B adv t at each training step.

Proof 1 Assume that B t and B t differ in the last tuple, x m (x m ).

Then,

Proof 2 Regarding the computation of h 1Bt = {θ

The sensitivity of a function h is defined as the maximum change in output, that can be generated by a change in the input (Lecuyer et al., 2018) .

Therefore, the global sensitivity of h 1 can be computed as follows:

following matrix norms (Operator norm, 2018): θ T 1 1,1 is the maximum 1-norm of θ 1 's columns.

By injecting Laplace noise Lap(

, and χ 2 drawn as a Laplace noise [Lap(

β , in our mechanism, the perturbed affine transformation h 1Bt is presented as:

This results in an ( 1 /γ)-DP affine transformation h 1Bt = {θ

Similarly, the perturbed inputs

where ∆ x is the sensitivity measuring the maximum change in the input layer that can be generated by a change in the batch B t and γ x = ∆ R m∆x .

Following (Lecuyer et al., 2018) , ∆ x can be computed as follows:

Consequently, Lemma 3 does hold.

Proof 3 Given χ 1 drawn as a Laplace noise [Lap(

d and χ 2 drawn as a Laplace noise

β , the perturbation of the coefficient φ ∈ Φ = { 1 2 h i , x i }, denoted as φ, can be rewritten as follows:

, we have that:

Consequently, the computation of R Bt (θ 1 ) preserves 1 -DP in Alg.

1.

In addition, the parameter optimization of R Bt (θ 1 ) only uses the perturbed data B t , which is ( 1 /γ x )-DP (Lemma 3), in the computations of h i , h i , x i , parameter gradients, and gradient descents at each step.

These operations do not access the original dataset B t ; therefore, they do not incur any additional information from the original data (the post-processing property in DP Dwork & Roth (2014) ).

As a result, the total privacy budget to learn the perturbed optimal parameters θ 1 in Alg.

1 is ( 1 /γ x + 1 )-DP.

Proof 4 Assume that B t and B t differ in the last tuple, and x m (x m ) be the last tuple in B t (B t ), we have that

Since y mk and y mk are one-hot encoding, we have that

Proof 5 Let B t and B t be neighboring batches of benign examples, and χ 3 drawn as Laplace noise [Lap(

|hπ| , the perturbations of the coefficients h πi y ik can be rewritten as:

Since all the coefficients are perturbed, and given ∆ L2 = 2|h π |, we have that

The computation of L 2Bt θ 2 preserves ( 1 /γ + 2 )-differential privacy.

The optimization of L 2Bt θ 2 does not access additional information from the original input x i ∈ B t .

Consequently, the optimal perturbed parameters θ 2 derived from L 2Bt θ 2 are ( 1 /γ + 2 )-DP.

Proof 6 First, we optimize for a single draw of noise during training (Line 3) and all the batches of perturbed benign examples are disjoint and fixed across epochs.

As a result, the computation of x i is equivalent to a data preprocessing step with DP, which does not incur any additional privacy budget consumption over T training steps (the post-processing property of DP) (Result 1).

That is different from repeatedly applying a DP mechanism on either the same or overlapping datasets causing the accumulation of the privacy budget.

Now, we show that our algorithm achieves DP at the dataset level D. Let us consider the computation of the first hidden layer, given any two neighboring datasets D and D differing at most one tuple x e ∈ D and x e ∈ D .

, we have that

By having disjoint and fixed batches, we have that:

From Eqs. 19, 20, and Lemma 3, we have that

Eqs. 20 and 21

As a result, the computation of h 1D is ( 1 /γ)-DP given the data D, since the Eq. 22 does hold for any tuple x e ∈ D. That is consistent with the parallel composition property of DP, in which batches can be considered disjoint datasets given h 1B as a DP mechanism (Dwork & Roth, 2014) .

This does hold across epochs, since batches B are disjoint and fixed among epochs.

At each training step t ∈ [1, T ], the computation of h 1Bt does not access the original data.

It only reads the perturbed batch of inputs B t , which is ( 1 /γ x )-DP (Lemma 3).

Following the post-processing property in DP (Dwork & Roth, 2014) , the computation of h 1Bt does not incur any additional information from the original data across T training steps.

Similarly, we show that the optimization of the function R Bt (θ 1 ) is ( 1 /γ x + 1 )-DP across T training steps.

As in Theorem 1 and Proof 3, we have that

, where B ∈ B. Given any two perturbed neighboring datasets D and D differing at most one tuple x e ∈ D and x e ∈ D :

From Eqs. 20, 23, and Theorem 1, we have that

Eqs. 23 and 24

As a result, the optimization of R D (θ 1 ) is ( 1 /γ x + 1 )-DP given the data D (which is 1 /γ x -DP (Lemma 3)), since the Eq. 25 does hold for any tuple x e ∈ D. This is consistent with the parallel composition property in DP (Dwork & Roth, 2014) , in which batches can be considered disjoint datasets and the optimization of the function on one batch does not affect the privacy guarantee in any other batch.

In addition, ∀t ∈ [1, T ], the optimization of R Bt (θ 1 ) does not use any additional information from the original data D. Consequently, the privacy budget is ( 1 /γ x + 1 ) across T training steps, following the post-processing property in DP (Dwork & Roth, 2014)

Similarly, we can also prove that optimizing the data reconstruction function R B adv t (θ 1 ) given the DP adversarial examples crafted in Eqs. 8 and 9, i.e., x adv j , is also ( 1 /γ x + 1 )-DP given t ∈ [1, T ] on the training data D. First, DP adversarial examples x adv j are crafted from perturbed benign examples x j .

As a result, the computation of the batch B adv t of DP adversarial examples is 1) ( 1 /γ x )-DP (the post-processing property of DP (Dwork & Roth, 2014) ), and 2) does not access the original data ∀t ∈ [1, T ].

In addition, the computation of h 1B adv t and the optimization of R B adv t (θ 1 ) correspondingly are 1 /γ-DP and 1 -DP.

In fact, the data reconstruction function R B adv t is presented as follows:

where h

, and x adv j = θ 1 h adv j .

The right summation component in Eq. 26 does not disclose any additional information, since the sign(·) function is computed from perturbed benign examples (the post-processing property in DP (Dwork & Roth, 2014) ).

Meanwhile, the left summation component has the same form with R Bt (θ 1 ) in Eq. 7.

Therefore, we can employ the Proof 3 in Theorem 1, by replacing the coefficients Φ = { In addition to the Result 4, by applying the same analysis in Result 3, we can further show that the optimization of R D adv (θ 1 ) is ( 1 /γ x + 1 )-DP given the DP adversarial examples D adv crafted using the data D across T training steps, since batches used to created DP adversarial examples are disjoint and fixed across epochs.

It is also straightforward to conduct the same analysis in Result 2, in order to prove that the computation of the first affine transformation h 1B

given the batch of DP adversarial examples B Regarding the output layer, the Algorithm 1 preserves ( 1 /γ + 2 )-DP in optimizing the adversarial objective function L Bt∪B adv t (θ 2 ) (Theorem 3).

We apply the same technique to preserve ( 1 /γ + 2 )-DP across T training steps given disjoint and fixed batches derived from the private training data D. In addition, as our objective functions R and L are always optimized given two disjoint batches B t and B adv t , the privacy budget used to preserve DP in these functions is ( 1 + 1 /γ + 2 ), following the parallel composition property in DP (Dwork & Roth, 2014)

With the Results 1-6, all the computations and optimizations in the Algorithm 1 are DP following the post-processing property in DP (Dwork & Roth, 2014) , by working on perturbed inputs and perturbed coefficients.

The crafting and utilizing processes of DP adversarial examples based on the perturbed benign examples do not disclose any additional information.

The optimization of our DP adversarial objective function at the output layer is DP to protect the ground-truth labels.

More importantly, the DP guarantee in learning given the whole dataset level D is equivalent to the DP guarantee in learning on disjoint and fixed batches across epochs.

Consequently, Algorithm 1 preserves ( 1 + 1 /γ x + 1 /γ + 2 )-DP in learning private parameters θ = {θ 1 , θ 2 } given the training data D across T training steps.

Note that the 1 /γ x is counted for the perturbation on the benign examples.

Theorem 4 does hold.

Proof 7 Thanks to the sequential composition theory in DP (Dwork & Roth, 2014) ,

As a result, we have

The sequential composition of the expected output is as:

Lemma 5 does hold.

Proof 8 ∀α ∈ l p (1), from Lemma 5, with probability ≥ η, we have that

In addition, we also have

Using the hypothesis (Eq. 14) and the first inequality (Eq. 27), we have that

Now, we apply the third inequality (Eq. 28), we have that

The Theorem 5 does hold.

Proof 9 ∀α ∈ l p (1), by applying Theorem 5, we havê

Furthermore, by applying group privacy, we have that

By applying Proof 8, it is straight to have

with probability ≥ η.

Proposition 1 does hold.

Recall that the Monte Carlo estimation is applied to estimate the expected valueÊf (x) = 1 n n f (x) n , where n is the number of invocations of f (x) with independent draws in the noise, i.e., ) in our case.

When 1 is small (indicating a strong privacy protection), it causes a notably large distribution shift between training and inference, given independent draws of the Laplace noise.

In fact, let us denote a single draw in the noise as

) used to train the function f (x), the model converges to the point that the noise χ 1 and 2χ 2 need to be correspondingly added into x and h in order to make correct predictions.

χ 1 can be approximated as Lap(χ 1 , ), where → 0.

It is clear that independent draws of the noise Lap(χ 1 , ).

These distribution shifts can also be large, when noise is large.

We have experienced that these distribution shifts in having independent draws of noise to estimatê Ef (x) can notably degrade the inference accuracy of the scoring function, when privacy budget 1 is small resulting in a large amount of noise injected to provide strong privacy guarantees.

To address this, one solution is to increase the number of invocations of f (x), i.e., n, to a huge number per prediction.

However, this is impractical in real-world scenarios.

We propose a novel way to draw independent noise following the distribution of χ 1 + /ψ) for the affine transformation h, where ψ is a hyper-parameter to control the distribution shifts.

This approach works well and does not affect the DP bounds and the provable robustness condition, since: (1) Our mechanism achieves both DP and provable robustness in the training process; and (2) It is clear thatÊf

is the n-th draw of the noise.

When n → ∞,Êf (x) will converge to 1 n n g a(x + χ 1 , θ 1 ) + 2χ 2 , θ 2 , which aligns well with the convergence point of the scoring function f (x).

Injecting χ 1 and 2χ 2 to x and h during the estimation ofÊf (x) yields better performance, without affecting the DP and the robustness bounds.

The MNIST database consists of handwritten digits (Lecun et al., 1998) .

Each example is a 28 × 28 size gray-level image.

The CIFAR-10 dataset consists of color images belonging to 10 classes, i.e., airplanes, dogs, etc.

The dataset is split into 50,000 training samples and 10,000 test samples (Krizhevsky & Hinton, 2009) .

The experiments were conducted on a single GPU, i.e., NVIDIA GTX TITAN X, 12 GB with 3,072 CUDA cores.

All the models share the same structure, consisting of 2 and 3 convolutional layers, respectively for MNIST and CIFAR-10 datasets.

Both fully-connected and convolution layers can be applied in the representation learning model a(x, θ 1 ).

Given convolution layer, the computation of each feature map needs to be DP; since each of them independently reads a local region of input neurons.

Therefore, the sensitivity ∆ R can be considered the maximal sensitivity given any single feature map in the first affine transformation layer.

In addition, each hidden neuron can only be used to reconstruct a unit patch of input units.

That results in d (Lemma 2) being the size of the unit patch connected to each hidden neuron, e.g., d = 9 given a 3 × 3 unit patch, and β is the number of hidden neurons in a feature map.

MNIST:

We used two convolutional layers (32 and 64 features).

Each hidden neuron connects with a 5x5 unit patch.

A fully-connected layer has 256 units.

The batch size m was set to 2,499, ξ = 1, ψ = 2.

I-FGSM, MIM, and MadryEtAl were used to draft l ∞ (µ) adversarial examples in training, with T µ = 10.

Learning rate t was set to 1e − 4.

Given a predefined total privacy budget t , 2 is set to be 0.1, and 1 is computed as: 1 = t− 2 (1+1/γ+1/γx) .

This will guarantee that ( 1 + 1 /γ x + 1 /γ + 2 ) = t .

∆ R = (14 2 + 2) × 25 and ∆ L2 = 2 × 256.

We used three convolutional layers (128, 128, and 256 features).

Each hidden neuron connects with a 4x4 unit patch in the first layer, and a 5x5 unit patch in other layers.

One fullyconnected layer has 256 neurons.

The batch size m was set to 1,851, ξ = 1.5, ψ = 10, and T µ = 3.

The ensemble of attacks A includes I-FGSM, MIM, and MadryEtAl.

We use data augmentation, including random crop, random flip, and random contrast.

Learning rate t was set to 5e − 2.

In the CIFAR-10 dataset, 2 is set to (1 + r/3.0) and 1 = (1 + 2r/3.0)/(1 + 1/γ + 1/γ x ), where r ≥ 0 is a ratio to control the total privacy budget t in our experiment.

For instance, given r = 0, we have

Computational Efficiency and Scalability.

In terms of computation efficiency, our mechanism does not consume any extra computational resources to train the model, compared with existing DP-preserving algorithms in deep learning (Phan et al., 2016; 2017b; a) .

The model invocations to approximate the robustness bounds can further be efficiently performed in a parallel process.

Regarding the scalability, with remarkably tightened global sensitivities, the impact of the size of deep neural networks in terms of the number of hidden layers and hidden neurons is significantly remedied, since 1) ∆ R and ∆ L2 are small, 2) we do not need to inject any noise into the computation of the network g(·), and 3) we do not redraw the noise in each training step t. In addition, our mechanism is not restricted to the type of activation functions.

That is similar to (Lecuyer et al., 2018; Phan et al., 2019) .

As a result, our mechanism has a great potential to be applied in larger deep neural networks using larger datasets.

Extensively investigating this property requires further study from both research and practice communities.

To compute how much error our polynomial approximation approaches (i.e., truncated Taylor expansions), R Bt (θ 1 ) (Eq. 6) and L Bt θ 2 , incur, we directly apply Lemma 4 in (Phan et al., 2016) , Lemma 3 in (Zhang et al., 2012) , and the well-known error bound results in (Apostol, 1967) .

Note that R Bt (θ 1 ) is the 1st-order Taylor series and L Bt θ 2 is the 2nd-order Taylor series.

Let us closely follow (Phan et al., 2016; Zhang et al., 2012; Apostol, 1967) to adapt their results into our scenario, as follows:

Given the truncated function R Bt (θ 1 ) = xi∈Bt θ 1j h i r , the average error of the approximation is bounded as

where θ 1 = arg min θ1 R Bt (θ 1 ), θ 1 = arg min θ1 R Bt (θ 1 ), L Bt (θ 2 ) is the original Taylor polynomial function of xi∈Bt L f (x i , θ 2 ), y i , θ 2 = arg min θ2 L Bt (θ 2 ), θ 2 = arg min θ2 L Bt (θ 2 ).

Proof 10 Let U = max θ1 R Bt (θ 1 ) − R Bt (θ 1 ) and S = min θ1 R Bt (θ 1 ) − R Bt (θ 1 ) .

We have that U ≥ R Bt ( θ 1 ) − R Bt ( θ 1 ) and ∀θ * 1 : S ≤ R Bt (θ * 1 ) − R Bt (θ * 1 ).

Therefore, we have

In addition, R Bt ( θ 1 ) − R Bt (θ * 1 ) ≤ 0, it is straightforward to have:

If U ≥ 0 and S ≤ 0 then we have:

Eq. 34 holds for every θ * 1 , including θ 1 .

Eq. 34 shows that the error incurred by truncating the Taylor series approximate function depends on the maximum and minimum values of R Bt (θ 1 ) − R Bt (θ 1 ).

This is consistent with (Phan et al., 2016; Zhang et al., 2012) .

To quantify the magnitude of the error, we rewrite R Bt (θ 1 ) − R Bt (θ 1 ) as:

where g 1j (x i , θ 1j ) = θ 1j h i and g 2j (x i , θ 1j ) = θ 1j h i .

By looking into the remainder of Taylor expansion for each j (i.e., following (Phan et al., 2016; Apostol, 1967) ), with z j ∈ [z lj − 1, z lj + 1], 1 |Bt| R Bt (θ 1j ) − R Bt (θ 1j ) must be in the interval .

This can be applied to the case of our autoencoder, as follows:

For the functions F 1j (z j ) = x ij log(1 + e −zj ) and F 2j (z j ) = (1 − x ij ) log(1 + e zj ), we have F

Consequently, Eq. 29 does hold.

Similarly, by looking into the remainder of Taylor expansion for each label k, Eq. 30 can be proved straightforwardly.

In fact, by using the 2nd-order Taylor series with K categories, we have that:

@highlight

Preserving Differential Privacy in Adversarial Learning with Provable Robustness to Adversarial Examples