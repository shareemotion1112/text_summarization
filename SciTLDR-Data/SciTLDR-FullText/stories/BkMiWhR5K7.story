We study the problem of generating adversarial examples in a black-box setting in which only loss-oracle access to a model is available.

We introduce a framework that conceptually unifies much of the existing work on black-box attacks, and demonstrate that the current state-of-the-art methods are optimal in a natural sense.

Despite this optimality, we show how to improve black-box attacks by bringing a new element into the problem: gradient priors.

We give a bandit optimization-based algorithm that allows us to seamlessly integrate any such priors, and we explicitly identify and incorporate two examples.

The resulting methods use two to four times fewer queries and fail two to five times less than the current state-of-the-art.

The code for reproducing our work is available at https://git.io/fAjOJ.

Recent research has shown that neural networks exhibit significant vulnerability to adversarial examples, or slightly perturbed inputs designed to fool the network prediction.

This vulnerability is present in a wide range of settings, from situations in which inputs are fed directly to classifiers BID23 BID3 to highly variable real-world environments BID12 .

Researchers have developed a host of methods to construct such attacks BID7 BID17 BID2 BID15 , most of which correspond to first order (i.e., gradient based) methods.

These attacks turn out to be highly effective: in many cases, only a few gradient steps suffice to construct an adversarial perturbation.

A significant shortcoming of many of these attacks, however, is that they fundamentally rely on the white-box threat model.

That is, they crucially require direct access to the gradient of the classification loss of the attacked network.

In many real-world situations, expecting this kind of complete access is not realistic.

In such settings, an attacker can only issue classification queries to the targeted network, which corresponds to a more restrictive black box threat model.

Recent work BID4 BID1 ) provides a number of attacks for this threat model.

BID4 show how to use a basic primitive of zeroth order optimization, the finite difference method, to estimate the gradient from classification queries and then use it (in addition to a number of optimizations) to mount a gradient based attack.

The method indeed successfully constructs adversarial perturbations.

It comes, however, at the cost of introducing a significant overhead in terms of the number of queries needed.

For instance, attacking an ImageNet BID21 classifier requires hundreds of thousands of queries.

Subsequent work improves this dependence significantly, but still falls short of fully mitigating this issue (see Section 4.1 for a more detailed analysis).

We revisit zeroth-order optimization in the context of adversarial example generation, both from an empirical and theoretical perspective.

We propose a new approach for generating black-box adversarial examples, using bandit optimization in order to exploit prior information about the gradient, which we show is necessary to break through the optimality of current methods.

We Table 1 : Summary of effectiveness of 2 and ∞ ImageNet attacks on Inception v3 using NES, bandits with time prior (Bandits T ), and bandits with time and data-dependent priors (Bandits T D ).

Note that in the first column, the average number of queries is calculated only over successful attacks, and we enforce a query limit of 10,000 queries.

For purposes of direct comparison, the last column calculates the average number of queries used for only the images that NES (previous SOTA) was successful on.

Our most powerful attack uses 2-4 times fewer queries, and fails 2-5 times less often.

Avg

Adversarial examples are natural inputs to a machine learning system that have been carefully perturbed in order to induce misbehaviour of the system, under a constraint on the magnitude of the pertubation (under some metric).

For image classifiers, this misbehaviour can be either classification as a specific class other than the original one (the targeted attack) or misclassification (the untargeted attack).

For simplicity and to make the presentation of the overarching framework focused, in this paper we restrict our attention to the untargeted case.

Both our algorithms and the whole framework can be, however, easily adapted to the targeted setting.

Also, we consider the most standard threat model in which adversarial perturbations must have p -norm, for some fixed p, less than some p .

Suppose that we have some classifier C(x) with a corresponding classification loss function L(x, y), where x is some input and y its corresponding label.

In order to generate a misclassified input from some input-label pair (x, y), we want to find an adversarial example x which maximizes L(x , y) but still remains p -close to the original input.

We can thus formulate our adversarial attack problem as the following constrained optimization task: DISPLAYFORM0 First order methods tend to be very successful at solving the problem despite its non-convexity BID7 BID2 BID15 .

A first order method used as the backbone of some of the most powerful white-box adversarial attacks for p bounded adversaries is projected gradient descent (PGD).

This iterative method, given some input x and its correct label y, computes a perturbed input x k by applying k steps of the following update (with x 0 = x) DISPLAYFORM1 Here, Π S is the projection onto the set S, B p (x , ε ) is the p ball of radius ε around x , η is the step size, and ∂U is the boundary of a set U .

Also, as is standard in continuous optimization, we make s l be the projection of the gradient ∇ x L(x l−1 , y) at x l−1 onto the unit p ball.

This way we ensure that s l corresponds to the unit p -norm vector that has the largest inner product with ∇ x L(x l−1 , y).(Note that, in the case of the 2 -norm, s l is simply the normalized gradient but in the case of, e.g., the ∞ -norm, s l corresponds to the sign vector, sgn (∇ x L(x l−1 , y)) of the gradient.)

So, intuitively, the PGD update perturbs the input in the direction that (locally) increases the loss the most.

Observe that due to the projection in (1), x k is always a valid perturbation of x, as desired.

The projected gradient descent (PGD) method described above is designed to be used in the context of so-called white-box attacks.

That is, in the setting where the adversary has full access to the gradient ∇ x L(x, y) of the loss function of the attacked model.

In many practical scenarios, however, this kind of access is not available-in the corresponding, more realistic black-box setting, the adversary has only access to an oracle that returns for a given input (x, y), only the value of the loss L(x, y).One might expect that PGD is thus not useful in such black-box setting.

It turns out, however, that this intuition is incorrect.

Specifically, one can still estimate the gradient using only such value queries.(In fact, this kind of estimator is the backbone of so-called zeroth-order optimization frameworks BID22 .)

The most canonical primitive in this context is the finite difference method.

This method estimates the directional derivative DISPLAYFORM0 Here, the step size δ > 0 governs the quality of the gradient estimate.

Smaller δ gives more accurate estimates but also decreases reliability, due to precision and noise issues.

Consequently, in practice, δ is a tunable parameter.

Now, we can just use finite differences to construct an estimate of the gradient.

To this end, one can find the d components of the gradient by estimating the inner products of the gradient with all the standard basis vectors e 1 , . . .

, e d : DISPLAYFORM1 We can then easily implement the PGD attack (c.f.

(1)) using this estimator: DISPLAYFORM2 Indeed, BID4 were the first to use finite differences methods in this basic form to power PGD-based adversarial attack in the black-box setting.

This basic attack was shown to be successful but, since its query complexity is proportional to the dimension, its resulting query complexity was prohibitively large.

For example, the Inception v3 BID24 classifier on the ImageNet dataset has dimensionality d=268,203 and thus this method would require 268,204 queries.

(It is worth noting, however, that BID4 developed additional methods to, at least partially, reduce this query complexity.)

In the light of the above discussion, one can wonder if the algorithm (4) can be made more queryefficient.

A natural idea here would be to avoid fully estimating the gradient and rely instead only on its imperfect estimators.

This gives rise to the following question: How accurate of an gradient estimate is necessary to execute a successful PGD attack?We examine this question first in the simplest possible setting: one in which we only take a single PGD step (i.e., the case of k = 1).

Previous work BID7 indicates that such an 0% 5% 10% 15% 20% 25% 30% 35% 40% k percent of ImageNet coordinates 0.2 0.4 0.6 0.8 adversariality rate random-k top-k FIG1 : The fraction of correctly estimated coordinates of sgn(∇ x L(x, y)) required to successfully execute the single-step PGD (also known as FGSM) attack, with = 0.05.

In the experiment, for each k, the top k percent -chosen either by magnitude (top-k) or randomly (random-k) -of the signs of the coordinates are set correctly, and the rest are set to +1 or −1 at random.

The adversariality rate is the portion of 1,000 random ImageNet images misclassified after one FGSM step.

For example, estimating only 20% of coordinates correctly leads to misclassification for > 60% of images.attack can already be quite powerful.

So, we study how the effectiveness of this attack varies with gradient estimator accuracy.

Our experiments, shown in FIG1 , suggest that it is feasible to generate adversarial examples without estimating correctly even most of the coordinates of the gradient.

For example, in the context of ∞ attacks, setting a randomly selected 20% of the coordinates in the gradient to match the true gradient (and making the remaining coordinates have random sign) is sufficient to fool the classifier on more than 60% images with single-step PGD.

Our experiments thus demonstrate that an adversary is likely to be able to cause a misclassification by performing the iterated PGD attack, even when driven by a gradient estimate that is largely imperfect.

The above discussion makes it clear that successful attacks do not require a perfect gradient estimation, provided this estimate is suitably constructed.

It is still unclear, however, how to efficiently find this kind of imperfect but helpful estimator.

Continuous optimization methodology suggests that the key characteristic needed from our estimator is for it to have a sufficiently large inner product with the actual gradient.

We thus capture this challenge as the following gradient estimation problem: Definition 1 (Gradient estimation problem).

For an input/label pair (x, y) and a loss function L, let g * = ∇ x L(x, y) be the gradient of L at (x, y).

Then the goal of the gradient estimation problem is to find a unit vector g maximizing the inner product DISPLAYFORM0 from a limited number of (possibly adaptive) function value queries L(x , y ). (The expectation here is taken over the randomness of the estimation algorithm.)One useful perspective on the above gradient estimation problem stems from casting the recovery of g * in (5) as an underdetermined vector estimation task.

That is, one can view each execution of the finite difference method (see (2)) as computing an inner product query in which we obtain the value of the inner product of g * and some chosen direction vector A i .

Now, if we execute k such queries, and k < d (which is the regime we are interested in), the information acquired in this process can be expressed as the following (underdetermined) linear regression problem Ag * = y, where the rows of the matrix A correspond to the queries A 1 , . . .

, A k and the entries of the vector y gives us the corresponding inner product values.

Relation to compressive sensing.

The view of the gradient estimation problem we developed bears striking similarity to the compressive sensing setting BID5 .

Thus one might wonder if the toolkit of that area could be applied here.

Compressive sensing crucially requires, however, certain sparsity structure in the estimated signal (here, in the gradient g * ) and, to our knowledge, the loss gradients do not exhibit such a structure. (We discuss this further in Appendix B.)The least squares method.

In light of this, we turn our attention to another classical signal-processing method: norm-minimizing 2 least squares estimation.

This method approaches the estimation problem posed in (5) by casting it as an undetermined linear regression problem of the form Ag * = b, where we can choose the matrix A (the rows of A correspond to inner product queries with g * ).

Then, it obtains the solution g to the regression problem by solving: min DISPLAYFORM1 A reasonable choice for A (via BID11 and related results) is the distancepreserving random Gaussian projection matrix, i.e. A ij normally distributed.

The resulting algorithm turns out to yield solutions that are approximately those given by Natural Evolution Strategies (NES), which ) previously applied to black-box attacks.

In particular, in Appendix A, we prove the following theorem.

Theorem 1 (NES and Least Squares equivalence).

Letx N ES be the Gaussian k-query NES estimator of a d-dimensional gradient g and letx LSQ be the minimal-norm k-query least-squares estimator of g. For any p > 0, with probability at least 1 − p we have that DISPLAYFORM2 Note that when we work in the underdetermined setting, i.e., when k d (which is the setting we are interested in), the right hand side bound becomes vanishingly small.

Thus, the equivalence indeed holds.

In fact, using the precise statement (given and proved in Appendix A), we can show that Theorem 1 provides us with a non-vacuous equivalence bound.

Further, it turns out that one can exploit this equivalence to prove that the algorithm proposed in Ilyas et al. FORMULA1 is not only natural but optimal, as the least-squares estimate is an information-theoretically optimal gradient estimate in the regime where k = d, and an error-minimizing estimator in the regime where k << d. Theorem 2 (Least-squares optimality (Proof in Appendix A)).

For a linear regression problem y = Ag with known A and y, unknown g, and isotropic Gaussian errors, the least-squares estimator is finite-sample efficient, i.e. the minimum-variance unbiased (MVU) estimator of the latent vector g. Theorem 3 (Least-squares optimality (Proof in Meir FORMULA1 ).

In the underdetermined setting, i.e. when k << d, the minimum-norm least squares estimate (x LSQ in Theorem 1) is the minimumvariance (and thus minimum-error, since bias is fixed) estimator with no empirical loss.

The optimality of least squares strongly suggests that we have reached the limit of query-efficiency of black-box adversarial attacks.

But is this really the case?

Surprisingly, we show that an improvement is still possible.

The key observation is that the optimality we established of least-squares (and by Theorem 1, the NES approach in ) holds only for the most basic setting of the gradient estimation problem, a setting where we assume that the target gradient is a truly arbitrary and completely unknown vector.

However, in the context we care about this assumption does not hold -there is actually plenty of prior knowledge about the gradient available.

Firstly, the input with respect to which we compute the gradient is not arbitrary and exhibits locally predictable structure which is consequently reflected in the gradient.

Secondly, when performing iterative gradient attacks (e.g. PGD), the gradients used in successive iterations are likely to be heavily correlated.

The above observations motivate our focus on prior information as an integral element of the gradient estimation problem.

Specifically, we enhance Definition 1 by making its objective DISPLAYFORM0 , where I is prior information available to us.

This change in perspective gives rise to two important questions: does there exist prior information that can be useful to us?, and does there exist an algorithmic way to exploit this information?

We show that the answer to both of these questions is affirmative.

Consider a gradient ∇ x L(x, y) of the loss function corresponding to some input (x, y).

Does there exist some kind of prior that can be extracted from the dataset {x i }, in general, and the input (x, y) in particular, that can be used as a predictor of the gradient?

We demonstrate that it is indeed the case, and give two example classes of such priors.

Time-dependent priors.

The first class of priors we consider are time-dependent priors, a standard example of which is what we refer to as the "multi-step prior." We find that along the trajectory taken by estimated gradients, successive gradients are in fact heavily correlated.

We show this empirically by taking steps along the optimization path generated by running the NES estimator at each point, and plotting the normalized inner product (cosine similarity) between successive gradients, given by Figure 2 demonstrates that there indeed is a non-trivial correlation between successive gradientstypically, the gradients of successive steps (using step size from ) have a cosine similarity of about 0.9.

Successive gradients continue to correlate at higher step sizes: Appendix B shows that the trend continues even at step size 4.0 (a typical value for the total perturbation bound ε).

This indicates that there indeed is a potential gain from incorporating this correlation into our iterative optimization.

To utilize this gain, we intend to use the gradients at time t − 1 as a prior for the gradient at time t, where both the prior and the gradient estimate itself evolve over iterations.

DISPLAYFORM0 Data-dependent priors.

We find that the time-dependent prior discussed above is not the only type of prior one can exploit here.

Namely, we can also use the structure of the inputs themselves to reduce query complexity (in fact, the existence of such data-dependent priors is what makes machine learning successful in the first place).In the case of image classification, a simple and heavily exploited example of such a prior stems from the fact that images tend to exhibit a spatially local similarity (i.e. pixels that are close together tend to be similar).

We find that this similarity also extends to the gradients: specifically, whenever two coordinates (i, j) and DISPLAYFORM1 To corroborate and quantify this phenomenon, we compare ∇ x L(x, y) with an average-pooled, or "tiled", version (with "tile length" k) of the same signal.

An example of such an average-blurred gradient can be seen in Appendix B. More concretely, we apply to the gradient the mean pooling operation with kernel size (k, k, 1) and stride (k, k, 1), then upscale the spatial dimensions by k. We then measure the cosine similarity between the average-blurred gradient and the gradient itself.

Our results, shown in Figure 3 , demonstrate that the gradients of images are locally similar enough to allow for average-blurred gradients to maintain relatively high cosine similarity with the actual gradients, even when the tiles are large.

Our results suggest that we can reduce the dimensionality of our problem by a factor of k 2 (for reasonably large k) and still estimate a vector pointing close to the same direction as the original gradient.

This factor, as we show later, leads to significantly improved black-box adversarial attack performance.

Given the availability of these informative gradient priors, we now need a framework that enables us to easily incorporate these priors into our construction of black-box adversarial attacks.

Our proposed method builds on the framework of bandit optimization, a fundamental tool in online convex optimization BID9 .

In the bandit optimization framework, an agent plays a game that consists of a sequence of rounds.

In round t, the agent must choose a valid action, and then by playing the action incurs a loss given by a loss function t (·) that is unknown to the agent.

After playing the action, he/she only learns the loss that the chosen action incurs; the loss function is specific to the round t and may change arbitrarily between rounds.

The goal of the agent is to minimize the average loss incurred over all rounds, and the success of the agent is usually quantified by comparing the total loss incurred to that of the best expert in hindsight (the best single-action policy).

By the nature of this formulation, the rounds of this game can not be treated as independent -to perform well, the agent needs to keep track of some latent record that aggregates information learned over a sequence of rounds.

This latent record usually takes a form of a vector v t that is constrained to a specified (convex) set K. As we will see, this aspect of the bandit optimization framework will provide us with a convenient way to incorporate prior information into our gradient prediction.

An overview of gradient estimation with bandits.

We can cast the gradient estimation problem as an bandit optimization problem in a fairly direct manner.

Specifically, we let the action at each round t be a gradient estimate g t (based on our latent vector v t ), and the loss t correspond to the (negative) inner product between this prediction and the actual gradient.

Note that we will never have a direct access to this loss function t but we are able to evaluate its value on a particular prediction vector g t via the finite differences method (2) (which is all that the bandits optimization framework requires us to be able to do).Just as this choice of the loss function t allows us to quantify performance on the gradient estimation problem, the latent vector v t will allow us to algorithmically incorporate prior information into our predictions.

Looking at the two example priors we consider, the time-dependent prior will be reflected by carrying over the latent vector between the gradient estimations at different points.

Data-dependent priors will be captured by enforcing that our latent vector has a particular structure.

For the specific prior we quantify in the preceding section (data-dependent prior for images), we will simply reduce the dimensionality of the latent vector via average-pooling ("tiling"), removing the need for extra queries to discern components of the gradient that are spatially close.

We now describe our bandit framework for adversarial example generation in more detail.

Note that the algorithm is general and can be used to construct black-box adversarial examples where the perturbation is constrained to any convex set ( p -norm constraints being a special case).

We discuss the algorithm in its general form, and then provide versions explicitly applied to the 2 and ∞ cases.

As previously mentioned, the latent vector v t ∈ K serves as a prior on the gradient for the corresponding round t -in fact, we make our prediction g t be exactly v t projected onto the appropriate space, and thus we set K to be an extension of the space of valid adversarial perturbations (e.g. R n for 2 examples, [−1, 1] n for ∞ examples).

Our loss function t is defined as DISPLAYFORM0 for a given gradient estimate g, where we access this inner product via finite differences.

Here, L(x, y) is the classification loss on an image x with true class y.

The crucial element of our algorithm will thus be the method of updating the latent vector v t .

We will adapt here the canonical "reduction from bandit information" BID9 .

Specifically, our update procedure is parametrized by an estimator ∆ t of the gradient ∇ v t (v), and a first-order update step DISPLAYFORM1 , which maps the latent vector v t and the estimated gradient of t with respect to v t (which we denote ∆ t ) to a new latent vector v t+1 .

The resulting general algorithm is presented as Algorithm 1.In our setting, we make the estimator ∆ of the gradient −∇ v ∇L(x, y), v of the loss be the standard spherical gradient estimator (see BID9 ).

We take a two-query estimate of the expectation, and employ antithetic sampling which results in the estimate being computed as DISPLAYFORM2 Algorithm 1 Gradient Estimation with Bandit Optimization DISPLAYFORM3 for each round t = 1, . . .

, T do

//

Our loss in round t is t (g t ) = − ∇ x L(x, y init ), g t

g t ← v t−1 6: DISPLAYFORM0 where u is a Gaussian vector sampled from N (0, {q 1 , q 2 } ← {v + δu, v − δu} // Antithetic samples 4: DISPLAYFORM1 // Note that due to cancellations we can actually evaluate ∆ with only two queries to L

return ∆ A crucial point here is that the above gradient estimator ∆ t parameterizing the bandit reduction has no direct relation to the "gradient estimation problem" as defined in Section 2.4.

It is simply a general mechanism by which we can update the latent vector v t in bandit optimization.

It is the actions g t (equal to v t ) which provide proposed solutions to the gradient estimation problem from Section 2.4.The choice of the update rule A tends to be natural once the convex set K is known.

For K = R n , we can simply use gradient ascent: DISPLAYFORM0 and the exponentiated gradients (EG) update when the constraint is an ∞ bound (i.e. K = [−1, 1] n ): DISPLAYFORM1 Finally, in order to translate our gradient estimation algorithm into an efficient method for constructing black-box adversarial examples, we interleave our iterative gradient estimation algorithm with an iterative update of the image itself, using the boundary projection of g t in place of the gradient (c.f.

FORMULA1 ).

This results in a general, efficient, prior-exploiting algorithm for constructing black-box adversarial examples.

The resulting algorithm in the 2 -constrained case is shown in Algorithm 3.

We evaluate our bandit approach described in Section 3 and the natural evolutionary strategies (NES) approach of on their effectiveness in generating untargeted adversarial examples.

We consider both the 2 and ∞ threat models on the ImageNet BID21 dataset, in terms of success rate and query complexity.

We further investigate loss and gradient estimate quality over the optimization trajectory in each method.

To show the method extends to other datasets, DISPLAYFORM0 x 0 ← x init // Adversarial image to be constructed 5:while C(x) = y init do 6: DISPLAYFORM1 7: ∆ t ← GRAD-EST(x t−1 , y init , v t−1 ) // Estimated Gradient of t DISPLAYFORM2

v t ← v t−1 + η · ∆ t 10: DISPLAYFORM0 we also compare to NES in the CIFAR-∞ threat model; in all threat models, we show results on Inception-v3, Resnet-50, and VGG16 classifiers.

In evaluating our approach, we test both the bandit approach with time prior (Bandits T ), and our bandit approach with the given examples of both the data and time priors (Bandits T D ).

We use 10,000 and 1,000 randomly selected images (scaled to [0, 1]) to evaluate all approaches on ImageNet and CIFAR-10 respectively.

For NES, Bandits T , and Bandits T D we found hyperparameters (given in Appendix C, along with the experimental parameters) via grid search.

For ImageNet, we record the effectiveness of the different approaches in both threat models in Table 1 ( 2 and ∞ perturbation constraints), where we show the attack success rate and the mean number of queries (of the successful attacks) needed to generate an adversarial example for the Inception-v3 classifier (results for other classifiers in Appendix F).

For all attacks, we limit the attacker to at most 10,000 oracle queries.

As shown in Table 1 , our bandits framework with both data-dependent and time prior (Bandits T D ), is six and three times less failure-prone than the previous state of the art (NES ) in the ∞ and 2 settings, respectively.

Despite the higher success rate, our method actually uses around half as many queries as NES.

In particular, when restricted to the inputs on which NES is successful in generating adversarial examples, our attacks are 2.5 and 5 times as query-efficient for the ∞ and 2 settings, respectively.

In Appendix G, we also compare against the AutoZOOM method of BID25 , where we show that our Bandits T D method at a higher 100% success rate is over 6 times as query-efficient.

Finally, we also have similar results for CIFAR-10 under the ∞ threat model, which can be found in Appendix E.We also further quantify the performance of our methods in terms of black-box attacks, and gradient estimation.

Specifically, we first measure average queries per success after reaching a certain success rate (Figure 4a) , which indicates the dependence of the query count on the desired success rate.

The data shows that for any fixed success rate, our methods are more query-efficient than NES, and (due to the exponential trend) suggest that the difference may be amplified for higher success rates.

We then plot the loss of the classifier over time (averaged over all images), and performance on the gradient estimation problem for both ∞ and 2 cases (which, crucially, corresponds directly to the expectation we maximize in (7).

We show these three plots for ∞ in Figure 4 , and show the results for 2 (which are extremely similar) in Appendix D, along with CDFs showing the success of each method as a function of the query limit.

We find that on every metric in both threat models, our methods strictly dominate NES in terms of performance.

All known techniques for generating adversarial examples in the black-box setting so far rely on either iterative optimization schemes (our focus) or so-called substitute networks and transferability.

In the first line of work, algorithms use queries to gradually perturb a given input to maximize a corresponding loss, causing misclassification.

BID19 presented the first such iterative attack on a special class of binary classifiers.

Later, BID26 Figure 4 : (left) Average number of queries per successful image as a function of the number of total successful images; at any desired success rate, our methods use significantly less queries per successful image than NES, and the trend suggests that this gap increases with the desired success rate. (center) The loss over time, averaged over all images; (right) The correlation of the latent vector with the true gradient g, which is precisely the gradient estimation objective we define.real-world system with black-box attacks.

Specifically, they fool PDF document malware classifier by using a genetic algorithms-based attack.

Soon after, Narodytska & Kasiviswanathan (2017) described the first black-box attack on deep neural networks; the algorithm uses a greedy search algorithm that selectively changes individual pixel values.

BID4 were the first to design black-box attack based on finite-differences and gradient based optimization.

The method uses coordinate descent to attack black-box neural networks, and introduces various optimizations to decrease sample complexity.

Building on the work of BID4 , designed a black-box attack strategy that also uses finite differences but via natural evolution strategies (NES) to estimate the gradients.

They then used their algorithm as a primitive in attacks on more restricted threat models.

In a concurrent line of work, BID20 introduce a method for attacking models with so-called substitute networks.

Here, the attacker trains a model -called a substitute network -to mimic the target network's decisions (obtained with black-box queries) , then uses (white-box) adversarial examples for the substitute network to attack the original model.

Adversarial examples generated with these methods BID20 ; BID14 tend to transfer to a target MNIST or CIFAR classifier.

We note, however, that for attacking single inputs, the overall query efficiency of this type of methods tends to be worse than that of the gradient estimation based ones.

Substitute models are also thus far unable to make targeted black-box adversarial examples.

We develop a new, unifying perspective on black-box adversarial attacks.

This perspective casts the construction of such attacks as a gradient estimation problem.

We prove that a standard least-squares estimator both captures the existing state-of-the-art approaches to black-box adversarial attacks, and actually is, in a certain natural sense, an optimal solution to the problem.

We then break the barrier posed by this optimality by considering a previously unexplored aspect of the problem: the fact that there exists plenty of extra prior information about the gradient that one can exploit to mount a successful adversarial attack.

We identify two examples of such priors: a "time-dependent" prior that corresponds to similarity of the gradients evaluated at similar inputs, and a "data-dependent" prior derived from the latent structure present in the input space.

Finally, we develop a bandit optimization approach to black-box adversarial attacks that allows for a seamless integration of such priors.

The resulting framework significantly outperforms state-of-the-art by a factor of two to six in terms of success rate and query efficiency.

Our results thus open a new avenue towards finding priors for construction of even more efficient black-box adversarial attacks.

We thank Ludwig Schmidt for suggesting the connection between LSQ and NES.

AM supported in part by NSF grants CCF-1553428 and CNS-1815221.

LE supported in part by a Siebel Foundation Scholarship and IBM Watson AI grant.

AI supported by an Analog Devices Fellowship.

Theorem 1 (NES and Least Squares equivalence).

Letx N ES be the Gaussian k-query NES estimator of a d-dimensional gradient g and letx LSQ be the minimal-norm k-query least-squares estimator of g. For any p > 0, with probability at least 1 − p we have that DISPLAYFORM0 and in particular, DISPLAYFORM1 with probability at least 1 − p, where DISPLAYFORM2 Proof.

Let us first recall our estimation setup.

We have k query vectors δ i ∈ R d drawn from an i.i.d Gaussian distribution whose expected squared norm is one, i.e. δ i ∼ N (0, DISPLAYFORM3 Let the vector y ∈ R k denote the inner products of δ i s with the gradient, i.e. DISPLAYFORM4 We define the matrix A to be a k × d matrix with the δ i s being its rows.

That is, we have Ag = y. Now, recall that the closed forms of the two estimators we are interested in are given bŷ DISPLAYFORM5 which implies that DISPLAYFORM6 We can bound the difference between these two inner products as DISPLAYFORM7 Now, to bound the first term in (12), observe that DISPLAYFORM8 and thus DISPLAYFORM9 (Note that the first term in the above sum has been canceled out.)

This gives us that DISPLAYFORM10 as long as AA T − I ≤ 1 2 (which, as we will see, is indeed the case with high probability).

Our goal thus becomes bounding AA T − I = λ max (AA T − I), where λ max (·) denotes the largest (in absolute value) eigenvalue.

Observe that AA T and −I commute and are simultaneously diagonalizable.

As a result, for any 1 ≤ i ≤ k, we have that the i-th largest eigenvalue λ i (AA T − I) of AA T −

I can be written as DISPLAYFORM11 So, we need to bound DISPLAYFORM12 To this end, recall that E[AA T ] = I (since the rows of A are sampled from the distribution N (0, 1 d I)), and thus, by the covariance estimation theorem of BID6 (see Corollary 7.2) (and union bounding over the two relevant events), we have that DISPLAYFORM13 and thus DISPLAYFORM14 with probability at least 1 − k k+1 p. To bound the second term in (12), we note that all the vectors δ i are chosen independently of the vector g and each other.

So, if we consider the set {ĝ,δ 1 , . . .

,δ k } of k + 1 corresponding normalized directions, we have (see, e.g., BID8 ) that the probability that any two of them have the (absolute value of) their inner product be larger than some ε = 2 log(2(k+1)/p) d is at most DISPLAYFORM15 .On the other hand, we note that each δ i is a random vector sampled from the distribution N (0, DISPLAYFORM16 , so we have that (see, e.g., Lemma 1 in BID13 ), for any 1 ≤ i ≤ k and any ε > 0, DISPLAYFORM17 .Theorem 2 (Least-Squares Optimality).

For a fixed projection matrix A and under the following observation model of isotropic Gaussian noise: y = Ag + ε where ε ∼ N (0, εId), the least-squares estimator as in Theorem 1,x LSQ = A T (AA T ) −1 y is a finite-sample efficient (minimum-variance unbiased) estimator of the parameter g.

Proving the theorem requires an application of the Cramer-Rao Lower Bound theorem:Theorem 3 (Cramer-Rao Lower Bound).

Given a parameter θ, an observation distribution p(x; θ), and an unbiased estimatorθ that uses only samples from p(x; θ), then (subject to Fisher regularity conditions trivially satisfied by Gaussian distributions), DISPLAYFORM0 Now, note that the Cramer-Rao bound implies that if the variance of the estimatorθ is the inverse of the Fisher matrix,θ must be the minimum-variance unbiased estimator.

Recall the following form of the Fisher matrix: DISPLAYFORM1 Now, suppose we had the following equality, which we can then simplify using the preceding equation: DISPLAYFORM2 Multiplying the preceding by [I(θ)] −1 on both the left and right sides yields: DISPLAYFORM3 which tells us that (15) is a sufficient condition for finite-sample efficiency (minimal variance).

We show that this condition is satisfied in our case, where we have y ∼ Ag + ε,θ =x LSQ , and θ = g. We begin by computing the Fisher matrix directly, starting from the distribution of the samples y: DISPLAYFORM4 ∂ log p(y; g) DISPLAYFORM5 Using FORMULA1 , DISPLAYFORM6 Finally, note that we can write: DISPLAYFORM7 which concludes the proof, as we have shown thatx LSQ satisfies the condition (15), which in turn implies finite-sample efficiency.

Claim 1.

Applying the precise bound that we can derive from Theorem 1 on an ImageNet-sized dataset (d = 300000) and using k = 100 queries (what we use in our ∞ threat model and ten times that used for our 2 threat model), DISPLAYFORM8 For 10 queries, DISPLAYFORM9 B OMITTED FIGURES

Compressed sensing approaches can, in some cases, solve the optimization problem presented in Section 2.4.

However, these approaches require sparsity to improve over the least squares method.

Here we show the lack of sparsity in gradients through a classifier on a set of canonical bases for images.

In FIG4 , we plot the fraction of 2 weight accounted for by the largest k components in randomly chosen image gradients when using two canonical bases: standard and wavelet (db4).

While lack of sparsity in these bases does not strictly preclude the existence of a basis on which gradients are sparse, it suggests the lack of a fundamental structural sparsity in gradients through a convolutional neural network.

We show in FIG6 that the correlation between successive gradients on the NES trajectory are signficantly correlated, even at much higher step sizes (up to 2 norm of 4.0, which is a typical value for ε, the total adversarial perturbation bound and thus an absolute bound on step size).

This serves as further motivation for the time-dependent prior.

The average number of queries used per successful image for each method when reaching a specified success rate: we compare NES , Bandits T (our method with time prior only), and Bandits T D (our method with both data and time priors) and find that our methods strictly dominate NES-that is, for any desired sucess rate, our methods take strictly less queries per successful image than NES.

Here, we give results for the CIFAR-10 dataset, comparing our best method (Bandits T D ) and NES.

We train Inception-v3, ResNet-50, and VGG16 classifiers by fine-tuning the standard PyTorch ImageNet classifiers.

As such, all images are upsampled to 224 × 224 (299 × 299) for .

Just as for ImageNet, we use a maximum ∞ perturbation of 0.05, where images are scaled to [0, 1].

Table 5 : Summary of effectiveness of ∞ CIFAR10 attacks on Inception v3, ResNet-50, and VGG16 (I, R, V) using NES and bandits with time and data-dependent priors (Bandits T D ).

Note that in the first column, the average number of queries is calculated only over successful attacks, and we enforce a query limit of 10,000 queries.

For purposes of direct comparison, the last column calculates the average number of queries used for only the images that NES (previous SOTA) was successful on.

Our most powerful attack uses 2-4 times fewer queries, and fails 2-22 times less often.

Table 1 ), VGG16, and ResNet50 classifiers.

Note that we do not fine-tune the hyperparameters to the new classifiers, but simply use the hyperparameters found for Inception-v3.

Nevertheless, our best method consistently outperforms NES on black-box attacks.

Table 6 : Summary of effectiveness of ∞ and 2 ImageNet attacks on Inception v3, ResNet-50, and VGG16 (I, R, V) using NES and bandits with time and data-dependent priors (Bandits T D ).

Note that in the first column, the average number of queries is calculated only over successful attacks, and we enforce a query limit of 10,000 queries.

For purposes of direct comparison, the last column calculates the average number of queries used for only the images that NES (previous SOTA) was successful on.

Our most powerful attack uses 2-4 times fewer queries, and fails 2-5 times less often.

To compare with the method of BID25 , we consider the same classifier and dataset (Inceptionv3 and Imagenet) under the same 2 threat model.

Note that BID25 use mean rather than maximum 2 perturbation to evaluate their attacks (since the method is based on a Lagrangian relaxation).

To ensure a fair comparison we compare against the average number of queries to reach the adversarial examples bounded within a pertubation budget of 2 · 10 −4 , which is explicitly reported byTu et al. (2018) .For the bandits approach, we used Bandits T , (the bandits method with the time prior) and Bandits T D (the bandits method with both time and data prior) and run the methods until 100% success is reached.

We use the same hyperparameters from the untargeted ImageNet experiments (given in Appendix C).

Our findings, given in Table 7 show that our best method achieves an 100% success rate, and an over 6-fold reduction in queries.

Note that the method of BID25 achieves 100% success rate in general, but only constrains the mean 2 perturbation, and thus actually achieves a strictly less than 100% success rate with this perturbation threshold.

Table 7 : Comparison against coordinate-based query efficient finite differences attacks from BID25 , using the ImageNet dataset, with a maximum 2 constraint of 0.0002 per-pixel normalized (which is equal to a max-2 threshold reported by BID25 ).

For our methods (Bandits T and Bandits T D ) we use the same hyperparameters as in our comparison to NES, which are given in Appendix C.

Avg.

Queries Success Rate AutoZOOM-BiLin BID25 15,064 <100% AutoZOOM-AE BID25 14,914 <100% Bandits T (Ours) 4455 100% Bandits T D (Ours) 2297 100%

@highlight

We present a unifying view on black-box adversarial attacks as a gradient estimation problem, and then present a framework (based on bandits optimization) to integrate priors into gradient estimation, leading to significantly increased performance.