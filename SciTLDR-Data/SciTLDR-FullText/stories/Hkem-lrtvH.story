Black-box adversarial attacks require a large number of attempts before finding successful adversarial examples that are visually indistinguishable from the original input.

Current approaches relying on substitute model training, gradient estimation or genetic algorithms often require an excessive number of queries.

Therefore, they are not suitable for real-world systems where the maximum query number is limited due to cost.

We propose a query-efficient black-box attack which uses Bayesian optimisation in combination with Bayesian model selection to optimise over the adversarial perturbation and the optimal degree of search space dimension reduction.

We demonstrate empirically that our method can achieve comparable success rates with 2-5 times fewer queries compared to previous state-of-the-art black-box attacks.

Deep learning algorithms are widely deployed in many real-world systems and are increasingly being used for tasks, ranging from identity verification , to financial services (Heaton et al., 2017) to autonomous driving (Bojarski et al., 2016) .

However, even the most accurate deep learning models can be easily deceived by perturbations which are visually imperceptible to the human eye (Szegedy et al., 2013; Carlini et al., 2016) .

The growing costs and risks associated with the potential model failures has led to the importance of studying adversarial attacks, both in assessing their robustness and their ability to detect such attacks.

In this paper we focus on highly practical adversarial attacks that fulfill the following two criteria.

First, the attack is designed for a black-box setting because in real-world examples, the attacker would normally have no knowledge of the target deep learning model and can only interact with the model by querying it.

Second, query efficiency is highly prioritised because in practical cases where the damage caused by the attack is high, the query budget available to the attacker will be highly limited due to the risk of being detected by the defence system or other high inherent costs (monetary or computational) of model evaluations.

Despite of the large array of adversarial attacks proposed in the literature, many of them are whitebox approaches that assume full access to the target model architecture and the ability of performing back-propagation to get gradient information (Moosavi-Dezfooli et al., 2016; Kurakin et al., 2016; Gu & Rigazio, 2014; Goodfellow et al., 2014; Carlini & Wagner, 2017) .

On the other hand, for black-box attacks, there are various techniques that have been used which do not require access to the model architecture.

One class of methods trains a white-box substitute model and attacks the target model with adversarial examples that successfully fool the substitute (Papernot et al., 2017) .

However, this type of method requires the availability of the original training data or large query data to train the substitute network and the performance is often limited by the mismatch between the substitute and the target models (Su et al., 2018) .

The second class of black-box attacks, which show better empirical performance than the substitute model approaches, numerically estimate the gradient of the target model by repeatedly querying it (Chen et al., 2017; Ilyas et al., 2018; Tu et al., 2018) and attack with the estimated gradient.

Although various techniques are employed to increase the query efficiency for the gradient estimation, they need an excessively large query budget to achieve a successful attack (Alzantot et al., 2018) .

Another line of work removes the need for gradient estimation and uses decision-based techniques (Brendel et al., 2017) or genetic algorithms (Alzantot et al., 2018) to generate adversarial examples.

One popular technique that is adopted by many black-box attacks (Chen et al., 2017; Alzantot et al., 2018; Tu et al., 2018) to significantly improve the query efficiency is to search for adversarial perturbations in a low-dimensional latent space (search dimensionality reduction).

However, learning the effective dimensionality of the latent search space can be challenging by itself, and has not been investigated by the prior works to the best of our knowledge.

In light of the above limitations, we propose a query efficient black-box attack that iteratively optimises over both the adversarial perturbation and the effective dimensionality of the latent search space.

Our main contributions are summarised as follows:

??? We introduce a novel gradient-free black-box attack method, BayesOpt attack, which uses Bayesian optimisation with Gaussian process surrogate models to find the effective adversarial example and is capable of dealing with high-dimensional image inputs.

??? We proposes a Bayesian technique which learns the optimal degree of search dimensionality reduction by harnessing our statistical surrogate and information from query data.

This technique can be incorporated naturally into our attack procedure, leading to efficient optimisation over both adversarial perturbation and the latent search space dimension.

??? We empirically demonstrate that under the L ??? constraint, our proposed attack method can achieve comparable success rate with about 2 to 5 times fewer model queries in comparison to current state-of-the-art query-efficient black box attack methods.

Most of the existing adversarial attacks (Moosavi-Dezfooli et al., 2016; Kurakin et al., 2016; Gu & Rigazio, 2014; Goodfellow et al., 2014; Carlini & Wagner, 2017) focus on whitebox settings, where the attacker can get full access to the target model and has complete knowledge of the architecture, weights and gradients to generate successful adversarial examples.

However, real-world systems are usually attacked in a black-box setting, where one has no knowledge about the model and can only observe the input-output correspondences by querying the model.

Here we give a brief overview of various existing black-box attacks.

Substitute model One class of black-box attacks uses data acquired from querying the target blackbox model to train a substitute model (Papernot et al., 2017) , which mimics the classification behaviour of the target model.

The adversary can then employ any white-box method to attack the fully observable substitute model and apply the successful adversarial example to the target model.

Such approaches rely on the transferability assumption that adversarial examples which are effective to the substitute model are also very likely to conceive the target model given their similar classification performance on the same data (Szegedy et al., 2013) .

Moreover, to provide training data for the substitute model, the adversary either requires information on the target model's training set, which is highly unrealistic in real-world applications, or needs to build a synthetic training set by querying the target model (Papernot et al., 2017) , which implies a large number of model evaluations and becomes hardly feasible for large models or complex datasets (Brendel et al., 2017) .

Gradient estimation An alternative to training a substitute model is to estimate the gradient via finite differences and use the estimated gradient information to produce attacks.

However, the naive coordinate-wise gradient estimation requires excessive queries to the target model (2 queries per coordinate per descent/attack step) and thus is not feasible for attacking models with high dimensional inputs (e.g. classifiers on ImageNet).

Chen et al. (2017) overcome this limitation by using stochastic coordinate gradient descent and selecting the batch of coordinates via importance sampling, introducing the state-of-the-art zeroth-order attack, ZOO, which achieves comparable attack success rate and perturbation costs as many white-box attacks.

Although ZOO makes it computationally tractable to perform black-box attack on high-dimensional image data, it still requires millions of queries to generate a successful adversarial example, making it impracticable for attacking real-world systems where model query can be expensive and the budget limited.

Improving on ZOO, AutoZOOM (Tu et al., 2018) significantly enhances the query efficiency by using random vectors to estimate the full gradient and adjusting the estimation parameter adaptively to trade off query efficiency vs. input perturbation cost.

More importantly, AutoZOOM shows the benefits of employing dimension reduction techniques in accelerating attacks (i.e. searching for adversarial perturbation in a low-dimensional latent space and decoding it back to the high-dimensional input space).

Parallel work by Ilyas et al. (2018) estimate the gradient through a modified version of natural evolution strategy which can be viewed as a finite-difference method over a random Gaussian basis.

The estimated gradient is then used with projected gradient descent, a white-box attack method, to generate adversarial examples.

Gradient-free optimisation As discussed, gradient-estimation approaches in general need an excessive number of queries to achieve successful attack.

Moreover, their dependence on the gradient information makes them less robust to defences which manipulate the gradients Brendel et al., 2017; Guo et al., 2017) .

Thus, truly gradient-free methods are more likely to bypass such defences.

One recent example, which has demonstrated state-of-the-art query efficiency, is GenAttack (Alzantot et al., 2018) .

GenAttack uses genetic algorithms to iteratively evolve a population of candidate adversarial examples.

Besides having an annealing scheme for mutation rate and range, GenAttack also adopts dimensional reduction techniques, similar to AutoZOOM, to improve the query efficiency.

In parallel to GenAttack, Brendel et al. (2017) introduce a decision-based attack, Boundary Attack, which only requires access to the final model decision.

Boundary Attack starts from a huge adversarial perturbation and then iteratively reduces the perturbation through a random walk along the decision boundary.

However, Boundary Attack takes about 72 times more queries than GenAttack to fool an undefended ImageNet model (Alzantot et al., 2018) .

Another recent approach introduced in Moon et al. (2019) reduces the search space to a discrete domain and subsequently uses combinatorial optimisation to find successful attacks, mostly focusing on the less challenging setting of untargeted attacks.

Finally, the prior works that use Bayesian optimisation for adversarial attacks (Suya et al., 2017; Zhao et al., 2019) only investigate the use of simple Gaussian process as the surrogate model and are limited in their performance.

The method proposed in (Suya et al., 2017) deals with the untargetted attack setting and only demonstrates the effectiveness of Bayesian optimisation in comparison to random search on a low-dimensional (d = 57) email attack task.

BO-ADMM proposed in (Zhao et al., 2019) works on image data but it applies Bayesian optimisation directly on the search space of image dimension to minimise the joint objective of attack loss and distortion loss.

Despite its query efficiency, BO-ADMM leads to poor-quality adversarial examples of large distortion loss.

We focus on the black-box attack setting, where the adversary has no knowledge about the network architecture, weights, gradient or training data of the target model f , and can only query the target model with an input x to observe its prediction scores on all C classes (i.e. f : (Tu et al., 2018; Alzantot et al., 2018) .

Moreover, we aim to perform targeted attacks, which is more challenging than untargeted attacks, subject to a constraint on the maximum change to any of the coordinates (i.e., a L ??? constraint) (Warde-Farley & Goodfellow, 2016; Alzantot et al., 2018) .

Specifically, targeted attacks refer to the case where given a valid input x origin of class t origin (i.e. arg max i???{1,...,C} f (x origin ) i = t origin ) and a target t = t origin , we aim to find an adversarial input x, which is close to x origin according to the L ??? ???norm, such that arg max i???{1,...,C} f (x) i = t. Untargeted adversarial attacks refer to the case that instead of classifying x origin as t origin , we try to find an input x so that arg max i???{1,...,C} f (x) i = t origin .

In our approach, we follow the convention to optimise over the perturbation ?? instead of the adversarial example x directly (Chen et al., 2017; Alzantot et al., 2018; Tu et al., 2018) .

Therefore, our problem can be formulated as:

Bayesian optimisation (BayesOpt) is a query-efficient approach to tackle global optimisation problems (Brochu et al., 2010) .

It is particularly useful when the objective function is a black-box and is very costly to evaluate.

There are 2 key components in BO: a statistical surrogate, such as a Gaussian process (GP) or a Bayesian neural network (BNN) which models the unknown objective, and an acquisition function ??(??) which is maximised to recommend the next query location by trading off exploitation and exploration.

The standard algorithm for BayesOpt is shown in Algorithm 3 in the Appendix.

4 BAYESOPT ATTACK

It has been shown that reducing the dimensionality of the search space in Equation (1) increases query efficiency significantly (Chen et al., 2017; Tu et al., 2018; Alzantot et al., 2018) .

Due to our focus on the small query regime where our surrogate model needs to be trained with a very small number of observation data, we adopt the previously suggested dimensionality reduction technique, bilinear resizing, to reduce the challenging problem of optimising over high-dimensional input space of x ??? R d to one over a relatively low-dimensional input space, setting x = x origin + g(??) where ?? ??? R Furthermore, we follow the approach of smoothing the discontinuous objective function in Equation (1), which has been found to be beneficial in previous work (Chen et al., 2017; Tu et al., 2018; Alzantot et al., 2018) .

Together with the dimensionality reduction, this leads to the following blackbox objective problem for our Bayesian optimisation:

where

We first use BayesOpt with a standard GP as the surrogate to solve for the black-box attack objective in the reduced input dimension in Equation (2).

The GP encodes our prior belief on the objective y :

which is specified by a mean function ?? and a kernel/covariance function k (we use the Matern-5/2 kernel in our work).

In our work, we normalise the objective function value and thus use a zeromean prior ??(??) = 0.

The predictive posterior distribution for y t at a test point ?? t conditioned on the observation data

where

where

The optimal GP hyper-parameters ?? * such as the length scales and variance of the kernel function k can be learnt by maximising the marginal likelihood p(D t???1 |??) which has analytic form as presented in Appendix F. For a detailed introduction on GPs, please refer to (Rasmussen, 2003) .

Based on the predictive posterior distribution, we construct the acquisition function ??(??|D t???1 ) to help select the next query point ?? t .

The acquisition function can be considered as a utility measure which balances the exploration and exploitation by giving higher utility to input regions where the functional value is high (high ?? y ) and where the model is very uncertain (high k y ).

The approach of using BayesOpt with a GP surrogate to attack the target model is described in Algorithm 1.

Update the surrogate model with D t 7: end for

Although techniques such as bilinear resizing are able to reduce the input dimension from the original image size (e.g. d = 3072 for CIFAR10 image) to a significantly lower dimension (e.g. d = 192), making the problem amenable to GP-based BO, the reduced search space for the adversarial attack is still considered very high dimensional for GP-based BayesOpt (which is usually applied on problems with d ??? 20).

There are two challenges in using BayesOpt for high dimensional problems.

The first is the curse of dimensionality in modelling the objective function.

When the unknown function is highdimensional, estimating it with non-parametric regression becomes very difficult because it is impossible to densely fill the input space with finite number of sample points, even if the sample size is very large (Gy??rfi et al., 2006) .

The second challenge is the computational difficulty in optimising the acquisition function.

The computation cost needed for optimising the acquisition function to within a desired accuracy grows exponentially with dimension (Kandasamy et al., 2015) .

We adopt the additive-GP model (Duvenaud et al., 2011; Kandasamy et al., 2015) to deal with the above-mentioned challenges associated with searching for adversarial perturbation in the high dimensional space.

The key assumption we make is that the objective can be decomposed into a sum of low-dimensional composite functions:

where

If we impose a GP prior for each y (j) , the prior for the overall objective y is also a GP:

The predictive posterior distribution for each subspace p(y

).

In our case, the exact decomposition (i.e. which input dimension belongs to which low-dimensional subspace) is unknown but we can learn it together with other GP hyperparameters by maximising marginal likelihood (Kandasamy et al., 2015) .

Note that in this case, the acquisition function is formulated based on p(y

for each subspace and is also optimised in the low-dimensional subspace, thus leading to much more efficient optimisation task.

The optimal perturbations in all the subspaces {?? (Aj ) * } M j=1 are then combined to give the next query point ?? t .

Generating the successful adversarial example x ??? R d by searching perturbation in a reduced dimension ?? ??? R d r has become a popular practice that leads to significant improvement in query efficiency (Chen et al., 2017; Tu et al., 2018; Alzantot et al., 2018) .

However, what the optimal d r is and how to decide it efficiently have not been investigated in previous work (Chen et al., 2017; Tu et al., 2018; Alzantot et al., 2018) .

As we shown empirically in Section 5.1, setting d r arbitrarily can lead to suboptimal attack performance in terms of query efficiency, attack success rate as well

2: Output:

The optimal reduced dimension d r * and the corresponding GP model 3: for j = 1, . . .

, N do 4:

Fit a GP model to D r is thus very important for adversarial attacks.

In this section, we propose a rigorous method, which is neatly compatible with our attack technique, to learn the optimal d r from the query information.

The optimal d r should be the one that both takes into consideration our prior knowledge on the discrete d r choices and at the same time best explain the observed query data.

Given that our BayesOpt attack uses a statistical surrogate (i.e. GP in our case) to model the unknown relation between the attack objective score y and the adversarial perturbation ??, this naturally translates to the criterion of maximising the posterior for d r : (Rasmussen, 2003) .

In most cases, our prior assumption is that we do not prefer one

The exact computation of the evidence term requires marginalisation over model hyper-parameters, which is intractable.

We approximate the integral with point estimates (i.e. marginal likelihood of our GP model):

.

For query efficiency, we project the same perturbation query data in the original image dimension

to different latent spaces to get corresponding low-dimensional training data sets for separate GP models.

The GP model that corresponds to d

r j ) of each GP surrogate.

The overall procedure for d r selection is described in Algorithm 2.

We would like to highlight that the use of the statistical surrogate in our BayesOpt attack approach enables us to naturally use the Bayesian model selection technique to learn the optimal d r that automatically enjoy the trade-off between data-fit quality and model complexity.

And we also show empirically in Section 5.3 that by automating and incorporating the learning of d r into our BayesOpt attack, we can gain higher success rate and query efficiency.

Other adversarial attacks methods can also use our proposed approach to decide d r but it would require the additional efforts of constructing statistical models to provide p(D t???1 |d r j ).

We empirically compare the performance of our BayesOpt attacks against the state-of-the-art blackbox methods such as ZOO (Chen et al., 2017) , AutoZOOM (Tu et al., 2018) and GenAttack (Alzantot et al., 2018 The target models that we attack follow the same architectures as that used in AutoZOOM and GenAttack; These are image classifiers for MNIST (a CNN with 99.5% test accuracy) and CIFAR10 (a CNN with 80% test accuracy).

Following the experiment design in (Tu et al., 2018) , we randomly select 50 correctly classified images from CIFAR10 test data and randomly select 7 correctly classified images from MNIST test data.

We then perform targeted attacks on these images.

Each selected image is attacked 9 times, targeting at all but its true class and this gives a total of 450 attack instances for CIFAR10 and 63 attack instances for MNIST.

We set ?? max = 0.3 for attacking MNIST and ?? max = 0.05 for CIFAR10, which are used in (Alzantot et al., 2018) .

We use the recommended parameter setting and their open sourced implementations for performing all competing attack methods (Chen et al., 2017; Tu et al., 2018 ; Alzantot et al., 2018).

We first empirically investigate the effect of the reduced dimensionality d r of the latent space in which we search for the adversarial perturbation.

We experiment with the GP-based BayesOpt attacks for the CIFAR10 classifier using reduced dimension of d r = {6 ?? 6 ?? 3, 8 ?? 8 ?? 3, 10 ?? 10 ?? 3, 12 ?? 12 ?? 3, 14 ?? 14 ?? 3, 16 ?? 16 ?? 3, 18 ?? 18 ?? 3} and perform targeted attacks on 5 target images with each image being attacked 9 times, leading to 45 attack instances.

We first investigate the attack success rate (ASR) achieved at different d r for all the 5 images.

The results on ASR out of the 9 targeted attacks for each of the 5 images, which are indicated by Image ID 1 to 5, are shown in the left subplot of Figure 2 .

It's evident that the d r which leads to highest attack success rate varies for different original images x origin .

We then examines how the d r affects the query efficiency and attack quality (average L 2 distance of the adversarial image from the original image) for the attack instances (e.g. make the classifier to mis-classify a airplane image as a cat) that GP-based BayesOpt can attack successfully at all d r .

We present the results on 4 attack instances of a airplane image in the middle and right subplots of Figure 2 .

We can see that even for the same original image and attack instances, varying d r can impact query efficiency and L 2 norm of the successful adversarial perturbation significantly.

For example, d r = 8 ?? 8 ?? 3 is most query efficient for attack instance 1 and 4 but is outperformed by other dimensions in attack instance 2 and 3.

Therefore, the importance of d r and the difficulty of finding the optimal d r for a specific target/image motivates us to derive our method for learning it automatically from the data.

As shown in the following sections, our d r learner does lead to more superior attack performance.

In this experiment, we limit the total query number to be 1000, which is slightly above the median query counts needed for GenAttack to make a successful attack on MNIST and CIFAR10 (Alzantot et al., 2018) .

We adopt the reduced search dimension recommended by AutoZOOM, d r = 14??14?? 3 for CIFAR10 and d r = 14??14??1 for MNIST.

For BayesOpt attack, each iteration requires 1 query to the objective function so we limit its iteration to 1000 and early terminate the BayesOpt attack algorithm when successful adversarial example is found.

AutoZOOM comprises 2 stages in their attack algorithm.

The first exploration stage aims to find the successful adversarial example.

Once a successful attack is found, it switches to the fine-tuning stage to reduce the perturbation cost (e.g. L 2 norm) of the successful attack.

We report its performance on the attack success rate and L 2 norm of adversarial perturbations after a budget of 1000 queries, which allows it to fine-tune the successful adversarial perturbations found.

Moreover, AutoZOOM uses L 2 norm to measure the perturbation costs but our method and GenAttack limit the search space via L ??? norm.

We observe that the successful attacks found by AutoZOOM incur much higher perturbation costs in terms of L ??? norm than GenAttack and our method.

Therefore, we only consider the final adversarial examples whose L ??? distances from the original images lie within [????? max , ?? max ] as the successful attacks 3 .

The results on MNIST in Table 1 show that all our BayesOpt attack can achieve a comparable success rate but at a much lower query count (71% less for simple GP-BO and 81% less for ADDGP-BO ) in comparison to GenAttack (Alzantot et al., 2018) and AutoZOOM (Tu et al., 2018) .

Note that GP-BO-auto-d r and ADDGP-BO) can achieve a attack success rate of 98% on MNIST(a) and 90% on CIFAR10 (b).

To achieve the same success rates, GenAttack(in purple) takes 1481 for MNIST and 4691 for CIFAR10 and AutoZOOM(in green) takes 1000 for MNIST and 3880 for CIFAR10.

ZOO's result of 0 query means that ZOO succeeds in attacking the 2 simplest image-target pairs at its first batch (batch size of 128) of adversarial perturbations but fails to make successful targeted attack on the other image-target pairs under the constraints.

As for results on attacking on CIFAR10, Table 2 shows that all our BayesOpt attack can achieve significantly higher success rate but again at a remarkably lower query counts than the existing blackbox approaches.

For example, our ADDGP-BO can achieve 18% higher success rate while using 37.4% less queries in terms of the median as compared to GenAttack.

In addition, our approaches also lead to better quality adversarial examples (Figure 1 ) which are closer to the original image than the benchmark methods as reflected by the lower average L 2 perturbation (20.5% less).

More importantly, this set of experiments also demonstrate the effectiveness of our Bayesian method for learning d r as GP-BO-auto-d r leads to 15% increase in attack success rate compared to GP-BO while maintaining the competitiveness in query efficiency and L 2 distance.

We finish by comparing the query efficiency, measuring the change in the attack success rate over query counts for all the methods.

We limit the query budget of our BayesOpt attacks to 1000 but let the competing methods to continue running until they achieve the same best attack success rate as our best BayesOpt attacks or exceeds a much higher limit (2000 queries for MNIST and 5000 queries for CIFAR10).

As shown in Figure 3 , BayesOpt attacks converge much faster to the high attack success rates than the other methods.

Specifically, ADDGP-BO takes only 584 queries to achieve the success rate of 98% for MNIST, which is 50% of the query count by AutoZOOM(1000) and 30% of that by GenAttack (1481).

As for CIFAR10, both ADDGP-BO and GP-BO-auto-d r takes around 890 queries to achieve a success rate of 87% which is 23% of the query count by AutoZOOM and 17% of that by GenAttack.

One point to note is that AutoZOOM appears to be slightly more query efficient than GenAttack in this set of experiments.

However, we need to bear in mind that although we limit the mean L inf norm of the adversarial perturbation found by AutoZOOM to ?? max , AutoZOOM still have the advantage of exploring beyond the ?? max for many dimensions.

We introduce a new black-box adversarial attack which leverages Bayesian optimisation to find successful adversarial perturbations with high query efficiency.

We also improve our attack by adopting an additive surrogate structure to ease the optimisation challenge over the typically high-dimensional task.

Moreover, we take full advantage of our statistical surrogate model and the available query data to learn the optimal degree of dimension reduction for the search space via Bayesian model selection.

In comparison to several existing black-box attack methods, our BayesOpt attacks can achieve high success rates with 2-5 times fewer queries while still producing adversarial examples that are closer to the original image (in terms of average L 2 distance).

We believe our BayesOpt attacks can be a competitive alternative for accessing the model robustness, especially in real-world applications where the available query budget is highly limited and the model evaluation is expensive or risky.

Here we present a generic algorithm for Bayesian optimisation.

: Attack objective value against BayesOpt iterations(query count) for using various BayesOpt methods to attack one CIFAR10 image of class label 9 (denoted by i).

Curves of different colours correspond to the 9 different target labels (denoted by t).

Convergence to 0 objective value indicates successful attack.

ADDGP-BO and GP-BO-auto-d r enjoy faster convergence and thus higher attack success rates than simple GP-BO.

We illustrate this via the case of attacking a CIFAR10 image of label 9(class truck) on the other 9 target labels with original label 9We plot the value of objective function (Equation 2), which is equal to the negative of attack loss, against the BayesOpt iterations/query counts.

We can see that our additive GP surrogate (ADDGP-BO) as well as the Bayesian learning of optimal d r (GP-BOauto-d r ) lead to faster convergence and thus higher attack success rate for this instance.

To learn the decomposition for the additive-GP surrogate in ADDGP-BO attack, we follow the approach proposed in (Kandasamy et al., 2015) to treat the decomposition as an additional hyperparameter and learn the optimal decomposition by maximising marginal likelihood.

However, exhaustive search over

4 possible decompositions is expensive.

We adopt a computationally cheap alternative by randomly selecting 20 decompositions and choosing the one with the largest marginal likelihood.

The method decomposition learning procedure is repeated every 40 BO iterations and we use M = 12 for CIFAR10.

We denote the method as ADDGP-BO-LD in this section but as ADDGP-BO in the rest of the paper.

We also experiment with another alternative way to learn the decomposition (ADDGP-BO-FD), which is similar to importance sampling in ZOO.

We group the pixels/dimensions together if the magnitude of average change in their pixel values over the past 5 BO iterations are closer (i.e. pixels that are subject to the most large adversarial perturbations are grouped together).

Again, we divide the dimensions into disjoint 12 groups as above to ensure fair comparison.

4 M is the number of subspaces and ds = |Aj| is the dimension of each subspaces 0 5001,000 2,000 3,000 4,000 5,000

Query counts The plots show the attack success rate of ADDGP-BO and GenAttack up to certain query counts.

Within a budget of 1797 queries, our proposed ADDGP-BO(in blue) can achieve a attack success rate of 78% on ImageNet.

To achieve the same success rates, GenAttack(in purple) takes 4711 queries, which is 2.6 times that of ADDGP-BO.

We compare the both types of decomposition learning methods using 20 randomly selected CI-FAR10 images, each of which is attacked on 9 other classes except its original class and a query budget of 1000.

The results are shown in Table 5 .

It is evident that learning decomposition by maximising marginal likelihood (ADDGP-BO-LD) can achieve higher attack success rate than pixelvalue-change-based decomposition learning (ADDGP-BO-FD) given the query budget.

To verify the feasibility/applicability of using BayesOpt to perform targeted attacks on ImageNet, we select 32 correctly classified images from the ImageNet and perform random targeted attacks with a query budget of 2000.

We adopt a hierarchical decoding process: 1) first performance BayesOpt(ADDGP-BO) on a reduced dimension of d Table 6 below.

Similar to the cases of MNIST and CIFAR10, ADDGP-BO achieves significantly higher (an increase of 59%) ASR while obtaining successful adversarial perturbations with lower average L 2 perturbation (14 % less) than GenAttack.

As shown in Figure 5 , ADDGP-BO only takes 1797 queries to achieve 78% ASR on these ImageNet images but GenAttack takes a budget of 4711 queries, which is 2.6 time than that of ADDGP-BO, to achieve the same ASR.

Gaussian processes (GPs) are popular models for inference in regression problems, especially when a quantification of the uncertainty around predictions is of relevance and little prior information is available to the inference problem.

These qualities, together with their analytical tractability, make them the most commonly used surrogate model in Bayesian optimisation.

A comprehensive overview on GPs can be found in Rasmussen & Williams (2006) .

Below we highlight the concepts of marginal likelihood and how to use it for hyperparameter optimisation and model selection based on the notation introduced in Section 4.2.

Hyperparameter tuning.

The key to finding the right hyperparameters ?? * in light of D t???1 in a principled way is given by the marginal likelihood in Equation (10).

where we have introduced a slightly augmented notation of K 1:t???1 ??? K 1:t???1 (??) to highlight the dependence on the kernel hyperparamaters ??.

In a truly Bayesian approach, one could shy away from fixing one set of hyperparameters and use Equation (10) to derive a posterior distribution of ?? based on one's prior beliefs about ?? expressed through some distribution p 0 (??).

However, as such an approach generally requires sampling using methods like Markov chain Monte Carlo (MCMC) due to intractability of integrals, in practice it is often easier and computationally cheaper to replace the fully Bayesian approach by a maximum likelihood approach and simply maximise the likelihood p(D t???1 |??) w.r.t.

??.

We follow this approach and perform a mix of grid-and gradient based search to maximise the logarithm of the r.h.s.

of Equation (10).

After evaluating a grid of 5 000 points, we start gradient ascent on each of the 5 most promising points using the following equation for the gradient (Rasmussen & Williams, 2006) to find the hyperparameters ?? * which maximise the marginal likelihood:

Model selection.

Once the optimal hyperparameters ?? * are found as described above, they can be plugged back into Equation (10) to choose amongst different models.

In Section 4.4 this is described for the case of choosing between different values of the reduced dimensionality d r .

@highlight

We propose a query-efficient black-box attack which uses Bayesian optimisation in combination with Bayesian model selection to optimise over the adversarial perturbation and the optimal degree of search space dimension reduction. 

@highlight

The authors propose to use Bayesian optimization with a GP surrogate for adversarial image generation, by exploiting additive structure and using Bayesian model selection to determine an optimal dimensionality reduction.