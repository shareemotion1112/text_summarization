In adversarial attacks to machine-learning classifiers, small perturbations are added to input that is correctly classified.

The perturbations yield adversarial examples, which are virtually indistinguishable from the unperturbed input, and yet are misclassified.

In standard neural networks used for deep learning, attackers can craft adversarial examples from most input to cause a misclassification of their choice.



We introduce a new type of network units, called RBFI units, whose non-linear structure makes them inherently resistant to adversarial attacks.

On permutation-invariant MNIST, in absence of adversarial attacks, networks using RBFI units match the performance of networks using sigmoid units, and are slightly below the accuracy of networks with ReLU units.

When subjected to adversarial attacks based on projected gradient descent or fast gradient-sign methods, networks with RBFI units retain accuracies above 75%, while ReLU or Sigmoid see their accuracies reduced to below 1%.

Further, RBFI networks trained on regular input either exceed or closely match the accuracy of sigmoid and ReLU network trained with the help of adversarial examples.



The non-linear structure of RBFI units makes them difficult to train using standard gradient descent.

We show that RBFI networks of RBFI units can be efficiently trained to high accuracies using pseudogradients, computed using functions especially crafted to facilitate learning instead of their true derivatives.

Machine learning via deep neural networks has been remarkably successful in a wide range of applications, from speech recognition to image classification and language processing.

While very successful, deep neural networks are affected by adversarial examples: small, especially crafter modifications of correctly classified input that are misclassified BID20 ).

The trouble with adversarial examples is twofold.

The modifications to regular input are so small as to be difficult or impossible to detect for a human: this has been shown both in the case of images BID20 ; BID14 ) and sounds BID9 ; BID5 ).

Further, the adversarial examples are in some measure transferable from one neural network to another BID7 ; BID14 ; BID16 ; BID22 ), so they can be crafted even without precise knowledge of the weights of the target neural network.

At a fundamental level, it is hard to provide guarantees about the behavior of a deep neural network, when every correctly classified input is tightly encircled by very similar, yet misclassified, inputs.

Thus far, the approach for obtaining neural networks that are more resistant to adversarial attacks has been to feed to the networks, as training data, an appropriate mix of the original training data, and adversarial examples BID7 ; BID12 ).

In training neural networks using adversarial examples, if the examples are generated via efficient heuristics such as the fast gradient sign method, the networks learn to associate the specific adversarial examples to the original input from which they were derived, in a phenomenon known as label leaking BID10 ; BID12 ; BID21 ).

This does not result in increased resistance to general adversarial attacks BID12 ; BID4 ).

If the adversarial examples used in training are generated via more general optimization techniques, as in BID12 ), networks with markedly increased resistance to adversarial attacks can be obtained, at the price of a more complex and computationally expensive training regime, and an increase in required network capacity.

We pursue here a different approach, proposing the use of neural network types that are, due to their structure, inherently impervious to adversarial attacks, even when trained on standard input only.

In BID7 ), the authors connect the presence of adversarial examples to the (local) linearity of neural networks.

In a purely linear form n i=1 x i w i , we can perturb each x i by , taking x i + if w i > 0, and x i − if w i < 0.

This causes an output perturbation of magnitude n i=1 |w i |, or nw forw the average modulus of w i .

When the number of inputs n is large, as is typical of deep neural networks, a small input perturbation can cause a large output change.

Of course, deep neural networks are not globally linear, but the insight of BID7 ) is that they may be sufficiently locally linear to allow adversarial attacks.

Following this insight, we develop networks composed of units that are highly non-linear.

The networks on which we settled after much experimentation are a variant of the well known radial basis functions (RBFs) BID0 ; BID6 BID15 ); we call our variant RBFI units.

RBFI units are similar to classical Gaussian RBFs, except for two differences that are crucial in obtaining both high network accuracy, and high resistance to attacks.

First, rather than being radially symmetrical, RBFIs can scale each input component individually; in particular, they can be highly sensitive to some inputs while ignoring others.

This gives an individual RBFI unit the ability to cover more of the input space than its symmetrical variants.

Further, the distance of an input from the center of the Gaussian is measured not in the Euclidean, or 2 , norm, but in the infinity norm ∞ , which is equal to the maximum of the differences of the individual components.

This eliminates all multi-input linearity from the local behavior of a RBFI: at any point, the output depends on one input only; the n in the above discussion is always 1 for RBFIs, so to say.

The "I" in RBFI stands for the infinity norm.

Using deeply nonlinear models is hardly a new idea, but the challenge has been that such models are typically difficult to train.

Indeed, we show that networks with RBFI units cannot be satisfactorily trained using gradient descent.

To get around this, we show that the networks can be trained efficiently, and to high accuracy, using pseudogradients.

A pseudogradient is computed just as an ordinary gradient, except that we artificially pretend that some functions have a derivative that is different from the true derivative, and especially crafted to facilitate training.

In particular, we use pseudoderivatives for the exponential function, and for the maximum operator, that enter the definition of Gaussian RBFI units.

Gaussians have very low derivative away from their center, which makes training difficult; our pseudoderivative artificially widens the region of detectable gradient around the Gaussian center.

The maximum operator appearing in the infinity norm has non-zero derivative only for one of its inputs at a time; we adopt a pseudogradient that propagates back the gradient to all of its inputs, according to their proximity in value to the maximum input.

Tampering with the gradient may seem unorthodox, but methods such as AdaDelta BID23 ), and even gradient descent with momentum, cause training to take a trajectory that does not follow pure gradient descent.

We simply go one step further, devising a scheme that operates at the granularity of the individual unit.

We show that with these two changes, RBFIs can be easily trained with standard random (pseudo)gradient descent methods, yielding networks that are both accurate, and resistant to attacks.

To conduct our experiments, we have implemented RBFI networks on top of the PyTorch framework BID18 ).

The code will be made available in a final version of the paper.

We consider permutation invariant MNIST, which is a version of MNIST in which the 28 × 28 pixel images are flattened into a one-dimensional vector of 784 values and fed as a feature vector to neural networks BID7 ).

On this test set, we show that for nets of 512,512,512,10 units, RBFI networks match the classification accuracy of networks of sigmoid units ((96.96 ± 0.14)% for RBFI vs. (96.88 ± 0.15)% for sigmoid), and are close to the performance of network with ReLU units ((98.62 ± 0.08)%).

When trained over standard training sets, RBFI networks retain accuracies over 75% for adversarial attacks that reduce the accuracy of ReLU and sigmoid networks to below 2% (worse than random).

We show that RBFI networks trained on normal input are superior to ReLU and sigmoid networks trained even with adversarial examples.

Our experimental results can be summarized as follows:• In absence of adversarial attacks, RBFI networks match the accuracy of sigmoid networks, and are slightly lower in accuracy than ReLU networks.• When networks are trained with regular input only, RBFI networks are markedly more resistant to adversarial attacks than sigmoid or ReLU networks.• In presence of adversarial attacks, RBFI networks trained on regualar input provide higher accuracy than sigmoid or ReLU networks, even when the latter are trained also on adversarial examples, and even when the adversarial examples are obtained via general projected gradient descent BID12 ).•

RBFI networks can be successfully trained with pseudogradients; the training via standard gradient descent yields instead markedly inferior results.• Appropriate regularization helps RBFI networks gain increased resistance to adversarial attacks.

Much work remains to be done, including experimenting with convolutional networks using RBFI units for images.

However, the results seem promising, in that RBFI seem to offer a viable alternative to current adversarial training regimes in achieving robustness to adversarial attacks.

Adversarial examples were first noticed in BID20 , where they were generated via the solution of general optimization problems.

In BID7 , a connection was established between linearity and adversarial attacks.

A fully linear form n i=1 x i w i can be perturbed by using x i + sign(w i ), generating an output change of magnitude · n i=1 |w i |.

In analogy, BID7 introduced the fast gradient sign method (FGSM) method of creating adversarial perturbations, by taking DISPLAYFORM0 , where ∇ i L is the loss gradient with respect to input i. The work also showed how adversarial examples are often transferable across networks, and it asked the question of whether it would be possible to construct non-linear structures, perhaps inspired by RBFs, that are less linear and are more robust to adversarial attacks.

This entire paper is essentially a long answer to the conjectures and suggestions expressed in BID7 .It was later discovered that training on adversarial examples generated via FGSM does not confer strong resistance to attacks, as the network learns to associate the specific examples generated by FGSM to the original training examples in a phenomenon known as label leaking BID10 ; BID12 ; BID21 .

The FGSM method for generating adversarial examples was extended to an iterative method, I-FGSM, in BID9 .

In BID21 , it is shown that using small random perturbations before applying FSGM enhances the robustness of the resulting network.

The network trained in BID21 using I-FSGM and ensemble method won the first round of the NIPS 2017 competition on defenses with respect to adversarial attacks.

Carlini and Wagner in a series of papers show that training regimes based on generating adversarial examples via simple heuristics, or combinations of these, in general fail to convey true resistance to attacks BID3 b) .

They further advocate measuring the resistance to attacks with respect to attacks found via more general optimization processes.

In particular, FGSM and I-FGSM rely on the local gradient, and training techniques that break the association between the local gradient and the location of adversarial examples makes networks harder to attack via FGSM and I-FGSM, without making the networks harder to attack via general optimization techniques.

In this paper, we follow this suggestion by using a general optimization method, projected gradient descent (PGD), to generate adversarial attacks and evaluate network robustness.

BID2 also shows that the technique of defensive distillation, which consists in appropriately training a neural network on the output of another BID17 , protects the networks from FGSM and I-FGSM attacks, but does not improve network resistance in the face of general adversarial attacks.

In BID12 it is shown that by training neural networks on adversarial examples generated via PGD, it is possible to obtain networks that are genuinely more resistant to adversarial examples.

The price to pay is a more computationally intensive training, and an increase in the network capacity required.

We provide an alternative way of reaching such resistance, one that does not rely on a new training regime.

In BID7 , the adversarial attacks are linked to the linearity of the models.

Following this insight, we seek to use units that do not exhibit a marked linear behavior, and specifically, units which yield small output variations for small variations of their inputs measured in infinity norm .

A linear form g(x) = i x i w i represents the norm-2 distance of the input vector x to a hyperplane perpendicular to vector w, scaled by |w| and its orientation.

It is not advantageous to simply replace this norm-2 distance with an infinity-norm distance, as the infinity-norm distance between a point and a plane is not a very useful concept.

It is preferable to consider the infinity-norm distance between points.

Hence, we define our units as variants of the classical Gaussian radial basis functions BID1 BID15 ).

We call our variant RBFI, to underline the fact that they are built using infinity norm.

An RBFI unit U(u, w) for an input in IR n is parameterized by two vectors of weights u = u 1 , . . . , u n and w = w 1 , . . . , w n Given an input x ∈ IR n , the unit produces output DISPLAYFORM0 where is the Hadamard, or element-wise, product.

In (1), the vector w is a point from which the distance to x is measured in infinity norm, and the vector u provides scaling factors for each coordinate.

Without loss of expressiveness, we require the scaling factors to be non-negative, that is, u i ≥ 0 for all 1 ≤ i ≤ n. The scaling factors provide the flexibility of disregarding some inputs x i , by having u i ≈ 0, while emphasizing the influence of other inputs.

Writing out (1) explicitly, we have: DISPLAYFORM1 The output of a RBFI unit is close to 1 only when x is close to w in the coordinates that have large scaling factors.

Thus, the unit is reminiscent of an And gate, with normal or complemented inputs, which outputs 1 only for one value of its inputs.

Logic circuits are composed both of And and of Or gates.

Thus, we introduce an Or RBFI unit by U OR (u, w) = 1 − U(u, w).

We construct neural networks out of RBFI units using layers consisting of And units, layers consisting of Or units, and mixed layers, in which the unit type is chosen at random at network initialization.

To form an intuitive idea of why networks with RBFI units might resist adversarial attacks, it is useful to compute the sensitivity of individual units to such attacks.

For x ∈ IR n and > 0, let B (x) = {x | x − x ∞ ≤ } be the set of inputs within distance from x in infinity norm.

Given a function f : IR n → IR, we call its sensitivity to adversarial attacks the quantity: DISPLAYFORM2 The sensitivity (3) represents the maximum change in output we can obtain via an input change within in infinity norm, as a multiple of itself.

For a single ReLU unit with weight vector w, the sensitivity is given by s = n i=1 |w i | = w 1 .

This formula can be understood by noting that the worst case for a ReLU unit corresponds to considering an input x for which the output is positive, and taking x i = x i + if w i > 0, and BID7 ).

Similarly, for a single sigmoid unit with weight vector w, we have s = 1 4 w 1 , where the factor of 1/4 corresponds to the maximum derivative of the sigmoid.

For a RBFI unit U(u, w), on the other hand, from (1) we have: DISPLAYFORM3 DISPLAYFORM4 ∞ .

Thus, the sensitivity of ReLU and Sigmoid units increases linearly with input size, whereas the sensitivity of RBFI units is essentially constant with respect to input size.

These formulas can be extended to bounds for whole networks.

For a ReLU network with K 0 inputs and layers of DISPLAYFORM5 where w (k) ij is the weight for input i of unit j of layer k, for 1 ≤ k ≤ K M .

We can compute an upper boundŝ for the sensitivity of the network via: DISPLAYFORM6 The formula for Sigmoid networks is identical except for the 1/4 factors.

Using similar notation, for RBFI networks we have: DISPLAYFORM7 By connecting in a simple way the sensitivity to attacks to the network weights, these formulas suggest the possibility of using weight regularization to achieve robustness: by adding cŝ to the loss function for c > 0, we might be able to train networks that are both accurate and robust to attacks.

We will show in Section 6.5 that such a regularization helps train more robust RBFI networks, but it does not help train more robust ReLU networks.

The non-linearities in (2) make neural networks containing RBFI units difficult to train using standard gradient descent, as we will show experimentally.

The problem lies in the shape of Gaussian functions.

Far from its peak for x = w, a function of the form (2) is rather flat, and its derivative may not be large enough to cause the vector of weights w to move towards useful places in the input space during training.

To obtain networks that are easy to train, we replace the derivatives for exp and max with alternate functions, which we call pseudoderivatives.

These pseudoderivatives are then used in the chain-rule computation of the loss gradient in lieu of the true derivatives, yielding a pseudogradient.

Exponential function.

In computing the partial derivatives of (1) via the chain rule, the first step consists in computing d dz e −z , which is of course equal to −e −z .

The problem is that −e −z is very close to 0 when z is large, and z in (2) is u (x − w) 2 ∞ , which can be large.

Hence, in the chain-rule computation of the gradient, we replace −e −z with the alternate "pseudoderivative" −1/ √ 1 + z, which has a much longer tail.

Max.

The gradient of y = max 1≤i≤n z i , of course, is given by ∂y ∂zi = 1 if z i = y, and ∂y ∂zi = 0 if z i < y.

The problem is that this transmits feedback only to the largest input(s).

This slows down training and can create instabilities.

We use as pseudoderivative e zi−y , so that some of the feedback is transmitted to inputs z i that approach y.

One may be concerned that by using the loss pseudogradient as the basis of optimization, rather than the true loss gradient, we may converge to solutions where the pseudogradient is null, and yet, we are not at a minimum of the loss function.

This can indeed happen.

We experimented with switching to training with true gradients once the pseudogradients failed to yield improvements; this increased the accuracy on the training set, but barely improved it on the testing set.

It is conceivable that more sophisticated ways of mixing training with regular and pseudo-gradients would allow training RBFI networks to higher accuracy on the testing set.

Given a correctly classified input x for a network, and a perturbation size > 0, an input x is an adversarial example for if x is misclassified, and x − x ∞ ≤ η.

Consider a network trained with cost function J(θ, x, y), where θ is the set of network parameters, x is the input, and y is the output.

Indicate with ∇ x J(θ, x , y) the gradient of J wrt its input x computed at values x of the inputs, parameters θ, and output y. For each input x belonging to the testing set, given a perturbation amount > 0, we produce adversarial examplesx with x −x ∞ ≤ using the following techniques.

Fast Gradient Sign Method (FGSM) BID7 ).

If the cost were linear around x, the optimal -max-norm perturbation of the input would be given by sign(∇ x J(θ, x, y) ).

This suggests taking as adversarial example: BID9 ).

Instead of computing a single perturbation of size using the sign of the gradient, we apply M perturbations of size /M , each computed from the endpoint of the previous one.

Precisely, the attack computes a sequencẽ x 0 ,x 1 , . . .

,x M , wherex 0 = x, and where eachx i+1 is obtained, for 0 ≤ i < M , by: DISPLAYFORM0

We then takex =x M as our adversarial example.

This attack is more powerful than its singlestep version, as the direction of the perturbation can better adapt to non-linear cost gradients in the neighborhood of x BID9 ).Projected Gradient Descent (PGD) BID12 ).

For an input x ∈ IR n and a given maximum perturbation size > 0, we consider the set B (x) ∩ [0, 1] n of valid inputs around x, and we perform projected gradient descent (PGD) in B (x) ∩ [0, 1] n of the negative loss with which the network has been trained (or, equivalently, projected gradient ascent wrt.

the loss).

By following the gradient in the direction of increasing loss, we aim at finding mis-classified inputs in B (x)∩ [0, 1] n .

As the gradient is non-linear, to check for the existence of adversarial attacks we perform the descent multiple times, each time starting from a point of B (x) ∩ [0, 1] n chosen uniformly at random.

Noise.

In addition to the above adversarial examples, we will study the robustness of our networks by feeding them inputs affected by noise.

For a testing input x and a noise amount ∈ [0, 1], we produce an -noisy versionx viax = (1 − )x + χ, where χ is a random element of the input space, which for MNIST is [0, 1] n .

We have implemented FGSM, I-FGSM, and PGD attacks for RBFI both relying on standard gradients, and relying on pseudogradients.

In the results, we denote pseudogradient-based results via RBFI [psd] .

The idea is that if pseudogradients are useful in training, they are likely to be useful also in attacking the networks, and an adversary may well rely on them.

BID4 show that many networks that resist FGSM and I-FGSM attacks can still be attacked by using general optimization-based methods.

Thus, they and argue that the evaluation of attack resistance should include general optimization methods; the PGD attacks we consider are an example of such methods.

6.1 EXPERIMENTAL SETUP Implementation.

We implemented RBFI networks in the PyTorch framework BID18 ).

In order to extend PyTorch with a new function f , it is necessary to specify the function behavior f (x), and the function gradient ∇ x f .

To implement RBFI, we extend PyTorch with two new functions: a LargeAttractorExp function, with forward behavior e −x and backward gradient propagation according to −1/ √ 1 + x, and SharedFeedbackMax, with forward behavior y = max n i=1 x i and backward gradient propagation according to e xi−y .

These two functions are used in the definition of RBFI units, as per FORMULA2 , with the AutoGrad mechanism of PyTorch providing backward (pseudo)gradient propagation for the complete networks.

Dataset.

We use the MNIST dataset BID11 ) for our experiments, following the standard setup of 60,000 training examples and 10,000 testing examples.

Each digit image was flattened to a one-dimensional feature vector of length 28 × 28 = 784, and fed to a fully-connected neural network; this is the so-called permutation-invariant MNIST.Neural networks.

We compared the accuracy of the following fully-connected network structures.• ReLU networks BID13 ) whose output is fed into a softmax, and the network is trained via cross-entropy loss.

Table 1 : Performance of 512-512-512-10 networks for MNIST testing input, and for input corrupted by adversarial attacks and noise computed with perturbation size = 0.3.• Sigmoid networks trained with square-error loss.• RBFI networks, trained using square-error loss.

For a RBFI network with m layers, we denote its type as RBFI(K 1 , . . .

, K m | t 1 , . . . , t m ), where K 1 , . . . , K m are the numbers of units in each layer, and where the units in layer i are And units if t i = ∧, Or units if t i = ∨, and are a random mix of And and Or units if t m = * .Square-error loss worked as well or better than other loss functions for Sigmoid and RBFI networks.

Unless otherwise noted, we use networks with layers 512, 512, 512, and 10 units, and in case of RBFI networks, we used geometry RBFI(512, 512, 512, 10 | ∧, ∨, ∧, ∨).

For RBFI networks we use a bound of [0.01, 3] for the components of the u-vectors, and of [0, 1] for the w-vectors, the latter corresponding to the value range of MNIST pixels.

We experimented with RBFI networks with various geometries, and we found the performance differences to be rather small, for reasons we do not yet fully understand.

We trained all networks with the AdaDelta optimizer BID23 ), which yielded good results for all networks considered.

Attacks.

We applied FGSM, I-FGSM, and noise attacks to the whole test set.

In I-FGSM attacks, we performed 10 iterations of (7).

As PGD attacks are considerably more computationally intensive, we apply them to one run only, and we compute the performance under PGD attacks for the first 5,000 examples in the test set.

For each input x in the test set, we perform 20 searches, or restarts.

In each search, we start from a random point in B (x) and we perform 100 steps of projected gradient descent using the AdaDelta algorithm to tune step size; if at any step a misclassified example is generated, the attack is considered successful.

In Table 1 we summarize the results on the accuracy and resistance to adversarial examples for networks trained on the standard MNIST training set.

The results are computed from 10 training runs for ReLU and Sigmoid networks, and from 5 runs for RBFI and RBFI [psd] .

In each run we used different seeds for the random generator used for weight initialization; each run consisted of 30 training epochs.

In a result of the form a ± e, a is the percentage accuracy, and e is the standard deviation in the accuracy of the individual runs.

In absence of perturbations, RBFI networks lose (1.66 ± 0.21)% performance compared to ReLU networks (from (98.62±0.07)% to (96.96±0.14)%), and perform comparably to sigmoid networks (the difference is below the standard deviation of the results).

When perturbations are present, in the form of adversarial attacks or noise, the performance of RBFI networks is superior.

We note that the FGSM and I-FGSM attacks performed using regular gradients are not effective against RBFI networks.

This phenomenon is called gradient masking: the gradient in proximity of valid inputs offers little information about the possible location of adversarial examples BID4 .

Pseudogradients do avoid gradient masking, and indeed the most effective attack against RBFI networks is I-FGSM performed using pseudogradients, which lowers the accuracy to (78.92 ± 1.91)% for = 0.3.

Including adversarial examples in the training set is the most common method used to make neural networks more resistant to adversarial attacks BID7 ; BID12 ).

We explored whether ReLU and Sigmoid networks trained via a mix of normal and adversarial exam- ples offer a resistance to adversarial attacks compared to that offered by RBFI networks trained on standard examples only.

For brevity, we omit the results for Sigmoid networks, as they were consistently inferior to those for ReLU networks.

We compared the performance of a RBFI network with that of ReLU network trained normally (indicated simply by ReLU), and with ReLU networks trained as follows:• ReLU(FGSM) and ReLU(I-FSGM): for each (x, t) in the training set, we construct an adversarial examplex via (6) or FORMULA10 , and we feed both (x, t) and (x, t) to the network for training.• ReLU(PGD): for each (x, t) in the training set, we perform 100 steps of projected gradient descent from a point chosen at random in B (x) ∩ [0, 1] n ; denoting by x the ending point of the projected gradient descent, we feed both (x, t) and (x , t) to the network for training.

We generated adversarial examples for training for = 0.3, which is consistent with BID12 .

Due to the high computational cost of adversarial training (and in particular, PGD adversarial training), we performed one run, and performed the training of ReLU networks for 10 epochs, which seemed sufficient for their accuracy to plateau.

The results are given in FIG1 .

Overall, the best networks may be the simple RBFI networks, trained without the use of adversarial examples: for each class of attack, they exhibit either the best performance, or they are very close in performance to the best performer; this is true for no other network type.

For PGD attacks, the best performance is obtained by ReLU(PGD) networks trained on PGD attacks, but this may be simply due to gradient masking: note that ReLU(PGD) networks do not perform well with respect to I-FGSM attacks.

We note that ReLU(FGSM) networks seem to learn that = 0.3 FGSM attacks are likely, but they have not usefully generalized the lesson, for instance, to attacks of size 0.1.

The S-shaped performance curve of ReLU(FGSM) with respect to FGSM or noise is known as label leaking: the network learns to recognize the original input given its perturbed version BID10 ).

We compared the performance achieved by training RBFI networks with standard gradients, and with pseudogradients.

After 30 epochs of training RBFI(512, 512, 512, 10 | * , * , * , ∨) networks, pseudogradients yielded (96.79 ± 0.17)% accuracy, while regular gradients only (86.35 ± 0.75)%.

On smaller networks, that should be easier to train, the gap even widened: for RBFI(128, 128, 10 | * , * , ∨) networks, it went from (95.00 ± 0.29)% for pseudogradients to (82.40 ± 3.72)% for regular gradients.

In Section 3, we developed upper bounds for the sensitivity of ReLU and RBFI networks to adversarial attacks on the basis of network weights.

It is reasonable to ask whether, using those upper bounds as weight regularizations, we might achieve robustness to adversarial attacks.

For ReLU networks, the answer is substantially negative.

We experimented adding to the loss used to train the network a term cŝ, for c ≥ 0 andŝ as in (4).

We experimented systematically for many values of c. Large values prevented the network from learning.

Smaller values resulted in little additional robustness: for = 0.3, simple FGSM attacks lowered the network accuracy to below 10%.For RBFI networks, regularization did help.

The choice of upper bound for the components of the u-vector influences the resistance of the trained networks to adversarial examples, as can be seen from (5).

In the experiments reported thus far, we used an upper bound of 3.

One may ask: would RBFI networks perform as well, if a higher bound were used?

The answer is yes, provided weight regularization is used in place of a tighter bound.

If we raise the bound to 10, and use no regularization, the accuracy under PGD attacks with = 0.3 drops from 93.32% to 83.62%.

By adding to the loss the regularization cŝ, for c = 0.0001 andŝ as in FORMULA8 , we can recover most of the lost accuracy, obtaining accuracy 89.38% at = 0.3.

In this paper, we have shown that non-linear structures such as RBFI can be efficiently trained using artificial, "pseudo" gradients, and can attain both high accuracy and high resistance to adversarial attacks.

@highlight

We introduce a type of neural network that is structurally resistant to adversarial attacks, even when trained on unaugmented training sets.  The resistance is due to the stability of network units wrt input perturbations.