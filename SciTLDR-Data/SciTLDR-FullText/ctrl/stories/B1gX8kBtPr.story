Training neural networks to be certifiably robust is critical to ensure their safety against adversarial attacks.

However, it is currently very difficult to train a neural network that is both accurate and certifiably robust.

In this work we take a step towards addressing this challenge.

We prove that for every continuous function $f$, there exists a network $n$ such that: (i) $n$ approximates $f$ arbitrarily close, and (ii) simple interval bound propagation of a region $B$ through $n$ yields a result that is arbitrarily close to the optimal output of $f$ on $B$. Our result can be seen as a Universal Approximation Theorem for interval-certified ReLU networks.

To the best of our knowledge, this is the first work to prove the existence of accurate, interval-certified networks.

Much recent work has shown that neural networks can be fooled into misclassifying adversarial examples (Szegedy et al., 2014) , inputs which are imperceptibly different from those that the neural network classifies correctly.

Initial work on defending against adversarial examples revolved around training networks to be empirically robust, usually by including adversarial examples found with various attacks into the training dataset (Gu and Rigazio, 2015; Papernot et al., 2016; Zheng et al., 2016; Athalye et al., 2018; Eykholt et al., 2018; Moosavi-Dezfooli et al., 2017; Xiao et al., 2018) .

However, while empirical robustness can be practically useful, it does not provide safety guarantees.

As a result, much recent research has focused on verifying that a network is certifiably robust, typically by employing methods based on mixed integer linear programming (Tjeng et al., 2019) , SMT solvers (Katz et al., 2017) , semidefinite programming (Raghunathan et al., 2018a) , duality (Wong and Kolter, 2018; Dvijotham et al., 2018b) , and linear relaxations (Gehr et al., 2018; Weng et al., 2018; Wang et al., 2018b; Zhang et al., 2018; Singh et al., 2018; Salman et al., 2019) .

Because the certification rates were far from satisfactory, specific training methods were recently developed which produce networks that are certifiably robust: Mirman et al. (2018) ; Raghunathan et al. (2018b) ; Wang et al. (2018a) ; Wong and Kolter (2018) ; Wong et al. (2018) ; Gowal et al. (2018) train the network with standard optimization applied to an over-approximation of the network behavior on a given input region (the region is created around the concrete input point).

These techniques aim to discover specific weights which facilitate verification.

There is a tradeoff between the degree of the over-approximation used and the speed of training and certification.

Recently, (Cohen et al., 2019b) proposed a statistical approach to certification, which unlike the non-probabilistic methods discussed above, creates a probabilistic classifier that comes with probabilistic guarantees.

So far, some of the best non-probabilistic results achieved on the popular MNIST (Lecun et al., 1998) and CIFAR10 (Krizhevsky, 2009 ) datasets have been obtained with the simple Interval relaxation (Gowal et al., 2018; Mirman et al., 2019) , which scales well at both training and verification time.

Despite this progress, there are still substantial gaps between known standard accuracy, experimental robustness, and certified robustness.

For example, for CIFAR10, the best reported certified robustness is 32.04% with an accuracy of 49.49% when using a fairly modest l ??? region with radius 8/255 (Gowal et al., 2018) .

The state-of-the-art non-robust accuracy for this dataset is > 95% with experimental robustness > 50%.

Given the size of this gap, a key question then is: can certified training ever succeed or is there a fundamental limit?

In this paper we take a step in answering this question by proving a result parallel to the Universal Approximation Theorem (Cybenko, 1989; Hornik et al., 1989) .

We prove that for any continuous function f defined on a compact domain ?? ??? R m and for any desired level of accuracy ??, there exists a ReLU neural network n which can certifiably approximate f up to ?? using interval bound propagation.

As an interval is a fairly imprecise relaxation, our result directly applies to more precise convex relaxations (e.g., Zhang et al. (2018); Singh et al. (2019) ).

Theorem 1.1 (Universal Interval-Certified Approximation, Figure 1 ).

Let ?? ??? R m be a compact set and let f : ?? ??? R be a continuous function.

For all ?? > 0, there exists a ReLU network n such that for all boxes [a, b] in ?? defined by points a, b ??? ?? where a k ??? b k for all k, the propagation of the box [a, b] using interval analysis through the network n, denoted n ([a, b]), approximates the set

We recover the classical universal approximation theorem (|f (x) ??? n(x)| ??? ?? for all x ??? ??) by considering boxes [a, b] describing points (x = a = b).

Note that here the lower bound is not [l, u] as the network n is an approximation of f .

Because interval analysis propagates boxes, the theorem naturally handles l ??? norm bound perturbations to the input.

Other l p norms can be handled by covering the l p ball with boxes.

The theorem can be extended easily to functions f : ?? ??? R k by applying the theorem component wise.

Practical meaning of theorem The practical meaning of this theorem is as follows: if we train a neural network n on a given training data set (e.g., CIFAR10) and we are satisfied with the properties of n (e.g., high accuracy), then because n is a continuous function, the theorem tells us that there exists a network n which is as accurate as n and as certifiable with interval analysis as n is with a complete verifier.

This means that if we fail to find such an n, then either n did not possess the required capacity or the optimizer was unsuccessful.

Focus on the existence of a network We note that we do not provide a method for training a certified ReLU network -even though our method is constructive, we aim to answer an existential question and thus we focus on proving that a given network exists.

Interesting future work items would be to study the requirements on the size of this network and the inherent hardness of finding it with standard optimization methods.

Universal approximation is insufficient We now discuss why classical universal approximation is insufficient for establishing our result.

While classical universal approximation theorems state that neural networks can approximate a large class of functions f , unlike our result, they do not state that robustness of the approximation n of f is actually certified with a scalable proof method (e.g., interval bound propagation).

If one uses a non scalable complete verifier instead, then the standard Universal approximation theorem is sufficient.

To demonstrate this point, consider the function f : R ??? R (Figure 2b ) mapping all x ??? 0 to 1, all x ??? 1 to 0 and all 0 < x < 1 to 1 ??? x and two ReLU networks n 1 (Figure 2a ) and n 2 (Figure 2c ) perfectly approximating f , that is n 1 (x) = f (x) = n 2 (x) for all x. For ?? = 1 4 , the interval certification that n 1 maps all

However, interval certification succeeds for n 2 , because n 2 ([0, 1]) = [0, 1] .

To the best of our knowledge, this is the first work to prove the existence of accurate, interval-certified networks.

After adversarial examples were discovered by Szegedy et al. (2014) , many attacks and defenses were introduced (for a survey, see Akhtar and Mian (2018) ).

Initial work on verifying neural network robustness used exact methods (Katz et al., 2017; Tjeng et al., 2019) on small networks, while later research introduced methods based on over-approximation (Gehr et al., 2018; Raghunathan et al., 2018a; Singh et al., 2018; Salman et al., 2019) aiming to scale to larger networks.

A fundamentally different approach is randomized smoothing (Li et al., 2019; L??cuyer et al., 2019; Cohen et al., 2019b) , in which probabilistic classification and certification with high confidence is performed.

As neural networks that are experimentally robust need not be certifiably robust, there has been significant recent research on training certifiably robust neural networks (Raghunathan et al., 2018b; Mirman et al., 2018; 2019; Wong and Kolter, 2018; Wong et al., 2018; Wang et al., 2018a; Gowal et al., 2018; Dvijotham et al., 2018a; Xiao et al., 2019; Cohen et al., 2019b) .

As these methods appear to have reached a performance wall, several works have started investigating the fundamental barriers in the datasets and methods that preclude the learning of a robust network (let alone a certifiably robust one) (Khoury and Hadfield-Menell, 2018; Schmidt et al., 2018; Tsipras et al., 2019) .

In our work, we focus on the question of whether neural networks are capable of approximating functions whose robustness can be established with the efficient interval relaxation.

Feasibility Results with Neural Networks Early versions of the Universal Approximation Theorem were stated by Cybenko (1989) and Hornik et al. (1989) .

Cybenko (1989) showed that networks using sigmoidal activations could approximate continuous functions in the unit hypercube, while Hornik et al. (1989) showed that even networks with only one hidden layer are capable of approximating Borel measurable functions. (2019a) provide an explicit construction to obtain the network.

We note that both of these works focus on Lipschitz continuous functions, a more restricted class than continuous functions, which we consider in our work.

In this section we provide the concepts necessary to describe our main result.

Adversarial Examples and Robustness Verification Let n : R m ??? R k be a neural network, which classifies an input x to a label t if n(x) t > n(x) j for all j = t.

For a correctly classified input x, an adversarial example is an input y such that x is imperceptible from y to a human, but is classified to a different label by n.

Frequently, two images are assumed to be "imperceptible" if there l p distance is at most .

The l p ball around an image is said to be the adversarial ball, and a network is said to be -robust around x if (Figure 3a ) using a ReLU network n = ?? 0 + k n k .

The ReLU networks n k (Figure 3c ) approximate the N -slicing of f (Figure 3b ), as a sum of local bumps ( Figure 6 ).

every point in the adversarial ball around x classifies the same.

In this paper, we limit our discussion to l ??? adversarial balls which can be used to cover to all l p balls.

The goal of robustness verification is to show that for a neural network n, input point x and label t, every possible input in an l ??? ball of size around x (written B ??? (x)) is also classified to t.

, and ?? ??? R ???0 .

Furthermore, we used to distinguish the function f from its interval-transformation f .

To illustrate the difference between f and f , consider f (

illustrating the loss in precision that interval analysis suffers from.

Interval analysis provides a sound over-approximation in the sense that for all function f , the values that

Furthermore all combinations f of +, ???, ?? and R are monotone, that is for

.

This will later be needed.

In this section, we provide an explanation of the proof of our main result, Theorem 4.6, and illustrate the main points of the proof.

The first step in the construction is to deconstruct the function f into slices {f k : ?? ??? [0,

for all x, where ?? 0 is the minimum of f (??).

We approximate each slice f k by a ReLU network ?? 2 ?? n k .

The network n approximating f up to ?? will be n(x) := ?? 0 + ?? 2 k n k (x).

The construction relies on 2 key insights, (i) the output of ?? 2 ?? n k can be confined to the interval [0, ?? 2 ], thus the loss of analysis precision is at most the height of the slice, and (ii) we can construct the networks n k using local bump functions, such that only 4 slices can contribute to the loss of analysis precision, two for the lower interval bound, two for the upper one.

The slicing {f k } 0???k<5 of the function f : [???2, 2] ??? R (Figure 3a) , mapping x to f (x) = ???x 3 + 3x is depicted in Figure 3b .

The networks n k are depicted in Figure 3c .

In this example, evaluating the interval-transformer of n, namely n on the box

Definition 4.1 (N -slicing (Figure 3b) ).

Let ?? ??? R m be a closed m-dimensional box and let f : ?? ??? R be continuous.

The N -slicing of f is a set of functions {f k } 0???k<N defined by

where

To construct a ReLU network satisfying the desired approximation property (Equation (1)) if evaluated on boxes in B(??), we need the ReLU network nmin capturing the behavior of min as a building block (similar to He et al. (2018) ).

It is given by

With the ReLU network nmin, we can construct recursively a ReLU network nmin N mapping N arguments to the smallest one (Definition A.8).

Even though the interval-transformation loses precision, we can establish bounds on the precision loss of nmin N sufficient for our use case (Appendix A).

Now, we use the clipping function R [ * ,1] := 1 ??? R(1 ??? x) clipping every value exceeding 1 back to 1 (Figure 5 ) to construct the local bumps ?? c w.r.t.

a grid G. G specifies the set of all possible local bumps we can use to construct the networks n k .

Increasing the finesse of G will increases the approximation precision.

Definition 4.2 (local bump, Figure 6 ).

M } ??? G be a set of grid points describing the corner points of a hyperrectangle in G.

We define a ReLU neural network ?? c :

We will describe later how M and c get picked.

A graphical illustration of a local bump for in two dimensions and c = { Figure 6 .

The local bump ?? c (x) evaluates to 1 for all x that lie within the convex hull of c, namely conv(c), after which ?? c (x) quickly decreases linearly to 0.

?? c has 1 + 2(2d ??? 1) + 2d ReLUs and 1 + log 2 (2d + 1) + 1 layers.

The formal proof is given in Appendix A. The next lemma shows, how a ReLU network n k can approximate the slice f k while simultaneously confining the loss of analysis precision.

Lemma 4.4.

Let ?? ??? R m be a closed box and let f : ?? ??? R be continuous.

For all ?? > 0 there exists a set of ReLU networks {n k } 0???k<N of size N ??? N approximating the N -slicing of f , {f k } 0???k<N (?? k as in Definition 4.1) such that for all boxes B ??? B(??)

and

It is important to note that in Equation (2) we mean f and not f .

The proof for Lemma 4.4 is given in Appendix A. In the following, we discuss a proof sketch.

Because ?? is compact and f is continuous, f is uniformly continuous by the Heine-Cantor Theorem.

So we can pick a M ??? N such that for all x, y ??? ?? satisfying ||y???x||

Next, we construct for every slice k a set ??? k of hyperrectangles on the grid G: if a box B ??? B(??) fulfills f (B) ??? ?? k+1 + ?? 2 , then we add a minimal enclosing hyperrectangle c ??? G such that B ??? conv(c) to ??? k , where conv(c) denotes the convex hull of c. This implies, using uniform continuity of f and that the grid G is fine enough, that f (conv(c)) ??? ?? k+1 .

Since there is only a finite number of possible hyperrectangles in G, the set ??? k is clearly finite.

The network fulfilling Equation (2) is

where ?? c is as in Definition 4.2.

The n k are depicted in Figure 3c .

Now, we see that Equation (2) holds by construction: For all boxes B ??? B(??) such that f ??? ?? k+1 + ?? 2 on B exists c ??? ??? k such that B ??? conv(c ) which implies, using Lemma 4.3, that ?? c (B) = [1, 1], hence

Similarly, if f (B) ??? ?? k ??? ?? 2 holds, then it holds for all c ??? ??? k that B does not intersect N (conv(c)).

Indeed, if a c ??? ??? k would violate this, then by construction, f (conv(c)) ??? ?? k+1 , contradicting f (B) ??? ?? k ???

where l := min f (B) and u := max f (B).

Proof.

Pick N such that the height of each slice is exactly ?? 2 , if this is impossible choose a slightly smaller ??.

Let {n k } 0???k<N be a series of networks as in Lemma 4.4.

Recall that ?? 0 = min f (??).

We define the ReLU network

Let B ??? B(??).

Thus we have for all k

Let p, q ??? {0, . . .

, N ??? 1} such that

Figure 7: Illustration of the proof for Theorem 4.5.

as depicted in Figure 7 .

Thus by Equation (4) for all k ??? {0, . . . , p ??? 2} it holds that n k (B) = [1, 1] and similarly, by Equation (5) for all k ??? {q + 2, . . .

, N ??? 1} it holds that n k (B) = [0, 0] .

Plugging this into Equation (3) after splitting the sum into three parts leaves us with

Applying the standard rules for interval analysis, leads to

where we used in the last step, that ?? 0 + k ?? 2 = ?? k .

For all terms in the sum except the terms corresponding to the 3 highest and lowest k we get

Indeed, from Equation (6) we know that there is

Similarly, from Equation (7) we know, that there is x ??? B such that f (x) ??? ?? q = ?? q???1 +

We know further, that if p + 3 ??? q, than there is an x ??? B such that f (x) ??? ?? p+3 = ?? p+2 +

If p + 3 > q the lower bound we want to prove becomes vacuous and only the upper one needs to be proven.

Thus we have

where l := min f (B) and u := max f (B).

where l, u ??? R m such that l k := min f (B) k and u k := max f (B) k for all k.

Proof.

This is a direct consequence of using Theorem 4.5 and the Tietze extension theorem to produce a neural network for each dimension d of the codomain of f .

Note that Theorem 1.1 is a special case of Theorem 4.6 with d = 1 to simplify presentation.

We proved that for all real valued continuous functions f on compact sets, there exists a ReLU network n approximating f arbitrarily well with the interval abstraction.

This means that for arbitrary input sets, analysis using the interval relaxation yields an over-approximation arbitrarily close to the smallest interval containing all possible outputs.

Our theorem affirmatively answers the open question, whether the Universal Approximation Theorem generalizes to Interval analysis.

Our results address the question of whether the interval abstraction is expressive enough to analyse networks approximating interesting functions f .

This is of practical importance because interval analysis is the most scalable non-trivial analysis.

Lemma A.1 (Monotonicity).

The operations +, ??? are monotone, that is for all

Further the operation * and R are monotone, that is for all

Proof.

[

Definition A.2 (N -slicing).

Let ?? ??? R m be a compact m-dimensional box and let f : ?? ??? R be continuous.

The N -slicing of f is a set of functions {f k } 0???k???N ???1 defined by

where

Proof.

Pick x ??? ?? and let l ??? {0, . . .

, N ??? 1} such that ?? l ??? f (x) ??? ?? l+1 .

Then

Definition A.4 (clipping).

Let a, b ??? R, a < b. We define the clipping function R [ * ,b] : R ??? R by

Lemma A.5 (clipping).

The function R [ * ,b] sends all x ??? b to x, and all x > b to b. Further,

Proof.

We show the proof for R [a,b] , the proof for R [ * ,b] is similar.

Definition A.6 (nmin).

We define the ReLU network nmin :

Lemma A.7 (nmin).

Let x, y ??? R, then nmin(x, y) = min(x, y).

Proof.

Because nmin is symmetric in its arguments, we assume w.o.l.g.

x ??? y.

Definition A.8 (nmin N ).

For all N ??? N ???1 , we define a ReLU network nmin N defined by

Proof.

The symmetry on abstract elements is immediate.

In the following, we omit some of to improve readability.

Claim: R(a+c)???R(???a???c) = a+c.

If a+c > 0 then ???a???c < 0 thus the claim in this case.

Indeed:

So the expression simplifies to

We proceed by case distinction:

By symmetry of nmin equivalent to Case 1.

Hence

Case 3: a ??? d < 0 and b ??? c > 0:

Proof.

By induction.

Base case:

Induction hypothesis: The property holds for N s.t.

0 < N ??? N ??? 1.

Induction step: Then it also holds for N :

Proof.

By induction: Let N = 2:

Lemma A.14 Induction hypothesis: The statement holds for all 2 ??? N ??? N ??? 1.

Proof.

Let N ??? N such that N ??? 2 ??max?????min ?? where ?? min := min f (??) and ?? max := max f (??).

For simplicity we assume ?? = [0, 1] m .

Using the Heine-Cantor theorem, we get that f is uniformly continuous, thus there exists a ?? > 0 such that ???x, y ??? ??.||y ??? x|| ??? < ?? ??? ||f (y) ??? f (x)|| <

<|TLDR|>

@highlight

We prove that for a large class of functions f there exists an interval certified robust network approximating f up to arbitrary precision.