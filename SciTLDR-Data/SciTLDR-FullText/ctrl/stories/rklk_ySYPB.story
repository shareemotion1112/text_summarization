In recent years several adversarial attacks and defenses have been proposed.

Often seemingly robust models turn out to be non-robust when more sophisticated attacks are used.

One way out of this dilemma are provable robustness guarantees.

While provably robust models for specific $l_p$-perturbation models have been developed, we show that they do not come with any guarantee against other $l_q$-perturbations.

We propose a new regularization scheme, MMR-Universal, for ReLU networks which enforces robustness wrt $l_1$- \textit{and} $l_\infty$-perturbations and show how that leads to the first provably robust models wrt any $l_p$-norm for $p\geq 1$.

The vulnerability of neural networks against adversarial manipulations (Szegedy et al., 2014; Goodfellow et al., 2015) is a problem for their deployment in safety critical systems such as autonomous driving and medical applications.

In fact, small perturbations of the input which appear irrelevant or are even imperceivable to humans change the decisions of neural networks.

This questions their reliability and makes them a target of adversarial attacks.

To mitigate the non-robustness of neural networks many empirical defenses have been proposed, e.g. by Gu & Rigazio (2015) ; Zheng et al. (2016) ; Papernot et al. (2016) ; Huang et al. (2016) ; Bastani et al. (2016) ; Madry et al. (2018) , but at the same time more sophisticated attacks have proven these defenses to be ineffective (Carlini & Wagner, 2017; Athalye et al., 2018; Mosbach et al., 2018) , with the exception of the adversarial training of Madry et al. (2018) .

However, even these l ∞ -adversarially trained models are not more robust than normal ones when attacked with perturbations of small l p -norms with p = ∞ (Sharma & Chen, 2019; Schott et al., 2019; Croce et al., 2019b; Kang et al., 2019) .

The situation becomes even more complicated if one extends the attack models beyond l p -balls to other sets of perturbations (Brown et al., 2017; Engstrom et al., 2017; Hendrycks & Dietterich, 2019; Geirhos et al., 2019) .

Another approach, which fixes the problem of overestimating the robustness of a model, is provable guarantees, which means that one certifies that the decision of the network does not change in a certain l p -ball around the target point.

Along this line, current state-of-theart methods compute either the norm of the minimal perturbation changing the decision at a point (e.g. Katz et al. (2017) ; Tjeng et al. (2019) ) or lower bounds on it (Hein & Andriushchenko, 2017; Raghunathan et al., 2018; Wong & Kolter, 2018) .

Several new training schemes like (Hein & Andriushchenko, 2017; Raghunathan et al., 2018; Wong & Kolter, 2018; Mirman et al., 2018; Croce et al., 2019a; Xiao et al., 2019; Gowal et al., 2018) aim at both enhancing the robustness of networks and producing models more amenable to verification techniques.

However, all of them are only able to prove robustness against a single kind of perturbations, typically either l 2 -or l ∞ -bounded, and not wrt all the l p -norms simultaneously, as shown in Section 5.

Some are also designed to work for a specific p (Mirman et al., 2018; Gowal et al., 2018) , and it is not clear if they can be extended to other norms.

The only two papers which have shown, with some limitations, non-trivial empirical robustness against multiple types of adversarial examples are Schott et al. (2019) and Tramèr & Boneh In this paper we aim at robustness against all the l p -bounded attacks for p ≥ 1.

We study the non-trivial case where none of the l p -balls is contained in another.

If p is the radius of the l p -ball for which we want to be provably robust, this requires:

q > p > q for p < q and d being the input dimension.

We show that, for normally trained models, for the l 1 -and l ∞ -balls we use in the experiments none of the adversarial examples constrained to be in the l 1 -ball (i.e. results of an l 1 -attack) belongs to the l ∞ -ball, and vice versa.

This shows that certifying the union of such balls is significantly more complicated than getting robust in only one of them, as in the case of the union the attackers have a much larger variety of manipulations available to fool the classifier.

We propose a technique which allows to train piecewise affine models (like ReLU networks) which are simultaneously provably robust to all the l p -norms with p ∈ [1, ∞].

First, we show that having guarantees on the l 1 -and l ∞ -distance to the decision boundary and region boundaries (the borders of the polytopes where the classifier is affine) is sufficient to derive meaningful certificates on the robustness wrt all l p -norms for p ∈ (1, ∞).

In particular, our guarantees are independent of the dimension of the input space and thus go beyond a naive approach where one just exploits that all l p -metrics can be upper-and lower-bounded wrt any other l q -metric.

Then, we extend the regularizer introduced in Croce et al. (2019a) so that we can directly maximize these bounds at training time.

Finally, we show the effectiveness of our technique with experiments on four datasets, where the networks trained with our method are the first ones having non-trivial provable robustness wrt l 1 -, l 2 -and l ∞ -perturbations.

It is well known that feedforward neural networks (fully connected, CNNs, residual networks, DenseNets etc.) with piecewise affine activation functions, e.g. ReLU, leaky ReLU, yield continuous piecewise affine functions (see e.g. Arora et al. (2018) ; Croce & Hein (2018) ).

Croce et al. (2019a) exploit this property to derive bounds on the robustness of such networks against adversarial manipulations.

In the following we recall the guarantees of Croce et al.

(2019a) wrt a single l p -perturbation which we extend in this paper to simultaneous guarantees wrt all the l p -perturbations for p in [1, ∞].

Let f : R d → R K be a classifier with d being the dimension of the input space and K the number of classes.

The classifier decision at a point x is given by arg max r=1,...,K f r (x).

In this paper we deal with ReLU networks, that is with ReLU activation function (in fact our approach can be easily extended to any piecewise affine activation function e.g. leaky ReLU or other forms of layers leading to a piecewise affine classifier as in Croce et al. (2019b) Denoting the activation function as σ (σ(t) = max{0, t} if ReLU is used) and assuming L hidden layers, we have the usual recursive definition of f as

where n l is the number of units in the l-th layer (

For the convenience of the reader we summarize from Croce & Hein (2018) the description of the polytope Q(x) containing x and affine form of the classifier f when restricted to Q(x).

We assume that x does not lie on the boundary between polytopes (this is almost always true as faces shared between polytopes are of lower dimension).

Let

This allows us to write f (l) (x) as composition of affine functions, that is

which we simplify as

A forward pass through the network is sufficient to compute V (l) and b (l) for every l. The polytope Q(x) is given as intersection of N = L l=1 n l half spaces defined by

Let q be defined via

. .

, K and s = c, which represent the N l p -distances of x to the hyperplanes defining the polytope Q(x) and the K − 1 l p -distances of x to the hyperplanes defining the decision boundaries in Q(x).

Finally, we define

as the minimum values of these two sets of distances (note that d

The l p -robustness r p (x) of a classifier f at a point x, belonging to class c, wrt the l p -norm is defined as the optimal value of the following optimization problem

where is S a set of constraints on the input, e.g. pixel values of images have to be in [0, 1].

The l p -robustness r p (x) is the smallest l p -distance to x of a point which is classified differently from c. Thus, r p (x) = 0 for misclassified points.

The following theorem from Croce et al. (2019a) , rephrased to fit the current notation, provides guarantees on r p (x).

Although Theorem 2.1 holds for any l p -norm with p ≥ 1, it requires to compute d B p (x) and d D p (x) for every p individually.

In this paper, exploiting this result and the geometrical arguments presented in Section 3, we show that it is possible to derive bounds on the robustness r p (x) for any p ∈ (1, ∞) using only information on r 1 (x) and r ∞ (x).

In the next section, we show that the straightforward usage of standard l p -norms inequalities does not yield meaningful bounds on the l p -robustness inside the union of the l 1 -and l ∞ -ball, since these bounds depend on the dimension of the input space of the network.

Figure 1 : Visualization of the l 2 -ball contained in the union resp.

the convex hull of the union of l 1 -and l ∞ -balls in R 3 .

First column: co-centric l 1 -ball (blue) and l ∞ -ball (black).

Second: in red the largest l 2 -ball completely contained in the union of l 1 -and l ∞ -ball.

Third: in green the convex hull of the union of the l 1 -and l ∞ -ball.

Fourth: the largest l 2 -ball (red) contained in the convex hull.

The l 2 -ball contained in the convex hull is significantly larger than that contained in the union of l 1 -and l ∞ -ball.

3 Minimal l p -norm of the complement of the union of l 1 -and l ∞ -ball and its convex hull

By the standard norm inequalities it holds, for every x ∈ R d , that

and thus a naive application of these inequalities yields the bound

However, this naive bound does not take into account that we know that x 1 ≥ 1 and x ∞ ≥ ∞ .

Our first result yields the exact value taking advantage of this information.

Thus a guarantee both for l 1 -and l ∞ -ball yields a guarantee for all intermediate l p -norms.

However, for affine classifiers a guarantee for B 1 and B ∞ implies a guarantee wrt the convex hull C of their union B 1 ∪ B ∞ .

This can be seen by the fact that an affine classifier generates two half-spaces, and the convex hull of a set A is the intersection of all half-spaces containing A. Thus, inside C the decision of the affine classifier cannot change if it is guaranteed not to change in B 1 and B ∞ , as C is completely contained in one of the half-spaces generated by the classifier (see Figure 1 for illustrations of B 1 , B ∞ , their union and their convex hull).

With the following theorem, we characterize, for any p ≥ 1, the minimal l p -norm over R d \ C.

where α =

5) (red) and its naive lower bound (4) (green).

We fix ∞ = 1 and show the results varying 1 ∈ (1, d), for d = 784 and d = 3072.

We plot the value (or a lower bound in case of (4)) of the minimal x 2 , depending on 1 , given by the different approaches (first and third plots).

The red curves are almost completely hidden by the green ones, as they mostly overlap, but can be seen for small values of x 1 .

Moreover, we report (second and fourth plots) the ratios of the minimal

The values provided by (6) are much larger than those of (5).

Note that our expression in Theorem 3.1 is exact and not just a lower bound.

Moreover, the minimal l p -distance of R d \ C to the origin in Equation (6) is independent from the dimension d, in contrast to the expression for the minimal l p -norm over R d \ U 1,∞ in (5) and its naive lower bound in (4), which are both decreasing for increasing d and p > 1.

In Figure  1 we compare visually the largest l 2 -balls (in red) fitting inside either U 1,∞ or the convex hull C in R 3 , showing that the one in C is clearly larger.

In Figure 2 we provide a quantitative comparison in high dimensions.

We plot the minimal l 2 -norm over R d \ C (6) (blue) and over R d \ U 1,∞ (5) (red) and its naive lower bound (4) (green).

We fix x ∞ = ∞ = 1 and vary

e. the dimensions of the input spaces of MNIST and CIFAR-10.

One sees clearly that the blue line corresponding to (6) is significantly higher than the other two.

In the second and fourth plots of Figure 2 we show, for each 1 , the ratio of the l 2 -distances given by (6) and (5).

The maximal ratio is about 3.8 for d = 784 and 5.3 for d = 3072, meaning that the advantage of (6) increases with d.

These two examples indicate that the l p -balls contained in C can be a few times larger than those in U 1,∞ .

Recall that we deal with piecewise affine networks.

If we could enlarge the linear regions on which the classifier is affine so that it contains the l 1 -and l ∞ -ball of some desired radii, we would automatically get the l p -balls of radii given by Theorem 3.1 to fit in the linear regions.

The next section formalizes the resulting robustness guarantees.

Combining the results of Theorems 2.1 and 3.1, in the next theorem we derive lower bounds on the robustness of a continuous piecewise affine classifier f , e.g. a ReLU network, at a point x wrt any l p -norm with p ≥ 1 using only d (2)).

for any p ∈ (1, ∞), with α =

It tries to push the k B closest hyperplanes defining Q(x) farther than γ B from x and the k D closest decision hyperplanes farther than γ D from x both wrt l p -metric.

In other words, MMR-l p aims at widening the linear regions around the training points so that they contain l p -balls of radius either γ B or γ D centered in the training points.

Using MMR-l p wrt a fixed l p -norm, possibly in combination with the adversarial training of Madry et al. (2018), leads to classifiers which are empirically resistant wrt l p -adversarial attacks and are easily verifiable by state-of-the-art methods to provide lower bounds on the true robustness.

where

We stress that, even if the formulation of MMR-Universal is based on MMR-l p , it is just thanks to the novel geometrical motivation provided by Theorem 3.1 and its interpretation in terms of robustness guarantees of Theorem 4.1 that we have a theoretical justification of MMR-Universal.

Moreover, we are not aware of any other approach which can enforce simultaneously l 1 -and l ∞ -guarantees, which is the key property of MMR-Universal.

The loss function which is minimized while training the classifier f is then, with {(

being the training set and CE the cross-entropy loss,

During the optimization our regularizer aims at pushing both the polytope boundaries and the decision hyperplanes farther than γ 1 in l 1 -distance and farther than γ ∞ in l ∞ -distance from the training point x, in order to achieve robustness close or better than γ 1 and γ ∞ respectively.

According to Theorem 4.1, this enhances also the l p -robustness for p ∈ (1, ∞).

Note that if the projection of x on a decision hyperplane does not lie inside

is just an approximation of the signed distance to the true decision surface, in which case Croce et al. (2019a) argue that it is an approximation of the local Cross-Lipschitz constant which is also associated to robustness (see Hein & Andriushchenko (2017) ).

The regularization parameters λ 1 and λ ∞ are used to balance the weight of the l 1 -and l ∞ -term in the regularizer, and also wrt the cross-entropy loss.

Note that the terms of MMR-Universal involving the quantities d

(x) penalize misclassification, as they take negative values in this case.

Moreover, we take into account the k B closest hyperplanes and not just the closest one as done in Theorems 2.1 and 4.1.

This has two reasons: first, in this way the regularizer enlarges the size of the linear regions around the training points more quickly and effectively, given the large number of hyperplanes defining each polytope.

Second, pushing many hyperplanes influences also the neighboring linear regions of Q(x).

This comes into play when, in order to get better bounds on the robustness at x, one wants to explore also a portion of the input space outside of the linear region Q(x), which is where Theorem 4.1 holds.

As noted in Raghunathan et al. (2018) ; Croce et al. (2019a); Xiao et al. (2019) , established methods to compute lower bounds on the robustness are loose or completely fail when using normally trained models.

In fact, their effectiveness is mostly related to how many ReLU units have stable sign when perturbing the input x within a given l p -ball.

This is almost equivalent to having the hyperplanes far from x in l p -distance, which is what MMR-Universal tries to accomplish.

This explains why in Section 5 we can certify the models trained with MMR-Universal with the methods of Wong & Kolter (2018) and Tjeng et al. (2019) .

We compare the models obtained via our MMR-Universal regularizer to state-of-the-art methods for provable robustness and adversarial training.

As evaluation criterion we use the robust test error, defined as the largest classification error when every image of the test set can be perturbed within a fixed set (e.g. an l p -ball of radius p ).

We focus on the l p -balls with p ∈ {1, 2, ∞}. Since computing the robust test error is in general an NP-hard problem, we evaluate lower and upper bounds on it.

The lower bound is the fraction of points for which an attack can change the decision with perturbations in the l p -balls of radius p (adversarial samples), that is with l p -norm smaller than p .

For this task we use the PGD-attack (Kurakin et al. (2017) (2018)).

In choosing the values of p for p ∈ {1, 2, ∞}, we try to be consistent with previous literature (e.g. Wong & Kolter (2018) ; Croce et al. (2019a) ) for the values of ∞ and 2 .

Equation (6) provides, given 1 and ∞ , a value at which one can expect l 2 -robustness (approximately 2 = √ 1 ∞ ).

Then we fix 1 such that this approximation is slightly larger than the desired 2 .

We show in Table 1 the values chosen for p , p ∈ {1, 2, ∞}, and used to compute the robust test error in Table 2 .

Notice that for these values no l p -ball is contained in the others.

Moreover, we compute for the plain models the percentage of adversarial examples given by an l 1 -attack (we use the PGD-attack) with budget 1 which have also l ∞ -norm smaller than or equal to ∞ , and vice versa.

These percentages are zero for all the datasets, meaning that being (provably) robust in the union of these l p -balls is much more difficult than in just one of them (see also C.1).

We train CNNs on MNIST, Fashion-MNIST (Xiao et al. (2017) et al. (2019a) , either alone or with adversarial training (MMR+AT) and the training with our regularizer MMR-Universal.

We use AT, KW, MMR and MMR+AT wrt l 2 and l ∞ , as these are the norms for which such methods have been used in the original papers.

More details about the architecture and models in C.3.

In Table 2 we report test error (TE) computed on the whole test set and lower (LB) and upper (UB) bounds on the robust test error obtained considering the union of the three l p -balls, indicated by l 1 + l 2 + l ∞ (these statistics are on the first 1000 points of the test set).

The lower bounds l 1 + l 2 + l ∞ -LB are given by the fraction of test points for which one of the adversarial attacks wrt l 1 , l 2 and l ∞ is successful.

The upper bounds l 1 + l 2 + l ∞ -UB are computed as the percentage of points for which at least one of the three l p -balls is not certified to be free of adversarial examples (lower is better).

This last one is the metric of main interest, since we aim at universally provably robust models.

In C.2 we report the lower and upper bounds for the individual norms for every model.

MMR-Universal is the only method which can give non-trivial upper bounds on the robust test error for all datasets, while almost all other methods aiming at provable robustness have l 1 + l 2 + l ∞ -UB close to or at 100%.

Notably, on GTS the upper bound on the robust test error of MMR-Universal is lower than the lower bound of all other methods except AT-(l 1 , l 2 , l ∞ ), showing that MMR-Universal provably outperforms existing methods which provide guarantees wrt individual l p -balls, either l 2 or l ∞ , when certifying the union l 1 +l 2 +l ∞ .

The test error is slightly increased wrt the other methods giving provable robustness, but the same holds true for combined adversarial training AT-(l 1 , l 2 , l ∞ ) compared to standard adv.

training AT-l 2 /l ∞ .

We conclude that MMR-Universal is the only method so far being able to provide non-trivial robustness guarantees for multiple l p -balls in the case that none of them contains any other.

We have presented the first method providing provable robustness guarantees for the union of multiple l p -balls beyond the trivial case of the union being equal to the largest one, establishing a baseline for future works.

Without loss of generality after a potential permutation of the coordinates it holds |x d | = x ∞ .

Then we get

, which finishes the proof.

Proof.

We first note that the minimum of the l p -norm over R d \ C lies on the boundary of C (otherwise any point on the segment joining the origin and y and outside C would have

where (1 − α ) ∞ < ∞ and

Thus S would not span a face as a convex combination intersects the interior of C. This implies that if 1 e j is in S then all the vertices v of B ∞ in S need to have v j = ∞ , otherwise S would not define a face of C. Analogously, if − 1 e j ∈ S then any vertex v of B ∞ in S has v j = − ∞ .

However, we note that out of symmetry reasons we can just consider faces of C in the positive orthant and thus we consider in the following just sets S which contain vertices of "positive type" 1 e j .

Let now S be a set (not necessarily defining a face of C) containing h ≤ k vertices of B 1 and d − h vertices of B ∞ and P the matrix whose columns are these points.

The matrix P has the form

−h is a matrix whose entries are either ∞ or − ∞ .

If the matrix P does not have full rank then the origin belongs to any hyperplane containing S, which means it cannot be a face of C. This also implies A has full rank if S spans a face of C.

We denote by π the hyperplane generated by the affine hull of S (the columns of P ) assuming that A has full rank.

Every point b belonging to the hyperplane π generated by S is such that there exists a unique a ∈ R d which satisfies

where 1 d1,d2 is the matrix of size d 1 × d 2 whose entries are 1.

The matrix (P , b ) ∈ R d+1,d+1 need not have full rank, so that

and then the linear system P a = b has a unique solution.

We define the vector v ∈ R d as solution of

, which is unique as P has full rank.

From their definitions we have P a = b and 1 T a = 1, so that

and thus

noticing that this also implies that any vector b ∈ R d such that b, v = 1 belongs to π (suppose that ∃q / ∈ π with q, v = 1, then define c as the solution of P c = q and then 1 = q, v = P c, v = c, P

T v = c, 1 which contradicts that q / ∈ π).

Applying Hölder inequality to (10) we get for any b ∈ π,

where 1 p + 1 q = 1.

Moreover, as p ∈ (1, ∞) there exists always a point b * for which (11) holds as equality.

In the rest of the proof we compute v q for any q > 1 when S is a face of C and then (11) yields the desired minimal value of b p over all b lying in faces of C.

which implies

Moreover, we have

Furthermore v 2 is defined as the solution of

We note that all the entries of

are either 1 or −1, so that the inner product between each row of A T and v 2 is a lower bound on the l 1 -norm of v 2 .

Since every entry of the r.h.s.

of the linear system is

, which combined with (13) leads to

In order to achieve equality u, v = v 1 it has to hold u i = sgn(v i ) for every v i = 0.

If at least two components of v were non-zero, the corresponding columns of A T would be identical, which contradicts the fact that A T has full rank.

Thus v 2 can only have one non-zero component which in absolute value is equal to

Thus, after a potential reordering of the components, v has the form

From the second condition in (13), we have

This means that, in order for S to define a face of C, we need h = k if α > 0, h ∈ {k − 1, k} if α = 0 (in this case choosing h = k − 1 or h = k leads to the same v, so in practice it is possible to use simply h = k for any α).

Once we have determined v, we can use again (10) and (11) to see that

Finally, for any v there exists b * ∈ π for which equality is achieved in (14).

Suppose that this b * does not lie in a face of C. Then one could just consider the line segment from the origin to b * and the point intersecting the boundary of C would have smaller l p -norm contradicting the just derived inequality.

Thus the b * realizing equality in (14) lies in a face of C.

Restricting the analysis to p = 2 for simplicity, we get

and one can check that δ * is indeed a maximizer.

Moreover, at δ * we have a ratio between the two bounds

We observe that the improvement of the robustness guarantee by considering the convex hull instead of the union is increasing with dimension and is ≈ 3.

Therefore the interior of the l 1 -ball of radius ρ 1 (namely, B 1 (x, ρ 1 )) and of the l ∞ -ball of radius ρ ∞ (B ∞ (x, ρ ∞ )) centered in x does not intersect with any of those hyperplanes.

This implies that {π j } j are intersecting the closure of

In Table 3 we compute the percentage of adversarial perturbations given by the PGD-attack wrt l p with budget p which have l q -norm smaller than q , for q = p (the values of p and q used are those from Table 1 ).

We used the plain model of each dataset.

The most relevant statistics of Table 3 are about the relation between the l 1 -and l ∞ -perturbations (first two rows).

In fact, none of the adversarial examples wrt l 1 is contained in the l ∞ -ball, and vice versa.

This means that, although the volume of the l 1 -ball is much smaller, even because of the intersection with the box constraints [0, 1] d , than that of the l ∞ -ball in high dimension, and most of it is actually contained in the l ∞ -ball, the adversarial examples found by l 1 -attacks are anyway very different from those got by l ∞ -attacks.

The choice of such p is then meaningful, as the adversarial perturbations we are trying to prevent wrt the various norms are non-overlapping and in practice exploit regions of the input space significantly diverse one from another.

Moreover, one can see that also the adversarial manipulations wrt l 1 and l 2 do not overlap.

Regarding the case of l 2 and l ∞ , for MNIST and F-MNIST it happens that the adversarial examples wrt l 2 are contained in the l ∞ -ball.

However, as one observes in Table 4 , being able to certify the l ∞ -ball is not sufficient to get non-trivial guarantees wrt l 2 .

In fact, all the models trained on these datasets to be provably robust wrt the l ∞ -norm, that is KW-l ∞ , MMR-l ∞ and MMR+AT-l ∞ , have upper bounds on the robust test error in the l 2 -ball larger than 99%, despite the values of the lower bounds are small (which means that the attacks could not find adversarial perturbations for many points).

Such analysis confirms that empirical and provable robustness are two distinct problems, and the interaction of different kinds of perturbations, as we have, changes according to which of these two scenarios one considers.

In Table 4 we report, for each dataset, the test error and upper and lower bounds on the robust test error, together with the p used, for each norm individually.

It is clear that training for provable l p -robustness (expressed by the upper bounds) does not, in general, yield provable l q -robustness for q = p, even in the case where the lower bounds are small for both p and q.

In order to compute the upper bounds on the robust test error in Tables 2 and 4 we use the method of Wong & Kolter (2018) for all the three l p -norms and that of Tjeng et al. (2019) only for the l ∞ -norm.

This second one exploits a reformulation of the problem in (3) in terms of mixed integer programming (MIP), which is able to exactly compute the solution of (3) for p ∈ {1, 2, ∞}. However, such technique is strongly limited by its high computational cost.

The only reason why it is possible to use it in practice is the exploitation of some presolvers which are able to reduce the complexity of the MIP.

Unfortunately, such presolvers are effective just wrt l ∞ .

On the other hand, the method of Wong & Kolter (2018) applies directly to every l p -norm.

This explains why the bounds provided for l ∞ are tighter than those for l 1 and l 2 .

The convolutional architecture that we use is identical to Wong & Kolter (2018) , which consists of two convolutional layers with 16 and 32 filters of size 4 × 4 and stride 2, followed by a fully connected layer with 100 hidden units.

The AT-l ∞ , AT-l 2 , KW, MMR and MMR+AT training models are those presented in Croce et al. (2019a) and available at https: //github.com/max-andr/provable-robustness-max-linear-regions.

We trained the AT-(l 1 , l 2 , l ∞ ) performing for each batch of the 128 images the PGD-attack wrt the three norms (40 steps for MNIST and F-MNIST, 10 steps for GTS and CIFAR-10) and then training on the point realizing the maximal loss (the cross-entropy function is used), for 100 epochs.

For all experiments with MMR-Universal we use batch size 128 and we train the models for 100 epochs.

Moreover, we use Adam optimizer of Kingma & Ba (2014) with learning rate of 5 × 10 −4 for MNIST and F-MNIST, 0.001 for the other datasets.

We also reduce the learning rate by a factor of 10 for the last 10 epochs.

On CIFAR-10 dataset we apply random crops and random mirroring of the images as data augmentation.

For training we use MMR-Universal as in (9) with k B linearly (wrt the epoch) decreasing from 20% to 5% of the total number of hidden units of the network architecture.

We also use a training schedule for λ p where we linearly increase it from λ p /10 to λ p during the first 10 epochs.

We employ both schemes since they increase the stability of training with MMR.

In order to determine the best set of hyperparameters λ 1 , λ ∞ , γ 1 , and γ ∞ of MMR, we perform a grid search over them for every dataset.

In particular, we empirically found that the optimal values of γ p are usually between 1 and 2 times the p used for the evaluation of the robust test error, while the values of λ p are more diverse across the different datasets.

Specifically, for the models we reported in Table 4 the following values for the (λ 1 , λ ∞ ) have been used: (3.0, 12.0) for MNIST, (3.0, 40.0) for F-MNIST, (3.0, 12.0) for GTS and (1.0, 6.0) for CIFAR-10.

In Tables 2 and 4 , while the test error which is computed on the full test set, the statistics regarding upper and lower bounds on the robust test error are computed on the first 1000 points of the respective test sets.

For the lower bounds we use the FAB-attack with the Figure 3: We show, for each dataset, the evolution of the test error (red), upper bound (UB) on the robust test error wrt l 1 (black), l 2 (cyan) and l ∞ (blue) during training.

Moreover, we report in green the upper bounds on the test error when the attacker is allowed to exploit the union of the three l p -balls.

The statistics on the robustness are computed at epoch 1, 2, 5, 10 and then every 10 epochs on 1000 points with the method of Wong & Kolter (2018) , using the models trained with MMR-Universal.

original parameters, 100 iterations and 10 restarts.

For PGD we use also 100 iterations and 10 restarts: the directions for the update step are the sign of the gradient for l ∞ , the normalized gradient for l 2 and the normalized sparse gradient suggested by Tramèr & Boneh (2019) with sparsity level 1% for MNIST and F-MNIST, 10% for GTS and CIFAR-10.

Finally we use the Liner Region Attack as in the original code.

For MIP (Tjeng et al. (2019) ) we use a timeout of 120s, that means if no guarantee is obtained by that time, the algorithm stops verifying that point.

We show in Figure 3 the clean test error (red) and the upper bounds on the robust test error wrt l 1 (black), l 2 (cyan), l ∞ (blue) and wrt the union of the three l p -balls (green), evaluated at epoch 1, 2, 5, 10 and then every 10 epochs (for each model we train for 100 epochs) for the models trained with our regularizer MMR-Universal.

For each dataset used in Section 5 the test error is computed on the whole test set, while the upper bound on the robust test error is evaluated on the first 1000 points of the test set using the method introduced in Wong & Kolter (2018) (the thresholds 1 , 2 , ∞ are those provided in Table 1 ).

Note that the statistics wrt l ∞ are not evaluated additionally with the MIP formulation of Tjeng et al. (2019) as the results in the main paper which would improve the upper bounds wrt l ∞ .

For all the datasets the test error keeps decreasing across epochs.

The values of all the upper bounds generally improve during training, showing the effectiveness of MMR-Universal.

We here report the robustness obtained training with MMR-l p +AT-l q with p = q on MNIST.

This means that MMR is used wrt l p , while adversarial training wrt l q .

In particular we test p, q ∈ {1, ∞}. In Table 5 we report the test error (TE), lower (LB) and upper bounds (UB) on the robust test error for such model, evaluated wrt l 1 , l 2 , l ∞ and l 1 + l 2 + l ∞ as done in Section 5.

It is clear that training with MMR wrt a single norm does not suffice to get provable guarantees in all the other norms, despite the addition of adversarial training.

In fact, for both the models analysed the UB equals 100% for at least one norm.

Note that the statistics wrt l ∞ in the plots do not include the results of the MIP formulation of Tjeng et al. (2019).

<|TLDR|>

@highlight

We introduce a method to train models with provable robustness wrt all the $l_p$-norms for $p\geq 1$ simultaneously.