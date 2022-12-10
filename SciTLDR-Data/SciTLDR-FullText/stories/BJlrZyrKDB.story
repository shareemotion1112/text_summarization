The use of deep learning for a wide range of data problems has increased the need for understanding and diagnosing these models, and deep learning interpretation techniques have become an essential tool for data analysts.

Although numerous model interpretation methods have been proposed in recent years, most of these procedures are based on heuristics with little or no theoretical guarantees.

In this work, we propose a statistical framework for saliency estimation for black box computer vision models.

We build a model-agnostic estimation procedure that is statistically consistent and passes the saliency checks of Adebayo et al. (2018).

Our method requires solving a linear program, whose solution can be efficiently computed in polynomial time.

Through our theoretical analysis, we establish an upper bound on the number of model evaluations needed to recover the region of importance with high probability, and build a new perturbation scheme for estimation of local gradients that is shown to be more efficient than the commonly used random perturbation schemes.

Validity of the new method is demonstrated through sensitivity analysis.

Deep learning models have achieved great predictive performance in many tasks.

However, these complex, often un-tractable models are difficult to interpret and understand.

This lack of interpretability is a major barrier for their wide adoption, especially in domains (e.g., medicine) where models need to be qualitatively understood and/or verified for robustness.

In order to address these issues, several interpretation approaches have been proposed in the last few years.

A group of methods are based on visualizations, either by quantifying the effect of particular neurons or features, or by creating new images that maximize the target score for specific classes (Erhan et al., 2009; Simonyan et al., 2013; Zeiler & Fergus, 2014) .

A large collection of the techniques build saliency maps by attributing the gradients of the neural network to the input image through various procedures or by finding perturbations that significantly change the output (Springenberg et al., 2014; Bach et al., 2015; Montavon et al., 2017; Shrikumar et al., 2017; Zhou et al., 2016; Selvaraju et al., 2017; Smilkov et al., 2017; Fong & Vedaldi, 2017; Adebayo et al., 2018a; Dumitru et al., 2018; Singla et al., 2019) .

Another class of approaches treat the deep learner as a black-box.

In this domain, Baehrens et al. (2010) use a Parzen window classifier to approximate the target classifier locally.

Ribeiro et al. (2016) propose the LIME procedure, where small perturbations on the instance are used to obtain additional samples with which a sparse linear model is fit.

Lundberg & Lee (2017) propose SHapley Additive exPlanation(SHAP), which combines the Shapley value from the game theory with the additive feature attribution methods.

They also make connections of the SHAP procedure with various existing methods including LRP, LIME and DeepLIFT.

Chen et al. (2019) propose L-and C-Shapley procedures which can reliably approximate the Shapley values in linear time with respect to the number of features.

Majority of the listed methods are heuristics which are constructed according to certain desirable qualities.

For these methods, it is not clear what the main estimand is, if it can be consistently estimated or if (and how) the estimand can be computed more efficiently.

In fact, according to the recent research by Adebayo et al. (2018b) , most methods with great visual inspection lack sensitivity to the model and the data generating process.

Theoretical explanation for why guided back-propagation and deconvolutional methods perform image recovery is provided by Nie et al. (2018) .

In this work, we propose a statistically valid technique for model-agnostic saliency estimation, and prove its consistency under reasonable assumptions.

Furthermore, our method passes the sanity checks given by Adebayo et al. (2018b) .

Through our analysis, we obtain insights into how to improve the accuracy and reliability of our approach.

We note that there is recent work by Burns et al. (2019) where they provide a saliency estimation technique with theoretical guarantees -more specifically, FDR control.

Although their procedure is very promising from a statistical perspective, and theoretically valid under a very general set of assumptions, their technique requires human input and has a significant computational load as it uses a generative model for filling in certain regions of the target image.

Our main contributions are as follows:

• We introduce a new saliency estimation framework for CNNs and propose a new method based on input perturbation.

Our procedure requires solving a linear program, and hence the estimates can be computed very efficiently.

Furthermore, the optimization problem can be recast as a "parametric simplex" (Vanderbei, 2014) , which allows the computation of the full solution path in an expedient manner.

• We establish conditions under which the significant pixels in the input can be identified with high probability.

We present finite-sample convergence rates that can be used to determine the number of necessary model evaluations.

• We find that the noise distribution for the perturbation has a substantial effect on the convergence rate.

We propose a new perturbation scheme which uses a highly correlated Gaussian, instead of the widely used independent Gaussian distribution.

In the following section, we define the linearly estimated gradient (LEG), which is the saliency parameter of interest (i.e. the estimand), and introduce our statistical framework.

In section 3, we propose a regularized estimation procedure for LEG that penalizes the anisotropic total-variation.

We provide our theoretical results in Section 4 and the result of our numerical comparisons in Section 5.

For a matrix B, we use vec(B) and vec −1 (B) to denote its vectorization and inverse vectorization, respectively.

The transpose of a matrix B is given by B T and we use B + for its pseudo-inverse .

The largest and smallest eigenvalue of a symmetric matrix B are denoted by λ max (B) and λ min (B).

For a set S, we use S C to denote its complement.

For a vector u ∈ R p and a set S ⊆ [1, . . .

, p], we use u S to refer to its components indexed by elements in S. The q-norm for a vector u is given by u q and we use B F r for the Frobenius norm of a matrix B. The vector of size p whose values are all equal to 1 is denoted by 1 p .

Similarly, we use 1 p1×p2 and 0 p1×p2 to denote a p 1 × p 2 matrix whose entries are equal to 1 and 0, respectively.

Finally, for a continuous distribution F , we use F + x 0 to denote a distribution that is mean-shifted by x 0 , i.e. F (z) = G(z − x 0 ) for all z, where

In gradient based saliency approaches, the main goal is to recover the gradient of the deep learner with respect to the input.

More specifically, let f (x) be a deep learner, f : X → [0, 1], where X is the input space, e.g., [0, 255] 28×28 for the MNIST dataset, where the input are given as 28 by 28 sized images.

In this notation, the output is the probability of a specific class, for instance P model (x is a 9); although this can be modified to check for comparative quantities by setting the output as f (x) = f 9 (x) − f 7 (x) = P model (x is a 9) − P model (x is a 7).

(1) Then, local saliency is defined as the derivative of f (·) with respect to the input, evaluated at a point of interest x 0 ∈ X , i.e. ∇ x f (x)| x=x0 .

However, in practice, local saliency is often too noisy and one instead uses an average of the gradient around x 0 (Shrikumar et al., 2017; Smilkov et al., 2017) .

In order to study the saliency procedure from a statistical perspective, we start by defining an estimand, whose definition is motivated by the LIME procedure (Ribeiro et al., 2016) .

Definition 1 (LEG).

For a continuous distribution F , an initial point x 0 ∈ X with X ⊂ R p1×p2 , and a function f : X → [−1, 1], the linearly estimated gradient (LEG), γ ∈ R p1×p2 is given by

LEG is based on a first order Taylor series expansion of the function f (x) around the point of interest x 0 .

The estimand is a proxy for the local gradient, and is the coefficient that gives the best linear approximation, in terms of the squared error, among all possible choices.

The distribution F determines the range of points the analyst wants to consider.

We visually demonstrate LEG on two toy examples with a single pixel (i.e. p 1 = p 2 = 1) in Figure 1 .

Figure 1a , we compare LEG to the gradient, which is very localized.

If f (x) is a highly varying function, then the gradient is too noisy, and the saliency score provided by LEG is more meaningful.

In Figure 1b , we show LEG for two different distributions.

For the distribution with the larger variance, LEG evaluates the input's effect on the output for a larger neighborhood around x 0 .

We note that the variance of F has a large effect on LEG.

As F converges to a point mass at 0, if f (x) is twice continuously differentiable in the neighborhood of x 0 , then γ → ∇ x f (x).

On the other hand, if F has high variance, then samples from x 0 + F are substantially different from x 0 and LEG might no longer be useful for interpreting the model at x 0 .

This phenomenon can also described in terms of local vs global interpretation: for F with a small variance, LEG provides a very local interpretation, i.e. a gradient that is valid in a small neighborhood around x 0 , and as the variance of F increases, LEG produces a more global interpretation, since a larger neighborhood around x 0 is considered in the calculation.

LEG has an analytical solution as the next lemma shows.

Lemma 1.

Let Z be the random variable with a centered distribution F , i.e. Z ∼ F and E[Z] = 0 p1×p2 .

Assume that covariance of vec(Z) exists, and is positive-definite.

Proof of the lemma is provided in the Appendix.

Lemma 1 shows that the LEG can be written as an affine transformation of a high dimensional integral where the integrand is (f (

This analysis also suggests an empirical estimate for the LEG, by replacing the expectation with the empirical mean.

The empirical mean can be obtained by sampling x from F + x 0 , calculating f (x), and then applying Lemma 1.

More formally, let x 1 , . . .

, x n be random samples from F + x 0 , and let y 1 , . . .

, y n be the function evaluations with

As the function f (x) is bounded and F has a positive-definite covariance matrix, then it follows that as n → ∞,γ → γ.

However, classical linear model theory (Ravishanker & Dey, 2001) shows that rate of the convergence is very slow, on the order of 1 λmin(Σ) p 1 p 2 /n, where p 1 and p 2 are the dimensions of X .

This severely limits the practicality of the empirical approach.

In the next section we propose to use regularization in order to obtain faster convergence rates.

For interpretation of image classifiers, one expects that the saliency scores are located at a certain region, i.e. a contiguous body or a union of such bodies.

This idea has lead to various procedures that estimate saliency scores by penalizing the local differences of the solution, often utilizing some form of the total variation (TV) penalty (Fong & Vedaldi, 2017) .

The approach is very sensible from a practical point of view: Firstly, it produces estimates that are easy to interpret as the important regions can be easily identified; secondly, penalization significantly shrinks the variance of the estimate and helps produce reliable solutions with less model evaluations.

In the light of the above, we propose to estimate the LEG coefficient with an anisotropic L 1 TV penalty.

For a hyperparameter, L ≥ 0, the TV-penalized LEG estimate is given as γ = vec −1 (g) where g is the solution of the following linear program

where D ∈ R (2p1p2−p1−p2)×(p1p2) is the differencing matrix with D i,j = 1, D i,k = −1 if the j th and the k th component of g are connected on the two dimensional grid.

Our method is based on the "high confidence set" approach which has been successful in numerous applications in high dimensional statistics (Candes & Tao, 2007; Cai et al., 2011; Fan, 2013) .

The set of g that satisfy the constraint in the formulation is our high confidence set; if L is chosen properly, this set contains the true LEG coefficient, γ(f, x 0 , F ), with high probability 1 .

This setup ensures that the distance between γ andγ is small.

When combined with the TV penalty in the objective function, the procedure seeks to find a solution that both belongs to the confidence set and has sparse differences on the grid.

Thus, the estimator is extremely effective at recovering γ that have small total variation.

The proposed method enjoys low computational complexity.

The problem in equation 4 is a linear program and can be solved in polynomial time, for instance by using a primal-dual interior-point method for which the time complexity is O (p 1 p 2 ) 3.5 (Nocedal & Wright, 2006) .

However, in practice, solutions can be obtained much faster using simplex solvers.

In our implementations, we use MOSEK, a commercial grade simplex solver by ApS (2019), and are able to obtain a solution in less than 3 seconds on a standard 8-core PC for a problem of size p 1 = p 2 = 28.

Additionally, the alternative formulation (provided in the Appendix) can be solved using parametric simplex approaches which yield the whole solution path in L (Vanderbei, 2014).

The last point is often a necessity in deployment when L needs to be tuned according to some criteria.

We note that the procedure does not require any knowledge about the underlying neural network and is completely model-agnostic.

In fact, in applications where security or privacy could be a concern and returning multiple prediction values needs to be avoided, the term given by n i=1 vec (ỹ i z i ) can be computed on the side and supplied alongside the prediction.

In Figure 2 , we show the resulting estimates of the method with n = 500 model evaluations for a VGG-19 (Simonyan & Zisserman, 2014) network.

For the distribution F , we use a multivariate Gaussian distribution with the proposed perturbation scheme in Section 4.2.

We computeγ separately for each channel, and then sum the absolute values of the different channels to obtain the final saliency score.

In this section, we analyze the procedure from a theoretical perspective and derive finite sample convergence rates of the proposed LEG-TV estimator.

As we noted earlier, this analysis also gives us insight on the properties of the ideal perturbation distribution.

We first present our condition, which has a major role in the convergence rate of our estimator.

The condition is akin to the restricted eigenvalue condition (Bickel et al., 2009 ) with adjustments specific to our problem.

Assumption 1.

Let D + be the pseudo-inverse of the differencing matrix D, and denote the elements of singular value decomposition of D as U, Θ, V where D = U ΘV T .

Furthermore, denote the last p 1 p 2 − p 1 − p 2 columns of U that correspond to zero singular values as U 2 .

For the covariance matrix Σ, and any set S with size s, it holds that κ > 0, where

The following theorem is our main result.

, where Z ∼ F and E[Z] = 0 p1×p2 .

Letγ be the LEG-TV estimate with L = 2 D + 1 log (p 1 p 2 / ) /n.

If Assumption 1 holds for the covariance matrix Σ with constant κ, then with probability 1 − ,

where m ∈ R is a mean shift parameter, s is the number of non-zero elements in Dγ The proof is built on top of the "high confidence set" approach of Fan (2013) .

In the proof, we first establish that, for an appropriately chosen value of L, γ * = γ(f, x 0 , F ) satisfies the constraint in equation 4 with high probability.

Then, we make use of TV sparsity ofγ and γ * to argue that the two quantities cannot be too far away from each other, since both are in the constraint set.

The full proof is provided in the Appendix.

Our theorem has two major implications:

1.

We can recover the true parameter as the number of model evaluations increase.

That is, TV penalized LEG is a statistically consistent model interpretation scheme.

Furthermore, our result states that, ignoring the log terms, one needs n = O(s (p 1 p 2 ) 1/2 ) many model evaluations to reliably recover γ * .

2.

Our bound depends on the constant κ, which further depends on the choice of Σ for the perturbation scheme.

It is possible to obtain faster rates of convergence with a carefully tuned choice of Σ. As a side note, since γ * also depends on Σ, the estimand changes when Σ is adjusted.

In other words, our result states that certain estimands require less samples.

We note that our procedure identifies the LEG coefficient up to a mean shift parameter, m, which is the average of the true LEG coefficient γ.

In practice, the average can be consistently estimated (for instance, using the empirical version of LEG in equation 3), and the mean can be subtracted to yield consistent estimates for γ.

However, in our numerical studies, we see that this mean shift is almost non-existent: LEG-TV yields solutions that has no mean differences with the LEG coefficient, which we define as the solution of the empirical version as n → ∞.

In our main result, we established that the convergence of our estimator depends on the quantity κ which is related to the spectral properties of Σ. In this subsection we explore the ramifications of the assumption.

Our main result in Theorem 1 states that the rate of convergence to the true LEG coefficient is inversely proportional to the term κ.

Thus, perturbation schemes for which the restricted eigenvalues are large, as defined in Definition 1, yield saliency maps that require less samples to estimate the LEG.

We note that most of the saliency estimation procedures that make use of perturbations take these perturbations to be independent, which results in a covariance matrix that is equal to the identity matrix, Σ = σ 2 I (p1p2)×(p1p2) for some σ 2 > 0.

For LEG estimation without penalization, i.e. using equation 1, this choice is also optimal as the convergence rates under the normal setup depend on 1/λ min (Σ).

However, when one seeks to find an estimate for which the solution is sparse in the TV norm, this choice is no longer ideal as demonstrated by our theorem.

In order to choose the covariance matrix of our perturbation scheme in a manner that maximizes the bound in equation 5, one also needs some prior information about the size of S, s. As that requires estimation of s, and a complex optimization procedure, we instead propose a heuristic: we choose Σ so that its eigenvectors match D + ∆ for vectors ∆ with unit-norm and U T 2 ∆ = 0.

This choice fixes p 1 p 2 − 1 many of the eigenvectors of Σ. For the last eigenvector, we use the one vector as it is orthogonal to the rest of the eigenvectors.

Our proposed perturbation scheme is as follows:

1.

Compute the singular value decomposition of D, and let D = U ΘV T .

for some choice of σ 2 > 0.

with the proposed Σ, the numerator in equation 5 reduces to σ 2 ∆ T ∆ and hence κ = σ 2 .

Without any additional assumptions on S, this is the maximal value for κ.

Figure 3: Selected eigenvectors of the proposed Σ. The eigenvectors, which contain the principal directions of the distribution, have maxima and minima in adjacent locations.

Distributions drawn with these properties perform as object detectors as they can be used to detect existence (or nonexistence) of significant pixels at these locations.

We plot some of the eigenvectors for our proposed Σ with p 1 = p 2 = 28 in Figure 3 .

These eigenvectors are the principal directions of the perturbation distribution F , and the samples drawn from F contain a combination of these directions.

We see these samples will have sharp contrasts at certain locations.

This result is very intuitive: The perturbation scheme is created for a specific problem where boundaries for objects are assumed to exist, and large jumps in the magnitude of the distribution help our method recover these boundaries efficiently.

We conclude this section with a demonstration of the perturbation scheme using Gaussian noise.

In Figure 4 , we plot a digit from the MNIST dataset (LeCun et al., 1998) , along with instances obtained by independent perturbation and by our suggested distribution.

LEG-TV procedure has two tuning parameters: (i) F , which determines the structure of the perturbation; and (ii) L, which controls the sparsity of the chosen interpretation.

Regarding F , we propose to use a multivariate Gaussian distribution as it is easy to sample from.

For Σ, we propose a theoretically driven heuristic for determining the correlation structure of Σ in Section 4.2.

However, the choice of the magnitude of Σ, i.e. σ 2 , is left to the user.

If this quantity is chosen too low, then the added perturbations are small in magnitude, and the predictions of the neural network do not change, resulting in a LEG near zero.

On the other hand, with a very large value of σ 2 , the results have too much variance as some of the pixel values are set to the minimum or the maximum pixel intensity.

In our implementations, we find that setting σ 2 to be between 0.05 and 0.30 results in reasonable solutions.

We determine this range by computing perturbations of various sizes on numerous images using the VGG-19 classifier.

The provided range is found to create perturbations large enough to change the prediction probabilities but small enough to avoid major changes in the image.

Most of our presented results are given for σ 2 = 0.10.

For the choice of L, we propose two solutions: The first is the theoretically suggested quantity given in Theorem 1, although this often results in estimates that are too conservative.

Our second method is a heuristic based on some of the quantities in the optimization problem and we use this for our demonstrations.

We set L = K L L max where K is a constant between 0 and 1 and L max is the smallest value of L for which the solution in equation 4 would result with g = 0; i.e.

.

We use K L = 0.05 or K L = 0.10 in our implementations.

We note that is possible to obtain the solution for all L by using a parametric simplex solver (Vanderbei, 2014) , or by starting with a large initial L, then using the solution of the program as a warm-start for a smaller choice of L. Both approaches return the solution path for all L, and might be more desirable in practice than relying on heuristics.

In this section, we demonstrate the robustness and validity of our procedure by two numerical experiments.

In Section 5.1, we perform sanity checks as laid out by Adebayo et al. (2018b) , and show that the LEG-TV estimator fails to detect objects when the weights of the neural network are chosen randomly.

In Section 5.2, we implement a sensitivity analysis in which we use various saliency methods to compute regions of importance, and then perturb these regions in order to see their effect on the prediction.

For the deep learner, we use VGG-19 (Simonyan & Zisserman, 2014) .

For computational efficiency, we compute saliency maps on a 28 by 28 grid (i.e.γ ∈ R 28×28 ) although the standard input for VGG-19 is 224 by 224.

The perturbations on the image are scaled up by 8 via upsampling in order for the dimensions to match.

In Adebayo et al. (2018b) , the validity of saliency estimation procedures are tested by varying the weights of the neural network.

In a technique named, "cascading randomization", authors propose to replace the fitted weights of a CNN layer by layer, and compute the saliency scores with each change.

As a deep learner with randomly chosen weights should have no prediction power, one expects to see the same effect in the resulting saliency scores: namely, as more of the weights are perturbed, the explanation offered by interpretability methods should become more and more meaningless.

Surprisingly, Adebayo et al. (2018b) show that most commonly adopted interpretation procedures provide some saliency even after full randomization, and conclude that these methods act as edge detectors.

Our procedure treats the classifier as a black-box and the explanations offered by LEG-TV are based solely on the predictions made by the neural network.

During the sanity check, when the weights of the neural network are randomly perturbed, the predictions change significantly and no longer depend on the input.

Thus, we expect the local linear approximations of the underlying function to be flat, which would result in saliency scores of zero for all of the pixels.

Finally, small artifacts that might arise in this process, such as positive or negative saliency scores with no spatial structure, should be smoothed over due to the TV penalty, further robustifying our procedure.

In order to verify our intuition, we perform cascading randomization on the weights of a VGG-19 network.

For all of the images in our analysis, we find that the LEG-TV estimate,γ, is reduced to zero after randomization of either the top (i.e. logits) or the second top layer (i.e. second fully connected layer).

The results of our experiment for two images are given in Figure 5 .

It is seen that after the weights are perturbed, the LEG-TV method fails to detect any signal that could be used for interpretation.

In fact, due to penalization, the estimate is set to zero.

These results show that the interpretation given by our proposed method is reliable and is dependent on the classifier.

...

...

Figure 5 : Results of the sanity check with cascading randomization.

The network weights are replaced by random numbers in a cascading order, starting from the last layer.

LEG is equal to zero for all pixel values immediately after the first randomization.

For our second validity test, we use various interpretation models to compute regions of high importance.

We then mask these regions by decreasing the value of the pixels to zero which is equivalent to painting them black.

We compute and assess the difference of the predictions for the target class with each perturbation.

We compare our method against four alternatives: GradCAM (Selvaraju et al., 2017), LIME (Ribeiro et al., 2016) , SHAP (Lundberg & Lee, 2017) and C-Shapley (Chen et al., 2019) .

The last three methods are chosen as they are model-agnostic, like LEG, and do not make use of the architecture of the neural network.

GradCAM is chosen due to its popularity.

The saliency maps using C-Shapley and LEG-TV are computed for a 28 by 28 grid.

In order to make the comparison between the methods more fair, we downsize the saliency maps resulting from GradCAM, LIME and SHAP to the same size.

Interestingly, we find that this step improves the performance of these estimators; that is, the perturbations identified using the low resolution saliency maps result in faster drops in the predicted score.

For LEG-TV, LIME and SHAP, the saliency scores are computed using 3000 model evaluations, where as C-Shapley requires 3136 (28×28×4) evaluations.

For LEG-TV, we provide two solutions, a sparse solution which corresponds to a larger choice of the penalty parameter L and a noisy solution which is obtained with a smaller choice of L, denoted by LEG and LEG0, respectively.

We present the results for 500 images that are randomly chosen from a subsample of the ImageNet dataset (Deng et al., 2009) 2 .

The average of the log odds ratios across the 500 images are provided in Figure 6 .

We see that as the size of the perturbation increases, the predictions for the target class drop for all of the methods.

The slope is sharpest for SHAP and LEG0, suggesting that these two methods identify pixels that are crucial for the predictions.

Figure 6 : Results of sensitivity analysis.

Log of the predicted probability for the target class is plotted versus the size of the perturbation.

The locations for the perturbations are determined by the saliency procedures.

Predictions should decrease at a fast rate for interpretability methods that can reliably identify regions of importance.

In that regard, SHAP and LEG0 appear to be the most accurate in determining the critical pixels, followed by LEG, GradCAM, C-Shapley and LIME.

In Figure 7 , we plot the top 10% most salient pixels according to different procedures for three images in the dataset.

The pixels chosen by SHAP appear to correspond to specific a convolution pattern and the chosen region is not contiguous.

On the other hand, pixels identified by LEG-TV are visually meaningful to the human eye and contain pixels that are more likely to be relevant for the prediction.

LEG-TV selects different parts of the crane in the first image, and the face of the Pekinese dog in the second.

In the last image, where a soap dispenser is misclassified as a soda bottle, LEG-TV relates the classification to the label and the barcode of the bottle -parts that are often seen on soda bottles.

For the same image, LEG-TV also selects the fixtures in the background, which could have been mistaken by the classifier as the cap of the soda bottle.

We have proposed a statistical framework for saliency estimation that relies on local linear approximations.

Utilizing the new framework, we have built a computationally efficient saliency estimator that has theoretical guarantees.

Using our theoretical analysis, we have identified how the sample complexity of the estimator can be improved by altering the model evaluation scheme.

Finally, we have shown through empirical studies that (i) unlike most of its competitors, our method passes the recently proposed sanity checks for saliency estimation; and (ii) pixels identified through our approach are highly relevant for the predictions, and our method often chooses regions with higher saliency compared to regions suggested by its alternatives.

Our linear program can also be recast by a change of variables and setting α = Dg.

In this case, the elements of α correspond to differences between adjoint pixels.

This program can be written as:

+ is the pseudo-inverse of D and U 2 is related to the left singular vectors of D. More precisely, letting D = U ΘV T denote the singular value decomposition of D, U 2 is the submatrix that corresponds to the columns of U for which Θ j is zero.

The linearity constraint ensures that the differences between the adjoint pixels is proper.

Derivation of the alternative formulation follows from Theorem 1 in Gaines et al. (2018) and is omitted.

This formulation can be expressed in the standard augmented form, i.e. min Ax=b,x≥0 c T x, by writ-

where y = 1 n n i=1f (x i )x i and m = 2p 1 p 2 −p 1 −p 2 .

The γ coefficient in the original formulation can be obtained by setting

A.2 PROOF OF THEOREM 1

Our proof depends on the following lemma.

Lemma 2.

For L ≥ 2 D + 1 log (p 1 p 2 / ) /n, γ * is in the feasibility set with probability 1 − , that is

Proof.

For ease of notation, let

We also assume that the images have been rescaled so that the maximum value ofx i is 1 (without rescaling, the maximum would be given as the largest intensity, i.e. 255).

Since, the function values are also in the range given by [-2,2], we can bound |z i,j |, that is

Under review as a conference paper at ICLR 2020

The proof follows by applying the McDiarmid's inequality (Vershynin, 2018) for each row of the difference and then taking the supremum over the terms.

By application of McDiarmid's inequality, we have that

Let L = 2 D + 1 log (p 1 p 2 /2 ) /n.

Then, taking a union bound over all variables, we have

Now note that that the feasibility set for any L ≥ L contains that of L and thus γ * is automatically included.

We now present the proof of the theorem.

Note that the technique is based on the Confidence Set approach by Fan (2013) .

In the proof, we use γ to refer to vec(γ) for ease of presentation.

Proof.

First, let the high probability set for which Lemma 2 holds by A. All of the following statements hold true for A. We let ∆ = D (γ − γ * ) .

We know that Dγ 1 ≤ Dγ * 1 since both are in the feasibility set, as stated in Lemma 2.

Let α * = Dγ * ,α = Dγ and define S = {j : α * j = 0}, and the complement of S as S C .

By assumption of the Theorem, we have that the cardinality of S is s, i.e. |S| = s. Now let ∆ S as the elements of ∆ in S. Then, using the above statement, one can show that ∆ S 1 ≥ ∆ S C 1 .

Note,

and ∆ S 1 ≥ ∆ S C 1 follows immediately.

Furthermore

where the last line uses the previous result.

Additionally, note that

where the first inequality follows by Holder's inequality and the second follows from Lemma 2 and the fact that bothγ and γ * are in the feasibility set for L = 2 D + 1 log (p 1 p 2 / ) /n.

We further bound the right hand side of the inequality by using the previous result, which gives

Next, we bound ∆ 2 by combining the previous results.

Now, by assumption of the Theorem, we have that

Dividing both sides by ∆ 2 , we obtain that

@highlight

We propose a statistical framework and a theoretically consistent procedure for saliency estimation.