Recent literature has demonstrated promising results on the training of Generative Adversarial Networks by employing a set of discriminators, as opposed to the traditional game involving one generator against a single adversary.

Those methods perform single-objective optimization on some simple consolidation of the losses, e.g. an average.

In this work, we revisit the multiple-discriminator approach by framing the simultaneous minimization of losses provided by different models as a multi-objective optimization problem.

Specifically, we evaluate the performance of multiple gradient descent and the hypervolume maximization algorithm on a number of different datasets.

Moreover, we argue that the previously proposed methods and hypervolume maximization can all be seen as variations of multiple gradient descent in which the update direction computation can be done efficiently.

Our results indicate that hypervolume maximization presents a better compromise between sample quality and diversity, and computational cost than previous methods.

Generative Adversarial Networks (GANs) BID13 offer a new approach to generative modeling, using game-theoretic training schemes to implicitly learn a given probability density.

Prior to the emergence of GAN architectures, realistic generative modeling remained elusive.

When offering unparalleled realism, GAN training remains fraught with stability issues.

Commonly reported shortcomings involved in the GAN game are the lack of useful gradients provided by the discriminator, and mode collapse, i.e. lack of diversity in the generator's samples.

Considerable research effort has been devoted in recent literature in order to overcome training instability 1 within the GAN framework.

Some architectures such as BEGAN BID4 ) have applied auto-encoders as discriminators and proposed a new loss to help stabilize training.

Methods such as TTUR BID16 , in turn, have attempted to define schedules for updating the generator and discriminator differently.

The PacGAN algorithm (Lin et al., 2017) proposes to modify the discriminator's architecture which will receive m concatenated samples as input, while modifications to alternate updates in SGD were introduced in (Yadav et al., 2017) .

These samples are jointly classified as either real or generated, and authors show that this enforces sample diversity.

In SNGAN (Miyato et al., 2018) , authors introduce spectral normalization on the discriminator aiming to ensure Lipschitz continuity, which is empirically shown to consistently yield high quality samples when different sets of hyperparameters are used.

Recent works have proposed to tackle GANs instability issues using multiple discriminators.

Neyshabur et al. (2017) propose a GAN variation in which one generator is trained against a set of discriminators, where each discriminator sees a fixed random projection of the inputs.

Prior work, including GMAN BID9 has also explored training against multiple discriminators.

In this paper, we build upon Neyshabur et al.'s introduced framework and propose reformulating the average loss minimization aiming to further stabilize GAN training.

Specifically, we propose treating the loss signal provided by each discriminator as an independent objective function.

To achieve this, we simultaneously minimize the losses using multi-objective optimization techniques.

Namely, we exploit previously introduced methods in literature such as the multiple gradient descent algorithm (MGD) BID7 .

However, due to MGD's prohibitively high cost in the case of large neural networks, we propose the use of more efficient alternatives such as maximization of the hypervolume of the region defined between a fixed, shared upper bound on those losses, which we will refer to as the nadir point η * , and each of the component losses.

In contrast to Neyshabur et al. (2017) 's approach, where the average loss is minimized when training the generator, hypervolume maximization (HV) optimizes a weighted loss, and the generator's training will adaptively assign greater importance to feedback from discriminators against which it performs poorly.

Experiments performed on MNIST show that HV presents a good compromise in the computational cost-samples quality trade-off, when compared to average loss minimization or GMAN's approach (low quality and cost), and MGD (high quality and cost).

Also, the sensitivity to introduced hyperparameters is studied and results indicate that increasing the number of discriminators consequently increases the generator's robustness along with sample quality and diversity.

Experiments on CIFAR-10 indicate the method described produces higher quality generator samples in terms of quantitative evaluation.

Moreover, image quality and sample diversity are once more shown to consistently improve as we increase the number of discriminators.

In summary, our main contributions are the following:1.

We offer a new perspective on multiple-discriminator GAN training by framing it in the context of multi-objective optimization, and draw similarities between previous research in GANs variations and MGD, commonly employed as a general solver for multi-objective optimization.

2.

We propose a new method for training multiple-discriminator GANs: Hypervolume maximization, which weighs the gradient contributions of each discriminator by its loss.

The remainder of this document is organized as follows: Section 2 introduces definitions on multiobjective optimization and MGD.

In Section 3 we describe prior relevant literature.

Hypervolume maximization is detailed in Section 4, with experiments and results presented in Section 5.

Conclusions and directions for future work are drawn in Section 6.

In this section we provide some definitions regarding multi-objective optimization literature which will be useful in the next sections.

Henceforth, the boldface notation will be used to indicate vector-valued variables.

Multi-objective optimization.

A multi-objective optimization problem is defined as BID6 : DISPLAYFORM0 where K is the number of objectives, Ω is the variables space and x = [x 1 , x 2 , ..., x n ] T ∈ Ω is a decision vector or possible solution to the problem.

F : Ω → R K is a set of K-objective functions that maps the n-dimensional variables space to the K-dimensional objective space.

Pareto-dominance.

Let x 1 and x 2 be two decision vectors.

x 1 is said to dominate x 2 (denoted by x 1 ≺ x 2 ) if and only if f i (x 1 ) ≤ f i (x 2 ) for all i ∈ {1, 2, . . .

, K} and f j (x 1 ) < f j (x 2 ) for some j ∈ {1, 2, . . .

, K}. If a decision vector x is dominated by no other vector in Ω, x is said to be non-dominated.

Pareto-optimality.

A decision vector x * ∈ Ω is said to be Pareto-optimal if and only if there is no x ∈ Ω such that x ≺ x * , i.e. x * is a non-dominated solution.

The Pareto-optimal Set (PS) is defined as the set of all Pareto-optimal solutions x ∈ Ω, i.e., P S = {x ∈ Ω|x is Pareto optimal}.

The set of all objective vectors F(x) such that x is Pareto-optimal is called Pareto front (PF), that is P F = {F(x) ∈ R K |x ∈ P S}.Pareto-stationarity.

Pareto-stationarity is a necessary condition for Pareto-optimality.

For f k differentiable everywhere for all k, F is said to be Pareto-stationary at the point x if there exists a set of scalars α k , k ∈ {1, . . .

, K}, such that: DISPLAYFORM1 Multiple Gradient Descent.

Multiple gradient descent BID7 Schäffler et al., 2002; Peitz & Dellnitz, 2018) was proposed for the unconstrained case of multi-objective optimization of F(x) assuming a convex, continuously differentiable and smooth f k (x) for all k. MGD finds a common descent direction for all f k by defining the convex hull of all ∇f k (x) and finding the minimum norm element within it.

Consider w * given by: DISPLAYFORM2 w * will be either 0 in which case x is a Pareto-stationary point, or w * = 0 and then w * is a descent direction for all f i (x).

Similar to gradient descent, MGD consists in finding the common steepest descent direction w * t at each iteration t, and then updating parameters with a learning rate λ according to DISPLAYFORM3 3 RELATED WORK

While we would prefer to always have strong gradients from the discriminator during training, the vanilla GAN makes this difficult to ensure, as the discriminator quickly learns to distinguish real and generated samples BID12 , thus providing no meaningful error signal to improve the generator thereafter.

BID9 proposed the Generative Multi-Adversarial Networks (GMAN) which consist in training the generator against a softmax weighted arithmetic average of K different discriminators, according to Eq. 4.

DISPLAYFORM0 where DISPLAYFORM1 , β ≥ 0, and L D k is the loss of discriminator k and defined as DISPLAYFORM2 where D k (x) and G(z) are the outputs of the k-th discriminator and the generator, respectively.

The goal of using the proposed averaging scheme is to privilege worse discriminators and thus providing more useful gradients to the generator during training.

Experiments were performed with β = 0 (equal weights), β → ∞ (only worst discriminator is taken into account), β = 1, and β learned by the generator.

Models with K = {2, 5} were tested and evaluated using a proposed metric and the Inception score (Salimans et al., 2016) .

However, results showed that the simple average of discriminator's losses provided the best values for both metrics in most of the considered cases.

Opposed to GMAN, Neyshabur et al. (2017) proposed training a GAN with K discriminators using the same architecture.

Each discriminator D k sees a different randomly projected lower-dimensional version of the input image.

Random projections are defined by a randomly initialized matrix W k , which remains fixed during training.

Theoretical results provided show that the distribution induced by the generator G will converge to the real data distribution p data , as long as there is a sufficient number of discriminators.

Moreover, discriminative tasks in the projected space are harder, i.e. real and fake samples are more alike, thus avoiding early convergence of discriminators, which leads to common stability issues in GAN training such as mode-collapse BID12 .

Essentially, the authors trade one hard problem for K easier subproblems.

The losses of each discriminator L D k are the same as shown in Eq. 5.

However, the generator loss L G is defined as simply the sum of the losses provided by each discriminator, as shown in Eq. 6.

This choice of L G does not exploit available information such as the performance of the generator with respect to each discriminator.

DISPLAYFORM3 3.2 HYPERVOLUME MAXIMIZATION Consider a set of solutions S for a multi-objective optimization problem.

The hypervolume H of S is defined as BID10 : DISPLAYFORM4 , where µ is the Lebesgue measure and η * is a point dominated by all x ∈ S (i.e. f i (x) is upper-bounded by η), referred to as nadir point.

H(S) can be understood as the size of the space covered by {F(x)|x ∈ S} BID3 .The hypervolume was originally introduced as a quantitative metric for coverage and convergence of Pareto-optimal fronts obtained through population based algorithms BID5 .

Methods based on direct maximization of H exhibit favorable convergence even in challenging scenarios, such as simultaneous minimization of 50 objectives BID3

We introduce a variation of the GAN game such that the generator solves the following multi-objective problem: DISPLAYFORM0 where each DISPLAYFORM1 .., K}, is the loss provided by the k-th discriminator.

Training proceeds as the usual formulation BID13 , i.e. with alternate updates between the discriminators and the generator.

Updates of each discriminator are performed to minimize the loss described in Eq. 5.A natural choice for generator's updates is the MGD algorithm, described in Section 2.

However, computing the direction of steepest descent w * before every parameter update step, as required in MGD, can be prohibitively expensive for large neural networks.

Therefore, we propose an alternative scheme for multi-objective optimization and argue that both our proposal and previously published methods can all be viewed as performing computationally more efficient versions of MGD update rule without the burden of having to solve a quadratric program, i.e. computing w * , every iteration.

Fleischer BID10 has shown that maximizing H yields Pareto-optimal solutions.

Since MGD converges to a set of Pareto-stationary points, i.e. a super-set of the Pareto-optimal solutions, hypervolume maximization yields a sub-set of the solutions obtained using MGD.We exploit the above mentioned property and define the generator loss as the negative loghypervolume, as defined in Eq. 8: DISPLAYFORM0 where the nadir point coordinate η is an upper bound for all l k .

In Fig. 1 we provide an illustrative example for the case where K = 2.

The highlighted region corresponds to e V .

Since the nadir point η * is fixed, V will only be maximized, and consequently L G minimized, if each l k is minimized.

Figure 1 : 2D example of the objective space where the generator loss is being optimized.

DISPLAYFORM1 Moreover, by adapting the results shown in (Miranda & Zuben, 2016) , the gradient of L G with respect to any generator's parameter θ is given by: DISPLAYFORM2 In other words, the gradient can be obtained by computing a weighted sum of the gradients of the losses provided by each discriminator, whose weights are defined as the inverse distance to the nadir point components.

This formulation will naturally assign more importance to higher losses in the final gradient, which is another useful property of hypervolume maximization.

Nadir point selection.

It is evident from Eq. 9 that the selection of η directly affects the importance assignment of gradients provided by different discriminators.

Particularly, as the quantity min k {η − l k } grows, the multi-objective GAN game approaches the one defined by the simple average of l k .

Previous literature has discussed in depth the effects of the selection of η in the case of population-based methods BID1 BID7 .

However, those results are not readily applicable for the single-solution case.

As will be shown in Section 5, our experiments indicate that the choice of η plays an important role in the final quality of samples.

Nevertheless, this effect becomes less relevant as the number of discriminators increases.

Similarly to (Miranda & Zuben, 2016) , we propose an adaptive scheme for η such that at iteration t: η t = δ max k {l k,t }, where δ > 1 is a user-defined parameter which will be referred to as slack.

This enforces min k {η − l k } to be higher when max k {l k,t } is high and low otherwise, which induces a similar behavior as an average loss when training begins and automatically places more importance on the discriminators in which performance is worse as training progresses.

Extra discussion and an illustrative example of the adaptation scheme adopted is presented in Appendix G.Comparison to average loss minimization.

The upper bound proven by Neyshabur et al. (2017) assumes that the marginals of the real and generated distributions are identical along all random projections.

Average loss minimization does not ensure equally good approximation between the marginals along all directions.

In case of a trade-off between discriminators, i.e. if decreasing the loss on a given projection increases the loss with respect to another one, the distribution of losses can be uneven.

With HV on the other hand, especially when η is reduced throughout training, overall loss will be kept high as long as there are discriminators with high loss.

This objective tends to prefer central regions of a trade-off, in which all discriminators present a roughly equally low loss.

All methods described previously for the solution of GANs with multiple discriminators, i.e. average loss minimization (Neyshabur et al., 2017 ), GMAN's weighted average BID9 and hypervolume maximization can be defined as MGD-like two-step algorithms consisting of:Step 1 -consolidating all gradients into a single update direction (compute the set α 1,...,K );Step 2 -updating parameters in the direction returned in step 1.

Definition of Step 1 for the different methods studied here can be seen in the following: DISPLAYFORM0 Average loss minimization (Neyshabur et al., 2017) : BID9 : DISPLAYFORM1 DISPLAYFORM2 We performed three sets of experiments aiming to analyze the following aspects: (i) How alternative methods for training GANs with multiple discriminators perform in comparison to MGD; (ii) How alternative methods perform in comparison to each other in terms of sample quality and coverage; and (iii) Whether the behavior induced by HV improves the results with respect to the baseline methods.

Firstly, we exploited the relatively low dimensionality of MNIST and used it as testbed for a comparison of MGD with the other approaches, i.e. average loss minimization (AVG), GMAN's weighted average loss, and HV, proposed in this work.

Moreover, multiple initializations and slack combinations were evaluated in order to investigate how varying the number of discriminators affects robustness to those factors.

Then, experiments were performed with CIFAR-10 while increasing the number of discriminators.

We evaluated HV's performance compared to baseline methods, and the effect in samples quality.

We also analyzed the impact on the diversity of generated samples by using the stacked MNIST dataset (Srivastava et al., 2017) .

Samples of generators trained on stacked MNIST, CIFAR-10, CelebA, and Cats dataset are shown in the Appendix.

In all experiments performed, the same architecture, set of hyperparameters and initialization were used for both AVG, GMAN and our proposed method.

The only different aspect is the generator loss.

Unless stated otherwise, Adam (Kingma & Ba, 2014) was used to train all the models with learning rate, β 1 and β 2 set to 0.0002, 0.5 and 0.999, respectively.

Mini-batch size was set to 64.

The Fréchet Inception Distance (FID) BID16 was employed for comparison.

Details on FID computation can be found in Appendix A.

We employed MGD in our experiments with MNIST.

In order to do so, a quadratic program has to be solved prior to every parameters update.

For this, we used the Scipy's implementation of the Serial Least Square Quadratic Program solver 2 .Three and four fully connected layers with LeakyReLU activations were used for the generator and discriminator, respectively.

Dropout was also employed in the discriminator and the random projection layer was implemented as a randomly initialized norm-1 fully connected layer, reducing the vectorized dimensionality of MNIST from 784 to 512.

A pretrained LeNet (LeCun et al., 1998) was used for FID computation.

Experiments over 100 epochs with 8 discriminators are reported in Fig. 2 and Fig. 3 .

In Fig. 2 , box-plots refer to 30 independent computations of FID over 10000 images sampled from the generator which achieved the minimum FID at train time.

FID results are measured at train time over 1000 images and the best values are reported in Fig. 3 along with the necessary time to achieve it.

MGD outperforms all tested methods.

However, its cost per iteration does not allow its use in more relevant datasets other than MNIST.

Hypervolume maximization, on the other hand, performs closest to MGD than the considered baselines, while introducing no relevant extra cost.

In Fig. 4 , we analyze convergence in the Pareto-stationarity sense by plotting the norm of the update direction for each method, given by || K k=1 α k ∇l k ||.

All methods converged to similar norms, leading to the conclusion that different Pareto-stationary solutions will perform differently in terms of quality of samples.

FID as a function of wall-clock time is shown in Figure 22 (Appendix H).HV sensitivity to initialization and choice of δ.

Analysis of the sensitivity of the performance with the choice of the slack parameter δ and initialization was performed under the following setting: models were trained for 50 epochs on MNIST with hypervolume maximization using 8, 16, 24 discriminators.

Three independent runs (different initializations) were executed with each δ = {1.05, 1.5, 1.75, 2} and number of discriminators, totalizing 36 final models.

FIG1 reports the box-plots obtained for 5 FID independent computations using 10000 images, for each of the 36 models obtained under the setting previously described.

Results clearly indicate that increasing the number of discriminators yields much smaller variation in the FID obtained by the final model.

We evaluate the performance of HV compared to baseline methods using the CIFAR-10 dataset.

FID was computed with a pretrained ResNet BID15 .

ResNet was trained on the 10-class classification task of CIFAR-10 up to approximately 95% test accuracy.

DCGAN (Radford et al., 2015) and WGAN-GP BID14 were included in the experiments for FID reference.

Same architectures as in (Neyshabur et al., 2017) were employed for all multi-discriminators settings.

An increasing number of discriminators was used.

Inception score as well as FID computed with other models are included in Appendix C.In Fig. 6 , we report the box-plots of 15 independent evaluations of FID on 10000 images for the best model obtained with each method across 3 independent runs.

Results once more indicate that HV outperforms other methods in terms of quality of the generated samples.

Moreover, performance clearly improves as the number of discriminators grows.

Fig. 7 shows the FID at train time, i.e. measured with 1000 generated samples after each epoch, for the best models across runs.

Models trained against more discriminators clearly converge to smaller values.

We report the norm of the update direction || K k=1 α k ∇l k || for each method in FIG4 , Appendix C. : FID estimated over 1000 generated images at train time.

Models trained against more discriminators achieve lower FID.Cost under the multiple discriminator setting.

We highlight that even though training with multiple discriminators may be more computationally expensive when compared to conventional approaches, such framework supports fully parallel training of the discriminators, a feature which is not trivially possible in other GAN settings.

For example in WGAN, the discriminator is serially updated multiple times for each generator update.

In Fig. 10 at Appendix C, we provide a comparison between the wall-clock time per iteration between all methods evaluated.

Serial implementations of discriminators updates with 8 and 16 discriminators were faster than WGAN-GP.

We repeat the experiments in (Srivastava et al., 2017) aiming to analyze how the number of discriminators impacts the sample diversity of the corresponding generator when trained using hypervolume maximization.

The stacked MNIST dataset is employed and results reported in (Lin et al., 2017) are used for comparison.

HV results for 8, 16, and 24 discriminators were obtained with 10k and 26k generator images averaged over 10 runs.

The number of covered modes along with the KL divergence between the generated mode distribution and test data are reported in Table 1 998.0 ± 1.8 0.120 ± 0.004 HV -24 disc.998.3 ± 1.1 0.116 ± 0.003 26k HV -8 disc.776.8 ± 6.4 1.115 ± 0.007 HV -16 disc.1000.0 ± 0.0 0.088 ± 0.002 HV -24 disc.1000.0 ± 0.0 0.084 ± 0.002 Table 1 : Number of covered modes and reverse KL divergence for stacked MNIST.As in previous experiments, results improved as we increased the number of discriminators.

All evaluated models using HV outperformed DCGAN, ALI, Unrolled GAN and VEEGAN.

Moreover, HV with 16 and 24 discriminators achieved state-of-the-art coverage values.

Thus, the increase in models' capacity via using more discriminators directly resulted in an improvement in generator's coverage.

Training details as well as architectures information are presented in Appendix B.

In this work we have shown that employing multiple discriminators is a practical approach allowing us to trade extra capacity, and thereby extra computational cost, for higher quality and diversity of generated samples.

Such an approach is complimentary to other advances in GANs training and can be easily used together with other methods.

We introduced a multi-objective optimization framework for studying multiple discriminator GANs, and showed strong similarities between previous work and the multiple gradient descent algorithm.

The proposed approach was observed to consistently yield higher quality samples in terms of FID.

Furthermore, increasing the number of discriminators was shown to increase sample diversity and generator robustness.

Deeper analysis of the quantity || K k=1 α k ∇l k || is the subject of future investigation.

We hypothesize that using it as a penalty term might reduce the necessity of a high number of discriminators.

In BID16 , authors proposed to use as a quality metric the squared Fréchet distance BID11 between Gaussians defined by estimates of the first and second order moments of the outputs obtained through a forward pass in a pretrained classifier of both real and generated data.

They proposed the use of Inception V3 (Szegedy et al., 2016) for computation of the data representation and called the metric Fréchet Inception Distance (FID), which is defined as: DISPLAYFORM0 where m d , Σ d and m g , Σ g are estimates of the first and second order moments from the representations of real data distributions and generated data, respectively.

We employ FID throughout our experiments for comparison of different approaches.

However, for each dataset in which FID was computed, the output layer of a pretrained classifier on that particular dataset was used instead of Inception.

m d and Σ d were estimated on the complete test partitions, which are not used during training.

Architectures of the generator and discriminator are detailed in TAB4 , respectively.

Batch normalization was used in all intermediate convolutional and fully connected layers of both models.

We employed RMSprop to train all the models with learning rate and α set to 0.0001 and 0.9, respectively.

Mini-batch size was set to 64.

The setup in (Lin et al., 2017 ) is employed and we build 128000 and 26000 samples for train and test sets, respectively.

Table 4 presents the best FID (computed with a pretrained ResNet) achieved by each approach at train time, along with the epoch in which it was achieved, for each of 3 independent runs.

Train time FIDs are computed using 1000 generated images.

Table 4 : Best FID obtained for each approach on 3 independent runs.

FID is computed on 1000 generated images after every epoch.

In FIG4 , we report the norm of the update direction || K k=1 α k ∇l k || of the best model obtained for each method.

Interestingly, different methods present similar behavior in terms of convergence in the Pareto-stationarity sense, i.e. the norm upon convergence is lower for models trained against more discriminators, regardless of the employed method.

We computed extra scores using 10000 images generated by the best model reported in Table 4 , i.e. the same models utilized to generate the results shown in Fig. 6 .

Both Inception score and FID were computed with original implementations, while FID-VGG and FID-ResNet were computed using a VGG and a ResNet we pretrained.

Results are reported with respect to DCGAN's scores.

Table 5 : Scores of different methods measure on generated CIFAR-10 samples.

DCGAN scores are used as reference values, and results report are the ratio between given model and DCGAN scores.

Inception score is better when high, whereas FIDs are better when low.

In TAB9 we present a comparison of minimum FID-ResNet obtained during training, along with computation cost in terms of time and space for different GANs, with both 1 and 24 discriminators.

The computational cost of training GANs under a multiple discriminator setting is higher by design, in terms of both FLOPS and memory, if compared with single discriminators settings.

However, a corresponding shift in performance is the result of the additional cost.

This effect was consistently observed considering 4 different well-known approaches, namely DCGAN (Radford et al., 2015) , Least-square GAN (LSGAN) (Mao et al., 2017), and HingeGAN (Miyato et al., 2018) .

The architectures of all single discriminator models follow the DCGAN, described in (Radford et al., 2015) .

For the 24 discriminators models, we used the architecture described in (Neyshabur et al., 2017) , which consists in removing the the normalization layers from DCGAN's discriminator and further adding the projection layer, inline with previous experiments reported for CIFAR-10 upscaled to 64x64.

All models were trained with minibatch size of 64 during 150 epochs.

Adam BID18 Furthermore, wall-clock time per iteration for different numbers of discriminators is shown in Fig. 10 for experiments with CIFAR-10 with serial updates of discriminators.

Notice that while the increase in cost in terms of FLOPS and memory is unavoidable when multiple discriminators settings is employed, wall-clock time can be made close to single discriminators cases since training with respect to different discriminators can be implemented in parallel.

On the other hand, extra cost in time introduced by other frameworks such as WGAN-GP or SNGAN cannot be trivially recovered.

All results reported in previous sections using CIFAR-10 were obtained with an upscaled version of the dataset.

Here, we thus run experiments with the dataset in its original resolution aiming to contextualize our proposed approach with respect to previously introduced methods.

To do so, we repeated similar experiments as reported in Miyato et al. (2018)- TAB4 , for the model referred to as standard CNN.

The same architecture is employed and the spectral normalization is removed from the discriminators.

Moreover, the same projection input is added in each of the discriminators.

Results in terms of both FID and Inception score, evaluated on top of 5000 generated images as in (Miyato et al., 2018) as well as with 10000 images, are reported in TAB11 for our proposed approach and our implementation of (Miyato et al., 2018) , along with the FID measured using a ResNet classifier trained in advance.

As can be seen, the addition of the multiple discriminators setting along with hypervolume maximization yields a relevant shift in performance for the DCGAN-like generator, taking all evaluated metrics to levels of recently proposed GANs.

In this experiment, we verify whether the proposed multiple discriminators setting is capable of generating higher resolution images.

For that, we employed the CelebA at a size of 128x128.

We used a similar architecture for both generator and discriminators networks as described in the previous experiments.

A convolutional layer with 2048 feature maps was added to both generator and discriminators architectures due to the increase in the image size.

Adam optimizer with the same set of hyperparameters as for CIFAR-10 and CelebA 64x64 was employed.

We trained models with 6, 8, and 10 discriminators during 24 epochs.

Samples from each generator are shown in FIG7 .

We show the proposed multiple-discriminators setting scales to higher resolution even in the small dataset regime, by reproducing the experiments presented in BID17 .

We used the same architecture for the generator.

For the discriminator, we removed batch normalization from all layers and used stride equal to 1 at the last convolutional layer, after adding the initial projection step.

The Cats dataset 3 was employed, we followed the same pre-processing steps, which, in our case, yielded 1740 training samples with resolution of 256x256.

Our model is trained using 24 discriminators and Adam optimizer with the same hyperparameters as for CIFAR-10 and CelebA previously described experiments.

In FIG3 we show generator's samples after 288 training epochs.

One epoch corresponds to updating over 27 minibatches of size 64.Figure 18: Cats generated using 24 discriminators after 288 training epochs.

In this experiment we illustrate and confirm the results introduced in (Neyshabur et al., 2017) , showing the effect of using an increasing number of random projections to train a GAN.

We trained models using average loss minimization with 1 to 6 discriminators on the CelebA dataset for 15 epochs.

Samples from the generator obtained in the last epoch are shown in FIG4 .

Generated samples are closer to real data as the number of random projections (and discriminators, consequently) increases.

<|TLDR|>

@highlight

We introduce hypervolume maximization for training GANs with multiple discriminators, showing performance improvements in terms of sample quality and diversity. 