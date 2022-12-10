Generative models often use human evaluations to determine and justify progress.

Unfortunately, existing human evaluation methods are ad-hoc: there is currently no standardized, validated evaluation that: (1) measures perceptual fidelity, (2) is reliable, (3) separates models into clear rank order, and (4) ensures high-quality measurement without intractable cost.

In response, we construct Human-eYe Perceptual Evaluation (HYPE), a human metric that is (1) grounded in psychophysics research in perception, (2) reliable across different sets of randomly sampled outputs from a model, (3) results in separable model performances, and (4) efficient in cost and time.

We introduce two methods.

The first, HYPE-Time, measures visual perception under adaptive time constraints to determine the minimum length of time (e.g., 250ms) that model output such as a generated face needs to be visible for people to distinguish it as real or fake.

The second, HYPE-Infinity, measures human error rate on fake and real images with no time constraints, maintaining stability and drastically reducing time and cost.

We test HYPE across four state-of-the-art generative adversarial networks (GANs) on unconditional image generation using two datasets, the popular CelebA and the newer higher-resolution FFHQ, and two sampling techniques of model outputs.

By simulating HYPE's evaluation multiple times, we demonstrate consistent ranking of different models, identifying StyleGAN with truncation trick sampling (27.6% HYPE-Infinity deception rate, with roughly one quarter of images being misclassified by humans) as superior to StyleGAN without truncation (19.0%) on FFHQ.

Historically, likelihood-based estimation techniques served as the de-facto evaluation metric for generative models BID18 BID5 .

But recently, with the application of generative models to complex tasks such as image and text generation BID14 BID34 , likelihood or density estimation grew no longer tractable BID46 .

Moreover, for high-dimensional problems, even likelihood-based evaluation has been called into question BID46 .

Consequently, most generative tasks today resort to analyzing model outputs BID41 BID43 BID11 BID21 BID7 BID37 .

These output evaluation metrics consist of either automatic algorithms that do not reach the ideals of likelihood-based estimation, or ad-hoc human-derived methods that are unreliable and inconsistent BID41 BID11 .Consider the well-examined and popular computer vision task of realistic face generation BID14 .

Automatic algorithms used for this task include Inception Score (IS) BID43 and Fréchet Inception Distance (FID) BID17 .

Both have been discredited for evaluation on non-ImageNet datasets such as faces BID2 BID40 BID6 BID38 .

They are also much more sensitive to visual corruptions such as salt and pepper noise than to semantic distortions such as swirled images BID17 .

So, while automatic metrics are consistent and standardized, they cannot fully capture the semantic side of perceptual fidelity BID6 .Realizing the constraints of the available automatic metrics, many generative modeling challenges resort to summative assessments that are completely human BID41 BID43 BID11 .

These human measures are (1) ad-hoc, each executed in idiosyncrasy without proof of reliability or grounding to theory, and (2) high variance in their estimates BID43 BID11 BID33 .

These characteristics combine to a lack of reliability, and downstream, (3) a lack of clear separability between models.

Theoretically, given sufficiently large sample sizes of human evaluators and model outputs, the law of large numbers would smooth out the variance and reach eventual convergence; but this would occur at (4) a high cost and a long delay.

In this paper, we present HYPE (HUMAN EYE PERCEPTUAL EVALUATION) that addresses these criteria in turn.

It: (1) measures the perceptual fidelity of generative model outputs via a grounded method inspired by psychophysics methods in perceptual psychology, (2) is a reliable and consistent estimator, (3) is statistically separable to enable a comparative ranking, and (4) ensures a cost and time efficient method through modern crowdsourcing techniques such as training and aggregation.

We present two methods of evaluation.

The first, called HYPE time , is drawn directly from psychophysics literature BID22 ) and displays images using adaptive time constraints to determine the time-limited perceptual threshold a person needs to distinguish real from fake BID9 .

The HYPE time score is understood as the minimum time, in milliseconds, that a person needs to see the model's output before they can distinguish it as real or fake.

Small HYPE time scores indicate that model outputs can be identified even at a glance; large scores suggest that people need to dedicate substantial time and attention.

The second method, called HYPE ∞ , is derived from the first to make it simpler, faster, and cheaper while maintaining reliability.

It measures human deception from fake images with no time constraints.

The HYPE ∞ score is interpretable as the rate at which people mistake fake images and real images, given unlimited time to make their decisions.

We demonstrate HYPE's performance on unconditional generation of human faces using generative adversarial networks (GANs) BID14 .

We evaluate four state-of-the-art GANs: WGAN-GP BID16 , BEGAN BID4 , ProGAN BID20 , and the most recent StyleGAN BID21 .

First, we track progress across the years on the popular CelebA dataset BID28 .

We derive a ranking based on perception (HYPE time , in milliseconds) and error rate (HYPE ∞ , as a percentage) as follows: StyleGAN (439.4ms, 50.7%), ProGAN (363.7ms, 40.3%), BEGAN (111.1ms, 10.0%), WGAN-GP (100.0ms, 3.8%).

A score of 500ms on HYPE time indicates that outputs from the model become indistinguishable from real, when shown for 500ms or less, but any more would start to reveal notable differences.

A score of 50% on HYPE ∞ represents indistinguishable results from real, conditioned on the real training set, while a score above 50% through 100% represents hyper-realism in which generated images appear more real than real ones when drawn from a mixed pool of both.

Next, we test StyleGAN trained on the newer FFHQ dataset BID21 , comparing between outputs generated when sampled with and without the truncation trick, a technique used to prune low-fidelity generated images BID7 BID21 .

We find that outputs generated with the truncation trick (363.2ms, 27.6%) significantly outperforms those without it (240.7ms, 19.0%), which runs counter to scores reported by FID.HYPE indicates that GANs have clear, measurable perceptual differences between them.

HYPE produces identical rankings between HYPE time and HYPE ∞ .

We also find that even the best eval- Images on the right exhibit the highest HYPE scores, the highest human perceptual fidelity.

uated model, StyleGAN trained on FFHQ and sampled with the truncation trick, only performs at 27.6% HYPE ∞ , suggesting substantial opportunity for improvement.

Finally, we show that we can reliably reproduce these results with 95% confidence intervals using 30 human evaluators at $60 in a task that takes 10 minutes.

While important measures, we do not focus on diversity, overfitting, entanglement, training stability, and computational and sample efficiency of the model BID6 BID29 and instead aim to construct the gold standard for human perceptual fidelity.

We deploy HYPE as a rapid solution for researchers to measure their generative models, requiring just a single click to produce reliable scores and measure progress.

We deploy HYPE at https://hype.stanford.edu, where researchers can upload a model and retrieve a HYPE score in 10 minutes for $60.

Future work would extend HYPE to adapt to other generative tasks such as text generation or abstractive summarization.

Model creators can choose to perform two different evaluations and receive two different scores: the HYPE time score, which gathers time-limited perceptual thresholds to measure the psychometric function and report the minimum time people need to make accurate classifications, and the HYPE ∞ score, a simplified approach which assesses people's error rate under no time constraint.

HYPE displays a series of images one by one to crowdsourced evaluators on Amazon Mechanical Turk and asks the evaluators to assess whether each image is real or fake.

Half of the images are drawn from the model's training set (e.g., FFHQ or CelebA), which constitute the real images.

The other half are drawn from the model's output.

We use modern crowdsourcing training and quality control techniques to ensure high quality labels BID31 .

Our first method, HYPE time , measures time-limited perceptual thresholds.

It is rooted in psychophysics literature, a field devoted to the study of how humans perceive stimuli, to evaluate human time thresholds upon perceiving an image.

Our evaluation protocol follows the procedure known as the adaptive staircase method (Cornsweet, 1962) (see FIG1 ).

An image is flashed for a limited length of time, after which the evaluator is asked to judge whether it is real or fake.

If the evaluator consistently answers correctly, the staircase descends and flashes the next image with less time.

If the evaluator is incorrect, the staircase ascends and provides more time.

This process requires sufficient iterations to converge on the minimum time needed for each evaluator to sustain correct guesses in a sample-efficient manner BID9 , producing what is known as the psychometric function BID51 , the relationship of timed stimulus exposure to accuracy.

For example, for an easily distinguishable set of generated images, a human evaluator would immediately drop to the lowest millisecond exposure.

However, for a harder set, it takes longer to converge and the person would remain at a longer exposure level in order to complete the task accurately.

The modal time value is the evaluator's perceptual threshold: the shortest exposure time at which they can maintain effective performance BID9 BID15 ).

HYPE time displays three blocks of staircases for each evaluator.

An image evaluation begins with a 3-2-1 countdown clock, each number displaying for 500 ms.

The sampled image is then displayed for the current exposure time.

Immediately after each image, four perceptual mask images are rapidly displayed for 30ms each.

These noise masks are distorted to prevent visual afterimages and further sensory processing on the image afterwards BID15 ).

We generate masks from the test images, using an existing texture-synthesis algorithm BID35 .

Upon each submission, HYPE time reveals to the evaluator whether they were correct.

Image exposure times fall in the range [100ms, 1000ms], which we derive from the perception literature BID13 .

All blocks begin at 500ms and last for 150 images (50% generated, 50% real), values empirically tuned from prior work BID9 BID10 .

Exposure times are raised at 10ms increments and reduced at 30ms decrements, following the 3-up/1-down adaptive staircase approach.

This 3-up/1-down approach theoretically leads to a 75% accuracy threshold that approximates the human perceptual threshold BID27 BID15 BID9 .Every evaluator completes multiple staircases, called blocks, on different sets of images.

As a result, we observe multiple measures for the model.

We employ three blocks, to balance quality estimates against evaluators' fatigue ( BID25 BID42 .

We average the modal exposure times across blocks to calculate a final value for each evaluator.

Higher scores indicate a better model, whose outputs take longer time exposures to discern from real.

Building on the previous method, we introduce HYPE ∞ : a simpler, faster, and cheaper method after ablating HYPE time to optimize for speed, cost, and ease of interpretation.

HYPE ∞ shifts from a measure of perceptual time to a measure of human deception rate, given infinite evaluation time.

The HYPE ∞ score gauges total error on the task, enabling the measure to capture errors on both fake and real images, and effects of hyperrealistic generation when fake images look even more realistic than real images.

HYPE ∞ requires fewer images than HYPE time to find a stable value, at a 6x reduction in time and cost (10 minutes per evaluator instead of 60 minutes, at the same rate of $12 per hour).

Higher scores are better, like HYPE time : a HYPE ∞ value of 10% indicates that only 10% of images deceive people, whereas 50% indicates that people are mistaking real and fake images at chance, rendering fake images indistinguishable from real.

Scores above 50% suggest hyperrealistic images, as evaluators mistake images at a rate greater than chance, on average mistaking more fake images to be real than real ones and vice versa.

HYPE ∞ shows each evaluator a total of 100 images: 50 real and 50 fake.

We calculate the proportion of images that were judged incorrectly, and aggregate the judgments over the n evaluators on k images to produce the final score for a given model.

To ensure that our reported scores are consistent and reliable, we need to sample sufficient model outputs, select suitable real images for comparison, and hire, qualify, and appropriately pay enough evaluators.

To ensure a wide coverage of images, we randomly select the fake and real images provided to workers from a pool of 5000 images (see Sampling sufficient model outputs, below).Comparing results between single evaluators can be problematic.

To ensure HYPE is reliable, we must use a sufficiently large number of evaluators, n, which can be treated as a hyperparameter.

To determine a suitable number, we use our experimental results (further discussed in the Results section) to compute bootstrapped 95% confidence intervals (CI) across various values of n evaluators.

To obtain a high-quality pool of evaluators, each is required to pass a qualification task.

Such a pre-task filtering approach, sometimes referred to as a person-oriented strategy, is known to outperform process-oriented strategies that perform post-task data filtering or processing BID31 .

Our qualification task displays 100 images (50 real and 50 fake) with no time limits.

Evaluators pass if they correctly classify 65% of both real and fake images.

This threshold should be treated as a hyperparameter and may change depending upon the GANs used in the tutorial and the desired discernment ability of the chosen evaluators.

We choose 65% based on the cumulative binomial probability of 65 binary choice answers out of 100 total answers: there is only a one in one-thousand chance that an evaluator will qualify by random guessing.

Unlike in the staircase task itself, fake qualification images are drawn equally from multiple different GANs.

This is to ensure an equitable qualification across all GANs, as to avoid a qualification that is biased towards evaluators who are particularly good at detecting one type of GAN.

The qualification is designed to be taken occasionally, such that a pool of evaluators can assess new models on demand.

Payment.

Evaluators are paid a base rate of $1 for working on the qualification task.

To incentivize evaluators to remained engaged throughout the task, all further pay after the qualification comes from a bonus of $0.02 per correctly labeled image.

This pay rate typically results in a wage of approximately $12 per hour, which is above a minimum wage in our local state.

Sampling sufficient model outputs.

The selection of K images to evaluate from a particular model is a critical component of a fair and useful evaluation.

We must sample a large enough number of images that fully capture a model's generative diversity, yet balance that against tractable costs in the evaluation.

We follow existing work on evaluating generative output by sampling K = 5000 generated images from each model BID43 BID32 BID49 and K = 5000 real images from the training set.

From these samples, we randomly select images to give to each evaluator.

Datasets.

We evaluate on two datasets of human faces:1.

CelebA-64 BID28 is popular dataset for unconditional image generation, used since 2015.

CelebA-64 includes 202,599 images of human faces, which we align and crop to be 64 × 64 pixel images using a standard mechanism.

We train all models without using attributes.2.

FFHQ-1024 BID21 ) is a newer dataset released in 2018 with StyleGAN and includes 70,000 images of size 1024 × 1024 pixels.

Architectures.

We evaluate on four state-of-the-art models trained on CelebA-64: StyleGAN (Karras et al., 2018), ProGAN BID20 , BEGAN BID4 , and WGAN-GP BID16 .

We also evaluate on two types of sampling from StyleGAN trained on FFHQ-1024: with and without the truncation trick, which we denote StyleGAN trunc and StyleGAN no-trunc respectively.

For parity on our best models across datasets, StyleGAN trained on CelebA-64 is sampled with the truncation trick.

We train StyleGAN, ProGAN, BEGAN, and WGAN-GP on CelebA-64 using 8 Tesla V100GPUs for approximately 5 days.

We use the official released pretrained StyleGAN model on FFHQ-1024 BID21 .We sample noise vectors from the d-dimensional spherical Gaussian noise prior z ∈ R d ∼ N (0, I) during training and test times.

We specifically opted to use the same standard noise prior for comparison, yet are aware of other priors that optimize for FID and IS scores BID7 .

We select training hyperparameters published in the corresponding papers for each model.

We evaluate all models for each task with the two HYPE methods: (1) HYPE time and (2) HYPE ∞ .Evaluator recruitment.

We recruit 360 total human evaluators across our 12 evaluations, each of which included 30 evaluators, from Amazon Mechanical Turk.

Each completed a single evaluation in {CelebA-64, FFHQ-1024} × {HYPE time , HYPE ∞ }.

To maintain a between subjects study in this evaluation, we did not allow duplicate evaluators across tasks or methods.

In total, we recorded (4 CelebA-64 + 2 FFHQ-1024) models × 30 evaluators × 550 responses = 99, 000 total responses for our HYPE time evaluation and (4 CelebA-64 + 2 FFHQ-1024) models × 30 evaluators × 100 responses = 18, 000 total responses for our HYPE ∞ evaluation.

Metrics.

For HYPE time , we report the modal perceptual threshold in milliseconds.

For HYPE ∞ , we report the error rate as a percentage of images, as well as the breakdown of this rate on real and fake images individually.

To show that our results for each model are separable, we report a oneway ANOVA with Tukey pairwise post-hoc tests to compare all models within each {CelebA-64, FFHQ-1024} × {HYPE time , HYPE ∞ } combination.

As mentioned previously, reliability is a critical component of HYPE, as an evaluation is not useful if a researcher can re-run it and get a different answer.

To show the reliability of HYPE, we use bootstrap BID12 , a form of simulation, to simulate what the results would be if we resample with replacement from this set of labels.

Our goal is to see how much variation we may get in the outcome.

We therefore report evaluator 95% bootstrapped confidence intervals, along with standard deviation of the bootstrap sample distribution.

Confidence intervals (CIs) are defined as the region that captures where the modal exposure might be estimated to be if the same sampling procedure were repeated many times.

For this and all following results, bootstrapped confidence intervals were calculated by randomly sampling 30 evaluators with replacement from the original set of evaluators across 10, 000 iterations.

Note that bootstrapped CIs do not represent that there necessarily exists substantial uncertainty-our reported modal exposure (for HYPE time ) or detection rate (for HYPE ∞ ) is still the best point estimate of the value.

We discuss bootstrapped CIs for other numbers of evaluators later on in the Cost Tradeoffs section.

First, we report results using the above datasets, models and metrics using HYPE time .

Next, we demonstrate the HYPE ∞ 's results approximates the ones from HYPE time at a fraction of the cost and time.

Next, we trade off the accuracy of our scores with time.

We end with comparisons to FID.

CelebA-64.

We find that StyleGAN trunc resulted in the highest HYPE time score (modal exposure time), at a mean of 439.3ms, indicating that evaluators required nearly a half-second of exposure to accurately classify StyleGAN trunc images (Table 1 ).

StyleGAN trunc is followed by ProGAN at 363.7ms, a 17% drop in time.

BEGAN and WGAN-GP are both easily identifiable as fake, so they are tied in third place around the minimum possible exposure time available of 100ms.

Both BEGAN and WGAN-GP exhibit a bottoming out effect -reaching our minimum time exposure of 100ms quickly and consistently 1 .

This means that humans can detect fake generated images at 100ms and possibly lower.

Thus, their scores are identical and indistinguishable.

To demonstrate separability between StyleGAN trunc , ProGAN, BEGAN, and WGAN-GP together, we report results from a one-way analysis of variance (ANOVA) test between all four models, where each model's input is the list of modes from each model's 30 evaluators.

The ANOVA results confirm that there is a statistically significant omnibus difference (F (3, 29) = 83.5, p < 0.0001).

Pairwise post-hoc analysis using Tukey tests confirms that all pairs of models are separable (all p < 0.05), with the exception of BEGAN and WGAN-GP (n.s.).FFHQ-1024.

We find that StyleGAN trunc resulted in a higher exposure time than StyleGAN no-trunc , at 363.2ms and 240.7ms, respectively (Table 2) .

While the 95% confidence intervals that represent a very conservative overlap of 2.7ms, an unpaired t-test confirms that the difference between the two models is significant (t(58) = 2.3, p = 0.02).

Table 3 : HYPE ∞ on four GANs trained on CelebA-64.

Evaluators were deceived most often by StyleGAN trunc images, followed by ProGAN, BEGAN, and WGAN-GP.

We also display the breakdown of the deception rate on real and fake images individually; counterintuitively, real errors increase with the errors on fake images, because evaluators become more confused and distinguishing factors between the two distributions become harder to discern.

We observe a consistently separable difference between StyleGAN trunc and StyleGAN no-trunc and clear delineations between models TAB2 .

HYPE ∞ ranks StyleGAN trunc (27.6%) above StyleGAN no-trunc (19.0%) with no overlapping CIs.

Separability is confirmed by an unpaired t-test (t(58) = 8.3, p < 0.001).

One of HYPE's goals is to be cost and time efficient.

When running HYPE, there is an inherent tradeoff between accuracy and time, as well as between accuracy and cost.

This is driven by the law of large numbers: recruiting additional evaluators in a crowdsourcing task often produces more consistent results, but at a higher cost (as each evaluator is paid for their work) and a longer amount of time until completion (as more evaluators must be recruited and they must complete their work).To manage this tradeoff, we run an experiment with HYPE ∞ on StyleGAN trunc .

We completed an additional evaluation with 60 evaluators, and compute 95% bootstrapped confidence intervals, choosing from 10 to 120 evaluators (Figure 4) .

We see that the CI begins to converge around 30 evaluators, our recommended number of evaluators to recruit and the default that we build into our system.

As FID is one of the most frequently used evaluation methods for unconditional image generation, it is imperative to compare HYPE against FID on the same models (Table 5) .

We show through Spearman rank-order correlation coefficients that FID is correlated with neither human judgment measure, not HYPE time (ρ = −0.0286) nor with HYPE ∞ (ρ = −0.0857), where a Spearman correlation of -1.0 is ideal because lower FID and higher HYPE scores indicate stronger models.

Meanwhile, HYPE time and HYPE ∞ exhibit strong correlation (ρ = 0.9429), where 1.0 is ideal because they are directly related.

We calculate FID across the standard protocol of evaluating 50K generated and 50K real images for both CelebA-64 and FFHQ-1024, reproducing scores for StyleGAN no-trunc .

Table 5 : HYPE scores compared to FID.

We put an asterisk on the most realistic GAN for each score (lower the better for FID, higher the better for HYPE).

FID scores do not correlate fully with the human evaluation scores of HYPE ∞ on both CelebA-64 and FFHQ-1024 tasks.

FID scores were calculated using 50K real (CelebA-64 or FFHQ-1024) and 50K generated images for each model.

Cognitive psychology.

We leverage decades of cognitive psychology to motivate how we use stimulus timing to gauge the perceptual realism of generated images.

It takes an average of 150ms of focused visual attention for people to process and interpret an image, but only 120ms to respond to faces because our inferotemporal cortex has dedicated neural resources for face detection BID39 BID8 .

Perceptual masks are placed between a person's response to a stimulus and their perception of it to eliminate post-processing of the stimuli after the desired time exposure BID44 .

Prior work in determining human perceptual thresholds BID15 generates masks from their test images using the texture-synthesis algorithm BID35 .

We leverage this literature to establish feasible lower bounds on the exposure time of images, the time between images, and the use of noise masks.

Success of automatic metrics.

Common generative modeling tasks include realistic image generation BID14 , machine translation BID0 , image captioning BID48 , and abstract summarization BID30 , among others.

These tasks often resort to automatic metrics like the Inception Score (IS) BID43 and Fréchet Inception Distance (FID) BID17 to evaluate images and BLEU BID34 , CIDEr BID47 and METEOR BID1 scores to evaluate text.

While we focus on how realistic generated content appears, other automatic metrics also measure diversity of output, overfitting, entanglement, training stability, and computational and sample efficiency of the model BID6 BID29 BID2 .

Our metric may also capture one aspect of output diversity, insofar as human evaluators can detect similarities or patterns across images.

Our evaluation is not meant to replace existing methods but to complement them.

Limitations of automatic metrics.

Prior work has asserted that there exists coarse correlation of human judgment to FID BID17 and IS BID43 , leading to their widespread adoption.

Both metrics depend on the Inception v3 Network BID45 , a pretrained ImageNet model, to calculate statistics on the generated output (for IS) and on the real and generated distributions (for FID).

The validity of these metrics when applied to other datasets has been repeatedly called into question BID2 BID40 BID6 BID38 .

Perturbations imperceptible to humans alter their values, similar to the behavior of adversarial examples BID26 .

Finally, similar to our metric, FID depends on a set of real examples and a set of generated examples to compute high-level differences between the distributions, and there is inherent variance to the metric depending on the number of images and which images were chosen-in fact, there exists a correlation between accuracy and budget (cost of computation) in improving FID scores, because spending a longer time and thus higher cost on compute will yield better FID scores BID29 .

Nevertheless, this cost is still lower than paid human annotators per image.

Human evaluations.

Many human-based evaluations have been attempted to varying degrees of success in prior work, either to evaluate models directly BID11 BID33 or to motivate using automated metrics BID43 BID17 .

Prior work also used people to evaluate GAN outputs on CIFAR-10 and MNIST and even provided immediate feedback after every judgment BID43 .

They found that generated MNIST samples have saturated human performance-that is, people cannot distinguish generated numbers from real MNIST numbers, while still finding 21.3% error rate on CIFAR-10 with the same model BID43 .

This suggests that different datasets will have different levels of complexity for crossing realistic or hyper-realistic thresholds.

The closest recent work to ours compares models using a tournament of discriminators BID33 BID24 .

The design would likely affect humans' absolute thresholds, as cognitive load may be of consideration; the number of humans required per task may require significant increase if evaluating fairly across all possible categories.

Practically, the most valuable direction for the community to pursue with HYPE is likely one that includes the most difficult categories, especially when progress on those is hard to measure using automatic metrics.

In the case of text generation (translation, caption generation), HYPE time may require much longer and much higher range adjustments to the perceptual time thresholds for text comprehensibility than those used in visual perception BID24 .Future Work.

We plan to extend HYPE to different imaging datasets and imaging tasks such as conditional image generation, as well as to text and video, such as translation BID34 and video captioning BID23 .

Future work would also explore budget-optimal estimation of HYPE scores and adaptive estimation of evaluator quality BID19 .

Additional improvements involve identifying images that require more evaluators BID50 .

We also aim to build in faster time exposures under 100ms -ideally down to 13ms, the minimum time exposure of human perception (Potter et al., 2014) -for tasks that require that level of granularity.

Doing so requires careful engineering solution, since 100ms appears to be the minimum time that is trustable before we are throttled by JavaScript paint and rendering times on modern browsers.

We will investigate the ecological validity of our methods -that is, whether HYPE's evaluation is representative of how a person would perceive a GAN in everyday life.

For instance, HYPE shows evaluators whether they classified an image correctly immediately after they answer.

While this is standard practice in the psychophysics literature for staircase tasks, it likely does not reflect how one might encounter generated content in everyday life.

Notably, in pilot studies, we found that without such feedback, evaluators were far less consistent and our metric would not be stable.

Finally, we plan to investigate whether the reliability of HYPE may be impacted by the month or year at which it is run, as the population of available crowdsourced workers may differ across these factors.

Anecdotally, we have found HYPE to be reliable regardless of the time of day.7 CONCLUSION HYPE provides researchers with two human evaluation methods for GANs that (1) are grounded in psychopisics to measure human perceptual fidelity directly, (2) provide task designs that result in consistent and reliable results, (3) distinguishes between different model performances through separable results, (4) is cost and time efficient.

We report two metrics: HYPE time and HYPE ∞ .

HYPE time uses time perceptual thresholds where longer time constraints are more difficult to achieve because they give humans more time to interpret the generated content and observe artifacts.

HYPE ∞ reports the error rate under unlimited time, where higher rates indicate a more realistic set of outputs.

We demonstrate the efficacy of our approach on unconditional image generation across four GANs {StyleGAN, ProGAN, BEGAN, WGAN-GP} and two datasets of human faces {CelebA-64, FFHQ-1024}, with two types of output sampling on StyleGAN {with the truncation trick, without the truncation trick}. To encourage progress of generative models towards human-level visual fidelity, we deploy our evaluation system at https://hype.stanford.edu, so anyone can upload and evaluate their models based on HYPE at the click of a button.

A. CONFIDENCE INTERVALS

@highlight

HYPE is a reliable human evaluation metric for scoring generative models, starting with human face generation across 4 GANs.