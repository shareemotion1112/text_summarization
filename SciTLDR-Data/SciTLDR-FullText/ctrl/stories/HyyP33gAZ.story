Class labels have been empirically shown useful in improving the sample quality of generative adversarial nets (GANs).

In this paper, we mathematically study the properties of the current variants of GANs that make use of class label information.

With class aware gradient and cross-entropy decomposition, we reveal how class labels and associated losses influence GAN's training.

Based on that, we propose Activation Maximization Generative Adversarial Networks (AM-GAN) as an advanced solution.

Comprehensive experiments have been conducted to validate our analysis and evaluate the effectiveness of our solution, where AM-GAN outperforms other strong baselines and achieves state-of-the-art Inception Score (8.91) on CIFAR-10.

In addition, we demonstrate that, with the Inception ImageNet classifier, Inception Score mainly tracks the diversity of the generator, and there is, however, no reliable evidence that it can reflect the true sample quality.

We thus propose a new metric, called AM Score, to provide more accurate estimation on the sample quality.

Our proposed model also outperforms the baseline methods in the new metric.

Generative adversarial nets (GANs) BID7 as a new way for learning generative models, has recently shown promising results in various challenging tasks, such as realistic image generation BID17 BID26 BID9 , conditional image generation BID12 BID2 BID13 , image manipulation ) and text generation BID25 .Despite the great success, it is still challenging for the current GAN models to produce convincing samples when trained on datasets with high variability, even for image generation with low resolution, e.g., CIFAR-10.

Meanwhile, people have empirically found taking advantages of class labels can significantly improve the sample quality.

There are three typical GAN models that make use of the label information: CatGAN BID20 builds the discriminator as a multi-class classifier; LabelGAN BID19 extends the discriminator with one extra class for the generated samples; AC-GAN BID18 jointly trains the real-fake discriminator and an auxiliary classifier for the specific real classes.

By taking the class labels into account, these GAN models show improved generation quality and stability.

However, the mechanisms behind them have not been fully explored BID6 .In this paper, we mathematically study GAN models with the consideration of class labels.

We derive the gradient of the generator's loss w.r.t.

class logits in the discriminator, named as class-aware gradient, for LabelGAN BID19 and further show its gradient tends to guide each generated sample towards being one of the specific real classes.

Moreover, we show that AC-GAN BID18 can be viewed as a GAN model with hierarchical class discriminator.

Based on the analysis, we reveal some potential issues in the previous methods and accordingly propose a new method to resolve these issues.

Specifically, we argue that a model with explicit target class would provide clearer gradient guidance to the generator than an implicit target class model like that in BID19 .

Comparing with BID18 , we show that introducing the specific real class logits by replacing the overall real class logit in the discriminator usually works better than simply training an auxiliary classifier.

We argue that, in BID18 , adversarial training is missing in the auxiliary classifier, which would make the model more likely to suffer mode collapse and produce low quality samples.

We also experimentally find that predefined label tends to result in intra-class mode collapse and correspondingly propose dynamic labeling as a solution.

The proposed model is named as Activation Maximization Generative Adversarial Networks (AM-GAN).

We empirically study the effectiveness of AM-GAN with a set of controlled experiments and the results are consistent with our analysis and, note that, AM-GAN achieves the state-of-the-art Inception Score (8.91) on CIFAR-10.In addition, through the experiments, we find the commonly used metric needs further investigation.

In our paper, we conduct a further study on the widely-used evaluation metric Inception Score BID19 and its extended metrics.

We show that, with the Inception Model, Inception Score mainly tracks the diversity of generator, while there is no reliable evidence that it can measure the true sample quality.

We thus propose a new metric, called AM Score, to provide more accurate estimation on the sample quality as its compensation.

In terms of AM Score, our proposed method also outperforms other strong baseline methods.

The rest of this paper is organized as follows.

In Section 2, we introduce the notations and formulate the LabelGAN BID19 and AC-GAN * BID18 ) as our baselines.

We then derive the class-aware gradient for LabelGAN, in Section 3, to reveal how class labels help its training.

In Section 4, we reveal the overlaid-gradient problem of LabelGAN and propose AM-GAN as a new solution, where we also analyze the properties of AM-GAN and build its connections to related work.

In Section 5, we introduce several important extensions, including the dynamic labeling as an alternative of predefined labeling (i.e., class condition), the activation maximization view and a technique for enhancing the AC-GAN * .

We study Inception Score in Section 6 and accordingly propose a new metric AM Score.

In Section 7, we empirically study AM-GAN and compare it to the baseline models with different metrics.

Finally we conclude the paper and discuss the future work in Section 8.

In the original GAN formulation BID7 , the loss functions of the generator G and the discriminator D are given as: DISPLAYFORM0 where D performs binary classification between the real and the generated samples and D r (x) represents the probability of the sample x coming from the real data.

The framework (see Eq. (1)) has been generalized to multi-class case where each sample x has its associated class label y ∈ {1, . . .

, K, K+1}, and the K+1 th label corresponds to the generated samples BID19 .

Its loss functions are defined as: DISPLAYFORM0 DISPLAYFORM1 where D i (x) denotes the probability of the sample x being class i.

The loss can be written in the form of cross-entropy, which will simplify our later analysis: DISPLAYFORM2 DISPLAYFORM3 where DISPLAYFORM4 H is the cross-entropy, defined as H(p, q)=− i p i log q i .

We would refer the above model as LabelGAN (using class labels) throughout this paper.

Besides extending the original two-class discriminator as discussed in the above section, BID18 proposed an alternative approach, i.e., AC-GAN, to incorporate class label information, which introduces an auxiliary classifier C for real classes in the original GAN framework.

With the core idea unchanged, we define a variant of AC-GAN as the following, and refer it as AC-GAN * : DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 where D r (x) and D f (x) = 1 − D r (x) are outputs of the binary discriminator which are the same as vanilla GAN, u(·) is the vectorizing operator that is similar to v(·) but defined with K classes, and C(x) is the probability distribution over K real classes given by the auxiliary classifier.

In AC-GAN, each sample has a coupled target class y, and a loss on the auxiliary classifier w.r.t.

y is added to the generator to leverage the class label information.

We refer the losses on the auxiliary classifier, i.e., Eq. FORMULA7 and FORMULA9 , as the auxiliary classifier losses.

The above formulation is a modified version of the original AC-GAN.

Specifically, we omit the auxiliary classifier loss E (x,y)∼G [H(u(y) , C(x))] which encourages the auxiliary classifier C to classify the fake sample x to its target class y. Further discussions are provided in Section 5.3.

Note that we also adopt the − log(D r (x)) loss in generator.

In this section, we introduce the class-aware gradient, i.e., the gradient of the generator's loss w.r.t.

class logits in the discriminator.

By analyzing the class-aware gradient of LabelGAN, we find that the gradient tends to refine each sample towards being one of the classes, which sheds some light on how the class label information helps the generator to improve the generation quality.

Before delving into the details, we first introduce the following lemma on the gradient properties of the cross-entropy loss to make our analysis clearer.

Lemma 1.

With l being the logits vector and σ being the softmax function, let σ(l) be the current softmax probability distribution andp denote the target probability distribution, then DISPLAYFORM0 For a generated sample x, the loss in LabelGAN is L lab DISPLAYFORM1 , as defined in Eq. (4).

With Lemma 1, the gradient of L lab G (x) w.r.t.

the logits vector l(x) is given as: DISPLAYFORM2 With the above equations, the gradient of L lab G (x) w.r.t.

x is: DISPLAYFORM3 where Figure 1: An illustration of the overlaid-gradient problem.

When two or more classes are encouraged at the same time, the combined gradient may direct to none of these classes.

It could be addressed by assigning each generated sample a specific target class instead of the overall real class.

DISPLAYFORM4 From the formulation, we find that the overall gradient w.r.t.

a generated example x is 1−D r (x), which is the same as that in vanilla GAN BID7 .

And the gradient on real classes is further distributed to each specific real class logit l k (x) according to its current probability ratio DISPLAYFORM5 Dr(x) .

As such, the gradient naturally takes the label information into consideration: for a generated sample, higher probability of a certain class will lead to a larger step towards the direction of increasing the corresponding confidence for the class.

Hence, individually, the gradient from the discriminator for each sample tends to refine it towards being one of the classes in a probabilistic sense.

That is, each sample in LabelGAN is optimized to be one of the real classes, rather than simply to be real as in the vanilla GAN.

We thus regard LabelGAN as an implicit target class model.

Refining each generated sample towards one of the specific classes would help improve the sample quality.

Recall that there are similar inspirations in related work.

BID4 showed that the result could be significantly better if GAN is trained with separated classes.

And AC-GAN BID18 introduces an extra loss that forces each sample to fit one class and achieves a better result.

In LabelGAN, the generator gets its gradients from the K specific real class logits in discriminator and tends to refine each sample towards being one of the classes.

However, LabelGAN actually suffers from the overlaid-gradient problem: all real class logits are encouraged at the same time.

Though it tends to make each sample be one of these classes during the training, the gradient of each sample is a weighted averaging over multiple label predictors.

As illustrated in Figure 1 , the averaged gradient may be towards none of these classes.

In multi-exclusive classes setting, each valid sample should only be classified to one of classes by the discriminator with high confidence.

One way to resolve the above problem is to explicitly assign each generated sample a single specific class as its target.

Assigning each sample a specific target class y, the loss functions of the revised-version LabelGAN can be formulated as: DISPLAYFORM0 DISPLAYFORM1 where v(y) is with the same definition as in Section 2.1.

The model with aforementioned formulation is named as Activation Maximization Generative Adversarial Networks (AM-GAN) in our paper.

And the further interpretation towards naming will be in Section 5.2.

The only difference between AM-GAN and LabelGAN lies in the generator's loss function.

Each sample in AM-GAN has a specific target class, which resolves the overlaid-gradient problem.

AC-GAN BID18 ) also assigns each sample a specific target class, but we will show that the AM-GAN and AC-GAN are substantially different in the following part of this section.

* is a combination of vanilla GAN and auxiliary classifier.

AM-GAN can naturally conduct adversarial training among all the classes, while in AC-GAN * , adversarial training is only conducted at the real-fake level and missing in the auxiliary classifier.

Both LabelGAN and AM-GAN are GAN models with K+1 classes.

We introduce the following cross-entropy decomposition lemma to build their connections to GAN models with two classes and the K-classes models (i.e., the auxiliary classifiers).

DISPLAYFORM0 With Lemma 2, the loss function of the generator in AM-GAN can be decomposed as follows: DISPLAYFORM1 The second term of Eq. (17) actually equals to the loss function of the generator in LabelGAN: DISPLAYFORM2 Similar analysis can be adapted to the first term and the discriminator.

Note that v r (x) equals to one.

Interestingly, we find by decomposing the AM-GAN losses, AM-GAN can be viewed as a combination of LabelGAN and auxiliary classifier (defined in Section 2.2).

From the decomposition perspective, disparate to AM-GAN, AC-GAN is a combination of vanilla GAN and the auxiliary classifier.

The auxiliary classifier loss in Eq. FORMULA0 can also be viewed as the cross-entropy version of generator loss in CatGAN: the generator of CatGAN directly optimizes entropy H(R(D(x))) to make each sample have a high confidence of being one of the classes, while AM-GAN achieves this by the first term of its decomposed loss H(R(v(x)), R(D(x))) in terms of cross-entropy with given target distribution.

That is, the AM-GAN is the combination of the cross-entropy version of CatGAN and LabelGAN.

We extend the discussion between AM-GAN and CatGAN in the Appendix B.

With the Lemma 2, we can also reformulate the AC-GAN * as a K+1 classes model.

Take the generator's loss function as an example: DISPLAYFORM0 In the K+1 classes model, the K+1 classes distribution is formulated as DISPLAYFORM1 AC-GAN introduces the auxiliary classifier in the consideration of leveraging the side information of class label, it turns out that the formulation of AC-GAN * can be viewed as a hierarchical K+1 classes model consists of a two-class discriminator and a K-class auxiliary classifier, as illustrated in FIG1 .

Conversely, AM-GAN is a non-hierarchical model.

All K+1 classes stay in the same level of the discriminator in AM-GAN.In the hierarchical model AC-GAN * , adversarial training is only conducted at the real-fake twoclass level, while misses in the auxiliary classifier.

Adversarial training is the key to the theoretical guarantee of global convergence p G = p data .

Taking the original GAN formulation as an instance, if generated samples collapse to a certain point x, i.e., p G (x) > p data (x), then there must exit another point x with p G (x ) < p data (x ).

Given the optimal D(x) = pdata (x) pG(x)+pdata(x) , the collapsed point x will get a relatively lower score.

And with the existence of higher score points (e.g. x ), maximizing the generator's expected score, in theory, has the strength to recover from the mode-collapsed state.

In practice, the p G and p data are usually disjoint , nevertheless, the general behaviors stay the same: when samples collapse to a certain point, they are more likely to get a relatively lower score from the adversarial network.

Without adversarial training in the auxiliary classifier, a mode-collapsed generator would not get any penalties from the auxiliary classifier loss.

In our experiments, we find AC-GAN is more likely to get mode-collapsed, and it was empirically found reducing the weight (such as 0.1 used in BID9 ) of the auxiliary classifier losses would help.

In Section 5.3, we introduce an extra adversarial training in the auxiliary classifier with which we improve AC-GAN * 's training stability and sample-quality in experiments.

On the contrary, AM-GAN, as a non-hierarchical model, can naturally conduct adversarial training among all the class logits.

In the above section, we simply assume each generated sample has a target class.

One possible solution is like AC-GAN BID18 , predefining each sample a class label, which substantially results in a conditional GAN.

Actually, we could assign each sample a target class according to its current probability estimated by the discriminator.

A natural choice could be the class which is of the maximal probability currently: y(x) argmax i∈{1,...,K} D i (x) for each generated sample x. We name this dynamic labeling.

According to our experiments, dynamic labeling brings important improvements to AM-GAN, and is applicable to other models that require target class for each generated sample, e.g. AC-GAN, as an alternative to predefined labeling.

We experimentally find GAN models with pre-assigned class label tend to encounter intra-class mode collapse.

In addition, with dynamic labeling, the GAN model remains generating from pure random noises, which has potential benefits, e.g. making smooth interpolation across classes in the latent space practicable.

Activation maximization is a technique which is traditionally applied to visualize the neuron(s) of pretrained neural networks BID16 b; BID5 ).The GAN training can be viewed as an Adversarial Activation Maximization Process.

To be more specific, the generator is trained to perform activation maximization for each generated sample on the neuron that represents the log probability of its target class, while the discriminator is trained to distinguish generated samples and prevents them from getting their desired high activation.

It is worth mentioning that the sample that maximizes the activation of one neuron is not necessarily of high quality.

Traditionally people introduce various priors to counter the phenomenon BID16 b) .

In GAN, the adversarial process of GAN training can detect unrealistic samples and thus ensures the high-activation is achieved by high-quality samples that strongly confuse the discriminator.

We thus name our model the Activation Maximization Generative Adversarial Network (AM-GAN).

Experimentally we find AC-GAN easily get mode collapsed and a relatively low weight for the auxiliary classifier term in the generator's loss function would help.

In the Section 4.3, we attribute mode collapse to the miss of adversarial training in the auxiliary classifier.

From the adversarial activation maximization view: without adversarial training, the auxiliary classifier loss that requires high activation on a certain class, cannot ensure the sample quality.

That is, in AC-GAN, the vanilla GAN loss plays the role for ensuring sample quality and avoiding mode collapse.

Here we introduce an extra loss to the auxiliary classifier in AC-GAN * to enforce adversarial training and experimentally find it consistently improve the performance: DISPLAYFORM0 where u(·) represents the uniform distribution, which in spirit is the same as CatGAN BID20 .Recall that we omit the auxiliary classifier loss E (x,y)∼G H u(y)] in AC-GAN * .

According to our experiments, E (x,y)∼G [H(u(y)] does improve AC-GAN * 's stability and make it less likely to get mode collapse, but it also leads to a worse Inception Score.

We will report the detailed results in Section 7.

Our understanding on this phenomenon is that: by encouraging the auxiliary classifier also to classify fake samples to their target classes, it actually reduces the auxiliary classifier's ability on providing gradient guidance towards the real classes, and thus also alleviates the conflict between the GAN loss and the auxiliary classifier loss.

One of the difficulties in generative models is the evaluation methodology BID22 .

In this section, we conduct both the mathematical and the empirical analysis on the widely-used evaluation metric Inception Score BID19 and other relevant metrics.

We will show that Inception Score mainly works as a diversity measurement and we propose the AM Score as a compensation to Inception Score for estimating the generated sample quality.

As a recently proposed metric for evaluating the performance of generative models, Inception Score has been found well correlated with human evaluation BID19 , where a publiclyavailable Inception model C pre-trained on ImageNet is introduced.

By applying the Inception model to each generated sample x and getting the corresponding class probability distribution C(x), Inception Score is calculated via DISPLAYFORM0 where E x is short of E x∼G andC DISPLAYFORM1 is the overall probability distribution of the generated samples over classes, which is judged by C, and KL denotes the Kullback-Leibler divergence.

As proved in Appendix D, E x KL C(x) C G can be decomposed into two terms in entropy: DISPLAYFORM2

A common understanding of how Inception Score works lies in that a high score in the first term H(C G ) indicates the generated samples have high diversity (the overall class probability distribution evenly distributed), and a high score in the second term −E x [H(C(x))] indicates that each individual sample has high quality (each generated sample's class probability distribution is sharp, i.e., it can be classified into one of the real classes with high confidence) BID19 .However, taking CIFAR-10 as an illustration, the data are not evenly distributed over the classes under the Inception model trained on ImageNet, which is presented in FIG3 .

It makes Inception Score problematic in the view of the decomposed scores, i.e., H(C G ) and −E x [H(C(x))].

Such as that one would ask whether a higher H(C G ) indicates a better mode coverage and whether a smaller H(C(x)) indicates a better sample quality.

DISPLAYFORM0 .

A common understanding of Inception Score is that: the value of H(C G ) measures the diversity of generated samples and is expected to increase in the training process.

However, it usually tends to decrease in practice as illustrated in (c). (C(x) ) score of CIFAR-10 training data is variant, which means, even in real data, it would still strongly prefer some samples than some others.

H(C(x)) on a classifier that pre-trained on CIFAR-10 has low values for all CIFAR-10 training data and thus can be used as an indicator of sample quality.

We experimentally find that, as in FIG2 , the value of H(C G ) is usually going down during the training process, however, which is expected to increase.

And when we delve into the detail of H(C(x)) for each specific sample in the training data, we find the value of H(C(x)) score is also variant, as illustrated in FIG3 , which means, even in real data, it would still strongly prefer some samples than some others.

The exp operator in Inception Score and the large variance of the value of H(C(x)) aggravate the phenomenon.

We also observe the preference on the class level in FIG3 , e.g., E x [H(C(x))]=2.14 for trucks, while E x [H(C(x))]=3.80 for birds.

It seems, for an ImageNet Classifier, both the two indicators of Inception Score cannot work correctly.

Next we will show that Inception Score actually works as a diversity measurement.

Since the two individual indicators are strongly correlated, here we go back to Inception Score's original formulation E x [KL(C(x) C G )].

In this form, we could interpret Inception Score as that it requires each sample's distribution C(x) highly different from the overall distribution of the generator C G , which indicates a good diversity over the generated samples.

As is empirically observed, a mode-collapsed generator usually gets a low Inception Score.

In an extreme case, assuming all the generated samples collapse to a single point, then C(x)=C G and we would get the minimal Inception Score 1.0, which is the exp result of zero.

To simulate mode collapse in a more complicated case, we design synthetic experiments as following: given a set of N points {x 0 , x 1 , x 2 , ..., x N −1 }, with each point x i adopting the distribution C(x i ) = v(i) and representing class i, where v(i) is the vectorization operator of length N , as defined in Section 2.1, we randomly drop m points, evaluate E x [KL(C(x) C G )] and draw the curve.

As is showed in Figure 5 , when N − m increases, the value of E x [KL(C(x) C G )] monotonically increases in general, which means that it can well capture the mode dropping and the diversity of the generated distributions.

DISPLAYFORM0 All of them works properly (going down) in the training process.

One remaining question is that whether good mode coverage and sample diversity mean high quality of the generated samples.

From the above analysis, we do not find any evidence.

A possible explanation is that, in practice, sample diversity is usually well correlated with the sample quality.

1 is generated, cannot be detected by E x [KL(C(x) C G )] score.

It means that with an accordingly pretrained classifier, E x [KL(C(x) C G )] score cannot detect intra-class level mode collapse.

This also explains why the Inception Network on ImageNet could be a good candidate C for CIFAR-10.

Exploring the optimal C is a challenge problem and we shall leave it as a future work.

However, there is no evidence that using an Inception Network trained on ImageNet can accurately measure the sample quality, as shown in Section 6.2.

To compensate Inception Score, we propose to introduce an extra assessment using an accordingly pretrained classifier.

In the accordingly pretrained classifier, most real samples share similar H(C(x)) and 99.6% samples hold scores less than 0.05 as showed in FIG3 , which demonstrates that H(C(x)) of the classifier can be used as an indicator of sample quality.

G is actually problematic when training data is not evenly distributed over classes, for that argmin H(C G ) is a uniform distribution.

To take theC train into account, we replace H(C G ) with a KL divergence betweenC train andC G .

So that DISPLAYFORM0 which requiresC G close toC train and each sample x has a low entropy C(x).

The minimal value of AM Score is zero, and the smaller value, the better.

A sample training curve of AM Score is showed in Figure 6 , where all indicators in AM Score work as expected.

1 1 Inception Score and AM Score measure the diversity and quality of generated samples, while FID BID10 measures the distance between the generated distribution and the real distribution.

Inception Score AM Score TAB0 7.04 ± 0.06 7.27 ± 0.07 --0.45 ± 0.00 0.43 ± 0.00 --GAN * 7.25 ± 0.07 7.31 ± 0.10 --0.40 ± 0.00 0.41 ± 0.00 --AC-GAN * 7.41 ± 0.09 7.79 ± 0.08 7.28 ± 0.07 7.89 ± 0.11 0.17 ± 0.00 0.16 ± 0.00 1.64 ± 0.02 1.01 ± 0.01 AC-GAN * + 8.56 ± 0.11 8.01 ± 0.09 10.25 ± 0.14 8.23 ± 0.10 0.10 ± 0.00 0.14 ± 0.00 1.04 ± 0.01 1.20 ± 0.01 LabelGAN 8.63 ± 0.08 7.88 ± 0.07 10.82 ± 0.16 8.62 ± 0.11 0.13 ± 0.00 0.25 ± 0.00 1.11 ± 0.01 1.37 ± 0.01 AM-GAN 8.83 ± 0.09 8.35 ± 0.12 11.45 ± 0.15 9.55 ± 0.11 0.08 ± 0.00 0.05 ± 0.00 0.88 ± 0.01 0.61 ± 0.01

To empirically validate our analysis and the effectiveness of the proposed method, we conduct experiments on the image benchmark datasets including CIFAR-10 and Tiny-ImageNet 2 which comprises 200 classes with 500 training images per class.

For evaluation, several metrics are used throughout our experiments, including Inception Score with the ImageNet classifier, AM Score with a corresponding pretrained classifier for each dataset, which is a DenseNet BID11 model.

We also follow BID18 and use the mean MS-SSIM BID23 of randomly chosen pairs of images within a given class, as a coarse detector of intra-class mode collapse.

A modified DCGAN structure, as listed in the Appendix F, is used in experiments.

Visual results of various models are provided in the Appendix considering the page limit, such as Figure 9 , etc.

The repeatable experiment code is published for further research 3 .

The first question is whether training an auxiliary classifier without introducing correlated losses to the generator would help improve the sample quality.

In other words, with the generator only with the GAN loss in the AC-GAN * setting. (referring as GAN * )As is shown in TAB0 , it improves GAN's sample quality, but the improvement is limited comparing to the other methods.

It indicates that introduction of correlated loss plays an essential role in the remarkable improvement of GAN training.

The usage of the predefined label would make the GAN model transform to its conditional version, which is substantially disparate with generating samples from pure random noises.

In this experiment, we use dynamic labeling for AC-GAN * , AC-GAN * + and AM-GAN to seek for a fair comparison among different discriminator models, including LabelGAN and GAN.

We keep the network structure and hyper-parameters the same for different models, only difference lies in the output layer of the discriminator, i.e., the number of class logits, which is necessarily different across models.

As is shown in TAB0 , AC-GAN * achieves improved sample quality over vanilla GAN, but sustains mode collapse indicated by the value 0.61 in MS-SSIM as in TAB1 .

By introducing adversarial Model Score ± Std.

DFM BID24 7.72 ± 0.13 Improved GAN BID19 8.09 ± 0.07 AC-GAN BID18 8.25 ± 0.07 WGAN-GP + AC BID9 8.42 ± 0.10 SGAN BID12 8.59 ± 0.12 AM-GAN (our work)8.91 ± 0.11 Splitting GAN BID8 8.87 ± 0.09 Real data 11.24 ± 0.12 Table 3 : Inception Score comparison on CIFAR-10.

Splitting GAN uses the class splitting technique to enhance the class label information, which is orthogonal to AM-GAN.training in the auxiliary classifier, AC-GAN * + outperforms AC-GAN * .

As an implicit target class model, LabelGAN suffers from the overlaid-gradient problem and achieves a relatively higher per sample entropy (0.124) in the AM Score, comparing to explicit target class model AM-GAN (0.079) and AC-GAN * + (0.102).

In the table, our proposed AM-GAN model reaches the best scores against these baselines.

We also test AC-GAN * with decreased weight on auxiliary classifier losses in the generator FORMULA0 relative to the GAN loss).

It achieves 7.19 in Inception Score, 0.23 in AM Score and 0.35 in MS-SSIM.

The 0.35 in MS-SSIM indicates there is no obvious mode collapse, which also conform with our above analysis.

AM-GAN achieves Inception Score 8.83 in the previous experiments, which significantly outperforms the baseline models in both our implementation and their reported scores as in Table 3 .

By further enhancing the discriminator with more filters in each layer, AM-GAN also outperforms the orthogonal work BID8 ) that enhances the class label information via class splitting.

As the result, AM-GAN achieves the state-of-the-art Inception Score 8.91 on CIFAR-10.

It's found in our experiments that GAN models with class condition (predefined labeling) tend to encounter intra-class mode collapse (ignoring the noise), which is obvious at the very beginning of GAN training and gets exasperated during the process.

In the training process of GAN, it is important to ensure a balance between the generator and the discriminator.

With the same generator's network structures and switching from dynamic labeling to class condition, we find it hard to hold a good balance between the generator and the discriminator: to avoid the initial intra-class mode collapse, the discriminator need to be very powerful; however, it usually turns out the discriminator is too powerful to provide suitable gradients for the generator and results in poor sample quality.

Nevertheless, we find a suitable discriminator and conduct a set of comparisons with it.

The results can be found in TAB0 .

The general conclusion is similar to the above, AC-GAN * + still outperforms AC-GAN * and our AM-GAN reaches the best performance.

It's worth noticing that the AC-GAN * does not suffer from mode collapse in this setting.

In the class conditional version, although with fine-tuned parameters, Inception Score is still relatively low.

The explanation could be that, in the class conditional version, the sample diversity still tends to decrease, even with a relatively powerful discriminator.

With slight intra-class mode collapse, the per-sample-quality tends to improve, which results in a lower AM Score.

A supplementary evidence, not very strict, of partial mode collapse in the experiments is that: the |

∂z | is around 45.0 in dynamic labeling setting, while it is 25.0 in the conditional version.

The LabelGAN does not need explicit labels and the model is the same in the two experiment settings.

But please note that both Inception Score and the AM Score get worse in the conditional version.

The only difference is that the discriminator becomes more powerful with an extended layer, which attests that the balance between the generator and discriminator is crucial.

We find that, without the concern of intra-class mode collapse, using the dynamic labeling makes the balance between generator and discriminator much easier.

DISPLAYFORM0 Note that we report results of the modified version of AC-GAN, i.e., AC-GAN * in TAB0 .

If we take the omitted loss E (x,y)∼G [H(u(y) , C(x))] back to AC-GAN * , which leads to the original AC-GAN (see Section 2.2), it turns out to achieve worse results on both Inception Score and AM Score on CIFAR-10, though dismisses mode collapse.

Specifically, in dynamic labeling setting, Inception Score decreases from 7.41 to 6.48 and the AM Score increases from 0.17 to 0.43, while in predefined class setting, Inception Score decreases from 7.79 to 7.66 and the AM Score increases from 0.16 to 0.20.This performance drop might be because we use different network architectures and hyper-parameters from AC-GAN BID18 .

But we still fail to achieve its report Inception Score, i.e., 8.25, on CIFAR-10 when using the reported hyper-parameters in the original paper.

Since they do not publicize the code, we suppose there might be some unreported details that result in the performance gap.

We would leave further studies in future work.

We plot the training curve in terms of Inception Score and AM Score in FIG6 .

Inception Score and AM Score are evaluated with the same number of samples 50k, which is the same as BID19 .

Comparing with Inception Score, AM Score is more stable in general.

With more samples, Inception Score would be more stable, however the evaluation of Inception Score is relatively costly.

A better alternative of the Inception Model could help solve this problem.

The AC-GAN * 's curves appear stronger jitter relative to the others.

It might relate to the counteract between the auxiliary classifier loss and the GAN loss in the generator.

Another observation is that the AM-GAN in terms of Inception Score is comparable with LabelGAN and AC-GAN * + at the beginning, while in terms of AM Score, they are quite distinguishable from each other.

In the CIFAR-10 experiments, the results are consistent with our analysis and the proposed method outperforms these strong baselines.

We demonstrate that the conclusions can be generalized with experiments in another dataset Tiny-ImageNet.

The Tiny-ImageNet consists with more classes and fewer samples for each class than CIFAR-10, which should be more challenging.

We downsize Tiny-ImageNet samples from 64×64 to 32×32 and simply leverage the same network structure that used in CIFAR-10, and the experiment result is showed also in TAB0 .

From the comparison, AM-GAN still outperforms other methods remarkably.

And the AC-GAN * + gains better performance than AC-GAN * .

In this paper, we analyze current GAN models that incorporate class label information.

Our analysis shows that: LabelGAN works as an implicit target class model, however it suffers from the overlaidgradient problem at the meantime, and explicit target class would solve this problem.

We demonstrate that introducing the class logits in a non-hierarchical way, i.e., replacing the overall real class logit in the discriminator with the specific real class logits, usually works better than simply supplementing an auxiliary classifier, where we provide an activation maximization view for GAN training and highlight the importance of adversarial training.

In addition, according to our experiments, predefined labeling tends to lead to intra-class mode collapsed, and we propose dynamic labeling as an alternative.

Our extensive experiments on benchmarking datasets validate our analysis and demonstrate our proposed AM-GAN's superior performance against strong baselines.

Moreover, we delve deep into the widelyused evaluation metric Inception Score, reveal that it mainly works as a diversity measurement.

And we also propose AM Score as a compensation to more accurately estimate the sample quality.

In this paper, we focus on the generator and its sample quality, while some related work focuses on the discriminator and semi-supervised learning.

For future work, we would like to conduct empirical studies on discriminator learning and semi-supervised learning.

We extend AM-GAN to unlabeled data in the Appendix C, where unsupervised and semi-supervised is accessible in the framework of AM-GAN.

The classifier-based evaluation metric might encounter the problem related to adversarial samples, which requires further study.

Combining AM-GAN with Integral Probability Metric based GAN models such as Wasserstein GAN could also be a promising direction since it is orthogonal to our work.

DISPLAYFORM0 Label smoothing that avoiding extreme logits value was showed to be a good regularization BID21 .

A general version of label smoothing could be: modifying the target probability of discriminator) BID19 proposed to use only one-side label smoothing.

That is, to only apply label smoothing for real samples: λ 1 = 0 and λ 2 > 0.

The reasoning of one-side label smoothing is applying label smoothing on fake samples will lead to fake mode on data distribution, which is too obscure.

DISPLAYFORM1 We will next show the exact problems when applying label smoothing to fake samples along with the log(1−D r (x)) generator loss, in the view of gradient w.r.t.

class logit, i.e., the class-aware gradient, and we will also show that the problem does not exist when using the − log(D r (x)) generator loss.

DISPLAYFORM2 The log(1−D r (x)) generator loss with label smoothing in terms of cross-entropy is DISPLAYFORM3 with lemma 1, its negative gradient is DISPLAYFORM4 DISPLAYFORM5 Gradient vanishing is a well know training problem of GAN.

Optimizing D r (x) towards 0 or 1 is also not what desired, because the discriminator is mapping real samples to the distribution with DISPLAYFORM6 The − log(D r (x)) generator loss with target [1−λ, λ] in terms of cross-entropy is DISPLAYFORM7 the negative gradient of which is DISPLAYFORM8 DISPLAYFORM9 Without label smooth λ, the − log(D r (x)) always * preserves the same gradient direction as log(1−D r (x)) though giving a difference gradient scale.

We must note that non-zero gradient does not mean that the gradient is efficient or valid.

The both-side label smoothed version has a strong connection to Least-Square GAN BID15 : with the fake logit fixed to zero, the discriminator maps real to α on the real logit and maps fake to β on the real logit, the generator in contrast tries to map fake sample to α.

Their gradient on the logit are also similar.

The auxiliary classifier loss of AM-GAN can also be viewed as the cross-entropy version of CatGAN: generator of CatGAN directly optimizes entropy H(R(D(x))) to make each sample be one class, while AM-GAN achieves this by the first term of its decomposed loss H(R(v(x)), R(D(x))) in terms of cross-entropy with given target distribution.

That is, the AM-GAN is the cross-entropy version of CatGAN that is combined with LabelGAN by introducing an additional fake class.

The discriminator of CatGAN maximizes the prediction entropy of each fake sample: DISPLAYFORM0 In AM-GAN, as we have an extra class on fake, we can achieve this in a simpler manner by minimizing the probability on real logits.

DISPLAYFORM1 If v r (K+1) is not zero, that is, when we did negative label smoothing BID19 , we could define R(v(K+1)) to be a uniform distribution.

DISPLAYFORM2 As a result, the label smoothing part probability will be required to be uniformly distributed, similar to CatGAN.

In this section, we extend AM-GAN to unlabeled data.

Our solution is analogous to CatGAN Springenberg (2015) .

Under semi-supervised setting, we can add the following loss to the original solution to integrate the unlabeled data (with the distribution denoted as p unl (x)): DISPLAYFORM0 C.2 UNSUPERVISED SETTING Under unsupervised setting, we need to introduce one extra loss, analogy to categorical GAN Springenberg (2015) : DISPLAYFORM1 where the p ref is a reference label distribution for the prediction on unsupervised data.

For example, p ref could be set as a uniform distribution, which requires the unlabeled data to make use of all the candidate class logits.

This loss can be optionally added to semi-supervised setting, where the p ref could be defined as the predicted label distribution on the labeled training data E x∼pdata [D(x)].

As a recently proposed metric for evaluating the performance of the generative models, the InceptionScore has been found well correlated with human evaluation BID19 , where a pre-trained publicly-available Inception model C is introduced.

By applying the Inception model to each generated sample x and getting the corresponding class probability distribution C(x), Inception Score is calculated via Inception Score = exp E x KL C(x) C G ,where E x is short of E x∼G andC G = E x [C(x)]

is the overall probability distribution of the generated samples over classes, which is judged by C, and KL denotes the Kullback-Leibler divergence which is defined as KL(p q) = i p i log pi qi = i p i log p i −

i p i log q i = −H(p) + H(p, q).An extended metric, the Mode Score, is proposed in BID3 to take the prior distribution of the labels into account, which is calculated via DISPLAYFORM0 where the overall class distribution from the training dataC train has been added as a reference.

We show in the following that, in fact, Mode Score and Inception Score are equivalent.

Lemma 3.

Let p(x) be the class probability distribution of the sample x, andp denote another probability distribution, then Ex H p(x),p = H Ex p(x) ,p .With Lemma 3, we have log(Inception Score) DISPLAYFORM1 log(Mode Score) DISPLAYFORM2

<|TLDR|>

@highlight

Understand how class labels help GAN training. Propose a new evaluation metric for generative models. 