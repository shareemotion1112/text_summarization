In order to alleviate the notorious mode collapse phenomenon in generative adversarial networks (GANs), we propose a novel training method of GANs in which certain fake samples can be reconsidered as real ones during the training process.

This strategy can reduce the gradient value that generator receives in the region where gradient exploding happens.

We show that the theoretical equilibrium between the generators and discriminations actually can be seldom realized in practice.

And this results in an unbalanced generated distribution that deviates from the target one, when fake datepoints overfit to real ones, which explains the non-stability of GANs.

We also prove that, by penalizing the difference between discriminator outputs and considering certain fake datapoints as real for adjacent real and fake sample pairs, gradient exploding can be alleviated.

Accordingly, a modified GAN training method is proposed with a more stable training process and a better generalization.

Experiments on different datasets verify our theoretical analysis.

In the past few years, Generative Adversarial Networks (GANs) Goodfellow et al. (2014) have been one of the most popular topics in generative models and achieved great success in generating diverse and high-quality images recently (Brock et al. (2019) ; Karras et al. (2019) ; ).

GANs are powerful tools for learning generative models, which can be expressed as a zero-sum game between two neural networks.

The generator network produces samples from the arbitrary given distribution, while the adversarial discriminator tries to distinguish between real data and generated data.

Meanwhile, the generator network tries to fool the discriminator network by producing plausible samples which are close to real samples.

When a final theoretical equilibrium is achieved, discriminator can never distinguish between real and fake data.

However, we show that a theoretical equilibrium often can not be achieved with discrete finite samples in datasets during the training process in practice.

Although GANs have achieved remarkable progress, numerous researchers have tried to improve the performance of GANs from various aspects ; Nowozin et al. (2016) ; Gulrajani et al. (2017) ; Miyato et al. (2018) ) because of the inherent problem in GAN training, such as unstability and mode collapse.

Arora et al. (2017) showed that a theoretical generalization guarantee does not be provided with the original GAN objective and analyzed the generalization capacity of neural network distance.

The author argued that for a low capacity discriminator, it can not provide generator enough information to fit the target distribution owing to lack of ability to detect mode collapse.

Thanh-Tung et al. (2019) argued that poor generation capacity in GANs comes from discriminators trained on discrete finite datasets resulting in overfitting to real data samples and gradient exploding when generated datapoints approach real ones.

As a result, Thanh-Tung et al. (2019) proposed a zero-centered gradient penalty on linear interpolations between real and fake samples (GAN-0GP-interpolation) to improve generalization capability and prevent mode collapse resulted from gradient exploding.

Recent work Wu et al. (2019) further studied generalization from a new perspective of privacy protection.

In this paper, we focus on mode collapse resulted from gradient exploding studied in Thanh-Tung et al. (2019) and achieve a better generalization with a much more stable training process.

Our contributions are as follows: discriminator with sigmoid function in the last layer removed D r = {x 1 , · · · , x n } the set of n real samples D g = {y 1 , · · · , y m } the set of m generated samples D f = {f 1 , · · · , f m } the candidate set of M 1 generated samples to be selected as real D F AR ⊂ {f 1 , · · · , f m } the set of M 0 generated samples considered as real 1.

We show that a theoretical equilibrium, when optimal discriminator outputs a constant for both real and generated data, is unachievable for an empirical discriminator during the training process.

Due to this fact, it is possible that gradient exploding happens when fake datapoints approach real ones, resulting in an unbalanced generated distribution that deviates from the target one.

2.

We show that when generated datapoints are very close to real ones in distance, penalizing the difference between discriminator outputs and considering fake as real can alleviate gradient exploding to prevent overfitting to certain real datapoints.

3.

We show that when more fake datapoints are moved towards a single real datapoint, gradients of the generator on fake datapoints very close to the real one can not be reduced, which partly explains the reason of a more serious overfitting phenomenon and an increasingly unbalanced generated distribution.

4.

Based on the zero-centered gradient penalty on data samples (GAN-0GP-sample) proposed in Mescheder et al. (2018) , we propose a novel GAN training method by considering some fake samples as real ones according to the discriminator outputs in a training batch to effectively prevent mode collapse.

Experiments on synthetic and real world datasets verify that our method can stabilize the training process and achieve a more faithful generated distribution.

In the sequel, we use the terminologies of generated samples (datapoints) and fake samples (datapoints) indiscriminately.

Tab.

1 lists some key notations used in the rest of the paper.

Unstability.

GANs have been considered difficult to train and often play an unstable role in training process Salimans et al. (2016) .

Various methods have been proposed to improve the stability of training.

A lot of works stabilized training with well-designed structures (Radford et al. (2015) ; Karras et al. (2018); Zhang et al. (2019); Chen et al. (2019) ) and utilizing better objectives (Nowozin et al. (2016); Zhao et al. (2016) ; ; Mao et al. (2017) ).

Gradient penalty to enforce Lipschitz continuity is also a popular direction to improve the stability including Gulrajani et al. (2017 ),Petzka et al. (2018 , Roth et al. (2017 ),Qi (2017 .

From the theoretical aspect, Nagarajan & Kolter (2017) showed that GAN optimization based on gradient descent is locally stable and Mescheder et al. (2018) proved local convergence for simplified zero-centered gradient penalties under suitable assumptions.

For a better convergence, a two time-scale update rule (TTUR) (Heusel et al. (2017) ) and exponential moving averaging (EMA) (Yazıcı et al. (2019) ) have also been studied.

Mode collapse.

Mode collapse is another persistent essential problem for the training of GANs, which means lack of diversity in the generated samples.

The generator may sometimes fool the discriminator by producing a very small set of high-probability samples from the data distribution.

Recent work (Arora et al. (2017) ; Arora et al. (2018) ) studied the generalization capacity of GANs and showed that the model distributions learned by GANs do miss a significant number of modes.

A large number of ideas have been proposed to prevent mode collapse.

Multiple generators are applied in Arora et al. (2017) , Ghosh et al. (2018) , Hoang et al. (2018) to achieve a more faithful distribution.

Mixed samples are considered as the inputs of discriminator in Lin et al. (2018) , Lucas et al. (2018) to convey information on diversity.

Recent work He et al. (2019) studied mode collapse from probabilistic treatment and Yamaguchi & Koyama (2019) from entropy of distribution.

In the original GAN Goodfellow et al. (2014) , the discriminator D maximizes the following objective:

The logistic sigmoid function σ(x) = 1 1+e −x is usually used in practice, leading to

and to prevent gradient collapse, the generator G maximizes

where D 0 is usually represented by a neural network.

Goodfellow et al. (2014) showed that the optimal discriminator D in Eqn.1 is

As the training progresses, p g will be pushed closer to p r .

If G and D are given enough capacity, a global equilibrium is reached when p r = p g , in which case the best strategy for D on supp(p r ) ∪ supp(p g ) is just to output 1 2 and the optimal value for Eqn.1 is 2 log( 1 2 ).

With finite training examples in training dataset D r in practice, an empirical version is applied to approximate Eqn.1, using

, where x i is from the set D r of n real samples and y i is from the set D g of m generated samples, respectively.

Mode collapse in the generator is attributed to gradient exploding in discriminator, according to Thanh-Tung et al. (2019) .

When a fake datapoint y 0 is pushed to a real datapoint x 0 and if |D(x 0 )− D(y 0 )| ≥ , is satisfied, the absolute value of directional derivative of D in the direction µ = x 0 − y 0 will approach infinity leading to gradient exploding:

Since the gradient ∇ y0 D(y 0 ) at y 0 outweights gradients towards other modes in a mini-batch, gradient exploding at datapoint y 0 will move multiple fake datapoints towards x 0 resulting in mode collapse.

Theoretically discriminator outputs a constant 1 2 when a global equilibrium is reached.

However in practice, discriminator can often easily distinguish between real samples and fake samples (Goodfellow et al. (2014) ; ), making a theoretical equilibrium unachievable.

Because the distribution p r of real data is unknown for discriminator, discriminator will always consider datapoints in the set D r of real samples as real while D g of generated samples as fake.

Even when generated distribution p g is equivalent to the target distribution p r , D r and D g is disjoint with probability 1 when they are sampled from two continuous distributions respectively (proposition 1 in Thanh-Tung et al. (2019) ).

In this case, actually p g is pushed towards samples in D r instead of the target distribution.

However, we show next because of the fact of an unachievable theoretical equilibrium for empirical discriminator during the training process, an unbalanced distribution would be generated that deviates from the target distribution.

Proposition 1.

For empirical discriminator in original GAN, unless p g is a discrete uniform distribution on D r , the optimal discriminator D output on D r and D g is not a constant 1 2 , since there exists a more optimal discriminator which can be constructed as a MLP with O(2d x ) parameters.

See Appendix A for the detailed proof.

If all the samples in D r can be remembered by discriminator and generator, and only if generated samples can cover D r uniformly, which means D g = D r , a theoretical equilibrium in discriminator can be achieved.

However, before generator covers all the samples in D r uniformly during the training process, the fact of an unachievable theoretical equilibrium makes it possible that there exists a real datapoint x 0 with a higher discriminator output than that of a generated datapoint y 0 .

When y 0 approaches x 0 very closely, gradient exploding and overfitting to a single datapoint happen, resulting an unbalanced distribution and visible mode collapse.

See the generated results on a Gaussian dataset of original GAN in Fig. 1a and 1e.

The generated distribution neither covers the target Gaussian distribution nor fits all the real samples in D r , making an unbalanced distribution visible.

Furthermore, in practice discriminator and generator are represented by a neural network with finite capacity and dataset D r is relatively huge, generator can never memorize every discrete sample resulting in a theoretical equilibrium unachievable.

In the following subsections, we are interested in the way of stabilizing the output of discriminator to alleviate gradient exploding to achieve a more faithful generated distribution.

Let's first consider a simplified scenario where a fake datapoint y 0 is close to a real datapoint x 0 .

Generator updates y 0 according to the gradient that the generator receives from the discriminator with respect to the fake datapoint y 0 , which can be computed as:

When y 0 approaches x 0 very closely and a theoretical discriminator equilibrium is not achieved here, namely D 0 (x 0 ) − D 0 (y 0 ) ≥ , the absolute value of directional derivative (∇ µ D 0 ) y0 in the direction µ = x 0 − y 0 at y 0 tends to explode and will outweigh directional derivatives in other

When y 0 is very close to x 0 , the norm of the gradient generator receives from the discriminator with respect to y 0 can be computed as:

If y 0 is in the neighborhood of x 0 , i.e.,

, where δ is a small positive value, we call {x 0 , y 0 } a close real and fake pair.

We are interested in reducing the approximated value of the gradient for a fixed pair {x 0 , y 0 } to prevent multiple fake datapoints overfitting to a single real one.

Note that the output of D 0 for real datapoint x 0 has a larger value than that of fake datapoint y 0 .

So for a fixed pair {x 0 , y 0 }, when D 0 (y 0 ) increases and D 0 (x 0 ) − D 0 (y 0 ) decreases, the target value decreases.

And, when D 0 (y 0 ) decreases and D 0 (x 0 ) − D 0 (y 0 ) increases, the target value increases, according to Eqn.

6.

Now we consider a more general scenario where for a real datapoint x 0 , in a set of n real samples, there are m 0 generated datapoints {y 1 , y 2 , · · · , y m0 } very close to x 0 in the set of m generated samples.

We are specially interested in the optimal discriminator output at x 0 and {y 1 , y 2 , · · · , y m0 }.

For simplicity, we make the assumption that discriminator outputs at these interested points are not affected by other datapoints in D r and D g .

We also assume discriminator has enough capacity to achieve optimum in this local region.

However, without any constraint, discriminator will consistently enlarge the gap between outputs for real datapoints and that for generated ones.

Thus, an extra constraint is needed to alleviate the difference between discriminator outputs on real and fake datapoints.

It comes naturally to penalize the L 2 norm of D 0 (x 0 ) − D 0 (y i ).

Denoting the discriminator output for x 0 , D 0 (x 0 ) as ξ 0 and D 0 (y i ) as ξ i , i = 1, · · · , m 0 , we have the following empirical discriminator objective:

where the interested term Based on Proposition 2, penalizing the difference between discriminator outputs on close real and fake pairs {x 0 , y i } can reduce the norm of ∇ y i L G (y i ) from Eqn.6, making it possible to move fake datapoints to other real datapoints instead of only being trapped at x 0 .

However in practice, it is hard to find the close real and fake pairs to penalize the corresponding difference between discriminator outputs.

If we directly penalize the L 2 norm of D 0 (x i ) − D 0 (y i )

when {x i , y i } are not a pair of close datapoints, ||∇ y i L G (y i )|| for y i may even get larger.

Consider D 0 (y i ) has a higher value than D 0 (x i ), which could happen when x i has more corresponding close fake datapoints than the real datapoint x yi corresponding to y i from Proposition 2.

Direct penalization will make D 0 (y i ) lower, then D 0 (x yi ) − D 0 (y i ) gets higher and ||∇ y i L G (y i )|| higher.

Thus in practice we could enforce a zero-centered gradient penalty of the form ||(∇D 0 ) v || 2 to stabilize discriminator output, where v can be real datapoints or fake datapoints.

Although Thanh-Tung et al. (2019) thought that discriminator can have zero gradient on the training dataset and may still have gradient exploding outside the training dataset, we believe a zero-centered gradient penalty can make it harder for discriminator to distinguish between real and fake datapoints and fill the gap between discriminator outputs on close real and fake pairs to prevent overfitting to some extent.

Fig. 1b and 1f alleviate overfitting phenomenon compared with no gradient penalty in Fig. 1a and 1e .

Thanh-Tung et al. (2019) proposed another zero-centered gradient penalty of the form ||(∇D 0 ) v || 2 , where v is a linear interpolation between real and fake datapoints, to prevent gradient exploding.

However, we consider it's not a very efficient method to fill the gap between discriminator outputs on close real and fake pairs.

To begin with, the results of direct linear interpolation between real and fake datapoints may not lie in supp(p r ) ∪ supp(p g ).

Although the author also considered the interpolation on latent codes, it needs an extra encoder which increases operational complexity.

Furthermore, for arbitrary pair of real and fake datapoints, the probability that linear interpolation between them lie where gradient exploding happens is close to 0, because large gradient happens when a fake datapoint approaches closely a real datapoint, resulting in the gap between discriminator outputs on close real and fake pairs hard to fill.

Based on Proposition 2, we also find that when more fake datapoints are moved to the corresponding real datapoint, ||∇ y i L G (y i )|| for a fake datpoint y i only to increase from Eqn.6.

It means with the training process going on, more fake datapoints tend to be attracted to one single real datapoint and it gets easier to attract much more fake datapoints to the real one.

It partly explains the unstability of GAN training process that especially during the later stage of training, similar generated samples are seen.

Compared with Fig. 1a, 1b and 1c at iter.100,000, Fig. 1e, 1f and 1g at iter.200,000 have a worse generalization and much more similar samples are generated with the training process going on.

In this subsection, we aim to make ||∇

y i L G (y i )||, i = 1, · · · , m 0 smaller for optimal empirical discriminator by considering some fake as real on close real and fake pairs based on the above discussions.

Suppose for each fake datapoint, it's considered as real datapoint with probability p 0 when training real datapoints, resulting in the following empirical discriminator objective:

where A is a binary random variable taking values in {0, 1} with Pr(A = 1) = p 0 and the interested term Note that only penalizing the difference between discriminator outputs on close real and fake pairs in Subsection 4.2 is just a special case of considering fake as real here when p 0 = 0.

Based on Proposition 3, considering fake datapoints as real with increasing probability p 0 for real datapoints training part can reduce the norm of ∇ y i L G (y i ) from Eqn.6.

It means when we consider more fake as real where large gradient happens for real training part, the attraction to the real datapoint for fake ones can be alleviate to make it easier to be moved to other real datapoints and prevent the overfitting to one single real datapoint.

Note that for a fixed p 0 , when the number m 0 of fake datapoints very close to the real one increases, more fake datapoints will be considered as real to alleviate the influences of increasing m 0 discussed in Subsection 4.2.

To overcome the problem of overfitting to some single datapoints and achieve a better generalization, we propose that fake samples generated by generator in real time can be trained as real samples in discriminator.

For original N real samples in a training batch in training process, we substitute them with N 0 real samples in D r and M 0 generated samples in D g , where N = N 0 + M 0 .

Our approach is mainly aimed at preventing large gradient in the region where many generated samples overfit one single real sample.

To find generated samples in these regions, we choose the generated ones with low discriminator output, owing to the reason that discriminator tends to have a lower output for the region with more generated datapoints approaching one real datapoint from Proposition 2.

Therefore, we choose needed M 0 generated samples denoted as set D F AR as real samples from a larger generated set D f containing M 1 generated samples {f 1 , f 2 , · · · , f M1 } according to corresponding discriminator output:

We also add a zero-centered gradient penalty on real datapoints Mescheder et al. (2018) based on the discussions in Subsection 4.2, resulting in the following empirical discriminator objective in a batch containing N real samples and M fake samples:

where

In practice, we usually let N = M .

Because some fake datapoints are trained as real ones, the zero-centered gradient penalty are actually enforced on the mixture of real and fake datapoints.

When we sample more generated datapoints for D f to decide the needed M 0 datapoints as real, the probability of finding the overfitting region with large gradient is higher.

When more fake datapoints in D F AR that are close to corresponding real ones are considered as real for training, it is equivalent to increase the value of p 0 in Subsection 4.3.

For a larger D F AR , the number of real samples N 0 will decrease for a fixed batchsize N and the speed to cover real ones may be slowed at the beginning of training owning to the reason that some fake datapoints are considered as real and discriminator will be not so confident to give fake ones a large gradient to move them to real ones.

Our method can stabilize discriminator output and prevent mode collapse caused by gradient exploding efficiently based on our theoretical analysis.

A more faithful generated distribution will be achieved in practice.

To test the effectiveness of our method in preventing an unbalanced distribution resulted from overfitting to only some real datapoints, we designed a dataset with finite real samples coming from a Gaussian distributions and trained MLP based GANs with different gradient penalties and our method on that dataset.

For gradient penalties in all GANs, the weight λ is set 10.

Training batch is set 64 and one quarter of the real training batch are generated samples picked from 256 generated samples according to discriminator output, namely M 0 = 16 and M 1 = 256 in Eqn.

11.

Learning rate is set 0.003 for both G and D. The result is shown in Fig.1 .

It can be observed that original GAN, GAN-0GP-sample and GAN-0GP-interpolation all have serious overfitting problem leading to a biased generated distribution with training process going on, while our method can generate much better samples with good generalization.

We also test our method on a mixture of 8 Gaussians dataset where random samples in different modes are far from each other.

The evolution of our method is depicted in Fig.2 .

We observe that although our method only covers 3 modes at the beginning, it can cover other modes gradually because our method alleviates the gradient exploding on close real and fake datapoints.

It is possible that fake datapoints are moved to other Gaussian modes when the attraction to other modes is larger than to the overfitted datapoints.

Hence, our method has the ability to find the uncovered modes to achieve a faithful distribution even when samples in high dimensional space are far from each other.

More synthetic experiments can be found in Appendix D. To test our method on real world data, we compare our method with GAN-0GP-sample on CIFAR-10 (Antonio et al. (2008)), CIFAR-100 (Antonio et al. (2008) ) and a more challenging dataset ImageNet (Russakovsky et al. (2015)) with ResNet-architectures similar with that in Mescheder et al. (2018) .

Inception score (Salimans et al. (2016) ) and FID (Heusel et al. (2017) ) are used as quantitative (2016) .

The FID score is evaluated on 10k generated images and statistics of data are calculated at the same scale of generation.

Better generation can be achieved with higher inception score and lower FID value.

The maximum number of iterations for CIFAR experiment is 500k, while for ImageNet is 600k because of training difficulty with much more modes.

We use the code from Mescheder et al. (2018) .

The weight λ for gradient penalty is also set 10.

Training batch is set 64 and for a better gradient alleviation on close real and fake datapoints, half of the real training batch are generated samples with M 0 = 32 and M 1 = 256 in Eqn.

11.

For CIFAR experiments, we use the RMSProp optimizer with α = 0.99 and a learning rate of 10 −4 .

For ImageNet experiments, we use the Adam optimizer with α = 0, β = 0.9 and TTUR with learning rates of 10 −4 and 3 × 10 −4 for the generator and discriminator respectively.

We use an exponential moving average with decay 0.999 over the weights to produce the final model.

The results on Inception score and FID are shown in Fig. 3 and 4.

Our method outperforms GAN-0GP-sample by a large margin.

As predicted in Section 5, the speed of our method to cover real ones could be slowed at the beginning of training with some fake considered as real.

However, our method can cover more modes and have a much better balanced generation than the baseline.

The losses of discriminator and generator during the process of CIFAR-10 training are shown in Fig.5 .

It can be observed that our method has a much more stable training process.

Owing to the overfitting to some single datapoints and an unbalanced generated distribution missing modes, the losses of discriminator and generator for GAN-0GP-sample gradually deviate from the optimal theoretical value, namely 2 log 2 ≈ 1.386 for discriminator and log 2 ≈ 0.693 for generator respectively.

However, our method has a much more stable output of discriminator to achieve the losses for discriminator and generator very close to theoretical case.

It proves practically that our method can stabilize discriminator output on close real and fake datapoints to prevent more datapoints from (a) (b) Figure 6 : Inception score and FID on ImageNet of GAN-0GP-sample and GAN-0GP-sample with our method trapped in a local region and has a better generalization.

The losses of discriminator and generator on CIFAR-100 and image samples can be found in Appendix D.

For the challenging ImageNet task, we train GANs to learn a generative model of all 1000 classes at resolution 64 × 64 with the limitation of our hardware.

However, our models are completely unsupervised learning models with no labels used instead of another 256 dimensions being used in latent code z as labels in Mescheder et al. (2018) .

The results in Fig. 6 show that our methods still outperforms GAN-0GP-sample on ImageNet.

Our method can produce samples of state of the art quality without using any category labels and stabilize the training process.

Random selected samples and losses of discriminator and generator during the training process can be found in Appendix D.

In this paper, we explain the reason that an unbalanced distribution is often generated in GANs training.

We show that a theoretical equilibrium for empirical discriminator is unachievable during the training process.

We analyze the affection on the gradient that generator receives from discriminator with respect to restriction on difference between discriminator outputs on close real and fake pairs and trick of considering fake as real.

Based on the theoretical analysis, we propose a novel GAN training method by considering some fake samples as real ones according to the discriminator outputs in a training batch.

Experiments on diverse datasets verify that our method can stabilize the training process and improve the performance by a large margin.

For empirical discriminator, it maximizes the following objective:

When p g is a discrete uniform distribution on D r , and generated samples in D g are the same with real samples in D r .

It is obvious that the discriminator outputs 1 2 to achieve the optimal value when it cannot distinguish fake samples from real ones.

For continues distribution p g , Thanh-Tung et al. (2019) has proved that an -optimal discriminator can be constructed as a one hidden layer MLP with O(d x (m + n)) parameters, namely D(x) ≥ 1 2 + 2 , ∀x ∈ D r and D(y) ≤ 1 2 − 2 , ∀y ∈ D g , where D r and D g are disjoint with probability 1.

In this case, discriminator objective has a larger value than the theoretical optimal version:

So the optimal discriminator output on D r and D g is not a constant 1 2 in this case.

Even discriminator has much less parameters than O(d x (m + n)), there exists a real datapoint x 0 and a generated datapoint y 0 satisfying D(x 0 ) ≥ 1 2 + 2 and D(y 0 ) ≤ 1 2 − 2 .

Whether p g is a discrete distribution only cover part samples in D r or a continues distribution, there exists a generated datapoint y 0 satisfying y 0 ∈ D r .

Assume that samples are normalized:

Let W 1 ∈ R 2×dx , W 2 ∈ R 2×2 and W 3 ∈ R 2 be the weight matrices, b ∈ R 2 offset vector and k 1 ,k 2 a constant, We can construct needed discriminator as a MLP with two hidden layer containing O(2d x ) parameters.

We set weight matrices

For any input v ∈ D r ∪ D g , the discriminator output is computed as:

where σ(x) = 1 1+e −x is the sigmoid function.

Let α = W 1 v − b, we have

where l < 1. Let β = σ(k 1 α), we have

as k 2 → ∞. Hence, for any input v ∈ D r ∪ D g , discriminator outputs

In this case, discriminator objective also has a more optimal value than the theoretical optimal version:

So the optimal discriminator output on D r and D g is also not a constant 1 2 in this case.

We rewrite f (ξ 0 , ξ 1 , · · · , ξ m0 ) here

To achieve the optimal value, let f (ξ i ) = 0, i = 0, · · · , m 0 and we have

It is obvious that ξ 1 = ξ 2 = · · · = ξ m0 = ξ.

Hence we have

We can solve

Substitute Eqn.

28 into Eqn.

26 and we get

We can also have from Eqn.

28 and Eqn.

26 respectively

Note that there must exist an optimal ξ 0 satisfying f (ξ 0 ) = 0 in Eqn.

29, so ξ 0 + ln(

We rewrite h(ξ 0 , ξ 1 , · · · , ξ m0 ) here

Let f (ξ i ) = 0, i = 0, · · · , m 0 and we have ξ 1 = ξ 2 = · · · = ξ m0 = ξ, 1 − σ(ξ 0 ) − 2nk 0 (ξ 0 − ξ) = 0,

We can solve ξ = − ln

The derivative of g(ξ 0 ) with respect to ξ 0 is computed as

g (ξ 0 ) > 0.

Hence ξ * 0 increases with p 0 increasing.

From Eqn.

33, we also have

we further know that ξ * increases and ξ * 0 − ξ * decreases with p 0 increasing.

(a) (b) Figure 7 : Losses of discriminator (not including regularization term) and generator on CIFAR-100 of GAN-0GP-sample and GAN-0GP-sample with our method

For synthetic experiment, the network architectures are the same with that in Thanh-Tung et al. (2019) .

While for real world data experiment, we use the similar architectures in Mescheder et al. (2018) .

We use Pytorch (Paszke et al. (2017) ) for development.

@highlight

 We propose a novel GAN training method by considering certain fake samples as real to alleviate mode collapse and stabilize training process.