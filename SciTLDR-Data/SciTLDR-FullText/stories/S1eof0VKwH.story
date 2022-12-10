The rich and accessible labeled data fuel the revolutionary success of deep learning.

Nonetheless, massive supervision remains a luxury for many real applications, boosting great interest in label-scarce techniques such as few-shot learning (FSL).

An intuitively feasible approach to FSL is to conduct data augmentation via synthesizing additional training samples.

The key to this approach is how to guarantee both discriminability and diversity of the synthesized samples.

In this paper, we propose a novel FSL model, called $\textrm{D}^2$GAN, which synthesizes Diverse and Discriminative features based on Generative Adversarial Networks (GAN).

$\textrm{D}^2$GAN secures discriminability of the synthesized features by constraining them to have high correlation with real features of the same classes while low correlation with those of different classes.

Based on the observation that noise vectors that are closer in the latent code space are more likely to be collapsed into the same mode when mapped to feature space, $\textrm{D}^2$GAN incorporates a novel anti-collapse regularization term, which encourages feature diversity by penalizing the ratio of the logarithmic similarity of two synthesized features and the logarithmic similarity of the latent codes generating them.

Experiments on three common benchmark datasets verify the effectiveness of $\textrm{D}^2$GAN by comparing with the state-of-the-art.

The rich and accessible labeled data fuel the revolutionary success of deep learning.

Nonetheless, massive supervision remains a luxury for many real applications, boosting great interest in label-scarce techniques such as few-shot learning (FSL).

An intuitively feasible approach to FSL is to conduct data augmentation via synthesizing additional training samples.

The key to this approach is how to guarantee both discriminability and diversity of the synthesized samples.

In this paper, we propose a novel FSL model, called D 2 GAN, which synthesizes Diverse and Discriminative features based on Generative Adversarial Networks (GAN).

D 2 GAN secures discriminability of the synthesized features by constraining them to have high correlation with real features of the same classes while low correlation with those of different classes.

Based on the observation that noise vectors that are closer in the latent code space are more likely to be collapsed into the same mode when mapped to feature space, D 2 GAN incorporates a novel anti-collapse regularization term, which encourages feature diversity by penalizing the ratio of the logarithmic similarity of two synthesized features and the logarithmic similarity of the latent codes generating them.

Experiments on three common benchmark datasets verify the effectiveness of D 2 GAN by comparing with the state-of-the-art.

@highlight

A new GAN based few-shot learning algorithm by synthesizing  diverse and discriminative Features

@highlight

A meta-learning method that learns a generative model that can augment the support set of a few-shot learner which optimizes a combination of losses.