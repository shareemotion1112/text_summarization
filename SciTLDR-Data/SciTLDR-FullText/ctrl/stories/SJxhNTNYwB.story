We present a new method for black-box adversarial attack.

Unlike previous methods that combined transfer-based and scored-based methods by using the gradient or initialization of a surrogate white-box model, this new method tries to learn a low-dimensional embedding using a pretrained model, and then performs efficient search within the embedding space to attack an unknown target network.

The method produces adversarial perturbations with high level semantic patterns that are easily transferable.

We show that this approach can greatly improve the query efficiency of black-box adversarial attack across different target network architectures.

We evaluate our approach on MNIST, ImageNet and Google Cloud Vision API, resulting in a significant reduction on the number of queries.

We also attack adversarially defended networks on CIFAR10 and ImageNet, where our method not only reduces the number of queries, but also improves the attack success rate.

The wide adoption of neural network models in modern applications has caused major security concerns, as such models are known to be vulnerable to adversarial examples that can fool neural networks to make wrong predictions (Szegedy et al., 2014) .

Methods to attack neural networks can be divided into two categories based on whether the parameters of the neural network are assumed to be known to the attacker: white-box attack and black-box attack.

There are several approaches to find adversarial examples for black-box neural networks.

The transfer-based attack methods first pretrain a source model and then generate adversarial examples using a standard white-box attack method on the source model to attack an unknown target network (Goodfellow et al., 2015; Madry et al., 2018; Carlini & Wagner, 2017; Papernot et al., 2016a) .

The score-based attack requires a loss-oracle, which enables the attacker to query the target network at multiple points to approximate its gradient.

The attacker can then apply the white-box attack techniques with the approximated gradient (Chen et al., 2017; Ilyas et al., 2018a; Tu et al., 2018) .

A major problem of the transfer-based attack is that it can not achieve very high success rate.

And transfer-based attack is weak in targeted attack.

On the contrary, the success rate of score-based attack has only small gap to the white-box attack but it requires many queries.

Thus, it is natural to combine the two black-box attack approaches, so that we can take advantage of a pretrained white-box source neural network to perform more efficient search to attack an unknown target black-box model.

In fact, in the recent NeurIPS 2018 Adversarial Vision Challenge (Brendel et al., 2018) , many teams transferred adversarial examples from a source network as the starting point to carry out black-box boundary attack (Brendel et al., 2017) .

N Attack also used a regression network as initialization in the score-based attack (Li et al., 2019a) .

The transferred adversarial example could be a good starting point that lies close to the decision boundary for the target network and accelerate further optimization.

P-RGF (Cheng et al., 2019) used the gradient information from the source model to accelerate searching process.

However, gradient information is localized and sometimes it is misleading.

In this paper, we push the idea of using a pretrained white-box source network to guide black-box attack significantly further, by proposing a method called TRansferable EMbedding based Black-box Attack (TREMBA).

TREMBA contains two stages: (1) train an encoder-decoder that can effectively generate adversarial perturbations for the source network with a low-dimensional embedding space; (2) apply NES (Natural Evolution Strategy) of (Wierstra et al., 2014) to the low-dimensional embedding space of the pretrained generator to search adversarial examples for the target network.

TREMBA uses global information of the source model, capturing high level semantic adversarial features that are insensitive to different models.

Unlike noise-like perturbations, such perturbations would have much higher transferablity across different models.

Therefore we could gain query efficiency by performing queries in the embedding space.

We note that there have been a number of earlier works on using generators to produce adversarial perturbations in the white-box setting (Baluja & Fischer, 2018; Xiao et al., 2018; Wang & Yu, 2019) .

While black-box attacks were also considered there, they focused on training generators with dynamic distillation.

These early approaches required many queries to fine-tune the classifier for different target networks, which may not be practical for real applications.

While our approach also relies on a generator, we train it as an encoder-decoder that produces a low-dimensional embedding space.

By applying a standard black-box attack method such as NES on the embedding space, adversarial perturbations can be found efficiently for a target model.

It is worth noting that the embedding approach has also been used in AutoZOOM (Tu et al., 2018) .

However, it only trained the autoencoder to reconstruct the input, and it did not take advantage of the information of a pretrained network.

Although it also produces structural perturbations, these perturbations are usually not suitable for attacking regular networks and sometimes its performance is even worse than directly applying NES to the images (Cheng et al., 2019; Guo et al., 2019) .

TREMBA, on the other hand, tries to learn an embedding space that can efficiently generate adversarial perturbations for a pretrained source network.

Compared to AutoZOOM, our new method produces adversarial perturbation with high level semantic features that could hugely affect arbitrary target networks, resulting in significantly lower number of queries.

We summarize our contributions as follows:

1.

We propose TREMBA, an attack method that explores a novel way to utilize the information of a pretrained source network to improve the query efficiency of black-box attack on a target network.

2.

We show that TREMBA can produce adversarial perturbations with high level semantic patterns, which are effective across different networks, resulting in much lower queries on MNIST and ImageNet especially for the targeted attack that has low transferablity.

3.

We demonstrate that TREMBA can be applied to SOTA defended models (Madry et al., 2018; Xie et al., 2018) .

Compared with other black-box attacks, TREMBA increases success rate by approximately 10% while reduces the number of queries by more than 50%.

There have been a vast literature on adversarial examples.

We will cover the most relevant topics including white-box attack, black-box attack and defense methods.

White-Box Attack White-box attack requires the full knowledge of the target model.

It was first discovered by (Szegedy et al., 2014) that adversarial examples could be found by solving an optimization problem with L-BFGS (Nocedal, 1980) .

Later on, other methods were proposed to find adversarial examples with improved success rate and efficiency (Goodfellow et al., 2015; Kurakin et al., 2016; Papernot et al., 2016b; Moosavi-Dezfooli et al., 2016) .

More recently, it was shown that generators can also construct adversarial noises with high success rate (Xiao et al., 2018; Baluja & Fischer, 2018) .

Black-Box Attack Black-box attack can be divided into three categories: transfer-based, score-based and decision-based.

It is well known that adversaries have high transferablity across different networks (Papernot et al., 2016a) .

Transfer-based methods generate adversarial noises on a source model and then transfer it to an unknown target network.

It is known that targeted attack is harder than untargeted attack for transfer-based methods, and using an ensemble of source models can improve the success rate (Liu et al., 2016) .

Score-based attack assumes that the attacker can query the output scores of the target network.

The attacker usually uses sampling methods to approximate the true gradient (Chen et al., 2017; Ilyas et al., 2018a; Li et al., 2019a; .

AutoZOOM tried to improve the query efficiency by reducing the sampling space with a bilinear transformation or an autoencoder (Tu et al., 2018) . (Ilyas et al., 2018b ) incorporated data and time prior to accelerate attacking.

In contrast to the gradient based method, (Moon et al., 2019) used combinatorial optimization to achieve good efficiency.

In decision-based attack, the attacker only knows the output label of the classifier.

Boundary attack and its variants are very powerful in this setting (Brendel et al., 2017; .

In NeutIPS 2018 Adversarial Vision Challenge (Brendel et al., 2018) , some teams combined transfer-based attack and decision-based attack in their attacking methods (Brunner et al., 2018) .

And in a similar spirit, N Attack also used a regression network as initialization in score-based attack (Li et al., 2019a) .

Gradient information from the surrogate model could also be used to accelerate the scored-based attack (Cheng et al., 2019) .

Defense Methods Several methods have been proposed to overcome the vulnerability of neural networks.

Gradient masking based methods add non-differential operations in the model, interrupting the backward pass of gradients.

However, they are vulnerable to adversarial attacks with the approximated gradient (Athalye et al., 2018; Li et al., 2019a) .

Adversarial training is the SOTA method that can be used to improve the robustness of neural networks.

Adversarial training is a minimax game.

The outside minimizer performs regular training of the neural network, and the inner maximizer finds a perturbation of the input to attack the network.

The inner maximization process can be approximated with FGSM (Goodfellow et al., 2015) , PGD (Madry et al., 2018) , adversarial generator (Wang & Yu, 2019) etc.

Moreover, feature denoising can improve the robustness of neural networks on ImageNet (Xie et al., 2018) .

be an input, and let F (x) be the output vector obtained before the softmax layer.

We denote F (x) i as the i-th component for the output vector and y as the label for the input.

For un-targeted attack, our goal is to find a small perturbation ?? such that the classifier predicts the wrong label, i.e. arg max F (x + ??) = y. And for targeted attack, we want the classifier to predicts the target label t, i.e. arg max F (x + ??) = t. The perturbation ?? is usually bounded by p norm: ?? p ??? ??, with a small ?? > 0.

Adversarial perturbations often have high transferablity across different DNNs.

Given a white-box source DNN F s with known architecture and parameters, we can transfer its white-box adversarial perturbation ?? s to a black-box target DNN F t with reasonably good success rate.

It is known that even if x + ?? s fails to be an adversarial example, ?? s can still act as a good starting point for searching adversarial examples using a score-based attack method.

This paper shows that the information of F s can be further utilized to train a generator, and performing search on its embedding space leads to more efficient black-box attacks of an unknown target network F t .

Adversarial perturbations can be generated by a generator network G. We explicitly divide the generator into two parts: an encoder E and a decoder D. The encoder takes the origin input x and output a latent vector z = E(x), where dim(z) dim(x).

The decoder takes z as the input and outputs an adversarial perturbation ?? = ?? tanh(D(z)) with dim(??) = dim(x).

In our new method, we will train the generator G so that ?? = ?? tanh(G(x)) can fool the source network F s .

Suppose we have a training set {(x 1 , y 1 ) , . . .

, (x n , y n )}, where x i denotes the input and y i denotes its label.

For un-targeted attack, we train the desired generator by minimizing the hinge loss used in the C&W attack (Carlini & Wagner, 2017) :

And for targeted, we use

where t denotes the targeted class and ?? is the margin parameter that can be used to adjust transferability of the generator.

A higher value of ?? leads to higher transferability to other models (Carlini & Wagner, 2017) .

We focus on ??? norm in this work.

By adding point-wise tanh function to an unnormalized output D(z), and scaling it with ??, ?? = ?? tanh(D(z)) is already bounded as ?? ??? < ??.

Therefore we employ this transformation, so that we do not need to impose the infinity norm constraint explicitly.

While hinge loss is employed in this paper, we believe other loss functions such the cross entropy loss will also work.

Given a new black-box DNN classifier F t (x), for which we can only query its output at any given point x.

As in (Ilyas et al., 2018a; Wierstra et al., 2014) , we can employ NES to approximate the gradient of a properly defined surrogate loss in order to find an adversarial example.

Denote the surrogate loss by L, rather than calculating ??? ?? L(x+??, y) directly, NES update ?? by using

The expectation can be approximated by taking finite samples.

And we could use the following equation to iteratively update ??:

where ?? is the learning rate, b is the minibatch sample size, ?? k is the sample from the gaussian distribution and [?????,??] represents a clipping operation, which projects ?? onto the ??? ball.

The sign function provides an approximation of the gradient, which has been widely used in adversarial attack (Ilyas et al., 2018a; Madry et al., 2018) .

However, it is observed that more effective attacks can be obtained by removing the sign function (Li et al., 2019b) .

Therefore in this work, we remove the sign function from Eqn (3) and directly use the estimated gradient.

Instead of performing search on the input space, TREMBA performs search on the embedding space z. The generator G explores the weakness of the source DNN F s so that D produces perturbations that can effective attack F s .

For a different unknown target network F t , we show that our method can still generate perturbations leading to more effective attack of F t .

Given an input x and its label y, we choose a starting point z 0 = E(x).

The gradient of z t given by NES can be estimated as:

where ?? k is the sample from the gaussian distribution N (z t , ?? 2 ).

Moreover, z t is updated with stochastic gradient descent.

The detailed procedure is presented in Algorithm 1.

We do not need to do projection explicitly since ?? already satisfies ?? ??? < ??.

Next we shall briefly explain why applying NES on the embedding space z can accelerate the search process.

Adversarial examples can be viewed as a distribution lying around a given input.

Usually this distribution is concentrated on some small regions, making the search process relatively slow.

After training on the source network, the adversarial perturbations of TREMBA would have high level semantic patterns that are likely to be adversarial patterns of the target network.

Therefore searching over z is like searching adversarial examples in a lower dimensional space containing likely adversarial patterns.

The distribution of adversarial perturbations in this space is much less concentrated.

It is thus much easier to find effective adversarial patterns in the embedding space.

We evaluated the number of queries versus success rate of TREMBA on undefended network in two datasets: MNIST (LeCun et al., 1998) and ImageNet (Russakovsky et al., 2015) .

Moreover, we evaluated the efficiency of our method on adversarially defended networks in CIFAR10 (Krizhevsky & Hinton, 2009 ) and ImageNet.

We also attacked Google Cloud Vision API to show TREMBA can generalize to truly black-box model.

1 We used the hinge loss from Eqn 1 and 2 as the surrogate loss for un-targeted and targeted attack respectively.

We compared TREMBA to four methods: (1) NES: Method introduced by (Ilyas et al., 2018a) , but without the sign function for reasons explained earlier.

(2) Trans-NES: Take an adversarial Algorithm 1 Black-Box adversarial attack on the embedding space Input:

Target Network F t ; Input x and its label y or the target class t; Encoder E; Decoder D; Standard deviation ??; Learning rate ??; Sample size b; Iterations T ; Bound for adversarial perturbation ?? Output: Adversarial perturbation ?? 1: z 0 = E(x) 2: for t = 1 to T do 3:

perturbation generated by PGD or FGSM on the source model to initialize NES.

(3) AutoZOOM: Attack target network with an unsupervised autoencoder described in (Tu et al., 2018) .

For fair comparisons with other methods, the strategy of choosing sample size was removed.

(4) P-RGF: Prior-guided random gradient-free method proposed in (Cheng et al., 2019) .

The P-RGF D (?? * ) version was compared.

We also combined P-RGF with initialization from Trans-NES PGD to form a more efficient method for comparison, denoted by Trans-P-RGF.

Since different methods achieve different success rates, we need to compare their efficiency at different levels of success rate.

For method i with success rate s i , the average number of queries is q i for all success examples.

Let q * denote the upper limit of queries, we modified the average number of queries to be q *

which unified the level of success rate and treated queries of failure examples as the upper limit on the number of queries.

Average queries sometimes could be misleading due to the the heavy tail distribution of queries.

Therefore we plot the curve of success rate at different query levels to show the detailed behavior of different attacks.

The upper limit on the number of queries was set to 50000 for all datasets, which already gave very high success rate for nearly all the methods.

Only correctly classified images were counted towards success rate and average queries.

And to fairly compare these methods, we chose the sample size to be the same for all methods.

We also added momentum and learning decay for optimization.

And we counted the queries as one if its starting point successfully attacks the target classifier.

The learning rate was fine-tuned for all algorithms.

We listed the hyperparameters and architectures of generators and classifiers in Appendix B and C.

We trained four neural networks on MNIST, denoted by ConvNet1, ConvNet1*, ConvNet2 and FCNet.

ConvNet1* and ConvNet1 have the same architecture but different parameters.

All the network achieved about 99% accuracy.

The generator G was trained on ConvNet1* using all images from the training set.

Each attack was tested on images from the MNIST test set.

The limit of ??? was ?? = 0.2.

We performed un-targeted attack on MNIST.

Table 1 lists the success rate and the average queries.

Although the success rate of TREMBA is slightly lower than Trans-NES in ConvNet1 and FCNet, their success rate are already close to 100% and TREMBA achieves about 50% reduction of queries compared with other attacks.

In contrast to efficient attack on ImageNet, P-RGF and Trans-P-RGF behaves very bad on MNIST.

Figure 4 .1 shows that TREMBA consistently achieves higher success rate at nearly all query levels.

We randomly divided the ImageNet validation set into two parts, containing 49000 and 1000 images respectively.

The first part was used as the training data for the generator G, and the second part was used for evaluating the attacks.

We evaluated the efficiency of all adversarial attacks on VGG19 (Simonyan & Zisserman, 2014) , Resnet34 (He et al., 2016) , DenseNet121 (Huang et al., 2017) and MobilenetV2 (Sandler et al., 2018) .

All networks were downloaded using torchvision package.

We set ?? = 0.03125. ) as the source model to improve transferablity (Liu et al., 2016) for both targeted and un-targeted attack.

TREMBA, Trans-NES, P-RGF and Trans-P-RGF all used the same source model for fair comparison.

We chose several target class.

Here, we show the result of attacking class 0 (tench) in Table 2 and Figure 2 .

And we leave the result of attacking other classes in Appendix A.1.

The average queries for TREMBA is about 1000 while nearly all the average queries for other methods are more than 6000.

TREMBA also achieves much lower queries for un-targeted attack on ImageNet.

The result is shown in Appendix A.2 due to space limitation.

And we also compared TREMBA with CombOpt (Moon et al., 2019) in the Appendix A.9.

Figure 3 shows the adversarial perturbations of different methods.

Unlike adversarial perturbations produced by PGD, the perturbations of TREMBA reflect some high level semantic patterns of the targeted class such as the fish scale.

As neural networks usually capture such patterns for classification, the adversarial perturbation of TREMBA would be more easy to transfer than the noise-like perturbation produced by PGD.

Therefore TREMBA can search very effectively for the target network.

More examples of perturbations of TREMBA are shown in Appendix A.3.

We performed attack on different ensembles of source model, which is shown in Appendix A.4.

TREMBA outperforms the other methods in different ensemble model.

And more source networks lead to better transferability for TREMBA, Trans-NES and Trans-P-RGF.

Varying ??: We also changed ?? and performed attack on ?? = 0.02 and ?? = 0.04.

As shown in Appendix A.5, TREMBA still outperforms the other methods despite using the G trained on ?? = 0.03125.

We also show the result of TREMBA for commonly used ?? = 0.05.

Sample size and dimension the embedding space: To justify the choice of sample size, we performed a hyperparameter sweep over b and the result is shown in Appendix A.6.

And we also changed the dimension of the embedding space for AutoZOOM and Trans-P-RGF.

As shown in Appendix A.7, the performance gain of TREMBA does not purely come from the diminishing of dimension of the embedding space.

This section presents the results for attacking defended networks.

We performed un-targeted attack on two SOTA defense methods on CIFAR10 and ImageNet.

MNIST is not studied since it is already robust against very strong white-box attacks.

For CIFAR10, the defense model was going through PGD minimax training (Madry et al., 2018) .

We directly used their model as the source network 2 , denoted by WResnet.

To test whether these methods can transfer to a defended network with a different architecture, we trained a defended ResNeXt (Xie et al., 2017) using the same method.

For ImageNet, we used the SOTA model 3 from (Xie et al., 2018) .

We used "ResNet152 Denoise" as the source model and transfered adversarial perturbations to the most robust "ResNeXt101 DenoiseAll".

Following the previous settings, we set ?? = 0.03125 for both CIFAR10 and ImageNet.

As shown in Table 3 , TREMBA achieves higher success rates with lower number of queries.

TREMBA achieves about 10% improvement of success rate while the average queries are reduced by more than 50% on ImageNet and by 80% on CIFAR10.

The curves in Figure 4 (a) and 4(b) show detailed behaviors.

The performance of AutoZOOM surpasses Trans-NES on defended models.

We suspect that low-frequency adversarial perturbations produced by AutoZOOM will be more suitable to fool the defended models than the regular networks.

However, the patterns learned by AutoZOOM are still worse than adversarial patterns learned by TREMBA from the source network.

An optimized starting point for TREMBA: z 0 = E(x) is already a good starting point for attacking undefended networks.

However, the capability of generator is limited for defended networks (Wang & Yu, 2019) .

Therefore, z 0 may not be the best starting point we can get from the defended source network.

To enhance the usefulness of the starting point, we optimized z on the source network by gradient descent and found

The method is denoted by TREMBA OSP (TREMBA with optimized starting point).

Figure 4 shows TREMBA OSP has higher success rate at small query levels, which means its starting point is better than TREMBA.

We also attacked the Google Cloud Vision API, which was much harder to attack than the single neural network.

Therefore we set ?? = 0.05 and perform un-targeted attack on the API, changing the top1 label to whatever is not on top1 before.

We chose 10 images for the ImageNet dataset and set query limit to be 500 due to high cost to use the API.

As shown Table 4 , TREMBA achieves much higher accuracy success rate and lower number of queries.

We show the example of successfully attacked image in Appendix A.8.

We propose a novel method, TREMBA, to generate likely adversarial patterns for an unknown network.

The method contains two stages: (1) training an encoder-decoder to generate adversarial perturbations for the source network; (2) search adversarial perturbations on the low-dimensional embedding space of the generator for any unknown target network.

Compared with SOTA methods, TREMBA learns an embedding space that is more transferable across different network architectures.

It achieves two to six times improvements in black-box adversarial attacks on MNIST and ImageNet and it is especially efficient in performing targeted attack.

Furthermore, TREMBA demonstrates great capability in attacking defended networks, resulting in a nearly 10% improvement on the attack success rate, with two to six times of reductions in the number of queries.

TREMBA opens up new ways to combine transfer-based and score-based attack methods to achieve higher efficiency in searching adversarial examples.

For targeted attack, TREMBA requires different generators to attack different classes.

We believe methods from conditional image generation (Mirza & Osindero, 2014 ) may be combined with TREMBA to form a single generator that could attack multiple targeted classes.

We leave it as a future work.

A EXPERIMENT RESULT A.1 TARGETED ATTACK ON IMAGENET Figure 9 shows result of the targeted attack on dipper, American chameleon, night snake, ruffed grouse and black swan.

TREMBA achieves much higher success rate than other methods at almost all queries level.

We used the same source model from targeted attack as the source model for un-targeted attack.

We report our evaluation results in Table 5 and Figure 5 .

Compared with Trans-P-RGF, TREMBA reduces the number of queries by more than a half in ResNet34, DenseNet121 and MobilenetV2.

Searching in the embedding space of generator remains very effective even when the target network architecture differs significantly from the networks in the source model.

Figure 5 : The success rate of un-targeted black-box adversarial attack at different query levels for undefended ImageNet models.

Figure 10 shows some examples of adversarial perturbations produced by TREMBA.

The first column is one image of the target class and other columns are examples of perturbations (amplified by 10 times).

It is easy to discover some features of the target class in the adversarial perturbation such as the feather for birds and the body for snakes.

We chose two more source ensemble models for evaluation.

The first ensemble contains VGG16 and Squeezenet.

And the second ensemble is consist of VGG16, Squeezenet and Googlenet.

Figure 6 shows our result for targeted attack for ImageNet.

We only compared Trans-NES PGD and Trans-P-RGF since they are the best variants from Trans-NES and P-RGF.

Figure 6 : We show the success rate at different query levels for targeted attack for different ensemble source networks.

V represents VGG16; S represents Squeezenet; G represents Googlenet; R represents Resnet18 A.5 VARYING ?? We chose ?? = 0.02 and ?? = 0.04 and performed targeted attack on ImageNet.

Although TREMBA used the same model that is trained on ?? = 0.03125, it still outperformed other methods, which shows that TREMBA can also generalize to different strength of adversarial attack with different ??.

For the commonly used ?? = 0.05, TREMBA also performs well.

The results are shown in Table 6,  Table 7 , and Figure 8 .

We performed a hyperparameter sweep over b on Densenet121 on un-targeted attack on ImageNet.

b = 20 may not be the best choice Trans-NES, but it is not the best for TREMBA, either.

Generally, the performance is not very sensitive to b, and TREMBA will also outperform other methods even if we fine-tune the sample size for all the methods.

We slightly changed the architecture of the autoencoder by adding max pooling layers and changing the number of filters and perform un-targeted attack on ImageNet.

More specifically, we added additional max pooling layers after the first and the fourth convolution layers and changed the number of filters of the last layer in the encoder to be 8.

Thus, the dimension of the embedding space would be 8 ?? 8 ?? 8.

And we also changed the factor of bilinear sampling in the decoder.

The remaining settings are the same in Appendix A.2.

As shown in Table 9 , this autoencoder is even worse than the original autoencoder despite small dimension of the embedding space.

In addition, we also changed to dimension of the data-dependent prior of Trans-P-RGF to match the dimension of TREMBA, whose performance is also not better than before.

They show that simply diminishing the size of the embedding space may not lead to better performance.

The performance gain of TREMBA comes beyond the effect of diminishing the dimension of the embedding space.

A.8 EXAMPLES OF ATTACKING GOOGLE CLOUD VISION API Figure 11 shows one example of attacking Google Cloud Vision API.

TREMBA successfully make the shark to be classified as green.

CombOpt is one of the SOTA score-based black-box attack.

We compared our method with it on the targeted and un-targeted attack on Imagenet.

The targeted attack is 0 and ?? = 0.03125.

As shown in Table 10 and Table 11 , TREMBA requires much lower queries than CombOpt.

It demonstrates the great improvement by combining the transfer-based and score-based attack.

B ARCHITECTURE OF CLASSIFIERS AND GENERATORS B.1 CLASSIFIER Table 12 lists the architectures of ConvNet1, ConvNet2 and FCNet.

The architecture of ResNeXt used in CIFAR10 is from https://github.com/prlz77/ResNeXt.pytorch.

We set the depth to be 20, the cardinality to be 8 and the widen factor to be 4.

Other architectures of classifiers are specified in the corresponding paper.

B.2 GENERATOR Table 13 lists the architectures of generator for three datasets.

For AutoZOOM, we find our architectures are not suitable and use the same generators in the corresponding paper.

We trained the generators with learning rate starting at 0.01 and decaying half every 50 epochs.

The whole training process was 500 epochs.

The batch size was determined by the memory of GPU.

Specifically, we set batch size to be 256 for MNIST and CIFAR10 defense model, 64 for ImageNet model.

All large ?? will work well for our method and we chose ?? = 200.0.

All the experiments were performed using pytorch on NVIDIA RTX 2080Ti.

Table 14 to 19 list the hyperparameters for all the algorithms.

The learning rate was fine-tuned for all the algorithms.

We set sample size b = 20 for all the algorithms for fair comparisons.

Table 18 : Hyperparameters for TREMBA.

Un-targeted Targeted

<|TLDR|>

@highlight

We present a new method that combines transfer-based and scored black-box adversarial attack, improving the success rate and query efficiency of black-box adversarial attack across different network architectures.