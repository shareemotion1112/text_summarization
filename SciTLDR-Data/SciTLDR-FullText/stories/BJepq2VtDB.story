We propose NovoGrad, an adaptive stochastic gradient descent method with layer-wise gradient normalization and decoupled weight decay.

In our experiments on neural networks for image classification, speech recognition, machine translation, and language modeling, it performs on par or better than well tuned SGD with momentum and Adam/AdamW.  Additionally, NovoGrad (1) is robust to the choice of learning rate and weight initialization, (2) works well in a large batch setting, and (3) has two times smaller memory footprint than Adam.

The most popular algorithms for training Neural Networks (NNs) are Stochastic Gradient Descent (SGD) with momentum (Polyak, 1964; Sutskever et al., 2013) and Adam (Kingma & Ba, 2015) .

SGD with momentum is the preferred algorithm for computer vision, while Adam is the most commonly used for natural language processing (NLP) and speech problems.

Compared to SGD, Adam is perceived as safer and more robust to weight initialization and learning rate.

However, Adam has certain drawbacks.

First, as noted in the original paper (Kingma & Ba, 2015) , the second moment can vanish or explode, especially during the initial phase of training.

To alleviate this problem, a learning rate (LR) warmup (Goyal et al., 2017 ) is typically used.

Adam often leads to solutions that generalize worse than SGD (Wilson et al., 2017) , and to improve Adam regularization, Loshchilov & Hutter (2019) proposed AdamW with decoupled weight decay.

Our motivation for this work was to find an algorithm which: (1) performs equally well for image classification, speech recognition, machine translation, and language modeling, (2) is robust to learning rate and weight initialization, (3) has strong regularization properties.

We start with Adam, and then (1) replace the element-wise second moment with the layer-wise moment, (2) compute the first moment using gradients normalized by layer-wise second moment, (3) and decouple weight decay (similar to AdamW) from normalized gradients.

The resulting algorithm, NovoGrad, combines SGD's and Adam's strengths.

We applied NovoGrad to a variety of large scale problems -image classification, neural machine translation, language modeling, and speech recognition -and found that in all cases, it performs as well or better than Adam/AdamW and SGD with momentum.

NovoGrad belongs to the family of Stochastic Normalized Gradient Descent (SNGD) optimizers (Hazan et al., 2015; Nesterov, 1984) .

SNGD uses only the direction of the stochastic gradient to update the weights, and the step size does not depend on the magnitude of that gradient.

Hazan et al. (2015) proved that the direction of the gradient was sufficient for convergence.

Ignoring the gradient magnitude makes SNGD robust to vanishing and exploding gradients.

SNGD with layer-wise gradient normalization was introduced by Singh et al. (2015) .

The method scales up small gradients, while keeping large gradients unchanged:

Similar to Adam, the weights are updated with the 1 st moment re-scaled by the 2 nd moment:

Adaptive methods like Adam generalize worse than SGD with momentum as was shown in Wilson et al. (2017) .

For example, Keskar & Socher (2017) proposed to use Adam during the initial stage only and then switch to SGD.

Luo et al. (2019) suggested to improve Adam regularization by limiting the factor 1 ??? vt to a certain range: limiting from above helps to decrease the training loss while limiting from below helps to generalize better.

Loshchilov & Hutter (2019) showed that Adam's weak regularization is due to the fact that the 2 nd moment normalization effectively turns off L2-regularization.

They proposed AdamW, which decouples the weight decay d ?? w t from the gradient and uses it directly in the weight update:

Adam needs to store the 2 nd moment, and this doubles the optimizer memory compared to SGD with momentum.

This affects large models like GPT-2 (Radford et al., 2019 ) with 1.5 billion parameters.

Shazeer & Stern (2018) proposed the AdaFactor algorithm, which replaced the full 2 nd moment with moving averages of the row and column sums of the squared gradients.

For a layer defined by an n ?? m matrix, this would reduce memory from O(n ?? m) to O(n + m).

NovoGrad consumes the same amount of memory as SGD with momentum.

NovoGrad is based on 3 ideas: (1) layer-wise 2 nd moments instead of 2 nd moment per each parameter, (2) gradients normalization with layer-wise 2 nd moments, (3) decoupled weight decay.

Let g l t be the stochastic gradient for layer l at step t. NovoGrad first computes the layer-wise 2 nd moment v l t using the norm ||g l t ||:

where 0 ??? ?? 2 ??? 1.

We use much smaller ?? 2 than in Adam, usually in the range [0.2, 0.5].

The moment v l t is used to normalize the gradient g l t before calculating the first moment m l t .

Similarly to AdamW, we decouple weight decay d ?? w t from the stochastic gradient, but we add it to normalized gradient before computing moment m

Algorithm 1 NovoGrad Parameters: Initial learning rate ?? 0 , moments ?? 1 , ?? 2 , weight decay d, number of steps T Weight initialization: t = 0, Initialize w 0 .

Moment initialization: t = 1, for each layer l set v

end for end while where 0 < ?? 1 < 1 is the momentum, typically in the same range as in SGD or Adam [0.9 ??? 0.95].

The first moment can be also computed via an exponential moving average in Adam-like style:

Finally, weights are updated the same way as in SGD with momentum.

Similar to Adam, one can construct a counter-example for NovoGrad in the stochastic convex optimization settings (Wilson et al., 2017) .

However, the "AMS-Grad" fix (Reddi et al., 2018) for Adam can also be applied in this case to guarantee NovoGrad convergence:

Following (Andrew M. Saxe & Ganguli, 2013; Ian J. Goodfellow & Saxe, 2015) we will use NovoGrad to train linear model composed of two linear layers w 1 , w 2 without any non-linearity.

The model y = (w 1 ?? w 2 ) ?? x should output 1 when x = 1.

This model is linear with respect to the inputs, but it is non-linear with respect to the weights, since they are factorized into the product of layers' weights.

Training the model is equivalent to the minimization of the loss L(w 1 , w 2 ) = (w 1 ??w 2 ???1) , 2015) .

The loss is not convex, and its minima are located on the hyperbola w 1 w 2 = 1 (see Figure 1 ).

Minima close to the points (???1, ???1) and (1, 1) are good "flat" minima which generalize well.

Minima close to the axes are "sharp" minima (Keskar et al., 2016) .

We trained the model with SGD with momentum, Adam, AdamW, and NovoGrad, using the same fixed learning rate, 3 weight decay, and weights initialization.

The model was trained for 500 steps.

Figure 2 shows the training trajectory and the zoomed-out area near the final point.

All algorithms behave in a similar way: first the trajectory goes to the curve w 2 = 1/w 1 , and then follows the hyperbola towards (1, 1) or (???1, ???1).

During the first phase, training loss decreases, and during the second phase, generalization improves.

SGD converges nicely toward (1, 1) but its trajectory is still slightly off of the optimal solution.

Adam oscillates wildly around hyperbola w 2 = 1/w 1 , while AdamW behaves much better since weight decay decoupling significantly reduces oscillations.

NovoGrad is the most stable out of four algorithms.

It exhibits better generalization and closely follows the minima curve because normalized gradients prevent trajectory from going far from it.

We also found that NovoGrad is more robust than other algorithms to the choice of learning rate, weight decay, and weight initialization (see for details Appendix A).

4 Each model was trained on a single DGX-1 machine with 8 NVIDIA V100 GPUs with gradient accumulation used for large batch training.

In all the experiments, NovoGrad performed on par or better than other algorithms.

We used ResNet-50 v2 (He et al., 2016) for ImageNet classification task (Russakovsky et al., 2015) .

We trained this model with three optimizers: SGD with momentum (SGD), AdamW, and NovoGrad.

All models have been trained with the batch size of 1024 for 100 epochs.

We used quadratic LR decay for SGD with momentum and cosine decay (Loshchilov & Hutter, 2016) for AdamW and NovoGrad.

We could not find any training recipe for ResNet-50 with AdamW, so we report the best accuracy we achieved after extensive hyper-parameter search.

We used only standard data augmentation methods: re-size, flip, random crop, and did not employ any additional training tricks (He et al., 2018) .

The single-crop validation accuracy for each algorithm is reported in Table 1 .

Hazan et al. (2015) showed that large batch size is beneficial for SNGD convergence, which motivated us to explore NovoGrad for large batch training.

We trained ResNet-50 v2 with batch sizes of 8K and 32K.

To compare with the previous methods, we train the model for 90 epochs using cosine LR decay.

To emulate a large batch, we used a mini-batch of 128 per GPU and accumulated gradients from several mini-batches before each weight update.

To establish the baseline for NovoGrad training with batch 32K we first used the method similar to proposed in Goyal et al. (2017) : scaling the learning rate linearly with the batch size and using LR warmup.

This method gives top-1=75.09% and top-5=92.27%.

We found that we get much better results when we increase both the learning rate ?? and the weight decay d to improve the regularization (see Table 2 ).

For comparison, we took 3 methods, which (1) use fixed batch size during training and (2) do not modify the original model.

All 3 methods employ SGD with momentum.

The first method (Goyal et al. (2017) ) scales LR linearly with batch size and uses the LR warmup to stabilize the initial training phase.

The second method (You et al. (2018) ) combines warmup with Layer-wise Adaptive Rate Scaling (LARS) (You et al., 2017) .

The last method (Codreanu et al. (2017) ) uses warmup and dynamic weight decay (WD).

NovoGrad outperformed all other methods without using any additional techniques like LR warmup (Goyal et al., 2017) , dynamic weight decay, special batch normalization initialization, etc.

Using warm-up (500 steps) we slightly improved top1 accuracy to 75.99% and top5 to 92.72%.

We conducted experiments with Jasper-10x5 (Li et al. (2019) ), a very deep convolutional neural acoustic model, on the LibriSpeech speech recognition task (Panayotov et al., 2015) .

Jasper was trained with SGD with momentum (SGD), Adam and NovoGrad for 400 epochs with a batch of 256, polynomial LR decay, and Layerwise Adaptive Rate Clipping (LARC).

5 We found that NovoGrad yields lower Word Error Rates (WER) comparing to SGD, especially for the long runs.

The model and training parameters are described in Li et al. (2019) .

We trained Jasper10x5 with batch sizes of 512, 4K, 8K, 16K and 32K on LibriSpeech.

In all cases, we trained the model for 400 epochs.

For batch size up to 8K we scaled LR linearly with the batch size and used LR warmup.

To scale batch to 16K and 32K we also increased weight decay (see Table 5 ).

The batch 16K leads to WER comparable to the baseline.

Batch 32K has higher WER due to the smaller number of training steps (9 weights updates per epoch).

Figure 3 shows WER on dev-clean during training for different batch sizes.

We trained Transformer-XL (Dai et al., 2019) , the state-of-the-art LM architecture on the wordlevel WikiText-103 (Merity et al., 2016) benchmark.

For all the experiments we used a 16-layer All other hyperparameters were taken from the original Transformer-XL paper, the source code was based on a publicly available implementation.

6 Each configuration was trained for 12 billion tokens which is approximately 117 epochs and 366K training iterations.

Figure 4 shows that NovoGrad exhibits a much smaller gap between training and validation perplexity compared to Adam, which results in better performance on the test set.

Longer training for 20B tokens does not lead to overfitting as the resulting validation and test perplexities improve even further.

We trained Transformer (Vaswani et al., 2017) on WMT 2014 English-to-German benchmark.

For all the experiments, we used a 12-layer Transformer-big model with 185M parameters (d model = 1024, d ff = 4096, h = 16) with the vocabulary of 8192 tokens based on joint source-target bytepair-encodings (Sennrich et al., 2015) .

For Adam and AdamW we used dropout of P drop = 0.3 and for NovoGrad we used P drop = 0.2.

We trained all algorithms with mixed-precision (Micikevicius et al., 2017) for 100K steps (approximately 150 epochs) with a 4K steps warmup on batches of up to 490K source and target tokens obtained via gradient accummulation (Ott et al., 2018) with cosine learning rate annealing.

We did not use checkpoint averaging, all the results are reported for the last checkpoint in the corresponding run.

w 2 ) ?? x should output 1 when x = 1.

The model is linear function of the inputs, but it is non-linear function of the weights, since they are factorized into the product of layers' weights.

Training the model is equivalent to the minimization of the loss L(w 1 , w 2 ) = (w 1 ?? w 2 ??? 1) 2 .

The loss is not convex, and its minima are located on the curve: w 1 w 2 = 1.

Minima close to the points (???1, ???1) and (1, 1) are good "flat" minima which generalize well.

Minima close to axes are "sharp" minima with bad generalization (see (Keskar et al., 2016) ).

The 2D-contour plot of the loss function shown on Figure 5 .

2 of linear model with two layers.

The loss functions has many global minima located on hyperbola w 2 = 1/w 1 .

Solutions near (???1, ???1) and (1, 1) are good "flat" minima, and solutions near axes are "sharp" minima.

We will study how the behavior of each algorithm depends on learning rate, weight decay and initialization.

We will train the model with each optimizer for 500 steps using the same learning rate, weight decay, and weights initialization.

To use the same learning rate for all optimizers, we will use the "gradient averaging" for NovoGrad.

We will also use the version of SGD with "gradient averaging" (similar to Adam): m t = ?? ?? m t???1 + (1 ??? ??) ?? g t .

For fixed learning rate this SGD version is equivalent to the regular SGD with momentum.

Training trajectories for the baseline (fixed learning rate 0.2, weight decay 0.1, and ?? 1 = 0.95, ?? 2 = 0.5.) are shown on the Figure 6 .

All algorithms first go to the curve w 2 = 1/w 1 , and then slide along hyperbola towards (1, 1) or (???1, ???1).

SGD is slightly off with respect to the optimal solution.

Adam oscillates wildly around line w 2 = 1/w 1 .

AdamW behaves better since weight decay decoupling significantly reduces osculations.

NovoGrad is the most stable out of four algorithms, it also shows much better generalization than other algorithms and converges to (1, 1) closely following the minima curve.

Next, we increased learning rate from 0.2 to 1.0 while keeping weight decay equal to 0.

Similarly, when we increased weight decay from 0.1 to 0.5 while keeping learning rate 0.2, all algorithms except NovoGrad diverge, while NovoGrad demonstrates high robustness to the weight decay choice (see Figure 8 ).

Finally, we started training from different initial point.

SGD and NovoGrad are most robust with respect to the initialization, while AdamW diverge (see Figure 9 ).

To summarize our experiments with linear neural network: NovoGrad is more robust than other algorithms to the choice of learning rate, weight decay, and weight initialization.

@highlight

NovoGrad -  an adaptive SGD method with layer-wise gradient normalization and decoupled weight decay. 