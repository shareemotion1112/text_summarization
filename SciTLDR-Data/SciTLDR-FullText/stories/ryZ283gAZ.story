Deep neural networks have become the state-of-the-art models in numerous machine learning tasks.

However, general guidance to network architecture design is still missing.

In our work, we bridge deep neural network design with numerical differential equations.

We show that many effective networks, such as ResNet, PolyNet, FractalNet and RevNet, can be interpreted as different numerical discretizations of differential equations.

This finding brings us a brand new perspective on the design of effective deep architectures.

We can take advantage of the rich knowledge in numerical analysis to guide us in designing new and potentially more effective deep networks.

As an example, we propose a linear multi-step architecture (LM-architecture) which is inspired by the linear multi-step method solving ordinary differential equations.

The LM-architecture is an effective structure that can be used on any ResNet-like networks.

In particular, we demonstrate that LM-ResNet and LM-ResNeXt (i.e. the networks obtained by applying the LM-architecture on ResNet and ResNeXt respectively) can achieve noticeably higher accuracy than ResNet and ResNeXt on both CIFAR and ImageNet with comparable numbers of trainable parameters.

In particular, on both CIFAR and ImageNet, LM-ResNet/LM-ResNeXt can significantly compress (>50%) the original networks while maintaining a similar performance.

This can be explained mathematically using the concept of modified equation from numerical analysis.

Last but not least, we also establish a connection between stochastic control and noise injection in the training process which helps to improve generalization of the networks.

Furthermore, by relating stochastic training strategy with stochastic dynamic system, we can easily apply stochastic training to the networks with the LM-architecture.

As an example, we introduced stochastic depth to LM-ResNet and achieve significant improvement over the original LM-ResNet on CIFAR10.

Deep learning has achieved great success in may machine learning tasks.

The end-to-end deep architectures have the ability to effectively extract features relevant to the given labels and achieve state-of-the-art accuracy in various applications BID3 ).

Network design is one of the central task in deep learning.

Its main objective is to grant the networks with strong generalization power using as few parameters as possible.

The first ultra deep convolutional network is the ResNet BID16 which has skip connections to keep feature maps in different layers in the same scale and to avoid gradient vanishing.

Structures other than the skip connections of the ResNet were also introduced to avoid gradient vanishing, such as the dense connections BID20 , fractal path BID27 and Dirac initialization BID50 .

Furthermore, there has been a lot of attempts to improve the accuracy of image classifications by modifying the residual blocks of the ResNet.

BID49 suggested that we need to double the number of layers of ResNet to achieve a fraction of a percent improvement of accuracy.

They proposed a widened architecture that can efficiently improve the accuracy.

BID51 pointed out that simply modifying depth or width of ResNet might not be the best way of architecture design.

Exploring structural diversity, which is an alternative dimension in network design, may lead to more effective networks.

In BID43 , BID51 , BID47 , and BID19 , the authors further improved the accuracy of the networks by carefully designing residual blocks via increasing the width of each block, changing the topology of the network and following certain empirical observations.

In the literature, the network design is mainly empirical.

It remains a mystery whether there is a general principle to guide the design of effective and compact deep networks.

Observe that each residual block of ResNet can be written as u n+1 = u n + ???tf (u n ) which is one step of forward Euler discretization (AppendixA.1) of the ordinary differential equation (ODE) u t = f (u) (E, 2017) .

This suggests that there might be a connection between discrete dynamic systems and deep networks with skip connections.

In this work, we will show that many state-of-the-art deep network architectures, such as PolyNet BID51 , FractalNet BID27 and RevNet BID12 , can be consider as different discretizations of ODEs.

From the perspective of this work, the success of these networks is mainly due to their ability to efficiently approximate dynamic systems.

On a side note, differential equations is one of the most powerful tools used in low-level computer vision such as image denoising, deblurring, registration and segmentation BID36 BID2 BID4 .

This may also bring insights on the success of deep neural networks in low-level computer vision.

Furthermore, the connection between architectures of deep neural networks and numerical approximations of ODEs enables us to design new and more effective deep architectures by selecting certain discrete approximations of ODEs.

As an example, we design a new network structure called linear multi-step architecture (LM-architecture) which is inspired by the linear multi-step method in numerical ODEs BID1 .

This architecture can be applied to any ResNet-like networks.

In this paper, we apply the LM-architecture to ResNet and ResNeXt BID47 ) and achieve noticeable improvements on CIFAR and ImageNet with comparable numbers of trainable parameters.

We also explain the performance gain using the concept of modified equations from numerical analysis.

It is known in the literature that introducing randomness by injecting noise to the forward process can improve generalization of deep residual networks.

This includes stochastic drop out of residual blocks BID21 and stochastic shakes of the outputs from different branches of each residual block BID11 .

In this work we show that any ResNet-like network with noise injection can be interpreted as a discretization of a stochastic dynamic system.

This gives a relatively unified explanation to the stochastic learning process using stochastic control.

Furthermore, by relating stochastic training strategy with stochastic dynamic system, we can easily apply stochastic training to the networks with the proposed LM-architecture.

As an example, we introduce stochastic depth to LM-ResNet and achieve significant improvement over the original LM-ResNet on CIFAR10.

The link between ResNet FIG0 ) and ODEs were first observed by E (2017), where the authors formulated the ODE u t = f (u) as the continuum limit of the ResNet u n+1 = u n + ???tf (u n ).

BID31 bridged ResNet with recurrent neural network (RNN), where the latter is known as an approximation of dynamic systems.

BID40 and BID30 also regarded ResNet as dynamic systems that are the characteristic lines of a transport equation on the distribution of the data set.

Similar observations were also made by BID5 ; they designed a reversible architecture to grant stability to the dynamic system.

On the other hand, many deep network designs were inspired by optimization algorithms, such as the network LISTA BID14 and the ADMM-Net BID48 .

Optimization algorithms can be regarded as discretizations of various types of ODEs BID18 , among which the simplest example is gradient flow.

Another set of important examples of dynamic systems is partial differential equations (PDEs), which have been widely used in low-level computer vision tasks such as image restoration.

There were some recent attempts on combining deep learning with PDEs for various computer vision tasks, i.e. to balance handcraft modeling and data-driven modeling.

BID32 and BID33 proposed to use linear combinations of a series of handcrafted PDE-terms and used optimal control methods to learn the coefficients.

Later, BID10 extended their model to handle classification tasks and proposed an learned PDE model (L-PDE).

However, for classification tasks, the dynamics (i.e. the trajectories generated by passing data to the network) should be interpreted as the characteristic lines of a PDE on the distribution of the data set.

This means that using spatial differential operators in the network is not essential for classification tasks.

Furthermore, the discretizations of differential operators in the L-PDE are not trainable, which significantly reduces the network's expressive power and stability.

BID28 proposed a feed-forward network in order to learn the optimal nonlinear anisotropic diffusion for image denoising.

Unlike the previous work, their network used trainable convolution kernels instead of fixed discretizations of differential operators, and used radio basis functions to approximate the nonlinear diffusivity function.

More recently, BID34 designed a network, called PDE-Net, to learn more general evolution PDEs from sequential data.

The learned PDE-Net can accurately predict the dynamical behavior of data and has the potential to reveal the underlying PDE model that drives the observed data.

In our work, we focus on a different perspective.

First of all, we do not require the ODE u t = f (u) associate to any optimization problem, nor do we assume any differential structures in f (u).

The optimal f (u) is learned for a given supervised learning task.

Secondly, we draw a relatively comprehensive connection between the architectures of popular deep networks and discretization schemes of ODEs.

More importantly, we demonstrate that the connection between deep networks and numerical ODEs enables us to design new and more effective deep networks.

As an example, we introduce the LM-architecture to ResNet and ResNeXt which improves the accuracy of the original networks.

We also note that, our viewpoint enables us to easily explain why ResNet can achieve good accuracy by dropping out some residual blocks after training, whereas dropping off sub-sampling layers often leads to an accuracy drop BID44 .

This is simply because each residual block is one step of the discretized ODE, and hence, dropping out some residual blocks only amounts to modifying the step size of the discrete dynamic system, while the sub-sampling layer is not a part of the ODE model.

Our explanation is similar to the unrolled iterative estimation proposed by BID13 , while the difference is that we believe it is the data-driven ODE that does the unrolled iterative estimation.

In this section we show that many existing deep neural networks can be consider as different numerical schemes approximating ODEs of the form u t = f (u).

Based on such observation, we introduce a new structure, called the linear multi-step architecture (LM-architecture) which is inspired by the well-known linear multi-step method in numerical ODEs.

The LM-architecture can be applied to any ResNet-like networks.

As an example, we apply it to ResNet and ResNeXt and demonstrate the performance gain of such modification on CIFAR and ImageNet data sets.

PolyNet FIG0 ), proposed by BID51 , introduced a PolyInception module in each residual block to enhance the expressive power of the network.

The PolyInception model includes polynomial compositions that can be described as DISPLAYFORM0 We observe that PolyInception model can be interpreted as an approximation to one step of the backward Euler (implicit) scheme (AppendixA.1): DISPLAYFORM1 Indeed, we can formally rewrite (I ??? ???tf ) ???1 as DISPLAYFORM2 Therefore, the architecture of PolyNet can be viewed as an approximation to the backward Euler scheme solving the ODE u t = f (u).

Note that, the implicit scheme allows a larger step size BID1 , which in turn allows fewer numbers of residual blocks of the network.

This explains why PolyNet is able to reduce depth by increasing width of each residual block to achieve state-of-the-art classification accuracy.

FractalNet BID27 FIG0 ) is designed based on self-similarity.

It is designed by repeatedly applying a simple expansion rule to generate deep networks whose structural layouts are truncated fractals.

We observe that, the macro-structure of FractalNet can be interpreted as the well-known Runge-Kutta scheme in numerical analysis.

Recall that the recursive fractal structure of FractalNet can be written as DISPLAYFORM3 For simplicity of presentation, we only demonstrate the FractalNet of order 2 (i.e. c ??? 2).

Then, every block of the FractalNet (of order 2) can be expressed as DISPLAYFORM4 , which resembles the Runge-Kutta scheme of order 2 solving the ODE u t = f (u) (see AppendixA.2).RevNet FIG0 ), proposed by BID12 , is a reversible network which does not require to store activations during forward propagations.

The RevNet can be expressed as the following discrete dynamic system DISPLAYFORM5 RevNet can be interpreted as a simple forward Euler approximation to the following dynamic syste??? DISPLAYFORM6 Note that reversibility, which means we can simulate the dynamic from the end time to the initial time, is also an important notation in dynamic systems.

There were also attempts to design reversible scheme in dynamic system such as BID35 .

DISPLAYFORM7

We have shown that architectures of some successful deep neural networks can be interpreted as different discrete approximations of dynamic systems.

In this section, we proposed a new struc-ture, called linear multi-step structure (LM-architecture), based on the well-known linear multi-step scheme in numerical ODEs (which is briefly recalled in Appendix A.3).

The LM-architecture can be written as follows DISPLAYFORM0 where k n ??? R is a trainable parameter for each layer n. A schematic of the LM-architecture is presented in Figure 2 .

Note that the midpoint and leapfrog network structures in BID5 are all special case of ours.

The LM-architecture is a 2-step method approximating the ODE u t = f (u).

Therefore, it can be applied to any ResNet-like networks, including those mentioned in the previous section.

As an example, we apply the LM-architecture to ResNet and ResNeXt.

We call these new networks the LM-ResNet and LM-ResNeXt.

We trained LM-ResNet and LM-ResNeXt on CIFAR BID25 ) and Imagenet BID37 , and both achieve improvements over the original ResNet and ResNeXt.

Implementation Details.

For the data sets CIFAR10 and CIFAR100, we train and test our networks on the training and testing set as originally given by the data set.

For ImageNet, our models are trained on the training set with 1.28 million images and evaluated on the validation set with 50k images.

On CIFAR, we follow the simple data augmentation in BID28 for training: 4 pixels are padded on each side, and a 32??32 crop is randomly sampled from the padded image or its horizontal flip.

For testing, we only evaluate the single view of the original 32??32 image.

Note that the data augmentation used by ResNet BID16 BID47 is the same as BID28 .

On ImageNet, we follow the practice in BID26 ; BID39 .

Images are resized with its shorter side randomly sampled in [256, 480] for scale augmentation BID39 .

The input image is 224??224 randomly cropped from a resized image using the scale and aspect ratio augmentation of BID42 .

For the experiments of ResNet/LM-ResNet on CIFAR, we adopt the original design of the residual block in BID17 , i.e. using a small two-layer neural network as the residual block with bn-relu-conv-bn-reluconv.

The residual block of LM-ResNeXt (as well as LM-ResNet164) is the bottleneck structure used by BID47 ) that takes the form 1 ?? 1, 64 3 ?? 3, 64 1 ?? 1, 256.

We start our networks with a single 3 ?? 3 conv layer, followed by 3 residual blocks, global average pooling and a fully-connected classifier.

The parameters k n of the LM-architecture are initialized by random sampling from U[???0.1, 0].

We initialize other parameters following the method introduced by BID15 .

On CIFAR, we use SGD with a mini-batch size of 128, and 256 on ImageNet.

During training, we apply a weight decay of 0.0001 for LM-ResNet and 0.0005 for LM-ResNeXt, and momentum of 0.9 on CIFAR.

We apply a weight decay of 0.0001 and momentum of 0.9 for both LM-ResNet and LM-ResNeXt on ImageNet.

For LM-ResNet on CIFAR10 (CIFAR100), we start with the learning rate of 0.1, divide it by 10 at 80 (150) and 120 (225) epochs and terminate training at 160 (300) epochs.

For LM-ResNeXt on CIFAR, we start with the learning rate of 0.1 and divide it by 10 at 150 and 225 epochs, and terminate training at 300 epochs.

Figure 2: LM-architecture is an efficient structure that enables ResNet to achieve same level of accuracy with only half of the parameters on CIFAR10.

Results.

Testing errors of our proposed LM-ResNet/LM-ResNeXt and some other deep networks on CIFAR are presented in TAB1 .

Figure 2 shows the overall improvements of LM-ResNet over ResNet on CIFAR10 with varied number of layers.

We also observe noticeable improvements of both LM-ResNet and LM-ResNeXt on CIFAR100.

BID47 claimed that ResNeXt can achieve lower testing error without pre-activation (pre-act).

However, our results show that LMResNeXt with pre-act achieves lower testing errors even than the original ResNeXt without pre-act.

Training and testing curves of LM-ResNeXt are plotted in Figure3.

In TAB1 , we also present testing errors of FractalNet and DenseNet BID20 on CIFAR 100.

We can see that our proposed LM-ResNeXt29 has the best result.

Comparisons between LM-ResNet and ResNet on ImageNet are presented in TAB2 .

The LM-ResNet shows improvement over ResNet with comparable number of trainable parameters.

Note that the results of ResNet on ImageNet are obtained from "https://github.com/KaimingHe/deep-residual-networks".

It is worth noticing that the testing error of the 56-layer LM-ResNet is comparable to that of the 110-layer ResNet on CIFAR10; the testing error of the 164-layer LM-ResNet is comparable to that of the 1001-layer ResNet on CI-FAR100; the testing error of the 50-layer LM-ResNet is comparable to that of the 101-layer ResNet on ImageNet.

We have similar results on LM-ResNeXt and ResNeXt as well.

These results indicate that the LM-architecture can greatly compress ResNet/ResNeXt without losing much of the performance.

We will justify this mathematically at the end of this section using the concept of modified equations from numerical analysis.

Explanation on the performance boost via modified equations.

Given a numerical scheme approximating a differential equation, its associated modified equation BID45 ) is another differential equation to which the numerical scheme approximates with higher order of accuracy than the original equation.

Modified equations are used to describe numerical behaviors of numerical schemes.

For example, consider the simple 1-dimensional transport equation u t = cu x .

101, pre-act 22.6 6.4Figure 3: Training and testing curves of ResNext29 (16x64d, pre-act) and and LM-ResNet29 (16x64d, pre-act) on CIFAR100, which shows that the LM-ResNeXt can achieve higher accuracy than ResNeXt.

Both the Lax-Friedrichs scheme and Lax-Wendroff scheme approximates the transport equation.

However, the associated modified equations of Lax-Friedrichs and Lax-Wendroff are DISPLAYFORM1 respectively, where r = 2???t ???x .

This shows that the Lax-Friedrichs scheme behaves diffusive, while the Lax-Wendroff scheme behaves dispersive.

Consider the forward Euler scheme which is associated to ResNet, DISPLAYFORM2 Thus, the modified equation of forward Euler scheme reads a??? DISPLAYFORM3 Consider the numerical scheme used in the LM-structure DISPLAYFORM4 Then, the modified equation of the numerical scheme associated to the LM-structure DISPLAYFORM5 Comparing FORMULA11 with (3), we can see that when k n ??? 0, the second order term?? of (3) is bigger than that of (2).

The term?? represents acceleration which leads to acceleration of the convergence of u n when f = ??????g BID41 ; BID46 .

When f (u) = L(u) with L being an elliptic operator, the term?? introduce dispersion on top of the dissipation, which speeds up the flow of u n .

In fact, this is our original motivation of introducing the LM-architecture (1).

Note that when the dynamic is truly a gradient flow, i.e. f = ??????g, the difference equation of the LM-structure has a stability condition ???1 ??? k n ??? 1.

In our experiments, we do observe that most of the coefficients are lying in (???1, 1) FIG1 .

Moreover, the network is indeed accelerating at the end of the dynamic, for the learned parameters {k n } are negative and close to ???1 FIG1 ).

Although the original ResNet BID16 did not use dropout, several work BID21 BID11 showed that it is also beneficial to inject noise during training.

In this section we show that we can regard such stochastic learning strategy as an approximation to a stochastic dynamic system.

We hope the stochastic dynamic system perspective can shed lights on the discovery of a guiding principle on stochastic learning strategies.

To demonstrate the advantage of bridging stochastic dynamic system with stochastic learning strategy, we introduce stochastic depth during training of LM-ResNet.

Our results indicate that the networks with proposed LM-architecture can also greatly benefit from stochastic learning strategies.

As an example, we show that the two stochastic learning methods introduced in BID21 and BID11 can be considered as weak approximations of stochastic dynamic systems.

Shake-Shake Regularization.

Gastaldi (2017) introduced a stochastic affine combination of multiple branches in a residual block, which can be expressed as DISPLAYFORM0 where ?? ??? U(0, 1).

To find its corresponding stochastic dynamic system, we incorporate the time step size ???t and consider DISPLAYFORM1 which reduces to the shake-shake regularization when ???t = 1.

The above equation can be rewritten as DISPLAYFORM2 Since the random variable DISPLAYFORM3 2 ), following the discussion in Appendix B, the network of the shake-shake regularization is a weak approximation of the stochastic dynamic system DISPLAYFORM4 where dB t is an N dimensional Brownian motion, 1 N ??1 is an N -dimensional vector whose elements are all 1s, N is the dimension of X and f i (X), and denotes the pointwise product of vectors.

Note from (4) that we have alternatives to the original shake-shake regularization if we choose ???t = 1.Stochastic Depth.

BID21 randomly drops out residual blocks during training in order to reduce training time and improve robustness of the learned network.

We can write the forward propagation as DISPLAYFORM5 where P(?? n = 1) = p n , P(?? n = 0) = 1 ??? p n .

By incorporating ???t, we consider DISPLAYFORM6 which reduces to the original stochastic drop out training when ???t = 1.

The variance of DISPLAYFORM7 is 1.

If we further assume that (1 ??? 2p n ) = O( ??? ???t), the condition(5) of Appendix B.2 is satisfied for small ???t.

Then, following the discussion in Appendix B, the network with stochastic drop out can be seen as a weak approximation to the stochastic dynamic system DISPLAYFORM8 Note that the assumption (1 ??? 2p n ) = O( ??? ???t) also suggests that we should set p n closer to 1/2 for deeper blocks of the network, which coincides with the observation made by BID21 Figure 8 ).In general, we can interpret stochastic training procedures as approximations of the following stochastic control problem with running cost DISPLAYFORM9 where L(??) is the loss function, T is the terminal time of the stochastic process, and R is a regularization term.

BID17 110,pre-act Orignial 6.37ResNet BID21 In this section, we extend the stochastic depth training strategy to networks with the proposed LMarchitecture.

In order to apply the theory of It?? process, we consider the 2nd order??? + g(t)??? = f (X) (which is related to the modified equation of the LM-structure (3)) and rewrite it as a 1st order ODE system??? = Y,??? = f (X) ??? g(t)Y. Following a similar argument as in the previous section, we obtain the following stochastic proces??? DISPLAYFORM10 which can be weakly approximated by DISPLAYFORM11 where P(?? n = 1) = p n , P(?? n = 0) = 1 ??? p n .

Taking ???t = 1, we obtain the following stochastic training strategy for LM-architecture DISPLAYFORM12 The above derivation suggests that the stochastic learning for networks using LM-architecture can be implemented simply by randomly dropping out the residual block with probability p.

Implementation Details.

We test LM-ResNet with stochastic training strategy on CIFAR10.

In our experiments, all hyper-parameters are selected exactly the same as in BID21 .

The probability of dropping out a residual block at each layer is a linear function of the layer, i.e. we set the probability of dropping the current residual block as DISPLAYFORM13 where l is the current layer of the network, L is the depth of the network and p L is the dropping out probability of the previous layer.

In our experiments, we select p L = 0.8 for LM-ResNet56 and p L = 0.5 for LM-ResNet110.

During training with SGD, the initial learning rate is 0.1, and is divided by a factor of 10 after epoch 250 and 375, and terminated at 500 epochs.

In addition, we use a weight decay of 0.0001 and a momentum of 0.9.Results.

Testing errors are presented in TAB3 .

Training and testing curves of LM-ResNet with stochastic depth are plotted in Figure5.

Note that LM-ResNet110 with stochastic depth training strategy achieved a 4.80% testing error on CIFAR10, which is even lower that the ResNet1202 reported in the original paper.

The benefit of stochastic training has been explained from difference perspectives, such as Bayesian BID23 and information theory BID38 BID0 .

The stochastic Brownian motion involved in the aforementioned stochastic dynamic systems introduces diffusion which leads to information gain and robustness.

In this section we briefly recall some concepts from numerical ODEs that are used in this paper.

The ODE we consider takes the form u t = f (u, t).

Interested readers should consult BID1 for a comprehensive introduction to the subject.

The simplest approximation of u t = f (u, t) is to discretize the time derivative u t by un+1???un ???t and approximate the right hand side by f (u n , t n ).

This leads to the forward (explicit) Euler scheme u n+1 = u n + ???tf (u n , t n ).If we approximate the right hand side of the ODE by f (u n+1 , t n+1 ), we obtain the backward (implicit) Euler scheme u n+1 = u n + ???tf (u n+1 , t n+1 ).The backward Euler scheme has better stability property than the forward Euler, though we need to solve a nonlinear equation at each step.

Runge-Kutta method is a set of higher order one step methods, which can be formulate a?? DISPLAYFORM0 a ij f (?? j , t n + c j ???t), DISPLAYFORM1 b j f (?? j , t n + c j ???t).Here,?? j is an intermediate approximation to the solution at time t n +c j ???t, and the coefficients {c j } can be adjusted to achieve higher order accuracy.

As an example, the popular 2nd-order Runge-Kutta takes the formx n+1 = x n + ???tf (x n , t n ),x n+1 = x n + ???t 2 f (x n , t n ) + ???t 2 f (x n+1 , t n+1 ).

Liear multi-step method generalizes the classical forward Euler scheme to higher orders.

The general form of a k???step linear multi-step method is given by where, ?? j , ?? j are scalar parameters and ?? 0 = 0, |?? j | + |?? j | = 0.

The linear multi-step method is explicit if ?? 0 = 0, which is what we used to design the linear multi-step structure.

In this section we follow the setting of Kesendal FORMULA11 and BID9 .

We first give the definition of Brownian motion.

The Brownian motion B t is a stochastic process satisfies the following assumptions

@highlight

This paper bridges deep network architectures with numerical (stochastic) differential equations. This new perspective enables new designs of more effective deep neural networks.