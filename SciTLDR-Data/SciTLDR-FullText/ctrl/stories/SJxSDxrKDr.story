We propose a new method to train neural networks based on a novel combination of adversarial training and provable defenses.

The key idea is to model training as a procedure which includes both, the verifier and the adversary.

In every iteration, the verifier aims to certify the network using convex relaxation while the adversary tries to find inputs inside that convex relaxation which cause verification to fail.

We experimentally show that this training method is promising and achieves the best of both worlds – it produces a model with state-of-the-art accuracy (74.8%) and certified robustness (55.9%) on the challenging CIFAR-10 dataset with a 2/255 L-infinity perturbation.

This is a significant improvement over the currently known best results of 68.3% accuracy and 53.9% certified robustness, achieved using a 5 times larger network than our work.

The discovery of adversarial examples in deep learning (Szegedy et al., 2013) has increased the importance of creating new training methods which produce accurate and robust neural networks with provable guarantees.

Existing work: adversarial and provable defenses Adversarial training (Goodfellow et al., 2015; Kurakin et al., 2017) provides a basis framework which augments the training procedure with adversarial inputs produced by an adversarial attack.

Madry et al. (2018) instantiated adversarial training using a strong iterative adversary and showed that their approach can train models which are highly robust against the strongest known adversarial attacks (Carlini & Wagner, 2017) .

This method has also been used to train robust ImageNet models (Xie et al., 2019) .

While promising, the main drawback of the method is that when instantiated in practice, via an approximation of an otherwise intractable optimization problem, it provides no guarantees -it does not produce a certificate that there are no possible adversarial attacks which could potentially break the model.

To address this lack of guarantees, recent line of work on provable defenses Raghunathan et al., 2018; has proposed to train neural networks which are certifiably robust under a specific attacker threat model.

However, these guarantees come at the cost of a significantly lower standard accuracy than models trained using adversarial training.

This setting raises a natural question -can we leverage ideas from both, adversarial training techniques and provable defense methods, so to obtain models with high accuracy and certified robustness?

This work: combining adversarial and provable defenses In this work, we take a step towards addressing this challenge.

We show that it is possible to train more accurate and provably robust neural networks using the same convex relaxations as those used in existing, state-of-the-art provable defense methods, but with a new, different optimization procedure inspired by adversarial training.

Our optimization works as follows: (i) to certify a property (e.g., robustness) of the network, the verifier produces a convex relaxation of all possible intermediate vector outputs in the neural network, then (ii) an adversary now searches over this (intermediate) convex regions in order to find, what we refer to as a latent adversarial example -a concrete intermediate input contained in the relaxation that when propagated through the network causes a misclassification that prevents verification, and finally (iii) the resulting latent adversarial examples are now incorporated into our training scheme using adversarial training.

Overall, we can see this method as bridging the gap between adversarial training and provable defenses (it can conceptually be instantiated with any convex relaxation).

We experimentally show that the method is promising and results in a neural network with state-of-theart 78.8% accuracy and 58.1% certified robustness on the challenging CIFAR-10 dataset with 2/255 L ∞ perturbation (the best known existing results are 68.3% accuracy and 53.9% certified robustness using 5 times larger network ).

• A new method we refer to as layerwise adversarial training which can train provably robust neural networks and conceptually bridges the gap between adversarial training and existing provable defense methods.

• Instantiation of layerwise adversarial training using linear convex relaxations used in prior work (accomplished by introducing a projection operator).

• Experimental results showing layerwise adversarial training can train neural network models which achieve both, state-of-the-art accuracy and certified robustness on CIFAR-10 with 2/255 L ∞ perturbation.

Overall, we believe the method presented in this work is a promising step towards training models that enjoy both, higher accuracy and higher certification guarantees.

An interesting item for future work would be to explore instantiations of the method with other convex relaxations than the one considered here.

We now discuss some of the closely related work on robustness of neural networks.

Heuristic adversarial defenses After the first introduction of adversarial examples (Szegedy et al., 2013; Biggio et al., 2013) , defense mechanisms to train robust neural networks were built based on the inclusion of adversarial examples to the training set (Kurakin et al., 2017; Goodfellow et al., 2015) .

Models trained using adversarial training with projected gradient descent (PGD) (Madry et al., 2018) were shown to be robust against the strongest known attacks (Carlini & Wagner, 2017) .

This is in contrast to other defense mechanisms which have been broken by new attack techniques (Athalye et al., 2018) .

While models trained using adversarial training achieve robustness against strong adversaries, there are no guarantees that model is robust against any kind of adversarial attack under the threat model considered.

Provable adversarial defenses Another line of work proposes to learn classifiers which come with robustness guarantees.

These approaches are based on linear or semidefinite (Raghunathan et al., 2018; Dvijotham et al., 2018a) relaxations, hybrid zonotope or interval bound propagation (Gowal et al., 2018) .

While these approaches currently obtain robustness guarantees, accuracy of these networks is relatively small and limits practical use of these methods.

There has also been recent work on certification of general neural networks, not necessarily trained in a special way.

These methods are based on SMT solvers (Katz et al., 2017) , mixed-integer linear programs , abstract interpretation , restricted polyhedra (Singh et al., 2019b) or combinations of those (Singh et al., 2019a) .

Another line of work proposes to replace neural networks with a randomized classifier (Lecuyer et al., 2018; Cohen et al., 2019; Salman et al., 2019a) which comes with probabilistic guarantees on its robustness.

While these approaches scale to larger datasets such as ImageNet (although with probabilistc instead of exact guarantees), their bounds come from the relationship between L 2 robustness and Gaussian distribution.

In this paper, we consider general verification problem (Qin et al., 2019) where input is not necessarily limited to an L p ball, but arbitrary convex set.

In this work we consider a threat model where an adversary is allowed to transform an input x ∈ R d0 into any point from a convex set S 0 (x) ⊆ R d0 .

For example, for a threat model based on L ∞ perturbations, the convex set will be defined as

Figure 1: An iteration of layerwise adversarial training.

Latent adversarial example x 1 is found in the convex region C 1 (x) and propagated through the rest of the layers in a forward pass which is shown with the blue line.

During backward pass, gradients are propagated through the same layers, shown with the red line.

Note that the first convolutional layer does not receive any gradients.

We now describe our layerwise adversarial training approach which yields a provable defense that bridges the gap between standard adversarial training and existing provable defenses.

Motivation: latent adversarial examples Consider an already trained neural network model h θ which we would like to certify using convex relaxations.

A fundamental issue here is that certification methods based on convex relaxations can struggle to prove the target property (e.g., robustness) due to the iterative accumulation of errors introduced by the relaxation.

More precisely, assume the neural network actually satisfies the property from Equation 1 for an input x, meaning that

.

Naturally, this also implies that the neural network behaves correctly in the latent space of its first hidden layer in the region S 1 (x).

Formally, this means that c T h 2:k θ (x 1 )+d < 0, ∀x 1 ∈ S 1 (x).

However, if one would use a certification method which replaces the region S 1 (x) by its convex relaxation C 1 (x), then it is possible that we would fail to certify our desired property.

This is due to the fact that there may exist an input

Of course, we could repeat the above thought experiment and possibly find more violating latent inputs in the set C i (x) \ S i (x) of any hidden layer i.

The existence of points found in the difference between a convex relaxation and the true region is a fundamental reason for the failure of certification methods based on convex approximations.

For convenience, we refer to such points as latent adversarial examples.

Next, we describe a method which trains the neural network in a way that aims to minimize the number of latent adversarial examples.

Layerwise provable optimization via convex relaxations Our key observation is that the two families of defense methods described earlier are in fact different ends of the same spectrum: methods based on adversarial training maximize the cross-entropy loss in the first convex region C 0 (x) while provable defenses maximize the same loss, but in the last convex region C k (x).

Both methods then backpropagate the loss through the network and update the parameters using SGD.

However, as explained previously, certification methods may fail even before the last layer due to the presence of latent adversarial examples in the difference of the regions C i (x) and S i (x).

A natural question then is -can we leverage adversarial training so to eliminate latent adversarial examples from hidden layers and obtain a provable network?

To this end, we propose adversarial training in layerwise fashion.

The initial phase of training is equivalent to adversarial training as used by Madry et al. (2018) .

In this phase in the inner loop we repeatedly find an input in C 0 (x) which maximizes the cross-entropy loss and update the parameters of the neural network so to minimize this loss using SGD.

Note that the outcome of this phase is a model which is highly robust against strong multi-step adversaries.

However, certification of this fact often fails due to the previously mentioned accumulation of errors in the particular convex relaxation being used.

The next step of our training method is visually illustrated in Figure 1 .

Here, we propagate the initial convex region through the first layer of the network and obtain the convex relaxation C 1 (x).

We then solve the optimization problem to find a concrete point x 1 inside of C 1 (x) which produces Algorithm 1: Layerwise adversarial training via convex relaxations Data: k-layer neural network h θ , training set (X , Y), learning rate η, step size α, inner steps n Result:

Update in parallel n times:

Freeze parameters θ l+1 of layer l + 1; 12 end the maximum loss when this point is propagated further through the network (this forward pass is shown with the blue line).

Finally, we backpropagate the final loss (red line) and update the parameters of the network so to minimize the loss.

Critically, we do not backpropagate through the convex relaxation in the first layer as standard provable defenses do Gowal et al., 2018) .

We instead freeze the first layer and stop backpropagation after the update of the second layer.

Because of this, our optimization problem is significantly easier -the neural network only has to learn to behave well on the concrete points that were found in the convex region C l (x).

This can be viewed as an extension of the robust optimization method that Madry et al. (2018) found to work well in practice.

We then proceed with the above process for later layers.

Formally, this training process amounts to (approximately) solving the following min-max optimization problem at the l-th step:

Note that for l = 0 this formulation is equivalent to the standard min-max formulation in Equation 2 because C 0 (x) = S 0 (x).

Our approach to solve this min-max optimization problem for every layer l is shown in Algorithm 1.

We initialize every batch by random sampling from the corresponding convex region.

Then, in every iteration we use projected gradient descent (PGD) to maximize the inner loss in 3.

We first update x j in the direction of the gradient of the loss and then project it back to C l (x j ) using the projection operator Π. Note that this approach assumes the existence of an efficient projection method to the particular convex relaxation the method is instantiated with.

In the next section, we show how to instantiate the training algorithm described above to a particular convex relaxation which is generally tighter than a hyperrectangle and where we derive an efficient projection operation.

So far we described the general approach of layerwise adversarial training.

Now we show how to instantiate it for a particular convex relaxation based on linear approximations.

If instead one would use interval approximation Gowal et al., 2018 ) as the convex relaxation, then all regions C l (x) will be hyperrectangles and projection to these sets is fast and simple.

However, the interval relaxation provides a coarse approximation which motivates the need to train with relaxations that provide tighter bounds.

Thus, we consider linear relaxations which are generally tighter than those based on intervals.

In particular we leverage the same relaxation which was previously proposed in ; Weng et al. (2018) ; Singh et al. (2018) as an effective way to certify neural networks.

Here, each convex region is represented as a set

Vector a l represents the center of the set and the matrix A l represents the affine transformation of the hypercube [−1, 1] m l .

The initial convex region C 0 (x) is represented using a 0 = x and A 0 = I d0 is a diagonal matrix.

Propagation of these convex regions through the network is out of the scope of this paper -a full description can be found in or Singh et al. (2018) .

At a high level, the convolutional and fully connected layers are handled by multiplying A l and a l by appropriate matrices.

To handle the ReLU activation, for ReLU units which cross 0, we apply a convex relaxation which amounts to multiplying A l and a l by appropriately chosen scalar values, depending whether the ReLU is activated or not.

Using this relaxation of ReLU, we can recursively obtain all convex regions C l (x).

In practice, A l e can be computed without explicitly constructing matrix A l because A l e = W l Λ l−1 W l−2 · · · M 0 e.

Then we can perform matrix-vector multiplication right to left to obtain vector A l e.

We provide more detailed description of this propagation in Appendix A.

Projection to linear convex regions To use our training method we now need to instantiate Algorithm 1 with a suitable projection operator Π C l (x) .

The key insight here is that the vector x ∈ C l (x) is uniquely determined by auxiliary vector e ∈ [−1, 1] m l where x = a l + A l e.

Then instead of directly solving for x which requires projecting to C l (x), we can solve for e instead which would uniquely determine x. Crucially, the domain of e is a hyperrectangle [−1, 1] m l which is easy to project to.

To visualize this further we provide an example in Figure 2 .

The goal is to project the red point x in the right picture to the convex region C l (x).

To project, we first perform change of variables to substitute x with e and then project e to the square [−1, 1] × [−1, 1] to obtain the blue point Π(e) on the left.

Then, we again perform change of variables to obtain the blue point Π(x ) on the right, the projection of x we were looking for.

Based on these observations, we modify Line 7 of Algorithm 1 to first update the coefficients e j using the following update rule: e j ← clip(e j + αA

Here clip is function which thresholds its argument between -1 and 1, formally clip(x, −1, 1) = min(max(x, −1), 1).

This is followed by an update to x j via x j ← a l + A l e j , completing the update step.

Sparse representation While our representation of convex regions with matrix A l and vector a l has clean mathematical properties, in practice, a possible issue is that the matrix A l can grow to be quite large.

Because of this, propagating it through the network can be memory intensive and prohibit the use of larger batches.

To overcome this difficulty, we first observe that A l is quite sparse.

We start with a very sparse, diagonal matrix A 0 at the input.

After each convolution, an element of matrix A l+1 is non-zero only if there is a non-zero element inside of its convolutional kernel in matrix A l .

We can leverage this observation to precompute positions of all non-zero elements in matrix A l+1 and compute their values using matrix multiplication.

This optimization is critical to enabling training to take place altogether.

An interesting item for future work is further optimizing the current relaxation (via a specialized GPU implementation) or developing more memory friendly relaxations, so to scale the training to larger networks.

After training a neural network via layerwise adversarial training, our goal is to certify the target property (e.g., robustness).

Here we leverage certification techniques which are not fast enough to be incorporated into the training procedure, but which can significantly boost the certification performance.

The linear relaxation of ReLU that we are using is parameterized by slopes λ of the linear relaxation.

Prior work which employed this relaxation Weng et al., 2018; Singh et al., 2018) chose these slopes in a greedy manner by minimizing the area of the relaxation.

During training we also choose λ in the same way.

However, during certification, we can also optimize for the values of λ that give rise to the convex region inside of which the maximum loss is minimized.

This optimization problem can be written as:

Solving this is computationally too expensive inside the training loop, but during certification it is feasible to approximate the solution.

We solve for λ using the Adam optimizer and clipping the elements between 0 and 1 after each update.

We remark that the idea of learning the slope is similar to Dvijotham et al. (2018b) who propose to optimize dual variables in a dual formulation, however here we stay in the primal formulation.

Combining convex relaxations with exact bound propagation During layerwise adversarial training we essentially train the network to be certified on all regions C 0 (x), ..., C k (x).

While computing exact regions S l (x) ⊆ C l (x) is not feasible during training, we can afford it to some extent during certification.

The idea is to first propagate the bounds using convex relaxations until one of the hidden layers l and obtain a region C l (x).

If training was successful, there should not exist a concrete point x l ∈ C l (x) which, if propagated through the network, violates the correctness property in Equation 1.

We can encode both, the property and the propagation of the exact bounds S l (x) using a Mixed-Integer Linear Programming (MILP) solver.

Note that we can achieve this because we represent the region C l (x) using a set of linear constraints, however, for general convex shapes this may not be possible.

We perform the MILP encoding using the formulation from .

It is usually possible to encode only the last two layers using MILP due to the poor scalability of these solvers for realistic network sizes.

One further improvement we also include is to tighten the convex regions C l (x) using refinement via linear programming as described in Singh et al. (2019a) .

We remark that this combination of convex relaxation and exact bound propagation does not fall under the recently introduced convex barrier to certification Salman et al. (2019b) .

We now present an evaluation of our training method on the challenging CIFAR-10 dataset.

Experimental setup We evaluate on a desktop PC with 2 GeForce RTX 2080 Ti GPU-s and 16-core Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz.

We use Gurobi as a MILP solver.

Our method is implemented in PyTorch and we plan to release both, the code and the trained models.

Neural network architecture All presented results are on a 4-layer convolutional network with 49 402 neurons: first 3 layers are convolutional layers with filter sizes 32, 32 and 128, kernel sizes 3, 4, 4 and strides 1, 2, 2 respectively.

These are followed by a fully connected layer with 250 hidden units.

After each of these layers, there is a ReLU activation.

Training We use batch size 50 and L1 regularization 0.00001 for training.

We perform optimization using Adam (Kingma & Ba, 2014) with initial learning rate 0.001 which is decreased by 10× every 100 epochs.

During layerwise training we start with perturbation which is 10% higher than the one we certify and we decrease it by 5% when the training progresses to the next layer.

Certification After training completes, we perform certification as follows: for every image, we first try to certify it using only linear relaxations (with the improvement of learned slopes, Section 6).

If this fails, we encode the last layer as MILP and try again.

Finally, if this fails we encode the ReLU activation after the last convolution using additional 300 binary variables and the rest using the triangle formulation Ehlers (2017) .

We consider an image to be not certifiable if we fail to verify it using these methods.

We always evaluate on the first 1 000 images from the test set.

Comparison to prior work We first train and certify using our method for the L ∞ perturbation 2/255.

Results are shown in Table 1 .

We always compare to the best reported and reproducible et al. (2018) 68.3 53.9 Gowal et al. (2018) 70.1 50.0 Zhang et al. (2019) 59.9 46.1 62.3 45.5 61.1 45.9 , as this improvement is also orthogonal to the method here.

Thus, we only consider their best single network architecture (inline with prior work Zhang et al. (2019) which compares to a single architecture).

We believe all methods listed in Table 1 , including ours, would benefit from additional techniques such as cascades, pre-training and leveraging unlabeled data.

Experimentally, we find that the neural network trained using our method substantially outperforms all existing approaches, both in terms of standard accuracy and certified robustness for 2/255.

Note that here we are using the same linear relaxation as , but our optimization procedure is different and shows significant improvements over the one used in their work.

We also run the same experiment for L ∞ perturbation 8/255.

Here we do not include comparison with Gowal et al. (2018) as their results were found to be not reproducible (Zhang et al., 2019; Mirman et al., 2019; Xu, 2019) .

These results are presented in Table 2 .

Here we substantially outperform all existing approaches in terms of standard accuracy.

However, in terms of certified robustness we are not able to achieve similar results to Zhang et al. (2019) whose method is based on a combination of interval approximation and linear relaxation.

The main issue is that our 4-layer network lacks capacity to solve this task -even if training only using standard adversarial training our empirical robustness does not go above ∼ 34%.

We remark that capacity was found to be one of the key components necessary to obtain a robust classifier (Madry et al., 2018) .

Due to promising results for 2/255, we believe achieving state-of-the-art results for 8/255 is very likely an issue of instantiating our method with a convex relaxation that is more memory efficient, which we believe is an interesting item for future work.

We presented a new method to train certified neural networks.

The key concept was to combine techniques from provable defenses using convex relaxations with those of adversarial training.

Our method achieves state-of-the-art 78.8% accuracy and 58.1% certified robustness on CIFAR-10 with a 2/255 L ∞ perturbation, significantly outperforming prior work when considering a single network (it also achieves competitive results on 8/255 L ∞ ).

The method is general and can be instantiated with any convex relaxation.

A promising future work item is scaling to larger networks: this will require tight convex relaxations with a low memory footprint that allow for efficient projection.

Here we provide additional details that were omitted in the main body of the paper.

In this section, we describe how to propagate convex relaxations of the form C l (x) = {a l + A l e | e ∈ [−1, 1] m l } through the network, for a single input x. As explained before, these relaxations were previously proposed in Weng et al. (2018) ; Singh et al. (2018) .

For the sake of completeness we describe them here using our notation.

Depending on the form of function h i θ representing operation applied at layer i we distinguish different cases.

Here we assume we have obtained region C i−1 (x) and our goal is to compute the region C i (x) using convex relaxation g i θ of the function h i θ .

Initial convex region Let be L ∞ radius that we are certifying.

We then compute minimum and maximum pixel values for each pixel as x l = max(0, x − ) and x u = min(1, x + ).

We define initial convex region as:

Convolutional and fully connected layers For both convolutional and fully connected layers, the update is given by

mi .

We can then compute:

Using this formula, we define convex region C i+1 (x) = {a i+1 + A i+1 e | e ∈ [−1, 1] mi+1 } where:

We will explain the transformation of a single element x i,j = a i,j + A T i,j e. We first compute lower bound l i,j and upper bound u i,j of element x i in the set C i (x):

In the other case where 0 is between l i,j and u i,j we define ReLU (x i,j ) = λ i,j x i,j + µe mi+j where e mi+j ∈ [−1, 1] is a coefficient for a new error term.

Formulas for λ i,j and µ i,j are the following:

This computation can also be written in the matrix form as ReLU (x i ) = Λ i+1 x i + M i+1 e new where Λ i+1 and M i+1 are diagonal matrices with elements computed as above.

Finally, new convex region C i+1 (x) = {a i+1 + A i+1 e | e ∈ [−1, 1] mi+1 } is defined as:

where [] denotes concatenation of matrices.

Here we describe how we apply random projection approach from to estimate the bounds during training.

While operate in dual framework, their method to statistically estimate the bounds during training can also be applied in primal framework which we use in this paper.

Recall that lower and upper bound for each neuron are computed as

Thus, we need to compute ||A i || 1 , which is L 1 norm of each row in the matrix A i .

Using the method from , based on the result from Li et al. (2007) , we estimate ||A i || 1 with the method of random projections.

Here A i is a matrix with d i rows and m i columns, where d i is dimensionality of the output vector in the i-th layer and m i is number of unit terms in the definition of region C i (x).

The method of random projections samples standard Cauchy random matrix R of dimensions m i × k (k is number of projections) and then estimates ||A i || 1 ≈ median(|A i R|).

To avoid computing the entire matrix A i we substitute:

In the formula above, we set M 0 = A 0 .

Now, Dvijotham et al. (2018a) 83.4 62.4

To calculate A i R we split R = [R 0 , R 2 , ..., R i−1 ] and compute:

Crucially, each summand can now be efficiently computed due to the associativity of matrix multiplication by performing the multiplication backwards.

In this section we present additional results on SVHN and MNIST datasets.

SVHN We evaluated on SVHN dataset.

For this experiment, we used convolutional network with 2 convolutional layers of kernel size 4 and stride 1 with 32 and 128 filters respectively.

These convolutional layers are followed by a fully connected layer with 200 neurons.

Each of the layers is followed by a ReLU activation function.

For our training, we started with perturbation 10% higher than the one we are certifying and decreased it by 5% when progressing to the next layer.

We trained each layer for 50 epochs and used L 1 regularization factor 0.0001.

Results of this experiment are shown in Table 3 .

We certified first 1 000 images in SVHN test dataset.

Our network has both, higher accuracy and higher certified robustness, than networks trained using other techniques for provable defense.

MNIST In the next experiment, we evaluted on MNIST dataset.

For this experiment, we used convolutional network with 2 convolutional layers with kernel sizes 5 and 4, and strides 2 followed by 1 fully connected layer.

Each of the layers is followed by a ReLU activation function.

For our training, we started with perturbation 10% higher than the one we are certifying and decreased it by 5% when progressing to the next layer.

We trained each layer for 50 epochs and used L 1 regularization factor 0.00001.

We certified first 1 000 images in MNIST test dataset.

For perturbation 0.1, convolutional layers have filter sizes 32 and 64, and fully connected layer has 150 neurons.

Results are presented in Table 4 .

Here, our numbers are comparable to those of state-of-the-art approaches.

For perturbation 0.3, convolutional layers have filter sizes 32 and 128, and fully connected layer has 400 neurons.

Results are presented in Table 4 .

Here, our certified robustness is somewhat lower than state-of-the-art.

We believe this is due to the imprecise estimates of lower and upper bund via random projections.

This is also reflected in relatively poor performance of who also rely on the same statistical estimates.

Thus, for MNIST dataset and perturbation 0.3 it is likely necessary to use exact propagation instead of the estimates.

However, this also induces large cost to the runtime.

During training, we use two regularization mechanisms to make our convex relaxation tighter, both previosuly proposed in .

First, we use L 1 -norm regularization which is known to induce sparsity in the weights of the network.

has shown that weight sparsity helps induce more stable ReLU units which in turn makes our convex relaxation tighter (as for stable ReLU units it is already precise).

Gowal et al. (2018) 98.9 97.7 Zhang et al. (2019) 99.0 94.

4 Wong et al. (2018) 98.9 96.3 Dvijotham et al. (2018a) 98.8 95.

6 Mirman et al. (2018) 98.7 95.8 99.0 95.6 Table 5 : Evaluation on MNIST dataset with L ∞ perturbation 0.3 Method Accuracy (%) Certified robustness (%)

Our work 97.6 84.

6 Gowal et al. (2018) 98.3 91.9 Zhang et al. (2019) 98.5 91.5 85.1 56.9 96.6 89.3 97.3 80.7

Second, in the i-th phase of training, we explicitly introduce a loss based on the volume of convex region C i+1 .

To make the relaxation tighter and minimize the volume, for each neuron j in layer i + 1 we add a loss of the form max(0, −l i+1,j ) max(0, u i+1,j ).

This loss corresponds to the area under ReLU relaxation, see e.g. Singh et al. (2018) for a derivation.

<|TLDR|>

@highlight

We propose a novel combination of adversarial training and provable defenses which produces a model with state-of-the-art accuracy and certified robustness on CIFAR-10. 