The brain performs unsupervised learning and (perhaps) simultaneous supervised learning.

This raises the question as to whether a hybrid of supervised and unsupervised methods will produce better learning.

Inspired by the rich space of Hebbian learning rules, we set out to directly learn the unsupervised learning rule on local information that best augments a supervised signal.

We present the Hebbian-augmented training algorithm (HAT) for combining gradient-based learning with an unsupervised rule on pre-synpatic activity, post-synaptic activities, and current weights.

We test HAT's effect on a simple problem (Fashion-MNIST) and find consistently higher performance than supervised learning alone.

This finding provides empirical evidence that unsupervised learning on synaptic activities provides a strong signal that can be used to augment gradient-based methods.

We further find that the meta-learned update rule is a time-varying function; thus, it is difficult to pinpoint an interpretable Hebbian update rule that aids in training.

We do find that the meta-learner eventually degenerates into a non-Hebbian rule that preserves important weights so as not to disturb the learner's convergence.

Backpropagation achieves great performance in neural net optimization, but might not be biologically plausible because most problems are not explicitly phrased as classification with true labels, because neurons only know local signals (e.g. synaptic density, ACh levels, current), and because backpropagation uses the computational graph, a separate data structure with no known biological basis.

Although some supervised training schemes are more biologically plausible (e.g. contrastive Hebbian learning [9] and equilibrium propagation [8] ), it's currently unknown whether the behavior of all neurons is accurately encapsulated by these models.

We speculate that some local, unsupervised learning occurs in the brain and demonstrate that the addition of local, unsupervised rules to standard backpropagation actually improves the speed and robustness of learning.

We begin by defining a local learning rule.

Consider two adjacent neurons i, j with weight w ij : given an impulse traversing i, j with activations v i , v j , a local learning rule computes updates ???w ij using local data v i , w ij , v j .

Note that by this definition, a local learning rule is unsupervised at face value.

Many neuroscientists have hypothesized specific functions that describe the brain's true (unsupervised) local learning rule.

Most such rules involve using the correlation of activations as part of the update rule.

Examples include Hebb's Rule [4] , Oja's Rule [7] , the Generalized Hebbian Algorithm [3] , and nonlineear Hebbian rules [5] .

It is not obvious which of these rules (if any) describe the true behavior of neurons.

We employ meta-learning (learning how to learn) as an investigative tool.

Optimization functions are algorithms too; it stands to reason that we can learn the best optimization function.

In the meta-learning framework, one model A learns a task (e.g. Fashion-MNIST) while another model B learns how to optimize A. Meta-learning has achieved great results in finding robust optimization schemes.

Andrychowicz et.

al. used meta-learning to find the best gradient-based optimization function (B learns to update A using A's gradients) [1] , and Chen et.

al. used meta-learning to find the best gradient-free optimization function (B learns to update A using only the sequence of A's losses).

[2] Finally, Metz et al. demonstrated a fully differentiable architecture for learning to learn unsupervised local rules and demonstrate better-than-random performance on a few-shot basis.

[6] If B consistently converges to some stable rule, we take it as strong evidence that this rule may occur in biological brains as well.

We therefore wish to extend Metz's approach to learning semisupervised local rules not only to improve performance but also to investigate the functional form of the meta-learned update rule.

The Hebbian-Augmented Training algorithm (HAT) is an algorithm that trains the neural net L twice per sample: using local, unsupervised rules on the forward pass and using backpropagationbased gradient descent on the backward pass.

Formally, we create 2 multilayer perceptrons: a learner L(?? | ?? L ) with parameters ?? L and a metalearner M (v i , v j , w ij | ?? M ) with parameters ?? M , which takes inputs v i , w ij , v j and returns ???w ij .

For a single sample ( x, y), we train L without supervision using M and x; we simultaneously train L and M with supervision using A and y.

On the forward pass, we compute activations for each layer.

For a given layer , we now have the inputs, outputs, and current weights -all of the inputs of local learning rule.

We can then apply the outputs of meta-learner M to update the weights of layer .

We then recompute the activations of layer using the new weights.

This process is done efficiently by convolution (for details, see Appendix A).

We compute the activations of the first layer 1 , update 1 , compute the activations of the second layer 2 , update 2 , and so on until we compute the predicted Weights?? y and update |L| .

On the backward pass, we backpropagate.

Since we recomputed the activations of each layer using weights updated by M , the weights of M are upstream of the weights of L in the computational graph; thus, a single iteration of the backpropagation algorithm will compute gradients for both M and L. Given a gradient ??? p for each parameter p ??? ?? L ??? ?? M , we then perform a supervised update p ??? p + A(p, ??? p ).

The key insight is that the convolution of the meta-learner over the weights of the learner forms a fully differentiable framework M L y.

Algorithm 1 Hebbian-Augmented Training Algorithm

for weights W ,

v +1 is a placeholder output as input to M 5:

Updates weight using local rule M 6:

Backpropagate loss H( v |L| , y).

for layer weight W in L and M do Backward pass 9: W ??? A

Apply gradient update using optimizer A 10:

return L, M Return updated learner and updated meta-learner

We hypothesize that the HAT algorithm will have three positive effects.

??? HAT will train the learner L faster since there are twice as many updates.

In ordinary backpropagation the metadata generated from the forward pass is computed and wasted; in HAT, the metadata is computed and used to generate a (potentially) useful update.

??? HAT will improve the convergence of L. The second update should introduce some stochasticity in the loss landscape since it is not directly tied to gradient descent, which may lead L into better local optima.

??? HAT will improve the performance of L when some examples are not labeled.

Backpropagation has no ability to learn from just the input x, while HAT is able to perform the unsupervised update.

We generate two learning curves to test these hypotheses: one with respect to time and one with respect to the proportion of labeled examples.

The charts below represent the aggregated learning curves of 100 pairs (L i , M i ).

We find that the effects of HAT on training are clearly positive.

The median accuracy of the neural nets trained by HAT is clearly increased along the learning curve, and the HAT-group neural nets reach a higher asymptotic value than the control group.

We do note that the two learning curves seem to inflect around the same point -HAT does not seem to cause a faster convergence, just a better one.

We attribute this to the meta-learner's convergence; it may take the meta-learner up to 0.5 epochs to start to have positive effects.

One potential concern with adding unsupervised meta-learner updates is that after the convergence of the base learner L, the meta-learner's continued output of non-zero updates might "bounce" the base learner out of an optimum.

Remarkably, we see in the above plot that the performance of the HAT-trained neural nets is quite stable for the entire 18 epochs of post-convergence duration.

To our surprise, we find that HAT is more effective when there are more labels, even though the self-supervised component of the algorithm is designed to take advantage of scarce labels.

We attribute this to slow convergence of the meta-learner M -when labels are scarce, the meta-learner may actually converge slower than the learner and thus provide bad update suggestions.

We would like insight into why HAT improves the training of neural nets over vanilla gradient descent.

Thus, we will analyze the functional form of the learned update rule M after it has fully converged.

Recall the setting from experiments 1 and 2: we generate 100 pairs of learners and meta-learners: (L i , M i ) for i ??? {1, ..., 100}. We then investigate the pointwise mean M of these meta-learners.

We first visualize the dependence of the function M on its inputs (v i , v j , w ij ).

We find that a remarkably linear dependence on v j explains almost all of the variance in the outputs of the meta-learned update rule.

This indicates that the rule is a "rich-get-richer" scheme: neurons that already fired with high magnitude will experience larger incoming weights and thus be encouraged to fire with high activation in the future.

This linear dependence is surprising since all of the hypothesized rules in neuroscience have a dependence on v i ??v j .

As a sanity check, we attempted to directly apply this update rule (???w ij ??? 2??v j ) without meta-learning to see if we can replicate HAT's performance improvement.

However, the results were decisively negative -HAT improves performance, but the a priori application of HAT's update rule decreases it.

We present three hypotheses:

??? Perhaps M learns a good update rule while L is training, then learns a degenerate rule once L has converged.

The sole purpose of this degenerate rule would be to not un-learn the important weights that have already converged (thus explaining the rich-gets-richer behavior of the rule f (??) = 2v j ).

Thus, analyzing the black-box function at epoch 20 is merely the wrong time -perhaps observing the meta-learned rule at epoch 1 would be more insightful and useful.

??? Perhaps M learns a good update rule in each run, and these update rules are all complex functions with no good low-order polynomial approximations; however, their pointwise mean (which is itself not a good local update rule) happens to be linear.

Thus, M is the wrong object to analyze and presents behaviors that are not indicative of the results of experiments 1 and 2.

??? Perhaps the learning of M is extremely transient.

For any given point in time, there is a different optimal learning rule, and our exercise in finding a fixed local, unsupervised update rule that is universal across training is futile.

The HAT algorithm demonstrates that local, unsupervised signals can provide performance-improving weight updates.

Neural nets under HAT converge to better asymptotic losses as long as there is sufficient time (> 0.5 epochs) and a sufficient number of labels (> 20% of the data is labeled).

The latter finding is surprising since the addition of an unsupervised learning algorithm depends on the presence of labels in order to deliver marginal benefits over gradient descent.

The underlying form of the learned rule that makes HAT successful is still a mystery; we find that while the meta-learner may learn a useful update rule during training, the meta-learner does not converge to this useful rule in the long run and instead devolves into a linear function ConvergedRule.

This converged function preserves fully-converged weights by reinforcing incoming weights for neurons with high activations.

The discovery that HAT does not stably converge to a function makes analysis quite difficult.

However, there is potential for future work to do more subtle analyses.

Imagine a time t during training in which the meta-learner M has converged to a useful function, but the learner L has not yet finished training.

A follow-up to this thesis might be to discover whether there such a time t exists, what the structure of M at time t is, and how M changes the weights of L at time t. One potential methodology might be to observe the function f not as a 3-dimensional function in (v i , w ij , v j ) but rather as a 4-dimensional function in (v i , w ij , v j , t).

Observing the function along the t-axis and checking for phase changes would shed light on whether a single useful update rule is learned during training or whether HAT's learning is truly transient and continuous.

If this follow-up were to succeed, then we could have an a priori rule to apply without having to metalearn update rules.

Extracting the local rules from multiple domains could either find that HAT learns a universal rule or that functional distance between two rules describes the "difference" between their originating domains.

??? Suppose we always metalearn the same rule, regardless of problem domain.

Optimal-Hebb is then a universal learning rule.

??? Suppose Optimal-Hebb is not universal for all problems.

For local rules R A , R B on problems A, B, integrating

gives an explicit measure for how similar A and B are.

This provides a systematic way to identify pairs of learning problems that are good candidates for transfer learning.

One implementation detail is notably not covered in the HAT pseudocode; this implementation detail patches an inadequacy in modern deep learning frameworks.

Given two neural net layers i and i+1 and minibatches of size B, we have B instances of | i | ?? i+1 neuron pairs, each of which has 3 salient properties (v i , w ij , v j ).

Therefore, we would like to apply the function M over the zeroth dimension of a tensor of size 3 ?? B ?? | i | ?? | i+1 | in order to compute the unsupervised weight updates.

However, as of this writing date, it is not possible to apply an arbitrary function M to slices of a tensor in parallel in any modern deep learning framework (e.g. Tensorflow, PyTorch, Keras); the reason is that this plays poorly with optimization of the computational graph.

We thus implement the application of M 's updates to the weights by convoluting M over a state tensor.

This is best clarified with an example.

Suppose we have a neural net with consecutive layers 1 , 2 of size 784 and 183, respectively.

Suppose further that we have batches of size 50.

Finally, suppose that we require a meta-learner that is a neural net of architecture 3 ?? 100 ?? 1.

We then copy the tensors along the boxed dimensions to stack them.

We instantiate M as a sequence of 3 composed functions:

1. a convolutional layer of kernel size 1 ?? 1 with 3 in-channels and 100 out-channels, 2. a ReLU activation, and 3.

a convolutional layer of kernel size 1 ?? 1 with 100 in-channels and 1 out-channels.

Applying this series of functions to a 1?? image with 3 channels is equivalent to passing the 3 channels into a neural net with architecture 3 ?? 100 ?? 1.

PyTorch (the framework used for this research) does not support the vectorization of arbitrary functions along torch tensors.

However, it does support (and heavily optimize for) convolutions.

Thus, we implement our neural net function M as a series of convolutions, and we convolve the function over the input tensor of size 3 ?? 50 ?? 183 ?? 784.

The output of M is of size 50 ?? 183 ?? 784; we average over the zeroth dimension to finally get a weight update of dimension 183 ?? 784, which is the same size as the original weight tensor.

@highlight

Metalearning unsupervised update rules for neural networks improves performance and potentially demonstrates how neurons in the brain learn without access to global labels.