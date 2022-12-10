Representations of sets are challenging to learn because operations on sets should be permutation-invariant.

To this end, we propose a Permutation-Optimisation module that learns how to permute a set end-to-end.

The permuted set can be further processed to learn a permutation-invariant representation of that set, avoiding a bottleneck in traditional set models.

We demonstrate our model's ability to learn permutations and set representations with either explicit or implicit supervision on four datasets, on which we achieve state-of-the-art results: number sorting, image mosaics, classification from image mosaics, and visual question answering.

Consider a task where each input sample is a set of feature vectors with each feature vector describing an object in an image (for example: person, table, cat).

Because there is no a priori ordering of these objects, it is important that the model is invariant to the order that the elements appear in the set.

However, this puts restrictions on what can be learned efficiently.

The typical approach is to compose elementwise operations with permutation-invariant reduction operations, such as summing (Zaheer et al., 2017) or taking the maximum (Qi et al., 2017) over the whole set.

Since the reduction operator compresses a set of any size down to a single descriptor, this can be a significant bottleneck in what information about the set can be represented efficiently (Qi et al., 2017; Le & Duan, 2018; Murphy et al., 2019) .We take an alternative approach based on an idea explored in Vinyals et al. (2015a) , where they find that some permutations of sets allow for easier learning on a task than others.

They do this by ordering the set elements in some predetermined way and feeding the resulting sequence into a recurrent neural network.

For instance, it makes sense that if the task is to output the top-n numbers from a set of numbers, it is useful if the input is already sorted in descending order before being fed into an RNN.

This approach leverages the representational capabilities of traditional sequential models such as LSTMs, but requires some prior knowledge of what order might be useful.

Our idea is to learn such a permutation purely from data without requiring a priori knowledge (section 2).

The key aspect is to turn a set into a sequence in a way that is both permutation-invariant, as well as differentiable so that it is learnable.

Our main contribution is a Permutation-Optimisation (PO) module that satisfies these requirements: it optimises a permutation in the forward pass of a neural network using pairwise comparisons.

By feeding the resulting sequence into a traditional model such as an LSTM, we can learn a flexible, permutation-invariant representation of the set while avoiding the bottleneck that a simple reduction operator would introduce.

Techniques used in our model may also be applicable to other set problems where permutation-invariance is desired, building on the literature of approaches to dealing with permutation-invariance (section 3).In four different experiments, we show improvements over existing methods (section 4).

The former two tasks measure the ability to learn a particular permutation as target: number sorting and image mosaics.

We achieve state-of-the-art performance with our model, which shows that our method is suitable for representing permutations in general.

The latter two tasks test whether a model can learn to solve a task that requires it to come up with a suitable permutation implicitly: classification from image mosaics and visual question answering.

We provide no supervision of what the permutation should be; the model has to learn by itself what permutation is most useful for the task at hand.

In the ordering cost C, elements of X are compared to each other (blue represents a negative value, red represents a positive value).

Gradients are applied to unnormalised permutations P (t) , which are normalised to proper permutations P (t) .Here, our model also beats the existing models and we improve the performance of a state-of-the-art model in VQA with it.

This shows that our PO module is able to learn good permutation-invariant representations of sets using our approach.

We will now describe a differentiable, and thus learnable model to turn an unordered set {x i } N with feature vectors as elements into an ordered sequence of these feature vectors.

An overview of the algorithm is shown in FIG0 and pseudo-code is available in Appendix A. The input set is represented as a matrix X = [x 1 , . . .

, x N ] T with the feature vectors x i as rows in some arbitrary order.

In the algorithm, it is important to not rely on the arbitrary order so that X is correctly treated as a set.

The goal is then to learn a permutation matrix P such that when permuting the rows of the input through Y = P X, the output is ordered correctly according to the task at hand.

When an entry P ik takes the value 1, it can be understood as assigning the ith element to the kth position in the output.

Our main idea is to first relate pairs of elements through an ordering cost, parametrised with a neural network.

This pairwise cost tells us whether an element i should preferably be placed before or after element j in the output sequence.

Using this, we can define a total cost that measures how good a given permutation is (subsection 2.1).

The second idea is to optimise this total cost in each forward pass of the module (subsection 2.2).

By minimising the total cost of a permutation, we improve the quality of a permutation with respect to the current ordering costs.

Crucially, the ordering cost function -and thus also the total cost function -is learned.

In doing so, the module is able to learn how to generate a permutation as is desired.

In order for this to work, it is important that the optimisation process itself is differentiable so that the ordering cost is learnable.

Because permutations are inherently discrete objects, a continuous relaxation of permutations is necessary.

For optimisation, we perform gradient descent on the total cost for a fixed number of steps and unroll the iteration, similar to how recurrent neural networks are unrolled to perform backpropagation-through-time.

Because the inner gradient (total cost differentiated with respect to permutation) is itself differentiable with respect to the ordering cost, the whole model is kept differentiable and we can train it with a standard supervised learning loss.

Note that as long as the ordering cost is computed appropriately (subsection 2.3), all operations used turn out to be permutation-invariant.

Thus, we have a model that respects the symmetries of sets while producing an output without those symmetries: a sequence.

This can be naturally extended to outputs where the target is not a sequence, but grids and lattices (subsection 2.4).

The total cost function measures the quality of a given permutation and should be lower for better permutations.

Because this is the function that will be optimised, it is important to understand what it expresses precisely.

The main ingredient for the total cost of a permutation is the pairwise ordering cost (details in subsection 2.3).

By computing it for all pairs, we obtain a cost matrix C where the entry C ij represents the ordering cost between i and j: the cost of placing element i anywhere before j in the output sequence.

An important constraint that we put on C is that C ij = −C ji .

In other words, if one ordering of i and j is "good" (negative cost), then the opposite ordering obtained by swapping them is "bad" (positive cost).

Additionally, this constraint means that C ii = 0.

This makes sure that two very similar feature vectors in the input will be similarly ordered in the output because their pairwise cost goes to 0.In this paper we use a straightforward definition of the total cost function: a sum of the ordering costs over all pairs of elements i and j. When considering the pair i and j, if the permutation maps i to be before j in the output sequence, this cost is simply C ij .

Vice versa, if the permutation maps i to be after j in the output sequence, the cost has to be flipped to C ji .

To express this idea, we define the total cost c : R N ×N → R of a permutation P as: DISPLAYFORM0 This can be understood as follows: If the permutation assigns element i to position u (so P iu = 1) and element j to position v (so P jv = 1), the sums over k and k simplify to 1 when v > u and −1 when v < u; permutation matrices are binary and only have one 1 in any row and column, so all other terms in the sums are 0.

That means that the term for each i and j becomes C ij when v > u and −C ij = C ji when v < u, which matches what we described previously.

Now that we can compute the total cost of a permutation, we want to optimise this cost with respect to a permutation.

After including the constraints to enforce that P is a valid permutation matrix, we obtain the following optimisation problem: DISPLAYFORM0 Optimisation over P directly is difficult due to the discrete and combinatorial nature of permutations.

To make optimisation feasible, a common relaxation is to replace the constraint that P ik ∈ {0, 1} with P ik ∈ [0, 1] BID9 .

With this change, the feasible set for P expands to the set of doublystochastic matrices, known as the Birkhoff or assignment polytope.

Rather than hard permutations, we now have soft assignments of elements to positions, analogous to the latent assignments when fitting a mixture of Gaussians model using Expectation-Maximisation.

Note that we do not need to change our total cost function after this relaxation.

Instead of discretely flipping the sign of C ij depending on whether element i comes before j or not, the sums over k and k give us a weight for each C ij that is based on how strongly i and j are assigned to positions.

This weight is positive when i is on average assigned to earlier positions than j and negative vice versa.

In order to perform optimisation of the cost under our constraints, we reparametrise P with the Sinkhorn operator S from Adams & Zemel (2011) (defined in Appendix B) so that the constraints are always satisfied.

We found this to lead to better solutions than projected gradient descent in initial experiments.

After first exponentiating all entries of a matrix, S repeatedly normalises all rows, then all columns of the matrix to sum to 1, which converges to a doubly-stochastic matrix in the limit.

DISPLAYFORM1 This ensures that P is always approximately a doubly-stochastic matrix.

P can be thought of as the unnormalised permutation while P is the normalised permutation.

By changing our optimisation to minimise P instead of P directly, all constraints are always satisfied and we can simplify the optimisation problem to min P c(P ) without any constraints.

It is now straightforward to optimise P with standard gradient descent.

First, we compute the gradient: DISPLAYFORM2 ∂c(P ) DISPLAYFORM3 From equation 4, it becomes clear that this gradient is itself differentiable with respect to the ordering cost C ij , which allows it to be learned.

In practice, both ∂c(P )/∂ P as well as ∂[∂c(P )/∂ P ]/∂C can be computed with automatic differentiation.

However, some implementations of automatic differentiation require the computation of c(P ) which we do not use.

In this case, implementing ∂c(P )/∂ P explicitly can be more efficient.

Also notice that if we define B jq = k >q P jk − k <q P jk , equation 4 is just the matrix multiplication CB and is thus efficiently computable.

For optimisation, P has to be initialised in a permutation-invariant way to preserve permutationinvariance of the algorithm.

In this paper, we consider a uniform initialisation so that all P ik = 1/N (PO-U model, left) and an initialisation that linearly assigns (Mena et al., 2018 ) each element to each position (PO-LA model, right).

DISPLAYFORM4 where w k is a different weight vector for each position k. Then, we perform gradient descent for a fixed number of steps T .

The iterative update using the gradient and a (learnable) step size η converges to the optimised permutation P (T ) : DISPLAYFORM5 One peculiarity of this is that we update P with the gradient of the normalised permutation P , not of the unnormalised permutation P as normal.

In other words, we do gradient descent on P but in equation 5 we set ∂P uv /∂ P pq = 1 when u = p, v = q, and 0 everywhere else.

We found that this results in significantly better permutations experimentally; we believe that this is because ∂P /∂ P vanishes too quickly from the Sinkhorn normalisation, which biases P away from good permutation matrices wherein all entries are close to 0 and 1 (Appendix D).The runtime of this algorithm is dominated by the computation of gradients of c(P ), which involves a matrix multiplication of two N × N matrices.

In total, the time complexity of this algorithm is T times the complexity of this matrix multiplication, which is Θ(N 3 ) in practice.

We found that typically, small values for T such as 4 are enough to get good permutations.

The ordering cost C ij is used in the total cost and tells us what the pairwise cost for placing i before j should be.

The key property to enforce is that the function F that produces the entries of C is anti-symmetric (F (x i , x j ) = −F (x j , x i )).

A simple way to achieve this is to define F as: DISPLAYFORM0 We can then use a small neural network for f to obtain a learnable F that is always anti-symmetric.

Lastly, C is normalised to have unit Frobenius norm.

This results in simply scaling the total cost obtained, but it also decouples the scale of the outputs of F from the step size parameter η to make optimisation more stable at inference time.

C is then defined as: DISPLAYFORM1 DISPLAYFORM2

In some tasks, it may be natural to permute the set into a lattice structure instead of a sequence.

For example, if it is known that the set contains parts of an image, it makes sense to arrange these parts back to an image by using a regular grid.

We can straightforwardly adapt our model to this by considering each row and column of the target grid as an individual permutation problem.

The total cost of an assignment to a grid is the sum of the total costs over all individual rows and columns of the grid.

The gradient of this new cost is then the sum of the gradients of these individual problems.

This results in a model that considers both row-wise and column-wise pairwise relations when permuting a set of inputs into a grid structure, and more generally, into a lattice structure.

The most relevant work to ours is the inspiring study by Mena et al. (2018) , where they discuss the reparametrisation that we use and propose a model that can also learn permutations implicitly in principle.

Their model uses a simple elementwise linear map from each of the N elements of the set to the N positions, normalised by the Sinkhorn operator.

This can be understood as classifying each element individually into one of the N classes corresponding to positions, then normalising the predictions so that each class only occurs once within this set.

However, processing the elements individually means that their model does not take relations between elements into account properly; elements are placed in absolute positions, not relative to other elements.

Our model differs from theirs by considering pairwise relations when creating the permutation.

By basing the cost function on pairwise comparisons, it is able to order elements such that local relations in the output are taken into account.

We believe that this is important for learning from permutations implicitly, because networks such as CNNs and RNNs rely on local ordering more than absolute positioning of elements.

It also allows our model to process variable-sized sets, which their model is not able to do.

Our work is closely related to the set function literature, where the main constraint is invariance to ordering of the set.

While it is always possible to simply train using as many permutations of a set as possible, using a model that is naturally permutation-invariant increases learning and generalisation capabilities through the correct inductive bias in the model.

There are some similarities with relation networks (Santoro et al., 2017) in considering all pairwise relations between elements as in our pairwise ordering function.

However, they sum over all non-linearly transformed pairs, which can lead to the bottleneck we mention in section 1.

Meanwhile, by using an RNN on the output of our model, our approach can encode a richer class of functions: it can still learn to simply sum the inputs, but it can also learn more complex functions where the learned order between elements is taken into account.

The concurrent work in (Murphy et al., 2019) discusses various approximations of averaging the output of a neural network over all possible permutations, with our method falling under their categorisation of a learned canonical input ordering.

Our model is also relevant to neural networks operating on graphs such as graph-convolutional networks (Kipf & Welling, 2017) .

Typically, a set function is applied to the set of neighbours for each node, with which the state of the node is updated.

Our module combined with an RNN is thus an alternative set function to perform this state update with.

Noroozi & Favaro FORMULA0 and BID7 show that it is possible to use permutation learning for representation learning in a self-supervised setting.

The model in BID7 is very similar to Mena et al. (2018) , including use of a Sinkhorn operator, but they perform significantly more processing on images with a large CNN (AlexNet) beforehand with the main goal of learning good representations for that CNN.

We instead focus on using the permuted set itself for representation learning in a supervised setting.

We are not the first to explore the usefulness of using optimisation in the forward pass of a neural network (for example, Stoyanov et al. FORMULA0 ; Domke FORMULA0 ; BID4 ).

However, we believe that we are the first to show the potential of optimisation for processing sets because -with an appropriate cost function -it is able to preserve permutation-invariance.

In OptNet BID1 , exact solutions to convex quadratic programs are found in a differentiable way through various techniques.

Unfortunately, our quadratic program is non-convex (Appendix E), which makes finding an optimal solution possibly NP-hard (Pardalos & Vavasis, 1991) .

We thus fall back to the simpler approach of gradient descent on the reparametrised problem to obtain a non-optimal, but reasonable solution.

Note that our work differs from learning to rank approaches such as BID5 and Severyn & Moschitti FORMULA0 , as there the end goal is the permutation itself.

This usually requires supervision on what the target permutation should be, producing a permutation with hard assignments at the end.

We require our model to produce soft assignments so that it is easily differentiable, since the main goal is not the permutation itself, but processing it further to form a representation of the set being permuted.

This means that other approaches that produce hard assignments such as Ptr-Net (Vinyals et al., 2015b) are also unsuitable for implicitly learning permutations, although using a variational approximation through Mena et al. FORMULA0 to obtain a differentiable permutation with hard assignments is a promising direction to explore for the future.

Due to the lack of differentiability, existing literature on solving minimum feedback arc set problems BID6 can not be easily used for set representation learning either.

Throughout the text, we will refer to our model with uniform assignment as PO-U, with linear assignment initialisation as PO-LA, and the model from Mena et al. FORMULA0 as LinAssign.

We perform a qualitative analysis of what comparisons are learned in Appendix C. Precise experimental details can be found in Appendix F and our implementation for all experiments is available at https: //github.com/Cyanogenoid/perm-optim for full reproducibility.

Some additional results including example image mosaic outputs can be found in Appendix G.

We start with the toy task of turning a set of random unsorted numbers into a sorted list.

For this problem, we train with fixed-size sets of numbers drawn uniformly from the interval [0, 1] and evaluate on different intervals to determine generalisation ability (for example: DISPLAYFORM0 We use the correctly ordered sequence as training target and minimise the mean squared error.

Following Mena et al. (2018) , during evaluation we use the Hungarian algorithm for solving a linear assignment problem with −P as the assignment costs.

This is done to obtain a permutation with hard assignments from our soft permutation.

Our PO-U model is able to sort all sizes of sets that we tried -5 to 1024 numbers -perfectly, including generalising to all the different evaluation intervals without any mistakes.

This is in contrast to all existing end-to-end learning-based approaches such as Mena et al. (2018) , which starts to make mistakes on [0, 1] at 120 numbers and no longer generalises to sets drawn from [1000, 1001] at 80 numbers.

Vinyals et al. (2015a) already starts making mistakes on 5 numbers.

Our stark improvement over existing results is evidence that the inductive biases due to the learned pairwise comparisons in our model are suitable for learning permutations, at least for this particular toy problem.

In subsection C.1, we investigate what it learns that allows it to generalise this well.

As a second task, we consider a problem where the model is given images that are split into n × n equal-size tiles and the goal is to re-arrange this set of tiles back into the original image.

We take these images from either MNIST, CIFAR10, or a version of ImageNet with images resized down to 64 × 64 pixels.

For this task, we use the alternative cost function described in subsection 2.4 to arrange the tiles into a grid rather than a sequence: this lets our model take relations within rows and columns into account.

Again, we minimise the mean squared error to the correctly permuted image and use the Hungarian algorithm during evaluation, matching the experimental setup in Mena et al. (2018) .

Due to the lack of reference implementation of their model for this experiment, we use our own implementation of their model, which we verified to reproduce their MNIST results closely.

Unlike them, we decide to not arbitrarily upscale MNIST images to get improved results for all models.

The mean squared errors for the different image datasets and different number of tiles an image is split into are shown in TAB0 .

First, notice that in essentially all cases, our model with linear assignment initialisation (PO-LA) performs best, often significantly so.

On the two more complex datasets CIFAR10 and ImageNet, this is followed by our PO-U model, then the LinAssign model.

We analyse what types of comparisons PO-U learns in subsection C.2.On MNIST, LinAssign performs better than PO-U on higher tile counts because images are always centred on the object of interest.

That means that many tiles only contain the background and end up completely blank; these tiles can be more easily assigned to the borders of the image by the LinAssign model than our PO-U model because the absolute position is much more important than the relative positioning to other tiles.

This also points towards an issue for these cases in our cost function: because two tiles that have the same contents are treated the same by our model, it is unable to place one blank tile on one side of the image and another blank tile on the opposite side, as this would require treating the two tiles differently.

This issue with backgrounds is also present on CIFAR10 to a lesser extent: notice how for the 3 × 3 case, the error of PO-U is much closer to LinAssign on CIFAR10 than on ImageNet, where PO-U is much better comparatively.

This shows that the PO-U model is more suitable for more complex images when relative positioning matters more, and that PO-LA is able to combine the best of both methods.

We now turn to tasks where the goal is not producing the permutation itself, but learning a suitable permutation for a different task.

For these tasks, we do not provide explicit supervision on what the permutation should be; an appropriate permutation is learned implicitly while learning to solve another task.

As the dataset, we use a straightforward modification of the image mosaic task.

The image tiles are assigned to positions on a grid as before, which are then concatenated into a full image.

This image is fed into a standard image classifier (ResNet-18 (He et al., 2015) ) which is trained with the usual cross-entropy loss to classify the image.

The idea is that the network has to learn some permutation of the image tiles so that the classifier can classify it accurately.

This is not necessarily the permutation that restores the original image faithfully.

One issue with this set-up we observed is that with big tiles, it is easy for a CNN to ignore the artefacts on the tile boundaries, which means that simply permuting the tiles randomly gets to almost the same test accuracy as using the original image.

To prevent the network from avoiding to solve the task, we first pre-train the CNN on the original dataset without permuting the image tiles.

Once it is fully trained, we freeze the weights of this CNN and train only the permutation mechanism.2 × 2 3 × 3 4 × 4 5 × 5 2 × 2 3 × 3 4 × 4 5 × 5 2 × 2 3 × 3 4 × 4 5 ×We show our results in TAB1 .

Generally, a similar trend to the image mosaic task with explicit supervision can be seen.

Our PO-LA model usually performs best, although for ImageNet PO-U is consistently better.

This is evidence that for more complex images, the benefits of linear assignment decrease (and can actually detract from the task in the case of ImageNet) and the importance of the optimisation process in our model increases.

With higher number of tiles on MNIST, even though PO-U does not perform well, PO-LA is clearly superior to only using LinAssign.

This is again due to the fully black tiles not being able to be sorted well by the cost function with uniform initialisation.

As the last task, we consider the much more complex problem of visual question answering (VQA): answering questions about images.

We use the VQA v2 dataset BID3 Goyal et al., 2017) , which in total contains around 1 million questions about 200,000 images from MS-COCO with 6.5 million human-provided answers available for training.

We use bottom-up attention features BID2 as representation for objects in the image, which for each image gives us a set (size varying from 10 to 100 per image) of bounding boxes and the associated feature vector that encodes the contents of the bounding box.

These object proposals have no natural ordering a priori.

We use the state-of-the-art BAN model (Kim et al., 2018) as a baseline and perform a straightforward modification to it to incorporate our module.

For each element in the set of object proposals, we concatenate the bounding box coordinates, features, and the attention value that the baseline model generates.

Our model learns to permute this set into a sequence, which is fed into an LSTM.

We take the last cell state of the LSTM to be the representation of the set, which is fed back into the baseline model.

This is done for each of the eight attention glimpses in the BAN model.

We include another baseline model (BAN + LSTM) that skips the permutation learning, directly processing the set with the LSTM.Our results on the validation set of VQA v2 are shown in TAB3 .

We improve on the overall performance of the state-of-the-art model by 0.37% -a significant improvement for this datasetwith 0.27% of this improvement coming from the learned permutation.

This shows that there is a substantial benefit to learning an appropriate permutation through our model in order to learn better set representations.

Our model significantly improves on the number category, despite the inclusion of a counting module BID2 specifically targeted at number questions in the baseline.

This is evidence that the representation learned through the permutation is non-trivial.

Note that the improvement through our model is not simply due to increased model size and computation: Kim et al. (2018) found that significantly increasing BAN model size, increasing computation time similar in scale to including our model, does not yield any further gains.

In this paper, we discussed our Permutation-Optimisation module to learn permutations of sets using an optimisation-based approach.

In various experiments, we verified the merit of our approach for learning permutations and, from them, set representations.

We think that the optimisation-based approach to processing sets is currently underappreciated and hope that the techniques and results in this paper will inspire new algorithms for processing sets in a permutation-invariant manner.

Of course, there is plenty of work to be done.

For example, we have only explored one possible function for the total cost; different functions capturing different properties may be used.

The main drawback of our approach is the cubic time complexity in the set size compared to the quadratic complexity of Mena et al. FORMULA0 , which limits our model to tasks where the number of elements is relatively small.

While this is acceptable on the real-world dataset that we used -VQA with up to 100 object proposals per image -with only a 30% increase in computation time, our method does not scale to the much larger set sizes encountered in domains such as point cloud classification.

Improvements in the optimisation algorithm may improve this situation, perhaps through a divide-and-conquer approach.

We believe that going beyond tensors as basic data structures is important for enabling higher-level reasoning.

As a fundamental mathematical object, sets are a natural step forward from tensors for modelling unordered collections.

The property of permutation invariance lends itself to greater abstraction by allowing data that has no obvious ordering to be processed, and we took a step towards this by learning an ordering that existing neural networks are able to take advantage of.

Algorithm 1 Forward pass of permutation-optimisation algorithm 1: Input: X ∈ R N ×M with x i as rows in arbitrary order 2: Learnable parameters: weights that parametrise F , step size η 3: DISPLAYFORM0 compute ordering costs (equation 10) 5: initialise P either uniform or linear assignment init (equation 6) 6: for t ← 1, T do 7: DISPLAYFORM1 normalise assignment with Sinkhorn operator (Appendix B)8:G ← ∂c(P )/∂P compute gradient of normalised assignment (equation 4) 9:P ← P − ηG gradient descent step on unnormalised assignment (equation 7) 10: end for 11: P ← S( P ) 12: Y ← P X permute rows of X to obtain output Y

The Sinkhorn operator S as defined in Adams & Zemel FORMULA0 is: DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 T r normalises each row, T c normalises each column of a square matrix X to sum to one.

This formulation is different from the normal Sinkhorn operator by Sinkhorn (1964) by exponentiating all entries first and running for a fixed number of steps L instead of for steps approaching infinity.

Mena et al. FORMULA0 include a temperature parameter on the exponentiation, which acts analogously to temperature in the softmax function.

In this paper, we fix L to 4.

First, we investigate what comparison function F is learned for the number sorting task.

We start with plotting the outputs of F for different pairs of inputs in Figure 2 .

From this, we can see that it learns a sensible comparison function where it outputs a negative number when the first argument is lower than the second, and a positive number vice versa.

The easiest way to achieve this is to learn f (x i , x j ) = x i , which results in F (x i , x j ) = x i − x j .

By plotting the outputs of the learned f in Figure 3 we can see that something close to this has indeed been learned.

The learned f mostly depends on the second argument and is a scaled and shifted version of it.

It has not learned to completely ignore the first argument, but the deviations from it are small enough that the cost function of the permutation is able to compensate for it.

We can see that there is a faint grey diagonal area going from (0, 0) to (1, 1) and to (1000, 1000), which could be an artifact from F having small gradients due to its skew-symmetry when two numbers are close to each other.

Next, we investigate the behaviour of F on the image mosaic task.

Since our model uses the outputs of F in the optimisation process, we find it easier to interpret F over f in the subsequent analysis.

We start by looking at the output of F 1 (costs for left-to-right ordering) and F 2 (costs for top-to-bottom ordering) for MNIST 2 × 2, shown in Figure 4 .

First, there is a clear entry in each row and column of both F 1 and F 2 that has the highest absolute cost (high colour saturation) whenever the corresponding tiles fit together correctly.

This shows that it successfully learned to be confident what order two tiles should be in when they fit together.

From the two 2-by-2 blocks of red and blue on the anti-diagonal, we can also see that it has learned that for the per-row comparisons (F 1 ), the tiles that should go into the left column should generally compare to less than (i.e. should be permuted to be to the left of) the tiles that go to the right.

Similarly, for the per-column comparisons (F 2 ) tiles that should be at the top compare to less than tiles that should be at the bottom.

Lastly, F 1 has a low absolute cost when comparing two tiles that belong in the same column.

These are the entries in the matrix at the coordinates (1, 2), (2, 1), (4, 3), and (3, 4).

This makes sense, as F 1 is concerned with whether one tile should be to the left or right of another, so tiles that belong in the same column should not have a preference either way.

A similar thing applies to F 2 for tiles that belong in the same column.

Figure 4: Outputs of F 1 (left half, row comparisons) and F 2 (right half, column comparisons) for pairs of tiles from an image in MNIST.

For F 1 , the tiles are sorted left-to-right if only F 1 was used as cost.

For F 2 , the tiles are sorted top-to-bottom if only F 2 was used as cost.

Blue indicates that the tile to the left of this entry should be ordered left of the tile at the top for F 1 , the tile on the left should be ordered above the tile at the top for F 2 .

The opposite applies for red.

The saturation of the colour indicates how strong this ordering is.

Next, we investigate what positions within the tiles F 1 and F 2 are most sensitive to.

This illustrates what areas of the tiles are usually important for making comparisons.

We do this by computing the gradients of the absolute values of F with respect to the input tiles and averaging over many inputs.

For MNIST 2 × 2 ( FIG4 , it learns no particular spatial pattern for F 1 and puts slightly more focus away from the centre of the tile for F 2 .

As we will see later, it learns something that is very content-dependent rather than spatially-dependent.

With increasing numbers of tiles on MNIST, it tends to focus more on edges, and especially on corners.

For the CIFAR10 dataset FIG5 ), there is a much clearer distinction between left-right comparisons for F 1 and top-bottom comparisons for F 2 .

For the 2 × 2 and 4 × 4 settings, it relies heavily on the pixels on the left and right borders for left-to-right comparisons, and top and bottom edges for top-to-bottom comparisons.

Interestingly, F 1 in the 3 × 3 setting (middle pair) on CIFAR10 focuses on the left and right halves of the tiles, but specifically avoids the borders.

A similar thing applies to F 2 , where a greater significance is given to pixels closer to the middle of the image rather than only focusing on the edges.

This suggests that it learns to not only match up edges as with the other tile numbers, but also uses the content within the tile to do more sophisticated content-based comparisons.

Figure 7: Gradient maps of pairs of tiles from MNIST for F 1 (left half) and F 2 (right half).

Each group of four consists of: tile 1, tile 2, gradient of F (t 1 , t 2 ) with respect to tile 1, gradient of F (t 2 , t 1 ) with respect to tile 2.Figure 8: Gradient maps of pairs of tiles from CIFAR10 for F 1 (left half) and F 2 (right half).

Each group of four consists of: tile 1, tile 2, gradient of F (t 1 , t 2 ) with respect to tile 1, gradient of F (t 2 , t 1 ) with respect to tile 2.

Lastly, we can look at the gradients of F with respect to the input tiles for specific pairs of tiles, shown in Figure 7 and Figure 8 .

This gives us a better insight into what changes to the input tiles would affect the cost of the comparison the most.

These figures can be understood as follows: for each pair of tiles, we have the corresponding two gradient maps next to them.

Brightening the pixels for the blue entries in these gradient maps would order the corresponding tile more strongly towards the left for F 1 and towards the top for F 2 .

The opposite applies to brightening the pixels with red entries.

Vice versa, darkening pixels with blue entries orders the tile more strongly towards the right for F 1 and the bottom for F 2 .

More saturated colours in the gradient maps correspond to greater effects on the cost when changing those pixels.

We start with gradients on the tiles for an input showing the digit 2 on MNIST 2 × 2 in Figure 7 .

We focus on the first row, left side, which shows a particular pair of tiles from this image and their gradients of F 1 (left-to-right ordering), and we share some of our observations here:• The gradients of the second tile show that to encourage the permutation to place it to the right of the first tile, it is best to increase the brightness of the curve in tile 2 that is already white (red entries in tile 2 gradient map) and decrease the black pixels around it (blue entries).

This means that it recognised that this type of curve is important in determining that it should be placed to the right, perhaps because it matches up with the start of the curve from tile 1.

We can imagine the curve in the gradient map of tile 2 roughly forming part of a 7 rather than a 2 as well, so it is not necessarily looking for the curve of a 2 specifically.• In the gradient map of the first tile, we can see that to encourage it to be placed to the left of tile 2, increasing the blue entries would form a curve that would make the first tile look like part of an 8 rather than a 2, completing the other half of the curve from tile 2.

This means that it has learned that to match something with the shape in tile 2, a loop that completes it is best, but the partial loop that we have in tile 1 satisfies part of this too.• Notice how the gradient of tile 1 changes quite a bit when going from row 1 to row 3, where it is paired up with different tiles.

This suggests that the comparison has learned something about the specific comparison between tiles being made, rather than learning a general trend of where the tile should go.

The latter is what a linear assignment model is limited to doing because it does not model pairwise interactions.• In the third row, we can see that even though the two tiles do not match up, there is a red blob on the left side of the tile 2 gradient map.

This blob would connect to the top part of the line in tile 1, so it makes sense that making the two tiles match up more on the border would encourage tile 2 to be ordered to the right of tile 1.Similar observations apply to the right half of Figure 7 , such as row 5, where tile 1 (which should go above tile 2) should have its pixels in the bottom left increased and tile 2 should have its pixels in the top left increased in order for tile 1 to be ordered before (i.e. above) tile 2 more strongly.

On CIFAR10 2 × 2 in Figure 8 , it is enough to focus on the borders of the tiles.

Here, it is striking how specifically it tries to match edge colours between tiles.

For example, consider the blue sky in the left half (F 1 ), row 6.

To order tile 1 to the left of tile 2, we should change tile 1 to have brighter sky and darker red on the right border, and also darken the black on the left border so that it matches up less well with the right border of tile 2, where more of the bright sky is visible.

For tile 2, the gradient shows that it should also match up more on the left border, and have increase the amount of bright pixels, i.e. sky, on the right border, again so that it matches up less well with the left border of tile 1 if they were to be ordered the opposing way.

First, the gradient of S(X) is: DISPLAYFORM0 where 1 is the indicator function that returns 1 if the condition is true and 0 otherwise.

We compared the entropy of the permutation matrices obtained with and without using the "proper" gradient with ∂S( P )/∂ P as term in it and found that our version has a significantly lower entropy.

To understand this, it is enough to focus on the first two terms in equation 16, which is essentially the gradient of a softmax function applied row-wise to P .Let x be a row in P and s i be the ith entry in the softmax function applied to x. Then, the gradient is: DISPLAYFORM1 Since this is a product of entries in a probability distribution, the gradient vanishes quickly as we move towards a proper permutation matrix (all entries very close to 0 or 1).

By using our alternative update and thus removing this term from our gradient, we can avoid the vanishing gradient problem.

Gradient descent is not efficient when the gradient vanishes towards the optimum and the optimumin our case a permutation matrix with exact ones and zeros as entries -is infinitely far away.

Since we prefer to use a small number of steps in our algorithm for efficiency, we want to reach a good solution as quickly as possible.

This justifies effectively ignoring the step size that the gradient suggests and simply taking a step in a similar direction as the gradient in order to be able to saturate the Sinkhorn normalisation sufficiently, thus obtaining a doubly stochastic matrix that is closer to a proper permutation matrix in the end.

We can write our total cost function as a quadratic program in the standard x T Qx form with linear constraints.

We leave out the constraints here as they are not particularly interesting.

First, we can define O ∈ R N ×N as: DISPLAYFORM0 and with it, Q ∈ R N 2 ×N 2 as: DISPLAYFORM1 Then we can write the cost function as: DISPLAYFORM2 where there is some bijection between a pair of indices (i, k) and the index l and p is a flattened version of P with p l = P ik .

Q is indefinite because the total cost can be negative: a uniform initialisation for P has a cost of 0, better permutations have negative cost, worse permutations have positive cost.

Thus, the problem is non-convex and the problem is possibly NP-hard.

Also, since we have flattened P into p, the number of optimisation variables is quadratic in the set size N .

Even if this were a convex quadratic program, methods such as OptNet BID1 have cubic time complexity in the number of optimisation variables, which makes it O(N 6 ) for our case.

All of our experiments can be reproduced using our implementation at https:// github.com/Cyanogenoid/perm-optim in PyTorch (Paszke et al., 2017) through the experiments/all.sh script.

For the former three experiments, we use the following hyperparameters throughout:• Optimiser:

Adam (Kingma & Ba, 2015) (default settings in PyTorch: β 1 = 0.9, β 2 = 0.999, = 10 −8 )• Initial step size η in inner gradient descent: 1.0All weights are initialised with Xavier initialisation (Glorot & Bengio, 2010) .

We choose the f within the ordering cost function F to be a small MLP.

The input to f has 2 times the number of dimensions of each element, obtained by concatenating the pair of elements.

This is done for all pairs that can be formed from the input set.

This is linearly projected to some number of hidden units to which a ReLU activation is applied.

Lastly, this is projected down to 1 dimension for sorting numbers and VQA, and 2 dimensions for assembling image mosaics (1 output for row-wise costs, 1 output for column-wise costs).

These outputs are used for creating the ordering cost matrix C. The ordering cost function F concatenates the two floats of each pair and applies a 2-layer MLP that takes the 2 inputs to 16 hidden units, ReLU activation, then to one output.

For evaluation, we switch to double precision floating point numbers.

This is because for the interval [1000, 1001] , as the set size increases, there are not enough unique single precision floats in that interval for the sets to contain only unique floats with high probability (the birthday problem).

Using double precision floats avoids this issue.

Note that using single precision floats is enough for the other intervals and smaller set sizes, and training is always done on the interval [0, 1] at single precision.

For all three image datasets from which we take images (MNIST, CIFAR10, ImageNet), we first normalise the inputs to have zero mean and standard deviation one over the dataset as is common practice.

For ImageNet, we crop rectangular images to be square by reducing the size of the longer side to the length of the shorter side (centre cropping).

Images that are not exactly divisible by the number of tiles are first rescaled to the nearest bigger image size that is exactly divisible.

Following Mena et al. (2018) , we process each tile with a 5 × 5 convolution with padding and stride 1, 2 × 2 max pooling, and ReLU activation.

This is flattened into a vector to obtain the feature vector for each tile, which is then fed into our F .

Unlike Mena et al. (2018), we decide not to arbitrarily upscale MNIST images by a factor of two, even when upscaling results in slightly better performance in general.

While we were able to mostly reproduce their MNIST results, we were not able to reproduce their ImageNet results for the 3 × 3 case.

In general, we observed that good settings for their model also improved the results of our PO-U and PO-LA models.

Better hyperparameters than what we used should improve all models similarly while keeping the ordering of how well they perform the same.

This task is also known as jigsaw puzzle (Noroozi & Favaro, 2016 ), but we decided on naming it image mosaics because the tiles are square which can lead to multiple solutions, rather than the typical unique solution in traditional jigsaw puzzles enforced by the different tile shapes.

We use the same setting as for the image mosaics, but further process the output image with a ResNet-18.

For MNIST and CIFAR10, we replace the first convolutional layer with one that has a 3 × 3 kernel size and no striding.

This ResNet-18 is first trained on the original dataset for 20 epochs (1 for ImageNet), though images may be rescaled if the image size is not divisible by the number of tiles per side.

All weights are then frozen and the permutation method is trained for 20 epochs (1 for ImageNet).

As stated previously, this is necessary in order for the ResNet-18 to not use each tile individually and ignore the resulting artefacts from the permuted tiles.

This is also one of the reasons why we downscale ImageNet images to 64 × 64 pixels.

Because the resulting image tiles are so big while the receptive field of ResNet-18 is relatively small if we were to use 256 × 256 images, the permutation artefacts barely affect results because they are only a small fraction of the globally-pooled features.

The permutation permutes each set of tiles, which are reconstructed (without use of the Hungarian algorithm) into an image, which is then processed by the ResNet-18.

We observed that the LinAssign model by Mena et al. (2018) consistently results in NaN values after Sinkhorn normalisation in this set-up, despite our Sinkhorn implementation using the numericallystable version of softmax with the exp-normalise trick.

We avoided this issue by clipping the outputs of their model into the [-10, 10] interval before Sinkhorn normalisation.

We did not observe these NaN issues with our PO-U model.

We use the official implementation of BAN as baseline without changing any of the hyperparameters.

We thus refer to Kim et al. (2018) for details of their model architecture and hyperparameters.

The only change to hyperparameters that we make is reducing the batch size from 256 to 112 due to the GPU memory requirements of the baseline model, even without our permutation mechanism.

The BAN model generates attention weights between all object proposals in a set and words of the question.

We take the attention weight for a single object proposal to be the maximum attention weight for that proposal over all words of the question, the same as in their integration of the counting module.

Each element of the set, corresponding to object proposals, is the concatenation of this attention logit, bounding box coordinates, and the feature vector projected from 2048 down to 8 dimensions.

We found this projection necessary to not inhibit learning of the rest of the model, which might be due to gradient clipping or other hyperparameters that are no longer optimal in the BAN model.

This set of object proposals is then permuted with T = 3 and a 2-layer MLP with hidden dimension 128 for f to produce the ordering costs.

The elements in the permuted sequence are weighted by how relevant each proposal is (sigmoid of the corresponding attention logit) and the sequence is then fed into an LSTM with 128 units.

The last cell state of the LSTM is the set representation which is projected, ReLUd, and added back into the hidden state of the BAN model.

The remainder of the BAN model is now able to use information from this set representation.

There are 8 attention glimpses, so we process each of these with a PO-U module and an LSTM with shared parameters across these 8 glimpses.

An interesting aspect we observed throughout all experiments is how the learned step size η changes during training.

At the start of training, it decreases from its initial value of 1, thus reducing the influence of the permutation mechanism.

Then, η starts rising again, usually ending up at a value above 1 at the end of training.

This can be explained by the ordering cost being very inaccurate at the start of training, since it has not been trained yet.

Through training, the ordering cost improves and it becomes more beneficial for the influence of the PO module on the permutation to increase.

In TAB4 , we show the accuracy corresponding to the results in TAB0 where the permutation has been trained with explicit supervision.

In FIG0 , FIG0 , and FIG0 , we show some example reconstructions that have been learnt by our PO-U model.

Starting from a uniform assignment at the top, the figures show reconstructions as a permutation is being optimised.

Generally, it is able to reconstruct most images fairly well.

Due to the poor quality of many of these reconstructions (particularly on ImageNet), the last two figures show reconstructions on the 2 × 2 versions of the datasets rather than 3 × 3.

In Table 5 , we show the mean squared error reconstruction loss corresponding to the results in TAB1 .

These show similar trends as before.

In FIG0 , FIG0 , and FIG0 , we show some example reconstructions that have been learnt by our PO-U model on 3 × 3 versions of the image datasets.

Because the quality of implicit CIFAR10 and ImageNet reconstructions are relatively poor, we also include FIG0 , and FIG0 on 2 × 2 versions.

Starting from a uniform assignment at the top, the figures show reconstructions as a permutation is being optimised.

The reconstructions here are clearly noisier than before due the supervision only being implicit.

This is evidence that while our method is superior to existing methods in terms of reconstruction error and accuracy of the classification, there is still plenty of room for improvement to allow for better implicitly learned permutations.

Keep in mind that it is not necessary for the permutation to produce the original image exactly, as long as the CNN can consistently recognise what the permutation method has learned.

Our models tend to naturally learn reconstructions that are more similar to the original image than the LinAssign model.

implicit supervision.

These examples have not been cherry-picked.

@highlight

Learn how to permute a set, then encode permuted set with RNN to obtain a set representation.