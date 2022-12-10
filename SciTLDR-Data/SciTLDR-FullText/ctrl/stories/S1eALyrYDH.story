In this paper, we propose an end-to-end deep learning model, called E2Efold, for RNA secondary structure prediction which can effectively take into account the inherent constraints in the problem.

The key idea of E2Efold is to directly predict the RNA base-pairing matrix, and use an unrolled constrained programming algorithm as a building block in the architecture to enforce constraints.

With comprehensive experiments on benchmark datasets, we demonstrate the superior performance of E2Efold: it predicts significantly better structures compared to previous SOTA (29.7% improvement in some cases in F1 scores and even larger improvement for pseudoknotted structures) and runs as efficient as the fastest algorithms in terms of inference time.

Ribonucleic acid (RNA) is a molecule playing essential roles in numerous cellular processes and regulating expression of genes (Crick, 1970) .

It consists of an ordered sequence of nucleotides, with each nucleotide containing one of four bases: Adenine (A), Guanine (G), Cytosine (C) and Uracile (U).

This sequence of bases can be represented as

x := (x 1 , . . .

, x L ) where x i ∈ {A, G, C, U }, which is known as the primary structure of RNA.

The bases can bond with one another to form a set of base-pairs, which defines the secondary structure.

A secondary structure can be represented by a binary matrix A * where A * ij = 1 if the i, j-th bases are paired (Fig 1) .

Discovering the secondary structure of RNA is important for understanding functions of RNA since the structure essentially affects the interaction and reaction between RNA and other cellular components.

Although secondary structure can be determined by experimental assays (e.g. X-ray diffraction), it is slow, expensive and technically challenging.

Therefore, computational prediction of RNA secondary structure becomes an important task in RNA research and is useful in many applications such as drug design (Iorns et al., 2007) .

(ii) Pseudo-knot (i) Nested Structure Research on computational prediction of RNA secondary structure from knowledge of primary structure has been carried out for decades.

Most existing methods assume the secondary structure is a result of energy minimization, i.e., A * = arg min A E x (A).

The energy function is either estimated by physics-based thermodynamic experiments (Lorenz et al., 2011; Markham & Zuker, 2008) or learned from data (Do et al., 2006) .

These approaches are faced with a common problem that the search space of all valid secondary structures is exponentially-large with respect to the length L of the sequence.

To make the minimization tractable, it is often assumed the base-pairing has a nested structure (Fig 2 left) , and the energy function factorizes pairwisely.

With this assumption, dynamic programming (DP) based algorithms can iteratively find the optimal structure for subsequences and thus consider an enormous number of structures in time O(L 3 ).

Although DP-based algorithms have dominated RNA structure prediction, it is notable that they restrict the search space to nested structures, which excludes some valid yet biologically important RNA secondary structures that contain 'pseudoknots', i.e., elements with at least two non-nested base-pairs (Fig 2 right) .

Pseudoknots make up roughly 1.4% of base-pairs (Mathews & Turner, 2006) , and are overrepresented in functionally important regions (Hajdin et al., 2013; Staple & Butcher, 2005) .

Furthermore, pseudoknots are present in around 40% of the RNAs.

They also assist folding into 3D structures (Fechter et al., 2001 ) and thus should not be ignored.

To predict RNA structures with pseudoknots, energy-based methods need to run more computationally intensive algorithms to decode the structures.

In summary, in the presence of more complex structured output (i.e., pseudoknots), it is challenging for energy-based approaches to simultaneously take into account the complex constraints while being efficient.

In this paper, we adopt a different viewpoint by assuming that the secondary structure is the output of a feed-forward function, i.e., A * = F θ (x), and propose to learn θ from data in an end-to-end fashion.

It avoids the second minimization step needed in energy function based approach, and does not require the output structure to be nested.

Furthermore, the feed-forward model can be fitted by directly optimizing the loss that one is interested in.

Despite the above advantages of using a feed-forward model, the architecture design is challenging.

To be more concrete, in the RNA case, F θ is difficult to design for the following reasons:

(i) RNA secondary structure needs to obey certain hard constraints (see details in Section 3), which means certain kinds of pairings cannot occur at all (Steeg, 1993) .

Ideally, the output of F θ needs to satisfy these constraints. (ii) The number of RNA data points is limited, so we cannot expect that a naive fully connected network can learn the predictive information and constraints directly from data.

Thus, inductive biases need to be encoded into the network architecture. (iii) One may take a two-step approach, where a post-processing step can be carried out to enforce the constraints when F θ predicts an invalid structure.

However, in this design, the deep network trained in the first stage is unaware of the post-processing stage, making less effective use of the potential prior knowledge encoded in the constraints.

In this paper, we present an end-to-end deep learning solution which integrates the two stages.

The first part of the architecture is a transformer-based deep model called Deep Score Network which represents sequence information useful for structure prediction.

The second part is a multilayer network called Post-Processing Network which gradually enforces the constraints and restrict the output space.

It is designed based on an unrolled algorithm for solving a constrained optimization.

These two networks are coupled together and learned jointly in an end-to-end fashion.

Therefore, we call our model E2Efold.

By using an unrolled algorithm as the inductive bias to design Post-Processing Network, the output space of E2Efold is constrained (see Fig 3 for an illustration), which makes it easier to learn a good model in the case of limited data and also reduces the overfitting issue.

Yet, the constraints encoded in E2Efold are flexible enough such that pseudoknots are included in the output space.

In summary, E2Efold strikes a nice balance between model biases for learning and expressiveness for valid RNA structures.

We conduct extensive experiments to compare E2Efold with state-of-the-art (SOTA) methods on several RNA benchmark datasets, showing superior performance of E2Efold including:

• being able to predict valid RNA secondary structures including pseudoknots;

• running as efficient as the fastest algorithm in terms of inference time;

• producing structures that are visually close to the true structure;

• better than previous SOTA in terms of F1 score, precision and recall.

Although in this paper we focus on RNA secondary structure prediction, which presents an important and concrete problem where E2Efold leads to significant improvements, our method is generic and can be applied to other problems where constraints need to be enforced or prior knowledge is provided.

We imagine that our design idea of learning unrolled algorithm to enforce constraints can also be transferred to problems such as protein folding and natural language understanding problems (e.g., building correspondence structure between different parts in a document).

Classical RNA folding methods identify candidate structures for an RNA sequence energy minimization through DP and rely on thousands of experimentally-measured thermodynamic parameters.

A few widely used methods such as RNAstructure , Vienna RNAfold (Lorenz et al., 2011) and UNAFold (Markham & Zuker, 2008) (Huang et al., 2019) achieved linear run time O(L) by applying beam search, but it can not handle pseudoknots in RNA structures.

The prediction of lowest free energy structures with pseudoknots is NP-complete (Lyngsø & Pedersen, 2000) , so pseudoknots are not considered in most algorithms.

Heuristic algorithms such as HotKnots (Andronescu et al., 2010) and Probknots (Bellaousov & Mathews, 2010) have been made to predict structures with pseudoknots, but the predictive accuracy and efficiency still need to be improved.

Learning-based RNA folding methods such as ContraFold (Do et al., 2006) and ContextFold (Zakov et al., 2011) have been proposed for energy parameters estimation due to the increasing availability of known RNA structures, resulting in higher prediction accuracies, but these methods still rely on the above DP-based algorithms for energy minimization.

A recent deep learning model, CDPfold , applied convolutional neural networks to predict base-pairings, but it adopts the dot-bracket representation for RNA secondary structure, which can not represent pseudoknotted structures.

Moreover, it requires a DP-based post-processing step whose computational complexity is prohibitive for sequences longer than a few hundreds.

Learning with differentiable algorithms is a useful idea that inspires a series of works (Hershey et al., 2014; Belanger et al., 2017; Ingraham et al., 2018; Chen et al., 2018; Shrivastava et al., 2019) , which shared similar idea of using differentiable unrolled algorithms as a building block in neural architectures.

Some models are also applied to structured prediction problems (Hershey et al., 2014; Pillutla et al., 2018; Ingraham et al., 2018) , but they did not consider the challenging RNA secondary structure problem or discuss how to properly incorporating constraints into the architecture.

OptNet (Amos & Kolter, 2017) integrates constraints by differentiating KKT conditions, but it has cubic complexity in the number of variables and constraints, which is prohibitive for the RNA case.

Dependency parsing in NLP is a different but related problem to RNA folding.

It predicts the dependency between the words in a sentence.

Similar to nested/non-nested structures, the corresponding terms in NLP are projective/non-projective parsing, where most works focus on the former and DP-based inference algorithms are commonly used (McDonald et al., 2005) .

Deep learning models (Dozat & Manning, 2016; Kiperwasser & Goldberg, 2016) are proposed to proposed to score the dependency between words, which has a similar flavor to the Deep Score Network in our work.

In the RNA secondary structure prediction problem, the input is the ordered sequence of bases x = (x 1 , . . .

, x L ) and the output is the RNA secondary structure represented by a matrix A * ∈ {0, 1} L×L .

Hard constraints on the forming of an RNA secondary structure dictate that certain kinds of pairings cannot occur at all (Steeg, 1993) .

Formally, these constraints are:

(i) Only three types of nucleotides combinations, B := {AU, U A}∪ {GC, CG} ∪ {GU, U G}, can form base-pairs.

(ii) No sharp loops are allowed.

∀|i − j| < 4, A ij = 0.

(iii) There is no overlap of pairs, i.e., it is a matching.

∀i, The space of all valid secondary structures contains all symmetric matrices A ∈ {0, 1} L×L that satisfy the above three constraints.

This space is much smaller than the space of all binary matrices {0, 1}

L×L .

Therefore, if we could incorporate these constraints in our deep model, the reduced output space could help us train a better predictive model with less training data.

We do this by using an unrolled algorithm as the inductive bias to design deep architecture.

In the literature on feed-forward networks for structured prediction, most models are designed using traditional deep learning architectures.

However, for RNA secondary structure prediction, directly using these architectures does not work well due to the limited amount of RNA data points and the hard constraints on forming an RNA secondary structure.

These challenges motivate the design of our E2Efold deep model, which combines a Deep Score Network with a Post-Processing Network based on an unrolled algorithm for solving a constrained optimization problem.

The first part of E2Efold is a Deep Score Network U θ (x) whose output is an L × L symmetric matrix.

Each entry of this matrix, i.e., U θ (x) ij , indicates the score of nucleotides x i and x j being paired.

The x input to the network here is the L × 4 dimensional one-hot embedding.

The specific architecture of U θ is shown in Fig 4.

It mainly consists of

by their exact and relative positions:

where {ψ j } is a set of n feature maps such as sin(·), poly(·), sigmoid(·), etc, and MLP(·) denotes multi-layer perceptions.

Such position embedding idea has been used in natural language modeling such as BERT (Devlin et al., 2018 ), but we adapted for RNA sequence representation; • a stack of Transformer Encoders (Vaswani et al., 2017) which encode the sequence information and the global dependency between nucleotides; • a 2D Convolution layers for outputting the pairwise scores.

With the representation power of neural networks, the hope is that we can learn an informative U θ such that higher scoring entries in U θ (x) correspond well to actual paired bases in RNA structure.

Once the score matrix U θ (x) is computed, a naive approach to use it is to choose an offset term s ∈ R (e.g., s = 0) and let A ij = 1 if U θ (x) ij > s.

However, such entry-wise independent predictions of A ij may result in a matrix A that violates the constraints for a valid RNA secondary structure.

Therefore, a post-processing step is needed to make sure the predicted A is valid.

This step could be carried out separately after U θ is learned.

But such decoupling of base-pair scoring and post-processing for constraints may lead to sub-optimal results, where the errors in these two stages can not be considered together and tuned together.

Instead, we will introduce a Post-Processing Network which can be trained end-to-end together with U θ to enforce the constraints.

The second part of E2Efold is a Post-Processing Network PP φ which is an unrolled and parameterized algorithm for solving a constrained optimization problem.

We first present how we formulate the post-processing step as a constrained optimization problem and the algorithm for solving it.

After that, we show how we use the algorithm as a template to design deep architecture PP φ .

Formulation of constrained optimization.

Given the scores predicted by U θ (x), we define the total score 1 2 i,j (U θ (x) ij − s)A ij as the objective to maximize, where s is an offset term.

Clearly, without structure constraints, the optimal solution is to take A ij = 1 when U θ (x) ij > s. Intuitively, the objective measures the covariation between the entries in the scoring matrix and the A matrix.

With constraints, the exact maximization becomes intractable.

To make it tractable, we consider a convex relaxation of this discrete optimization to a continuous one by allowing A ij ∈ [0, 1].

Consequently, the solution space that we consider to optimize over is A(x) := A ∈ [0, 1] L×L | A is symmetric and satisfies constraints (i)-(iii) in Section 3 .

To further simplify the search space, we define a nonlinear transformation

, where • denotes element-wise multiplication.

Matrix M is defined as M (x) ij := 1 if x i x j ∈ B and also |i − j| ≥ 4, and M (x) ij := 0 otherwise.

From this definition we can see that M (x) encodes both constraint (i) and (ii).

With transformation T , the resulting matrix is non-negative, symmetric, and satisfies constraint (i) and (ii).

Hence, by defining A := T (Â), the solution space is simplified as

Finally, we introduce a 1 penalty term Â 1 := i,j |Â ij | to make A sparse and formulate the post-processing step as: ( ·, · denotes matrix inner product, i.e., sum of entry-wise multiplication)

The advantages of this formulation are that the variablesÂ ij are free variables in R and there are only L inequality constraints A1 ≤ 1.

This system of linear inequalities can be replaced by a set of nonlinear equalities relu(A1 − 1) = 0 so that the constrained problem can be easily transformed into an unconstrained problem by introducing a Lagrange multiplier λ ∈ R

Algorithm for solving it.

We use proximal gradient (derived in Appendix B) for maximization and gradient descent for minimization.

In each iteration,Â and λ are updated alternatively by:

where

soft threshold:

gradient step:

where α, β are step sizes and γ α , γ β are decaying coefficients.

When it converges at T , an approximate solution Round A T = T (Â T ) is obtained.

With this algorithm operated on the learned U θ (x), even if this step is disconnected to the training phase of U θ (x), the final prediction works much better than many other existing methods (as reported in Section 6).

Next, we introduce how to couple this post-processing step with the training of U θ (x) to further improve the performance.

We design a Post-Processing Network, denoted by PP φ , based on the above algorithm.

After it is defined, we can connect it with the deep score network U θ and train them jointly in an end-to-end fashion, so that the training phase of U θ (x) is aware of the post-processing step.

Algorithm 1: Post-Processing Network PP φ (U, M )

Algorithm 2:

The specific computation graph of PP φ is given in Algorithm 1, whose main component is a recurrent cell which we call PPcell φ .

The computation graph is almost the same as the iterative update from Eq. 3 to Eq. 6, except for several modifications:

• (learnable hyperparameters) The hyperparameters including step sizes α, β, decaying rate γ α , γ β , sparsity coefficient ρ and the offset term s are treated as learnable parameters in φ, so that there is no need to tune the hyperparameters by hand but automatically learn them from data instead.

• (fixed # iterations) Instead of running the iterative updates until convergence, PPcell φ is applied recursively for T iterations where T is a manually fixed number.

This is why in Fig 3 the output space of E2Efold is slightly larger than the true solution space.

• (smoothed sign function) Resulted from the gradient of relu(·), the update step in Eq. 4 contains a sign(·) function.

However, to push gradient through PP φ , we require a differentiable update step.

Therefore, we use a smoothed sign function defined as softsign(c) := 1/(1 + exp(−kc)), where k is a temperature.

• (clipÂ) An additional step,Â ← min(Â, 1), is included to make the output A t at each iteration stay in the range [0, 1] L×L .

This is useful for computing the loss over intermediate results

, for which we will explain more in Section 5.

With these modifications, the Post-Processing Network PP φ is a tuning-free and differentiable unrolled algorithm with meaningful intermediate outputs.

Combining it with the deep score network, the final deep model is

5 END-TO-END TRAINING ALGORITHM Given a dataset D containing examples of input-output pairs (x, A * ), the training procedure of E2Efold is similar to standard gradient-based supervised learning.

However, for RNA secondary structure prediction problems, commonly used metrics for evaluating predictive performances are F1 score, precision and recall, which are non-differentiable.

Since F1 = 2TP/(2TP + FP + FN), we define a loss function to mimic the negative of F1 score as:

Assuming that ij A * ij = 0, this loss is well-defined and differentiable on [0, 1] L×L .

Precision and recall losses can be defined in a similar way, but we optimize F1 score in this paper.

It is notable that this F1 loss takes advantages over other differentiable losses including 2 and cross-entropy losses, because there are much more negative samples (i.e. A ij = 0) than positive samples (i.e. A ij = 1).

A hand-tuned weight is needed to balance them while using 2 or crossentropy losses, but F1 loss handles this issue automatically, which can be useful for a number of problems (Wang et al., 2016; .

L×L in each iteration.

This allows us to add auxiliary losses to regularize the intermediate results, guiding it to learn parameters which can generate a smooth solution trajectory.

More specifically, we use an objective that depends on the entire trajectory of optimization:

where

)

and γ ≤ 1 is a discounting factor.

Empirically, we find it very useful to pre-train U θ using logistic regression loss.

Also, it is helpful to add this additional loss to Eq. 9 as a regularization.

We compare E2Efold with the SOTA and also the most commonly used methods in the RNA secondary structure prediction field on two benchmark datasets.

It is revealed from the experimental results that E2Efold achieves 29.7% improvement in terms of F1 score on RNAstralign dataset and it infers the RNA secondary structure as fast as the most efficient algorithm (LinearFold) among existing ones.

An ablation study is also conducted to show the necessity of pushing gradient through the post-processing step.

Experiments On RNAStralign.

We divide RNAStralign dataset into training, testing and validation sets by stratified sampling (see details in Table 7 and Fig 6) , so that each set contains all RNA types.

We compare the performance of E2Efold to six methods including CDPfold, LinearFold, Mfold, RNAstructure (ProbKnot), RNAfold and CONTRAfold.

Both E2Efold and CDPfold are learned from the same training/validation sets.

For other methods, we directly use the provided packages or web-servers to generate predicted structures.

We evaluate the F1 score, Precision and Recall for each sequence in the test set.

Averaged values are reported in Table 2 .

As suggested by Mathews (2019), for a base pair (i, j), the following predictions are also considered as correct:

, so we also reported the metrics when one-position shift is allowed.

As shown in Table 2 , traditional methods can achieve a F1 score ranging from 0.433 to 0.624, which is consistent with the performance reported with their original papers.

The two learning-based methods, CONTRAfold and CDPfold, can outperform classical methods with reasonable margin on some criteria.

E2Efold, on the other hand, significantly outperforms all previous methods across all criteria, with at least 20% improvement.

Notice that, for almost all the other methods, the recall is usually higher than precision, while for E2Efold, the precision is higher than recall.

That can be the result of incorporating constraints during neural network training.

Fig 5 shows the distributions of F1 scores for each method.

It suggests that E2Efold has consistently good performance.

To estimate the performance of E2Efold on long sequences, we also compute the F1 scores weighted by the length of sequences, such that the results are more dominated by longer sequences.

Detailed results are given in Appendix D.3.

Test On ArchiveII Without Re-training.

To mimic the real world scenario where the users want to predict newly discovered RNA's structures which may have a distribution different from the training dataset, we directly test the model learned from RNAStralign training set on the ArchiveII dataset, without re-training the model.

To make the comparison fair, we exclude sequences that are overlapped with the RNAStralign dataset.

We then test the model on sequences in ArchiveII that have overlapping RNA types (5SrRNA, 16SrRNA, etc) with the RNAStralign dataset.

Results are shown in Table 3 .

It is understandable that the performances of classical methods which are not learning- based are consistent with that on RNAStralign.

The performance of E2Efold, though is not as good as that on RNAStralign, is still better than all the other methods across different evaluation criteria.

In addition, since the original ArchiveII dataset contains domain sequences (subsequences), we remove the domains and report the results in Appendix D.4, which are similar to results in Table 3 .

Inference Time Comparison.

We record the running time of all algorithms for predicting RNA secondary structures on the RNAStralign test set, which is summarized in Table 4 .

LinearFold is the most efficient among baselines because it uses beam pruning heuristic to accelerate DP.

CDPfold, which achieves higher F1 score than other baselines, however, is extremely slow due to its DP post-processing step.

Since we use a gradient-based algorithm which is simple to design the PostProcessing Network, E2Efold is fast.

On GPU, E2Efold has similar inference time as LinearFold.

Pseudoknot Prediction.

Even though E2Efold does not exclude pseudoknots, it is not sure whether it actually generates pseudoknotted structures.

Therefore, we pick all sequences containing pseudoknots and compute the averaged F1 score only on this set.

Besides, we count the number of pseudoknotted sequences that are predicted as pseudoknotted and report this count as true positive (TP).

Similarly we report TN, FP and FN in Table 5 along with the F1 score.

Most tools exclude pseudoknots while RNAstructure is the most famous one that can predict pseudoknots, so we choose it for comparison.

RNAstructure CONTRAfold true structure RNAstructure CONTRAfold E2Efold true structure true structure E2Efold true structure E2Efold

Visualization.

We visualize predicted structures of three RNA sequences in the main text.

More examples are provided in appendix (Fig 8 to 14) .

In these figures, purple lines indicate edges of pseudoknotted elements.

Although CDPfold has higher F1 score than other baselines, its predictions are visually far from the ground-truth.

Instead, RNAstructure and CONTRAfold produce comparatively more reasonable visualizations among all baselines, so we compare with them.

These two methods can capture a rough sketch of the structure, but not good enough.

For most cases, E2Efold produces structures most similar to the ground-truths.

Moreover, it works surprisingly well for some RNA sequences that are long and very difficult to predict.

Ablation Study.

To exam whether integrating the two stages by pushing gradient through the post-process is necessary for performance of E2Efold, we conduct an ablation study (Table 6 ).

We test the performance when the post-processing step is disconnected with the training of Deep Score Network U θ .

We apply the post-processing step (i.e., for solving augmented Lagrangian) after U θ is learned (thus the notation "U θ + PP" in Table 6 ).

Although "U θ + PP" performs decently well, with constraints incorporated into training, E2Efold still has significant advantages over it.

Discussion.

To better estimate the performance of E2Efold on different RNA types, we include the per-family F1 scores in Appendix D.5.

E2Efold performs significantly better than other methods in 16S rRNA, tRNA, 5S RNA, tmRNA, and telomerase.

These results are from a single model.

In the future, we can view it as multi-task learning and further improve the performance by learning multiple models for different RNA families and learning an additional classifier to predict which model to use for the input sequence.

We propose a novel DL model, E2Efold, for RNA secondary structure prediction, which incorporates hard constraints in its architecture design.

Comprehensive experiments are conducted to show the superior performance of E2Efold, no matter on quantitative criteria, running time, or visualization.

Further studies need to be conducted to deal with the RNA types with less samples.

Finally, we believe the idea of unrolling constrained programming and pushing gradient through post-processing can be generic and useful for other constrained structured prediction problems.

Here we explain the difference between our approach and other works on unrolling optimization problems.

First, our view of incorporating constraints to reduce output space and to reduce sample complexity is novel.

Previous works (Hershey et al., 2014; Belanger et al., 2017; Ingraham et al., 2018) did not discuss these aspects.

The most related work which also integrates constraints is OptNet (Amos & Kolter, 2017) , but its very expensive and can not scale to the RNA problem.

Therefore, our proposed approach is a simple and effective one.

Second, compared to (Chen et al., 2018; Shrivastava et al., 2019) , our approach has a different purpose of using the algorithm.

Their goal is to learn a better algorithm, so they commonly make their architecture more flexible than the original algorithm for the room of improvement.

However, we aim at enforcing constraints.

To ensure that constraints are nicely incorporated, we keep the original structure of the algorithm and only make the hyperparameters learnable.

Finally, although all works consider end-to-end training, none of them can directly optimize the F1 score.

We proposed a differentiable loss function to mimic the F1 score/precision/recall, which is effective and also very useful when negative samples are much fewer than positive samples (or the inverse).

The maximization step in Eq. 2 can be written as the following minimization:

Consider the quadratic approximation of −f (Â) centered atÂ t :

and rewrite the optimization in Eq. 10 as

Next, we define proximal mapping as a function depending on α as follows:

Since we always useÂ •Â instead ofÂ in our problem, we can take the absolute value |prox α (Ȧ t+1 )| = relu(|Ȧ t+1 | − αρ) without loss of generality.

Therefore, the proximal gradient step isȦ

A t+1 ← relu(|Ȧ t+1 | − αρ) (correspond to Eq. 5).

More specifically, in the main text, we write ∂f ∂Ât

The last equation holds sinceÂ t will remain symmetric in our algorithm if the initialÂ 0 is symmetric.

Moreover, in the main text, α is replaced by α · γ t α .

We used Pytorch to implement the whole package of E2Efold.

Deep Score Network.

In the deep score network, we used a hyper-parameter, d, which was set as 10 in the final model, to control the model capacity.

In the transformer encoder layers, we set the number of heads as 2, the dimension of the feed-forward network as 2048, the dropout rate as 0.1.

As for the position encoding, we used 58 base functions to form the position feature map, which goes through a 3-layer fully-connected neural network (the number of hidden neurons is 5 * d) to generate the final position embedding, whose dimension is L by d. In the final output layer, the pairwise concatenation is carried out in the following way: Let X ∈ R L×3d be the input to the final output layers in Figure 4 (which is the concatenation of the sequence embedding and position embedding).

The pairwise concatenation results in a tensor Y ∈ R L×L×6d defined as

where

, and X(j, :) ∈ R 3d .

In the 2D convolution layers, the the channel of the feature map gradually change from 6 * d to d , and finally to 1.

We set the kernel size as 1 to translate the feature map into the final score matrix.

Each 2D convolution layer is followed by a batch normalization layer.

We used ReLU as the activation function within the whole score network.

Post-Processing Network.

In the PP network, we initialized w as 1, s as log(9), α as 0.01, β as 0.1, γ α as 0.99, γ β as 0.99, and ρ as 1.

We set T as 20.

Training details.

During training, we first pre-trained a deep score network and then fine-tuned the score network and the PP network together.

To pre-train the score network, we used binary crossentropy loss and Adam optimizer.

Since, in the contact map, most entries are 0, we used weighted loss and set the positive sample weight as 300.

The batch size was set to fully use the GPU memory, which was 20 for the Titan Xp card.

We pre-train the score network for 100 epochs.

As for the fine-tuning, we used binary cross-entropy loss for the score network and F1 loss for the PP network and summed up these two losses as the final loss.

The user can also choose to only use the F1 loss or use another coefficient to weight the loss estimated on the score network U θ .

Due to the limitation of the GPU memory, we set the batch size as 8.

However, we updated the model's parameters every 30 steps to stabilize the training process.

We fine-tuned the whole model for 20 epochs.

Also, since the data for different RNA families are imbalanced, we up-sampled the data in the small RNA families based on their size.

For the training of the score network U θ in the ablation study, it is exactly the same as the training of the above mentioned process.

Except that during the fine-tune process, there is the unrolled number of iterations is set to be 0.

D.1 DATASET STATISTICS Figure 6 : The RNAStralign length distribution.

To compare the differences among these data distributions, we can test the following hypothesis:

The approach that we adopted is the permutation test on the unbiased empirical Maximum Mean Discrepancy (MMD) estimator:

where

contains M i.i.d.

samples from a distribution P 2 , and k(·, ·) is a string kernel.

Since we conduct stratified sampling to split the training and testing dataset, when we perform permutation test, we use stratified re-sampling as well (for both Hypothese (a) and (b)).

The result of the permutation test (permuted 1000 times) is reported in Figure 7 .

The result shows (a) Hypothesis P(RNAStr train ) = P(RNAStr test ) can be accepted with significance level 0.1.

(b) Hypothesis P(RNAStr train ) = P(ArchiveII) is rejected since the p-value is 0.

Therefore, the data distribution in ArchiveII is very different from the RNAStralign training set.

A good performance on ArchiveII shows a significant generalization power of E2Efold.

For long sequences, E2Efold still performs better than other methods.

We compute F1 scores weighted by the length of sequences (Table 8 ), such that the results are more dominated by longer sequences.

The third row reports how much F1 score drops after reweighting.

Since domain sequence (subsequences) in ArchiveII are explicitly labeled, we filter them out in ArchiveII and recompute the F1 scores ( Table 9 ).

The results do not change too much before or after filtering out subsequences.

To balance the performance among different families, during the training phase we conducted weighted sampling of the data based on their family size.

With weighted sampling, the overall F1 score (S) is 0.83, which is the same as when we did equal-weighted sampling.

The per-family results are shown in Table 10 .

RNAstructure CDPfold true structure LinearFold Mfold CONTRAfold RNAfold

<|TLDR|>

@highlight

A DL model for RNA secondary structure prediction, which uses an unrolled algorithm in the architecture to enforce constraints.