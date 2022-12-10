In representation learning (RL), how to make the learned representations easy to interpret and less overfitted to training data are two important but challenging issues.

To address these problems, we study a new type of regularization approach that encourages the supports of weight vectors in RL models to have small overlap, by simultaneously promoting near-orthogonality among vectors and sparsity of each vector.

We apply the proposed regularizer to two models: neural networks (NNs) and sparse coding (SC), and develop an efficient ADMM-based algorithm for regularized SC.

Experiments on various datasets demonstrate that weight vectors learned under our regularizer are more interpretable and have better generalization performance.

In representation learning (RL), two critical issues need to be considered.

First, how to make the learned representations more interpretable?

Interpretability is a must in many applications.

For instance, in a clinical setting, when applying deep learning (DL) and machine learning (ML) models to learn representations for patients and use the representations to assist clinical decision-making, we need to explain the representations to physicians such that the decision-making process is transparent, rather than being black-box.

Second, how to avoid overfitting?

It is often the case that the learned representations yield good performance on the training data, but perform less well on the testing data.

How to improve the generalization performance on previously unseen data is important.

In this paper, we make an attempt towards addressing these two issues, via a unified approach.

DL/ML models designed for representation learning are typically parameterized with a collection of weight vectors, each aiming at capturing a certain latent feature.

For example, neural networks are equipped with multiple layers of hidden units where each unit is parameterized by a weight vector.

In another representation learning model -sparse coding BID74 , a dictionary of basis vectors are utilized to reconstruct the data.

In the interpretation of RL models, a major part is to interpret the learned weight vectors.

Typically, elements of a weight vector have one-to-one correspondence with observed features and a weight vector is oftentimes interpreted by examining the top observed-features that correspond to the largest weights in this vector.

For instance, when applying SC to reconstruct documents that are represented with bag-of-words feature vectors, each dimension of a basis vector corresponds to one word in the vocabulary.

To visualize/interpret a basis vector, one can inspect the words corresponding to the large values in this vector.

To achieve better interpretability, various constraints have been imposed on the weight vectors.

Some notable ones are: (1) Sparsity BID80 -which encourages most weights to be zero.

Observed features that have zeros weights are considered to be irrelevant and one can focus on interpreting a few non-zero weights.

(2) Diversity BID81 -which encourages different weight vectors to be mutually "different" (e.g., having larger angles ).

By doing this, the redundancy among weight vectors is reduced and cognitively one can map each weight vector to a physical concept in a more unambiguous way.

(3) Non-negativeness BID66 -which encourages the weights to be nonnegative since in certain scenarios (e.g., bag of words representation of documents), it is difficult to make sense of negative weights.

In this paper, we propose a new perspective of interpretability: less-overlapness, which encourages the weight vectors to have small overlap in supports 1 .

By doing this, each weight vector is anchored on a unique subset of observed features without being redundant with other vectors, which greatly facilitates interpretation.

For example, if topic models BID47 are learned in such a way, each topic will be characterized by a few representative words and the representative words of different topics are different.

Such topics are more amenable for interpretation.

Besides improving interpretability, less-overlapness helps alleviate overfitting.

It imposes a structural constraint over the weight vectors, thus can effectively shrink the complexity of the function class induced by the RL models and improve the generalization performance on unseen data.

To encourage less-overlapness, we propose a regularizer that simultaneously encourages different weight vectors to be close to being orthogonal and each vector to be sparse, which jointly encourage vectors' supports to have small overlap.

The major contributions of this work include:• We propose a new type of regularization approach which encourages less-overlapness, for the sake of improving interpretability and reducing overfitting.• We apply the proposed regularizer to two models: neural networks and sparse coding (SC), and derive an efficient ADMM-based algorithm for the regularized SC problem.• In experiments, we demonstrate the empirical effectiveness of this regularizer.

In this section, we propose a nonoverlapness-promoting regularizer and apply it to two models.

We assume the model is parameterized by m vectors W = {w i } m i=1 .

For a vector w, its support s(w) is defined as {i|w i = 0} -the indices of nonzero entries in w. We first define a scoreõ(w i , w j ) to measure the overlap between two vectors: DISPLAYFORM0 which is the Jaccard index of their supports.

The smallerõ(w i , w j ) is, the less overlapped the two vectors are.

For m vector, the overlap score is defined as the sum of pairwise scores DISPLAYFORM1 This score function is not smooth, which will result in great difficulty for optimization if used as a regularizer.

Instead, we propose a smooth function that is motivated fromõ(w i , w j ) and can achieve a similar effect as o(W).

The basic idea is: to encourage small overlap, we can encourage (1) each vector has a small number of non-zero entries and (2) the intersection of supports among vectors is small.

To realize (1), we use an L1 regularizer to encourage the vectors to be sparse.

To realize (2), we encourage the vectors to be close to being orthogonal.

For two sparse vectors, if they are close to orthogonal, then their supports are landed on different positions.

As a result, the intersection of supports is small.

We follow the method proposed by to promote orthogonality.

To encourage two vectors w i and w j to be close to being orthogonal, one can make their 2 norm w i 2 , w j 2 close to one and their inner product w i w j close to zero.

Based on this, one can promote orthogonality among a set of vectors by encouraging the Gram matrix G (G ij = w i w j ) of these vectors to be close to an identity matrix I. Since w i w j and zero are off the diagonal of G and I respectively, and w i 2 2 and one are on the diagonal of G and I respectively, encouraging G close to I essentially makes w i w j close to zero and w i 2 close to one.

As a result, w i and w j are encouraged to be close to being orthogonal.

In , one way proposed to measure the "closeness" between two matrices is to use the log-determinant divergence (LDD) BID64 .

The LDD between two m × m positive definite matrices X and Y is defined as D(X, Y) = tr(XY −1 ) − log det(XY −1 ) − m where tr(·) denotes matrix trace.

The closeness between G and I can be achieved by encouraging their LDD D(G, I) = tr(G) − log det(G) − m to be small.

Combining the orthogonality-promoting LDD regularizer with the sparsity-promoting L1 regularizer together, we obtain the following LDD-L1 regularizer DISPLAYFORM2 where γ is a tradeoff parameter between these two regularizers.

As verified in experiments, this regularizer can effectively promote non-overlapness.

The formal analysis of the relationship between Eq. (3) and Eq.(2) will be left for future study.

It is worth noting that either L1 or LDD alone is not sufficient to reduce overlap.

As illustrated in FIG0 (a) where only L1 is applied, though the two vectors are sparse, their supports are completely overlapped.

In FIG0 (b) where the LDD regularizer is applied, though the two vectors are very close to orthogonal, their supports are completely overlapped since they are dense.

In FIG0 (c) where the LDD-L1 regularizer is used, the two vectors are sparse and are close to being orthogonal.

As a result, their supports are not overlapped.

In this section, we apply the LDD-L1 regularizer to two models.

Neural Networks In a neural network (NN) with L hidden layers, each hidden layer l is equipped with m (l) units and each unit i is connected with all units in layer l − 1.

Hidden unit i at layer l is parameterized by a weight vector w (l)i .

These hidden units aim at capturing latent features underlying data.

For DISPLAYFORM0 in each layer l, we apply the LDD-L1 regularizer to encourage them to have small overlap.

An LDD-L1 regularized NN problem (LDD-L1-NN) can be defined in the following way: DISPLAYFORM1 where DISPLAYFORM2 ) is the objective function of this NN.Sparse Coding Given n data samples X ∈ R d×n where d is the feature dimension, we aim to use a dictionary of basis vectors W ∈ R d×m to reconstruct X, where m is the number of basis vectors.

Each data sample x is reconstructed by taking a sparse linear combination of the basis vectors x ≈ m j=1 α j w j , where {α j } m j=1 are the linear coefficients and most of them are zero.

The reconstruction error is measured using the squared L2 norm x− m j=1 α j w j 2 2 .

To achieve sparsity among the codes, L1 regularization is utilized: m j=1 |α j | 1 .

To avoid the degenerated case where most coefficients are zero and the basis vectors are of large magnitude, L2 regularization is applied to the basis vectors: w j 2 2 .

We apply the LDD-L1 regularizer to encourage the supports of basis vectors to have small overlap.

Putting these pieces together, we obtain the LDD-L1 regularized SC (LDD-L1-SC) problem FORMULA0 until convergence of the problem defined in Eq. FORMULA11 until convergence of the problem defined in Eq. FORMULA6 3 ALGORITHM For LDD-L1-NNs, a simple subgradient descent algorithm is applied to learn the weight parameters.

For LDD-L1-SC, we solve it by alternating between A and W: (1) updating A with W fixed; (2) updating W with A fixed.

These two steps alternate until convergence.

With W fixed, the subproblem defined over A is DISPLAYFORM3 DISPLAYFORM4 which can be decomposed into n Lasso problems: DISPLAYFORM5 where a i is the coefficient vector of the i-th sample.

Lasso can be solved by many algorithms, such as proximal gradient descent (PGD) BID75 .

Fixing A, the sub-problem defined over W is: DISPLAYFORM6 We solve this problem using an ADMM-based algorithm.

First, we write the problem into an equivalent form DISPLAYFORM7 Then we write down the augmented Lagrangian function DISPLAYFORM8 We minimize this Lagrangian function by alternating among W, U and W. DISPLAYFORM9 which is a Lasso problem and can be solved using PGD BID75 .Update U The update equation of U is simple.

DISPLAYFORM10 The subproblem defined on W is DISPLAYFORM11 which can be solved using a coordinate descent algorithm.

The derivation is given in the next subsection.

In each iteration of the CD algorithm, one basis vector is chosen for update while the others are fixed.

Without loss of generality, we assume it is w 1 .

The sub-problem defined over w 1 is min w1 DISPLAYFORM0 To obtain the optimal solution, we take the derivative of the objective function and set it to zero.

First, we discuss how to compute the derivative of logdet(W W) w.r.t w 1 .

According to the chain rule, we have DISPLAYFORM1 where (W W) DISPLAYFORM2 :,1 denotes the first column of (W W) DISPLAYFORM3 According to the inverse of block matrix DISPLAYFORM4 where DISPLAYFORM5 where DISPLAYFORM6 To this end, we obtain the full gradient of the objective function in Eq. FORMULA0 : DISPLAYFORM7 Setting the gradient to zero, we get DISPLAYFORM8 The matrix A = W ¬1 (W ¬1 W ¬1 ) −1 W ¬1 is idempotent, i.e., AA = A, and its rank is m − 1.

According to the property of idempotent matrix, the first m − 1 eigenvalues of A equal to one and the rest equal to zero.

Thereafter, the first m − 1 eigenvalues of M = I − A equal to zero and the rest equal to one.

Based on this property, Eq.(24) can be simplified as DISPLAYFORM9 After simplification, it is a quadratic function where γ has a closed form solution.

Then we plug the solution of γ into Eq.(23) to get the solution of w 1 .

In these section, we present experimental results.

The studies were performed on three models: sparse coding (SC) for document modeling, long short-term memory (LSTM) (Hochreiter & Schmidhuber, 1997) network for language modeling and convolutional neural network (CNN) BID63 for image classification.

We used four datasets.

The SC experiments were conducted on two text datasets: 20-Newsgroups 2 (20-News) and Reuters Corpus 3 Volume 1 (RCV1).

The 20-News dataset contains newsgroup documents belonging to 20 categories, where 11314, 3766 and 3766 documents were used for training, validation and testing respectively.

The original RCV1 dataset contains documents belonging to 103 categories.

Following BID48 , we chose the largest 4 categories which contain 9625 documents, to carry out the study.

The number of training, validation and testing documents are 5775, 1925, 1925 respectively.

For both datasets, stopwords were removed and all words were changed into lower-case.

Top 1000 words with the highest document frequency were selected to form the vocabulary.

We used tf-idf to represent documents and the feature vector of each document is normalized to have unit L2 norm.

The LSTM experiments were conducted on the Penn Treebank (PTB) dataset BID69 , which consists of 923K training, 73K validation, and 82K test words.

Following (Mikolov et al.) , top 10K words with highest frequency were selected to form the vocabulary.

All other words are replaced with a special token UNK.The CNN experiments were performed on the CIFAR-10 dataset 4 .

It consists of 32x32 color images belonging to 10 categories, where 50,000 images were used for training and 10,000 for testing.

5000 training images were used as the validation set for hyperparameter tuning.

We augmented the dataset by first zero-padding the images with 4 pixels on each side, then randomly cropping the padded images to reproduce 32x32 images.

First of all, we verify whether the LDD-L1 regularizer is able to promote non-overlapness.

The study is performed on the SC model and the 20-News dataset.

The number of basis vectors was set to 50.

For 5 choices of the regularization parameter of LDD-L1: {10 −4 , 10 −3 , · · · , 1}, we ran the LDD-L1-SC model until convergence and measured the overlap score (defined in Eq.2) of the basis vectors.

The tradeoff parameter γ inside LDD-L1 is set to 1.

FIG1 shows that the overlap score consistently decreases as the regularization parameter of LDD-L1 increases, which implies that LDD-L1 can effectively encourage non-overlapness.

As a comparison, we replaced LDD-L1 with LDD-only and L1-only, and measured the overlap scores.

As can be seen, for LDD-only, the overlap score remains to be 1 when the regularization parameter increases, which indicates that LDD alone is not able to reduce overlap.

This is because under LDD-only, the vectors remain dense, which renders their supports to be completely overlapped.

Under the same regularization parameter, LDD-L1 achieves lower overlap score than L1, which suggests that LDD-L1 is more effective in promoting non-overlapness.

Given that γ -the tradeoff parameter associated with the L1 norm in Representative Words 1 crime, guns 2 faith, trust 3 worked, manager 4 weapons, citizens 5 board, uiuc 6 application, performance, ideas 7 service, quality 8 bible, moral 9 christ, jews, land, faq Table 1 : Representative words of 9 exemplar basis vectors LDD-L1 -is set to 1, the same regularization parameter λ imposes the same level of sparsity for both LDD-L1 and L1-only.

Since LDD-L1 encourages the vectors to be mutually orthogonal, the intersection between vectors' supports is small, which consequently results in small overlap.

This is not the case for L1-only, which hence is less effective in reducing overlap.

In this section, we examine whether the weight vectors learned under LDD-L1 regularization are more interpretable, using SC as a study case.

For each basis vector w learned by LDD-L1-SC on the 20-News dataset, we use the words (referred to as representative words) that correspond to the supports of w to interpret w. Table 1 shows the representative words of 9 exemplar vectors.

By analyzing the representative words, we can see vector 1-9 represent the following semantics respectively: crime, faith, job, war, university, research, service, religion and Jews.

The representative words of these vectors have no overlap.

As a result, it is easy to associate each vector with a unique concept, in other words, easy to interpret.

FIG2 visualizes the learned vectors where the black dots denote vectors' supports.

As can be seen, the supports of different basis vectors are landed over different words and their overlap is very small.

In this section, we verify whether LDD-L1 is able to reduce overfitting.

The studies were performed on SC, LSTM and CNN.

In each experiment, the hyperparameters were tuned on the validation set.

Sparse Coding For 20-News, the number of basis vectors in LDD-L1-SC is set to 50.

λ 1 , λ 2 , λ 3 and λ 4 are set to 1, 1, 0.1 and 0.001 respectively.

For RCV1, the number of basis vectors is set to 200.

λ 1 , λ 2 , λ 3 and λ 4 are set to 0.01, 1, 1 and 1 respectively.

We compared LDD-L1 with LDD-only and L1-only.

To evaluate the model performance quantitatively, we applied the dictionary learned on the training data to infer the linear coefficients (A in Eq.4) of test documents, then performed k-nearest neighbors (KNN) classification on A. TAB2 shows the classification accuracy on test sets of 20-News and RCV1 and the gap 5 between the accuracy on training and test sets.

Without regularization, SC achieves a test accuracy of 0.592 on 20-News, which is lower than the training accuracy by 0.119.

This suggests that an overfitting to training data occurs.

With LDD-L1 regularization, the test accuracy is improved to 0.612 and the gap between training and test accuracy is reduced to 0.099, demonstrating the ability of LDD-L1 in alleviating overfitting.

Though LDD alone and L1 alone improve test accuracy and reduce train/test gap, they perform less well than LDD-L1, which indicates that for overfitting reduction, encouraging non-overlapness is more effective than solely promoting orthogonality or solely promoting sparsity.

Similar observations are made on the RCV1 dataset.

Interestingly, the test accuracy achieved by LDD-L1-SC on RCV1 is even better than the training accuracy.

The LSTM network architecture follows the word language model (PytorchTM) provided in Pytorch 6 .

The number of hidden layers is set to 2.

The embedding size is 1500.

The size of hidden state is 1500.

The word embedding and softmax weights are tied.

The number of training epochs is 40.

Dropout with 0.65 is used.

The initial learning rate is 20.

Gradient clipping threshold is 0.25.

The size of mini-batch is 20.

In LSTM training, the network is unrolled for 35 iterations.

Perplexity is used for evaluating language modeling performance (lower is better).

The weight parameters are initialized uniformly between [-0.1, 0.1].

The bias parameters are initialized as 0.

We compare with the following regularizers: (1) L1 regularizer; (2) orthogonality-promoting regularizers based on cosine similarity (CS) , incoherence (IC) , mutual angle (MA) , decorrelation (DC) , angular constraint (AC) and LDD .

TAB3 shows the perplexity on the PTB test set.

Without regularization, PytorchLM achieves a perplexity of 72.3.

With LDD-L1 regularization, the perplexity is significantly reduced to 71.1.

This shows that LDD-L1 can effectively improve generalization performance.

Compared with the sparsity-promoting L1 regularizer and orthogonality-promoting regularizers, LDD-L1 -which promotes non-overlapness by simultaneously promoting sparsity and orthogonality -achieves lower perplexity.

For the convenience of readers, we also list the perplexity achieved by other state of the art deep learning models.

The LDD-L1 regularizer can be applied to these models as well to potentially boost their performance.

BID88 .

The depth and width are set to 28 and 10 respectively.

The networks are trained using SGD, where the epoch number is 200, the learning rate is set to 0.1 initially and is dropped by 0.2 at 60, 120 and 160 epochs, the minibatch size is 128 and the Nesterov momentum is 0.9.

The dropout probability is 0.3 and the L2 weight decay is 0.0005.

Model performance is measured using error rate, which is the median of 5 runs.

We compared with (1) L1 regularizer; (2) orthogonality-promoting regularizers including CS, IC, MA, DC, AC, LDD and one based on locally constrained decorrelation (LCD) .

Table 4 shows classification errors on CIFAR-10 test set.

Compared with the unregularized WideResNet which achieves an error rate of 3.89%, the proposed LDD-L1 regularizer greatly reduces the error to 3.60%.

LDD-L1 outperforms the L1 regularizer and orthogonality-promoting regularizers, demonstrating that encouraging non-overlapness is more effective than encouraging sparsity alone or orthogonality alone in reducing overfitting.

The error rates achieved by other state of the art methods are also listed.

The interpretation of representation learning models has been widely studied.

BID50 develop a two-level neural attention model that detects influential variables in a reverse time order and use these variables to interpret predictions.

Lipton (2016) discuss a taxonomy of both the desiderata and methods in interpretability research.

BID62 propose to use influence functions to trace a model's prediction back to its training data and identify training examples that are most relevant to a prediction.

BID53 integrate topics extracted from human descriptions into neural networks via an interpretive loss and then use a prediction-difference maximization algo- Table 4 : Classification error (%) on CIFAR-10 test set rithm to interpret the learned features of each neuron.

Our method is orthogonal to these existing approaches and can be potentially used with them together to further improve interpretability.

In this paper, we propose a new type of regularization approach that encourages the weight vectors to have less-overlapped supports.

The proposed LDD-L1 regularizer simultaneously encourages the weight vectors to be sparse and close to being orthogonal, which jointly produces the effects of less overlap.

We apply this regularizer to two models: neural networks and sparse coding (SC), and derive an efficient ADMM-based algorithm for solving the regularized SC problem.

Experiments on various datasets demonstrate the effectiveness of this regularizer in alleviating overfitting and improving interpretability.

<|TLDR|>

@highlight

We propose a new type of regularization approach that encourages non-overlapness in representation learning, for the sake of improving interpretability and reducing overfitting.

@highlight

The paper introduces a matrix regularizer to simultaneously induce both sparsity and approximate orthogonality.

@highlight

The paper studies a regularization method to promote sparsity and reduce the overlap among the supports of the weight vectors in the learned representations to enhance interpretability and avoid overfitting

@highlight

The paper proposed a new regularization approach that simultaneously encourages the weight vectors (W) to be sparse and orthogonal to each other.