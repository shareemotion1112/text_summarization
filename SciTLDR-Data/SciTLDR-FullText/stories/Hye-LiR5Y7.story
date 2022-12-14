We present SOSELETO (SOurce SELEction for Target Optimization), a new method for exploiting a source dataset to solve a classification problem on a target dataset.

SOSELETO is based on the following simple intuition: some source examples are more informative than others for the target problem.

To capture this intuition, source samples are each given weights; these weights are solved for jointly with the source and target classification problems via a bilevel optimization scheme.

The target therefore gets to choose the source samples which are most informative for its own classification task.

Furthermore, the bilevel nature of the optimization acts as a kind of regularization on the target, mitigating overfitting.

SOSELETO may be applied to both classic transfer learning, as well as the problem of training on datasets with noisy labels; we show state of the art results on both of these problems.

Deep learning has made possible many remarkable successes, leading to state of the art algorithms in computer vision, speech and audio, and natural language processing.

A key ingredient in this success has been the availability of large datasets.

While such datasets are common in certain settings, in other scenarios this is not true.

Examples of the latter include "specialist" scenarios, for instance a dataset which is entirely composed of different species of tree; and medical imaging, in which datasets on the order of hundreds to a thousand are common.

A natural question is then how one may apply the techniques of deep learning within these relatively data-poor regimes.

A standard approach involves the concept of transfer learning: one uses knowledge gleaned from the source (data-rich regime), and transfers it over to the target (data-poor regime).

One of the most common versions of this approach involves a two-stage technique.

In the first stage, a network is trained on the source classification task; in the second stage, this network is adapted to the target classification task.

There are two variants for this second stage.

In feature extraction (e.g. ), only the parameters of the last layer (i.e. the classifier) are allowed to adapt to the target classification task; whereas in fine-tuning (e.g. BID12 ), the parameters of all of the network layers (i.e. both the features/representation and the classifier) are allowed to adapt.

The idea is that by pre-training the network on the source data, a useful feature representation may be learned, which may then be recycled -either partially or completely -for the target regime.

This two-stage approach has been quite popular, and works reasonably well on a variety of applications.

Despite this success, we claim that the two-stage approach misses an essential insight: some source examples are more informative than others for the target classification problem.

For example, if the source is a large set of natural images and the target consists exclusively of cars, then we might expect that source images of cars, trucks, and motorcycles might be more relevant for the target task than, say, spoons.

However, this example is merely illustrative; in practice, the source and target datasets may have no overlapping classes at all.

As a result, we don't know a priori which source examples will be important.

Thus, we propose to learn this source filtering as part of an end-to-end training process.

The resulting algorithm is SOSELETO: SOurce SELEction for Target Optimization.

Each training sample in the source dataset is given a weight, corresponding to how important it is.

The shared source/target representation is then optimized by means of a bilevel optimization.

In the interior level, the source minimizes its classification loss with respect to the representation parameters, for fixed values of the sample weights.

In the exterior level, the target minimizes its classification loss with respect to both the source sample weights and its own classification layer.

The sample weights implicitly control the representation through the interior level.

The target therefore gets to choose the source samples which are most informative for its own classification task.

Furthermore, the bilevel nature of the optimization acts as a kind of regularization on the target, mitigating overfitting, as the target does not directly control the representation parameters.

Finally, note that the entire processtraining of the shared representation, target classifier, and source weights -happens simultaneously.

We pause here to note that the general philosophy behind SOSELETO is related to the literature on instance reweighting for domain adaptation, see for example BID32 .

However, there is a crucial difference between SOSELETO and this literature, which is related to the difference between domain adaptation and more general transfer learning.

Domain adaptation is concerned with the situation in which there is either full overlap between the source and target label sets; or in some more recent work BID43 , partial but significant overlap.

Transfer learning, by contrast, refers to the more general situation in which there may be zero overlap between label sets, or possibly very minimal overlap.

(For example, if the source consists of natural images and the target of medical images.)

The instance reweighting literature is concerned with domain adaptation; the techniques are therefore relevant to the case in which source and target have the same labels.

SOSELETO is quite different: it makes no such assumptions, and is therefore a more general approach which can be applied to both "pure" transfer learning, in which there is no overlap between source and target label sets, as well as domain adaptation. (Note also a further distinction with domain adaptation: the target is often -though not always -taken to be unlabelled in domain adaptation.

This is not the case for our setting of transfer learning.)Above, we have illustrated how SOSELETO may be applied to the problem of transfer learning.

However, the same algorithm can be applied to the problem of training with noisy labels.

Concretely, we assume that there is a large noisy dataset, as well as a much smaller clean dataset; the latter can be constructed cheaply through careful hand-labelling, given its small size.

Then if we take the source to be the large noisy dataset, and the target to the small clean dataset, SOSELETO can be applied to the problem.

The algorithm will assign high weights to samples with correct labels and low weights to those with incorrect labels, thereby implicitly denoising the source, and allowing for an accurate classifier to be trained.

The remainder of the paper is organized as follows.

Section 2 presents related work.

Section 3 presents the SOSELETO algorithm, deriving descent equations as well as convergence properties of the bilevel optimization.

Section 4 presents results of experiments on both transfer learning as well as training with noisy labels.

Section 5 concludes.

Transfer learning As described in Section 1, the most common techniques for transfer learning are feature extraction and fine-tuning, see for example and BID12 , respectively.

An older survey of transfer learning techniques may be found in BID25 .

Domain adaptation BID28 is concerned with transferring knowledge when the source and target classes are the same.

Earlier techniques aligned the source and target via matching of feature space statistics ; BID19 ; subsequent work used adversarial methods to improve the domain adaptation performance BID10 ; BID36 ; .In this paper, we are more interested in transfer learning where the source and target classes are different.

A series of recent papers BID20 ; BID26 BID4 b) address domain adaptation that is closer to our setting.

In particular, BID5 examines "partial transfer learning", the case in which there is partial overlap between source and target classes (particularly when the target classes are a subset of the source).

This setting is also dealt with in BID3 .

BID11 examine the scenario where the source and target classes are completely different.

Similar to SOSELETO, they propose selecting a portion of the source dataset.

However, the selection is not performed in an end-to-end fashion, as in SOSELETO; rather, selection is performed prior to training, by finding source examples which are similar to the target dataset, where similarity is measured by using filter bank descriptors.

Another recent work of interest is , which focuses on a slightly different scenario: the target consists of a very small number of labelled examples (i.e. the few-shot regime), but a very large number of unlabelled examples.

Training is achieved via an adversarial loss to align the source and the target representations, and a special entropy-based loss for the unlabelled part of the data.

Instance reweighting for domain adaptation is a well studied technique, demonstrated e.g. in Covariate Shift methods Shimodaira FORMULA6 ; BID31 .

In these works, the source and target label spaces are the same.

We, however, allow for different -even entirely nonoverlapping -classes in the source and target.

Crucially, we do not make assumptions on the similarity of the distributions nor do we explicitly optimize for it.

The same distinction applies for the recent work of , and for the partial overlap assumption of BID43 .

In addition, these two works propose an unsupervised approach, whereas our proposed method is completely supervised.

Covariate shift determines the weighting for an instance as the ratio of its probability of being in the training set and being in the prediction set.

Consequently, the feature vectors are used in re-weighting, regardless of their labels.

This renders covariate shift unsuitable for handling noisy labels.

Our re-weighing scheme is instead gradient-based and as we show next performs well in this task.

Learning with noisy labels Classification with noisy labels is a longstanding problem in the machine learning literature, see the review paper BID9 and the references therein.

Within the realm of deep learning, it has been observed that with sufficiently large data, learning with label noise -without modification to the learning algorithms -actually leads to reasonably high accuracy BID14 ; BID34 .The setting that is of greatest interest to us is when the large noisy dataset is accompanied by a small clean dataset.

BID33 introduce an additional noise layer into the CNN which attempts to adapt the output to align with the noisy label distribution; the parameters of this layer are also learned.

BID39 use a more general noise model, in which the clean label, noisy label, noise type, and image are jointly specified by a probabilistic graphical model.

Both the clean label and the type of noise must be inferred given the image, in this case by two separate CNNs.

consider the same setting, but with additional information in the form of a knowledge graph on labels.

Other recent work on label noise includes BID27 , which shows that adding many copies of an image with noisy labels to a clean dataset barely dents performance; Malach & ShalevShwartz (2017) , in which two separate networks are simultaneously trained, and a sample only contributes to the gradient descent step if there is disagreement between the networks (if there is agreement, that probably means the label is wrong); and BID8 , which analyzes theoretically the situations in which CNNs are more and less resistant to noise.

A pair of papers BID18 ; combine ideas of learning with label noise with instance reweighting.

Bilevel optimization Bilevel optimization problems have a nested structure: the interior level (sometimes called the lower level) is a standard optimization problem; and the exterior level (sometimes called the upper level) is an optimization problem where the objective is a function of the optimal arguments from the interior level.

A branch of mathematical programming, bilevel optimization has been extensively studied within this community BID6 ; Bard (2013) .

For recent developments, readers are referred to the review paper BID30 .

Bilevel optimization has been used in both machine learning, e.g. BID1 and computer vision, e.g. BID24 .

We have two datasets.

The source set is the data-rich set, on which we can learn extensively.

It is denoted by {(x DISPLAYFORM0 , where as usual x s i is the i th source training image, and y s i is its corresponding label.

The second dataset is the target set, which is data-poor; but it is this set which ultimately interests us.

That is, the goal in the end is to learn a classifier on the target set, and the source set is only useful insofar as it helps in achieving this goal.

The target set is denoted DISPLAYFORM1 , and it is assumed that is much smaller than the source set, i.e. n t n s .Our goal is to exploit the source set to solve the target classification problem.

The key insight is that not all source examples contribute equally useful information in regards to the target problem.

For example, suppose that the source set consists of a broad collection of natural images; whereas the target set consists exclusively of various breeds of dog.

We would assume that any images of dogs in the source set would help in the target classification task; images of wolves might also help, as might cats.

Further afield it might be possible that objects with similar textures as dog fur might be useful, such as rugs.

On the flip side, it is probably less likely that images of airplanes and beaches will be relevant (though not impossible).

However, the idea is not to come with any preconceived notions (semantic or otherwise) as to which source images will help; rather, the goal is to let the algorithm choose the relevant source images, in an end-to-end fashion.

We assume that the source and target classifier networks have the same architecture, but different network parameters.

In particular, the architecture is given by DISPLAYFORM2 where ?? is last layer, or possibly last few layers, and ?? constitutes all of the remaining layers.

We will refer to ?? colloquially as the "classifier", and to ?? as the "features" or "representation".

(This is consistent with the usage in related papers, see for example .)

Now, the source and target will share features, but not classifiers; that is, the source network will be given by F (x; ??, ?? s ), whereas the target network will be F (x; ??, ?? t ).

The features ?? are shared between the two, and this is what allows for transfer learning.

The weighted source loss is given by DISPLAYFORM3 where ?? j ??? [0, 1] is a weight assigned to each source training example; and (??, ??) is a per example classification loss, in this case cross-entropy.

The use of the weights ?? j will allow us to decide which source images are most relevant for the target classification task.

The target loss is standard: DISPLAYFORM4 As noted in Section 1, this formulation allows us to address both the transfer learning problem as well as learning with label noise.

In the former case, the source and target may have non-overlapping label spaces; high weights will indicate which source examples have relevant knowledge for the target classification task.

In the latter case, the source is the noisy dataset, the target is the clean dataset, and they share a classifier (i.e. ?? t = ?? s ) as well as a label space; high weights will indicate which source examples do not have label noise, and are therefore reliable.

In either case, the target is much smaller than the source.

The question now becomes: how can we combine the source and target losses into a single optimization problem?

A simple idea is to create a weighted sum of source and target losses.

Unfortunately, issues are likely to arise regardless of the weight chosen.

If the target is weighted equally to the source, then overfitting may likely result given the small size of the target.

On the other hand, if the weights are proportional to the size of the two sets, then the source will simply drown out the target.

A more promising idea is to use bilevel optimization.

Specifically, in the interior level we find the optimal features and source classifier as a function of the weights ??, by minimizing the source loss: DISPLAYFORM5 In the exterior level, we minimize the target loss, but only through access to the source weights; that is, we solve: min DISPLAYFORM6 Why might we expect this bilevel formulation to succeed?

The key is that the target only has access to the features in an indirect manner, by controlling which source examples are included in the source classification problem.

Thus, the target can influence the features chosen, but only in this roundabout way.

This serves as an extra form of regularization, mitigating overfitting, which is the main threat when dealing with a small set such as the target.

Implementing the bilevel optimization is rendered somewhat challenging due to the need to solve the optimization problem in the interior level (1).

Note that this optimization problem must be solved at every point in time; thus, if we choose to solve the optimization (2) for the exterior level via gradient descent, we will need to solve the interior level optimization (1) at each iteration of the gradient descent.

This is clearly inefficient.

Furthermore, it is counter to the standard deep learning practice of taking small steps which improve the loss.

Thus, we instead propose the following procedure.

At a given iteration, we will take a gradient descent step for the interior level problem (1): DISPLAYFORM7 where m is the iteration number; ?? p is the learning rate (where the subscript p stands for "parameters", to distinguish it from a second learning rate for ??, to appear shortly); and Q(??, ?? s ) is a matrix whose j th column is given by DISPLAYFORM8 Thus, Equation (3) leads to an improvement in the features ??, for a fixed set of source weights ??.

Note that there will be an identical descent equation for the classifier ?? s , which we omit for clarity.

Given this iterative version of the interior level of the bilevel optimization, we may now turn to the exterior level.

Plugging Equation (3) into Equation FORMULA6 gives the following problem: DISPLAYFORM9 DISPLAYFORM10 where we have suppressed Q's arguments for readability.

We can then take a gradient descent step of this equation, yielding: DISPLAYFORM11 where in the final line, we have made use of the fact that ?? p is small.

Of course, there will also be a descent equation for the classifier ?? t .

The resulting update scheme is quite intuitive: source example weights are update according to how well they align with the target aggregated gradient.

We have not yet dealt with the weight constraint.

That is, we would like to explicitly require that each ?? j ??? [0, 1].

We may achieve this by requiring ?? j = ??(?? j ) where the new variable ?? j ??? R, and ?? : R ??? [0, 1] is a sigmoid-type function.

As shown in Appendix A, for a particular piecewise linear sigmoid function, replacing the Update Equation (4) with a corresponding update equation for ?? is equivalent to modifying Equation (4) to read DISPLAYFORM12 where CLIP [0, 1] clips the values below 0 to be 0; and above 1 to be 1.

FORMULA7 and FORMULA12 , along with the descent equations for the source and target classifiers ?? s and ?? t .

As usual, the whole operation is done on a mini-batch basis, rather than using the entire set; note that if processing is done in parallel, then source minibatches are taken to be non-overlapping, so as to avoid conflicts in the weight updates.

SOSELETO is summarized in Algorithm 1.

Note that the target derivatives ???L t /????? and ???L t /????? t are evaluated over a target mini-batch; we suppress this for clarity.

In terms of time-complexity, we note that each iteration requires both a source batch and a target batch; assuming identical batch sizes, this means that SOSELETO requires about twice the time as the ordinary source classification problem.

Regarding space-complexity, in addition to the ordinary network parameters we need to store the source weights ??.

Thus, the additional relative spacecomplexity required is the ratio of the source dataset size to the number of network parameters.

This is obviously problem and architecture dependent; a typical number might be given by taking the source dataset to be Imagenet ILSVRC-2012 (size 1.2M) and the architecture to be ResNeXt-101 BID40 (size 44.3M parameters), yielding a relative space increase of about 3%.Convergence properties SOSELETO is only an approximation to the solution of a bilevel optimization problem.

As a result, it is not entirely clear whether it will even converge.

In Appendix B, we demonstrate a set of sufficient conditions for SOSELETO to converge to a local minimum of the target loss L t .

We briefly discuss some implementation details.

In all experiments, we use the SGD optimizer without learning rate decay, and we use ?? ?? = 1.

We initialize the ??-values to be 1, and in practice clip them to be in the slightly expanded range [0, 1.1]; this allows more relevant source points some room to grow.

Other settings are experiment specific, and are discussed in the relevant sections.

To illustrate how SOSELETO works on the problem of learning with noisy labels, we begin with a synthetic experiment, see FIG0 .

The setting is straightforward: the source dataset consists of 500 points which lie in R 2 .

There are two labels / classes, and the ideal separator between the classes is the y-axis.

However, of the 500 points, 100 are corrupted: that is, they lie on the wrong side of the separator.

This is shown in FIG0 , in which one class is shown as white triangles and the second as black pluses.

The target dataset is a set of 50 points, which are "clean", in the sense that they lie on the correct sides of the separator.

(For the sake of simplicity, the target set is not illustrated.)

SOSELETO is run for 100 epochs.

In FIG0 and 1(c), we choose a threshold of 0.1 on the weights ??, and colour the points accordingly.

In particular, in FIG0 (b) the clean (i.e. correctly labelled) instances which are above the threshold are labelled in green, while those below the threshold are labelled in red; as can be seen, all of the clean points lie above the threshold for this choice of threshold, meaning that SOSELETO has correctly identified all of the clean points.

In FIG0 (c), the noisy (i.e. incorrectly labelled) instances which are below the threshold are labelled in green; and those above the threshold are labelled in red.

In this case, SOSELETO correctly identifies most of these noisy labels by assigning them small weights (below 0.1); in fact, 92 out of 100 points are assigned such small weights.

The remaining 8 points, those shown in red, are all near the separator, and it is therefore not very surprising that SOSELETO mislabels them.

All told, using this particular threshold the algorithm correctly accounts for 492 out of 500 points, i.e. 98.4%.

FIG0 .

In FIG0 (e), a plot is shown of mean weight vs. training epoch for clean instances and noisy instances; the width of each plot is the 95% confidence interval of the weights of that type.

All weights are initialized at 0.5; after 100 epochs, the clean instances have a mean weight of about 0.8, whereas the noisy instances have a mean weight of about 0.05.

The evolution is exactly as one would expect.

FIG0 (e) examines the role of the threshold, chose as 0.1 in the above discussion; although 0.1 is a good choice in this case, the good behaviour is fairly robust to choices in the range of 0.1 to 0.4.

We now turn to a real-world setting of the problem of learning with label noise.

We use a noisy version of CIFAR-10 Krizhevsky & Hinton (2009), following the settings used in BID33 ; BID39 .

In particular, an overall noise level is selected.

Based on this, a label confusion matrix is chosen such that the diagonal entries of the matrix are equal to one minus the noise level, and the off-diagonals are chosen randomly (while maintaining the matrix's stochasticity).

Noisy labels are then sampled according to this confusion matrix.

We run experiments for various overall noise levels.

The target consists of a small clean dataset.

CIFAR-10's train set consists of 50K images; of this 50K, both BID33 ; BID39 set aside 10K clean examples for pre-training, a necessary step in both of these algorithms.

In contrast, we use a smaller clean dataset of half the size, i.e. 5K examples while the rest of the 45K samples are noisy.

We compare our results to the two state of the art methods BID33 ; BID39 , as they both address the same setting as we do -the large noisy dataset is accompanied by a small clean dataset, with no extra side-information available.

In addition, we compare with the baseline of simply training on the noisy labels without modification.

In all cases, Caffes CIFAR-10 Quick cif architecture has been used.

For SOSELETO, we use the following settings: ?? p = 10 ???4 , the target batch-size is 32, and the source batch-size is 256.

We use a larger source batch-size to enable more ??-values to be affected quickly.

Results are shown in TAB1 for three different overall noise levels, 30%, 40%, and 50%.

Performance is reported for CIFAR-10's test set, which is of size 10K.

(Note that the competitors' performance numbers are taken from BID39 .)

SOSELETO achieves state of the art on all three noise levels, with considerably better performance than both BID33 and BID39 : between 2.6% to 3.2% absolute improvement.

Furthermore, it does so in each case with only half of the clean samples used in BID33 BID39 .We perform further analysis by examining the ??-values that SOSELETO chooses on convergence, see Figure 4 .2.

To visualize the results, we imagine thresholding the training samples in the source set on the basis of their ??-values; we only keep those samples with ?? greater than a given threshold.

By increasing the threshold, we both reduce the total number of samples available, as well as change the effective noise level, which is the fraction of remaining samples which have incorrect labels.

We may therefore plot these two quantities against each other, as shown in Figure 4 .2; we show three plots, one for each noise level.

Looking at these plots, we see for example that for the 30% noise level, if we take the half of the training samples with the highest ??-values, we are left with only about 4% which have incorrect labels.

We can therefore see that SOSELETO has effectively filtered out the incorrect labels in this instance.

For the 40% and 50% noise levels, the corresponding numbers are about 10% and 20% incorrect labels; while not as effective in the 30% noise level, SOSELETO is still operating as designed.

Further evidence for this is provided by the large slopes of all three curves on the righthand side of the graph.

We now examine the performance of SOSELETO on a transfer learning task.

In order to provide a challenging setting, we choose to (a) use source and target sets with disjoint label sets, and (b) use a very small target set.

In particular, the source dataset is chosen to the subset of Google Street View House Numbers (SVHN) BID23 corresponding to digits 0-4.

SVHN's train set is of size 73,257 images, with about half of those belonging to the digits 0-4.

The target dataset is a very small subset of MNIST BID16 corresponding to digits 5-9.

While MNIST's train set is of size 60K, with 30K corresponding to digits 5-9, we use very small subsets: either 20 or 25 images, with equal numbers sampled from each class (4 and 5, respectively).

Thus, as mentioned, there is no overlap between source and target classes, making it a true transfer learning (rather than domain adaptation) problem; and the small target set size adds further challenge.

Furthermore, this task has already been examined in .We compare our results with the following techniques.

Target only, which indicates training on just the target set; standard fine-tuning; Matching Nets BID38 , a few-shot technique which is relevant given the small target size; fine-tuned Matching Nets, in which the previous result is then fine-tuned on the target set; and two variants of the Label Efficient Learning technique -one which includes fine-tuning plus a domain adversarial loss, and the other the full technique presented in .

Note that besides the target only and fine-tuning approaches, all other approaches depend on unlabelled target data.

Specifically, they use all of the remaining MNIST 5-9 examples -about 30,000 -in order to aid in transfer learning.

SOSELETO, by contrast, does not make use of any of this data.

For each of the above methods, the simple LeNet architecture BID16 was used.

For SOSELETO, we use the following settings: ?? p = 10 ???2 , the source batch-size is 32, and the target batch-size is 10 (it is chosen to be small since the target itself is very small).

Additionally, the SVHN images were resized to 28 ?? 28, to match the MNIST size.

The performance of the various methods is shown in TAB2 , and is reported for MNIST's test set which is of size 10K.

We have divided TAB2 into two parts: those techniques which use the 30K examples of unlabelled data, and those which do not.

SOSELETO has superior performance to all of the techniques which do not use unlabelled data.

Furthermore, SOSELETO has superior performance to all of the techniques which do use unlabelled data, except the Label Efficient technique.

It is noteworthy in particular that SOSELETO outperforms the few-shot techniques, despite not being designed to deal with such small amounts of data.

In Appendix C we further analyze which SVHN instances are considered more useful than others by SOSELETO, by transfering all of SVHN classes to MNSIT 5-9.

Two-stage SOSELETO Finally, we note that although SOSELETO is not designed to use unlabelled data, one may do so using the following two-stage procedure.

Stage 1: run SOSELETO as described above.

Stage 2: use the learned SOSELETO classifier to classify the unlabelled data.

This will now constitute a dataset with noisy labels, and SOSELETO can now be run in the mode of training with label noise, where the noisily labelled unsupervised data is now the source, and the target remains the same small clean set.

In the case of n t = 25, this procedure elevates the accuracy to above 92%.

We have presented SOSELETO, a technique for exploiting a source dataset to learn a target classification task.

This exploitation takes the form of joint training through bilevel optimization, in which the source loss is weighted by sample, and is optimized with respect to the network parameters; while the target loss is optimized with respect to these weights and its own classifier.

We have derived an efficient algorithm for performing this bilevel optimization, through joint descent in the network parameters and the source weights, and have analyzed the algorithm's convergence properties.

We have empirically shown the effectiveness of the algorithm on both learning with label noise, as well as transfer learning problems.

An interesting direction for future research involves incorporating an additional domain alignment term into SOSELETO, in the case where the source and target dataset have overlapping labels.

We note that SOSELETO is architecture-agnostic, and thus may be easily deployed.

Furthermore, although we have focused on classification tasks, the technique is general and may be applied to other learning tasks within computer vision; this is an important direction for future research.

Recall that our goal is to explicitly require that ?? j ??? [0, 1].

We may achieve this by requiring DISPLAYFORM0 where the new variable ?? j ??? R, and ??(??) is a kind of piecewise linear sigmoid function.

Now we will wish to replace the Update Equation (4), the update for ??, with a corresponding update equation for ??.

This is straightforward.

Define the Jacobian ?????/????? by ????? ????? ij = ????? i ????? jThen we modify Equation (4) to read DISPLAYFORM1 The Jacobian is easy to compute analytically: where CLIP [0, 1] clips the values below 0 to be 0; and above 1 to be 1.

DISPLAYFORM2

SOWETO is only an approximation to the solution of a bilevel optimization problem.

As a result, it is not entirely clear whether it will even converge.

In this section, we demonstrate a set of sufficient conditions for SOWETO to converge to a local minimum of the target loss L t .To this end, let us examine the change in the target loss from iteration m to m + 1:

DISPLAYFORM0 Now, we can use the evolution of the weights ??.

Specifically, we substitute Equation FORMULA11

@highlight

Learning with limited training data by exploiting "helpful" instances from a rich data source.  