Curriculum learning and Self paced learning are popular topics in the machine learning that suggest to put the training samples in order by considering their difficulty levels.

Studies in these topics show that starting with a small training set and adding new samples according to difficulty levels improves the learning performance.

In this paper we experimented that we can also obtain good results by adding the samples randomly without a meaningful order.

We compared our method with classical training, Curriculum learning, Self paced learning and their reverse ordered versions.

Results of the statistical tests show that the proposed method is better than classical method and similar with the others.

These results point a new training regime that removes the process of difficulty level determination in Curriculum and Self paced learning and as successful as these methods.

BID1 named Curriculum learning the idea of following an order related with difficulty of the samples during training which provides an optimization for non convex objectives.

After this many researchers tried to find the most efficient curriculum to get the best yield with this approach.

In BID15 's study conventional curriculum learning did not work so well and they developed a new version.

BID14 proposed three different curriculum strategies for language model adaptation of recurrent neural networks.

In the field of computer vision BID13 looked for the best order of tasks to learn.

Although these models have better generalization performance with the proposed curriculum methods it is not known whether the tried methods ensures the best curriculum.

A curriculum is work-specific so could not be applicable for another work.

In order to use the curriculum logic in different applications BID10 suggested a method that the learner decides itself which samples are easy or difficult at every stage.

This method called Self paced learning was combined with Curriculum learning which provides prior information by BID9 .

In another work BID7 introduced a method to automatically select the syllabus to follow for the neural networks.

BID12 also proposed a way to learn simple subtasks before the complex tasks and achieved better results than using manually designed curriculum.

In some cases higher learning performance could be obtainable by adding some noises to easy-tohard ordering of the samples.

BID8 gave preference to both easy and diverse samples and outperform the conventional Self paced learning BID10 ) algorithm.

Emphasizing the uncertain samples suggested by BID3 lead to more accurate and robust SGD training.

BID0 explored the inversed versions of the Self paced learning and Self paced learning with diversity BID8 ) and demonstrated that these methods performed slightly better than their standard variants.

Consistent with the literature we have showed in our previous work () that using both curriculum and anti-curriculum strategies improving generalization performance in a wide application area.

These researches brings a question to minds: While it is natural and logical to obtain better results by sorting the samples from easy-to-hard why it is also better to sort the samples from hard-to-easy?In this study we point that to start with a small training set and add new samples in both curriculum and anti-curriculum learning makes these methods better.

So we claim that it is possible to have better results only by adding new samples stage-by-stage without a meaningful order.

We experimented two ordering types related with difficulty (easy-to-hard and hard-to-easy) and our method without a meaningful order.

Training was carried out by adding a new group to the training set at every stage.

We compared the proposed method with two strategies.

First one is Curriculum learning which we give the difficulty levels of the samples as pre-information.

Second one is Self paced learning which the trained network determines the difficulty levels of the samples at each stage.

All methods including usual baseline training have been compared by using paired T-test and the results are examined.

We explored 2 ordering types determined by 2 different strategies and our method without ordering.

In the Curriculum learning(CL) learner follows a pre-determined curriculum during training.

For this reason it is necessary to know the difficulty of the samples.

This information can be given by defining the specific curriculum of the task or labeling the samples at the beginning.

Even though labeling with difficulty levels is easy for artificial data sets, it is costly and demanding for real world data sets.

Besides that, a sample which is easy for humans may not be easy for machines.

In this study, we used an ensemble method to automatically determine the difficulty of the samples.

We created an ensemble with Bagging method BID2 ) and calculated decision consistency for each training sample accordingly the difficulty levels are determined.

Then, the ordered training set which is easy-to-hard was grouped for Curriculum learning and hard-to-easy ordered training set was grouped for Anti-curriculum learning(ACL).For CL, the samples are sorted firstly then separated into same sized groups by dividing the number of training samples into the number of stages.

Remainings are added to the first stage so that the number of new samples to add at each stage is equal.

Training starts from the easiest group and continues by adding the groups one-by-one according to the easiness.

Hardest group is given with the entire training set at the last stage and training is completed in stages same number as groups.

For ACL, the training set is sorted by difficulty in descending order and then separated into groups.

Training starts with the group that is the most difficult.

At the last stage of the training, the group includes the easiest samples with the entire training set.

The groups given at each stage are shown in TAB0 when the training set considered as sorted according to the difficulty levels.

At each step a new group of samples are given together with the previous samples for both methods.

In the last stage the entire training set is given like the baseline training.

The path for CL is given in Algorithm 1.

The same algorithm is applied for ACL with reverse order training set.

Curriculum learning starts with random initialized weights in the first stage.

The optimum weights found at each stage taken as the initial weights of the next stage.

In other words, next stage of the optimization starts from the minimum found in the previous stage.

Self paced learning(SPL) is a solution for finding the difficulty levels in Curriculum learning.

In this method learner determines the instances to learn according to the current situation of the objective for t = 1 to n ??? 1 do 10: DISPLAYFORM0 end for

return ?? n 14: end procedure function.

Training starts with a group of random samples and the samples that best fit the current model space are labeled as easy at each step.

Weights are updated by easy labeled examples at the next step.

The network is trained with more examples in the next step and all training set is given in the final step.

This is realized by annealing the self pace parameter.

Same number of samples at each stage can be added to the training set to make the training progress homogeneous.

In this work the only parameter to set for SPL is the number of stages.

The number of samples to add at each stage is equal like as CL.

Training starts with randomly selected samples.

At the end of each stage, the samples are sorted by ascending order according to the losses calculated with MSE.

The samples to give in the next stage are selected from this order.

The most consistent samples with the model space are selected for the next stage.

For the Self paced learning-Inversed(SPLI), the most outlying samples have priority.

Same as SPL, training starts with random samples.

Training samples in the next stage are selected by considering the incompatibility with the model space.

All training set is sorted by loss in descending order, samples with the highest loss are selected for the next stage.

The path for SPL is given in Algorithm 2.

The same algorithm is applied for SPLI, only training set is sorted by reverse order at each stage.

Training in the next stage is started with using previous solutions in SPL also as in CL.

SPL does not guarantee that the all samples from previous stage will be taken again in the next stage.

The samples best adapted with the objective function are taken among all training samples and some of the samples used in the previous stage may not be inside these samples.

For example, if there is noisy samples in the first stage these probably will not be in the training set at the second stage.

Algorithm 2 Self paced learning 1: T ??? random ordered training set 2: n ??? number of stages 3: s ??? number of samples to add at each stage 4: f ??? non-convex objective function of neural network with parameters ?? 5: procedure SPL(T, n, s, f ) 6: Randomly initialize the parameters ?? 0 7: T 0 = randomly chosen s sample from T 8: DISPLAYFORM0 for t = 1 to n ??? 1 do 10:D t = training set sorted by loss of f (T, ?? t ) ascending order 11:T t = first (t + 1) * s sample of D t 12: DISPLAYFORM1 end for 14:return ?? n 15: end procedure 2.3 RANDOM ORDERED GROWING SETS CL, SPL and their inverse versions have a common point: At each next step the number of training samples is increasing with addition of a new group.

Due to the both versions (easy-to-hard and hard-to-easy) have better performance than the classical method it is considered that this common feature may provide an optimization.

We investigated the total weight change in each iteration for all methods in the Appendix A. In this case, it is important to train with accumulating groups rather than giving the samples with a meaningful order.

Therefore, dividing the unordered training set into the groups and adding a new group at each stage could also raise the generalization performance.

In the Random ordered growing sets(ROGS) method unsorted training set is divided into groups with equal number of samples.

Training is carried out in the same fashion as starting with the first group then adding a new group at each stage without a meaningful order.

Algorithm 3 shows the way of training with Random ordered growing sets.

Same as the other methods, ROGS uses the found solutions in the previous stages as initial weights in the next stage.

Algorithm 3 Random ordered growing sets 1: T ??? random ordered training set 2:

n ??? number of stages 3: s ??? number of samples to add at each stage 4: f ??? non-convex objective function of neural network with parameters ?? 5: procedure ROGS(T, n, s, f ) 6: Randomly initialize the parameters ?? 0 7: T 0 = randomly chosen s sample from T 8: DISPLAYFORM2 for t = 1 to n ??? 1 do 10:T t = randomly choose (t + 1) * s sample from T 11: DISPLAYFORM3 end for

return ?? n 14: end procedure Examined 5 methods find the optimum weights with the same objective: DISPLAYFORM0 where s denotes the number of samples to add at each stage, parameters at each stage determined by same amount of samples for all methods.

x i , y i are taken from pre-sorted easy-to-hard training set for CL, from pre-sorted hard-to-easy training set for ACL, from the training set that sorted easy-tohard at each stage for SPL and from the training set that sorted hard-to-easy at each stage for SPLI.

In ROGS method samples are selected from non-ordered training set.

Definitions: (a) The loss function (??) with one parameter ?? ??? R is given as in (2) where N is the number of training instances and the loss function for each individual instance i (??) is given as in (3).

DISPLAYFORM0 (b) The point ?? B is a definite local minimum of (??).

Geometric representation of a definite local minimum point is shown in FIG0 (a) with the loss functions.

We show the loss functions for 10 instances with thin lines and their average with the bold line.

If we denote the expected value of (??) as E[ (??)], expected values of the loss function in the given points can be ordered as in (4).

DISPLAYFORM1 Expected values of the derivatives of the loss function in the given points can be written as in FORMULA8 and the derivatives are given in FIG0 .

DISPLAYFORM2 DISPLAYFORM3 (b) More than half of the individual instances are less complex than average of all instances.

DISPLAYFORM4 Lemma 1: Probability density function of the derivatives in the ?? B point has a skewed distribution with zero mean.

Proof: Derivatives of the loss functions is under the point ?? B in FIG0 (b) for more than half of the individual instances.

However i (?? B ) values are high for the above instances.

If we denote the probability density function of the derivatives in the ?? B as Pr( i (?? B )), 1) Definition (c) gives that ?? B is a local minimum therefore Pr( i (?? B )) has zero mean.2) If ??( (??)) = 2 as in FIG0 (b) then ??( i (??)) <= 2 from Assumption (a) and ??( i (??)) < 2 for more than half of the instances from Assumption (b).

i (?? B ) < 0 for the instances that has ??( i (??)) < 2.

i (?? B ) < 0 for more than half of the instances therefore Pr( i (?? B )) has a skewed distribution.

Corollary 1: If we take a subset with k instances from the training set we can denote: DISPLAYFORM5 Expected value for the derivative of the loss function of the subset at the point ?? B is less than zero.

DISPLAYFORM6 Proof: Probability of being negative for E[ s (?? B )] depends on k and always high from Lemma 1.Theorem 1: If we take a subset with k instances and train with batch gradient descent, we obtain a better local minimum for this subset.

Proof: Optimization in the (??) surface which starts from ?? A will probably end at ?? B .

However in the s (??) surface optimization will be continue to find a better minimum for the current set according to Corollary 1.Theorem 2: To continue the optimization with all samples provides a better local minimum for all training set than stopping at the minimum of the subset.

Proof: Error surface for all instances ( (??)) and some subsample instances ( sx (??), sy (??), sz (??), sw (??)) are given in FIG2 .

These examples shows all possible situations for the stopping points by considering ?? A , ?? B and ?? C .

We suppose to start the optimization from ?? A .-If we use (??) it will be stop at ?? B .-If we use sx (??) it will be stop at ?? sx in the range ?? B < ?? sx < ?? C .

If we finish the optimization here, error on the surface for all instances will be high (E[ (?? sx )] > E[ (?? B )]).

Therefore optimizing in the (??) subsequently prevents to stay at a worse point.-If we use (?? sy ) it will be stop at ?? sy in the range ?? A < ?? sy < ?? B .

Here E[ (?? sy )]

> E[ (?? B )] therefore optimization for all instances is necessary.-If we use (?? sz ) it will be stop at ?? sz in the range ?? C < ?? sz and it is possible to obtain a better local minimum with optimization on (??) surface by starting from the local minimum of the subset.-If we use (?? sw ) it will be stop at ?? sw in the range ?? A > ?? sw .

Here the stopping point can be worse for all instances.

Therefore it is possible to avoid from worse point by optimization on the (??) surface.

We can obtain a better local minimum with training with growing sets.

Proof: Subset of the training set provides to get a better local minimum as in Theorem 1.

By the same way subset of the subset surface can be provide a better local minimum for the subset.

To optimize the surface of the whole training set first optimize the following surfaces respectively: DISPLAYFORM7

We trained 3-layer artificial neural network which has 10 neurons in the hidden layer with stochastic gradient descent with momentum for all methods.

Stop condition is the rising validation error series for 6 times during the training.

Baseline and all other methods applied with incremental training.

We set the number of stages as 25 for all methods with growing sets.

The network must provide the stopping condition at each stage to pass the next stage.

Optimized weights in the previous stage are given as initial weights in the next stage.

We have tested the suggested methods in 36 data sets retrieved from UCI repository 1 with the sample number range from 57 to 20000.

We divide each data set 5 folds, 3 for training, 1 for validation and 1 for testing at each experiment.

For each data set 20 error rates (MSE) obtained with 4x5 fold cross validation are compared by 0.95 significance level paired T-test.

Results of the comparisons are in TAB1 . (Abbreviations are as follows: Curriculum learning=CL, Anti-curriculum learning=ACL, Self paced Learning=SPL, Self paced learning-Inversed=SPLI, Random ordered growing sets=ROGS) Each cell contains the win/tie/loss information for corresponding row against corresponding column.

For example, Curriculum learning wins versus Baseline in 12 data sets, ties in 24 data sets, not loses in any data set.

Results of the comparisons with Baseline for all data sets are given in the Appendix B with the number of samples, number of features, number of classes and average Baseline MSE.

If the comparison result of the corresponding method against Baseline is win it is marked with W, tie marked with T and loss marked with L.It is seen that training with ROGS is better than Baseline method in 17 data sets.

Considering the comparisons with the Baseline, ROGS method wins in more data sets than CL, ACL and SPL.

Obtaining good results without a meaningful order shows that only giving the training set as growing subsets provides an optimization.

Additionally, SPLI method wins against Baseline in more than half of the data sets.

This method seems as the best method in terms of total number of wins.

It is striking that CL and SPL methods did not lose against Baseline in any data set.

This shows the robustness of these methods in noisy data sets.

It is possible to give priority to noises with hard-toeasy and random ordered methods so training may be misdirected in the data sets with high error rates.

In such data sets giving priority to easy examples provides a safer training.

We drew our attention that both versions of training with easy-to-hard ordered and hard-to easy ordered samples have better performance.

That led us to investigate what common issues they have.

We considered that their common point is growing the training sets during training.

Therefore, instead of ordering the samples according to difficulty we only added some samples randomly at each stage.

In these experiments we obtained similar results with Curriculum, Anti-curriculum, Self Paced and Self Paced-Inversed methods which are related to difficulty levels.

According to these results, we can claim that the success of Curriculum learning and Self paced learning approaches not comes from the fact that they follow a meaningful order but trained by growing training sets.

In FIG0 we showed some examples for the individual instances.

We started the optimization from ?? A , instances under this point are considered as easy and above are difficult.

If we take an easy instance it is possible or not to guide the optimization to a better minimum.

It will be stop at the local minimum ?? B in the worst case.

Similarly if we take a difficult instance it is possible or not to obtain a better result.

Implementation results also showed that both easy-to-hard and hard-to-easy ordered methods can be successful.

Therefore ordering of the samples are not so important to guide the optimization.

It is a better situation to shorten the distance between ?? B and ?? C in FIG2 to bypass the local minimum.

When the points are same for a saddle point, training with growing sets will probably overcome this point and find a better minimum.

This is a good condition when considering saddle points are so much than local minimums in high dimensional functions as mentioned in BID4 .On many data sets with different distributions we used ensemble method to automatically determine the difficulty of the samples for curriculum learning.

Pre-processing for difficulty level determination can be thought to caused slowdown.

However it has provided a faster neural network training than SPL.

Also it could be said that ensemble method set a better ordering than SPL by considering their number of wins against Baseline.

ACL and SPLI, which are the inverse versions of the CL and SPL methods, has performed poorly in some high error rated data sets.

The effect of giving the samples at different points during the training has been studied in BID5 .

In these methods, noisy examples may be effecting the output more because of giving at the beginning.

Nevertheless, the inverse versions of the approaches have better performance than their standard versions.

However, CL and SPL methods did not lose in any data set so this shows they have a robust aspect.

It is thought that these methods must have a theoretical explanation about ensuring resistance to noises.

BID6 studied on why these methods are effectiveness especially on big and noisy data.

SPLI method has the most winning against Baseline.

In this method strategy of selecting the samples to learn at each stage reminds pool-based active learning BID11 ) in which the learner wants to learn the uncertain samples of the unlabeled data pool.

Also non-loss of CL and SPL, and more wins of ACL and SPLI shows the necessity of determining the valuable-example-based curriculum instead of easiness-based-curriculum for the future work.

??? Previous work on CL and SPL has made experiments on specific domains.

In this study, the performance of these methods on various data sets has been shown.??? We proposed a new method that need not to determine the difficulty levels as in CL and SPL, so faster and as successful as these methods.

In order to find the common feature of CL, SPL and their reverse versions, we examined the total weight change in each iteration during training.

Graphs of each method for vehicle data set are given in the FIG3 .

In these graphs, total weight change is increasing and decreasing during stages.

This may be indicates a feature that provides avoidance from the local minimum.

By adding a new group at each stage, it is possible to continue with a larger step from the minimum of the previous stage.

Iterations of the last stage which we give the all training set is marked with orange.

When the whole training set is given it reaches the minimum in a shorter time than baseline in all methods.

This implies the whole training set has been started from a better minimum in the methods with growing sets than random initializing in the baseline.

The common feature of the methods are not related to the ordering of the samples.

Only the growth of the training set at each stage with the addition of a new group made to continue the optimization with avoiding local minimum.

In this case training with growing sets without ordering can be also provide a better minimum.

@highlight

We propose that training with growing sets stage-by-stage provides an optimization for neural networks.

@highlight

The authors compare curriculum learning to learning in a random order with stages that add a new sample of examples to the previously, randomly constructed set

@highlight

This paper studies the influence of ordering in the Curriculum and Self paced learning, and shows that to some extent the ordering of training instances is not important.