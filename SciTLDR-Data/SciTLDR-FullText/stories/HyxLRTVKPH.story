In most practical settings and theoretical analyses, one assumes that a model can be trained until convergence.

However, the growing complexity of machine learning datasets and models may violate such assumptions.

Indeed, current approaches for hyper-parameter tuning and neural architecture search tend to be limited by practical resource constraints.

Therefore, we introduce a formal setting for studying training under the non-asymptotic, resource-constrained regime, i.e., budgeted training.

We analyze the following problem: "given a dataset, algorithm, and fixed resource budget, what is the best achievable performance?" We focus on the number of optimization iterations as the representative resource.

Under such a setting, we show that it is critical to adjust the learning rate schedule according to the given budget.

Among budget-aware learning schedules, we find simple linear decay to be both robust and high-performing.

We support our claim through extensive experiments with state-of-the-art models on ImageNet (image classification), Kinetics (video classification), MS COCO (object detection and instance segmentation), and Cityscapes (semantic segmentation).

We also analyze our results and find that the key to a good schedule is budgeted convergence, a phenomenon whereby the gradient vanishes at the end of each allowed budget.

We also revisit existing approaches for fast convergence and show that budget-aware learning schedules readily outperform such approaches under (the practical but under-explored) budgeted training setting.

Deep neural networks have made an undeniable impact in advancing the state-of-the-art for many machine learning tasks.

Improvements have been particularly transformative in computer vision (Huang et al., 2017b; He et al., 2017) .

Much of these performance improvements were enabled by an ever-increasing amount of labeled visual data (Russakovsky et al., 2015; Kuznetsova et al., 2018) and innovations in training architectures (Krizhevsky et al., 2012; He et al., 2016) .

However, as training datasets continue to grow in size, we argue that an additional limiting factor is that of resource constraints for training.

Conservative prognostications of dataset sizes -particularly for practical endeavors such as self-driving cars (Bojarski et al., 2016) , assistive medical robots (Taylor et al., 2008) , and medical analysis (Fatima & Pasha, 2017) -suggest one will train on datasets orders of magnitude larger than those that are publicly available today.

Such planning efforts will become more and more crucial, because in the limit, it might not even be practical to visit every training example before running out of resources (Bottou, 1998; Rai et al., 2009 ).

We note that resource-constrained training already is implicitly widespread, as the vast majority of practitioners have access to limited compute.

This is particularly true for those pursuing research directions that require a massive number of training runs, such as hyper-parameter tuning (Li et al., 2017) and neural architecture search (Zoph & Le, 2017; Cao et al., 2019; Figure 1 : We formalize the problem of budgeted training, in which one maximizes performance subject to a fixed training budget.

We find that a simple and effective solution is to adjust the learning rate schedule accordingly and anneal it to 0 at the end of the training budget.

This significantly outperforms off-the-shelf schedules, particularly for small budgets.

This plot shows several training schemes (solid curves) for ResNet-18 on ImageNet.

The vertical axis in the right plot is normalized by the validation accuracy achieved by the full budget training.

The dotted green curve indicates an efficient way of trading off computation with performance.

Instead of asking "what is the best performance one can achieve given this data and algorithm?", which has been the primary focus in the field so far, we decorate this question with budgeted training constraints as follows: "what is the best performance one can achieve given this data and algorithm within the allowed budget?".

Here, the allowed budget refers to a limitation on the total time, compute, or cost spent on training.

More specifically, we focus on limiting the number of iterations.

This allows us to abstract out the specific constraint without loss of generality since any one of the aforementioned constraints could be converted to a finite iteration limit.

We make the underlying assumption that the network architecture is constant throughout training, though it may be interesting to entertain changes in architecture during training (Rusu et al., 2016; Wang et al., 2017) .

Much of the theoretical analysis of optimization algorithms focuses on asymptotic convergence and optimality (Robbins & Monro, 1951; Nemirovski et al., 2009; Bottou et al., 2018) , which implicitly makes use of an infinite compute budget.

That said, there exists a wide body of work (Zinkevich, 2003; Kingma & Ba, 2015; Reddi et al., 2018; Luo et al., 2019) that provide performance bounds which depend on the iteration number, which apply even in the non-asymptotic regime.

Our work differs in its exploration of maximizing performance for a fixed number of iterations.

Importantly, the globally optimal solution may not even be achievable in our budgeted setting.

Given a limited budget, one obvious strategy might be data subsampling (Bachem et al., 2017; Sener & Savarese, 2018) .

However, we discover that a much more effective, simpler, and under-explored strategy is adopting budget-aware learning rate schedules -if we know that we are limited to a single epoch, one should tune the learning schedule accordingly.

Such budget-aware schedules have been proposed in previous work (Feyzmahdavian et al., 2016; Lian et al., 2017) , but often for a fixed learning rate that depends on dataset statistics.

In this paper, we specifically point out linearly decaying the learning rate to 0 at the end of the budget, may be more robust than more complicated strategies suggested in prior work.

Though we are motivated by budget-aware training, we find that a linear schedule is quite competitive for general learning settings as well.

We verify our findings with state-of-the-art models on ImageNet (image classification), Kinetics (video classification), MS COCO (object detection and instance segmentation), and Cityscapes (semantic segmentation).

We conduct several diagnostic experiments that analyze learning rate decays under the budgeted setting.

We first observe a statistical correlation between the learning rate and the full gradient magnitude (over the entire dataset).

Decreasing the learning rate empirically results in a decrease in the full gradient magnitude.

Eventually, as the former goes to zero, the latter vanishes as well, suggesting that the optimization has reached a critical point, if not a local minimum 1 .

We call this phenomenon budgeted convergence and we find it generalizes across budgets.

On one hand, it implies that one should decay the learning rate to zero at the end of the training, even given a small budget.

On the other hand, it implies one should not aggressively decay the learning rate early in the optimization (such as the case with an exponential schedule) since this may slow down later progress.

Finally, we show that linear budget-aware schedules outperform recently-proposed fast-converging methods that make use of adaptive learning rates and restarts.

Our main contributions are as follows:

??? We introduce a formal setting for budgeted training based on training iterations and provide an alternative perspective for existing learning rate schedules.

??? We discover that budget-aware schedules are handy solutions to budgeted training.

Specifically, our proposed linear schedule is more simple, robust, and effective than prior approaches, for both budgeted and general training.

??? We provide an empirical justification of the effectiveness of learning rate decay based on the correlation between the learning rate and the full gradient norm.

Learning rates.

Stochastic gradient descent dates back to Robbins & Monro (1951) .

The core is its update step: w t = w t???1 ??? ?? t g t , where t (from 1 to T ) is the iteration, w are the parameters to be learned, g is the gradient estimator for the objective function 2 F , and ?? t is the learning rate, also known as step size.

Given base learning rate ?? 0 , we can define the ratio ?? t = ?? t /?? 0 .

Then the set of {?? t } T t=1 is called the learning rate schedule, which specifies how the learning rate should vary over the course of training.

Our definition differs slighter from prior art as it separates the base learning rate and learning rate schedule.

Learning rates are well studied for (strongly) convex cost surfaces and we include a brief review in Appendix H.

Learning rate schedule for deep learning.

In deep learning, there is no consensus on the exact role of the learning rate.

Most theoretical analysis makes the assumption of a small and constant learning rate (Du et al., 2018a; b; Hardt et al., 2016) .

For variable rates, one hypothesis is that large rates help move the optimization over large energy barriers while small rates help converge to a local minimum (Loshchilov & Hutter, 2017; Huang et al., 2017a; Kleinberg et al., 2018) .

Such hypothesis is questioned by recent analysis on mode connectivity, which has revealed that there does exist a descent path between solutions that were previously thought to be isolated local minima (Garipov et al., 2018; Dr??xler et al., 2018; Gotmare et al., 2019) .

Despite a lack of theoretical explanation, the community has adopted a variety of heuristic schedules for practical purposes, two of which are particularly common:

??? step decay: drop the learning rate by a multiplicative factor ?? after every d epochs.

The default for ?? is 0.1, but d varies significantly across tasks.

??? exponential: ?? t = ?? t .

There is no default parameter for ?? and it requires manual tuning.

State-of-the-art codebases for standard vision benchmarks tend to employ step decay (Xie & Tu, 2015; Huang et al., 2017b; He et al., 2017; Carreira & Zisserman, 2017; Wang et al., 2018; Yin et al., 2019; Ma et al., 2019) , whereas exponential decay has been successfully used to train Inception networks (Szegedy et al., 2015; .

In spite of their prevalence, these heuristics have not been well studied.

Recent work proposes several new schedules (Loshchilov & Hutter, 2017; Smith, 2017; Hsueh et al., 2019) , but much of this past work limits their evaluation to CIFAR and ImageNet.

For example, SGDR (Loshchilov & Hutter, 2017) advocates for learning-rate restarts based on the results on CIFAR, however, we find the unexplained form of cosine decay in SGDR is more effective than the restart technique.

Notably, Mishkin et al. (2017) Adaptive learning rates.

Adaptive learning rate methods (Tieleman & Hinton, 2012; Kingma & Ba, 2015; Reddi et al., 2018; Luo et al., 2019) adjust the learning rate according to the local statistics of the cost surface.

Despite having better theoretical bounds under certain conditions, they do not generalize as well as momentum SGD for benchmark tasks that are much larger than CIFAR (Wilson et al., 2017) .

We offer new insights by evaluating them under the budgeted setting.

We show fast descent can be trivially achieved through budget-aware schedules and aggressive early descent is not desirable for achieving good performance in the end.

Learning rate schedules are often defined assuming unlimited resources.

As we argue, resource constraints are an undeniable practical aspect of learning.

One simple approach for modifying an existing learning rate schedule to a budgeted setting is early-stopping.

Fig 1 shows that one can dramatically improve results of early stopping by more than 60% by tuning the learning rate for the appropriate budget.

To do so, we simply reparameterize the learning rate sequence with a quantity not only dependent on the absolute iteration t, but also the training budget T :

Definition (Budget-Aware Schedule).

Let T be the training budget, t be the current step, then a training progress p is t/T .

A budget-aware learning rate schedule is

where f (p) is the ratio of learning rate at step t to the base learning rate ?? 0 .

At first glance, it might be counter-intuitive for a schedule to not depend on T .

For example, for a task that is usually trained with 200 epochs, training 2 epochs will end up at a solution very distant from the global optimal no matter the schedule.

In such cases, conventional wisdom from convex optimization suggests that one should employ a large learning rate (constant schedule) that efficiently descends towards the global optimal.

However, in the non-convex case, we observe empirically that a better strategy is to systematically decay the learning rate in proportion to the total iteration budget.

Budge-Aware Conversion (BAC).

Given a particular rate schedule ?? t = f (t), one simple method for making it budget-aware is to rescale it, i.e., ?? p = f (pT 0 ), where T 0 is the budget used for the original schedule.

For instance, a step decay for 90 epochs with two drops at epoch 30 and epoch 60 will convert to a schedule that drops at 1/3 and 2/3 training progress.

Analogously, an exponential schedule 0.99 t for 200 epochs will be converted into (0.99

It is worth noting that such an adaptation strategy already exists in well-known codebases (He et al., 2017) for training with limited schedules.

Our experiments confirm the effectiveness of BAC as a general strategy for converting many standard schedules to be budget-aware (Tab 1).

For our remaining experiments, we regard BAC as a known technique and apply it to our baselines by default. (He et al., 2016) .

The numbers are classification accuracy on the validation set.

The 100% budget refers to training for 200 epochs.

"step-d1" denotes step decay dropping once at training progress 50%.

Please refer to Sec 4.1 for the complete setup.

Recent schedules: Interestingly, several recent learning rate schedules are implicitly defined as a function of progress p = t T , and so are budget-aware by our definition: ??? poly (Jia et al., 2014) :

No parameter other than ?? = 0.9 is used in published work.

??? cosine (Loshchilov & Hutter, 2017) :

(1 ??? ??)(1 + cos(??p)).

?? specify a lower bound for the learning rate, which defaults to zero.

??? htd (Hsueh et al., 2019) :

Here ?? has the same representation as in cosine.

It is reported that L = ???6 and U = 3 performs the best.

The poly schedule is a feature in Caffe (Jia et al., 2014 ) and adopted by the semantic segmentation community (Chen et al., 2018; Zhao et al., 2017) .

The cosine schedule is a byproduct in work that promotes learning rate restarts (Loshchilov & Hutter, 2017) .

The htd schedule is recently proposed (Hsueh et al., 2019) , which however, contains only limited empirical evaluation.

None of these analyze their budget-aware property or provides intuition for such forms of decay.

These schedules were Table 2 : Comparison of learning rate schedules on CIFAR-10.

The 1st, 2nd and the 3rd place under each budget are color coded.

The number here is the classification accuracy and each one is the average of 3 independent runs.

"step-dx" denotes decay x times at even intervals with ?? = 0.1.

For "exp" and "step" schedules, BAC (Sec 3.1) is applied in place of early stopping.

We can see linear schedule surpasses other schedules under almost all budgets.

treated as "yet another schedule".

However, our definition of budget-aware makes these schedules stand out as a general family.

Inspired by existing budget-aware schedules, we borrow an even simpler schedule from the simulated annealing literature (Kirkpatrick et al., 1983; McAllester et al., 1997; Nourani & Andresen, 1998) 3 :

In Fig 2, we compare linear schedule with various existing schedules under the budget-aware setting.

Note that this linear schedule is completely parameter-free.

This property is particularly desirable in budgeted training, where little budget exists for tuning such a parameter.

The excellent generalization of linear schedule across budgets (shown in the next section) might imply that the cost surface of deep learning is to some degree self-similar.

Note that a linear schedule, together with other recent budget-aware schedules, produces a constant learning rate in the asymptotic limit i.e., lim T ?????? (1 ??? t/T ) = 1.

Consequently, such practically high-performing schedules tend to be ignored in theoretical convergence analysis (Robbins & Monro, 1951; Bottou et al., 2018) .

In this section, we first compare linear schedule against other existing schedules on the small CIFAR-10 dataset and then on a broad suite of vision benchmarks.

The CIFAR-10 experiment is designed to extensively evaluate each learning schedule while the vision benchmarks are used to verify the observation on CIFAR-10.

We provide important implementation settings in the main text while leaving the rest of the details to Appendix K. In addition, we provide in Appendix A the evaluation with a large number of random architectures in the setting of neural architecture search.

CIFAR-10 ( Krizhevsky & Hinton, 2009 ) is a dataset that contains 70,000 tiny images (32 ?? 32).

Given its small size, it is widely used for validating novel architectures.

We follow the standard setup for dataset split (Huang et al., 2017b) , which is randomly holding out 5,000 from the 50,000 training images to form the validation set.

For each budget, we report the best validation accuracy among epochs up till the end of the budget.

We use ResNet-18 (He et al., 2016) as the backbone architecture and utilize SGD with base learning rate 0.1, momentum 0.9, weight decay 0.0005 and a batch size 128.

We study learning schedules in several groups: (a) constant (equivalent to not using any schedule).

(b) & (c) exponential and step decay, both of which are commonly adopted schedules.

(d) htd (Hsueh et al., 2019) , a quite recent addition and not yet adopted in practice .

We take the parameters with the best-reported performance (???6, 3).

Note that this schedule decays much slower initially than the linear schedule (Fig 2) .

(e) the smooth-decaying schedules (small curvature), which consists of cosine (Loshchilov & Hutter, 2017) , poly (Jia et al., 2014 ) and the linear schedule.

As shown in Tab 2, the group of schedules that are budget-aware by our definition, outperform other schedules under all budgets.

The linear schedule in particular, performs best most of the time including the typical full budget case.

Noticeably, when exponential schedule is well-tuned for this task (?? = 0.97), it fails to generalize across budgets.

In comparison, the budget-aware group does not require tuning but generalizes much better.

Within the budget-aware schedules, cosine, poly and linear achieve very similar results.

This is expected due to the fact that their numerical similarity at each step (Fig 2) .

These results might indicate that the key for a robust budgeted-schedule is to decay smoothly to zero.

Based on these observations and results, we suggest linear schedule should be the "go-to" budget-aware schedule.

In the previous section, we showed that linear schedule achieves excellent performance on CIFAR-10, in a relatively toy setting.

In this section, we study the comparison and its generalization to practical large scale datasets with various state-of-the-art architectures.

In particular, we set up experiments to validate the performance of linear schedule across tasks and budgets.

Ideally, one would like to see the performance of all schedules in Fig 2 on vision benchmarks.

Due to resource constraints, we include only the off-the-shelf step decay and the linear schedule.

Note our CIFAR-10 experiment suggests that using cosine and poly will achieve similar performance as linear, which are already budget-aware schedules given our definition, so we focus on linear schedule in this section.

More evaluation between cosine, poly and linear can be found in Appendix A & D.

We consider the following suite of benchmarks spanning many flagship vision challenges:

Image classification on ImageNet.

ImageNet (Russakovsky et al., 2015) is a widely adopted standard for image classification task.

We use ResNet-18 (He et al., 2016) and report the top-1 accuracy on the validation set with the best epoch.

We follow the step decay schedule used in (Huang et al., 2017b; PyTorch, 2019) , which drops twice at uniform interval (?? = 0.1 at p ??? { 1 3 , 2 3 }).

We set the full budget to 100 epochs (10 epochs longer than typical) for easier computation of the budget.

Table 3 : Robustness of linear schedule across budgets, tasks and architectures.

Linear schedule significantly outperforms step decay given limited budgets.

Note that the off-the-shelf decay for each dataset has different parameters optimized for the specific dataset.

For all step decay schedules, BAC (Sec 3.1) is applied to boost their budgeted performance.

To reduce stochastic noise, we report the average and the standard deviation of 3 independent runs.

See Sec 4.2 for the metrics of each task (the higher the better for all tasks).

Object detection and instance segmentation on MS COCO.

MS COCO (Lin et al., 2014 ) is a widely recognized benchmark for object detection and instance segmentation.

We use the standard COCO AP (averaged over IoU thresholds) metric for evaluating bounding box output and instance mask output.

The AP of the final model on the validation set is reported in our experiment.

We use the challenge winner Mask R-CNN (He et al., 2017 ) with a ResNet-50 backbone and follow its setup.

For training, we adopt the 1x schedule (90k iterations), and the off-the-shelf (He et al., 2017) step decay that drops 2 times with ?? = 0.1 at p ??? { 2 3 , 8 9 }.

Semantic segmentation on Cityscapes.

Cityscapes (Cordts et al., 2016 ) is a dataset commonly used for evaluating semantic segmentation algorithms.

It contains high quality pixel-level annotations of 5k images in urban scenarios.

The default evaluation metric is the mIoU (averaged across class) of the output segmentation map.

We use state-of-the-art model PSPNet (Zhao et al., 2017 ) with a ResNet-50 backbone and the full budget is 400 epochs as in standard set up.

The mIoU of the best epoch is reported.

Interestingly, unlike other tasks in this series, this model by default uses the poly schedule.

For complete evaluation, we add step decay that is the same in our ImageNet experiment in Tab 3 and include the off-the-shelf poly schedule in Tab E.

Video classification on Kinetics with I3D.

Kinetics (Kay et al., 2017 ) is a large-scale dataset of YouTube videos focusing on human actions.

We use the 400-category version of the dataset and a variant of I3D (Carreira & Zisserman, 2017) with training and data processing code publicly available (Wang et al., 2018) .

The top-1 accuracy of the final model is used for evaluating the performance.

We follow the 4-GPU 300k iteration schedule (Wang et al., 2018) , which features a step decay that drops 2 times with ?? = 0.1 at p ??? { 1 2 , 5 6 }.

If we factor in the dimension of budgets, Tab 3 shows a clear advantage of linear schedule over step decay.

For example, on ImageNet, linear achieves 51.5% improvement at 1% of the budget.

Next, we consider the full budget setting, where we simply swap out the off-the-shelf schedule with linear schedule.

We observe better (video classification) or comparable (other tasks) performance after the swap.

This is surprising given the fact that linear schedule is parameter-free and thus not optimized for the particular task or network.

In summary, the smoothly decaying linear schedule is a simple and effective strategy for budgeted training.

It significantly outperforms traditional step decay given limited budgets, while achieving comparable performance with the normal full budget setting.

t || vanishes over time (color curves) as learning rate ?? t (black curves) decays.

The first row shows that the dynamics of full gradient norm correlate with the corresponding learning rate schedule while the second row shows that such phenomena generalize across budgets for budget-aware schedules.

Such generalization is most obvious in plot (h), which overlays the full gradient norm across different budgets.

If a schedule does not decay to 0, the gradient norm does not vanish.

For example, if we train a budget-unaware exponential schedule for 50 epochs (c), the full gradient norm at that time is around 1.5, suggesting this is a schedule with insufficient final decay of learning rate.

In this section, we summarize our empirical analysis with a desiderata of properties for effective budget-aware learning schedules.

We highlight those are inconsistent with conventional wisdom and follow the experimental setup in Sec 4.1 unless otherwise stated.

???F (x i , y i ).

We empirically find that the dynamics of ||g * t || over time highly correlates with the learning rate ?? t (Fig 3) .

As the learning rate vanishes for budget-aware schedules, so does the gradient magnitude.

We call this "vanishing gradient" phenomenon budgeted convergence.

This correlation suggests that decaying schedules to near-zero rates (and using BAC) may be more effective than early stopping.

As a side note, budgeted convergence resonates with classic literature that argues that SGD behaves similar to simulated annealing (Bottou, 1991) .

Given that ?? t and ||g * t || decrease, the overall update ||????? t g t || also decreases 4 .

In other words, large moves are more likely given large learning rates in the beginning, while small moves are more likely given small learning rates in the end.

However, the exact mechanism by which the learning rate influences the gradient magnitude remains unclear.

Desideratum: don't waste the budget.

Common machine learning practise often produces multiple checkpointed models during a training run, where a validation set is used to select the best one.

Such additional optimization is wasteful in our budgeted setting.

Tab 4 summarizes the progress point at which the best model tends to be found.

Step decay produces an optimal model somewhat towards the end of the training, while linear and poly are almost always optimal at the precise end of the training.

This is especially helpful for state-of-the-art models where evaluation can be expensive.

For example, validation for Kinetics video classification takes several hours.

Budget-aware schedules require validation on only the last few epochs, saving additional compute.

Table 4 : Where does one expect to find the model with the highest validation accuracy within the training progress?

Here we show the best checkpoint location measured in training progress p and averaged for each schedule across budgets greater or equal than 10% and 3 different runs.

Aggressive early descent.

Guided by asymptotic convergence analysis, faster descent of the objective might be an apparent desideratum of an optimizer.

Many prior optimization methods explicitly call for faster decrease of the objective (Kingma & Ba, 2015; Clevert et al., 2016; Reddi et al., 2018) .

In contrast, we find that one should not employ aggressive early descent because large learning rates can prevent budgeted convergence.

Consider AMSGrad (Reddi et al., 2018) , an adaptive learning rate that addresses a convergence issue with the widely-used Adam optimizer (Kingma & Ba, 2015) .

Fig 4 shows that while AMSGrad does quickly descend over the training objective, it still underperforms budget-aware linear schedules over any given training budget.

To examine why, we derive the equivalent rate ?? t for AMSGrad (Appendix B) and show that it is dramatically larger than our defaults, suggesting the optimizer is too aggressive.

We include more adaptive methods for evaluation in Appendix E.

Warm restarts.

SGDR (Loshchilov & Hutter, 2017 ) explores periodic schedules, in which each period is a cosine scaling.

The schedule is intended to escape local minima, but its effectiveness has been questioned (Gotmare et al., 2019).

Fig 5 shows that SDGR has faster descent but is inferior to budget-aware schedules for any budget (similar to the adaptive optimizers above).

Additional comparisons can be found in Appendix F. Whether there exists a method that achieves promising anytime performance and budgeted performance at the same time remains an open question. (Loshchilov & Hutter, 2017 ) with linear schedules.

(a) SGDR makes slightly faster initial descent of the training loss, but is surpassed at each given budget by the linear schedule.

(b) for SGDR, the correlation between full gradient norm ||g * t || and learning rate ??t is also observed.

Warm restart does not help to achieve better budgeted performance.

This paper introduces a formal setting for budgeted training.

Under this setup, we observe that a simple linear schedule, or any other smooth-decaying schedules can achieve much better performance.

Moreover, the linear schedule even offers comparable performance on existing visual recognition tasks for the typical full budget case.

In addition, we analyze the intriguing properties of learning rate schedules under budgeted training.

We find that the learning rate schedule controls the gradient magnitude regardless of training stage.

This further suggests that SGD behaves like simulated annealing and the purpose of a learning rate schedule is to control the stage of optimization.

In the main text, we list neural architecture search as an application of budgeted training.

Due to resource constraint, these methods usually train models with a small budget (10-25 epochs) to evaluate their relative performance (Cao et al., 2019; Cai et al., 2018; Real et al., 2019) .

Under this setting, the goal is to rank the performance of different architectures instead of obtaining the best possible accuracy as in the regular case of budgeted training.

Then one could ask the question that whether budgeted training techniques help in better predicting the relative rank.

Unfortunately, budgeted training has not been studied or discussed in the neural architecture search literature, it is unknown how well models only trained with 10 epochs can tell the relative performance of the same ones that are trained with 200 epochs.

Here we conduct a controlled experiment and show that proper adjustment of learning schedule, specifically the linear schedule, indeed improves the accuracy of rank prediction.

We adapt the code in (Cao et al., 2019) to generate 100 random architectures, which are obtained by random modifications (adding skip connection, removing layer, changing filter numbers) on top of ResNet-18 (He et al., 2017) .

First, we train these architectures on CIFAR-10 given full budget (200 epochs), following the setting described in Sec 4.1.

This produces a relative rank between all pairs of random architectures based on the validation accuracy and this rank is considered as the target to predict given limited budget.

Next, every random architecture is trained with various learning schedules under various small budgets.

For each schedule and each budget, this generates a complete rank.

We treat this rank as the prediction and compare it with the target full-budget rank.

The metric we adopt is Kendall's rank correlation coefficient (?? ), a standard statistics metric for measuring rank similarity.

It is based on counting the inversion pairs in the two ranks and (?? + 1)/2 is approximately the probability of estimating the rank correctly for a pair.

We consider the following schedules: (1) constant, it might be possible that no learning rate schedule is required if only the relative performance is considered.

(2) step decay (?? = 0.1, decay at p ??? { The results suggest that with more budget, we can better estimate the full-budget rank between architectures.

And even if only relative performance is considered, learning rate decay should be applied.

Specifically, smooth-decaying schedule, such as linear or cosine, are preferred over step decay.

We list some additional details about the experiment.

To reduce stochastic noise, each configuration under both the small and full budget is repeated 3 times and the median accuracy is taken.

The fullbudget model is trained with linear schedule, similar results are expected with other schedules as evidenced by the CIFAR-10 results in the main text (Tab 2).

Among the 100 random architectures, 21 cannot be trained, the rest of 79 models have validation accuracy spanning from 0.37 to 0.94, with the distribution mass centered at 0.91.

Such skewed and widespread distribution is the typical case in neural architecture search.

We remove the 21 models that cannot be trained for our experiments.

We take the epoch with the best validation accuracy for each configuration, so the drawback of constant or step decay not having the best model at the very end does not affect this experiment (see Sec 5).

Table C : Tab B normalized by the full-budget accuracy and then averaged across architectures.

Linear schedule achieves solutions closer to their full-budget performance than the rest of schedules under small budgets.

To reinforce our claim that linear schedule generalizes across different settings, we compare budgeted performance of various schedules on random architectures generated in the previous section.

We present two versions of the results.

The first is to directly average the validation accuracy of different architecture with each schedule and under each budget (Tab B).

The second is to normalize by dividing the budgeted accuracy by the full-budget accuracy of the same architecture and then average across different architectures (Tab C).

The second version assumes all architectures enjoy equal weighting.

Under both cases, linear schedule is the most robust schedule across architectures under various budgets.

In Sec 5, we use equivalent learning rate to compare AMSGrad (Reddi et al., 2018) with momentum SGD.

Here we present the derivation for the equivalent learning rate ?? t .

Let ?? 1 , ?? 2 and be hyper-parameters, then the momentum SGD update rule is:

while the AMSGrad update rule is:

Comparing equation 4 with 10, we obtain the equivalent learning rate: Table D : Comparison with offline data subsampling.

"Subset" meets the budget constraint by randomly subsample the dataset prior to training, while "full" uses all the data, but restricting the number of iterations.

Note that budget-aware schedule is used for "full".

Note that the above equation holds per each weight.

For Fig 4a, we take the median across all dimensions as a scalar summary since it is a skewed distribution.

The mean appears to be even larger and shares the same trend as the median.

In our experiments, we use the default hyper-parameters (which also turn out to have the best validation accuracy): ??

Data subsampling is a straight-forward strategy for budgeted training and can be realized in several different ways.

In our work, we limit the number of iterations to meet the budget constraint and this effectively limits the number of data points seen during the training process.

An alternative is to construct a subsampled dataset offline, but keep the same number of training iterations.

Such construction can be done by random sampling, which might be the most effective strategy for i.i.d (independent and identically distributed) dataset.

We show in Tab D that even our baseline budgeaware step decay, together with a limitation on the iterations, can significantly outperform this offline strategy.

For the subset setting, we use the off-the-shelf step decay (step-d2) while for the full set setting, we use the same step decay but with BAC applied (Sec 3.1).

For detailed setup, we follow Sec 4.1, of the main text.

Of course, more complicated subset construction methods exist, such as core-set construction (Bachem et al., 2017) .

However, such methods usually requires a feature summary of each data point and the computation of pairwise distance, making such methods unsuitable for extremely large dataset.

In addition, note that our subsampling experiment is conducted on CIFAR-10, a well-constructed and balanced dataset, making smarter subsampling methods less advantageous.

Consequently, the result in Tab D can as well provides a reasonable estimate for other complicated subsampling methods.

In the main text, we compare linear schedule against step decay for various tasks.

However, the off-the-shelf schedule for PSPNet (Zhao et al., 2017) is poly instead of step decay.

Therefore, we include the evaluation of poly schedule on Cityscapes (Cordts et al., 2016) in Tab E. Given the similarity of poly and linear (Fig 2) , and the opposite results on CIFAR-10 and Cityscapes, it is inconclusive that one is strictly better than the other within the smooth-decaying family.

However, these smooth-decaying methods both outperform step decay given limited budgets.

Figure A: Comparison between budget-aware linear schedule and adaptive learning rate methods on CIFAR-10.

We see while adaptive learning rate methods appear to descent faster than full budget linear schedule, at each given budget, they are surpassed by the corresponding linear schedule.

In the main text we compare linear schedule with AMSGrad (Reddi et al., 2018) (the improved version over Adam (Kingma & Ba, 2015) ), we further include the classical method RMSprop (Tieleman & Hinton, 2012) and the more recent AdaBound (Luo et al., 2019) .

We tune these adaptive methods for CIFAR-10 and summarize the results in Fig A. We observe the similar conclusion that budgetaware linear schedule outperforms adaptive methods for all given budgets.

Like SGD, those adaptive learning rate methods also takes input a parameter of base learning rate, which can also be annealed using an existing schedule.

Although it is unclear why one needs to anneal an adaptive methods, we find that it in facts boosts the performance ("AMSGrad + Linear" in Fig A) .

This section provides additional evaluation to show that learning rate restart produces worse results than our proposed budgeted training techniques under budgeted setting.

In (Loshchilov & Hutter, 2017) , both a new form of decay (cosine) and the technique of learning rate restart are proposed.

To avoid confusion, we use "cosine schedule", or just "cosine", to refer to the form of decay and SGDR to a schedule of periodical cosine decays.

The comparison with cosine schedule is already included in the main text.

Here we focus on evaluating the periodical schedule.

SGDR requires two parameters to specify the periods: T 0 , the length of the first period; T mult , where i-th period has length T i = T 0 T i???1 mult .

In Fig B, we plot the off-the-shelf SGDR schedule with T 0 = 10 (epoch), T mult = 2.

The validation accuracy plot (on the right) shows that it might end at a very poor solution (0.8460) since it is not budget-aware.

Therefore, we consider two settings to compare Figure B: One issue with off-the-shelf SGDR (T 0 = 10, T mult = 2) is that it is not budget-aware and might end at a poor solution.

We convert it to a budget aware schedule by setting it to restart n times at even intervals across the budget and n = 2 is shown here (SGDR-r2).

Table G : Comparison with SGDR under budget-aware setting.

"SGDR-r1" refers to restarting learning rate once at midpoint of the training progress, and "SGDR-r2" refers to restarting twice at even interval.

linear schedule with SGDR.

The first is to compare only at the end of each period of SGDR, where budgeted convergence is observed.

The second is to convert SGDR into a budget-aware schedule by setting the schedule to restart n times at even intervals across the budget.

The results under the first and second setting is shown in Tab F and Tab G respectively.

Under both budget-aware and budget-unaware setting, linear schedule outperforms SGDR.

For detailed setup, we follow Sec 4.1, of the main text and take the median of 3 runs.

In Sec 5, we refer to validation accuracy curve for training on CIFAR-10, which we provide here in Fig C.

For convex cost surfaces, constant learning rates are guaranteed to converge when less or equal than 1/L, where L is the Lipschitz constant for the gradient of the cost function ???F (Bottou et al., 2018) .

Another well-known result ensures convergence for sequences that decay neither too fast nor too slow (Robbins & Monro, 1951) :

One common such instance in convex optimization is ?? t = ?? 0 /t.

For non-convex problems, similar results hold for convergence to a local minimum (Bottou et al., 2018) .

Unfortunately, there does not exist a theory for learning rate schedules in the context of general non-convex optimization.

Table I : Comparison between linear and step decay with different initial learning rate under full budget setting.

On one hand, we see that linear schedule outperforms step decay under various initial learning rate.

On the other hand, we see that initial learning rate is still a very important hyper-parameter that needs to be tuned even with budget-aware, smooth-decaying schedules.

Image classification on ImageNet.

We adapt both the network architecture (ResNet-18) and the data loader from the open source PyTorch ImageNet example 5 .

The base learning rate used is 0.1 and weight decay 5 ?? 10 ???4 .

We train using 4 GPUs with asynchronous batch normalization and batch size 128.

Video classification on Kinetics with I3D.

The 400-category version of the dataset is used in the evaluation.

We use an open source codebase 6 that has training and data processing code publicly available.

Note that the codebase implements a variant of standard I3D (Carreira & Zisserman, 2017) that has ResNet as the backbone.

We follow the configuration of run i3d baseline 300k 4gpu.sh, which specifies a base learning rate 0.005 and a weight decay 10 ???4 .

Only learning rate schedule is modified in our experiments.

We train using 4 GPUs with asynchronous batch normalization and batch size 32.

Object detection and instance segmentation on MS COCO.

We use the open source implementation of Mask R-CNN 7 , which is a PyTorch re-implementation of the official codebase Detectron in the Caffe 2 framework.

We only modify the part of the code for learning rate schedule.

The codebase sets base learning rate to 0.02 and weight decay 10 ???4 .

We train with 8 GPUs (batch size 16) and keep the built-in learning rate warm up mechanism, which is an implementation technique that increases learning rate for 0.5k iterations and is intended for stabilizing the initial phase of multi-GPU training (Goyal et al., 2017) .

The 0.5k iterations are kept fixed for all budgets and learning rate decay is applied to the rest of the training progress.

Semantic segmentation on Cityscapes.

We adapt a PyTorch codebase obtained from correspondence with the authors of PSPNet.

The base learning rate is set to 0.01 with weight decay 10 ???4 .

The training time augmentation includes random resize, crop, rotation, horizontal flip and Gaussian blur.

We use patch-based testing time augmentation, which cuts the input image to patches of 713 ?? 713 and processes each patch independently and then tiles the patches to form a single output.

For overlapped regions, the average logits of two patches are taken.

We train using 4 GPUs with synchronous batch normalization and batch size 12.

@highlight

Introduce a formal setting for budgeted training and propose a budget-aware linear learning rate schedule

@highlight

This work presents a technique for tuning the learning rate for Neural Network training when under a fixed number of epochs.

@highlight

This paper analyzed which learning rate schedule should be used when the number of iteration is limited using an introduced concept of BAS (Budget-Aware Schedule).