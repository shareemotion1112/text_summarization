We explore the behavior of a standard convolutional neural net in a setting that introduces classification tasks sequentially and requires the net to master new tasks while preserving mastery of previously learned tasks.

This setting corresponds to that which human learners face as they acquire domain expertise, for example, as an individual reads a textbook chapter-by-chapter.

Through simulations involving sequences of 10 related tasks, we find reason for optimism that nets will scale well as they advance from having a single skill to becoming domain experts.

We observed two key phenomena.

First, forward facilitation---the accelerated learning of task n+1 having learned n previous tasks---grows with n. Second, backward interference---the forgetting of the n previous tasks when learning task n+1---diminishes with n.  Forward facilitation is the goal of research on metalearning, and reduced backward interference is the goal of research on ameliorating catastrophic forgetting.

We find that both of these goals are attained simply through broader exposure to a domain.

We explore the behavior of a standard convolutional neural net in a setting that introduces classification tasks sequentially and requires the net to master new tasks while preserving mastery of previously learned tasks.

This setting corresponds to that which human learners face as they acquire domain expertise, for example, as an individual reads a textbook chapter-by-chapter.

Through simulations involving sequences of 10 related tasks, we find reason for optimism that nets will scale well as they advance from having a single skill to becoming domain experts.

We observed two key phenomena.

First, forward facilitation-the accelerated learning of task n + 1 having learned n previous tasks-grows with n. Second, backward interference-the forgetting of the n previous tasks when learning task n + 1-diminishes with n. Forward facilitation is the goal of research on metalearning, and reduced backward interference is the goal of research on ameliorating catastrophic forgetting.

We find that both of these goals are attained simply through broader exposure to a domain.

In a standard supervised learning setting, neural networks are trained to perform a single task, such as classification, defined in terms of a discriminative distribution p(y | x, D) for labels y conditioned on input x given a data set D. Although such models are useful in engineering applications, they do not reflect the breadth of human intelligence, which depends on the capability to perform arbitrary tasks in a context-dependent manner.

Multitask learning BID6 is concerned with performing any one of n tasks, usually by having multiple heads on a neural network to produce outputs appropriate for each task, cast formally in terms of the distribution p(y i | x, D 1 , . . .

, D n ), where the subscript denotes a task index and i ∈ {1, . . .

, n} is an arbitrary task.

When related, multiple tasks can provide a useful inductive bias to extract shared structure (Caruana, 1993) , and as a regularization method to guide toward solutions helpful on a variety of problems BID26 .Multitask learning is typically framed in terms of simultaneous training on all tasks, but humans and artificial agents operating in naturalistic settings more typically tackle tasks sequentially and need to maintain mastery of previously learned tasks as they acquire a new one.

Consider students reading a calculus text in which each chapter presents a different method.

Early on, engaging with a chapter and its associated exercises will lead to forgetting of the material they had previously mastered.

However, as more knowledge is acquired, students learn to effectively scaffold knowledge and eventually are able to leverage prior experience to integrate the new material with the old.

As the final chapters are studied, students have built a strong conceptual framework which facilitates the integration of new material with little disruption of the old.

In this article, we study the machine-learning analog of our hypothetical students.

The punch line of the article is that a generic neural network trained sequentially to acquire and maintain mastery of multiple tasks behaves similarly to human learners, exhibiting faster acquisition of new knowledge and less disruption of previously acquired knowledge with diverse domain experience.

Early research investigating sequential training observed catastrophic forgetting BID18 , characterized by a dramatic drop in task 1 performance following training on task 2, i.e., the accuracy of the model p(y 1 | x, D 1 → D 2 ) is significantly lower than accuracy of the model p(y 1 | x, D 1 ), where the arrow denotes training sequence.

BID23 review efforts to quantify and reduce catastrophic forgetting, including specialized mechanisms that aim to facilitate sequential learning.

BID28 BID3 BID29 .

Metalearning assesses facilitation that arises on task n from having previously learned tasks 1, 2, . . . , n − 1.

Success in metalearning is measured by a reduction in training-trialsto-criterion or an increase in model accuracy given finite training for the n'th task, p(y n |x, D 1 → . . .

→ D n ), relative to the first task, p(y 1 | x, D 1 ).

Some metalearning approaches, such as MAML BID9 or SNAIL BID19 offer mechanisms to encourage transfer between tasks, while other approaches employ recurrence to modify the learning procedure itself BID0 BID30 .Catastrophic forgetting and metalearning have a complementary relationship.

Whereas catastrophic forgetting reflects backward interference of a new task on previously learned tasks, metalearning reflects forward facilitation of previously learned tasks on a new task.1 Whereas catastrophic forgetting has focused on the first task learned, metalearning has focused on the last task learned.

We thus view these two topics as endpoints of a continuum.

Surprisingly, we are not aware of any work that systematically examines these two topics in conjunction with one another.

To unify the topics, this article examines the continuum from the first task to the n'th.

We devised a setting in which we train a model on a sequence of related tasks and investigate the consequences of introducing each new task i. We measure how many training trials are required to learn the i'th task while maintaining performance on tasks 1 . . .

i−1.

Simultaneously, we measure how performance drops on tasks 1 . . .

i − 1 after introducing task i and how many trials are required to retrain tasks 1 . . .

i − 1.

We believe that examining scaling behavior-performance as a function of i-is critical to assessing the efficacy of sequential multitask learning.

Scaling behavior has been mostly overlooked in recent deep-learning research, which is odd considering its central role in computational complexity theory, and therefore, in assessing whether existing algorithms offer any home for extend to human-scale intelligence.

The tasks we train are defined over images consisting of multiple synthetic shapes having different colors and textures FIG1 ).

The tasks involve yes/no responses to questions about whether an image contains certain objects or properties, such as "is there a red object?" or "is there a spherical object?"

We generate a series consisting of 10 episodes; in each episode, a new task is introduced (more 1 In the psychology literature, backward interference is referred to as retroactive interference BID22 Postman, 1961) .

In the machine learning literature, the more general terms backward and forward transfer are sometimes used (Lopez-Paz & Ranzato, 2017) .

details to follow on the tasks).

A model is trained de novo on episode 1, and then continues training for the remaining episodes.

In episode i, training involves a mix of examples drawn from tasks 1-i until an accuracy criterion of 95% is attained on a hold-out set for all tasks.

To balance training on the newest task (task i in episode i) and retraining on previous tasks, we adapt the methodology of BID21 : half the training set consists of examples from the newest task, and the other half consists of an equal number of examples from each of the previous tasks 1 through i−1.

In episode 1, only the single task is trained.

Each epoch of training consists of one pass through each of the training images.

These images can be assigned to arbitrary tasks.

In each epoch, we roughly balance the number of yes and no target responses for each task.

We turn now from this overview to details of the images, tasks, and architecture.

Image generation.

We leverage the CLEVR (Johnson et al., 2017) image generation codebase to produce 160 × 120 pixel color images each with 4 or 5 objects that varied along three dimensions: shape, color, and texture.

To balance the dimensions, we introduced additional features in each dimension to ensure 10 feature values per dimension.

We synthesized 45,000 images for a training set, roughly balancing the count of each feature across images.

An additional 5,000 images were generated for a hold-out set.

Each image could used for any task.

Each epoch of training involved one pass through all images, with a random assignment of images to task each epoch to satisfy the constraint on the distribution of tasks.

Tasks.

For each replication of our simulation, we select one of the three dimensions and randomize the order of the ten within-dimension tasks.

To reduce sensitivity of the results to order, we performed replications using a Latin square design (Bailey, 2008, ch. 9) , guaranteeing that within a block of ten replications, each task will appear in each ordinal position exactly once.

We constructed six such Latin square blocks for each of the three dimensions, resulting in 180 total simulation replications.

Because we observed no meaningful differences across task dimensions (see Appendix), the results we report below collapse across dimension.

Architecture.

We report experiments using a basic vision architecture with four convolutional layers followed by four fully connected layers.

The convolutional layers- with 16, 32, 48, and 64 filters successively-each have 3x3 kernels with stride 1 and padding 1, followed by ReLU nonlinearities, batch normalization, and 2x2 max pooling.

The fully-connected layers have 512 units in each, also with ReLU nonlinearities.

Note that our model is generic and is not specialized for metalearning or for preventing catastrophic forgetting.

Instead of having one output head for each task, task is specified as a component of the input.

Similar to Sort-of-CLEVR BID27 , task is coded as a one-hot input vector.

Task representation is concatenated to the output of the last convolutional layer before passing it to the first fully-connected layer.

Figure 2a depicts hold-out accuracy for a newly introduced task as a function of the number of training trials.

Curve colors indicate the task's ordinal position in the series of episodes, with cyan being the first and magenta being the tenth.

Not surprisingly, task accuracy improves monotonically over training trials.

But notably, metalearning is evidenced because the accuracy of task i + 1 is strictly higher than the accuracy of task i for i > 2.

FIG2 shows the accuracy of the task introduced in the first episode (y 1 ) as it is retrained each episode.2 Not surprisingly, task accuracy improves monotonically with the number of times trained, indicating a relearning savings.

But notably, the catastrophic forgetting present in early episodes vanishes by the tenth episode.

To analyze our simulations more systematically, we remind the reader that the simulation sequence presents fiftyfive opportunities to assess learning: the task introduced in episode 1 (i.e., ordinal position 1) is trained ten times, the task introduced in episode 2 is trained nine times, and so forth, until the task introduced in episode 10, which is trained only once.

FIG2 provide two views on the amount of training to reach an accuracy criterion of 95%-the dashed line in FIG2 .

The data are plotted either as a function of the number of times a task is retrained FIG2 or as a function of the episode number FIG2 , with the curves color coded as in FIG2 .

The roughly log-log linear curves offer evidence of power-law decrease in the retraining effort required to reach criterion.

(We discuss the exception points shortly.)

Backward interference diminishes both as a function of the number of times a task is relearned FIG2 ) and the amount of domain experi-ence, as indexed by the episode number FIG2 .

Figures 2e,f show an alternative view of backward interference by plotting accuracy after a fixed amount of retraining.

The conditions that require the least number of trials to criterion FIG2 ) also achieve the highest accuracy after a small amount of training FIG2 .To examine forward facilitation, we focus on the newest task introduced, the highlighted curve in FIG2 .

Starting at the third episode, we observe forward facilitation, evidenced by both a reduced number of examples required to learn the new task, as well as higher accuracy after a fixed amount of training.

Similar forward facilitation occurs not just for the newest tasks, but even for relearning older tasks, as reflected in the black-to-copper curves.

FIG2 -and strong backward interference is observed for the old taskas indicated by the crossover of the cyan curve in Figures 2c,e.

This finding suggests that to understand properties of neural nets, we must look beyond training on just two tasks, which is often the focus of research in transfer learning and catastrophic forgetting.

We explored the behavior of a standard convolutional neural net for classification tasks in a setting that introduces tasks sequentially and requires the net to master new tasks while preserving mastery of previously learned tasks.

This setting corresponds to that which human learners face as they become experts in a domain, for example, as they read a textbook chapter by chapter.

Our network exhibits six interesting properties:1.

Forward facilitation is observed once the net has acquired sufficient expertise in the domain, as evidenced by requiring less training to learn new tasks as a function of the number of related tasks learned (see highlighted black curve in FIG2 BID8 BID8 .

5.

Training performance improves according to a power function of the number of tasks learned, controlling for experience on a task (the slope of the curves in FIG2 , and also according to a power function of the amount of training a given task has received, controlling for number of tasks learned (the slope of the curves in FIG2 ).

Power-law learning is a robust characteristic of human skill acquisition, observed on a range of behavioral measures BID20 BID7 .

6.

Catastrophic forgetting is evidenced primarily for task 1 when task 2 is learned-the canonical case studied in the literature.

However, the model becomes more robust as it acquires sufficient domain experience, and eventually the relearning effort becomes negligible (see copper curves in FIG2 ,f).

The anomalous behavior of task 2 is noteworthy, yielding a transition behavior that is perhaps analogous to the "zero one infinity" rule coined by Willem van der Poel.

We are able to identify these interesting phenomena because our simulations examined scaling behavior and not just effects of one task on a second-the typical case for studying catastrophic forgetting-or the effects of many tasks on a subsequent task-the typical case for metalearning and few-shot learning.

Studying the entire continuum from the first task to the n'th is quite revealing.

We found strong evidence for improved learning performance with broader domain expertise, and further investigation is merited.

We are beginning investigations that examine how similar tasks must be to facilitate one another: how does scaling behavior change when the tasks dimensions switch across successive episodes (e.g., from color to shape to texture)?

Our preliminary results suggest that the domain knowledge acquired is quite general and extends to other dimensions of the images.

We are also examining the scaling properties of metalearning methods that are explicitly designed to facilitate transfer.

The results presented in this article can serve as a baseline to measure the magnitude of facilitation that the specialized methods offer.

A holy grail of sorts would be to identify methods that demonstrate backward facilitation, where training on later tasks improves performance on earlier tasks, and compositional generalization BID11 BID10 BID17 , where learning the interrelationship among earlier tasks allows new tasks to be performed on the first trial.

Humans demonstrate the former under rare conditions BID1 BID12 ; the latter is common in human behavior, as when individuals are able to perform a task immediately from instruction .

277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329

@highlight

We study the behavior of a CNN as it masters new tasks while preserving mastery for previously learned tasks