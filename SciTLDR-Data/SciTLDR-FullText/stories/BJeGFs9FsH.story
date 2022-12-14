There were many attempts to explain the trade-off between accuracy and adversarial robustness.

However,  there  was  no  clear  understanding  of  the  behaviors  of  a  robust  classifier  which  has human-like robustness.

We  argue  (1)  why  we  need  to  consider  adversarial  robustness  against  varying  magnitudes  of perturbations not only focusing on a fixed perturbation threshold, (2) why we need to use different method to generate adversarially perturbed samples that can be used to train a robust classifier and measure the robustness of classifiers and (3) why we need to prioritize adversarial accuracies with different magnitudes.

We introduce Lexicographical Genuine Robustness (LGR) of classifiers that combines the above requirements.

We also suggest a candidate oracle classifier called "Optimal Lexicographically Genuinely  Robust  Classifier  (OLGRC)"  that  prioritizes  accuracy  on  meaningful  adversarially perturbed  examples  generated  by  smaller  magnitude  perturbations.

The  training  algorithm  for estimating OLGRC requires lexicographical optimization unlike existing adversarial training methods.

To apply lexicographical optimization to neural network, we utilize Gradient Episodic Memory (GEM) which was originally developed for continual learning by preventing catastrophic forgetting.

Even though deep learning models have shown promising performances in image classification tasks [6] , most deep learning classifiers mis-classify imperceptibly perturbed images, i.e. adversarial examples [7] .

This vulnerability can occur even when the adversarial attacks were applied before they print the images, and the printed images were read through a camera [8] .

That result shows real-world threats of classifiers can exist.

In addition, adversarial examples for a classifier can be transferable to other models [3] .

This transferability of adversarial examples [9] enables attackers to exploit a target model with limited access to the target classifier.

This kinds of attacks is called black-box attacks.

An adversarially perturbed sample refers to the result of the perturbation (adversary generation) methods that has increased adversarial loss usually starting from an original sample.

It is important to notice that an adversarially perturbed sample of a classifier may not be an adversarial example, which will be explained later in subsection 1.1.

It can be just a non-adversarial perturbed sample (see Figure 4) .

Adversary generation methods try to effectively increase adversarial loss using the available information of the target classifier.

Methods to generate adversarially perturbed samples include Fast Gradient Sign Method (FGSM) [3] , Basic Iterative Method (BIM) [8] , Projected Gradient Descent (PGD) [10] , Distributionally Adversarial Attack (DAA) [11] and Interval Attack [12] .

We will use the following terminology for the following paragraphs unless we specify otherwise.

x is an original sample, y is corresponding label for x, is the perturbation norm, sign indicates the element-wise sign function and L(??, x, y) is the loss of the classifier parameterized by ??.

Fast Gradient Sign Method (FGSM) [3] generates the adversarial result x F GSM with the following formula.

x F GSM = x + sign [??? x L(??, x, y)].

FGSM was suggested from the hypothesis that linear behaviors of classifiers are enough to cause adversarial susceptibility of models.

The formula was obtained by applying local linearization of the cost function and finding the optimal perturbation.

Note that we only show formula for l ??? norm (max norm) attacks, but we can easily get the formula for other attacks when we replace the sign function with identity function or others.

In order to get the strongest attacks using first order information of the models, Projected gradient descent (PGD) generates the adversarial result x P GD by applying iterative steps like the following.

, y) where x + S is the set of allowed perturbation region for sample x that is limited by , x (0) is the random starting points in x + S, x+S [] means the projection result on x + S and ?? is the step size [10] .

Note that we also only show formula for l ??? norm attacks.

Basic Iterative Method (BIM) [8] use the same iterative steps with PGD method except that it start from a fixed starting point, i.e. x (0) = x for BIM.

Adversarial training [3] was developed to avoid adversarial vulnerability of a classifier.

It tries to reduce the weighted summation of standard loss (empirical risk) E [L(??, x, y)] and adversarial loss E [L(??, x , y)], i.e. ??E [L(??, x, y)] + (1 ??? ??)E [L(??, x , y)] where ?? is a hyperparameter for adversarial training, and x is an adversarially perturbed sample from x with x ??? x ??? .

(Usually, ?? = 0.5 is used for adversarial training.)

By considering both standard and adversarially perturbed samples, adversarial training try to increase accuracies on both clean 3 and adversarially perturbed samples.

In the literatures on adversarial training, inner maximization of a classifier refers to generating adversarial attacks, i.e. generating adversarially perturbed samples x * that maximally increase the loss.

And outer minimization refers to minimizing the adversarial loss of the model.

Madry et al. [10] explained that inner maximization and outer minimization of the loss can train models that are robust against adversarial attacks.

However, adversarial training [3] has shown some issues.

As some researches on adversarial robustness explained the trade-offs between accuracy on clean data and adversarial robustness [13] [14] [15] [16] , when we used adversarial training, we can get a classifier whose accuracy is lower than using standard (non-adversarial) training method [13, 17] .

Also, a research studied samples whose perceptual classes are changed due to perturbation, but not in the model's prediction, what they called "invariance-based adversarial examples" [18] .

They found that classifiers trained with adversarial training can be more susceptible to invariance-based adversarial examples.

We define three properties of human-like classification: (1) human-like classification is robust against varying magnitudes of adversarially perturbed samples and not just on a fixed maximum norm perturbations, (2) when we think about adversarially perturbed samples with increasing magnitudes, a human-like classifier does not consider already considered samples multiple times and (3) human-like classification prioritizes adversarial accuracies with smaller Figure 1 : Examples of confusing near image pairs with different classes of MNIST training dataset [20] .

The l 2 norms of the pairs are 2.399, 3.100 and 3.131 from left to right.

From these examples, we can say the exclusive belongingness assumption may not be realistic.

perturbation norm.

The objective of this paper is to design and train a classifier whose robustness is more resemblance to robustness of human than a model trained by standard adversarial training [3] .

We introduce Lexicographical Genuine Robustness (LGR) of classifiers to combine three properties.

Using LGR can prevent the problem of training classifiers with lower accuracy on clean data by considering the third property.

LGR also enable to avoid what we named "pseudo adversarial examples" of models which are conceptually similar to invariance-based adversarial examples [18] .

From LGR, we introduce a candidate oracle classifier called "Optimal Lexicographically Genuinely Robust Classifier (OLGRC)".

We know move on to more precise definition of adversarial examples and the detailed explanation of our ideas.

The definition of adversarial example by Biggio et al. [19] was used in many theoretical analysis of adversarial robustness [13] [14] [15] [16] .

These analyses showed adversarial examples are inevitable and there is a trade-off between accuracy on clean data and adversarial robustness, i.e. accuracy on adversarially perturbed samples.

However, we argue why simply increasing adversarial robustness can get a classifier whose behavior is different from humans.

Problem setting 1.

In a clean input set X ??? R d , let every sample x exclusively belong to one of the classes Y, and their classes will be denoted as c x .

A classifier f assigns a class label from Y for each sample x ??? R d .

Assume f is parameterized by ?? and L(??, x, y) is the loss of the classifier provided the input x and the label y ??? Y.

Note that this exclusive belonging assumption is introduced to simplify the analysis and it can be unrealistic.

In a real situation, 1) input information might not be enough to perfectly predict the class, 2) input samples might contain noises which erase class information, 3) some input samples might be better to give non-exclusive class (see Figure 1 ), or 4) sometimes labels might also contain some noises due to mistakes.

Definition 1.

Given a clean sample x ??? X and a maximum permutation norm (threshold) , a perturbed sample x is an adversarial example by the definition of Biggio et al. [19] if x ??? x ??? and f (x ) = c x .

Note that some perturbed samples x and some adversarial examples may also belong to X .

Although generated by adversary generation methods mentioned in subsection 1.0.1, perturbed samples are not necessarily adversarial examples (see Figure 4) .

For example, when allowed perturbation norm is too small, predicted class of adversarially perturbed samples x can be c x .

We are only focus on the analysis using l p norm for measuring the distance, but the concept of adversarial examples and our analysis are not confined to these metrics.

Many ideas in our analysis can might be applied to adversarial examples based on relaxed versions of metrics.

Let's consider the classification task on MNIST dataset [20] .

We will use the norms calculated by viewing an image input as a (flatten) 784 dimensional vector.

The smallest l 2 norm 5 of the image pair for different class on training data is 2.399.

However, as the l 2 norm of the nearest image pair for class 6 and 7 on training data is 5.485, when we train a classifier using adversarial training with = 2.399 (one can use half of the minimum distance = 2.399 2 = 1.200 to consider the distance between decision boundary and the samples in different classes, but our explanation doesn't consider that approach), the trained classifier might mis-classifies a perturbed image of digit 6 as an image of digit 7 when perturbation norm is 2.5 (> 2.399) even if such perturbation norm is smaller than the half (2.742) of the expected minimum norm 5.485.

Hence, we might want a classifier who is also robust when is larger than 2.399.

If we want a classifier which has no adversarial example when = 5.4 < 5.485, we need to have a classifier that outputs original class for every training image and perturbed images when norms of the perturbations are at most 5.4.

However, the l 2 norm of the nearest image pair for class 4 and 9 on training data is 2.830 (see Figure  2 ).

What does it mean?

As the image on the bottom left can also be considered as an adversarial example perturbed from top left, the classifier needs to classify its class is 9 when it was an original image and its class is 4 when it was an adversarial example.

Can we have a classifier with such psychometric power that knows the previous history of images?

Do we have such psychometric power?

Or, are we not robustness enough [1] even for classifying MNIST data [20] ?

The more important question we need to ask is "Do we really want such kinds of ability?".

And we answer the answer is "No!".

This confusion arises from the gap between our intuitive understanding [7] and Biggio et al. [19] 's definition of adversarial examples.

(Note that the reason we encounter these kinds of problems is not because we are doing multiclass classification.

A similar problem can occur in binary classification problems as shown in Figure 3 .)

Even though intuitive definition of adversarial examples are samples which are generated by applying imperceptible perturbations to clean samples [7] , by relying on the Biggio et al. [19] 's definition of adversarial examples, some theoretical analyses tried to analyze the adversarial robustness even when the norms of the adversarial perturbations are big enough so that they can change the perceptual classes of samples.

Definition 2.

Let us distinguish two kinds of class for a given clean sample x ??? X and its corresponding perturbed sample x .

5 l??? norm was more commonly used in the literature as l??? norm of the perturbation should be large in order to change the perceptual class [21] .

However, a classifier trained by adversarial training with l??? adversary is susceptible to adversarial attacks with l0 or l2 adversary [22] which suggests that we also need to consider l0 and l2 norm robustness.

??? De facto class: c x of the clean sample x. c x of the perturbed sample x if x ??? X .

De facto class is undefined for some perturbed sample x ??? X C = R d ??? X .

??? De jure class: c x of the clean sample x. c x of the perturbed sample x , i.e. original class of the perturbed example x .

Intuitively speaking, de facto class of a sample is current perceptual class of a sample.

The name "De jure" can be debatable and confusing, but it follows the tradition of the researchers who consider the original class of a perturbed sample as a legitimate class and try to increase robustness based on that even if the perturbation can change the de facto class.

One thing to notice is that we can change the de facto class by perturbating a clean sample x when large perturbation is allowed, but we can't change the de jure class of it.

De facto class and de jure class are not dependent on the classifier f .

Definition 3.

Furthermore, we distinguish two kinds of adversarial example x when x was the original sample of x before the adversarial perturbation.

??? Pseudo adversarial example: an adversarial example x by definition 1 whose de facto class is different from its de jure class, i.e. c x = c x .

??? Genuine adversarial example: an adversarial example x by definition 1 whose de facto class is undefined, i.e.

x ??? X C .

Note that even though the classifier f determines whether a given perturbed sample x is an adversarial example or not, it doesn't affect whether an adversarial example x is a pseudo adversarial example or genuine adversarial example.

For a given sample, the history (whether it was perturbed or not) of a sample will determine whether the sample is a clean sample x or a perturbed sample x .

For a perturbed sample x , its de jure class c x and the classifier f will determine whether it is a non-adversarial perturbed sample or an adversarial example.

Finally, the existence of de facto class will determine whether an adversarial example is a genuine or pseudo adversarial example.

With our definitions, let's consider again the classification task on MNIST dataset [20] (see Figure 2 ).

When we think about adversarial examples for = 5.4, again, the image on the bottom left can be considered as an adversarial example perturbed from the top left image.

As its de facto class is 9 no matter it's a clean or adversarial example, it is a pseudo adversarial example of top left image if it was an adversarial example.

Do we want our classifier to be robust against pseudo adversarial examples?

Short answer to this question is "No, we don't need to.".

When we consider the classification process of humans, we do not care about whether a given sample was a clean or perturbed sample, i.e. the previous history of the sample.

We only care about the most likely class of the current sample and such class is close to the concept of de facto class.

And this principle was commonly used in many visual assessment of adversarial robustness [10, 13, [23] [24] [25] even if some of them follow the definition of adversarial example of Biggio et al. [19] .

Let's consider a general situation where a classifier f tries to increase the adversarial robustness for a perturbation norm which is large enough so that the perturbation results can change the de facto classes of some samples.

In other words, the classifier tries to assign de jure class even for pseudo adversarial examples.

This implies that the classifier tries to assign perceptually wrong classes for pseudo adversarial examples who are currently equivalent to clean examples, and this will decrease accuracy on clean data without increasing human-like robustness on these samples.

Hence, not only we don't need to increase robustness against pseudo adversarial examples, but also we should avoid increasing robustness against them in order to get a model with human-like robustness (Note that robustness will be calculated by de jure classes of pseudo adversarial examples).

Let's compare the training tasks when we only have clean samples and when we only have perturbed samples.

Perturbed samples can be derived from clean samples and theoretically they can take any values in their allowed perturbation regions.

Because of that perturbed samples have more uncertainty than clean samples.

In order words, clean samples have more information than perturbed samples.

This observation can lead to a preference to prefer using clean samples when we train a model.

When we think about a training task with both clean and perturbed samples, the preference will be correspond to increasing natural accuracy before we consider the accuracy on perturbed samples.

This preference can be generalized to a principle that we prioritize the adversarial accuracy on smaller perturbation norm.

From the above explanations, we can summarize the properties of human classification or human-like robustness.

1.

Human classification is robust against adversarially perturbed samples generated from varying magnitudes of perturbations and not just fixed maximum norm perturbations.

2.

The previous history of a sample has no effect in classification.

Only the current sample will determine the classification result.

From this, a human-like classifier avoids assigning de jure class for pseudo adversarial examples.

More generally, a human-like classifier avoids considering already considered samples several times.

3.

Human classification prioritizes the robustness for smaller perturbation norm than the robustness for larger perturbation norm.

The question arising from the second property is "How do we know a given adversarial example is a pseudo adversarial example or genuine adversarial example?".

It would be trivial when we know the data distribution and predefined classes for all data like the toy example in section 2.

However, in practice, we only have limited training data and hard to know the data distribution.

We introduce a method to estimate whether a perturbed sample x has de facto class or not, and thus try to avoid using pseudo adversarial examples for adversarial training and measure the robustness of classifiers.

We then combine this with a lexicographical optimization method.

Before further diving into the adversarial robustness of classifiers, we give the mathematical definitions of the accuracies.

We define natural accuracy and adversarial accuracies for given maximum perturbation norm and exact perturbation norm.

Note that 1 () is an indicator function which has value 1 if the condition in the bracket holds and value 0 if the condition in the bracket doesn't hold.

??? Natural accuracy:

??? (Standard) Adversarial accuracy (by maximum perturbation norm):

where adversarially perturbed sample x * = argmax

??? (Standard) Adversarial accuracy (by exact perturbation norm):

where adversarially perturbed sample x * = argmax

??? Genuine adversarial accuracy (by maximum perturbation norm):

where S max ( ) = x ??? X |???x ??? X C : x ??? x ??? and adversarially perturbed sample x * = argmax

??? Genuine adversarial accuracy (by exact perturbation norm):

where S exact ( ) = x ??? X |???x ??? X C : x ??? x = and adversarially perturbed sample x * = argmax

Note that the only difference of adversarial accuracies by maximum perturbation norm and exact perturbation norm is that their allowed regions of adversarially perturbed sample x * , i.e. x : x ??? x ??? vs. x : x ??? x = .

The reason why we are separating them will be explained later.

Due to the additional requirement x ??? X C in adversarially perturbed sample x , pseudo adversarial examples will not be considered in genuine adversarial accuracy and thus give more meaningful adversarial accuracy.

Depending on X , genuine adversarial accuracies can be undefined.

In other word, genuine adversarial accuracies will be undefined when S max = ??? or S exact = ???.

Definition 5.

We define adversarial accuracy functions a : [0, ???) ??? [0, 1] for a classifier f .

These functions are defined by measuring adversarial accuracies with varying perturbation norms, but genuine adversarial accuracy function uses slightly modified formula.

??? (Standard) Adversarial accuracy function (by maximum perturbation norm):

] where x * = argmax ??? (Standard) Adversarial accuracy function (by exact perturbation norm):

] where x * = argmax

??? Genuine adversarial accuracy function (by exact perturbation norm):

previously allowed perturbation region X = x ??? R d : x ??? x < where x ??? X and x * = argmax

Likewise, the only difference of adversarial accuracy functions by maximum perturbation norm and exact perturbation norm is that their allowed regions of adversarially perturbed sample x * .

Adversarial accuracy function will be also called the change of adversarial accuracy.

Genuine adversarial accuracy function will be conventionally also called the change of genuine adversarial accuracy even if it is not strictly correct.

We don't define genuine adversarial accuracy function by maximum perturbation norm.

One thing to notice in the S exact ( ) in the definition of genuine adversarial accrucacy function is that it useX , i.e. the closure of X .

The reason we are usingX instead of X will be explained in subsection 2.2.

The additional requirement used in genuine adversarial accuracy function was x ??? X C = R d ??? X rather than x ??? X C .

It is because we consider the situation where we continuously increase the exact perturbation norm and we want to ignore already considered points for calculation of adversarial accuracy with smaller perturbation norm.

This can also be considered as using samples in previously allowed perturbation region X as a new clean input set X = X .

Let's think about a toy example (see Figure 5 ) with predefined (pre-known) classes in order to simplify the analysis.

There are only two classes ???1 and 1, i.e. Y = {???1, 1}, and

, i.e. we assume uniform prior probability.

Let's define three classifiers f 1 , f 2 and f 3 for this toy example (see Figure 6 ).

When step function step(x) is defined

Notice that natural accuracy for all three classifiers is 1.

We now explain the change of adversarial accuracy for f 1 (x) by exact perturbation norm (see top right of Figure  7 ).

When 0 < ??? 1, we can change the predicted class for x ??? [1, 1 + ) by subtracting , and we can't change the predicted class for x / ??? [1, 1 + ), thus standard adversarial accuracy will be 1 ??? 1 2 .

When 1 < ??? 2, there will be same amount of adversarial examples with = 1, thus (standard) adversarial accuracy will be 1 ??? 1 2 = 1 2 .

When 2 < ??? 3, we can still change the predicted class for x ??? [1, 2) by subtracting .

Addition to that we can also change the predicted class for x ??? [1 ??? , ???1) by adding and (standard) adversarial accuracy will be ???

where step(x) = 1 for x ??? 0 and step(x) = ???1 for x < 0.

Top: Change of (standard) adversarial accuracy for f 1 (x) by maximum perturbation norm (left) and exact perturbation norm x (right) where x = x ??? x, Middle: Change of adversarial accuracy for f 2 (x) by maximum perturbation norm (left) and exact perturbation norm x (right), Bottom: Change of adversarial accuracy for f 3 (x) by maximum perturbation norm (left) and exact perturbation norm x (right).

Observed behaviors of f 2 and f 3 will be same when we compare the adversarial accuracy by maximum perturbation norm , however, observed behaviors of f 2 and f 3 are different when we compare the adversarial accuracy by exact perturbation norm

x .

When we think about the change of adversarial accuracy for f 2 (x) by exact perturbation norm , by similar analysis, we can check it will be look like middle right graph in Figure 7 when ??? 5.

However, intriguing phenomenon occurs when > 5.

When 5 < ??? 6, x ??? [1, ??? 4) cannot change the predicted class as subtracting or adding will result in the same class 1, thus adversarial accuracy will be ???5

2 .

If ??? 6, adversarial accuracy will be The change of adversarial accuracy for f 3 (x) by exact perturbation norm can be understand similarly with f 2 (x).

Now, we move on to the explanation for the changes of genuine adversarial accuracy for f 1 (x), f 2 (x) and f 3 (x) (see Figure 8 ).

When 0 < ??? 1, previously allowed perturbation region X = (???2 ??? , ???1 + ) ??? (1 ??? , 2 + ).

When > 1, previously allowed perturbation region X = (???2 ??? , 2 + ).

For calculation of genuine adversarial accuracies, we will consider four points, i.e. S exact ( ) = {???2 ??? , ???1 + , 1 ??? , 2 + }, when 0 < ??? 1 (point 0 will be counted twice when = 1) and two points, i.e. S exact ( ) = {???2 ??? , 2 + }, when > 1.

Note that if we did not use closure in the definition of S exact ( ), S exact ( ) = {???2 ??? , 1 ??? }, when 0 < ??? 1 and S exact ( ) = {???2 ??? }, when > 1.

This will ignore many points and can not measure proper robustness of classifiers.

In the change of genuine adversarial accuracy for f 1 (x), when 0 < ??? 1, ???2 ??? , ???1 + and 2 + will be non-adversarial perturbed samples and 1 ??? will be adversarial example, and thus a gen;exact ( ) = 3 4 = 0.75.

When > 1, ???2 ??? and 2 + will be non-adversarial perturbed samples, and thus its genuine adversarial accuracy is 1.

When considering the change of genuine adversarial accuracy for f 2 (x), for 0 < < 1, ???2 ??? , ???1 + , 1 ??? and 2 + will be non-adversarial perturbed samples, and thus a gen;exact ( ) = 1.

When = 1, ???2 ??? , 1 ??? and 2 + will be non-adversarial perturbed samples and ???1 + will be adversarial example, and thus a gen;exact (1) = 3 4 = 0.75 (Actually, 1 ??? = 0 = ???1 + , but they counted twice.).

When 1 < ??? 2, ???2 ??? and 2 + will be non-adversarial perturbed samples, and thus a gen;exact ( ) = 1.

However, when > 2, only 2 + will be non-adversarial perturbed samples and ???2 ??? will be adversarial example, and thus a gen;exact ( ) = We introduce Lexicographical (Standard) Robustness (LSR or LR) which is a total preorder based on adversarial accuracy functions by the exact perturbation norm .

Furthermore, we explain why LSR is not enough to specify a human-like classifier and why we need Lexicographical Genuine Robustness (LGR).

From this, we suggest a candidate oracle classifier what we called "Optimal Lexicographically Genuinely Robust Classifier (OLGRC)".

Let's say we have two classifiers f 1 and f 2 for given data D ??? X ?? Y (Here, we are considering general classifiers and not f 1 and f 2 for our toy example.).

Let a 1 , a 2 : [0, ???) ??? [0, 1] be the corresponding standard adversarial accuracy functions by exact perturbation norm for f 1 and f 2 , respectively.

Definition 6.

We define a total preorder of classifiers called Lexicographical Standard Robustness (LSR).

??? We say "f 2 is lexicographically more robust (LR) than

??? "f 2 is lexicographically equivalently robust (LR) with f 1 " or denote "

The reason why we consider adversarial robustness against varying magnitudes of perturbations and not a fixed maximum perturbation norm is that increasing robustness on a fixed maximum perturbation norm will not give a classifier that has human-like robustness as explained in 1.2 (The first property in 1.3.).

The defined (total) preorder prioritizes the robustness for smaller perturbation norm because more information in the samples can be lost when larger perturbation is allowed, and thus adversarial accuracy for larger perturbation norm is less important (The third property in 1.3.).

This prioritization is also related to the observation that we need to avoid increasing robustness against pseudo adversarial examples who are more likely to occur when the magnitude of the perturbation is large (It is also connected to the second property in 1.3, but in an incomplete way as samples used for adversarial accuracy with small perturbation magnitude can be repeatedly used for larger perturbation magnitudes.).

Furthermore, there is also a reason for using adversarial accuracy by exact perturbation norm not by maximum perturbation norm.

That was because using adversarial accuracy by exact perturbation norm enables further discretibility as shown in Figure 7 .

Let's go back to the toy example 2 and three classifiers f 1 , f 2 and f 3 for that toy example.

According to the Lexicographical Standard Robustness (LSR), we have f 1 < LR f 3 < LR f 2 .

Then, can we say f 3 is better than f 1 , and f 2 is better than f 3 ?

Well, it is true in terms of Standard Robustness only.

However, in the following subsection 3.2, we argue why f 2 can be better than f 3 in other aspects.

One thing to note here is that if we define f 4 (x) = step(x) ??? step(x ??? 4), we can check f 2 = LR f 4 while f 2 = f 4 .

Hence, LSR doesn't have an antisymmetric property, thus it is not a partial order.

In the previous subsection, we explained that the total preorder based on Lexicographical Standard Robustness (LSR) can handle the first and third properties in subsection 1.3, but only incompletely for the second property.

To also handle the second property, we use genuine adversarial accuracy function which ignores already considered points for calculation of adversarial accuracy.

Let's say we have two classifiers f 1 and f 2 for given data D ??? X ?? Y (Again, we are referring general classifiers and not f 1 and f 2 for our toy example.).

Let a 1 , a 2 : [0, ???) ??? [0, 1] be the corresponding genuine adversarial accuracy functions by exact perturbation norm for f 1 and f 2 , respectively.

??? We say "f 2 is lexicographically genuinely more robust (LGR) than

??? "f 2 is lexicographically genuinely equivalently robust (LGR) with f 1 " or denote "

Let's go back again to the toy example 2 and classifiers f 1 , f 2 and f 3 for that toy example.

According to the Lexicographical Genuine Robustness (LGR), we have

Let's consider the perturbations needed to change the predicted classification results.

Similar to the gradients of differentiable function, the perturbations can be considered as interpretations of classifier as they can change the predicted class.

When we think about changing de facto classes, we need positive perturbation x = x ??? x ??? (2, 4) in order to change class ???1 to class 1, and we need negative perturbation x ??? (???4, ???2) in order to change class 1 to class ???1.

From this we can see that direction of the perturbations can explain the change of de facto classes.

Considering the perturbations needed to change the predicted classes for classifier f 3 , we need positive perturbation x ??? (1, ???) in order to change class ???1 to class 1, and we need negative perturbation x ??? (??????, ???1) in order to change class 1 to class ???1.

Note that direction of the perturbations can explain the change of de facto classes.

Considering the perturbations needed to change the predicted classes for classifier f 2 , we need negative perturbation x ??? (???6, ???1) in order to change class 1 to class ???1.

However, not only positive perturbation x ??? (1, ???) can change class ???1 to class 1, but also negative perturbation x ??? (??????, ???2) can change class ???1 to class 1.

Hence, the direction of the perturbations no longer explain the change of de facto classes for f 2 .

We saw that the directions of the perturbations of classifier f 3 explain more the change of de facto classes than classifier f 2 .

Also, when the Occam's razor principle was considered, we would prefer classifier f 3 over f 2 as they have same standard adversarial robustness for x ??? 4 and f 2 has one more decision boundary point than f 3 , i.e. more complex than f 3 .

Optimal Lexicographically Genuinely Robust Classifier (OLGRC) is defined as the maximal classifier based on Lexicographical Genuine Robustness (LGR), i.e. this classifier o satisfies either o = LGR g or o > LGR g for any classifier g. OLGRC is determined by expanding explored regions.

If each expansion step is (almost everywhere) uniquely determined and expansion can fill the whole space R d , there will be unique OLGRC (in almost everywhere sense).

Whether there is unique OLGRC (in almost everywhere sense) or not will be determined by the definition of the metric.

We do not cover the detailed conditions for uniqueness.

The behavior of OLGRC is similar to the behavior of the support vector machine (SVM) [26] in that its boundary tries to maximize its distance (margin) to the data points.

However, linear SVM can only be trained for linearly separable problems even if we assume exclusive belonging settings.

On the other hand, Kernel SVM tries to maximize its distance based on the norms of the feature space.

Thus, it is probably vulnerable to adversarial attacks in the input set while OLGRC tries to maximize its distance based on the norms of the input set in order to increase adversarial robustness.

When we think about the problem setting in the toy example 2, the classifier f 3 is the OLGRC as it's impossible to have a classifier whose change of genuine adversarial accuracy is higher than f 3 .

We are going to use l 1 , l 2 , ?? ?? ?? to denote loss functions in this section unlike section 1.1 which were used to represent l p norms.

As mentioned in the second properties of the human classification, we need a method that estimates whether a perturbed sample x has de facto class or not to avoid using pseudo adversarial examples in adversarial training.

To do that, we train a discriminator that is trained to distinguish clean samples and adversarially perturbed samples.

Even if its classification is incomplete because of the overlapping samples, this discriminator allows us to avoid using pseudo adversarial examples for adversarial training.

Note that this discriminator has a similar role with the discriminator in Generative Adversarial Nets [27] in that its gradients will be used to generate adversarial examples.

In our training method, we will use different magnitudes of perturbations

Then, the discriminator will assign corresponding classes for each magnitude.

As we need to estimate previously allowed perturbation region X , we provides two different inputs for each class: adversarially perturbed samples

L(??, x , c x ) and their opponents x * * = argmin

.

When we have a discriminator, we will use lexicographical optimization [2] (that will be mentioned in 4.2) to prioritize by avoid generating samples in the previously allowed perturbation region using the discriminator, i.e. to make x * ??? X C , and to make the perturbed samples adversarial.

Gradient Episodic Memory (GEM) [4] was originally developed to prevent catastrophic forgetting [5] which indicates the situation when a network was trained on some tasks, and trained on a new task after finishing the train on the previous tasks, then the network performs poorly (forget to perform well) on the previous tasks.

Gradient Episodic Memory (GEM) is a method that enables to minimize the loss for task t without increasing losses for all previous task k < t locally.

It is based on first-order approximation of the loss and angles between different loss gradients.

To our best knowledge, the lexicographical optimization [2] of neural networks was only used to avoid catastrophic forgetting in continual learning 6 [4] .

However, we argue that lexicographical optimization of neural networks is not only needed for traditional multi-task learning (MTL), but also for single task learning (STL).

Single task learning problems can be described as learning tasks that have only one target loss.

However, we often add regularization terms in the training loss in order to prevent over-fitting.

As reducing the main loss (target loss) is more important than reducing the regularization terms, we can use lexicographical optimization by prioritizing the main loss.

Progressively growing generative adversarial networks [28] uses images with different complexity.

We can also think about using lexicographical optimization in their model so that the discriminator and the generator make sure to correctly learn simple structures first.

As Lexicographical Genuine Robustness (LGR) also considers multiple accuracies with preference, it can also be considered as a problem that requires lexicographical optimization.

To better understand GEM [4] , let us assume that there are losses l 1 (??), ?? ?? ?? , l T (??) with lexicographical preference, in other words, we want to reduce l t (??) without increasing l 1 (??), ?? ?? ?? , l t???1 (??) for t ??? {1, ?? ?? ?? , T }.

We also have (pre-projection) parameter updates g 1 , ?? ?? ?? , g T where g t = ??? ???l t (??) and is a learning rate.

We locally satisfy lexicographical improvement when g t , g k ??? 0, ???k < t [4] .

If it is not satisfied, we can project g t to a nearestg t such that it satisfies g t , g k ??? 0, ???k < t, i.e.g t = argmin g,g k ???0,???k<t g t ???g 2 .

As this problem is a quadratic program (QP) problem, they suggested solving this by its dual problem and recoveringg t when t p (where p is the number of parameters in the neural network).

Unlike continual learning that only reduces loss for current task without forgetting previous tasks [4] , we will reduce multiple losses simultaneously with lexicographical preferences.

For each lexicographical training step, we can apply weights update for task 1 to task T .

But, it requires to calculateg t for each task t and require much computational complexity.

In stead of applying several small steps for different tasks, we suggest to apply only one combined weights update for each lexicographical training step.

We will call this approach as "Onestep method".

When we have suggested parameter updatesg 1 , ?? ?? ?? ,g T , let's consider their weighted meang Onestep = T t=1 ?? tgt where ?? 1 , ?? ?? ?? , ?? T ??? 0 and

It means that we can have same lexicographical training effect by simply applying the combined weights updateg Onestep .

Considering adversarial robustness for different perturbation itself is not new.

As standard accuracy can be considered as adversarial accuracy with zero perturbation, measuring standard accuracy and adversarial accuracy can be regarded as an example.

Recently, a research [29] considered the model's robustness against multiple perturbation types and suggested adversarial training schemes for multiple perturbation types.

However, their adversarial training methods ("Max" and "Avg" strategies) did not consider the different importance of adversarial accuracy with different magnitudes.

Similar concepts with pseudo adversarial examples and problems of using them for adversarial training have been studied.

The concept of pseudo adversarial example is similar to the concept called "invariance-based adversarial example" [18] whose predicted class by f is the same with the original class c x even if the predicted class by an oracle classifier o is changed.

However, their definition requires an oracle classifier o which is hard to be defined while our definition requires predefined class for samples in clean data set X , and thus it is easier to do theoretical analysis.

Invalid adversarial example [30] is also similar to pseudo adversarial example.

Their definition assumes data distribution of each class should be a manifold which limits the behavior of data distribution while we don't set manifold requirement in order to consider every possible situation.

There were some attempts to balance the accuracy of clean data and accuracy on perturbed samples of classifiers.

MixTrain [12] uses adversarial training by dynamically adjusting the hyperparameter ?? for adversarial training.

TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization) method [31] tries to minimize the difference between predicted results of adversarially perturbed samples and predicted results clean samples instead of minimizing the difference between predicted results of adversarially perturbed samples and clean labels.

To our best knowledge, there were no attempts to understand the different importance of adversarial accuracies of different magnitudes and prioritized training methods for adversarial robustness.

We also handle the problem of simply increasing standard adversarial robustness, i.e. simply finding classifiers who are lexicographically more robust (LSR) than others.

In order to compare the different training methods, we experimented with 5 different training methods: standard (non-adversarial) training, standard adversarial training [3] , TRADES [31] , OLSRC and OLGRC.

OLSRC refers to the model that trained by applying Onestep method in subsection 4.3 without applying the adversary generation method that avoids generating samples in the previously allowed perturbation region in subsection 4.1.

OLGRC refers to the model that trained by applying Onestep method in subsection 4.3 with adversary generation method that avoid generating samples in the previously allowed perturbation region in subsection 4.1.

We used PGD method [10] (using exact perturbation norms) to generate adversarially perturbed samples.

We used ADAM algorithm [32] to train the discriminator for OLGRC.

We found that using mini-batch training for lexicographical optimization might not work well and lexicographical optimization [2] would require full batch training.

As lexicographical optimization uses different weights update from a standard method (which is using more than one objective function), simply using mini-batch gradients update can result in catastrophic forgetting in other mini-batches.

In other words, even if a weights update with lexicographical optimization can improve losses for current mini-batch satisfying the lexicographical improvement, as losses functions on different mini-batch will be different from current mini-batch, the current weight update can increase losses in different mini-batch.

In order to avoid this problem, we applied full batch training in all experiments.

We did not plot for the changes of genuine adversarial accuracy in our experiments.

We think it is unnecessary to plot them for toy example 2.

It is impossible to plot the changes of genuine adversarial accuracy for the MNIST experiment as we don't know the actual data distribution and don't have predefined classes for all data.

(Note that even if we can use discriminators to estimate them, the discriminators depends on the trained classifiers and estimated changes of genuine adversarial accuracy may not be comparable.)

We randomly generated 100 training and 100 test samples from the toy example.

Fully connected neural network with one hidden layer (with 256 hidden neurons and leaky-ReLU non-linearity with parameter 0.2) was used for experiments.

Full batch training was used with learning rate of 0.015 for 1000 epochs (iterations).

Gradient descent algorithm was used for weights update.

6.1.1 The first experiment: examining the effect of lexicographical optimization [2] In order to see the effect of lexicographical optimization [2] in adversarial training, in this experiment, we only used perturbation norm 4 for adversarial attacks.

We used ?? = 0.5 for standard adversarial training [3] .

In order to apply comparable effects on the training of OLSRC, we used ?? 1 , ?? 2 = 0.5 for weights of Onestep method and 10 ???10 was used for numerical stability in GEM algorithm [4] .

We used 1 ?? = 1.0 for TRADES [31] training.

Standard (non-adversarial) training and OLGRC were not experimented.

Comparing the change of accuracies and losses by iterations in Figure 9 , we can observe that training processes of standard adversarial training [3] and TRADES [31] are not stable and classifiers can not be trained properly as both methods don't have prioritization of losses.

(It seems TRADES method is less fluctuating than standard adversarial training, but it could be because of different effect of loss function.)

On the other hand, training of OLSRC is much more stable as it prioritizes natural cross-entropy loss.

Comparing the plots for change of adversarial accuracy in Figure 10 , the final classifier obtained by standard adversarial training [3] achieved 0 for both natural accuracy and adversarial accuracy (perturbation norm: 4).

The final classifier obtained by TRADES [31] training achieved 1.0 natural accuracy and almost 0 adversarial accuracy (perturbation norm: 4).

However, it might achieve 1.0 natural accuracy by chance considering the fluctuating training accuracy.

Final OLSRC achieved 1.0 natural accuracy and about 0.5 adversarial accuracy (perturbation norm: 4).

In order to see the effect of avoiding already explored regions, in the second experiment, we used exact perturbation norms 1, 2, 3, 4, 5, 6 for adversarial attacks.

Only OLSRC and OLGRC were experimented.

We used

for weights of Onestep method and 10 ???3 was used for numerical stability in GEM algorithm [4] .

When the generated adversarially perturbed samples using discriminator were observed, we can check that the perturbation process avoids already explored regions even though it is incomplete.

For example, when = 4, 5, perturbed samples went to the right direction without making any mistake (Recall that previously allowed perturbation region X is (???2 ??? , 2 + ) when > 1, and in order to avoid already explored points, perturbed samples need to move outward.).

Estimated p(x ??? X C |x ) also roughly capture the regions that need to be explored.

Comparing the final classifiers and changes of adversarial accuracy in Figure 12 , we can observe the shape of the trained OLGRC and its changes of adversarial accuracy are quite similar to f 3 in section 2 which is the theoretical OLGRC.

Notice that it was not achievable when we only used training method for OLSRC as shown in the figure.

Finding a human-like classifier

Figure 11: Plotted graphs show estimated probabilities that the input is not in the previously allowed perturbation region, i.e. estimated p(x ??? X C |x ).

Red: [???2, ???1) and blue: [1, 2) dashed lines represent the regions for class ???1 and class 1.

Generated adversarially perturbed samples using discriminator were color plotted class ???1: red and class 1: blue.

In order to prevent catastrophic forgetting in different mini-batches in mini-batch training, we only used randomly sampled 2000 samples as training data and full batch training was used with learning rate 0.001 for 2000 epochs (iterations).

Note that our results will not be comparable with other previous analysis on MNIST data because we are using smaller training data.

For this experiment, we used common architecture with two convolution layers and two fully connected layers which can be found at https://github.com/MadryLab/mnist_challenge.

In order to speed up the training, we applied the ADAM algorithm [32] after projections were applied because of the easiness of implementation.

However, as applying adaptive gradient optimization after projections might violate lexicographical improvements, we speculate that it would be better to apply projection after adaptive gradient optimization method was applied.

We used the Projected Gradient Descent method [10] with 40 iterations to generates adversarial attacks.

Only l 2 norm 4 adversarial attack is used for adversarial training for l 2 norm robust model and only l ??? norm 0.3 adversarial attack is used for adversarial training for l ??? norm robust model.

We used ?? = 0.5 for standard adversarial training [3] .

In order to apply comparable effects on the training of OLSRC and OLGRC, we used ?? 1 , ?? 2 = 0.5 for weights of Onestep method.

We used 1 ?? = 1.0 for TRADES [31] training.

Note that due to different formulation of losses training results of TRADES will not be directly comparable with other training methods.

When we compare the results of different training methods (shown in Table 1 ), we can notice that using OLSRC and OLGRC are better than standard adversarial training [3] Table 2 : Results on test data when l 2 norm attacks were used for training and test expectation, trained OLSRC was not lexicographically more robust than trained OLGRC (even on the trained data).

It could be the result of simultaneously reducing more than one loss and applying the ADAM [32] after projections were applied.

(When it comes to natural accuracy for both experiments, TRADES [31] achieved the best result.

It could be because of different formulation of losses.

It also achieved the smallest training loss in both experiments among adversarially trained models.

Results on training data were not shown.)

In this work, we explained why existing adversarial training methods cannot train a classifier that has human-like robustness.

We identified three properties of human-like classification: (1) human-like classification should be robust against varying magnitudes of adversarially perturbed samples and not just on a fixed maximum norm perturbations, (2) when we consider robustness on increasing magnitudes of adversarial perturbations, a human-like classifier should avoid considering already considered points multiple times, and (3) human-like classification need to prioritize the robustness against adversarially perturbed samples with smaller perturbation norm.

The suggested properties explain why previous methods for adversarial training and evaluation can be incomplete.

For example, the second property explains why commonly used evaluation of adversarial robustness may not fully reveal our intuitive understanding of human-like robustness as standard adversarial accuracies don't avoid pseudo adversarial examples.

We defined a candidate oracle classifier called Optimal Lexicographically Genuinely Robust Classifier (OL-GRC).

OLGRC is (almost everywhere) uniquely determined when dataset and norm were given.

In order to train a OLGRC, we suggested a method to generate adversarially perturbed samples using a discriminator.

We proposed to use Gradient Episodic Memory (GEM) [4] for lexicographical optimization [2] and an approach to applying GEM when simultaneously reducing multiple losses with lexicographical preferences.

From the first experiment on the toy example from section 2, we showed that lexicographical optimization enables stable training even when other adversarial training methods failed to do so.

The second experiment on the same toy example showed that we can use discriminator to roughly generate adversarially perturbed samples by avoiding already explored regions.

Because of that, we could train a classifier that is similar to the theoretical OLGRC.

From the experiment on the MNIST data, we showed that our methods (OLSRC and OLGRC) achieved better performances on natural accuracy and adversarial accuracy than using standard adversarial training method [3] .

In our work, we applied GEM [4] method to adversarial training which is not traditionally a multi-task learning (MTL) problem.

This perspective also leads us to use multiobjective optimization [33] (without lexicographical preference) to the problems those were not considered as such.

For example, one can use multiobjective optimization to train a single ensemble model that reduces losses in different datasets instead of training different models separately and averaging them.

Multiobjective optimization can be used to find an efficient black-box attack by finding adversarial examples that can fool a list of models.

By replacing the calculation of an average, it can also be used to smoothen the interpretations of a model [34] .

Gradient episodic memory (GEM) [4] with standard gradient descent optimization method is slow and it needs to be combined with adaptive gradient update algorithms.

One needs to try applying adaptive gradient update algorithms before the projection was applied.

Also, GEM with mini-batch training cannot prevent not increasing losses in the other mini-batches.

It is a serious limitation in deep learning applications.

Future work needs to find a way to handle this problem.

To simplify the problem of finding a human-like classifier, we assumed the exclusive belonging which is unrealistic in many problems.

We need analysis when this assumption is violated.

We might need to consider easing the lexicographical preference as we expect to get the accuracy that is less than 1 when the exclusive belonging assumption is violated.

Another approach would be estimating the hypothetical original data which satisfies the exclusive belonging assumption.

In that approach, we consider current data are obtained by adding some input or label noises to the unknown original data.

Our training method will find a classifier that is robust against one form (l 1 , l 2 or l ??? ) of adversarial attacks with different magnitudes.

However, we need to find a classifier that is robust against many forms of adversarial attacks (including shift, rotation [35] , spatial transformation [36] , etc.) with different magnitudes as attackers can try different kinds of attacks to exploit the classifier.

Our model suggest a (almost everywhere) unique classifier that is robust against one form of adversarial robustness with some conditions.

Because of this, in order to find a classifier that is robust against many forms of adversaries, we need to define a combined metric (or its generalization).

Figure 13 , Below: Change of adversarial accuracy for f (x) by exact perturbation norm x .

Notice that f does not satisfy non-increasing Lexicographical Standard Robustness property for 2 < x < 3.

We can think of the same toy example in section 2 except for the prior probability.

For example, p(c = ???1) = .

Then, it might be more reasonable to depending on a weighted version of when we define the change of adversarial accuracy functions, i.e. it will be depending on p(c=cx) rather than depending on .

9.3 Interpretation of a classifier: Negative Adversarial Remover (NAR) Definition 8.

We define decision boundary (DB) for a classifier f and a class c ??? Y.

??? Decision boundary : DB c = x ??? R d : ???N (x), ???x 1 , x 2 ??? N (x) such that f (x 1 ) = c, f (x 2 ) = c where N (x) is a neighborhood of x.

Note that when f is calculated from an accessible differentiable function g, i.e. f (x) = argmax c???Y g(x) c , DB c is not equivalent to N B c = x ??? R d : ???N (x), ???x 1 , x 2 ??? N (x) such that g(x 1 ) c ??? p(C = c), g(x 2 ) c < p(C = c) x ??? R d : g(x) c = p(C = c) when prior is not uniform.

N B c will be called neutral boundary (NB).

Definition 9.

We define negative adversarial remover (NAR) and nearest decision boundary point (NDBP) for a classifier f , a sample x ??? R d and a class c ??? Y.

??? Negative adversarial remover : NAR c (x) = ??? argmin Note that NAR and NDBP for a sample x can be more than one points.

One can check that x = NAR c (x) + NDBP c (x).

This indicates that when f (x) = c, NAR c (x) can be an interpretation of the sample x as it is the perturbation that change a point in the decision boundary, i.e. NDBP c (x), to sample x. NDBP c (x) is also similar to the concept called baseline in Integrated Gradients interpretation method [37] while NDBP c (x) is dependent on sample x unlike baseline which will be predefined by users.

If f is calculated from an accessible differentiable function g, i.e.

f (x) = argmax c???Y g(x) c , we can use DeepFool algorithm [38] or Fast Adaptive Boundary (FAB)-attack [39] to estimate NAR c (x) when c = f (x).

If we only have f , we can use Boundary attack [40] or HopSkipJumpAttack [41] to estimate NAR c (x) when c = f (x).

@highlight

We try to design and train a classifier whose adversarial robustness is more resemblance to robustness of human.