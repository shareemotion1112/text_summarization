During the last years, a remarkable breakthrough has been made in AI domain thanks to artificial deep neural networks that achieved a great success in many machine learning tasks in computer vision, natural language processing, speech recognition, malware detection and so on.

However, they are highly vulnerable to easily crafted adversarial examples.

Many investigations have pointed out this fact and different approaches have been proposed to generate attacks while adding a limited perturbation to the original data.

The most robust known method so far is the so called C&W attack [1].

Nonetheless, a countermeasure known as fea- ture squeezing coupled with ensemble defense showed that most of these attacks can be destroyed [6].

In this paper, we present a new method we call Centered Initial Attack (CIA) whose advantage is twofold : first, it insures by construc- tion the maximum perturbation to be smaller than a threshold fixed beforehand, without the clipping process that degrades the quality of attacks.

Second, it is robust against recently introduced defenses such as feature squeezing, JPEG en- coding and even against a voting ensemble of defenses.

While its application is not limited to images, we illustrate this using five of the current best classifiers on ImageNet dataset among which two are adversarialy retrained on purpose to be robust against attacks.

With a fixed maximum perturbation of only 1.5% on any pixel, around 80% of attacks (targeted) fool the voting ensemble defense and nearly 100% when the perturbation is only 6%.

While this shows how it is difficult to defend against CIA attacks, the last section of the paper gives some guidelines to limit their impact.

Since the skyrocketing of data volumes and parallel computation capacities with GPUs during the last years, deep neural networks (DNN) have become the most effective approaches in solving many machine learning problems in several domains like computer vision, speech recognition, games playing etc.

They are even intended to be used in critical systems like autonomous vehicle BID17 , BID18 .

However, DNN as they are currently built and trained using gradient based methods, are very vulnerable to attacks a.k.a.

adversarial examples BID1 .

These examples aim to fool a classifier to make it predict the class of an input as another one, different from the real class, after bringing only a very limited perturbation to this input.

This can obviously be very dangerous when it comes to systems where human life is in stake like in self driven vehicles.

Companies IT networks and plants are also vulnerable if DNN based intrusion detection systems were to be deployed BID20 .Many approaches have been proposed to craft adversarial examples since the publication by Szegedy et al. of the first paper pointing out DNN vulnerability issue BID4 .

In their work, they generated adversarial examples using box-constrained L-BFGS.

Later in BID1 , a fast gradient sign method (FGSM) that uses gradients of a loss function to determine in which direction the pixels intensity should be changed is presented.

It is designed to be fast not optimize the loss function.

Kurakin et al. introduced in BID13 a straightforward simple improvement of this method where instead of taking a single step of size in the direction of the gradient-sign, multiple smaller steps are taken, and the result is clipped in the end.

Papernot et al. introduced in BID3 an attack, optimized under L0 distance, known as the Jacobian-based Saliency Map Attack (JSMA).

Another simple attack known as Deepfool is provided in [34] .

It is an untargeted attack technique optimized for the L2 distance metric.

It is efficient and produces closer adversarial examples than the L-BFGS approach discussed earlier.

Evolutionary algorithms are also used by authors in BID14 to find adversarial example while maintaining the attack close to the initial data.

More recently, Carlini and Wagner introduced in BID0 the most robust attack known to date as pointed out in BID5 .

They consider different optimization functions and several metrics for the maximum perturbation.

Their L2-attack defeated the most powerful defense known as distillation BID7 .

However, authors in BID6 showed that feature squeezing managed to destroy most of the C&W attacks.

Many other defenses have been published, like adversarial training BID3 , gradient masking BID9 , defenses based on uncertainty using dropout BID10 as done with Bayesian networks, based on statistics BID11 , BID12 , or principal components BID22 , BID23 .

Later, while we were carrying out our investigation, paper BID16 showed that not less than ten defense approaches, among which are the previously enumerated defenses, can be defeated by C&W attacks.

It also pointed out that feature squeezing also can be defeated but no thorough investigation actually was presented.

Another possible defense but not investigated is based on JPEG encoding when dealing with images.

It has never been explicitly attacked even after it is shown in BID13 that most attacks are countered by this defense.

Also, to our knowledge, no investigation has been conducted when dealing with ensemble defenses.

Actually, attacks transferability between models that is well investigated and demonstrated in BID19 in the presence of an oracle (requesting defense to get labels to train a substitute model) is not guaranteed at all when it is absent.

Finally, when the maximum perturbation added to the original data is strictly limited, clipping is needed at the end of training (adversarial crafting) even if C&W attacks are used.

The quality of crafted attacks is therefore degraded as the brought perturbation during the training is brutally clipped.

We tackle all these points in our work while introducing a new attack we call Centered Initial Attack (CIA).

This approach considers the perturbation limits by construction and consequently no alteration is done on the CIA resulting examples.

To make it clearer for the reader, an example is given below to illustrate the clipping issue.

FIG0 shows a comparison between CIA and C&W L2 attack before and after clipping on an example, a guitar targeted as a potpie with max perturbation equal to 4.0 (around 1.5%).

The same number of iterations FORMULA4 is considered for both methods.

As can be seen on FIG0 , CIA generates the best attack with 96% confidence.

C&W is almost as good with a score of 95% but it is degraded to 88% after applying the clipping to respect the imposed max perturbation.

Avoiding this degradation due to clipping is the core motivation of our investigation.

The remaining of this paper is organized as follows.

Section I presents some mathematical formulations and the principle of CIA strategy.

Then Section II investigates the application of CIA against ensemble defense, feature squeezing and JPEG encoding defenses.

Then Section III provides some guidelines to limit the impact of CIA attacks.

Finally, we give some possible future investigations in the conclusion.

Before entering into details of CIA, let us give some useful formulations.

A neural network can be seen as a function F (x) = y that accepts an input x and produces an output y. The function F depends actually on some model parameters often called weights an biases.

These are the variables that are adjusted during the learning process to fit the training data on one hand and generalize well to unseen data on the other hand.

Since they do not change in our models, we omit them in our notations.

The input x can be a vector or an array of any dimension.

So, without loss of generality, we consider x ??? n as it can be flattened in any case.

So, the i th component of x is noted x i with integer i ??? [1, n].Since we consider m-class classifiers, the output is calculated using the sof tmax function.

The output y = F (x) can be seen as a vector of m probabilities p j with j ??? [1, m].

The component with the biggest value gives the predicted class C(x).

This can be written as : DISPLAYFORM0 We note the output corresponding to the correct class as C c (x).

An adversarial examplex is crafted as a non targeted attack so as to get C(x) = C c (x) or a targeted one to get C(x) = t where t is the target class.

Crafting an example can then be formulated using a loss function L(F, x) to maximize the probability of getting a class different from the correct one.

Cross entropy is used in our work.

The adversarial examplex can be written asx = x + ?? where ?? is the added perturbation.

In our work we constrain ?? to be within domain [??????, ???] as considered for instance in Google Brain-Kaggle competition [16] , ??? being the maximum perturbation.

With the existing approaches, adversarial examples are generated through some iterations then clipped in the end to respect this constraint.

With C&W attacks for instance, the loss function includes a norm term ?? to minimize perturbation ?? but the clipping is still needed as we saw above.

The main idea behind Centered Initial Attack is to find for each component i the center x * i of the domain in whichx i is allowed to be (green segment on Figure.

2).

This is not trivial since we have to insure at the same time the componentx i to be within another domain [

?? i , ?? i ] to be valid, ?? i and ?? i are respectively the minimum and maximum values that can be taken by the i To find the center of domain definition ofx, three cases are to be considered actually, not four since ??? i is much smaller than (?? i ??? ?? i ), as can be seen on FIG2 .

For recall, ??? i is the i th component of ???.The three cases are: DISPLAYFORM1 DISPLAYFORM2 Now if we consider a continuous differentiable function g such that: g : ??? [???1, +1], then we can write every componentx i as: DISPLAYFORM3 This equation can be rewritten using arrays as: DISPLAYFORM4 where operator is the elementwise product and r is a new variable on which we optimize the loss function.

Finally, the loss function can be written as: DISPLAYFORM5 No constraint is to be considered on the variable r since it is well defined in domain (??????, +???).

Any initialization of r is possible but we consider it as zero for simplicity.

So, the initial attackx is therefore different from x whereas it is centered in its domain of definition (green segment).

This explains the CIA attack name.

Regarding g, many continuous functions can be used.

For instance we tried three functions: DISPLAYFORM6 Obviously other functions can be considered.

In our experiments, as they all lead to similar results, we always considered tanh.

It is interesting to note that with CIA, we can define a different maximum perturbation from a component x i to another x j .

Likewise, it is easy for instance to limit the crafting to only a portion of an image by considering a zero max perturbation on the other regions, without changing anything in the training algorithm.

This is an advantage with regard to existing approaches as it is difficult with the current machine learning frameworks to select from the same array only some variables to optimize on.

Gradients masking would be a solution but not desired as it is a clipping operation.

An example of such partial crafting is displayed on Figure.

3 where only a 50px band on the top and right sides is modified.

We generated it using ??? = 32 to make the difference visible on the paper but the image (spider) is also classified as the target (dog) even with smaller values.

Also, any gradient descent optimizer can be used to craft attacks as the case with BID0 .

Adam BID15 turns out to be the fastest in our experiments.

We used it for training and considered 20 iterations, a good compromise between computation time and attacks crafting convergence, for all the adversarial examples crafting.

To reproduce the results, the Adam hyperparameters to be considered are {learning rate = 0.2, DISPLAYFORM0 Finally, it is worth noting that CIA is not limited to images.

It can be used for any type of data with bounded continuous features.

In order to check the effectiveness of CIA attacks, we consider mainly targeted attacks as they are more difficult to craft against three different strategies of defense; ensemble defense with many classifiers, feature squeezing, and JPEG encoding.

A combination of these defenses is also considered as we will see later.

In this paper, we consider only white box attacks where we have full access to defense models parameters.

Other works BID19 pointed out the transferability property between models when it is possible to quest the defense classifier and get the labels back to train a substitute model to be used for crafting attacks.

When it is the case, an attack generated using the substitute is likely to remain an attack on the defense.

When it is not the case however, this transferability is inexistent as we will see below.

So, attacking as many models as possible at once is required.

In order to check the robustness of CIA in attacking many classifiers at once, we consider the five best classifiers on ImageNet dataset : Inception V3 (IncV3 a), Inception V4 (IncV4), InceptionResnet V2 (IncRes a), adversarialy trained Inception V3(IncV3 b) and adversarialy InceptionResnet V2 (IncRes b).

The accuracy of these classifiers is around 80% on the whole ImageNet dataset.

The accuracies on 1000 images dataset [17] we consider are showed in TAB0 .

In this experiment, we attack IncV3 b classifier and present the success rate of targeted attacks and the miss-classification rate of each classifier.

The results are showed in TAB1 .

As can be seen on TAB1 , the transferability is inexistent whatever the maximum perturbation used when looking at the targeted attacks success rate.

However, a small increase in misclassification rate is noticed especially with IncV3 a, raising from 3.9% to 7%.

This was verified when attacking any other classifier alone and checking the impact on the others.

This demonstrates clearly the need to attack many classifiers at once.

To do so, we considered an optimization using a sum of losses, each loss being related to one classifier: DISPLAYFORM0 where F i is the function relative to the i th classifier.

A weighed version can be considered to target a classifier more aggressively than another.

In our experiments, they are all attacked equally.

The results are displayed in TAB2 .

As we can see on TAB2 , the success rate of the targeted attacks against the voting ensemble is high, around 80% for ??? = 4.0 and approaching 100% for ??? = 16.0.

It is also interesting to notice that the success rate of attacking IncV3 b has decreased compared to the case when it was attacked alone.

With ??? = 4.0 for instance, it went from 97.6% to 92,5%.

This can be explained by the fact that the gradients are balanced in a way to change the input in a direction that minimizes all the losses at the same time.

As a conclusion of this section, attacking an ensemble defense using CIA is effective when we have complete access to all defense models (white box attacks).

Section 3.3 will show that the transferability to an anknown model, while attacking four among the five, is limited but not negligible (more than 30%).

This defense approach has been developed to counter attacks using smoothing filters BID6 .

The intuition behind this idea was that smoothing removes the sharp changes brought while crafting adversarial examples.

There are different feature squeezing possibilities but we consider spatial smoothing in the current study.

The other ones will be addressed in future experiments.

While adding a filter, one should care about the possible loss of accuracy of the defense classifier.

Authors in BID6 showed that a 3x3 filter is a good compromise that gives an effective defense while limiting the loss of accuracy.

We noticed it too in the current investigation and therefore considered this kernel size.

Also, different smoothing strategies are possible like Gaussian, diagonal, mean, etc.

As the results are quite similar in our experiments we carried out, we consider the mean filter in the sequel for its calculation simplicity.

The filter used for defense can actually be replaced by a convolution layer before the neural network as shown in Figure.

4.

Figure 4 : spatial smoothing modeling using a convolution layer.

As can be seen on Figure 4 , adding a convolution layer results in a new network that could be represented using a new function F .

We can therefore simply craft an adversarial example using this new function.

As a start, we conducted an attack against only one network (IncV3 a).

The results are shown in TAB3 .

As we can notice in TAB3 , the success rate of targeted attacks and the miss-classification rate are nearly 100%.

This means that the spatial smoothing based defense is not effective.

Once again, the transferability between models is inexistent.

An interesting point is to check the success of the same attacks when the defense is not actually using spatial smoothing for sure.

Said in other words, we suspect the defense to use spatial smoothing but we are not sure about it.

So, we craft attacks as before for a guarantee.

Or not!

The results are showed in TAB4 .As we can clearly see, the attacks success rate depleted to only 3.2% and the misclassification rate to 18.2%.

The attack is therefore not effective in case of filter based defense uncertainty.

This result disagrees with the intuition behind spatial smoothing as an efficient defense against attacks presented in BID6 .

As demonstrated indeed, the filtered adversarial example can be an effective attack but not the unfiltered one !

Then, how to overcome this issue for a more robust attack ?

To answer this question, we consider a hybrid network where both filtered and non filtered inputs are used for optimization as represented on Figure.

5.

The loss function to be used is given as a sum of two terms (weighing would be useful for more robustness) as follows: DISPLAYFORM0 where a, b are real positive numbers.

We conducted a new experiment using this hybrid loss function and the results are given in TAB5 .

6.3% TAB5 shows that the success rate is this time very high in both cases: 98.4% in filter based defense and nearly 100% in no filter defense.

We conclude that this attack is robust whether filtering is used or not in defense.

Another question arises from previous results given the lack of transferability of attacks between models.

What if an ensemble defense is used and filter use is uncertain ?

Once again, we consider the sum of all losses, but use the hybrid losses this time as follows: DISPLAYFORM1 where L Hi is the hybrid loss relative to the i th classifier.

The results are presented in TAB6 Even with ??? = 4.0 and considering only targeted attacks, the success and miss-classification rates are high, around 50% when all classifiers use filters and much higher (around 80%) when no filters are used.

Other experiments we conducted showed that attacks rate for filter based defense can be improved by assigning a greater weight b (twice the weight a) in the hybrid loss equation FORMULA9 .

An investigation conducted by authors in BID13 showed that most adversarial examples are countered if they are JPEG encoded before classifying them.

Lets check if it is the case with CIA attacks.

We conducted an attack against IncV3 a and classified the adversarial examples after being JPEG encoded.

The encoding uses different compression quality values Q. A higher Q means a better quality of image with a bigger size however since it undergoes less loss and compression.

The results are displayed on TAB7 .

TAB7 shows that CIA is robust when performing non targeted attacks as almost 100% of them are successful with Q = 80 and around 50% with Q = 20.

The targeted attacks are less successful with the highest score of 20% when Q = 80 and 0% when Q = 20.The result is somehow mitigated with regard to targeted attaks results.

Indeed, one has to keep in mind that a Q = 20 would not be a reasonable defense as this will degrade highly the accuracy of the classifier.

Nonetheless, we tried to improve the attacks success score by finding a suitable approximation JPEG transformations.

Obviously, JPEG encoding cannot be modeled accurately using a differentiable function that can be included in crafting examples process as we did before with spatial smoothing.

As a brief recall, this encoding implies the passage from RGB space to another color space called YCbCr where Y is brightness, Cb and Cr components represent the chrominance.

Actually, humans can see considerably more fine detail in the brightness of an image (the Y component) than in the hue and color saturation of an image (the Cb and Cr components).

Considering this fact, Cb and Cr can be downsampled by a factor of 2 or 3 without sensitive change of receptivity to human eye.

Another fact is the eye not being sensitive to sharp changes in images.

So, removing the high frequencies from the spectral space after a DCT (Discrete Cosine Transform) of an image would not affect its quality remarkably too.

These are the important facts used when making a JPEG compression.

Other steps like dividing the image into blocks, the quantization of frequency components, the encoding of these components and so on are thoroughly well documented for interested readers BID21 .

We do not take them into account.

Our idea for approximating JPEG is represented in FIG5 .As shown on FIG5 , we first transform RGB images to YCrCb space using a function T (product an sum operations) then we filter each component using a convolution layer.

Given the facts enumerated before about filtering high frequencies and down sampling the chrominance, we consider a mean 3x3 kernel for brightness component and a 6x6 kernel for Cr and Cb.

Once filtered, the result is brought back to RGB space using T ???1 before feeding the neural network.

The results of crafting attacks against IncV3 a using this approximation are given in TAB8 .

As can be noticed, the results are almost the same as those of TAB7 .

This result is a bit untriguing as if all the filters used in the approximation are inexistent.

However, a similar remark as in the filterbased defense case can be made.

The attacks crafted using JPEG encoding are no longer attacks if JPEG is not used by the defense.

This means that the considered approximation of JPEG encoding is not that bad but not enough accurate to give strong attacks.

It has obviously to be improved.

We are working on it.

As we saw previously, defending against attacks is hard as along as the defense can be modeled using a function that could be added to form a new network that can be used for crafting examples.

As a conclusion, one has to find a transformation that is hardly represented or approximated using simple functions.

This is obviously not trivial as we saw with JPEG defense.

Another, and more short term realizable defense, is to consider a big number of well performing classifiers including limited accuracy variance transformations like spatial smoothing.

As we saw in TAB6 , the success rate decreased to around 50% when using five classifiers including filters for defense.

This is not guaranteed but it worth being investigated.

For instance, when we attack all the classifiers except IncV4, the transferability is somehow limited.

Indeed, the success rate of attacks is almost 100% against the attacked classifiers whereas it is only around 34% for IncV4 as can be noticed on TAB0 .

In this paper we presented a new strategy called CIA for crafting adversarial examples while insuring the maximum perturbation added to the original data to be smaller than a fixed threshold.

We demonstrated also its robustness against some defenses, feature squeezing, ensemble defenses and even JPEG encoding.

For future work, it would be interesting to investigate the transferability of CIA attacks to the physical world as it is shown in BID13 that only a very limited amount of FGDM attacks, around 20%, survive this transfer.

Another interesting perspective is to consider partial crafting attacks while selecting regions taking into account the content of the data.

With regard to images for instance, this would be interesting to hide attacks with big but imperceptible perturbations.

<|TLDR|>

@highlight

 In this paper, a new method we call Centered Initial Attack (CIA) is provided. It insures by construction the maximum perturbation to be smaller than a threshold fixed beforehand, without the clipping process.