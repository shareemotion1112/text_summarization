Deep learning models are vulnerable to adversarial examples crafted by applying human-imperceptible perturbations on benign inputs.

However, under the black-box setting, most existing adversaries often have a poor transferability to attack other defense models.

In this work, from the perspective of regarding the adversarial example generation as an optimization process, we propose two new methods to improve the transferability of adversarial examples, namely Nesterov Iterative Fast Gradient Sign Method (NI-FGSM) and Scale-Invariant attack Method (SIM).

NI-FGSM aims to adapt Nesterov accelerated gradient into the iterative attacks so as to effectively look ahead and improve the transferability of adversarial examples.

While SIM is based on our discovery on the scale-invariant property of deep learning models, for which we leverage to optimize the adversarial perturbations over the scale copies of the input images so as to avoid "overfitting” on the white-box model being attacked and generate more transferable adversarial examples.

NI-FGSM and SIM can be naturally integrated to build a robust gradient-based attack to generate more transferable adversarial examples against the defense models.

Empirical results on ImageNet dataset demonstrate that our attack methods exhibit higher transferability and achieve higher attack success rates than state-of-the-art gradient-based attacks.

Deep learning models have been shown to be vulnerable to adversarial examples Szegedy et al., 2014) , which are generated by applying human-imperceptible perturbations on benign input to result in the misclassification.

In addition, adversarial examples have an intriguing property of transferability, where adversarial examples crafted by the current model can also fool other unknown models.

As adversarial examples can help identify the robustness of models (Arnab et al., 2018) , as well as improve the robustness of models by adversarial training , learning how to generate adversarial examples with high transferability is important and has gained increasing attentions in the literature.

Several gradient-based attacks have been proposed to generate adversarial examples, such as onestep attacks and iterative attacks (Kurakin et al., 2016; .

Under the white-box setting, with the knowledge of the current model, existing attacks can achieve high success rates.

However, they often exhibit low success rates under the black-box setting, especially for models with defense mechanism, such as adversarial training (Madry et al., 2018; and input modification Xie et al., 2018) .

Under the black-box setting, most existing attacks fail to generate robust adversarial examples against defense models.

In this work, by regarding the adversarial example generation process as an optimization process, we propose two new methods to improve the transferability of adversarial examples: Nesterov Iterative Fast Gradient Sign Method (NI-FGSM) and Scale-Invariant attack Method (SIM).

• Inspired by the fact that Nesterov accelerated gradient (Nesterov, 1983 ) is superior to momentum for conventionally optimization (Sutskever et al., 2013) , we adapt Nesterov accelerated gradient into the iterative gradient-based attack, so as to effectively look ahead and improve the transferability of adversarial examples.

We expect that NI-FGSM could replace the momentum iterative gradient-based method in the gradient accumulating portion and yield higher performance.

• Besides, we discover that deep learning models have the scale-invariant property, and propose a Scale-Invariant attack Method (SIM) to improve the transferability of adversarial examples by optimizing the adversarial perturbations over the scale copies of the input images.

SIM can avoid "overfitting" on the white-box model being attacked and generate more transferable adversarial examples against other black-box models.

• We found that combining our NI-FGSM and SIM with existing gradient-based attack methods (e.g., diverse input method (Xie et al., 2019) ) can further boost the attack success rates of adversarial examples.

Extensive experiments on the ImageNet dataset (Russakovsky et al., 2015) show that our methods attack both normally trained models and adversarially trained models with higher attack success rates than existing baseline attacks.

Our best attack method, SI-NI-TI-DIM (Scale-Invariant Nesterov Iterative FGSM integrated with translation-invariant diverse input method), reaches an average success rate of 93.5% against adversarially trained models under the black-box setting.

For further demonstration, we evaluate our methods by attacking the latest robust defense methods Xie et al., 2018; Liu et al., 2019; Jia et al., 2019; Cohen et al., 2019) .

The results show that our attack methods can generate adversarial examples with higher transferability than state-of-theart gradient-based attacks.

2.1 NOTATION Let x and y true be a benign image and the corresponding true label, respectively.

Let J(x, y true ) be the loss function of the classifier (e.g. the cross-entropy loss).

Let x adv be the adversarial example of the benign image x. The goal of the non-targeted adversaries is to search an adversarial example x adv to maximize the loss J(x adv , y true ) in the p norm bounded perturbations.

To align with previous works, we focus on p = ∞ in this work to measure the distortion between x adv and x. That is x adv − x ∞ ≤ , where is the magnitude of adversarial perturbations.

Several attack methods have been proposed to generate adversarial examples.

Here we provide a brief introduction.

generates an adversarial example x adv by maximizing the loss function J(x adv , y true ) with one-step update as:

where sign(·) function restricts the perturbation in the L ∞ norm bound.

Iterative Fast Gradient Sign Method (I-FGSM).

Kurakin et al. (2016) extend FGSM to an iterative version by applying FGSM with a small step size α:

where Clip x (·) function restricts generated adversarial examples to be within the -ball of x.

Projected Gradient Descent (PGD).

PGD attack (Madry et al., 2018 ) is a strong iterative variant of FGSM.

It consists of a random start within the allowed norm ball and then follows by running several iterations of I-FGSM to generate adversarial examples.

Momentum Iterative Fast Gradient Sign Method (MI-FGSM).

integrate momentum into the iterative attack and lead to a higher transferability for adversarial examples.

Their update procedure is formalized as follows:

where g t is the accumulated gradient at iteration t, and µ is the decay factor of g t .

Diverse Input Method (DIM).

Xie et al. (2019) optimize the adversarial perturbations over the diverse transformation of the input image at each iteration.

The transformations include the random resizing and the random padding.

DIM can be naturally integrated into other gradient-based attacks to further improve the transferability of adversarial examples.

Translation-Invariant Method (TIM).

Instead of optimizing the adversarial perturbations on a single image, Dong et al. (2019) use a set of translated images to optimize the adversarial perturbations.

They further develop an efficient algorithm to calculate the gradients by convolving the gradient at untranslated images with a kernel matrix.

TIM can also be naturally integrated with other gradientbased attack methods.

The combination of TIM and DIM, namely TI-DIM, is the current strongest black-box attack method.

Carlini & Wagner attack (C&W).

C&W attack (Carlini & Wagner, 2017 ) is an optimization-based method which directly optimizes the distance between the benign examples and the adversarial examples by solving:

arg min

It is a powerful method to find adversarial examples while minimizing perturbations for white-box attacks, but it lacks the transferability for black-box attacks.

Various defense methods have been proposed to against adversarial examples, which can fall into the following two categories.

Adversarial Training.

One popular and promising defense method is adversarial training Szegedy et al., 2014; Zhai et al., 2019; Song et al., 2020) , which augments the training data by the adversarial examples in the training process.

Madry et al. (2018) develop a successful adversarial training method, which leverages the projected gradient descent (PGD) attack to generate adversarial examples.

However, this method is difficult to scale to large-scale datasets (Kurakin et al., 2017) .

Tramr et al. (2018) propose ensemble adversarial training by augmenting the training data with perturbations transferred from various models , so as to further improve the robustness against the black-box attacks.

Currently, adversarial training is still one of the best techniques to defend against adversarial attacks.

Input Modification.

The second category of defense methods aims to mitigate the effects of adversarial perturbations by modifying the input data.

Guo et al. (2018) discover that there exists a range of image transformations, which have the potential to remove adversarial perturbations while preserving the visual information of the images.

Xie et al. (2018) mitigate the adversarial effects through random transformations.

propose high-level representation guided denoiser to purify the adversarial examples.

Liu et al. (2019) propose a JPEG-based defensive compression framework to rectify adversarial examples without impacting classification accuracy on benign data.

Jia et al. (2019) leverage an end-to-end image compression model to defend adversarial examples.

Although these defense methods perform well in practice, they can not tell whether the model is truly robust to adversarial perturbations.

Cohen et al. (2019) use randomized smoothing to obtain an ImageNet classifier with certified adversarial robustness.

Similar with the process of training neural networks, the process of generating adversarial examples can also be viewed as an optimization problem.

In the optimizing phase, the white-box model being attacked to optimize the adversarial examples can be viewed as the training data on the training process.

And the adversarial examples can be viewed as the training parameters of the model.

Then in the testing phase, the black-box models to evaluate the adversarial examples can be viewed as the testing data of the model.

From the perspective of the optimization, the transferability of the adversarial examples is similar with the generalization ability of the trained models .

Thus, we can migrate the methods used to improve the generalization of models to the generation of adversarial examples, so as to improving the transferability of adversarial examples.

Many methods have been proposed to improve the generalization ability of the deep learning models, which can be split to two aspects: (1) better optimization algorithm, such as Adam optimizer(Kingma & Ba, 2014); (2) data augmentation (Simonyan & Zisserman, 2014) .

Correspondingly, the methods to improve the transferability of adversarial examples can also be split to two aspects: (1) better optimization algorithm, such as MI-FGSM, which applies the idea of momentum; (2) model augmentation (i.e., ensemble attack on multiple models), such as the work of , which considers to attack multiple models simultaneously.

Based on above analysis, we aim to improve the transferability of adversarial examples by applying the idea of Nesterov accelerated gradient for optimization and using a set of scaled images to achieve model augmentation.

Nesterov Accelerated Gradient (NAG) (Nesterov, 1983 ) is a slight variation of normal gradient descent, which can speed up the training process and improve the convergence significantly.

NAG can be viewed as an improved momentum method, which can be expressed as:

Typical gradient-based iterative attacks (e.g., I-FGSM) greedily perturb the images in the direction of the sign of the gradient at each iteration, which usually falls into poor local maxima, and shows weak transferability than single-step attacks (e.g., FGSM).

show that adopting momentum (Polyak, 1964) into attacks can stabilize the update directions, which helps to escape from poor local maxima and improve the transferability.

Compared to momentum, beyond stabilize the update directions, the anticipatory update of NAG gives previous accumulated gradient a correction that helps to effectively look ahead.

Such looking ahead property of NAG can help us escape from poor local maxima easier and faster, resulting in the improvement on transferability.

We integrate NAG into the iterative gradient-based attack to leverage the looking ahead property of NAG and build a robust adversarial attack, which we refer to as NI-FGSM (Nesterov Iterative Fast Gradient Sign Method) .

Specifically, we make a jump in the direction of previous accumulated gradients before computing the gradients in each iteration.

Start with g 0 = 0, the update procedure of NI-FGSM can be formalized as follows:

where g t denotes the accumulated gradients at the iteration t, and µ denotes the decay factor of g t .

Besides considering a better optimization algorithm for the adversaries, we can also improve the transferability of adversarial examples by model augmentation.

We first introduce a formal definition of loss-preserving transformation and model augmentation as follows.

Definition 1 Loss-preserving Transformation.

Given an input x with its ground-truth label y true and a classifier f (x) : x ∈ X → y ∈ Y with the cross-entropy loss J(x, y), if there exists an input transformation T (·) that satisfies J(T (x), y true ) ≈ J(x, y true ) for any x ∈ X , we say T (·) is a loss-preserving transformation.

Definition 2 Model Augmentation.

Given an input x with its ground-truth label y true and a model f (x) : x ∈ X → y ∈ Y with the cross-entropy loss J(x, y), if there exists a loss-preserving transformation T (·), then we derive a new model by f (x) = f (T (x)) from the original model f .

we define such derivation of models as model augmentation.

Intuitively, similar to the generalization of models that can be improved by feeding more training data, the transferability of adversarial examples can be improved by attacking more models simultaneously.

enhance the gradient-based attack by attacking an ensemble of models.

However, their approach requires training a set of different models to attack, which has a large computational cost.

Instead, in this work, we derive an ensemble of models from the original model by model augmentation, which is a simple way of obtaining multiple models via the loss-preserving transformation.

To get the loss-preserving transformation, we discover that deep neural networks might have the scale-invariant property, besides the translation invariance.

Specifically, the loss values are similar for the original and the scaled images on the same model, which is empirically validated in Section 4.2.

Thus, the scale transformation can be served as a model augmentation method.

Driven by the above analysis, we propose a Scale-Invariant attack Method (SIM), which optimizes the adversarial perturbations over the scale copies of the input image:

arg max

where S i (x) = x/2 i denotes the scale copy of the input image x with the scale factor 1/2 i , and m denotes the number of the scale copies.

With SIM, instead of training a set of models to attack, we can effectively achieve ensemble attacks on multiple models by model augmentation.

More importantly, it can help avoid "overfitting" on the white-box model being attacked and generate more transferable adversarial examples.

For the gradient processing of crafting adversarial examples, NI-FGSM introduces a better optimization algorithm to stabilize and correct the update directions at each iteration.

For the ensemble attack of crafting adversarial examples, SIM introduces model augmentation to derive multiple models to attack from a single model.

Thus, NI-FGSM and SIM can be naturally combined to build a stronger attack, which we refer to as SI-NI-FGSM (Scale-Invariant Nesterov Iterative Fast Gradient Sign Method).

The algorithm of SI-NI-FGSM attack is summarized in Algorithm 1.

In addition, SI-NI-FGSM can be integrated with DIM (Diverse Input Method), TIM (TranslationInvariant Method) and TI-DIM (Translation-Invariant with Diverse Input Method) as SI-NI-DIM, SI-NI-TIM and SI-NI-TI-DIM, respectively, to further boost the transferability of adversarial examples.

The detailed algorithms for these attack methods are provided in Appendix A.

In this section, we provide experimental evidence on the advantage of the proposed methods.

We first provide experimental setup, followed by the exploration of the scale-invariance property for deep learning models.

We then compare the results of the proposed methods with baseline methods in Section 4.3 and 4.4 on both normally trained models and adversarially trained models.

Beyond the defense models based on adversarial training, we also quantify the effectiveness of the proposed methods on other advanced defense in Section 4.5.

Additional discussions, the comparison between NI-FGSM and MI-FGSM and the comparison with classic attacks, are in Section 4.6.

Codes are available at https://github.com/JHL-HUST/SI-NI-FGSM.

Input: A clean example x with ground-truth label y true ; a classifier f with loss function J; Input: Perturbation size ; maximum iterations T ; number of scale copies m and decay factor µ. Update g t+1 by g t+1 = µ · g t +

Dataset.

We randomly choose 1000 images belonging to the 1000 categories from ILSVRC 2012 validation set, which are almost correctly classified by all the testing models.

For normally trained models, we consider Inception-v3 (Inc-v3) (Szegedy et al., 2016) , Inception-v4 (Inc-v4), Inception-Resnet-v2 (IncRes-v2) (Szegedy et al., 2017) and Resnet-v2-101 (Res-101) (He et al., 2016) .

For adversarially trained models, we consider Inc-v3 ens3 , Inc-v3 ens4 and IncRes-v2 ens (Tramr et al., 2018) .

Additionally, we include other advanced defense models: high-level representation guided denoiser (HGD) , random resizing and padding (R&P) (Xie et al., 2018) , NIPS-r3 1 , feature distillation (FD) (Liu et al., 2019) , purifying perturbations via image compression model (Comdefend) (Jia et al., 2019) and randomized smoothing (RS) (Cohen et al., 2019) .

Baselines.

We integrate our methods with DIM (Xie et al., 2019) , TIM, and TI-DIM (Dong et al., 2019) , to show the performance improvement of SI-NI-FGSM over these baselines.

Denote our SI-NI-FGSM integrated with other attacks as SI-NI-DIM, SI-NI-TIM, and SI-NI-TIM-DIM, respectively.

Hyper-parameters.

For the hyper-parameters, we follow the settings in with the maximum perturbation as = 16, number of iteration T = 16, and step size α = 1.6.

For MI-FGSM, we adopt the default decay factor µ = 1.0.

For DIM, the transformation probability is set to 0.5.

For TIM, we adopt the Gaussian kernel and the size of the kernel is set to 7 × 7.

For our SI-NI-FGSM, the number of scale copies is set to m = 5.

To validate the scale-invariant property of deep neural networks, we randomly choose 1,000 original images from ImageNet dataset and keep the scale size in the range of [0.1, 2.0] with a step size 0.1.

Then we feed the scaled images into the testing models, including Inc-v3, Inc-v4, IncRes-2, and Res-101, to get the average loss over 1,000 images.

As shown in Figure 1 , we can easily observe that the loss curves are smooth and stable when the scale size is in range [0.1, 1.3].

That is, the loss values are very similar for the original and scaled images.

So we assume that the scale-invariant property of deep models is held within [0.1, 1.3], and we leverage the scale-invariant property to optimize the adversarial perturbations over the scale copies of the input images.

In this subsection, we integrate our SI-NI-FGSM with TIM, DIM and TI-DIM, respectively, and compare the black-box attack success rates of our extensions with the baselines under single model setting.

As shown in Table 1 , our extension methods consistently outperform the baseline attacks by 10% ∼ 35% under the black-box setting, and achieve nearly 100% success rates under the white-box setting.

It indicates that SI-NI-FGSM can serve as a powerful approach to improve the transferability of adversarial examples.

Following the work of (Liu et al., 2016) , we consider to show the performance of our methods by attacking multiple models simultaneously.

Specifically, we attack an ensemble of normally trained models (including Inc-v3, Inc-v4, IncRes-v2 and Res-101) with equal ensemble weights using TIM, SI-NI-TIM, DIM, SI-NI-DIM, TI-DIM and SI-NI-TI-DIM, respectively.

As shown in Table 2 , our methods improve the attack success rates across all experiments over the baselines.

In general, our methods consistently outperform the baseline attacks by 10% ∼ 30% under the black-box setting.

Especially, SI-NI-TI-DIM, the extension by combining SI-NI-FGSM with TI-DIM, can fool the adversarially trained models with a high average success rate of 93.5%.

It indicates that these advanced adversarially trained models provide little robustness guarantee under the black-box attack of SI-NI-TI-DIM.

Besides normally trained models and adversarially trained models, we consider to quantify the effectiveness of our methods on other advanced defenses, including the top-3 defense solutions in the NIPS competition (high-level representation guided denoiser (HGD, rank-1) , random resizing and padding (R&P, rank-2) (Xie et al., 2018) and the rank-3 submission (NIPS-r3), and three recently proposed defense methods (feature distillation (FD) (Liu et al., 2019) , purifying perturbations via image compression model (Comdefend) (Jia et al., 2019) and randomized smoothing (RS) (Cohen et al., 2019) ).

We compare our SI-NI-TI-DIM with MI-FGSM , which is the top-1 attack solution in the NIPS 2017 competition, and TI-DIM (Dong et al., 2019) , which is state-of-the-art attack.

We first generate adversarial examples on the ensemble models, including Inc-v3, Inc-v4, IncResv2, and Res-101 by using MI-FGSM, TI-DIM, and SI-NI-TI-DIM, respectively.

Then, we evaluate the adversarial examples by attacking these defenses.

As shown in Table 3 , our method SI-NI-TI-DIM achieves an average attack success rate of 90.3%, surpassing state-of-the-art attacks by a large margin of 14.7%.

By solely depending on the trans- ferability of adversarial examples and attacking on the normally trained models, SI-NI-TI-DIM can fool the adversarially trained models and other advanced defense mechanism, raising a new security issue for the development of more robust deep learning models.

Some adversarial examples generated by SI-NI-TI-DIM are shown in Appendix B. .

The adversarial examples are crafted on Inc-v3 with various number of iterations ranging from 4 to 16, and then transfer to attack Inc-v4 and IncRes-v2.

As shown in Figure 2 , NI-FGSM yields higher attack success rates than MI-FGSM with the same number of iterations.

In another view, NI-FGSM needs fewer number of iterations to gain the same attack success rate of MI-FGSM.

The results not only indicate that NI-FGSM has a better transferability, but also demonstrate that with the property of looking ahead, NI-FGSM can accelerate the generation of adversarial examples.

Comparison with classic attacks.

We consider to make addition comparison with classic attacks, including FGSM (Goodfellow et al., 2014), I-FGSM (Kurakin et al., 2016) , PGD (Madry et al., 2018) and C&W (Carlini & Wagner, 2017) .

As shown in Table 4 , our methods achieve 100% attack success rate which is the same as C&W under the white-box setting, and significantly outperform other methods under the black-box setting.

In this work, we propose two new attack methods, namely Nesterov Iterative Fast Gradient Sign Method (NI-FGSM) and Scale-Invariant attack Method (SIM), to improve the transferability of adversarial examples.

NI-FGSM aims to adopt Nesterov accelerated gradient method into the gradientbased attack, and SIM aims to achieve model augmentation by leveraging the scale-invariant property of models.

NI-FGSM and SIM can be naturally combined to build a robust attack, namely SI-NI-FGSM.

Moreover, by integrating SI-NI-FGSM with the baseline attacks, we can further improve the transferability of adversarial examples.

Extensive experiments demonstrate that our methods not only yield higher success rates on adversarially trained models but also break other strong defense mechanism.

Our work of NI-FGSM suggests that other momentum methods (e.g. Adam) may also be helpful to build a strong attack, which will be our future work, and the key is how to migrate the optimization method to the gradient-based iterative attack.

Our work also shows that deep neural networks have the scale-invariant property, which we utilized to design the SIM to improve the attack transferability.

However, it is not clear why the scale-invariant property holds.

Possibly it is due to the batch normalization at each convolutional layer, that may mitigate the impact of the scale change.

We will also explore the reason more thoroughly in our future work.

The algorithm of SI-NI-TI-DIM attack is summarized in Algorithm 2.

We can get the SI-NI-DIM attack algorithm by removing Step 10 of Algorithm 2, and get the SI-NI-TIM attack algorithm by removing T (·; p) in Step 7 of Algorithm 2.

Input: A clean example x with ground-truth label y true ; a classifier f with loss function J; Input: Perturbation size ; maximum iterations T ; number of scale copies m and decay factor µ. for i = 0 to m − 1 do sum the gradients over the scale copies of the input image 7:

Get the gradients by ∇ x J(T (S i (x nes t ); p), y true ) apply random resizing and padding to the inputs with the probability p Convolve the gradients by g = W * g convolve gradient with the pre-defined kernel W

Update g t+1 by g t+1 = µ · g t +

We visualize 12 randomly selected benign images and their corresponding adversarial images in Figure 3 .

The adversarial images are crafted on the ensemble models, including Inc-v3, Inc-v4, IncRes-v2 and Res-101, using the proposed SI-NI-TI-DIM.

We see that these generated adversarial perturbations are human imperceptible.

@highlight

We proposed a Nesterov Iterative Fast Gradient Sign Method (NI-FGSM) and a Scale-Invariant attack Method (SIM) that can boost the transferability of adversarial examples for image classification.