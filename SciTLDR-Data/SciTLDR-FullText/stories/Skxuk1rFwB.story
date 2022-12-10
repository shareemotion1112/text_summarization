Training neural networks with verifiable robustness guarantees is challenging.

Several existing approaches utilize linear relaxation based neural network output bounds under perturbation, but they can slow down training by a factor of hundreds depending on the underlying network architectures.

Meanwhile, interval bound propagation (IBP) based training is efficient and significantly outperforms linear relaxation based methods on many tasks, yet it may suffer from stability issues since the bounds are much looser especially at the beginning of training.

In this paper, we propose a new certified adversarial training method, CROWN-IBP, by combining the fast IBP bounds in a forward bounding pass and a tight linear relaxation based bound, CROWN, in a backward bounding pass.

CROWN-IBP is computationally efficient and consistently outperforms IBP baselines on training verifiably robust neural networks.

We conduct large scale experiments on MNIST and CIFAR datasets, and outperform all previous linear relaxation and bound propagation based certified defenses in L_inf robustness.

Notably, we achieve 7.02% verified test error on MNIST at epsilon=0.3, and 66.94% on CIFAR-10 with epsilon=8/255.

The success of deep neural networks (DNNs) has motivated their deployment in some safety-critical environments, such as autonomous driving and facial recognition systems.

Applications in these areas make understanding the robustness and security of deep neural networks urgently needed, especially their resilience under malicious, finely crafted inputs.

Unfortunately, the performance of DNNs are often so brittle that even imperceptibly modified inputs, also known as adversarial examples, are able to completely break the model (Goodfellow et al., 2015; Szegedy et al., 2013) .

The robustness of DNNs under adversarial examples is well-studied from both attack (crafting powerful adversarial examples) and defence (making the model more robust) perspectives (Athalye et al., 2018; Carlini & Wagner, 2017a; b; Goodfellow et al., 2015; Madry et al., 2018; Papernot et al., 2016; Xiao et al., 2019b; 2018b; c; Eykholt et al., 2018; Chen et al., 2018; Xu et al., 2018; Zhang et al., 2019b) .

Recently, it has been shown that defending against adversarial examples is a very difficult task, especially under strong and adaptive attacks.

Early defenses such as distillation (Papernot et al., 2016) have been broken by stronger attacks like C&W (Carlini & Wagner, 2017b) .

Many defense methods have been proposed recently (Guo et al., 2018; Song et al., 2017; Buckman et al., 2018; Ma et al., 2018; Samangouei et al., 2018; Xiao et al., 2018a; 2019a) , but their robustness improvement cannot be certified -no provable guarantees can be given to verify their robustness.

In fact, most of these uncertified defenses become vulnerable under stronger attacks (Athalye et al., 2018; He et al., 2017) .

Several recent works in the literature seeking to give provable guarantees on the robustness performance, such as linear relaxations (Wong & Kolter, 2018; Mirman et al., 2018; Wang et al., 2018a; Dvijotham et al., 2018b; Weng et al., 2018; Zhang et al., 2018) , interval bound propagation (Mirman et al., 2018; Gowal et al., 2018) , ReLU stability regularization (Xiao et al., 2019c) , and distributionally robust optimization (Sinha et al., 2018) and semidefinite relaxations (Raghunathan et al., 2018a; Dvijotham et al.) .

Linear relaxations of neural networks, first proposed by Wong & Kolter (2018) , is one of the most popular categories among these certified defences.

They use the dual of linear programming or several similar approaches to provide a linear relaxation of the network (referred to as a "convex adversarial polytope") and the resulting bounds are tractable for robust optimization.

However, these methods are both computationally and memory intensive, and can increase model training time by a factor of hundreds.

On the other hand, interval bound propagation (IBP) is a simple and efficient method for training verifiable neural networks (Gowal et al., 2018) , which achieved state-of-the-art verified error on many datasets.

However, since the IBP bounds are very loose during the initial phase of training, the training procedure can be unstable and sensitive to hyperparameters.

In this paper, we first discuss the strengths and weakness of existing linear relaxation based and interval bound propagation based certified robust training methods.

Then we propose a new certified robust training method, CROWN-IBP, which marries the efficiency of IBP and the tightness of a linear relaxation based verification bound, CROWN (Zhang et al., 2018) .

CROWN-IBP bound propagation involves a IBP based fast forward bounding pass, and a tight convex relaxation based backward bounding pass (CROWN) which scales linearly with the size of neural network output and is very efficient for problems with low output dimensions.

Additional, CROWN-IBP provides flexibility for exploiting the strengths of both IBP and convex relaxation based verifiable training methods.

The efficiency, tightness and flexibility of CROWN-IBP allow it to outperform state-of-the-art methods for training verifiable neural networks with ∞ robustness under all settings on MNIST and CIFAR-10 datasets.

In our experiment, on MNIST dataset we reach 7.02% and 12.06% IBP verified error under ∞ distortions = 0.3 and = 0.4, respectively, outperforming the state-of-the-art baseline results by IBP (8.55% and 15.01%).

On CIFAR-10, at = 2 255 , CROWN-IBP decreases the verified error from 55.88% (IBP) to 46.03% and matches convex relaxation based methods; at a larger , CROWN-IBP outperforms all other methods with a noticeable margin.

Neural network robustness verification algorithms seek for upper and lower bounds of an output neuron for all possible inputs within a set S, typically a norm bounded perturbation.

Most importantly, the margins between the ground-truth class and any other classes determine model robustness.

However, it has already been shown that finding the exact output range is a non-convex problem and NP-complete (Katz et al., 2017; Weng et al., 2018) .

Therefore, recent works resorted to giving relatively tight but computationally tractable bounds of the output range with necessary relaxations of the original problem.

Many of these robustness verification approaches are based on linear relaxations of non-linear units in neural networks, including CROWN (Zhang et al., 2018) , DeepPoly (Singh et al., 2019) , Fast-Lin (Weng et al., 2018) , DeepZ (Singh et al., 2018) and Neurify (Wang et al., 2018b) .

We refer the readers to (Salman et al., 2019b ) for a comprehensive survey on this topic.

After linear relaxation, they bound the output of a neural network f i (·) by linear upper/lower hyper-planes:

where a row vector

is the product of the network weight matrices W (l) and diagonal matrices D (l) reflecting the ReLU relaxations for output neuron i; b L and b U are two bias terms unrelated to ∆x.

Additionally, Dvijotham et al. (2018c; a) ; Qin et al. (2019) solve the Lagrangian dual of verification problem; Raghunathan et al. (2018a; b); Dvijotham et al. propose semidefinite relaxations which are tighter compared to linear relaxation based methods, but computationally expensive.

Bounds on neural network local Lipschitz constant can also be used for verification (Zhang et al., 2019c; Hein & Andriushchenko, 2017) .

Besides these deterministic verification approaches, randomized smoothing can be used to certify the robustness of any model in a probabilistic manner (Cohen et al., 2019; Salman et al., 2019a; Lecuyer et al., 2018; Li et al., 2018) .

To improve the robustness of neural networks against adversarial perturbations, a natural idea is to generate adversarial examples by attacking the network and then use them to augment the training set (Kurakin et al., 2017) .

More recently, Madry et al. (2018) showed that adversarial training can be formulated as solving a minimax robust optimization problem as in (2).

Given a model with parameter θ, loss function L, and training data distribution X , the training algorithm aims to minimize the robust loss, which is defined as the maximum loss within a neighborhood {x + δ|δ ∈ S} of each data point x, leading to the following robust optimization problem:

Madry et al. (2018) proposed to use projected gradient descent (PGD) to approximately solve the inner max and then use the loss on the perturbed example x + δ to update the model.

Networks trained by this procedure achieve state-of-the-art test accuracy under strong attacks (Athalye et al., 2018; Wang et al., 2018a; Zheng et al., 2018) .

Despite being robust under strong attacks, models obtained by this PGD-based adversarial training do not have verified error guarantees.

Due to the nonconvexity of neural networks, PGD attack can only compute the lower bound of robust loss (the inner maximization problem).

Minimizing a lower bound of the inner max cannot guarantee (2) is minimized.

In other words, even if PGD-attack cannot find a perturbation with large loss, that does not mean there exists no such perturbation.

This becomes problematic in safety-critical applications since those models need certified safety.

Verifiable adversarial training methods, on the other hand, aim to obtain a network with good robustness that can be verified efficiently.

This can be done by combining adversarial training and robustness verification-instead of using PGD to find a lower bound of inner max, certified adversarial training uses a verification method to find an upper bound of the inner max, and then update the parameters based on this upper bound of robust loss.

Minimizing an upper bound of the inner max guarantees to minimize the robust loss.

There are two certified robust training methods that are related to our work and we describe them in detail below.

Dvijotham et al. (2018b) .

Since the bound propagation process of a convex adversarial polytope is too expensive, several methods were proposed to improve its efficiency, like Cauchy projection (Wong et al., 2018) and dynamic mixed training (Wang et al., 2018a) .

However, even with these speed-ups, the training process is still slow.

Also, this method may significantly reduce a model's standard accuracy (accuracy on natural, unmodified test set).

As we will demonstrate shortly, we find that this method tends to over-regularize the network during training, which is harmful for obtaining good accuracy.

Interval Bound Propagation (IBP).

Interval Bound Propagation (IBP) uses a very simple rule to compute the pre-activation outer bounds for each layer of the neural network.

Unlike linear relaxation based methods, IBP does not relax ReLU neurons and does not consider the correlations between neurons of different layers, yielding much looser bounds.

Mirman et al. (2018) proposed a variety of abstract domains to give sound over-approximations for neural networks, including the "Box/Interval Domain" (referred to as IBP in Gowal et al. (2018) ) and showed that it could scale to much larger networks than other works (Raghunathan et al., 2018a) could at the time.

Gowal et al. (2018) demonstrated that IBP could outperform many state-of-the-art results by a large margin with more precise approximations for the last linear layer and better training schemes.

However, IBP can be unstable to use and hard to tune in practice, since the bounds can be very loose especially during the initial phase of training, posing a challenge to the optimizer.

To mitigate instability, Gowal et al. (2018) use a mixture of regular and minimax robust cross-entropy loss as the model's training loss.

Notation.

We define an L-layer feed-forward neural network recursively as:

where h (0) (x) = x, n 0 represents input dimension and n L is the number of classes, σ is an elementwise activation function.

We use z to represent pre-activation neuron values and h to represent Table 1 : IBP trained models have low IBP verified errors but when verified with a typically much tighter bound, including convex adversarial polytope (CAP) (Wong et al., 2018) and CROWN (Zhang et al., 2018) , the verified errors increase significantly.

CROWN is generally tighter than convex adversarial polytope however the gap between CROWN and IBP is still large, especially at large .

We used a 4-layer CNN network for all datasets to compute these bounds.

1 post-activation neuron values.

Consider an input example x k with ground-truth label y k , we define a set of S(x k , ) = {x| x − x k ∞ ≤ } and we desire a robust network to have the property

We define element-wise upper and lower bounds for z (l) and

Verification Specifications.

Neural network verification literature typically defines a specification vector c ∈ R n L , that gives a linear combination for neural network output: c f (x).

In robustness verification, typically we set c i = 1 where i is the ground truth class label, c j = −1 where j is the attack target label and other elements in c are 0.

This represents the margin between class i and class j.

For an n L class classifier and a given label y, we define a specification matrix C ∈ R n L ×n L as:

otherwise (note that the y-th row contains all 0)

Importantly, each element in vector m := Cf (x) ∈ R n L gives us margins between class y and all other classes.

We define the lower bound of Cf (x) for all x ∈ S(x k , ) as m(x k , ), which is a very important quantity: when all elements of m(x k , ) > 0, x k is verifiably robust for any perturbation with ∞ norm less than .

m(x k , ) can be obtained by a neural network verification algorithm, such as convex adversarial polytope, IBP, or CROWN.

Additionally, Wong & Kolter (2018) showed that for cross-entropy (CE) loss:

(4) gives us the opportunity to solve the robust optimization problem (2) via minimizing this tractable upper bound of inner-max.

This guarantees that max x∈S(x k , ) L(f (x), y) is also minimized.

Interval Bound Propagation (IBP) Interval Bound Propagation (IBP) uses a simple bound propagation rule.

For the input layer we set x L ≤ x ≤ x U element-wise.

For affine layers we have:

where |W (l) | takes element-wise absolute value.

Note that h (0) = x U and h (0) = x L 2 .

And for element-wise monotonic increasing activation functions σ,

1 We implemented CROWN with efficient CNN support on GPUs: https://github.com/huanzhang12/CROWN-IBP 2 For inputs bounded with general norms, IBP can be applied as long as this norm can be converted to per-neuron intervals after the first affine layer.

For example, for p norms (1 ≤ p ≤ ∞) Hölder's inequality can be applied at the first affine layer to obtain h (1) and h (1) , and IBP rule for later layers do not change.

We found that IBP can be viewed as training a simple augmented ReLU network which is friendly to optimizers (see Appendix A for more discussions).

We also found that a network trained using IBP can obtain good verified errors when verified using IBP, but it can get much worse verified errors using linear relaxation based verification methods, including convex adversarial polytope (CAP) by Wong & Kolter (2018) (equivalently, Fast-Lin by Weng et al. (2018) ) and CROWN (Zhang et al., 2018) .

Table 1 demonstrates that this gap can be very large on large .

However, IBP is a very loose bound during the initial phase of training, which makes training unstable and hard to tune; purely using IBP frequently leads to divergence.

Gowal et al. (2018) proposed to use a schedule where is gradually increased during training, and a mixture of robust cross-entropy loss with natural cross-entropy loss as the objective to stabilize training: In Figure 1 we train a small 4-layer MNIST model and we linearly increase from 0 to 0.3 in 60 epochs.

We plot the ∞ induced norm of the 2nd CNN layer during the training process of CROWN-IBP and (Wong et al., 2018) .

The norm of weight matrix using (Wong et al., 2018) does not increase.

When becomes larger (roughly at = 0.2, epoch 40), the norm even starts to decrease slightly, indicating that the model is forced to learn smaller norm weights.

Meanwhile, the verified error also starts to ramp up possibly due to the lack of capacity.

We conjecture that linear relaxation based training over-regularizes the model, especially at a larger .

However, in CROWN-IBP, the norm of weight matrices keep increasing during the training process, and verifiable error does not significantly increase when reaches 0.3.

Another issue with current linear relaxation based training or verification methods is their high computational and memory cost, and poor scalability.

For the small network in Figure 1 , convex adversarial polytope (with 50 random Cauchy projections) is 8 times slower and takes 4 times more memory than CROWN-IBP (without using random projections).

Convex adversarial polytope scales even worse for larger networks; see Appendix J for a comparison.

Overview.

We have reviewed IBP and linear relaxation based methods above.

As shown in Gowal et al. (2018) , IBP performs well at large with much smaller verified error, and also efficiently scales to large networks; however, it can be sensitive to hyperparameters due to its very imprecise bound at the beginning phase of training.

On the other hand, linear relaxation based methods can give tighter lower bounds at the cost of high computational expenses, but it over-regularizes the network at large and forbids us to achieve good standard and verified accuracy.

We propose CROWN-IBP, a new certified defense where we optimize the following problem (θ represents the network parameters):

where our lower bound of margin m(x, ) is a combination of two bounds with different natures: IBP, and a CROWN-style bound (which will be detailed below); L is the cross-entropy loss.

Note that the combination is inside the loss function and is thus still a valid lower bound; thus (4) still holds and we are within the minimax robust optimization theoretical framework.

Similar to IBP and TRADES (Zhang et al., 2019a) , we use a mixture of natural and robust training loss with parameter κ, allowing us to explicitly trade-off between clean accuracy and verified accuracy.

In a high level, the computation of the lower bounds of CROWN-IBP (m CROWN-IBP (x, )) consists of IBP bound propagation in a forward bounding pass and CROWN-style bound propagation in a backward bounding pass.

We discuss the details of CROWN-IBP algorithm below.

Forward Bound Propagation in CROWN-IBP.

In CROWN-IBP, we first obtain z (l) and z (l) for all layers by applying (5), (6) and (7).

Then we will obtain m IBP (x, ) = z (L) (assuming C is merged into W (L) ).

The time complexity is comparable to two forward propagation passes of the network.

Linear Relaxation of ReLU neurons Given z (l) and z (l) computed in the previous step, we first check if some neurons are always active (z

where Zhang et al. (2018) propose to adaptively select α k = 1 when z

k | and 0 otherwise, which minimizes the relaxation error.

Following (10), for an input vector z (l) , we effectively replace the ReLU layer with a linear layer, giving upper or lower bounds of the output:

where D (l) and D (l) are two diagonal matrices representing the "weights" of the relaxed ReLU layer.

Other general activation functions can be supported similarly.

In the following we focus on conceptually presenting the algorithm, while more details of each term can be found in the Appendix.

Backward Bound Propagation in CROWN-IBP.

Unlike IBP, CROWN-style bounds start bounding from the last layer, so we refer to it as backward bound propagation (not to be confused with the back-propagation algorithm to obtain gradients).

Suppose we want to obtain the lower bound

(we assume the specification matrix C has been merged into

), which can be bounded linearly by Eq. (11).

CROWN-style bounds choose the lower bound of σ(z

i,k is positive, and choose the upper bound otherwise.

We then merge W (L) and the linearized ReLU layer together and define:

where b

with the next linear layer, which is straight forward by plugging in

Then we continue to unfold the next ReLU layer σ(z (L−2) ) using its linear relaxations, and compute

in a similar manner as in (12).

Along with the bound propagation process, we need to compute a series of matrices,

.

At this point, we merged all layers of the network into a linear layer: z

For ReLU networks, convex adversarial polytope (Wong & Kolter, 2018) uses a very similar bound propagation procedure.

CROWN-style bounds allow an adaptive selection of α i in (10), thus often gives better bounds (e.g., see Table 1 ).

We give details on each term in Appendix L.

Computational Cost.

Ordinary CROWN (Zhang et al., 2018) and convex adversarial polytope (Wong & Kolter, 2018) use (13)

as the final layer of the network.

For each layer m, we need a different set of m A matrices, defined as A m,(l) , l ∈ {m − 1, · · · , 0}. This causes three computational issues:

• Computation of all A m,(l) matrices is expensive.

Suppose the network has n neurons for all L − 1 intermediate and input layers and n L n neurons for the output layer (assuming L ≥ 2), the time complexity of ordinary CROWN or convex adversarial polytope is O(

A ordinary forward propagation only takes O(Ln 2 ) time per example, thus ordinary CROWN does not scale up to large networks for training, due to its quadratic dependency in L and extra Ln times overhead.

• When both W (l) and W (l−1) represent convolutional layers with small kernel tensors K (l) and

, there are no efficient GPU operations to form the matrix

and K (l−1) .

Existing implementations either unfold at least one of the convolutional kernels to fully connected weights, or use sparse matrices to represent W (l) and W (l−1) .

They suffer from poor hardware efficiency on GPUs.

In CROWN-IBP, we use IBP to obtain bounds of intermediate layers, which takes only twice the regular forward propagate time (O(Ln 2 )), thus we do not have the first and second issues.

The time complexity of the backward bound

, only n L times slower than forward propagation and significantly more scalable than ordinary CROWN (which is Ln times slower than forward propagation, where typically n n L ).

The third convolution issue is also not a concern, since we start from the last specification layer W (L) which is a small fully connected layer.

Suppose we need to compute

on GPUs using the transposed convolution operator with kernel K (L−1) , without unfolding any convoluational layers.

Conceptually, the backward pass of CROWN-IBP propagates a small specification matrix W (L) backwards, replacing affine layers with their transposed operators, and activation function layers with a diagonal matrix product.

This allows efficient implementation and better scalability.

Benefits of CROWN-IBP.

Tightness, efficiency and flexibility are unique benefits of CROWN-IBP:

• CROWN-IBP is based on CROWN, a tight linear relaxation based lower bound which can greatly improve the quality of bounds obtained by IBP to guide verifiable training and improve stabability;

• CROWN-IBP avoids the high computational cost of convex relaxation based methods : the time complexity is reduced from O(

, well suited to problems where the output size n L is much smaller than input and intermediate layers' sizes; also, there is no quadratic dependency on L. Thus, CROWN-IBP is efficient on relatively large networks;

• The objective (9) is strictly more general than IBP and allows the flexibility to exploit the strength from both IBP (good for large ) and convex relaxation based methods (good for small ).

We can slowly decrease β to 0 during training to avoid the over-regularization problem, yet keeping the initial training of IBP more stable by providing a much tighter bound; we can also keep β = 1 which helps to outperform convex relaxation based methods in small regime (e.g., = 2/255 on CIFAR-10).

Models and training schedules.

We evaluate CROWN-IBP on three models that are similar to the models used in (Gowal et al., 2018) on MNIST and CIFAR-10 datasets with different ∞ perturbation norms.

Here we denote the small, medium and large models in Gowal et al. (2018) as DM-small, DM-medium and DM-large.

During training, we first warm up (regular training without robust loss) for a fixed number of epochs and then increase from 0 to train using a ramp-up schedule of R epochs.

Similar techniques are also used in many other works (Wong et al., 2018; Wang et al., 2018a; Gowal et al., 2018) .

For both IBP and CROWN-IBP, a natural cross-entropy (CE) loss with weight κ (as in Eq (9)) may be added, and κ is scheduled to linearly decrease from κ start to κ end within R ramp-up epochs.

Gowal et al. (2018) used κ start = 1 and κ end = 0.5.

To understand the trade-off between verified accuracy and standard (clean) accuracy, we explore two more settings: κ start = κ end = 0 (without natural CE loss) and κ start = 1, κ end = 0.

For β, a linear schedule during the ramp-up period is used, but we always set β start = 1 and β end = 0, except that we set β start = β end = 1 for CIFAR-10 at = 2 255 .

Detailed model structures and hyperparameters are in Appendix C. Our training code for IBP and CROWN-IBP, and pre-trained models are publicly available 3 .

Metrics.

Verified error is the percentage of test examples where at least one element in the lower bounds m(x k , ) is < 0.

It is an guaranteed upper bound of test error under any ∞ perturbations.

We obtain m(x k , ) using IBP or CROWN-IBP (Eq. 13).

We also report standard (clean) errors and errors under 200-step PGD attack.

PGD errors are lower bounds of test errors under ∞ perturbations.

Comparison to IBP.

Table 2 represents the standard, verified and PGD errors under different for each dataset with different κ settings.

We test CROWN-IBP on the same model structures in Table 1 of Gowal et al. (2018) .

These three models' architectures are presented in Table A in the Appendix.

Here we only report the DM-large model structure in as it performs best under all setttings; small and medium models are deferred to Table C in the Appendix.

When both κ start = κ end = 0, no natural CE loss is added and the model focuses on minimizing verified error, but the lack of natural CE loss may lead to unstable training, especially for IBP; the κ start = 1, κ end = 0.5 setting emphasizes on minimizing standard error, usually at the cost of slightly higher verified error rates.

κ start = 1, κ end = 0 typically achieves the best balance.

We can observe that under the same κ settings, CROWN-IBP outperforms IBP in both standard error and verified error.

The benefits of CROWN-IBP is significant especially when model is large and is large.

We highlight that CROWN-IBP reduces the verified error rate obtained by IBP from 8.21% to 7.02% on MNIST at = 0.3 and from 55.88% to 46.03% on CIFAR-10 at = 2/255 (it is the first time that an IBP based method outperforms results from (Wong et al., 2018) , and our model also has better standard error).

We also note that we are the first to obtain verifiable bound on CIFAR-10 at = 16/255.

Trade-off Between Standard Accuracy and Verified Accuracy.

To show the trade-off between standard and verified accuracy, we evaluate DM-large CIFAR-10 model with test = 8/255 under different κ settings, while keeping all other hyperparameters unchanged.

For each κ end = {0.5, 0.25, 0}, we uniformly choose 11 κ start ∈ [1, κ end ] while keeping all other hyper-parameters unchanged.

A larger κ start or κ end tends to produce better standard errors, and we can explicitly control the trade-off between standard accuracy and verified accuracy.

In Figure 2 we plot the standard and verified errors of IBP and CROWN-IBP trained models with different κ settings.

Each cluster on the figure has 11 points, representing 11 different κ start values.

Models with lower verified errors tend to have higher standard errors.

However, CROWN-IBP clearly outperforms IBP with improvement on both standard and verified accuracy, and pushes the Pareto front towards the lower left corner, indicating overall better performance.

To reach the same verified error of 70%, CROWN-IBP can reduce standard error from roughly 55% to 45%.

Training Stability.

To discourage hand-tuning on a small set of models and demonstrate the stability of CROWN-IBP over a broader range of models, we evaluate IBP and CROWN-IBP on a variety of small and medium sized model architectures (18 for MNIST and 17 for CIFAR-10), detailed in Appendix D. To evaluate training stability, we compare verified errors under different ramp-up schedule length (R = {30, 60, 90, 120} on CIFAR-10 and R = {10, 15, 30, 60} on MNIST) Table 4 of Gowal et al. (2018) are evaluated using mixed integer programming (MIP) and linear programming (LP), which are strictly smaller than IBP verified errors but computationally expensive.

For a fair comparison, we use the IBP verified errors reported in their Table 3 .

† According to direct communications with Gowal et al. (2018) , achieving the 68.44% IBP verified error requires to adding an extra PGD adversarial training loss.

Without adding PGD, the verified error is 72.91% (LP/MIP verified) or 73.52% (IBP verified).

Our result should be compared to 73.52%.

‡ Although not explicitly mentioned, the CIFAR-10 models in (Gowal et al., 2018) are trained using train = 1.1 test.

We thus follow their settings.

§ We use βstart = βend = 1 for this setting, and thus CROWN-IBP bound (β = 1) is used to evaluate the verified error.

and different κ settings.

Instead of reporting just the best model, we compare the best, worst and median verified errors over all models.

Our results are presented in Figure 3 : (a) is for MNIST with = 0.3; (c),(d) are for CIFAR with = 8/255.

We can observe that CROWN-IBP achieves better performance consistently under different schedule length.

In addition, IBP with κ = 0 cannot stably converge on all models when schedule is short; under other κ settings, CROWN-IBP always performs better.

We conduct additional training stability experiments on MNIST and CIFAR-10 dataset under other model and settings and the observations are similar (see Appendix H).

We propose a new certified defense method, CROWN-IBP, by combining the fast interval bound propagation (IBP) bound and a tight linear relaxation based bound, CROWN.

Our method enjoys high computational efficiency provided by IBP while facilitating the tight CROWN bound to stabilize training under the robust optimization framework, and provides the flexibility to trade-off between the two.

Our experiments show that CROWN-IBP consistently outperforms other IBP baselines in both standard errors and verified errors and achieves state-of-the-art verified test errors for ∞ robustness.

Given a fixed neural network (NN) f (x), IBP gives a very loose estimation of the output range of f (x).

However, during training, since the weights of this NN can be updated, we can equivalently view IBP as an augmented neural network, which we denote as an IBP-NN ( Figure A) .

Unlike a usual network which takes an input x k with label y k , IBP-NN takes two points x L = x k − and x U = x k + as inputs (where x L ≤ x ≤ x U , element-wisely).

The bound propagation process can be equivalently seen as forward propagation in a specially structured neural network, as shown in Figure A .

After the last specification layer C (typically merged into W (L) ), we can obtain m(x k , ).

Then, −m(x k , ) is sent to softmax layer for prediction.

Importantly, since [m(x k , )] y k = 0 (as the y k -th row in C is always 0), the top-1 prediction of the augmented IBP network is y k if and only if all other elements of m(x k , ) are positive, i.e., the original network will predict correctly for all x L ≤ x ≤ x U .

When we train the augmented IBP network with ordinary cross-entropy loss and desire it to predict correctly on an input x k , we are implicitly doing robust optimization (Eq. (2)).

The simplicity of IBP-NN may help a gradient based optimizer to find better solutions.

On the other hand, while the computation of convex relaxation based bounds can also be cast as an equivalent network (e.g., the "dual network" in Wong & Kolter (2018)), its construction is significantly more complex, and sometimes requires non-differentiable indicator functions (the sets I + , I − and I in Wong & Kolter (2018)).

As a consequence, it can be challenging for the optimizer to find a good solution, and the optimizer tends to making the bounds tighter naively by reducing the norm of weight matrices and over-regularizing the network, as demonstrated in Figure 1 .

Both IBP and CROWN-IBP produce lower bounds m(x, ), and a larger lower bound has better quality.

To measure the relative tightness of the two bounds, we take the average of all bounds of training examples:

A positive value indicates that CROWN-IBP is tighter than IBP.

In Figure B we plot this averaged bound differences during schedule for one MNIST model and one CIFAR-10 model.

We can observe that during the early phase of training when the schedule just starts, CROWN-IBP produces significantly better bounds than IBP.

A tighter lower bound m(x, ) gives a tighter upper bound for max δ∈S L(x + δ; y; θ), making the minimax optimization problem (2) more effective to solve.

When the training schedule proceeds, the model gradually learns how to make IBP bounds tighter and eventually the difference between the two bounds become close to 0.

Why CROWN-IBP stabilizes IBP training?

When taking a randomly initialized network or a naturally trained network, IBP bounds are very loose.

But in Table 1 , we show that a network trained using IBP can eventually obtain quite tight IBP bounds and high verified accuracy; the network can adapt to IBP bounds and learn a specific set of weights to make IBP tight and also correctly classify examples.

However, since the training has to start from weights that produce loose bounds for IBP, the beginning phase of IBP training can be challenging and is vitally important.

We observe that IBP training can have a large performance variance across models and initializations.

Also IBP is more sensitive to hyper-parameter like κ or schedule length; in Figure 3 , many IBP models converge sub-optimally (large worst/median verified error).

The reason for instability is that during the beginning phase of training, the loose bounds produced by IBP make the robust loss (9) ineffective, and it is challenging for the optimizer to reduce this loss and find a set of good weights that produce tight IBP verified bounds in the end.

Conversely, if our bounds are much tighter at the beginning, the robust loss (9) always remains in a reasonable range during training, and the network can gradually learn to find a good set of weights that make IBP bounds increasingly tighter (this is obvious in Figure B) .

Initially, tighter bounds can be provided by a convex relaxation based method like CROWN, and they are gradually replaced by IBP bounds (using β start = 1, β end = 0), eventually leading to a model with learned tight IBP bounds in the end.

The goal of these experiments is to reproduce the performance reported in (Gowal et al., 2018) and demonstrate the advantage of CROWN-IBP under the same experimental settings.

Specifically, to reproduce the IBP results, for CIFAR-10 we train using a large batch size and long training schedule on TPUs (we can also replicate these results on multi-GPUs using a reasonable amount of training time; see Section F).

Also, for this set of experiments we use the same code base as in Gowal et al. (2018) .

For model performance on a comprehensive set of small and medium sized models trained on a single GPU, please see Table D in Section F, as well as the training stability experiments in Section 4 and Section H.

The models structures (DM-small, DM-medium and DM-large) used in Table C and Table 2 are listed in Table A .

These three model structures are the same as in Gowal et al. (2018) .

Training hyperparameters are detailed below:

• For MNIST IBP baseline results, we follow exact the same set of hyperparameters as in (Gowal et al., 2018) .

We train 100 epochs (60K steps) with a batch size of 100, and use a warm-up and ramp-up duration of 2K and 10K steps.

Learning rate for Adam optimizer is set to 1 × 10 −3 and decayed by 10X at steps 15K and 25K.

Our IBP results match their reported numbers.

Note that we always use IBP verified errors rather than MIP verified errors.

We use the same schedule for CROWN-IBP with train = 0.2 ( test = 0.1) in Table C  and Table 2 .

For train = 0.4, this schedule can obtain verified error rates 4.22%, 7.01% and 12.84% at test = {0.2, 0.3, 0.4} using the DM-Large model, respectively.

• For MNIST CROWN-IBP with train = 0.4 in Table C and Table 2 , we train 200 epochs with a batch size of 256.

We use Adam optimizer and set learning rate to 5 × 10 −4 .

We warm up with 10 epochs' regular training, and gradually ramp up from 0 to train in 50 epochs.

We reduce the learning rate by 10X at epoch 130 and 190.

Using this schedule, IBP's performance becomes worse (by about 1-2% in all settings), but this schedule improves verified error for CROWN-IBP at test = 0.4 from 12.84% to to 12.06% and does do affect verified errors at other test levels.

• For CIFAR-10, we follow the setting in Gowal et al. (2018) and train 3200 epochs on 32 TPU cores.

We use a batch size of 1024, and a learning rate of 5 × 10 −4 .

We warm up for 320 epochs, and ramp-up for 1600 epochs.

Learning rate is reduced by 10X at epoch 2600 and 3040.

We use random horizontal flips and random crops as data augmentation, and normalize images according to per-channel statistics.

Note that this schedule is slightly different from the schedule used in (Gowal et al., 2018); we use a smaller batch size due to TPU memory constraints (we used TPUv2 which has half memory capacity as TPUv3 used in (Gowal et al., 2018) ), and also we decay learning rates later.

We found that this schedule improves both IBP baseline performance and CROWN-IBP performance by around 1%; for example, at = 8/255, this improved schedule can reduce verified error from 73.52% to 72.68% for IBP baseline (κ start = 1.0, κ end = 0.5) using the DM-Large model.

Hyperparameter κ and β.

We use a linear schedule for both hyperparameters, decreasing κ from κ start to κ end while increasing β from β start to β end .

The schedule length is set to the same length as the schedule.

In both IBP and CROWN-IBP, a hyperparameter κ is used to trade-off between clean accuracy and verified accuracy.

Figure 2 shows that κ end can significantly affect the trade-off, while κ start has minor impacts compared to κ end .

In general, we recommend κ start = 1 and κ end = 0 as a safe starting point, and we can adjust κ end to a larger value if a better standard accuracy is desired.

The setting κ start = κ end = 0 (pure minimax optimization) can be challenging for IBP as there is no natural loss as a stabilizer; under this setting CROWN-IBP usually produces a model with good (sometimes best) verified accuracy but noticeably worse standard accuracy (on CIFAR-10 = 8 255 the difference can be as large as 10%), so this setting is only recommended when a model with best verified accuracy is desired at a cost of noticeably reduced standard accuracy.

Compared to IBP, CROWN-IBP adds one additional hyperparameter, β.

β has a clear meaning: balancing between the convex relaxation based bounds and the IBP bounds.

β start is always set to 1 as we want to use CROWN-IBP to obtain tighter bounds to stabilize the early phase of training when IBP bounds are very loose; β end determines if we want to use a convex relaxation based bound (β end = 1) or IBP based bound (β end = 0) after the schedule.

Thus, we set β end = 1 for the case where convex relaxation based method (Wong et al., 2018) can outperform IBP (e.g., CIFAR-10 = 2/255, and β end = 0 for the case where IBP outperforms convex relaxation based bounds.

We do not tune or grid-search this hyperparameter.

Gowal et al. (2018) .

"CONV k w×h+s" represents a 2D convolutional layer with k filters of size w×h using a stride of s in both dimensions.

"FC n" = fully connected layer with n outputs.

Last fully connected layer is omitted.

All networks use ReLU activation functions.

In all our training stability experiments, we use a large number of relatively small models and train them on a single GPU.

These small models cannot achieve state-of-the-art performance but they can be trained quickly and cheaply, allowing us to explore training stability over a variety of settings, and report min, median and max statistics.

We use the following hyperparameters:

• For MNIST, we train 100 epochs with batch size 256.

We use Adam optimizer and the learning rate is 5 × 10 −4 .

The first epoch is standard training for warming up.

We gradually increase linearly per batch in our training process with a schedule length of 60.

We reduce the learning rate by 50% every 10 epochs after schedule ends.

No data augmentation technique is used and the whole 28 × 28 images are used (normalized to 0 -1 range).

• For CIFAR, we train 200 epoch with batch size 128.

We use Adam optimizer and the learning rate is 0.1%.

The first 10 epochs are standard training for warming up.

We gradually increase linearly per batch in our training process with a schedule length of 120.

We reduce the learning rate by 50% every 10 epochs after schedule ends.

We use random horizontal flips and random crops as data augmentation.

The three channels are normalized with mean (0.4914, 0.4822, 0.4465) and standard deviation (0.2023, 0.1914, 0.2010) .

These numbers are per-channel statistics from the training set used in (Gowal et al., 2018) .

All verified error numbers are evaluated on the test set using IBP, since the networks are trained using IBP (β = 0 after reaches the target train ), except for CIFAR = 2 255 where we set β = 1 to compute the CROWN-IBP verified error.

Table B gives the 18 model structures used in our training stability experiments.

These model structures are designed by us and are not used in Gowal et al. (2018) .

Most CIFAR-10 models share the same structures as MNIST models (unless noted on the table) except that their input dimensions are different.

Model A is too small for CIFAR-10 thus we remove it for CIFAR-10 experiments.

Models A -J are the "small models" reported in Figure 3 .

Models K -T are the "medium models" reported in Figure 3 .

For results in Table 1 , we use a small model (model structure B) for all three datasets.

These MNIST, CIFAR-10 models can be trained on a single NVIDIA RTX 2080 Ti GPU within a few hours each.

In Table 2 we report results from the best DM-Large model.

Table C presents the verified, standard (clean) and PGD attack errors for all three model structures used in (Gowal et al., 2018 ) (DM-Small, DM-Medium and DM-Large) trained on MNIST and CIFAR-10 datasets.

We evaluate IBP and CROWN-IBP under the same three κ settings as in Table 2 .

We use hyperparameters detailed in Section C to train these models.

We can see that given any model structure and any κ setting, CROWN-IBP consistently outperforms IBP.

Table B : Model structures used in our training stability experiments.

We use ReLU activations for all models.

We omit the last fully connected layer as its output dimension is always 10.

In the table, "Conv k w × w + s" represents to a 2D convolutional layer with k filters of size w × w and a stride of s. Model A -J are referred to as "small models" and model K to T are referred to as "medium models".

In this section we present additional experiments on a variety of smaller MNIST and CIFAR-10 models which can be trained on a single GPU.

The purpose of this experiment is to compare model performance statistics (min, median and max) on a wide range of models, rather than a few hand selected models.

The model structures used in these experiments are detailed in Table B .

In Table D, we present the best, median and worst verified and standard (clean) test errors for models trained on MNIST and CIFAR-10 using IBP and CROWN-IBP.

Although these small models cannot achieve state-of-the-art performance, CROWN-IBP's best, median and worst verified errors among all model structures consistently outperform those of IBP.

Especially, in many situations the worst case verified error improves significantly using CROWN-IBP, because IBP training is not stable on some of the models.

It is worth noting that in this set of experiments we explore a different setting: train = test .

We found that both IBP and CROWN-IBP tend to overfit to training dataset on MNIST with small , thus verified errors are not as good as presented in Table C .

This overfitting issue can be alleviated by using train > test (as used in Table 2 and Table C) , or using an explicit 1 regularization, which will be discussed in detail in Section I.

To further test the training stability of CROWN-IBP, we run each MNIST experiment (using selected models in Table B ) 5 times to get the mean and standard deviation of the verified and standard errors on test set.

Results are presented in Table E Table E : Means and standard deviations of verified and standard errors of 10 MNIST models trained using CROWN-IBP.

The architectures of these models are presented in Table B .

We run each model 5 times to compute its mean and standard deviation.

regularization term in CROWN-IBP training helps when train = 0.1 or 0.2.

The verified and standard errors on the training and test sets with and without regularization can be found in Table F .

We can see that with a small 1 regularization added (λ = 5 × 10 −5 ) we can reduce verified errors on test set significantly.

This makes CROWN-IBP results comparable to the numbers reported in convex adversarial polytope (Wong et al., 2018) ; at = 0.1, the best model using convex adversarial polytope training can achieve 3.67% verified error, while CROWN-IBP achieves 3.60% best certified error on the models presented in Table F .

The overfitting is likely caused by IBP's strong learning power without over-regularization, which also explains why IBP based methods significantly outperform linear relaxation based methods at larger values.

Using early stopping can also improve verified error on test set; see Figure D. J TRAINING TIME In Table G we present the training time of CROWN-IBP, IBP and convex adversarial polytope (Wong et al., 2018) on several representative models.

All experiments are measured on a single RTX 2080 Ti GPU with 11 GB RAM except for 2 DM-Large models where we use 4 RTX 2080 Ti GPUs to speed up training.

We can observe that CROWN-IBP is practically 1.5 to 3.5 times slower than IBP.

Theoretically, CROWN-IBP is up to n L = 10 times slower 4 than IBP; however usually the total training time is less than 10 times since the CROWN-IBP bound is only computed during the ramp-up phase, and CROWN-IBP has higher GPU computation intensity and thus better GPU utilization than IBP.

convex adversarial polytope (Wong et al., 2018) (Wong et al., 2018) .

Using random projections alone is not sufficient to scale purely linear relaxation based methods to larger datasets, thus we advocate a combination of IBP bounds with linear relaxation based methods as in CROWN-IBP, which offers good scalability and stability.

We also note that the random projection based acceleration can also be applied to the backward bound propagation (CROWN-style bound) in CROWN-IBP to further speed up CROWN-IBP.

The use of 32 TPUs for our CIFAR-10 experiments is not necessary.

We use TPUs mainly for obtaining a completely fair comparison to IBP (Gowal et al., 2018) , as their implementation was TPU-based.

Since TPUs are not widely available, we additionally implemented CROWN-IBP using multi-GPUs.

We train the best models in Table 2 on 4 RTX 2080Ti GPUs.

As shown in Table H , we can achieve comparable verified errors using GPUs, and the differences between GPU and TPU training are around ±0.5%.

Training time is reported in Table G. L EXACT FORMS OF THE CROWN-IBP BACKWARD BOUND CROWN (Zhang et al., 2018 ) is a general framework that replaces non-linear functions in a neural network with linear upper and lower hyperplanes with respect to pre-activation variables, such that the entire neural network function can be bounded by a linear upper hyperplane and linear lower hyperplane for all x ∈ S (S is typically a norm bounded ball, or a box region):

CROWN achieves such linear bounds by replacing non-linear functions with linear bounds, and utilizing the fact that the linear combinations of linear bounds are still linear, thus these linear bounds 1 We use β start = β end = 1 for this setting, the same as in Table 2 , and thus CROWN-IBP bound is used to evaluate the verified error.

can propagate through layers.

Suppose we have a non-linear vector function σ, applying to an input (pre-activation) vector z, CROWN requires the following bounds in a general form:

In general the specific bounds A σ , b σ , A σ , b σ for different σ needs to be given in a case-by-case basis, depending on the characteristics of σ and the preactivation range z ≤ z ≤ z. In neural network common σ can be ReLU, tanh, sigmoid, maxpool, etc.

Convex adversarial polytope (Wong et al., 2018) is also a linear relaxation based techniques that is closely related to CROWN, but only for ReLU layers.

For ReLU such bounds are simple, where A σ , A σ are diagonal matrices, b σ = 0:

where D and D are two diagonal matrices:

, if z k > 0, i.e., this neuron is always active 0, if z k < 0, i.e., this neuron is always inactive α, otherwise, any 0 ≤ α ≤ 1 (15)

1, if z k > 0, i.e., this neuron is always active 0, if z k < 0, i.e., this neuron is always inactive

if z k > 0, i.e., this neuron is always active 0, if z k < 0, i.e., this neuron is always inactive

Note that CROWN-style bounds require to know all pre-activation bounds z (l) and z (l) .

We assume these bounds are valid for x ∈ S. In CROWN-IBP, these bounds are obtained by interval bound propagation (IBP).

With pre-activation bounds z (l) and z (l) given (for x ∈ S), we rewrite the CROWN lower bound for the special case of ReLU neurons:

Theorem L.1 (CROWN Lower Bound).

For a L-layer neural network function f (x) : R n0 → R n L , ∀j ∈ [n L ], ∀x ∈ S, we have f j (x) ≤ f j (x), where

if l ∈ {0, · · · , L − 1}.

@highlight

We propose a new certified adversarial training method, CROWN-IBP, that achieves state-of-the-art robustness for L_inf norm adversarial perturbations.