Differently from the popular Deep Q-Network (DQN) learning, Alternating Q-learning (AltQ) does not fully fit a target Q-function at each iteration, and is generally known to be unstable and inefficient.

Limited applications of AltQ mostly rely on substantially altering the algorithm architecture in order to improve its performance.

Although Adam appears to be a natural solution, its performance in AltQ has rarely been studied before.

In this paper, we first provide a solid exploration on how well AltQ performs with Adam.

We then take a further step to improve the implementation by adopting the technique of parameter restart.

More specifically, the proposed algorithms are tested on a batch of Atari 2600 games and exhibit superior performance than the DQN learning method.

The convergence rate of the slightly modified version of the proposed algorithms is characterized under the linear function approximation.

To the best of our knowledge, this is the first theoretical study on the Adam-type algorithms in Q-learning.

Q-learning (Watkins & Dayan, 1992 ) is one of the most important model-free reinforcement learning (RL) problems, which has received considerable research attention in recent years (Bertsekas & Tsitsiklis, 1996; Even-Dar & Mansour, 2003; Hasselt, 2010; Lu et al., 2018; Achiam et al., 2019) .

When the state-action space is large or continuous, parametric approximation of the Q-function is often necessary.

One remarkable success of parametric Q-learning in practice is its combination with deep learning, known as the Deep Q-Network (DQN) learning (Mnih et al., 2013; 2015) .

It has been applied to various applications in computer games (Bhatti et al., 2016) , traffic control (Arel et al., 2010) , recommendation systems (Zheng et al., 2018; Zhao et al., 2018) , chemistry research (Zhou et al., 2017) , etc.

Its on-policy continuous variant (Silver et al., 2014) has also led to great achievements in robotics locomotion (Lillicrap et al., 2016) .

The DQN algorithm is performed in a nested-loop manner, where the outer loop follows an one-step update of the Q-function (via the empirical Bellman operator for Q-learning), and the inner loop takes a supervised learning process to fit the updated (i.e., target) Q-function with a neural network.

In practice, the inner loop takes a sufficiently large number of iterations under certain optimizer (e.g. stochastic gradient descent (SGD) or Adam) to fit the neural network well to the target Q-function.

In contrast, a conventional Q-learning algorithm runs only one SGD step in each inner loop, in which case the overall Q-learning algorithm updates the Q-function and fits the target Q-function alternatively in each iteration.

We refer to such a Q-learning algorithm with alternating updates as Alternating Q-learning (AltQ).

Although significantly simpler in the update rule, AltQ is well known to be unstable and have weak performance (Mnih et al., 2016) .

This is in part due to the fact that the inner loop does not fit the target Q-function sufficiently well.

To fix this issue, Mnih et al. (2016) proposed a new exploration strategy and asynchronous sampling schemes over parallel computing units (rather than the simple replay sampling in DQN) in order for the AltQ algorithm to achieve comparable or better performance than DQN.

As another alternative, Knight & Lerner (2018) proposed a more involved natural gradient propagation for AltQ to improve the performance.

All these schemes require more sophisticated designs or hardware support, which may place AltQ less advantageous compared to the popular DQN, even with their better performances.

This motivates us to ask the following first question.

• Q1:

Can we design a simple and easy variant of the AltQ algorithm, which uses as simple setup as DQN and does not introduce extra computational burden and heuristics, but still achieves better and more stable performance than DQN?

In this paper, we provide an affirmative answer by introducing novel lightweight designs to AltQ based on Adam.

Although Adam appears to be a natural tool, its performance in AltQ has rarely been studied yet.

Thus, we first provide a solid exploration on how well AltQ performs with Adam (Kingma & Ba, 2014) , where the algorithm is referred to as AltQ-Adam.

We then take a further step to improve the implementation of AltQ-Adam by adopting the technique of parameter restart (i.e., restart the initial setting of Adam parameters every a few iterations), and refer to the new algorithm as AltQ-AdamR. This is the first time that restart is applied for improving the performance of RL algorithms although restart has been used for conventional optimization before.

In a batch of 23 Atari 2600 games, our experiments show that both AltQ-Adam and AltQ-AdamR outperform the baseline performance of DQN by 50% on average.

Furthermore, AltQ-AdamR effectively reduces the performance variance and achieves a much more stable learning process.

In our experiments for the linear quadratic regulator (LQR) problems, AltQ-AdamR converges even faster than the model-based value iteration (VI) solution.

This is a rather surprising result given that the model-based VI has been treated as the performance upper bound for the Q-learning (including DQN) algorithms with target update .

Regarding the theoretical analysis of AltQ algorithms, their convergence guarantee has been extensively studied (Melo et al., 2008; Chen et al., 2019b) .

More references are given in Section 1.1.

However, all the existing studies focus on the AltQ algorithms that take a simple SGD step.

Such theory is not applicable to the proposed AltQ-Adam and AltQ-AdamR that implement the Adam-type update.

Thus, the second intriguing question we address here is described as follows.

• Q2:

Can we provide the convergence guarantee for AltQ-Adam and AltQ-AdamR or their slightly modified variants (if these two algorithms do not always converge by nature)?

It is well known in optimization that Adam does not always converge, and instead, a slightly modified variant AMSGrad proposed in Reddi et al. (2018) has been widely accepted as an alternative to justify the performance of Adam-type algorithms.

Thus, our theoretical analysis here also focuses on such slightly modified variants AltQ-AMSGrad and AltQ-AMSGradR of the proposed algorithms.

We show that under the linear function approximation (which is the structure that the current tools for analysis of Q-learning can handle), both AltQ-AMSGrad and AltQ-AMSGradR converge to the global optimal solution under standard assumptions for Qlearning.

To the best of our knowledge, this is the first non-asymptotic convergence guarantee on Q-learning that incorporates Adam-type update and momentum restart.

Furthermore, a slight adaptation of our proof provides the convergence rate for the AMSGrad for conventional strongly convex optimization which has not been studied before and can be of independent interest.

Notations We use x := x 2 = √ x T x to denote the 2 norm of a vector x, and use x ∞ = max i |x i | to denote the infinity norm.

When x, y are both vectors, x/y, xy, x 2 , √ x are all calculated in the element-wise manner, which will be used in the update of Adam and AMSGrad.

We denote [n] = 1, 2, . . .

, n, and x ∈ Z as the largest integer such that x ≤ x < x + 1.

Empirical performance of AltQ: AltQ algorithms that strictly follow the alternating updates are rarely used in practice, particularly in comparison with the well-accepted DQN learning and its improved variants of dueling network structure (Wang et al., 2016 ), double Q-learning (Hasselt, 2010 and variance exploration and sampling schemes (Schaul et al., 2015) .

Mnih et al. (2016) proposed the asynchronous one-step Q-learning that is conceptually close to AltQ with competitive performance against DQN.

However, the algorithm still relies on a slowly moving target network like DQN, and the multi-thread learning also complicates the computational setup.

Lu et al. (2018) studied the problem of value overestimation and proposed the non-delusional Q-learning algorithm that employed the so-called pre-conditioned Q-networks, which is also computationally complex.

Knight & Lerner (2018) proposed a natural gradient propagation for AltQ to improve the performance, where the gradient implementation is complex.

In this paper, we propose two simple and computationally efficient schemes to improve the performance of AltQ.

We consider a Markov decision process with a considerably large or continuous state space S ⊂ R M and action space A ⊂ R N with a non-negative bounded reward function R :

We define U (s) ⊂ A as the admissible set of actions at state s, and π : S → A as a feasible stationary policy.

We seek to solve a discrete-time sequential decision problem with γ ∈ (0, 1) as follows:

Let J (s) := J π (s) be the optimal value function when applying the optimal policy π .

The corresponding optimal Q-function can be defined as

where s ∼ P (·|s, a) and we use the same notation hereafter when no confusion arises.

In other words, Q (s, a) is the reward of an agent who starts from state s and takes action a at the first step and then follows the optimal policy π thereafter.

This paper focuses on the Alternating Q-learning (AltQ) algorithm that uses a parametric function Q(s, a; θ) to approximate the Q-function with a parameter θ of finite and relatively small dimension.

The update rule of AltQ-learning is given by

where α t is the step size at time t.

It is immediate from the equations that AltQ performs the update by taking one step temporal target update and one step parameter learning in an alternating fashion.

As DQN is also included in this work for performance comparison.

We recall the update of DQN in the following as reference.

Differently from AltQ, DQN updates the parameters in a nested loop.

Within the t-th inner loop, DQN first obtains the target Q-function as in Equation (5), and then uses a neural network to fit the target Q-function by running Y steps of a certain optimization algorithm as Equation (6).

The update rule of DQN is given as follows.

TQ(s, a; θ

where Optimizer can be SGD or Adam for example, and Equation (6) is thus a supervised learning process with TQ(s, a; θ 0 t )) as the "supervisor".

At the t-th outer loop, DQN performs the so-called target update as θ

In practice, when one of the momentum-based optimizers is adopted for Equation (6), such as Adam, it is only initialized once at the beginning of the first inner loop.

The historical gradient terms then accumulate throughout multiple inner loops with different targets.

While this stabilizes the DQN training empirically, it is still lack of theoretical understanding on how the optimizer affects the training with various moving targets.

As we will discuss in detail in Section 5, the analysis of AltQ with Adam can potentially shed light on such ambiguity and inspire future work for this matter.

Note that AltQ and DQN mainly differ at how the Q-function evolves after each step of sampling.

A fair comparison between the algorithms should be made without introducing dramatic difference on gradient propagation (Knight & Lerner, 2018) , policy structure, exploration and sampling strategies (Mnih et al., 2016) .

In practice, the vanilla AltQ is often slow in convergence and unstable with high variance.

To improve the performance, we propose to incorporate Adam and restart schemes, which are easy to implement and yield improved performance than DQN.

In this section, we first describe how to incorporate Adam to the AltQ algorithm, and then introduce a novel implementation scheme to improve the performance of AltQ with Adam.

AltQ with Adam-type update We propose a new AltQ algorithm with Adam-type update (AltQAdam) as described in Algorithm 1.

Its update is similar to the well-known Adam (Kingma & Ba, 2014) .

The iterations evolve by updating the exponentially decaying average of historical gradients (m t ) and squared historical gradients (v t ).

The hyper-parameters β 1 , β 2 are used to exponentially decrease the rate of the moving averages.

The difference between Algorithm 1 and the standard Adam in supervised learning is that in AltQ, there is no fixed target to "supervise" the learning process.

The target is always moving along with iteration t, leading to more noisy gradient estimations.

The proposed algorithm sheds new light on the possibility of using Adam to deal with such unique challenge brought by RL.

AltQ-Adam with momentum restart We also introduce the restart technique to AltQ-Adam and propose AltQ-AdamR as Algorithm 2.

Traditional momentum-based algorithms largely depend on the historical gradient direction.

When part of the historical information is incorrect, the estimation error tends to accumulate.

The restart technique can be employed to deal with this issue.

One way to restart the momentum-based methods is to initialize the momentum at some restart iteration.

That is, at restart iteration r, we reset m r , v r , i.e., m r = 0, v r = 0, which yields θ r+1 = θ r .

It is an intuitive implementation technique to adjust the trajectory from time to time, and can usually help mitigate the aforementioned problem while keeping fast convergence property.

For the implementation, we execute the restart periodically with a period r. It turns out that the restart technique can significantly improve the numerical performance, which can be seen in Section 4.

We empirically evaluate the proposed algorithms in this section.

The linear quadratic regulator (LQR) is a direct numerical demonstration of the convergence analysis under linear function approximation which will be discussed in the next section.

Atari 2600 game (Bellemare et al., 2013; Brockman et al., 2016) , a classic benchmark for DQN evaluations, is also used to show the effectiveness of the proposed algorithms for complicated tasks.

In practice, we also make a small adjustment to the proposed algorithms.

That is, we re-scale the loss term of L(θ t ) :=Q t (s, a; θ t ) − TQ(s, a; θ t )

TQ(s, a; θ t ) = R(s, a) + γ max a Q (s , a ; θ t ) 4:

:

if mod(t, r) = 0 then 4:

TQ(s, a; θ t ) = R(s, a) + γ max a Q (s , a ; θ t ) 7:

8:

:

with some scaling factorτ ∈ (0, 1], which is beneficial for stabilizing the learning process.

We find that in both experiments, AltQ-AdamR outperforms both AltQ-Adam and DQN in terms of convergence speed and variance reduction.

Compared with DQN in the empirical experiments of Atari games, under the same hyper-parameter settings, AltQ-Adam and AltQ-AdamR improve the performance of DQN by 50% on average.

We numerically validate the proposed algorithms through an infinite-horizon discrete-time LQR problem whose background can be found in Appendix A.1.

A typical model-based solution (with known dynamics), known as the discrete-time algebraic Riccati equation (DARE), is adopted to derive the optimal policy u t = −K x t .

The performance of the learning algorithm is then evaluated at each step of iterate t with the Euclidean norm K t − K .

The performance result for each method is averaged over 10 trials with different random seeds.

All algorithms share the same set of random seeds and are initialized with the same θ 0 .

The hyper-parameters of the learning settings are also consistent and further details are shown in Table 1 .

Note that for all the implementations, we also adopt the double Q-update (Hasselt, 2010) to help prevent over-estimations of the Q-value.

The performance results are seen in Figure 1 .

Here we highlight main observations from the LQR experiments.

• AltQ-AdamR outperforms DARE In ideal cases where data sampling perfectly emulates the system dynamics and the target is accurately learned in each inner loop, DARE for LQR would become equivalent to the DQN-like update if the neural network is replaced with a parameterzied linear function.

In practice, such ideal conditions are difficult to satisfy, and hence the actual Q-learning with target update is usually far slower (in terms of number of steps of target updates) than DARE.

Note that AltQ-AdamR performs significantly well and even converges faster than DARE, and thus implies it is faster than the most well-performing Q-learning with target update.

• AltQ-AdamR outperforms AltQ-Adam Overall, under the same batch sampling scheme and restart period, AltQ-AdamR achieves a faster convergence and smaller variance than AltQ-Adam.

Step

We apply the proposed AltQ algorithms to more challenging tasks of deep convolutional neural network playing a group of Atari 2600 games.

The particular DQN we train to compare against adopts the dueling network structure (Wang et al., 2016) , double Q-learning setup (Van Hasselt et al., 2016) , -greedy exploration and experience replay (Mnih et al., 2013) .

Adam is also adopted, without momentum restart, as the optimizer for the inner-loop supervised learning process.

AltQ-Adam and AltQ-AdamR are implemented using the identical setup of network construction, exploration and sampling strategies.

We test all the three algorithms with a batch of 23 Atari games.

The choice of 10 million steps of iteration is a common setup for benchmark experiments with Atari games.

Although this does not guarantee the best performance in comparison with more time-consuming training with 50 million steps or more, it is sufficient to illustrate different performances among the selected methods.

The software infrastructure is based on the baseline implementation of OpenAI.

Selections of the hyperparameters are listed in Table 2 .

We summarize the results in Figure 2 .

The overall performance is illustrated by first normalizing the return of each method with respect to the results obtained from DQN, and then averaging the performance of all 23 games to obtain the mean return and standard deviation.

Considering we use a smaller buffer size than common practice, DQN is not consistently showing improved return over all tested games.

Therefore, the self-normalized average return of DQN in Figure 2 is not strictly increasing from 0 to 100%.

Overall, both AltQ-Adam and AltQ-AdamR achieve significant improvement in comparison with the DQN results.

While AltQ-Adam is suffering from a higher variance, periodic restart (AltQAdamR) resolves the issue efficiently with an on-par performance on average and far smaller variance.

Specifically, in terms of the maximum average return, AltQ-Adam and AltQ-AdamR perform no worse then DQN on 17 and 20 games respectively out of the 23 games being evaluated.

In this section, we characterize the convergence guarantee for the proposed AltQ-learning algorithms.

Furthermore, like most of the related papers, we focus on convergence analysis under the linear approximation class.

Understanding the analytical behavior in the linear case is an important stepping stone to understand general cases such as deep neural network.

A linear approximation of the Q-functionQ(s, a; θ) can be written aŝ

where θ ∈ R d , and Φ : S × A → R d is a vector function of size d, and the elements of Φ represent the nonlinear kernel (feature) functions.

we introduce AltQ-AMSGradR which applies the same update rule as Algorithm 3, but resets m t ,v t with a period of r, i.e., m t = 0,v t = 0, ∀t = kr, k = 1, 2, · · · .

Algorithm 3 AltQ-AMSGrad 1: Input: α, λ, θ 1 , β 1 , β 2 , m 0 = 0,v 0 = 0.

2: for t = 1, 2, . . . , T do 3:

5:

Our theoretical analysis here focuses on the slight variants, AltQ-AMSGrad and AltQ-AMSGradR. Before stating the theorems, we first introduce some technical assumptions for our analysis.

Assumption 1.

At each iteration t, the noisy gradient is unbiased and uniformly bounded, i.e. g t =ḡ t + ξ t with Eξ t = 0 whereḡ t = E[g t ], and g t < G ∞ , ∀t.

Thus g t ∞ < G ∞ and g t 2 < G 2 ∞ .

Assumption 2. (Chen et al., 2019b, Lemma 6 .7) The equationḡ(θ) = 0 has a unique solution θ , which implies that there exists a c > 0, such that for any θ ∈ R d we have

Assumption 3.

The domain D ⊂ R d of approximation parameters is a ball originating at θ = 0 with bounded diameter containing θ .

That is, there exists D ∞ , such that θ m − θ n < D ∞ , ∀θ m , θ n ∈ D, and θ ∈ D.

Assumption 1 is standard in the theoretical analysis of Adam-type algorithms (Chen et al., 2019a; Zhou et al., 2018) .

Under linear function approximation and given Assumption 3 and bounded r(·), Assumption 1 is almost equivalent to the assumption of bounded φ(·) which is commonly taken in related RL work (Tsitsiklis & Van Roy, 1997; Bhandari et al., 2018) .

Assumption 2 has been proved as a key technical lemma in Chen et al. (2019b) under certain assumptions.

Such an assumption appears to be the weakest in the existing studies of the theoretic guarantee for Q-learning with function approximation.

, β 1t = β 1 λ t and δ = β 1 /β 2 with δ, λ ∈ (0, 1) for t = 1, 2, . . .

in Algorithm 3.

Given Assumptions 1 ∼ 3, the output of AltQ-AMSGrad satisfies:

where

.

In Theorem 1, B 1 , B 2 , B 3 in the bound in Equation (10) are constants and independent of time.

Therefore, under the choice of the stepsize and hyper-parameters in Algorithm 3, AltQ-AMSGrad achieves a convergence rate of O

T which is justified in Duchi et al. (2011) .

Remark 1.

Our proof of convergence here has two major differences from that for AMSGrad in Reddi et al. (2018): (a) The two algorithms are quite different.

AltQ-AMSGrad is a Q-learning algorithm alternatively finding the best policy, whereas AMSGrad is an optimizer for conventional optimization and does not have alternating nature.

(b) Our analysis is on the convergence rate whereas Reddi et al. (2018) provides regret bound.

In fact, a slight modification of our proof also provides the convergence rate of AMSGrad for conventional strongly convex optimization, which can be of independent interest.

Moreover, our proof avoids the theoretical error in the proof in Reddi et al. (2018)

In the following theorem, we provide the convergence result for AltQ-AMSGradR. Theorem 2. (Convergence of AltQ-AMSGradR) Under the same condition of Theorem 1, the output of AltQ-AMSGradR satisfies:

where

, and B 3 =

We propose two types of the accelerated AltQ algorithms, and demonstrate their superior performance over the state-of-the-art through a linear quadratic regulator problem and a batch of 23 Atari 2600 games.

Notably, Adam is not the only scheme in the practice for general optimization.

Heavy ball (Ghadimi et al., 2015) and Nesterov (Nesterov, 2013) are also popular momentum-based methods.

When adopting such methods in AltQ-learning for RL problems, however, we tend to observe a less stable learning process than AltQ-Adam.

This is partially caused by the fact that they optimize over a shorter historical horizon of updates than Adam.

Furthermore, the restart scheme provides somewhat remarkable performance in our study.

It is thus of considerable future interest to further investigate the potential of such a scheme.

One possible direction is to develop an adaptive restart mechanism with changing period determined by an appropriately defined signal of restart.

This will potentially relieve the effort in hyper-parameter tuning of finding a good fixed period.

We discuss more details on the experiment setup and provide further results that are not included in Section 4.

The linear quadratic regulator (LQR) problem is of great interest for control community where Lewis et al. applies PQL to both discrete-time problems Lewis & Vamvoudakis, 2011) and continuous-time problems (Vamvoudakis, 2017; ).

We empirically validate the proposed algorithms through an infinite-horizon discrete-time LQR problem defined as

subject to

where u t = π(x t ).

A typical model-based solution (with known A and B) considers the problem backwards in time and iterates a dynamic equation known as the discrete-time algebraic Riccati equation (DARE):

with the cost-to-go P being positive definite.

The optimal policy satisfies u t = −K x t with

For experiments, we parameterize a quadratic Q-function with a matrix parameter H in the form of

The corresponding linear policy satisfies u = −Kx, and K = H −1 uu H ux .

The performance of the learning algorithm is then evaluated at each step of iterate i with the Euclidean norm K i − K 2 .

We list detailed experiments of the 23 Atari games evaluated with the proposed algorithms in Figure 3 .

All experiments are executed with the same set of two random seeds.

Each task takes about 20-hour of wall-clock time on a GPU instance.

All three methods being evaluated share similar training time.

AltQ-Adam and AltQ-AdamR can be further accelerated in practice with a more memory-efficient implementation considering the target network is not required.

We keep our implementation of proposed algorithms consistent with the DQN we are comparing against.

Other techniques that are not included in this experiment are also compatible with AltQ-Adam and AltQ-AdamR, such like asynchronous exploration (Mnih et al., 2013) and training with decorrelated loss (Mavrin et al., 2019) .

Overall, AltQ-Adam significantly increases the performance by over 100% in some of the tasks including Asterix, BeamRider, Enduro, Gopher, etc.

However, it also illustrates certain instability with complete failure on Amidar and Assault.

This is mostly caused by the sampling where we are using a relevantly small buffer size with 10% of the common configured size in Atari games with experience replay.

Notice that those failures tend to appear when the -greedy exploration has evolved to a certain level where the immediate policy is effectively contributing to the accumulated experience.

This potentially amplifies the biased exploration that essentially leads to the observed phenomenon.

Interstingly, AltQ-AdamR that incorporates the restart scheme resolves the problem of high variance of average return brought by AltQ-Adam and provides a more consistent performance across the task domain.

This implies that momentum restart effectively corrects the accumulated error and stabilizes the training process.

Clearly Π D,V 1/4 t (θ ) = θ due to Assumption 3.

We start from the update of θ t when t ≥ 2.

where (i) follows from the Cauchy-Schwarz inequality, and (ii) holds becausev t+1,i ≥v t,i , ∀t, ∀i.

Next, we take the expectation over all samples used up to time step t on both sides, which still preserves the inequality.

Since we consider i.i.d.

sampling case, by letting F t be the filtration of all the sampling up to time t, we have

Thus we have

where (i) follows from Equation (18), (ii) follows due to Assumption 2 and 1 − β 1t > 0, (iii)

follows from β 1t < β 1 < 1 and E θ t − θ 2 > 0, and (iv) follows from V 1/4

∞ by Lemma 1 and Assumption 3.

We note that (iii) is the key step to avoid the error in the proof in Reddi et al. (2018) , where we can directly bound 1 − β 1t , which is impossible in Reddi et al. (2018) .

By rearranging the terms in the above inequality and taking the summation over time steps, we have

where (i) follows from α t < α t−1 .

With further adjustment of the first term in the right hand side of the last inequality, we can then bound the sum as

where (i) follows from Assumption 3 and becausev 1/2 t,i αt >v 1/2 t−1,i αt−1 , and (ii) follows from Lemmas 1 -3.

Finally, applying the Jensen's inequality yields

We conclude our proof by further applying the bound in Equation (19) to Equation (20).

To prove the convergence for AltQ-AMSGradR, the major technical development beyond the proof of Theorem 1 lies in dealing with the parameter restart.

More specifically, the moment approximation terms are reset every r steps, i.e., m kr =v kr = 0 for k = 1, 2, . . .

, which implies θ kr+1 = θ kr for k = 1, 2, . . . .

For technical convenience, we define θ 0 = θ 1 .

Using the arguments similar to Equation (19), in a time window that does not contain a restart (i.e. kr ≤ S ≤ (k + 1)r − 1) we have

where (i) follows from Equation (19) and (ii) follows from θ kr+1 = θ kr due to the definition of restart.

Then we take the summation over the total time steps and obtain

<|TLDR|>

@highlight

New Experiments and Theory for Adam Based Q-Learning

@highlight

This paper provides a convergence result for traditional Q-learning with linear function approximation when using an Adam-like update. 

@highlight

This paper describes a method to improve the AltQ algorithm by using a combination of an Adam optimizer and regularly restarting the internal parameters of the Adam optimizer.