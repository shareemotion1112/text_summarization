Conservation laws are considered to be fundamental laws of nature.

It has broad application in many fields including physics, chemistry, biology, geology, and engineering.

Solving the differential equations associated with conservation laws is a major branch in computational mathematics.

Recent success of machine learning, especially deep learning, in areas such as computer vision and natural language processing, has attracted a lot of attention from the community of computational mathematics and inspired many intriguing works in combining machine learning with traditional methods.

In this paper, we are the first to explore the possibility and benefit of solving nonlinear conservation laws using deep reinforcement learning.

As a proof of concept, we focus on 1-dimensional scalar conservation laws.

We deploy the machinery of deep reinforcement learning to train a policy network that can decide on how the numerical solutions should be approximated in a sequential and spatial-temporal adaptive manner.

We will show that the problem of solving conservation laws can be naturally viewed as a sequential decision making process and the numerical schemes learned in such a way can easily enforce long-term accuracy.

Furthermore, the learned policy network is carefully designed to determine a good local discrete approximation based on the current state of the solution, which essentially makes the proposed method a meta-learning approach.

In other words, the proposed method is capable of learning how to discretize for a given situation mimicking human experts.

Finally, we will provide details on how the policy network is trained, how well it performs compared with some state-of-the-art numerical solvers such as WENO schemes, and how well it generalizes.

Our code is released anomynously at \url{https://github.com/qwerlanksdf/L2D}.

Conservation laws are considered to be one of the fundamental laws of nature, and has broad applications in multiple fields such as physics, chemistry, biology, geology, and engineering.

For example, Burger's equation, a very classic partial differential equation (PDE) in conservation laws, has important applications in fluid mechanics, nonlinear acoustics, gas dynamics, and traffic flow.

Solving the differential equations associated with conservation laws has been a major branch of computational mathematics (LeVeque, 1992; 2002) , and a lot of effective methods have been proposed, from classic methods such as the upwind scheme, the Lax-Friedrichs scheme, to the advanced ones such as the ENO/WENO schemes (Liu et al., 1994; Shu, 1998) , the flux-limiter methods (Jerez Galiano & Uh Zapata, 2010) , and etc.

In the past few decades, these traditional methods have been proven successful in solving conservation laws.

Nonetheless, the design of some of the high-end methods heavily relies on expert knowledge and the coding of these methods can be a laborious process.

To ease the usage and potentially improve these traditional algorithms, machine learning, especially deep learning, has been recently incorporated into this field.

For example, the ENO scheme requires lots of 'if/else' logical judgments when used to solve complicated system of equations or high-dimensional equations.

This very much resembles the old-fashioned expert systems.

The recent trend in artificial intelligence (AI) is to replace the expert systems by the so-called 'connectionism', e.g., deep neural networks, which leads to the recent bloom of AI.

Therefore, it is natural and potentially beneficial to introduce deep learning in traditional numerical solvers of conservation laws.

In the last few years, neural networks (NNs) have been applied to solving ODEs/PDEs or the associated inverse problems.

These works can be roughly classified into three categories according to the way that the NN is used.

The first type of works propose to harness the representation power of NNs, and are irrelevant to the numerical discretization based methods.

For example, Raissi et al. (2017a; b) ; Yohai Bar-Sinai (2018) treated the NNs as new ansatz to approximate solutions of PDEs.

It was later generalized by Wei et al. (2019) to allow randomness in the solution which is trained using policy gradient.

More recent works along this line include (Magiera et al., 2019; Michoski et al., 2019; Both et al., 2019) .

Besides, several works have focused on using NNs to establish direct mappings between the parameters of the PDEs (e.g. the coefficient field or the ground state energy) and their associated solutions (Khoo et al., 2017; Khoo & Ying, 2018; Fan et al., 2018b) .

Furthermore, Han et al. (2018) ; Beck et al. (2017) proposed a method to solve very high-dimensional PDEs by converting the PDE to a stochastic control problem and use NNs to approximate the gradient of the solution.

The second type of works focus on the connection between deep neural networks (DNNs) and dynamic systems (Weinan, 2017; Chang et al., 2017; Lu et al., 2018; Long et al., 2018b; Chen et al., 2018) .

These works observed that there are connections between DNNs and dynamic systems (e.g. differential equations or unrolled optimization algorithms) so that we can combine deep learning with traditional tools from applied and computational mathematics to handle challenging tasks in inverse problems (Long et al., 2018b; a; Qin et al., 2018) .The main focus of these works, however, is to solve inverse problems, instead of learning numerical discretizations of differential equations.

Nonetheless, these methods are closely related to numerical differential equations since learning a proper discretization is often an important auxiliary task for these methods to accurately recover the form of the differential equations.

The third type of works, which target at using NNs to learn new numerical schemes, are closely related to our work.

However, we note that these works mainly fall in the setting of supervised learning (SL).

For example, Discacciati et al. (2019) proposed to integrate NNs into high-order numerical solvers to predict artificial viscosity; Ray & Hesthaven (2018) trained a multilayer perceptron to replace traditional indicators for identifying troubled-cells in high-resolution schemes for conservation laws.

These works greatly advanced the development in machine learning based design of numerical schemes for conservation laws.

Note that in Discacciati et al. (2019) , the authors only utilized the one-step error to train the artificial viscosity networks without taking into account the longterm accuracy of the learned numerical scheme.

Ray & Hesthaven (2018) first constructed several functions with known regularities and then used them to train a neural network to predict the location of discontinuity, which was later used to choose a proper slope limiter.

Therefore, the training of the NNs is separated from the numerical scheme.

Then, a natural question is whether we can learn discretization of differential equations in an end-to-end fashion and the learned discrete scheme also takes long-term accuracy into account.

This motivates us to employ reinforcement learning to learn good solvers for conservation laws.

The main objective of this paper is to design new numerical schemes in an autonomous way.

We propose to use reinforcement learning (RL) to aid the process of solving the conservation laws.

To our best knowledge, we are the first to regard numerical PDE solvers as a MDP and to use (deep) RL to learn new solvers.

We carefully design the proposed RL-based method so that the learned policy can generate high accuracy numerical schemes and can well generalize in varied situations.

Details will be given in section 3.

Here, we first provide a brief discussion on the benefits of using RL to solve conservation laws (the arguments apply to general evolution PDEs as well):

• Most of the numerical solvers of conservation law can be interpreted naturally as a sequential decision making process (e.g., the approximated grid values at the current time instance definitely affects all the future approximations).

Thus, it can be easily formulated as a Markov Decision Process (MDP) and solved by RL.

• In almost all the RL algorithms, the policy π (which is the AI agent who decides on how the solution should be approximated locally) is optimized with regards to the values Q π (s 0 , a 0 ) = r(s 0 , a 0 ) + ∞ t=1 γ t r(s t , a t ), which by definition considers the long-term accumulated reward (or, error of the learned numerical scheme), thus could naturally guarantee the long-term accuracy of the learned schemes, instead of greedily deciding the local approximation which is the case for most numerical PDEs solvers.

Furthermore, it can gracefully handle the cases when the action space is discrete, which is in fact one of the major strength of RL.

• By optimizing towards long-term accuracy and effective exploration, we believe that RL has a good potential in improving traditional numerical schemes, especially in parts where no clear design principles exist.

For example, although the WENO-5 scheme achieves optimal order of accuracy at smooth regions of the solution (Shu, 1998) , the best way of choosing templates near singularities remains unknown.

Our belief that RL could shed lights on such parts is later verified in the experiments: the trained RL policy demonstrated new behaviours and is able to select better templates than WENO and hence approximate the solution better than WENO near singularities.

• Non-smooth norms such as the infinity norm of the error is often used to evaluate the performance of the learned numerical schemes.

As the norm of the error serves as the loss function for the learning algorithms, computing the gradient of the infinity norm can be problematic for supervised learning, while RL does not have such problem since it does not explicitly take gradients of the loss function (i.e. the reward function for RL).

• Learning the policy π within the RL framework makes the algorithm meta-learning-like (Schmidhuber, 1987; Bengio et al., 1992; Andrychowicz et al., 2016; Li & Malik, 2016; Finn et al., 2017) .

The learned policy π can decide on which local numerical approximation to use by judging from the current state of the solution (e.g. local smoothness, oscillatory patterns, dissipation, etc).

This is vastly different from regular (non-meta-) learning where the algorithms directly make inference on the numerical schemes without the aid of an additional network such as π.

As subtle the difference as it may seem, meta-learning-like methods have been proven effective in various applications such as in image restoration (Jin et al., 2017; Fan et al., 2018a; Zhang et al., 2019) .

See (Vanschoren, 2018) for a comprehensive survey on meta-learning.

• Another purpose of this paper is to raise an awareness of the connection between MDP and numerical PDE solvers, and the general idea of how to use RL to improve PDE solvers or even finding brand new ones.

Furthermore, in computational mathematics, a lot of numerical algorithms are sequential, and the computation at each step is expert-designed and usually greedy, e.g., the conjugate gradient method, the fast sweeping method (Zhao, 2005) , matching pursuit (Mallat & Zhang, 1993) , etc.

We hope our work could motivate more researches in combining RL and computational mathematics, and stimulate more exploration on using RL as a tool to tackle the bottleneck problems in computational mathematics.

Our paper is organized as follows.

In section 2 we briefly review 1-dimensional conservation laws and the WENO schemes.

In section 3, we discuss how to formulate the process of numerically solving conservation laws into a Markov Decision Process.

Then, we present details on how to train a policy network to mimic human expert in choosing discrete schemes in a spatial-temporary adaptive manner by learning upon WENO.

In section 4, we conduct numerical experiments on 1-D conservation laws to demonstrate the performance of our trained policy network.

Our experimental results show that the trained policy network indeed learned to adaptively choose good discrete schemes that offer better results than the state-of-the-art WENO scheme which is 5th order accurate in space and 4th order accurate in time.

This serves as an evidence that the proposed RL framework has the potential to design high-performance numerical schemes for conservation laws in a data-driven fashion.

Furthermore, the learned policy network generalizes well to other situations such as different initial conditions, mesh sizes, temporal discrete schemes, etc.

The paper ends with a conclusion in section 5, where possible future research directions are also discussed.

In this paper, we consider solving the following 1-D conservation laws:

For example, f = u 2 2 is the famous Burger's Equation.

We discretize the (x, t)-plane by choosing a mesh with spatial size ∆x and temporal step size ∆t, and define the discrete mesh points (x j , t n ) by

We denote x j+

2 )∆x.

The finite difference methods will produce approximations U n j to the solution u(x j , t n ) on the given discrete mesh points.

We denote pointwise values of the true solution to be u n j = u(x j , t n ), and the true point-wise flux values to be f

WENO (Weighted Essentially Non-Oscillatory) (Liu et al., 1994 ) is a family of high order accurate finite difference schemes for solving hyperbolic conservation laws, and has been successful for many practical problems.

The key idea of WENO is a nonlinear adaptive procedure that automatically chooses the smoothest local stencil to reconstruct the numerical flux.

Generally, a finite difference method solves Eq.1 by using a conservative approximation to the spatial derivative of the flux:

where u j (t) is the numerical approximation to the point value u(x j , t) andf j+ 1 2

is the numerical flux generated by a numerical flux policŷ

which is manually designed.

Note that the term "numerical flux policy" is a new terminology that we introduce in this paper, which is exactly the policy we shall learn using RL.

In WENO, π f works as follows.

Using the physical flux values {f j−2 , f j−1 , f j }, we could obtain a 3 th order accurate polynomial interpolationf

, where the indices {j − 2, j − 1, j} is called a 'stencil'.

We could also use the stencil {j−1, j, j+1}, {j, j+1, j+2} or {j+1, j+2, j+3} to obtain another three interpolantŝ f .

The key idea of WENO is to average (with properly designed weights) all these interpolants to obtain the final reconstruction:f j+

The weight w i depends on the smoothness of the stencil.

A general principal is: the smoother is the stencil, the more accurate is the interpolant and hence the larger is the weight.

To ensure convergence, we need the numerical scheme to be consistent and stable (LeVeque, 1992) .

It is known that WENO schemes as described above are consistent.

For stability, upwinding is required in constructing the flux.

The most easy way is to use the sign of the Roe speedā j+

to determine the upwind direction: ifā j+ 1 2 ≥ 0, we only average among the three interpolantsf .

Some further thoughts.

WENO achieves optimal order of accuracy (up to 5) at the smooth region of the solutions (Shu, 1998) , while lower order of accuracy at singularities.

The key of the WENO method lies in how to compute the weight vector (w 1 , w 2 , w 3 , w 4 ), which primarily depends on the smoothness of the solution at local stencils.

In WENO, such smoothness is characterized by handcrafted formula, and was proven to be successful in many practical problems when coupled with high-order temporal discretization.

However, it remains unknown whether there are better ways to combine the stencils so that optimal order of accuracy in smooth regions can be reserved while, at the same time, higher accuracy can be achieved near singularities.

Furthermore, estimating the upwind directions is another key component of WENO, which can get quite complicated in high-dimensional situations and requires lots of logical judgments (i.e. "if/else").

Can we ease the (some time painful) coding and improve the estimation at the aid of machine learning?

In this section we present how to employ reinforcement learning to solve the conservation laws given by Eq.1.

To better illustrate our idea, we first show in general how to formulate the process of numerically solving a conservation law into an MDP.

We then discuss how to incorporate a policy network with the WENO scheme.

Our policy network targets at the following two key aspects of WENO: (1) Can we learn to choose better weights to combine the constructed fluxes?

(2) Can we learn to automatically judge the upwind direction, without complicated logical judgments?

Compute the numerical fluxf

j+s ), e.g., using the WENO scheme

), e.g., using the Euler scheme

As shown in Algorithm 1, the procedure of numerically solving a conservation law is naturally a sequential decision making problem.

The key of the procedure is the numerical flux policy π f and the temporal scheme π t as shown in line 6 and 8 in Algorithm 1.

Both policies could be learned using RL.

However, in this paper, we mainly focus on using RL to learn the numerical flux policy π f , while leaving the temporal scheme π t with traditional numerical schemes such as the Euler scheme or the Runge-Kutta methods.

A quick review of RL is given in the appendix.

Now, we show how to formulate the above procedure as an MDP and the construction of the state S, action A, reward r and transition dynamics P .

Algorithm 2 shows in general how RL is incorporated into the procedure.

In Algorithm 2, we use a single RL agent.

Specifically, when computing U n j :

• The state for the RL agent is s

j+s ), where g s is the state function.

• In general, the action of the agent is used to determine how the numerical fluxesf n j+ 1 2 andf n j− 1 2 is computed.

In the next subsection, we detail how we incorporate a n j to be the linear weights of the fluxes computed using different stencils in the WENO scheme.

•

The reward should encourage the agent to generate a scheme that minimizes the error between its approximated value and the true value.

Therefore, we define the reward function as r n j = g r (U n j−r−1 − u n j−r−1 , · · · , U n j+s − u n j+s ), e.g., a simplest choice is g r = −|| · || 2 .

• The transition dynamics P is fully deterministic, and depends on the choice of the temporal scheme at line 10 in Algorithm 2.

Note that the next state can only be constructed when we have obtained all the point values in the next time step, i.e., s n+1 j = g s (U n j−r−1 , ..., U n j+s ) does not only depends on action a n j , but also on actions a n j−r−1 , ..., a n j+s (action a n j can only determine the value U n j ).

This subtlety can be resolved by viewing the process under the framework of multi-agent RL, in which at each mesh point j we use a distinct agent A RL j , and the next state s n+1 j = g s (U n j−r−1 , ..., U n j+s ) depends on these agents' joint action a n j = (a n j−r−1 , ..., a n j+s ).

However, it is impractical to train J different agents as J is usually very large, therefore we enforce the agents at different mesh point j to share the same weight, which reduces to case of using just a single agent.

The single agent can be viewed as a counterpart of a human designer who decides on the choice of a local scheme based on the current state in traditional numerical methods.

Compute the action a

), e.g., the Euler scheme

Compute the reward r .

14 Return the well-trained RL policy π RL .

We now present how to transfer the actions of the RL policy to the weights of WENO fluxes.

Instead of directly using π RL to generate the numerical flux, we use it to produce the weights of numerical fluxes computed using different stencils in WENO.

Since the weights are part of the configurations of the WENO scheme, our design of action essentially makes the RL policy a meta-learner, and enables more stable learning and better generalization power than directly generating the fluxes.

Specifically, at point x j (here we drop the time superscript n for simplicity), to compute the numerical fluxf j− .

Note that the determination of upwind direction is automatically embedded in the RL policy since it generates four weights at once.

For instance, when the roe speedā j+ ≈ 0.

Note that the upwind direction can be very complicated in a system of equations or in the high-dimensional situations, and using the policy network to automatically embed such a process could save lots of efforts in algorithm design and implementation.

Our numerical experiments show that π RL can indeed automatically determine upwind directions for 1D scalar cases.

Although this does not mean that it works for systems and/or in high-dimensions, it shows the potential of the proposed framework and value for further studies.

In this section, we describe training and testing of the proposed RL conservation law solver and compare it with WENO.

More comparisons and discussions can be found in the appendix.

In this subsection, we explain the general training setup.

We train the RL policy network on the Burger's equation, whose flux is computed as f (u) = 1 2 u 2 .

In all the experiments, we set the left-shift r = 2 and the right shift s = 3.

The state function g s (s j ) = g s (U j−r−1 , ..., U j+s ) will generate two vectors: s l = (f j−r−1 , ..., f j+s−1 ,ā j− respectively.

s l and s r will be passed into the same policy neural network π RL θ to produce the desired actions, as described in section 3.2.

The reward function g r simply computes the infinity norm, i.e., g r (U j−r−1 −

u j−r−1 , ..., U j+s − u j+s ) = −||(U j−r−1 −

u j−r−1 , ..., U j+s − u j+s )|| ∞ .

The policy network π RL θ is a feed-forward Multi-layer Perceptron with 6 hidden layers, each has 64 neurons and use Relu (Goodfellow et al., 2016) as the activation function.

We use the Deep Deterministic Policy Gradient Algorithm (Lillicrap et al., 2015) to train the RL policy.

To guarantee the generalization power of the trained RL agent, we randomly sampled 20 initial conditions in the form u 0 (x) = a + b · func(cπx), where |a| + |b| ≤ 3.5, func ∈ {sin, cos} and c ∈ {2, 4, 6}. The goal of generating such kind of initial conditions is to ensure they have similar degree of smoothness and thus similar level of difficulty in learning.

The computation domain is −1 ≤ x ≤ 1 and 0 ≤ t ≤ 0.8 with ∆x = 0.02, ∆t = 0.004, and evolve steps N = 200 (which ensures the appearance of shocks).

When training the RL agent, we use the Euler scheme for temporal discretization.

The true solution needed for reward computing is generated using WENO on the same computation domain with ∆x = 0.001, ∆t = 0.0002 and the 4th order Runge-Kutta (RK4).

In the following, we denote the policy network that generates the weights of the WENO fluxes (as described in section 3.2) as RL-WENO.

We randomly generated another different 10 initial conditions in the same form as training for testing.

We compare the performance of RL-WENO and WENO.

We also test whether the trained RL policy can generalize to different temporal discretization schemes, mesh sizes and flux functions that are not included in training.

Table 1 and Table 2 with the 2-norm taking over all x) between the approximated solution U and the true solution u, averaged over 250 evolving steps (T = 1.0) and 10 random initial values.

Numbers in the bracket shows the standard deviation over the 10 initial conditions.

Several entries in the table are marked as '-' because the corresponding CFL number is not small enough to guarantee convergence.

Recall that training of the RL-WENO was conducted with Euler time discretization, (∆x, ∆t) = (0.02, 0.004), T = 0.8 and f (u) = 1 2 u 2 .

Our experimental results show that, compared with the high order accurate WENO (5th order accurate in space and 4th order accurate in time), the linear weights learned by RL not only achieves smaller errors, but also generalizes well to: 1) longer evolving time (T = 0.8 for training and T = 1.0 for testing); 2) new time discretization schemes (trained on Euler, tested on RK4); 3) new mesh sizes (see Table 1 and Table 2 for results of varied ∆x and ∆t); and 4) a new flux function (trained on f (u) = Table 2 ).

Figure 1 shows some examples of the solutions.

As one can see, the solutions generated by RL-WENO not only achieve the same accuracy as WENO at smooth regions, but also have clear advantage over WENO near singularities which is particularly challenging for numerical PDE solvers and important in applications.

Figure 2 shows that the learned numerical flux policy can indeed correctly determine upwind directions and generate local numerical schemes in an adaptive fashion.

More interestingly, Figure 2 further shows that comparing to WENO, RL-WENO seems to be able to select stencils in a different way from it, and eventually leads to a more accurate solution.

This shows that the proposed RL framework has the potential to surpass human experts in designing numerical schemes for conservation laws.

In this paper, we proposed a general framework to learn how to solve 1-dimensional conservation laws via deep reinforcement learning.

We first discussed how the procedure of numerically solving conservation laws can be naturally cast in the form of Markov Decision Process.

We then elaborated how to relate notions in numerical schemes of PDEs with those of reinforcement learning.

In particular, we introduced a numerical flux policy which was able to decide on how numerical flux should be designed locally based on the current state of the solution.

We carefully design the action of our RL policy to make it a meta-learner.

Our numerical experiments showed that the proposed RL based solver was able to outperform high order WENO and was well generalized in various cases.

As part of the future works, we would like to consider using the numerical flux policy to inference more complicated numerical fluxes with guaranteed consistency and stability.

Furthermore, we can use the proposed framework to learn a policy that can generate adaptive grids and the associated numerical schemes.

Lastly, we would like consider system of conservation laws in 2nd and 3rd dimensional space.

A COMPLEMENTARY EXPERIMENTS

We first note that most of the neural network based numerical PDE solvers cited in the introduction requires retraining when the initialization, terminal time, or the form of the PDE is changed; while the proposed RL solver is much less restricted as shown in our numerical experiments.

This makes proper comparisons between existing NN-based solvers and our proposed solver very difficult.

Therefore, to demonstrate the advantage of our proposed RL PDE solver, we would like to propose a new SL method that does not require retraining when the test setting (e.g. initialization, flux function, etc.) is different from the training.

However, as far as we are concerned, it is challenging to design such SL methods without formulating the problem into an MDP.

One may think that we can use WENO to generate the weights for the stencil at a particular grid point on a dense grid, and use the weights of WENO generated from the dense grid as the label to train a neural network in the coarse grid.

But such setting has a fatal flaw in that the stencils computed in the dense grids are very different from those in the coarse grids, especially near singularities.

Therefore, good weights on dense grids might perform very poorly on coarse grids.

In other words, simple imitation of WENO on dense grids is not a good idea.

One might also argue that instead of learning the weights of the stencils, we could instead generate the discrete operators, such as the spatial discretization of (u), etc., on a dense grid, and then use them as labels to train a neural network in the supervised fashion on a coarse grid.

However, the major problem with such design is that there is no guarantee that the learned discrete operators obey the conservation property of the equations, and thus they may also generalize very poorly.

After formulating the problem into a MDP, there is indeed one way that we can use back-propagation (BP) instead of RL algorithms to optimize the policy network.

Because all the computations on using the stencils to calculate the next-step approximations are differentiable, we can indeed use SL to train the weights.

One possible way is to minimize the error (e.g. 2 norm) between the approximated and the true values, where the true value is pre-computed using a more accurate discretization on a fine mesh.

The framework to train the SL network is described in Algorithm 3.

Note that the framework to train the SL network is essentially the same as that of the proposed RL-WENO (Algorithm 2).

The only difference is that we train the SL network using BP and the RL network using DDPG.

Compute

), e.g., the Euler scheme U However, we argue that the main drawback of using SL (BP) to optimize the stencils in such a way is that it cannot enforce long-term accuracy and thus cannot outperform the proposed RL-WENO.

To support such claims, we have added experiments using SL to train the weights of the stencils, and the results are shown in table 3 and 4.

The SL policy is trained till it achieves very low loss (i.e., converges) in the training setting.

However, as shown in the table, the SL-trained policy does not perform well overall.

To improve longer time stability, one may argue that we could design the loss of SL to be the accumulated loss over multiple prediction steps, but in practice as the dynamics of our problem (computations for obtaining multiple step approximations) is highly non-linear, thus the gradient flow through multiple steps can be highly numerically unstable, making it difficult to obtain a decent result.

As mentioned in section 2.2, WENO itself already achieves an optimal order of accuracy in the smooth regions.

Since RL-WENO can further improve upon WENO, it must have obtained higher accuracy especially near singularities.

Here we provide additional demonstrations on how RL-WENO performs in the smooth/singular regions.

We run RL-WENO and WENO on a set of initial conditions, and record the approximation errors at every locations and then separate the errors in the smooth and singular regions for every time step.

We then compute the distribution of the errors on the entire spatial-temporal grids with multiple initial conditions.

The results are shown in figure 3 .

In figure 3 , the x-axis is the logarithmic (base 10) value of the error and the y-axis is the number of grid points whose error is less than the corresponding value on the x-axis, i.e., the accumulated distribution of the errors.

The results show that RL-WENO indeed performs better than WENO near singularities.

RL-WENO even achieves better accuracy than WENO in the smooth region when the flux function is

A.3 INFERENCE TIME OF RL-WENO AND WENO

In this subsection we report the inference time of RL-WENO and WENO.

Although the computation complexity of the trained RL policy (a MLP) is higher than that of WENO, we could parallel and accelerate the computations using GPU.

Our test is conducted in the following way: for each grid size ∆x, we fix the initial condition as u 0 (x) = 1 + cos(6πx), the evolving time T = 0.8 and the flux function f = u 2 .

We then use RL-WENO and WENO to solve the problem 20 times, and report the average running time.

For completeness, we also report the relative error of RL-WENO and WENO in each of these grid sizes in table 6.

Note that the relative error is computed on average of several initial functions, and our RL-WENO policy is only trained on grid (∆x, ∆t) = (0.02, 0.004).

For RL-WENO, we test it on both CPU and on GPU; For WENO, we test it purely on CPU, with a well-optimized version (e.g., good numpy vectorization in python), and a poor-implemented version (e.g., no vectorization, lots of loops).

The CPU used for the tests is a custom Intel CORE i7, and the GPU is a custom NVIDIA GTX 1080.

The results are shown in From the table we can tell that as ∆x decreases, i.e., as the grid becomes denser, all methods, except for the RL-WENO (GPU), requires significant more time to finish the computation.

The reason that the time cost of the GPU-version of RL-WENO does not grow is that on GPU, we can compute all approximations in the next step (i.e., to compute (U , which dominates the computation cost of the algorithm) together in parallel.

Thus, the increase of grids does not affect much of the computation time.

Therefore, for coarse grid, well-optimized WENO indeed has clear speed advantage over RL-WENO (even on GPU), but on a much denser grid, RL-WENO (GPU) can be faster than well-optimized WENO by leveraging the paralleling nature of the algorithm.

B REVIEW OF REINFORCEMENT LEARNING B.1 REINFORCEMENT LEARNING Reinforcement Learning (RL) is a general framework for solving sequential decision making problems.

Recently, combined with deep neural networks, RL has achieved great success in various tasks such as playing video games from raw screen inputs (Mnih et al., 2015) , playing Go (Silver et al., 2016) , and robotics control (Schulman et al., 2017) .

The sequential decision making problem RL tackles is usually formulated as a Markov Decision Process (MDP), which comprises five elements: the state space S, the action space A, the reward r : S × A → R, the transition probability of the environment P : S × A × S → [0, 1], and the discounting factor γ.

The interactions between an RL agent and the environment forms a trajectory τ = (s 0 , a 0 , r 0 , ..., s T , a T , r T , ...).

The return of τ is the discounted sum of all its future rewards:

Similarly, the return of a state-action pair (s t , a t ) is:

A policy π in RL is a probability distribution on the action A given a state S: π : S × A → [0, 1].

We say a trajectory τ is generated under policy π if all the actions along the trajectory is chosen following π, i.e., τ ∼ π means a t ∼ π(·|s t ) and s t+1 ∼ P (·|s t , a t ).

Given a policy π, the value of a state s is defined as the expected return of all the trajectories when the agent starts at s and then follows π:

Similarly, the value of a state-action pair is defined as the expected return of all trajectories when the agent starts at s, takes action a, and then follows π:

As aforementioned in introduction, in most RL algorithms the policy π is optimized with regards to the values Q π (s, a), thus naturally guarantees the long-term accumulated rewards (in our setting, the long-term accuracy of the learned schemes).

Bellman Equation, one of the most important equations in RL, connects the value of a state and the value of its successor state: The goal of RL is to find a policy π to maximize the expected discounted sum of rewards starting from the initial state s 0 , J(π) = E s0∼ρ [V π (s 0 )], where ρ is the initial state distribution.

If we parameterize π using θ, then we can optimize it using the famous policy gradient theorem:

where ρ π θ is the state distribution deduced by the policy π θ .

In this paper we focus on the case where the action space A is continuous, and a lot of mature algorithms has been proposed for such a case, e.g., the Deep Deterministic Policy Gradient (DDPG) (Lillicrap et al., 2015) , the Trust Region Policy Optimization algorithm (Schulman et al., 2015) , and etc.

<|TLDR|>

@highlight

We observe that numerical PDE solvers can be regarded as Markov Desicion Processes, and propose to use Reinforcement Learning to solve 1D scalar Conservation Laws