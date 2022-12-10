Transferring knowledge across tasks to improve data-efficiency is one of the open key challenges in the area of global optimization algorithms.

Readily available algorithms are typically designed to be universal optimizers and, thus, often suboptimal for specific tasks.

We propose a novel transfer learning method to obtain customized optimizers within the well-established framework of Bayesian optimization, allowing our algorithm to utilize the proven generalization capabilities of Gaussian processes.

Using reinforcement learning to meta-train an acquisition function (AF) on a set of related tasks, the proposed method learns to extract implicit structural information and to exploit it for improved data-efficiency.

We present experiments on a sim-to-real transfer task as well as on several simulated functions and two hyperparameter search problems.

The results show that our algorithm (1) automatically identifies structural properties of objective functions from available source tasks or simulations, (2) performs favourably in settings with both scarse and abundant source data, and (3) falls back to the performance level of general AFs if no structure is present.

Global optimization of black-box functions is highly relevant for a wide range of real-world tasks.

Examples include the tuning of hyperparameters in machine learning, the identification of control parameters or the optimization of system designs.

Such applications oftentimes require the optimization of relatively low-dimensional ( 10D) functions where each function evaluation is expensive in either time or cost.

Furthermore, there is typically no gradient information available.

In this context of data-efficient global black-box optimization, Bayesian optimization (BO) has emerged as a powerful solution (Mockus, 1975; Brochu et al., 2010; Snoek et al., 2012; Shahriari et al., 2016 ).

BO's data efficiency originates from a probabilistic surrogate model which is used to generalize over information from individual data points.

This model is typically given by a Gaussian process (GP), whose well-calibrated uncertainty prediction allows for an informed explorationexploitation trade-off during optimization.

The exact manner of performing this trade-off, however, is left to be encoded in an acquisition function (AF).

There is wide range of AFs available in the literature which are designed to yield universal optimization strategies and thus come with minimal assumptions about the class of target objective functions.

To achieve optimal data-efficiency on new instances of previously seen tasks, however, it is crucial to incorporate the information obtained from these tasks into the optimization.

Therefore, transfer learning (or warm-starting) is an important and active field of research.

Indeed, in many practical applications, optimizations are repeated numerous times in similar settings, underlining the need for specialized optimizers.

Examples include hyperparameter optimization which is repeatedly done for the same machine learning model on varying datasets or the optimization of control parameters for a given system with varying physical configurations.

Following recent approaches (Swersky et al., 2013; Feurer et al., 2018; Wistuba et al., 2018) , we argue that it is beneficial to perform transfer learning for global black-box optimization in the framework of BO to retain the proven generalization capabilities of its underlying GP surrogate model.

To not restrict the expressivity of this model, we propose to implicitly encode the task structure in a specialized AF, i.e., in the optimization strategy.

We realize this encoding via a novel method which meta-learns a neural AF, i.e., a neural network representing the AF, on a set of training tasks.

The meta-training is performed using reinforcement learning, making the proposed approach applicable to the standard BO setting, where we do not assume access to objective function gradients.

Our contributions are (1) a novel transfer learning method allowing the incorporation of implicit structural knowledge about a class of objective functions into the framework of BO through learned neural AFs to increase data-efficiency on new task instances, (2) an automatic and practical metalearning procedure for training such neural AFs which is fully compatible with the black-box optimization setting, i.e, not requiring gradients, and (3) the demonstration of the efficiency and practical applicability of our approach on a challenging hardware control task, hyperparameter optimization problems, as well as a set of synthetic functions.

The general idea of improving the performance or convergence speed of a learning system on a given set of tasks through experience on similar tasks is known as learning to learn, meta-learning or transfer learning and has attracted a large amount of interest in the past while remaining an active field of research (Schmidhuber, 1987; Hochreiter et al., 2001; Thrun and Pratt, 1998; Lake et al., 2016) .

In the context of meta-learning optimization, a large body of literature revolves around learning local optimization strategies.

One line of work focuses on learning improved optimizers for the training of neural networks, e.g., by directly learning update rules (Bengio et al., 1991; Runarsson and Jonsson, 2000) or by learning controllers for selecting appropriate step sizes for gradient descent (Daniel et al., 2016) .

Another direction of research considers the more general setting of replacing the gradient descent update step by neural networks which are trained using either reinforcement learning (Li and Malik, 2016; or in a supervised fashion (Andrychowicz et al., 2016; Metz et al., 2019) .

Finn et al. (2017) , Nichol et al. (2018) , and Flennerhag et al. (2019) propose approaches for initializing machine learning models through meta-learning to be able to solve new learning tasks with few gradient steps.

We are currently aware of only one work tackling the problem of meta-learning global black-box optimization (Chen et al., 2017) .

In contrast to our proposed method, the authors assume access to gradient information and choose a supervised learning approach, representing the optimizer as a recurrent neural network operating on the raw input vectors.

Based on statistics of the optimization history accumulated in its memory state, this network directly outputs the next query point.

In contrast, we consider transfer learning applications where gradients are typically not available.

A number of articles address the problem of increasing BO's data-efficiency via transfer learning, i.e., by incorporating data from similar optimizations on source tasks into the current target task.

A range of methods accumulate all available source and target data in a single GP and make the data comparable via a ranking algorithm (Bardenet et al., 2013) , standardization or multi-kernel GPs (Yogatama and Mann, 2014) , multi-task GPs (Swersky et al., 2013) , the GP noise model (Theckel Joy et al., 2016) , or by regressing on prediction biases (Shilton et al., 2017) .

These approaches naturally suffer from the cubic scaling behaviour of GPs, which can be tackled for instance by replacing the GP model, e.g., with Bayesian neural networks (Springenberg et al., 2016) with task-specific embedding vectors or adaptive Bayesian linear regression (Perrone et al., 2018) with basis expansions shared across tasks via a neural network.

Recently, Neural Processes were proposed by Garnelo et al. (2018) as another interesting alternative for GPs.

Other approaches retain the GP surrogate model and combine individual GPs for source and target tasks in an ensemble model with the weights adjusted according to the GP uncertainties , dataset similarities , or estimates of the GP generalization performance on the target task (Feurer et al., 2018) .

Similarly, Golovin et al. (2017) form a stack of GPs by iteratively regressing onto the residuals w.r.t.

the most recent source task.

In contrast to our proposed method, many of these approaches rely on hand-engineered dataset features to measure the relevance of source data for the target task.

Such features have also been used to pick promising initial configurations for BO (Feurer et al., 2015a; b; c) .

The method being closest in spirit and capability to our approach is proposed by Wistuba et al. (2018) .

It is similar to the aforementioned ensemble techniques with the important difference that the source and target GPs are not combined via a surrogate model but via a new AF, the socalled transfer acquisition function (TAF).

This AF is defined to be a weighted superposition of the predicted improvements according to the source GPs and the expected improvement according to the target GP.

The weights are adjusted either according to the GP's uncertainty prediction (mixture-ofexperts, TAF-ME) or using a ranking-based approach (TAF-R).

Viewed in this context, our method also combines knowledge from source and target tasks in a new AF.

Our weighting is determined in a meta-learning phase and can automatically be regulated during the optimization on the target task to adapt on-line to the specific objective function at hand.

Furthermore, our method does not store and evaluate many source GPs during optimization as the knowledge from the source datasets is encoded directly in the network weights of the learned neural AF.

This allows MetaBO to incorporate large amounts of source data while the applicability of TAF is restricted to a comparably small number of source tasks.

We are aiming to find a global optimum x * ∈ arg max x∈D f (x) of some unknown bounded real-

The only means of acquiring information about f is via (possibly noisy) evaluations at points in D. Thus, at each optimization step t ∈ {1, 2, . . . }, the optimizer has to decide for the iterate x t ∈ D based on the optimization history

.

In particular, the optimizer does not have access to gradient information.

To assess the performance of global optimization algorithms, it is natural to use the simple regret

where x + t is the best input vector found by an algorithm up to and including step t. The proposed method relies on the framework of BO and is trained using reinforcement learning.

Therefore, we give a short overview of the notation used in both frameworks.

Bayesian Optimization In Bayesian optimization (BO) (Shahriari et al., 2016) , one specifies a prior belief about the objective function f and at each step t builds a probabilistic surrogate model conditioned on the current optimization history H t .

Typically, a Gaussian process (GP) (Rasmussen and Williams, 2005) is employed as surrogate model in which case the resulting posterior belief about f is given in terms of a mean function µ t (x) ≡ E { f (x) | H t } and a variance function σ 2 t (x) ≡ V { f (x) | H t }, for which closed-form expressions are available.

To determine the next iterate x t based on the belief about f given H t , a sampling strategy is defined in terms of an acquisition function (AF) α ( · | H t ) : D → R. The AF outputs a score value at each point in D such that the next iterate is defined to be given by x t ∈ arg max x∈D α ( x | H t ).

The strength of the resulting optimizer is largely based upon carefully designing the AF to trade-off exploration of unknown versus exploitation of promising areas in D. Well known general-purpose AFs are probability of improvement (PI) (Kushner, 1964) , GP-upper confidence bound (GP-UCB) (Srinivas et al., 2010) , and expected improvement (EI) (Mockus, 1975) .

Reinforcement Learning Reinforcement learning (RL) allows an agent to learn goal-oriented behavior via trial-and-error interactions with its environment (Sutton and Barto, 1998) .

This interaction process is formalized as a Markov decision process: at step t the agent senses the environment's state s t ∈ S and uses a policy π : S → P (A) to determine the next action a t ∈ A. Typically, the agent explores the environment by means of a probabilistic policy, i.e., P (A) denotes the probability measures over A. The environment's response to a t is the next state s t+1 , which is drawn from a probability distribution with density p ( s t+1 | s t , a t ).

The agent's goal is formulated in terms of a scalar reward r t = r (s t , a t , s t+1 ), which the agent receives together with s t+1 .

The agent aims to maximize the expected cumulative discounted future reward η (π) when acting according to π and starting from some state s 0 ∈ S, i.e., η (π) ≡ E π T t=1 γ t−1 r t s 0 .

Here, T denotes the episode length and γ ∈ (0, 1] is a discount factor.

We devise a global black-box optimization method that is able to automatically identify and exploit structural properties of a given class of objective functions for improved data-efficiency.

We stay within the framework of BO, enabling us to exploit the powerful generalization capabilities of a GP surrogate model.

The actual optimization strategy which is informed by this GP is classically encoded in a hand-designed AF.

Instead, we meta-train on a set of source tasks to replace this AF

Neural AF in the BO loop Policy architecture Figure 1 : Different levels of the MetaBO framework.

Left panel: structure of the training loop for meta-learning neural AFs using RL (PPO).

Middle panel: the classical BO loop with a neural AF α θ .

At test time, there is no difference to classical BO, i.e., x t is given by the arg max of the AF output.

During training, the AF corresponds to the RL policy evaluated on an adaptive set ξ t in the optimization domain.

The outputs are interpreted as logits of a categorical distribution from which the actions a t = x t are sampled.

This sampling procedure is detailed in the right panel.

We indicate by the dashed curve and small two-headed arrows that α θ is a function defined on the whole domain D which can be evaluated at arbitrary points ξ t,i to form the categorical distribution representing the policy π θ .

by a neural network but retain all other elements of the proven BO-loop (middle panel of Fig. 1 ).

To distinguish the learned AF from a classical AF α, we call such a network a neural acquisition function and denote it by α θ , indicating that it is parametrized by a vector θ.

We dub the resulting algorithm MetaBO.

The minimal set of inputs to AFs in BO is given by the pointwise GP posterior prediction µ t (x) and σ t (x).

Since transfer learning relies on the identification and comparison of relevant structure between tasks, the incorporation of additional information in the sampling strategy is crucial.

In our setting, this is achieved via extending the set of inputs to the neural AF by additional features to enable it to evaluate sample locations.

Thus, in addition to assessing the mean µ t = µ t | x and variance σ t = σ t | x at potential sample locations, the neural AF also receives the input location x itself.

Furthermore, we add to the set of input features the current optimization step t and the optimization budget T , as these features can be valuable for adjusting the exploration-exploitation trade-off (Srinivas et al., 2010) .

Therefore,

This architecture allows learning a scalable neural AF that can be evaluated at arbitrary points x ∈ D and which can be used as a plug-in feature in any state-of-the-art BO framework.

In particular, if differentiable activation functions are chosen, a neural AF constitutes a differentiable mapping D → R, allowing to utilize gradient-based optimization strategies to find its maximum when used in the BO loop during evaluation.

We further emphasize that after the training phase the resulting neural AF is fully defined, i.e., there is no need to calibrate any AF-related hyperparameters.

Let F be the class of objective functions for which we want to learn a neural acquisition function α θ .

For instance, let F be the set of objective functions resulting from different physical configurations of a laboratory experiment or the set of loss functions used for training a machine learning model evaluated on different data sets.

Often, such function classes have structure which which we aim to exploit for data-efficient optimization on further instances of related tasks.

In many relevant cases, it is straightforward to obtain approximations to F, i.e., a set of functions F which capture the most relevant properties of F (e.g., through numerical simulations or from previous hyperparameter optimization tasks (Eggensperger et al., 2013; Wistuba et al., 2018) ) but are much cheaper to evaluate.

During the offline meta-training phase, the proposed algorithm makes use of such cheap approximations to identify the implicit structure of F and to adapt θ to obtain a data-efficient optimization strategy customized to F.

Training Procedure In the general BO setting, gradients of F are assumed to be unavailable.

This is oftentimes also true for the functions in F , for instance, if F comprises numerical simulations.

Therefore, we resort to RL (PPO, (Schulman et al., 2015a) ) as the meta-algorithm, as it does not require gradients of the objective functions.

Tab.

1 translates the MetaBO-setting into RL parlance.

We aim to shape the mapping α θ (x) during meta-training in such a way that its maximum location corresponds to promising sampling points x for optimization.

The meta-algorithm PPO explores its state space using a parametrized stochastic policy π θ from which the actions a t = x t are sampled depending on the current state s t , i.e., a t ∼ π θ ( · | s t ).

As the meta-algorithm requires access to the global information contained in the GP posterior prediction, the state s t at optimization step t formally corresponds to the entire functions µ t and σ t (together with the aforementioned additional input features to the neural AF).

To connect the neural AF α θ with the policy π θ and to arrive at a practical implementation, we evaluate µ t and σ t on a discrete set of points ξ ≡ {ξ n } N n=1 ⊂ D and feed these evaluations through the neural AF α θ one at a time, yielding one scalar output value

.

These outputs are interpreted as the logits of a categorical distribution, i.e., we arrive at the policy architecture

Thus, the proposed policy evaluates the same neural acquisition function α θ at arbitrarily many input points ξ i and preferably samples points ξ i with high α θ (ξ i ).

This incentivizes the meta-algorithm to adjust θ such that promising locations ξ i are attributed high values of α θ (ξ i ).

As for all BO approaches, calculating a sufficiently fine static set of of evaluation points ξ is challenging for higher dimensional settings.

Instead, we build on the approach proposed by (Snoek et al., 2012) and continuously adapt ξ to the current state of α θ .

At each step t, α θ is first evaluated on a static and relatively coarse Sobol grid (Sobol, 1967) ξ global spanning the whole domain D. Subsequently, local maximizations of α θ from the k points corresponding to the best evaluations are started.

We refer the reader to App.

B.1 for details.

We denote the resulting set of local maxima by ξ local,t .

Finally, we define ξ = ξ t ≡ ξ local,t ∪ ξ global .

The adaptive local part of this set enables the agent to exploit what it has learned so far by picking points which look promising according to the current neural AF while the static global part maintains exploration.

The final characteristics of the learned AF are controlled through the choice of reward function during meta-training.

For the presented experiments we emphasized fast convergence to the optimum by setting r t ≡ −R t , i.e., the negative simple regret (or a logarithmically transformed version, r t ≡ − log 10 R t ).

This choice does not penalize explorative evaluations which do not yield an immediate improvement and serves as normalization of the functions f ∈ F .

We emphasize that the knowledge about the true maximum is only needed during training and that cases in which it is not even known at training time do not limit the applicability of our method, as a cheap approximation (e.g., by evaluating the function on a coarse grid) can also be utilized.

The left panel of Fig. 1 depicts the resulting training loop graphically.

The outer loop corresponds to the RL meta-training iterations, each performing a policy update step π θi → π θi+1 .

To approximate the gradients of the meta-training loss function, in the inner loop we record a batch of episodes, i.e., a set of (s t , a t , r t )-tuples, by rolling out the current policy π θi .

At the beginning of each episode, we draw some function f from the training set F and fix an optimization budget T .

In each iteration of the inner loop we determine the adaptive set ξ t and feed the state s t , i.e., the GP posterior evaluated on this set, through the policy which yields the action a t = x t .

We then evaluate f at x t and use the result to compute the reward r t and to update the optimization history:

Finally, the GP is conditioned on the updated optimization history H t+1 to obtain the next state s t+1 . .

For TAF, we evaluated both the ranking-based version (TAF-R-50) and the mixture-of-experts version (TAF-ME-50).

Results for TAF-20 are moved to App.

A.5, Fig. 12 .

MetaBO-50 outperformed EI by clear margin, especially in early stages of the optimization.

After few steps to identify the objective function, MetaBO-50 also outperforms both flavors of TAF over wide ranges of the optimization budget.

Note that MetaBO-50 was trained on the same set of source tasks as TAF.

However, MetaBO can naturally learn from the whole set of available source tasks (MetaBO).

We trained MetaBO on a wide range of function classes and compared the performance of the resulting neural AFs with the general-purpose AF expected improvement (EI) 1 as well as the transfer acquisition function framework (TAF) which proved to be the current state-of-the-art solution for transfer learning in BO in an extensive experimental study (Wistuba et al., 2018) .

We tested both the ranking-based version (TAF-R) and the mixture-of-experts version (TAF-ME).

We refer the reader to App.

A for a more detailed experimental investigation of MetaBO's performance.

MetaBO (ours) MetaBO-50 (ours) EI TAF-R-100 TAF-ME-100 Transfer to the hardware depicted in (c), 10 BO runs.

MetaBO learns robust neural AFs with very strong early-time performance and on-line adaption to the target objectives, reliably yielding stabilizing controllers after less than ten BO iterations while TAF-ME, TAF-R, and EI explore too heavily.

We move the results for M TAF = 50 to App.

A.5, Fig. 13 .

Here MetaBO benefits from its ability to learn from the whole set of available source data, while TAF's applicability is restricted to a comparably small number of source tasks.

objective functions and spread these tasks uniformly over the whole range of translations and scalings (MetaBO-50, TAF-R-50, TAF-ME-50).

We used N TAF = 100 data points for each source GP of TAF.

We also tested both flavors of TAF for M = 20 source tasks (N TAF = 50) and observed that TAF's performance does not necessarily increase with more source data, rendering the choice of source tasks cumbersome.

To avoid clutter, we moved the results for TAF-20 to App.

A.5, cf.

Fig. 12 .

Fig. 2 shows the performance on unseen functions drawn randomly from F .

MetaBO-50 outperformed EI by large margin, in particular at early stages of the optimization, by making use of the structural knowledge acquired during the meta-learning phase.

Furthermore, MetaBO-50 outperformed both flavors of TAF-50 over wide ranges of the optimization budget.

This is due to its ability to learn sampling strategies which go beyond a combination of a prior over x ∈ D and a standard AF (as is the case for TAF).

Indeed, note that MetaBO spends some initial non-greedy evaluations to identify the objective function at hand, resulting in much more efficient optimization strategies.

We investigate this behaviour further on simple toy experiments and using easily interpretable baseline methods in App.

A.1.

We further emphasize that MetaBO does not require the user to choose a suitable set of source tasks but it can naturally learn from the whole set of available source tasks by randomly picking a new source task from F at the beginning of each BO iteration and aggregating this information in the neural AF weights to obtain optimal optimization strategies.

We also trained this full version of MetaBO (labelled MetaBO) on the global optimization benchmark functions, obtaining performance comparable with MetaBO-50.

We demonstrate below that for more complex experiments, such as the simulation-to-real task, MetaBO's ability to learn from the full set of available source tasks is crucial for efficient transfer learning.

We investigate the dependence of MetaBO's performance on the number of source tasks in more detail in App.

A.3.

As a final test on synthetic functions, we evaluated the neural AFs on objective functions outside of the training distribution as this can give interesting insights into the nature of the problems under consideration.

We move the results of this experiment to App.

A.2.

Simulation-to-Real Task Sample efficiency is of special interest for the optimization of real world systems.

In cases where an approximate model of the system can be simulated, the proposed approach can be used to improve the data-efficiency on the real system.

To demonstrate this, we evaluated MetaBO on a 4D control task on a Furuta pendulum (Furuta et al., 1992) .

We applied BO to tune the four feedback gains of a linear state-feedback controller used to stabilize the pendulum in the upward equilibrium position.

To assess the performance of a given controller, we employed a logarithmic quadratic cost function (Bansal et al., 2017 ) with a penalty term if no stabilization could be achieved.

We emphasize that the cost function is rather sensitive to the control gains, resulting in a challenging black-box optimization problem.

Figure 4: Performance on two 2D hyperparameter optimization tasks (SVM and AdaBoost).

We trained MetaBO on precomputed data for 35 randomly chosen datasets and also used these as source tasks for TAF.

The remaining 15 datasets were used for this evaluation.

MetaBO learned extremely data-efficient sampling strategies on both experiments, outperforming the benchmark methods by clear margin.

Note that the optimization domain is discrete and therefore tasks can be solved exactly, corresponding to zero regret.

To meta-learn the neural AF, we employed a simple numerical simulation of the Furuta pendulum which models only the most basic physical effects.

The training distribution was then generated by sampling the free physical parameters (two lengths, two masses), uniformly on a range of 75% -125% around the measured parameters of the hardware (Quanser QUBE -Servo 2, 4 Fig. 3(c) ).

We also used this simulation to generate source tasks for TAF-50 and TAF-100 (N TAF = 200).

We move the results for TAF-50 to App.

A.5, Fig. 13 .

Fig. 3(a) show the performance on objective functions from simulation.

MetaBO learned a sophisticated sampling strategy, using its prior knowledge about the class of objective functions to adapt on-line to the target objective function which yields very strong optimization performance.

In contrast, TAF does not adapt the weighting of source and target tasks online to the specific objective function at hand which leads to excessive explorative behaviour on this complex class of objective functions.

By comparing the performance of MetaBO and MetaBO-50 in simulation, we find that on this challenging optimization problem, our architecture's ability to automatically incorporate large amounts of source data is indeed beneficial.

The results in App.

A.3 indicate that this task indeed requires large amounts of source data to be solved efficiently.

This is underlined by the results on the hardware, on which we evaluated the full version of MetaBO (which showed more promising performance in simulation than MetaBO-50) and the baseline AFs obtained by training on data from simulation without any changes.

Fig. 3(b) shows that MetaBO learned a neural AF which generalizes well from the simulated objectives to the hardware task and was thereby able to rapidly adjust to its specific properties.

This resulted in very data-efficient optimization on the target system, consistently yielding stabilizing controllers after less than ten BO iterations.

In comparison, the benchmark AFs required many samples to identify promising regions of the search space and therefore did not reliably find stabilizing controllers within the budget of 25 optimization steps.

As it provides interesting insights into the nature of the studied problem, we investigate MetaBO's generalization performance to functions outside of the training distribution in App.

A.2.

We emphasize, however, that the intended use case of our method is on unseen functions drawn from the training distribution.

Indeed, this distribution obtained from simulation can be modelled in such a way that the true system parameters lie inside of it with high confidence.

Hyperparameter Optimization We also considered two hyperparameter optimization (HPO) problems on RBF-based SVMs and AdaBoost.

As proposed in Feurer et al. (2018) and Wistuba et al. (2018) , we used precomputed results of training these models on 50 datasets with 144 parameter configurations (RBF kernel parameter, penalty parameter C) for the SVMs and 108 configurations (number of product terms, number of iterations) for AdaBoost.

We randomly split this data into 35 source datasets used for training of MetaBO as well as for TAF and evaluated the resulting optimizers on the remaining 15 datasets.

5 We emphasize that MetaBO did not use more source data than TAF in this experiment, underlining again its broad applicability in practice to application scopes with both scarse and abundant source data.

The results (Fig. 4) show that MetaBO learned extremely data-efficient neural AFs which surpassed EI und TAF on both experiments.

General Function Classes Finally, we evaluated the performance of MetaBO on function classes without any particular structure structure.

In such cases, it is desirable to obtain neural AFs which fall back on the performance level of general-purpose AFs such as EI.

We generated the training set by sampling functions from GP priors with squared exponential and Matern-5/2 kernels with varying lengthscales.

The results of our experiments show that MetaBO indeed performs at least on-par with the general-purpose AF EI, cf.

App.

A.4, Fig. 11 .

We introduced MetaBO, a novel approach for transfer learning in the framework of BO.

Via a flexible meta-learning approach we inject prior knowledge directly into the optimization strategy of BO using neural AFs.

Our experiments on several real-world optimization tasks show that our method consistently outperforms the popular general-purpose AF EI as well as the state-of-the-art solution TAF for warmstarting BO, for instance in simulation-to-real settings or on hyperparameter search tasks.

Our approach is broadly applicable to a wide range of practical problems, covering both the cases of scarse and abundant source data.

The resulting neural AFs generalize well beyond the training distribution, allowing our algorithm to perform robustly unseen problems.

In future work, we aim to tackle the multi-task multi-fidelity setting (Valkov et al., 2018) , where we expect MetaBO's sample efficiency to be of high impact.

We provide additional experimental results to demonstrate that MetaBO's neural AFs learn representations that go beyond standard AFs combined with a prior over x ∈ D.

To obtain intuition about the kind of search strategies MetaBO is able to learn, we devised two classes of simple one-dimensional toy objective functions.

The first class of objective functions (Rhino-1, cf.

Fig. 5 ) is generated by applying random translations sampled uniformly from t ∈ [−0.2, 0.2] to a fixed function f R1 which is given by the superposition of two non-normalized Gaussian bumps with different heights and widths but fixed distance,

where we define N ( x| µ, σ) ≡ exp(−1/2 · (x − µ) 2 /σ 2 ).

The second class of objective functions (Rhino-2, cf.

Fig. 6 ) is given by uniformly sampling the parameter h ∈ [0.6, 0.9] of the function

In both of these experiments it is intuitively clear that the optimal search strategy involves a nongreedy step as the first evaluation to identify the specific instance of the function class.

Indeed, in all instances of the function classes, the smaller and wider bumps overlap and encode information about the position of the sharp global optimum.

Therefore, an optimal strategy spends the first evaluation at a fixed position x 0 where all smaller and wider bumps have non-negligible heights y 0 .

Then, for both function classes, the global optimum x * can be determined exactly from y 0 , such that x * can be found in the second step.

Figs. 5, 6 show that MetaBO is indeed able to find such non-greedy optimization strategies, which go far beyond a simple combination of a prior over x with a standard AF.

As mentioned in the main part of this paper, we suppose that MetaBO employs similar strategies on more complex function classes.

For instance, on the global optimization benchmark functions (Fig. 2) we observe that MetaBO consistently starts off higher than the pre-informed TAF which suggests that it learned to spend a few non-greedy evaluations at the beginning to identify the specific instance of the target function.

Additional Baseline Methods To provide further evidence that MetaBO's neural AFs learn representations that go beyond a simple prior over x ∈ D, we show results for two additional baseline AFs which rely on a naive combination of such a prior obtained from the source tasks and a standard AF.

We define the AF GMM-UCB as the following convex combination of a Gaussian Mixture Model (GMM) with n comps components fitted on the best designs from each of the M source tasks:

Here, UCB is defined as

and we choose β = 2 as is common in BO.

Note that GMM-UCB is similar in spirit to the TAFapproach used as a baseline in the main part of this paper.

However, TAF uses more principled methods (TAF-ME, TAF-RANKING) to adaptively determine the weights between its prior and posterior parts (observed improvement on the source tasks and EI on the target task, respectively).

Furthermore, we define EPS-GREEDY as the AF which samples in each optimization step with probability without replacement from the set of best designs of each of the source tasks and uses standard EI with probability 1 − .

To obtain optimal performance of GMM-UCB and EPS-GREEDY, we chose the parameters for these methods by grid search on the test set w.r.t.

the median simple regret summed from t = 0 to t = T = 30.

For GMM-UCB we tested w on 10 linearly spaced points in [0.0, 1.0] as well as a schedule which reduces w from 1.0 to 0.0 over the course of one episode.

Furthermore, we tested numbers of GMM-components n comp ∈ {1, 2, 3, 4, 5}. For EPS-GREEDY we also evaluated on 10 linearly spaced points in [0.0, 1.0] and a schedule which reduces from 1.0 to 0.0 over an episode.

Figure 5: Visualization of three episodes from the one-dimensional Rhino-1 task.

Each instance of this task is generated by randomly translating an objective function with two peaks of different heights and widths.

The distance between the local and global optimum is constant for each instance.

Each column of this figure correspond to one episode with three optimization steps.

The uppermost row corresponds to the prior state before the objective function was queried.

The fourth row depicts the state after three evaluations.

Each subfigure shows the GP mean (dashed blue line), GP standard deviation (blue shaded area), and the ground truth function (black) in the upper panel as well as the neural AF in the lower panel.

Dashed red lines indicate the maxima of the ground truth function and of the neural AF.

Red and green crosses indicate the recorded data (the red cross corresponds to the most recent data point).

MetaBO learns a sophisticated sampling strategy, spending a non-greedy evaluation at a position where the smaller but wider peaks overlap for every instance of the function class at the beginning of each episode to gain information about the location of the global optimum.

Using this strategy, MetaBO is able to find the global optimum very efficiently.

Figure 6: Visualization of three episodes from the one-dimensional Rhino-2 task.

Each instance of this task is generated by sampling the height h of a wide bump at a fixed location x = 0.2 and placing a sharp peak at x = h. Each column of this figure correspond to one episode with two optimization steps.

The uppermost row corresponds to the prior state before the objective function was queried.

The third row depicts the state after two evaluations.

Each subfigure shows the GP mean (dashed blue line), GP standard deviation (blue shaded area), and the ground truth function (black) in the upper panel as well as the neural AF in the lower panel.

Dashed red lines indicate the maxima of the ground truth function and of the neural AF.

Red and green crosses indicate the recorded data (the red cross corresponds to the most recent data point).

MetaBO learns a sophisticated sampling strategy, spending a non-greedy evaluation at x ≈ 0.2 to gain information about the location of the global optimum.

Using this strategy, MetaBO is able to find the global optimum very efficiently.

We present results for two additional baseline methods (GMM-UCB, EPS-GREEDY) which rely on a simple combination of prior knowledge from M = 50 source tasks and can thus be easily interpreted.

As MetaBO produces more sophisticated search strategies, these approaches are not able to surpass MetaBO's performance.

In Fig. 7 we display the performance of GMM-UCB and EPS-GREEDY on the global optimization benchmark functions Branin, Goldstein-Price, and Hartmann-3 with the optimal parameter configurations (cf.

Tab.

2) and with M = 50 source tasks.

MetaBO outperforms both GMM-UCB and EPS-GREEDY which provides additional evidence that neural AFs learn representations which go beyond a simple combination of standard AFs with a prior over x ∈ D.

As described in the main part of this paper, MetaBO's primary use case is transfer learning, i.e., to speed up optimization on target functions similar to the source objective functions.

Put differently, we are mainly interested in MetaBO's performance on unseen functions drawn from the training distribution.

Nevertheless, studying MetaBO's generalization performance to functions outside of the training distribution can give interesting insights into the nature of the tasks we considered in the main part.

Therefore, we present here a study of MetaBO's generalization performance on the global optimization benchmark functions (Fig. 8) as well as on the simulation-to-real experiment (Fig. 9) . , red square) on Branin, Goldstein-Price, and Hartmann-3.

We evaluated the neural AFs on 100 test distributions with disjoint ranges of translations and scalings, each corresponding to one tile of the heatmap.

The x-and y-labels of each tile denote the lower bounds of the translations t and scalings s of the respective test distribution from which the parameters were sampled uniformly (for each dimension we sampled the translation and its sign independently).

The color encodes the number of optimization steps required to reach a given regret threshold.

White tiles indicate that this threshold could not be reached withtin T = 30 optimization steps.

The regret threshold was fixed for each function separately: we set it to the 1%-percentile of function evaluations on a Sobol grid of one million points in the domain of the original objective functions.

Number of steps to regret threshold I n t e n d e d u s e c a s e Figure 9 : Generalization of neural AFs to functions outside of the training distribution (0.75% to 1.25% of measured physical parameters, red square) on the simulation-to-real task.

We evaluated neural AFs on test distributions with disjoint ranges of physical parameters (masses and lengths of the pendulum and arm).

We sampled each physical parameter p i uniformly on

.

Therefore, f = 0.9 corresponds to the interval containing the measured parameters.

We plot f on the x-axis and the number of steps required to reach a regret threshold of R = 1.0 on the y-axis.

Following our experience, this corresponds approximately to the regret that has to be reached in simulation to allow stabilization on the real system.

We emphasize that the intended use case of MetaBO is on systems inside of the training distribution marked in red, as this distribution is chosen such that the true parameters are located inside of it with high confidence when taking into account the measurement uncertainty.

Note that for small f the system becomes very hard to stabilize (lightweight and short pendula) which is why the regret threshold cannot be reached within 30 steps for f ≤ 0.5.

We argued in the main part of this paper that a main advantage of MetaBO over existing transfer learning methods for BO is its ability to process a very large amount of source data because it does not store all available data in GP models (in contrast to TAF) but rather accumulates the data in the neural AF weights.

For tasks where source data is abundant (e.g., when it comes from simulations, cf.

Fig. 3) , this frees the user from having to select a small subset of representative source tasks by hand, which can be intricate or even impossible for complex tasks.

In addition, we showed in our experiments that MetaBO's applicability is not restricted to such cases, but also performs favourably with the same amount of source data as presented to the baseline methods on tasks which do not require a very large amount of source data to be solved efficiently (cf.

Figs. 2, 4) .

In Fig. 10 we provide further evidence for this point by plotting the performance of MetaBO for different numbers M of source tasks on the Branin function and the Furuta stabilization task in simulation.

The results indicate that on the Branin function a small number of source tasks is already sufficient to obtain strong optimization performance.

In contrast, the more complex stabilization task requires a much larger amount of source data to be solved reliably. (Fig. 2(a) ) and on the stabilization task for the Furuta pendulum in simulation (Fig. 3(a) ).

We show the number of steps MetaBO requires to reach a given performance in terms of median regret over 100 test functions in dependence of the number M of source tasks.

As in the main part of this paper, we chose a constant budget of T = 30 on the Branin function and of T = 50 on the stabilization task.

The dashed red line indicates the number of source tasks seen by the full version of MetaBO (a new function is sampled from the training distribution at the beginning of each optimization episode) at the point of convergence of meta-training.

For the Branin function we chose the regret threshold R = 10 , which corresponds to the final performance at t = 30 steps of TAF as presented in the main part of this paper ( Fig. 2(a) ).

For the Furuta stabilization task, we chose the regret threshold R = 1.0, which corresponds approximately to the regret that has to be reached in simulation to allow stabilization on the real system.

We investigated MetaBO's performance on general function classes without any specific structure except some correlation lengthscale.

We present this experiment as a sanity check to show that MetaBO falls back at least on the performance of general-purpose optimization strategies which is to be expected when there is no special structure present in the objective functions which could be exploited in a transfer learning setting.

We performed two different experiments of this type.

For the first experiment, we sampled such objective functions from a GP prior with squared-exponential (RBF) kernel with lengthscales drawn uniformly from ∈ [0.05, 0.5].

For the second experiment, we used a GP prior with Matern-5/2 kernel with the same range of lengthscales.

For the latter experiment we also used the Matern-5/2 kernel (in contrast to the RBF kernel used in all other experiments) as the kernel of the model GP to avoid model mismatch.

For both types of function classes we trained MetaBO on D = 3 dimensional tasks and excluded the x-feature to study a dimensionality-agnostic version of MetaBO.

Indeed, we evaluated the resulting neural AFs without retraining for different dimensionalities D ∈ {3, 4, 5}. The results (Fig. 11) show that MetaBO is capable of learning neural AFs which perform slightly better than or on on-par with the benchmark AFs on these general function classes.

MetaBO ( As we excluded the x-feature from the neural AF inputs during training, the resulting AFs can be applied to functions of different dimensionalities.

We evaluated each AF on D = 4 and D = 5 without retraining MetaBO.

We report simple regret w.r.t.

the best observed function value, determined separately for each function in the test set.

Global Optimization Benchmark Functions We provide the full set of results for the experiment on the global optimization benchmark functions.

In Fig. 12 we also include results for TAF with M = 20, showing that TAF's performance does not necessarily increase with more source data.

MetaBO (ours) MetaBO - .

For TAF, we evaluated both the ranking-based version (TAF-R-50) and the mixture-of-experts version (TAF-ME-50).

MetaBO-50 outperformed EI by clear margin, especially in early stages of the optimization.

After few steps to identify the objective function, MetaBO-50 also outperforms both flavors of TAF over wide ranges of the optimization budget.

Note that MetaBO-50 was trained on the same set of source tasks as TAF.

However, MetaBO can naturally learn from the whole set of available source tasks (MetaBO).

We provide the full set of results for the experiment on the global optimization benchmark functions, including the results for TAF-50, cf.

Fig. 13 .

MetaBO (ours) MetaBO-50 (ours) EI TAF-R-50 TAF-R-100 TAF-ME-50 TAF-ME-100 Transfer to the hardware depicted in (c), 10 BO runs.

MetaBO learns robust neural AFs with very strong early-time performance and on-line adaption to the target objectives, reliably yielding stabilizing controllers after less than ten BO iterations while TAF-ME, TAF-R, and EI explore too heavily.

Here MetaBO benefits from its ability to learn from the whole set of available source data, while TAF's applicability is restricted to a comparably small number of source tasks.

To foster reproducibility, we provide a detailed explanation of the settings used in our experiments and make source code available online.

In what follows, we explain all hyperparameters used in our experiments and summarize them in Tab.

4.

We emphasize that we used the same MetaBO hyperparameters for all our experiments, making our method easily applicable in practice.

Gaussian Process Surrogate Models We used the implementation GPy (GPy, 2012) with squaredexponential kernels with automatic relevance determination and a Gaussian noise model and tuned the corresponding hyperparameters (noise variance, kernel lengthscales, kernel signal variance) off-line by fitting a GP to the objective functions in the training and test sets using type-2 maximum likelihood.

We also used the resulting hyperparameters for the source GPs of TAF.

We emphasize that our method is fully compatible with other (on-line) hyperparameter optimization techniques, which we did not use in our experiments to arrive at a consistent and fair comparison with as few confounding factors as possible.

Baseline AFs As is standard, we used the parameter-free version of EI.

For TAF, we follow Wistuba et al. (2018) and evaluate both the ranking-based as well as the product-of-experts versions.

We detail the specific choices for the number of source tasks M and the number of datapoints N TAF contained in each source GP in the main part of this paper.

For EI we used the midpoint of the optimization domain D as initial design.

For TAF we did not use an initial design as it utilizes the information contained in the source tasks to warmstart BO.

Note that MetaBO also works without any initial design.

Maximization of the AFs Our method is fully compatible with any state-of-the-art method for maximizing AFs.

In particular our neural AFs can be optimized using gradient-based techniques.

We chose to switch off any confounding factors related to AF maximization and used a hierarchical gridding approach for all evaluations as well as during training of MetaBO.

For the experiments with continuous domains D, i.e. all experiments except the HPO task, we first put a multistart Sobol grid with N MS points over the whole optimization domain and evaluated the AF on this grid.

Afterwards, we implemented local searches from the k maximal evaluations via centering k local Sobol grids with N LS points, each spanning one unit cell of the uniform grid, around the k maximal evaluations.

The AF maximum is taken to be the maximal evaluation of the AF on these k Sobol grids.

For the HPO task, the AF maximum can be determined exactly, as the domain is discrete.

We use the trust-region policy gradient method Proximal Policy Optimization (PPO) (Schulman et al., 2017) as the algorithm to train the neural AF.

Reward Function If the true maximum of the objective functions is not known at training time, we compute R t with respect to an approximate maximum and define the reward to be given by r t ≡ −R t .

This is the case for the experiment on general function classes (GP samples) where we used grid search to approximate the maximum as well as for the simulation-to-real task on the Furuta pendulum where we used the performance of a linear quadratic regulator (LQR) controller as an approximate maximum.

For the experiments on the global optimization benchmark functions as well as on the HPO tasks, we do know the exact value of the global optimum.

In these cases, we use a logarithmic transformation of the simple regret, i.e., r t = − log 10 R t as the reward signal.

Note that we also consistently plot the logarithmic simple regret in our evaluations for these cases.

Neural AF Architecture We used multi-layer perceptrons with relu-activation functions and four hidden layers with 200 units each to represent the neural AFs. (Schulman et al., 2015b) 0.98

Value Function Network To reduce the variance of the gradient estimates for PPO, a value function V π (s t ), i.e., an estimator for the expected cumulative reward from state s t , can be employed (Schulman et al., 2015b) .

In this context, the optimization step t and the budget T are particularly informative features, as for a given sampling strategy on a given function class they allow quite reliable predictions of future regrets.

Thus, we propose to use a separate neural network to learn a value function of the form V π (s t ) = V π (t, T ).

We used the same network architecture to learn the value functions as we used for the neural AFs.

Computation Time For training MetaBO, we employed ten parallel CPU-workers to record the data batches and one GPU to perform the policy updates.

Depending on the complexity of the objective function evaluations, training a neural AF for a given function class took between approximately 30 min and 10 h on this moderately complex architecture.

We shortly list some additional details specific to the experiments presented in this article.

Simulation-to-real Task The task was to stabilize a Furuta pendulum (Furuta et al., 1992) for 5 s around the upper equilibrium position using a linear state-feedback controller.

If the controller was not able to stabilize the system or if the voltage applied to the motor exceeded some safety limit, we added a penalty term to the cost function proportional to the remaining time the pendulum would have had to be stabilized for successfully completing the task.

The numerical simulation we used to train MetaBO was based on the nonlinear dynamics equations of the Furuta pendulum and did only contain the most basic physical effects.

In particular, effects like friction and stiction were not modeled.

As the true maximum of the objective functions is not known, we used the cost accumulated by a linear quadratic regulator (LQR) controller as an approximate maximum to compute the simple regret.

Hyperparameter Optimization Tasks We performed 7-fold cross validation on the training datasets to determine at which iteration to stop the meta-training.

We determined approximate maxima on the objective functions sampled from a GP prior via grid search.

@highlight

We perform efficient and flexible transfer learning in the framework of Bayesian optimization through meta-learned neural acquisition functions.

@highlight

The authors present MetaBO which uses reinforcement learning to meta-learn the acquisition function for Bayesian Optimization, showing increasing sample efficiency on new tasks.

@highlight

The authors propose a meta-learning based alternative to standard acquisition functions (AFs), whereby a pretrained neural network outputs acquisition values as a function of hand-chosen features.