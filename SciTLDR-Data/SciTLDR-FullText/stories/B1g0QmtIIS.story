Reservoir computing is a powerful tool to explain how the brain learns temporal sequences, such as movements, but existing learning schemes are either biologically implausible or too inefficient to explain animal performance.

We show that a network can learn complicated sequences with a reward-modulated Hebbian learning rule if the network of reservoir neurons is combined with a second network that serves as a dynamic working memory and provides a spatio-temporal backbone signal to the reservoir.

In combination with the working memory, reward-modulated Hebbian learning of the readout neurons performs as well as FORCE learning, but with the advantage of a biologically plausible interpretation of both the learning rule and the learning paradigm.

Learning complex temporal sequences that extend over a few seconds -such as a movement to grab a bottle or to write a number on the blackboard -looks easy to us but is challenging for computational brain models.

A common framework for learning temporal sequences is reservoir computing (alternatively called liquid computing or echo-state networks) [1, 2, 3] .

It combines a reservoir, a recurrent network of rate units with strong, but random connections [4] , with a linear readout that feeds back to the reservoir.

Training of the readout weights with FORCE, a recursive least-squares estimator [1] , leads to excellent performance on many tasks such as motor movements.

The FORCE rule is, however, biologically implausible: update steps of synapses are rapid and large, and require an immediate and precisely timed feedback signal.

A more realistic alternative to FORCE is the family of reward-modulated Hebbian learning rules [5, 6, 7] , but plausibility comes at a price: when the feedback (reward minus expected reward) is given only after a long delay, reward-modulated Hebbian plasticity is not powerful enough to learn complex tasks.

Here we combine the reservoir network with a second, more structured network that stores and updates a two-dimension continuous variable as a "bump" in an attractor [8, 9] .

The activity of the attractor network acts as a dynamic working memory and serves as input to the reservoir network ( fig. 1 ).

Our approach is related to that of feeding an abstract oscillatory input [10] or a "temporal backbone signal" [11] into the reservoir in order to overcome structural weaknesses of reservoir computing that arise if large time spans need to be covered.

In computational experiments, we show that a dynamic working memory that serves as an input to a reservoir network facilitates reward-modulated Hebbian learning in multiple ways: it makes a biologically plausible three-factor rule as efficient as FORCE; it admits a delay in the feedback signal; and it allows a single reservoir network to learn and perform multiple tasks.

Our architecture is simple: the attractor network (the "memory") receives some task-specific input and produces a robust two-dimensional neural trajectory; the reservoir network (the "motor cortex") shapes its dynamics with this trajectory, and produces a potentially high-dimensional output ( fig. 1 ).

Figure 1: Model architecture: a moving 2D bump (left: activity bump (red) surrounded by inactive neurons (blue)) in an attractor network (left circle with two bump trajectories) projects to a reservoir (right circle); the output z(t) is read out from the reservoir and approximates the target function.

Attractor network.

Following [9] , the bump attractor consists of 2500 neurons evolving as

where x is the vector of firing rates evolving with time constant τ m , e is the task-specific external input, h is an adaptation variable with time constant τ a and s is the strength of adaptation.

The weight matrix J = J s + J h has two parts.

The symmetric part J s creates a two-dimensional translationinvariant structure resulting in bump-like stable activity patterns, whereas J h represents structural noise.

Due to the adaptation h, the bump moves across a path defined by the initial conditions and structural noise, creating long-lasting reliable activity patterns which also depend on the input e.

Reservoir network.

The reservoir learns to approximate a target function f (t) with the output z(t) by linearly combining the firing rate r with readout weights W ro : z = W ro r + η η η ≡ẑ + η η η with readout noise η η η.

We use the same number of neurons (1000) and parameters as [1, 6] ,

where u is the membrane potential, ξ ξ ξ is the firing rate noise, W attr scales attractor input with coupling c, W rec and λ regulate chaotic activity [4] , and W fb implements the feedback loop.

Learning rule.

We use the reward-modulated Hebbian rule of [6] for the readout weights W ro ,

wherex denotes low-pass filtering of x, such that z(t) −z(t) ≈ η η η(t).

The reward modulation M (t) tracks performance P (t) as

The update rule is an example of a NeoHebbian three-factor learning rule [12] and mimics gradient descent if we ignore the feedback loop [5] .

For model details, see appendix A.

In fig. 2 , the learning rules are compared on 50 target functions sampled from a Gaussian Process (GP) with exponential squared kernel (σ 2 = 10 4 to match the complexity of hand-picked functions from [1, 6] ).

After each training period, we measure performance with normalized cross-correlation between the output and the target (ranging from -1 to 1, where 1 is a perfect match) on a single trial with frozen weights.

Details are provided in appendix A; code: https://github.com/neuroai-workshopanon-1224113/working-memory-facilitating-reservoir-learning.

When tested on one-second signals similar to those of [1, 6] (two insets in fig. 2 ), the full network with attractor input and reward-modulated Hebbian learning learns faster and more reliably than reward-modulated Hebbian learning without the input from the attractor network.

After about 90 training trials, the full network achieves the performance of the FORCE rule (for which training error approaches one in the first trial, [1] , while test error does so after 30-50 trials, [6] ; fig. 2A ).

For target signals that extend over 10 seconds (same smoothness of the target functions, two insets in fig. 2B ), the reward-modulated Hebbian rule achieves a performance of 1 after 200 trials if combined with input from the attractor network ( fig. 2B ) but fails completely without the attractor network (tuning of the hyperparameters on a logarithmic scale did not help; data not shown).

Thus a threefactor learning rule succeeds to learn complex tasks if combined with a temporally structured input from the attractor network.

FORCE learning needs a feedback signal at every time step.

Standard reward-modulated Hebbian learning can support very small delays, but fails if updates are less frequent than every few ms [6] .

In our approach ( fig. 2C ), proposed updates are summed up in the background, but applied only at the end of a one-second trial.

We find that even with such a temporally sparse update, learning is still possible.

The input from the dynamic working memory is necessary to achieve this task: when the strength of the input from the attractor network gradually decreases, performance drops; in the total absence of attractor input (c = 0.0; note that the reservoir still receives weak input noise) learning completely fails.

Strikingly, delayed updates do not hurt performance, and the system achieves high (> 0.9) cross-correlation in fewer than 100 training trials if the input from the attractor network is strong enough.

The transient drop in performance shortly after the start in fig. 2C is likely due to W ro = 0 in the beginning, meaning that the output is uncorrelated with the firing rates, and therefore the cumulative weight update does not approximate gradient information.

It is well known that reservoir networks can learn multiple tasks given different initial conditions with both FORCE [1] and the reward-modulated Hebbian rule [6] .

We want to check whether this also holds for our approach.

We conjecture that different inputs to the attractor network generate unique neural trajectories [9] that can be exploited by the reservoir network.

To test this hypothesis, we train the network to produce hand-written digits.

The static input to the attractor comes from the pre-processed MNIST dataset (network inputs are taken from one of the last layers of a deep network trained to classify MNIST) in order to provide a realistic input to the attractor network which transforms the static input into dynamic trajectories (noiseless, fig. 3B , and noisy, fig. 3D ).

We record 50 attractor trajectories used for training (used 4 times each, resulting in 2000 training trials) and 50 for testing of each digit (1 second each), where each trajectory corresponds to a distinct input pattern.

The reservoir learns a single drawing for each class.

The variance of the structural noise in the attractor network is 3 times larger compared to the previous experiments in order to produce more robust bump trajectories ( fig. 3D ).

The reward-modulated Hebbian rule masters 10 out of 10 digits when driven by a noiseless input from the attractor network ( fig. 3A) .

In the presence of noise in the attractor network ( fig. 3D ), the performance is imperfect for "five" and "six" (fig. 3C ).

We checked that FORCE learning with the same noisy input did not improve the performance (data not shown).

Note that a linear readout of the attractor (without the reservoir) would be insufficient: first, sometimes single digit trajectories are very dissimilar (e.g. the different zero's in fig. 3D) ; second, at points where trajectories cross each other, a delay-less linear readout must produce the same output, no matter what the digit is.

We showed that a dynamic working memory can facilitate learning of complex tasks with biologically plausible three-factor learning rules.

Our results indicate that, when combined with a bump attractor, reservoir computing with reward-modulated learning can be as efficient as FORCE [1] , a widely used but biologically unrealistic rule.

The proposed network relies on a limited number of trajectories in the attractor network.

To increase its capacity, a possible future direction would be to combine input from the attractor network with another, also input-specific, but transient input that would bring the reservoir into a different initial state.

In this case the attractor network would work as a time variable (as in [9] ), and the other input as the control signal (as in [1] ).

Apart from the biological relevance, the proposed method might be used for real-world applications of reservoir computing (e.g. wind forecasting [13] ) as it is computationally less expensive than FORCE.

It might also be an interesting alternative for learning in neuromorphic devices.

Simulation details.

Both networks were simulated with the Euler method with the step size dt = 1 ms.

The attractor network dynamics was recorded after a 100 ms warm up period to allow creation of the bump solution, during which it received additional input from the images in section 3.3.

Training was done consequently, without breaks in the dynamics between trials.

For testing, the network started from the preceding training state and continued with frozen weights.

After testing, the pre-training activity was restored.

The code for experiments is available at https://github.com/neuroai-workshopanon-1224113/working-memory-facilitating-reservoir-learning.

Test functions.

Gaussian process test function were drawn from

Forcing both ends of the function to be zero and denoting x = (0, T − 1) , z = (1, . . .

, T − 2) , we sample test functions as

where T is either 10 3 (short tasks) or 10 4 (long tasks).

We chose σ 2 to roughly match the complexity of targets from [6] (σ 2 = 10 4 ).

50 random functions were tested on 50 random reservoirs that nevertheless received the same attractor input (W attr was not resampled).

In section 3.3, the same reservoir was used for all runs.

The noisy input for section 3.3 was taken from an intermediate layer of a deep network trained to classify MNIST, and the noiseless input stimulated only a 5 by 5 square of neurons (unique for each digit).

Attractor network parameters.

The time constants were τ m = 30 ms, τ a = 400 ms.

Adaptation strength was s = 1.5.

The external input e was drawn independently for each neuron from a Gaussian distribution N (1, 0.0025

2 ).

In section 3.3, the task-specific input was added to the noisy one.

For the connectivity matrix J = J s + J h , the noisy part was drawn independently as (J h ) ij ∼ N (0, σ 2 /N attr ), with N attr = 2500 and σ = 2 in all experiments except for section 3.3, where we used σ = 6 for more robust trajectories.

The symmetric part arranged the neurons on a 2D grid, such that every neuron i had its coordinates x i and y i ranging from 0 to 49.

The connectivity led to mutual excitation of nearby neurons and inhibition of the distant ones,

The bump center (used in fig. 3B and D) corresponded to the mean of the activity on the torus.

Denoting activity of each neuron as r(x, y), the center on the x axis was calculated as

where "angle" computes the counterclockwise angle of a complex variable (ranging from 0 to 2π).

Reservoir network parameters.

The time constant was τ = 50 ms, and total coupling strength was λ = 1.5.

The readout weights W ro were initialized to zero.

The feedback weights were drawn independently from a uniform distribution as (W fb ) ij ∼ U(−1, 1).

Both the recurrent connections and the weights from the attractor to the reservoir were drawn independently as (W rec ) ij , (W attr ) ij ∼ N (0, 1/pN res ) · Be(p), with p = 0.1, N res = 1000, and Be being the Bernoulli distribution.

A new reservoir, and thus W rec , was sampled for each new test function.

The matrix W rec was the same for all tasks except the last one in section 3.3.

State noise ξ ξ ξ and exploratory noise η η η were generated independently from the uniform distribution as ξ i ∼ U(−0.05, 0.05), η i ∼ U(−0.5, 0.5).

When attractor was present, the reservoir neurons also received weak independent noise drawn from N (0, 0.0025).

Learning rule.

Low-pass filtering was done as x(t + dt) =x(t) + dt (x(t) −x(t))/τ f , τ f = 5 ms,x(0) = 0.

The learning rate η(t) was computed as η(t) = η 0 /(1 + t/τ l ) (η 0 = 5 · 10 −4 , τ l = 2 · 10 4 ms) and held at η 0 in section 3.2 to make conclusions independent of the decay.

@highlight

We show that a working memory input to a reservoir network makes a local reward-modulated Hebbian rule perform as well as recursive least-squares (aka FORCE)