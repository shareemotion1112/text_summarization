Gated recurrent units (GRUs) were inspired by the common gated recurrent unit, long short-term memory (LSTM), as a means of capturing temporal structure with less complex memory unit architecture.

Despite their incredible success in tasks such as natural and artificial language processing, speech, video, and polyphonic music, very little is understood about the specific dynamic features representable in a GRU network.

As a result, it is difficult to know a priori how successful a GRU-RNN will perform on a given data set.

In this paper, we develop a new theoretical framework to analyze one and two dimensional GRUs as a continuous dynamical system, and classify the dynamical features obtainable with such system.

We found rich repertoire that includes stable limit cycles over time (nonlinear oscillations), multi-stable state transitions with various topologies, and homoclinic orbits.

In addition, we show that any finite dimensional GRU cannot precisely replicate the dynamics of a ring attractor, or more generally, any continuous attractor, and is limited to finitely many isolated fixed points in theory.

These findings were then experimentally verified in two dimensions by means of time series prediction.

Recurrent neural networks (RNNs) have been widely used to capture and utilize sequential structure in natural and artificial languages, speech, video, and various other forms of time series.

The recurrent information flow within RNN implies that the data seen in the past has influence on the current state of the RNN, forming a mechanism for having memory through (nonlinear) temporal traces.

Unfortunately, training vanilla RNNs (which allow input data to directly interact with the hidden state) to capture long-range dependences within a sequence is challenging due to the vanishing gradient problem BID8 .

Several special RNN architectures have been proposed to mitigate this issue, notably the long short-term memory (LSTM; BID9 ) which explicitly guards against unwanted corruption of the information stored in the hidden state until necessary.

Recently, a simplification of the LSTM called the gated recurrent unit (GRU; BID1 ) has become wildly popular in the machine learning community thanks to its performance in machine translation BID0 , speech BID16 , music BID2 , video BID4 , and extracting nonlinear dynamics underlying neural data BID15 .

As a variant of the vanilla LSTM, GRUs incorporate the use of forget gates, but lack an output gate BID5 .

While this feature reduces the number of required parameters, LSTM has been shown to outperform GRU on neural machine translation BID0 .

In addition, certain mechanistic tasks, specifically unbounded counting, come easy to LSTM networks but not to GRU networks BID18 .

Despite these empirical findings, we lack systematic understanding of the internal time evolution of GRU's memory structure and its capability to represent nonlinear temporal dynamics.

In general, a RNN can be written as h t+1 = f (h t , x t ) where x t is the current input in a sequence indexed by t, f is a point-wise nonlinear function, and h t represents the hidden memory state that carries all information responsible for future output.

In the absence of input, the hidden state h t can evolve over time on its own: DISPLAYFORM0 where f (·) := f (·, 0) for notational simplicity.

In other words, we can consider the temporal evolution of memory stored within RNN as a trajectory of a dynamical system defined by (1).

Then we can use dynamical systems theory to investigate the fundamental limits in the expressive power of RNNs in terms of their temporal features.

We develop a novel theoretical framework to study the dynamical features fundamentally attainable, in particular, given the particular form of GRU.

We then validate the theory by training GRUs to predict time series with prescribed dynamics.

The GRU uses two internal gating variables: the update gate z t which protects the d-dimensional hidden state h t ∈ R d and the reset gate r t which allows overwriting of the hidden state and controls the interaction with the input x t ∈ R p .z t = σ(W z x t +U z h t−1 + b z ) (update gate) (2) r t = σ(W r x t +U r h t−1 + b r ) (reset gate) (3) h t = (1 − z t ) tanh(W h x t +U h (r t h t−1 + b h )) + z t h t−1 (hidden state)where W z , W r , W h ∈ R d×p and U z , U r , U h ∈ R d×d are the parameter matrices, b z , b r , b h ∈ R d are bias vectors, represents the Hadamard product, and σ(z) = 1/(1 + e −z ) is the element-wise logistic sigmoid function.

Note that the hidden state is asymptotically contained within [−1, 1] d due to the saturating nonlinearities, implying if the state is initialized outside of this trapping region, it must eventually enter it in finite time and remain in it for all later time.

Note that the update gate z t controls how fast each dimension at the hidden state decays, providing an adaptive time constant for memory.

Specifically, as lim zt→0 h t = h t−1 , GRUs can implement perfect memory of the past and ignore x t .

Hence, a d-dimensional GRU is capable of keeping a near constant memory through the update gate-near constant since 0 < [z t ] j < 1, where [·] j denotes j-th component of a vector.

Moreover, the autoregressive weights (mainly U h and U r ) can support time evolving memory BID13 ) considered this a hindrance and proposed removing all complex dynamic behavior in a simplified GRU).To investigate the memory structure further, let us consider the dynamics of hidden state in the absence of input, i.e. x t = 0, ∀t, which is of the form (1).

To utilize the rich descriptive language of continuous time dynamical system theory, we consider the following continuous time limit of the (autonomous) GRU time evolution: DISPLAYFORM0 dt .

Since both σ(·) and tanh(·) are smooth, this continuous limit is justified.

The update gate z(t) again plays the role of a state-dependent time constant for memory decay.

Furthermore, since 1 − z(t) > 0, it does not change the topological structure of the dynamics (only speed).

For the following theoretical analysis sections (3 & 4), we can safely ignore the effects of z(t).

A derivation of the continuous-time GRU can be found in appendix A.

For a single GRU * (d = 1), (7) reduces to a one dimensional dynamical system where every variable is a scalar.

The expressive power of a single GRU is quite limited, as only three stability structures (topologies) exist (see appendix B): (A) a single stable node, (B) a stable node and a half-stable node, and (C) two stable nodes separated by an unstable node (see Fig. 1 ).

The corresponding temporal features are (A) decay to a fixed value, (B) decay to a fixed value, but from one direction halt at an intermediate value until perturbed, or (C) decay to one of two fixed values (bistability).

The bistability can be used to capture a binary latent state in the sequence.

It should be noted that a one dimensional continuous time autonomous system cannot exhibit oscillatory behavior, as is the case here BID7 .

Figure 1: Three possible types of one dimensional flow for a single GRU.

Whenḣ > 0, h(t) increases.

This flow is indicated by a rightward arrow.

Nodes ({h |ḣ(h) = 0}) are represented as circles and classified by their stability BID14 .The topology the GRU takes is determined by its parameters.

If the GRU begins in a region of the parameter space corresponding to (A), we can smoothly vary the parameters to transverse (B) in the parameter space, and end up at (C).

This is commonly known as a saddle-node bifurcation.

Speaking generally, a bifurcation is the change in topology of a dynamical system, resulting from a smooth change in parameters.

The point in the parameters space at which the bifurcation occurs is called the bifurcation point, and we will refer to the fixed point that changes its stability at the bifurcation point as the bifurcation fixed point.

This corresponds to the parameters underlying (B) in our previous example.

The codimension of a bifurcation is the number of parameters which must vary in order to achieve the bifurcation.

In the case of our example, a saddle-node bifurcation is codimension-1 BID12 .

Right before transitioning to (B), from (A), the flow near where the half-stable node would appear can exhibit arbitrarily slow flow.

We will refer to these as slow points BID17 .

We will see that the addition of a second GRU opens up a substantial variety of possible topological structures when compared with the use of a single GRU.

For notational simplicity, we denote the two dimensions of h as x and y. We visualize the flow fields defined by (7) in 2-dimension as phase portraits which reveal the topological structures of interest.

For starters, the phase portrait of two independent bistable GRUs can be visualized as FIG1 .

It clearly shows 4 stable states as expected, with a total of 9 stable fixed points.

This could be thought of as a continuoustime continuous-space implementation of a finite state machine with 4 states FIG1 .

The 3 types of observed fixed points-stable (sinks), unstable (sources), and saddle points-exhibit locally linear dynamics, however, the global geometry is nonlinear and their topological structures can vary depending on their arrangement.

We explored stability structures attainable by two GRUs.

Due to the relatively large number of observed topologies, this section's main focus will be on demonstrating all observed local dynamical features obtainable by two GRUs.

In addition, existence of two non-local dynamical features will be presented.

A complete catalog of all observed topologies can be found in the appendix C, along with the parameters of every phase portrait depicted in this paper.

Before proceeding, let us take this time to describe all the local dynamical features observed.

In addition to the previously mentioned three types of fixed points, two GRUs can exhibit a variety of bifurcation fixed points, resulting from regions of parameter space that separate all topologies restricted to simple fixed points (i.e stable, unstable, and saddle points).

Behaviorally speaking, these fixed points act as hybrids between the previous three, resulting in a much richer set of obtainable dynamics.

These bifurcation fixed points fall into two categories, separated by codimension.

More specifically, two GRUs have been seen to feature both codimension-1 and codimension-2 bifurcation (fixed) points.

Beginning with codimension-1, we have the saddle-node bifurcation fixed point, as expected from its existence in the single GRU case.

We can further classify these points into two types.

These can be thought of as both the fusion of a stable fixed point and a saddle point, and the fusion of an unstable fixed point and a saddle point.

We will refer to these fixed points as saddle-node bifurcation fixed points of the first kind and second kind respectively.

One type of codimension-2 bifurcation fixed point that has been observed in the two GRU system acts as the fusion of all three simple fixed points.

More specifically, these points arise from the fusion of a stable fixed point, unstable fixed point, and two saddle points.

All of these local structures are depicted in figure 3.While the existence of simple fixed points was already demonstrated (see FIG1 ).

FIG2 demonstrates the maximum number of fixed points observed in a two GRU system, for a given set of parameters.

A closer look at this system reveals its potential interpretation as a continuous analogue of 5-discrete states with input-driven transitions, similar to that depicted in figure 2, implying additional GRUs are needed for any Markov process modeled in this manner, requiring more than five discrete states.

We conjecture that the system depicted in FIG2 is the only eleven fixed point structure obtainable with two GRUs, as all observed structures containing the same number of fixed points are topologically equivalent to one another.

The addition of bifurcation fixed points opens the door to dynamically realize more sophisticated models.

Take for example the four state system depicted in FIG2 .

If the hidden state is set to initialize in the first quadrant of phase space, the trajectory will flow towards the codimension-2 bifurcation fixed point at the origin.

Introducing noise through the input will stochastically cause the trajectory to approach the stable fixed point at (-1,-1) either directly, or by first flowing into one of the two saddle-node bifurcation fixed points of the first kind.

Models of this sort can be used in a variety of applications, such as neural decision making (Wong & Wang FORMULA7 , BID3 ).

We will begin our investigation into the non-local dynamics observed with two GRUs by showing the existence of homoclinic orbits.

A trajectory initialized on a homoclinic orbit will approach the same fixed point in both forward and backward time.

We observe that two GRUs can exhibit one or two bounded planar regions of homoclinic orbits for a given set of parameters, as shown in FIG3 and 4B respectively.

Any trajectory initialized in one of these regions will flow into the codimension-2 bifurcation fixed point at the origin, regardless of which direction time flows in.

This featured behavior enables the accurate depiction of various models, including neuron spiking BID10 .In regards to the second non-local dynamic feature, it can be shown that two GRUs can exhibit an Andronov-Hopf bifurcation, whereby a stable fixed point bifurcates into an unstable fixed point surrounded by a limit cycle.

Behaviorally speaking, a limit cycle is a type of attractor, in the sense that there exists a defined basin of attraction.

However, unlike a stable fixed point, where trajectories initialized in the basin of attraction flow towards a single point, a limit cycle pulls trajectories into a stable periodic orbit around the unstable fixed point at its center.

To demonstrate this phenomenon, let (8) define the parameters of (7).

DISPLAYFORM0 where α ∈ R + .If α = π 3 , the system has a single stable fixed point (stable spiral), as depicted in FIG4 .

If we continuously decrease α, the system undergoes an Andronov-Hopf bifurcation approximately about α = π 3.8 .

As α continuously decreases, the orbital period increases, and as the nullclines can be made arbitrarily close together, the length of this orbital period can be set arbitrarily, up to machine accuracy.

FIG4 shows an example of a relatively short orbital period, and figure 5C depicts the behavior seen for slower orbits.

the system exhibits a single stable fixed point at the origin FIG4 ).

If α decreases continuously, a limit cycle emerges around the fixed point, and the fixed point changes stability FIG4 .

Allowing α to decrease further increases the size and orbital period of the limit cycle FIG4 ).

The bottom row represents the hidden state as a function of time, for a single trajectory (denoted by black trajectories in each corresponding phase portrait)With finite-fixed point topologies and global structures out of the way, the next logical question to ask is, can two GRUs exhibit an infinite number of fixed points (countable or uncountable)?

Such behavior is often desirable in models that require stationary attraction to non-point structures, such as line attractors and ring attractors.

The short answer to this question is no. Lemma 1.

Any two dimensional GRU can only have finitely many simple fixed points.

This follows from Lefschetz theory BID6 .

The detailed proof can be found in appendix D, and is intended to give the reader intuition behind the result presented in the claim extended to aribitrary dimensional GRU in theorem 1.Theorem 1.

Any finite dimensional GRU can only have finitely many simple fixed points and bifurcation fixed points.

Proof.

By definition of simple fixed points, the Jacobian of the dynamics at those fixed points have nonzero real parts, making them Lefschetz fixed points.

Since GRU dynamics is asymptotically bounded on the compact set [−1, 1] d , where d is the number of GRUs, it follows from Lefshetz theory BID6 ) that there are finitely many simple fixed points.

Furthermore, by construction, a bifurcation fixed point can only exist within a stability structure if and only if there exists a separate topology, such that the simple fixed points making up each bifurcation fixed point exist isolated from one another.

Therefore, there can only exist finitely many isolated bifurcation fixed points.

This eliminates the possibility of having countably many fixed points.

Next, we show that there cannot be uncountably many non-isolated fixed points.

Theorem 2.

Any finite dimensional GRU cannot have a continuous attractor.

Proof.

We provide a sketch of a proof.

Let h DISPLAYFORM1 .

Now for any unit norm vector k ∈ S d−1 , and for any δ > 0, we can show that there exist an > 0 such that, ḣ (h * + δk) −ḣ(h * ) = ḣ (h * + δk) > .

This can be argued by taking three cases into consideration, (a) U r k = 0 and U h (σ(b r ) k) = 0, (b) U r k = 0 and U h (σ(b r ) k) = 0, and (c) U r k = 0.

In each case, it reverts to a 1-dimensional problem where it can be trivially shown to have no continuous attractor around h * along direction k. Thus, we conclude that there is no continuous attractor in any direction.

Despite this limitation, an approximation of a line attractor can be created using two GRUs.

This approximation can be made arbitrarily close to an actual line attractor on a finite region in phase space, thereby satisfying computational needs on an arbitrary interval when scaled.

We will refer to this phenomenon as a pseudo-line attractor.

Figure 6 depicts an example of such an attractor.

We conclude this section with a discussion of slow points in the two GRU system.

As a logical extension to the single GRU system, slow points occur where the nullclines are sufficiently close together, but do not intersect, as demonstrated in figure 7 .

Given the previously discussed classes of dynamic behavior for two GRUs, slow points can only exist so long as the potential for a saddle-node bifurcation fixed point is possible in the location of the desired slow point, given an appropriate change in parameters, as they result from the collision and annihilation of two fixed points.

This observation is consistent with the single GRU case, as slow points can only exist for a single fixed point.

This would imply that given the one fixed point case, a maximum of five slow points are possible.

However, this would imply that there must exist a six fixed point case by which five of the six fixed points exist at saddle-node bifurcation fixed points, which has not been observed (see appendix C).

Despite this shortcoming, four simultaneous slow points are obtainable, as shown in FIG4 .

Figure 7: An example of a slow point about the origin, obtainable with two GRUs.

Initial conditions satisfying y < −x are attracted to the slow point at the origin before a secondary attraction to the sink.

7A depicts the phase portrait of the system, and 7B shows the dynamics of the hidden state for a single initial condition (denoted by a black trajectory on 7A).

As a means to put our theory to practice, in this section we explore several examples of time series prediction of continuous time planar dynamical systems using two GRUs.

Results from the previous section indicate what dynamical features can be learned by this RNN, and suggest cases by which training will fail.

All of the following computer experiments consist of an RNN, by which the hidden layer is made up of two GRUs, followed by a linear output layer.

The network is trained to make a 29-step prediction from a given initial observation, and no further input through prediction.

As such, to produce accurate predictions, the RNN must rely solely on the hidden layer dynamics.

We train the network to minimize the following multi-step loss function: DISPLAYFORM0 where θ are the parameters of the GRU and linear readout, T = 29 is the prediction horizon, w i (t) is the i-th time series generated by the true system, andŵ(k; w 0 ) is k-step the prediction given w 0 .The hidden states are initialized at zero for each trajectory.

The RNN is then trained for 4000 epochs, using ADAM (Kingma & Ba, 2014) in whole batch mode to minimize the loss function, i.e., the mean square error between the predicted trajectory and the data.

N traj = 667 time series were used for training.

FIG7 depicts the experimental results of the RNN's attempt at learning each dynamical system we describe below.

To test if two GRUs can learn a limit cycle, we use a simple nonlinear oscillator called the FitzHugh-Nagumo Model.

The FitzHugh-Nagumo model is defined by: DISPLAYFORM0 where in this experiment we will chose τ = 12.5, a = 0.7, b = 0.8, and I ext = N (0.7, 0.04).

Under this choice of model parameters, the system will exhibit an unstable fixed point (unstable spiral) surrounded by a limit cycle FIG7 .As shown in section 4, two GRUs are capable of representing this topology.

The results of this experiment verify this claim FIG7 , as two GRUs can capture topologically equivalent dynamics.

As discussed in section 4, two GRUs can exhibit a pseudo-line attractor, by which the system mimics an analytic line attractor.

We will use the simplest representation of a planar line attractor: This system will exhibit a line attractor along the y-axis, at x = 0 FIG7 ).

Trajectories will flow directly perpendicular towards the attractor.

We added white Gaussian noise N (0, 0.1I) in the training data.

While the hidden state dynamics of the trained network do not perfectly match that of an analytic line attractor, there exists a subinterval near each of the fixed points acting as a pseudo-line attractor FIG7 ).

As such, the added affine transformation (linear readout) can scale and reorient this subinterval as is required by a given problem, thereby mimicking a line attractor.

DISPLAYFORM0

For this experiment, a dynamical system representing a standard ring attractor of radius one is used: DISPLAYFORM0 This system exhibits an attracting ring, centered around an unstable fixed point.

We added Gaussian noise N (0, 0.1I) to the training data.

Two GRUs will not be able to accurately capture the system's continuous attractor dynamics as expected from theorem 3.

The results of this experiment are demonstrated in FIG7 .

As expected, the RNN fails to capture the proper dynamical features of the ring attractor.

Rather, the hidden state dynamics fall into an observed finite fixed point topology (see case xxix in appendix C).

In addition, we robustly see this over multiple initializations, and the quality of approximation improves as the dimensionality of GRU increases FIG8 ).

Our analysis shows the rich but limited classes of dynamics the GRU can approximate in one, two, and arbitrary dimensions.

We developed a new theoretical framework to analyze GRUs as a continuous dynamical system, and showed that two GRUs can exhibit a variety of expressive dynamic features, such as limit cycles, homoclinic orbits, and a substantial catalog of stability structures and bifurcations.

However, we also showed that finitely many GRUs cannot exhibit the dynamics of an arbitrary continuous attractor.

These claims were then experimentally verified in two dimensions.

We believe these findings also unlock new avenues of research on the trainability of recurrent neural networks.

Although we have analyzed GRUs only in 1-and 2-dimensions in near exhaustive, we believe that the insights extends to higher-dimensions.

We leave rigorous analysis of higher-dimensional GRUs as future work.

A CONTINUOUS TIME SYSTEM DERIVATION We begin with the fully gated GRU as a discrete time system, where the input vector x t has been set equal to zero, as depicted in FORMULA0 - FORMULA0 , where is the Hadamard product, and σ is the sigmoid function.

DISPLAYFORM0 We recognize that (15) is a forward Euler discretization of a continuous time dynamical system.

This allows us to consider the underlying continuous time dynamics on the basis of the discretization.

The following steps are a walk through of the derivation:Since z t is a bounded function on R ∀t, there exists a functionz t , such that z t +z t = 1 at each time step (due to the symmetry of z t ,z t is the result of vertically flipping z t about 0.5, the midpoint of its range).

As such, we can rewrite (15) withz t as depicted in FORMULA0 .

DISPLAYFORM1 Let h(t) ≡ h t−1 .

As a result, we can sayz t ≡z(t) and r t ≡ r(t), as depicted in FORMULA7 .

DISPLAYFORM2 Dividing both sides of the equation by ∆t yields (24).

DISPLAYFORM3 If we take the limit as ∆t → 0, we get the analogous continuous time system to (13) -(15), DISPLAYFORM4 whereḣ ≡

Finally, we can rewrite (25) as follows: DISPLAYFORM0 where DISPLAYFORM1

The fixed points of our continuous time system (25) exist where the derivativeḣ = 0.

In the single GRU case, the Hadamard product reduces to standard scalar multiplication, yielding, DISPLAYFORM0 where z(t) and r(t) are defined by (27) and FORMULA7 respectively, and h * ∈ R represents a solution of (28).We can divide out z(t) − 1, indicating that the update gate does not play a part in the stability of the system.

For simplicity, lets expand r(t) in (28) by its definition (22).

TAB1 depicts all observed topologies of multiple-fixed point structures using two GRUs.

Figure 10 displays an example of a phase portrait from a two GRU system for each case listed in 2.

Note that all fixed points are denoted by a red dot, regardless of classification.

TAB2 lists the parameters used for each of the observed cases.

Note that all the update gate parameters are set to zero.

DISPLAYFORM1

Each case in this paper was discovered by hand by considering the geometric constraints on the structure of nullclines for both the decoupled and coupled system (i.e reset gate inactive and active respectively).

An exhaustive analysis on the one dimensional GRU allowed for a natural extension into the two dimensional decoupled GRU.

Upon establishing a set of base cases (a combinatorial argument regarding all possible ways the decoupled nullclines [topologically conjugate to linear and cubic polynomials] can intersect) From these base cases, the reset gate can be used as a means of bending and manipulating structure of the decoupled nullclines in order to obtain new intersection patterns in the coupled system.

DISPLAYFORM0 Figure 10: Thirty six multiple fixed-point topologies obtainable with two GRUs, depicted in phase space.

Orange and pink lines represent the x and y nullclines respectively.

Red dots indicate fixed points.

Each subfigure contains 64 purple lines, indicating trajectories in forward time, whose initial conditions were chosen to be evenly spaced on the vertices of a square grid on [−1.5, 1.5] 2 .

Direction of the flow is determined by the black arrows, and the underlaying color map represents the magnitude of the velocity of the flow in log scale.

DISPLAYFORM1 Figure 10: Thirty six multiple fixed-point topologies obtainable with two GRUs, depicted in phase space.

Orange and pink lines represent the x and y nullclines respectively.

Red dots indicate fixed points.

Each subfigure contains 64 purple lines, indicating trajectories in forward time, whose initial conditions were chosen to be evenly spaced on the vertices of a square grid on [−1.5, 1.5] 2 .

Direction of the flow is determined by the black arrows, and the underlaying color map represents the magnitude of the velocity of the flow in log scale.

DISPLAYFORM2 Figure 10: Thirty six multiple fixed-point topologies obtainable with two GRUs, depicted in phase space.

Orange and pink lines represent the x and y nullclines respectively.

Red dots indicate fixed points.

Each subfigure contains 64 purple lines, indicating trajectories in forward time, whose initial conditions were chosen to be evenly spaced on the vertices of a square grid on [−1.5, 1.5] 2 .

Direction of the flow is determined by the black arrows, and the underlaying color map represents the magnitude of the velocity of the flow in log scale.

We begin this proof by showing that all fixed points obtainable with two GRUs are Lefschetz fixed points.

To show this is the case let (37) expand our previous notation.

We set all elements in U z and b z to zero, as the update gate plays no part in the topology of (7) (shown in appendix B).

DISPLAYFORM3 We can now rewrite (7) expanded in terms of the individual elements of U h , U r , b h , and b r , as shown in FORMULA3 and (39).

DISPLAYFORM4 DISPLAYFORM5 If 0 is not an eigenvalue of the Jacobian matrix of (38) and (39) at a fixed point, the fixed point is said to be Lefshetz.

DISPLAYFORM6 , and F y = dẏ dy .

where DISPLAYFORM7 γ F x = U h11 U r11 xe −Ur11x−Ur12y−br1(e −Ur11x−Ur12y−br1 + 1) Let J denote the Jacobian matrix of (38) and (39).

DISPLAYFORM8 Note that we can rewrite (38) and (39) as follows: DISPLAYFORM9 DISPLAYFORM10 where θ represents the set of parameters, f (x, y, θ) = U h11 x 1+e −(U r11 x+U r12 y+b r1 ) + U h12 y 1+e −(U r21 x+U r22 y+b r2 ) + b h1 , and g(x, y, θ) = U h21 x 1+e −(U r11 x+U r12 y+b r1 ) + U h22 y 1+e −(U r21 x+U r22 y+b r2 ) + b h2 An ordered pair (x, y) is a fixed point of FORMULA1 and FORMULA7 As such, we can say the following: x = tanh(f (x, y, θ)) = u(x, y, θ) (53) y = tanh(g(x, y, θ)) = v(x, y, θ)If we let λ represent the eigenvalues of (48), the characteristic equation of FORMULA1 is as follows: λ 2 + λ(−1 − sech 2 (f (x, y, θ)) ∂f ∂x − sech 2 (g(x, y, θ)) ∂g ∂y ) + 1 4 (1 − sech 2 (f (x, y, θ))sech 2 (g(x, y, θ)) ∂f ∂y ∂g ∂x )We can rewrite (55) in terms of u(x, y, θ) and v(x, y, θ) as shown in (56) DISPLAYFORM11 where u Setting λ = 0 and simplifying yields the following constraint: DISPLAYFORM12 which can be realized as follows: We observe that sech 2 (f (x, y, θ))sech 2 (g(x, y, θ)) ∈ (0, 1), which implies that ∂f ∂y ∂g ∂x ∈ (1, ∞) However, from (53) and (54), we see that f (x, y, θ) = tanh −1 (x) and g(x, y, θ) = tanh −1 (y) at a critical point.

Which implies ∂f ∂y ∂g ∂x = 0 / ∈ (1, ∞).

Therefore λ = 0 ∀θ.

This implies that (38) and (39) is a Lefschetz map.

Since (38) and (39) are asymptotically bound to (−1, 1) 2 , we can always find a finite time t 0 such that x, y ∈ (−1, 1) 2 ∀t > t 0 .

Therefore, for every trajectory initialized outside of the trapping region, we can always find a point on [−1, 1] 2 that arises as the transition of that initial condition flowing into the trapping region.

This implies that (38) and (39) can be thought of as existing on a compact set, and therefore has a finite number of simple fixed points BID6 .

<|TLDR|>

@highlight

We classify the the dynamical features one and two GRU cells can and cannot capture in continuous time, and verify our findings experimentally with k-step time series prediction. 

@highlight

The authors analyse GRUs with hidden sizes of one and two as continuous-time dynamical systems, claiming that the expressive power of the hidden state representation can provide prior knowledge on how well a GRU will perform on a given dataset

@highlight

This paper analyzes GRUs from a dynamical systems perspective, and shows that 2d GRUs can be trained to adopt a variety of fixed points and can approximate line attractors, but cannot mimic a ring attractor.

@highlight

Converts GRU equations into continuous time and uses theory and experiemnts to study 1- and 2-dimensional GRU networks and showcase every variety of dynamical topology available in these systems