With the success of modern machine learning, it is becoming increasingly important to understand and control how learning algorithms interact.

Unfortunately, negative results from game theory show there is little hope of understanding or controlling general n-player games.

We therefore introduce smooth markets (SM-games), a class of n-player games with pairwise zero sum interactions.

SM-games codify a common design pattern in machine learning that includes some GANs, adversarial training, and other recent algorithms.

We show that SM-games are amenable to analysis and optimization using first-order methods.

As artificial agents proliferate, it is increasingly important to analyze, predict and control their collective behavior (Parkes and Wellman, 2015; Rahwan et al., 2019) .

Unfortunately, despite almost a century of intense research since von Neumann (1928) , game theory provides little guidance outside a few special cases such as two-player zero-sum, auctions, and potential games (Monderer and Shapley, 1996; Nisan et al., 2007; Vickrey, 1961; von Neumann and Morgenstern, 1944) .

Nash equilibria provide a general solution concept, but are intractable in almost all cases for many different reasons (Babichenko, 2016; Daskalakis et al., 2009; Hart and Mas-Colell, 2003) .

These and other negative results (Palaiopanos et al., 2017) suggest that understanding and controlling societies of artificial agents is near hopeless.

Nevertheless, human societies -of billions of agents -manage to organize themselves reasonably well and mostly progress with time, suggesting game theory is missing some fundamental organizing principles.

In this paper, we investigate how markets structure the behavior of agents.

Market mechanisms have been studied extensively (Nisan et al., 2007) .

However, prior work has restricted to concrete examples, such as auctions and prediction markets, and strong assumptions, such as convexity.

Our approach is more abstract and more directly suited to modern machine learning where the building blocks are neural nets.

Markets, for us, encompass discriminators and generators trading errors in GANs (Goodfellow et al., 2014) and agents trading wins and losses in StarCraft (Vinyals et al., 2019) .

The paper introduces a class of games where optimization and aggregation make sense.

The phrase requires unpacking.

"Optimization" means gradient-based methods.

Gradient descent (and friends) are the workhorse of modern machine learning.

Even when gradients are not available, gradient estimates underpin many reinforcement learning and evolutionary algorithms.

"Aggregation" means weighted sums.

Sums and averages are the workhorses for analyzing ensembles and populations across many fields.

"Makes sense" means we can draw conclusions about the gradient-based dynamics of the collective by summing over properties of its members.

As motivation, we present some pathologies that arise in even the simplest smooth games.

Examples in section 2 show that coupling strongly concave profit functions to form a game can lead to uncontrolled behavior, such as spiraling to infinity and excessive sensitivity to learning rates.

Hence, one of our goals is to understand how to 'glue together agents' such that their collective behavior is predictable.

Section 3 introduces a class of games where simultaneous gradient ascent behaves well and is amenable to analysis.

In a smooth market (SM-game), each player's profit is composed of a personal objective and pairwise zero-sum interactions with other players.

Zero-sum interactions are analogous to monetary exchange (my expenditure is your revenue), double-entry bookkeeping (credits balance debits), and conservation of energy (actions cause equal and opposite reactions).

SM-games explicitly account for externalities.

Remarkably, building this simple bookkeeping mechanism into games has strong implications for the dynamics of gradient-based learners.

SM-games generalize adversarial games (Cai et al., 2016) and codify a common design pattern in machine learning, see section 3.1.

Section 4 studies SM-games from two points of view.

Firstly, from that of a rational, profit-maximizing agent that makes decisions based on first-order profit forecasts.

Secondly, from that of the game as a whole.

SM-games are not potential games, so the game does not optimize any single function.

A collective of profit-maximizing agents is not rational because they do not optimize a shared objective (Drexler, 2019) .

We therefore introduce the notion of legibility, which quantifies how the dynamics of the collective relate to that of individual agents.

Finally, section 5 applies legibility to prove some basic theorems on the dynamics of SM-games under gradient-ascent.

We show that (i) Nash equilibria are stable; (ii) that if profits are strictly concave then gradient ascent converges to a Nash equilibrium for all learning rates; and (iii) the dynamics are bounded under reasonable assumptions.

The results are important for two reasons.

Firstly, we identify a class of games whose dynamics are, at least in some respects, amenable to analysis and control.

The kinds of pathologies described in section 2 cannot arise in SM-games.

Secondly, we identify the specific quantities, forecasts, that are useful to track at the level of individual firms and can be meaningfully aggregated to draw conclusions about their global dynamics.

It follows that forecasts should be a useful lever for mechanism design.

A wide variety of machine learning markets and agent-based economies have been proposed and studied: Abernethy and Frongillo (2011); Balduzzi (2014) ; Barto et al. (1983); Baum (1999); Hu and Storkey (2014) ; Kakade et al. (2003; 2005) ; Kearns et al. (2001) ; Kwee et al. (2001) ; Lay and Barbu (2010); Minsky (1986) ; Selfridge (1958); Storkey (2011); Storkey et al. (2012) ; Sutton et al. (2011); Wellman and Wurman (1998) .

The goal of this paper is different.

Rather than propose another market mechanism, we abstract an existing design pattern and elucidate some of its consequences for interacting agents.

Our approach draws on work studying convergence in generative adversarial networks (Balduzzi et al., 2018; Gemp and Mahadevan, 2018; Gidel et al., 2019; Mescheder, 2018; Mescheder et al., 2017) , related minimax problems (Abernethy et al., 2019; Bailey and Piliouras, 2018) , and monotone games (Gemp and Mahadevan, 2017; Nemirovski et al., 2010; Tatarenko and Kamgarpour, 2019 ).

We consider dynamics in continuous time dw dt = ξ(w) in this paper.

Discrete dynamics, w t+1 ← w t + ξ(w) require a more delicate analysis, e.g. Bailey et al. (2019) .

In particular, we do not claim that optimizing GANs and SM-games is easy in discrete time.

Rather, our analyis shows that it is relatively easy in continuous time, and therefore possible in discrete time, with some additional effort.

The contrast is with smooth games in general, where gradient-based methods have essentially no hope of finding local Nash equilibria even in continuous time.

Figure 1: Effect of learning rates in two games.

Note: x-axis is log-scale.

Left: "half a game", e.g. 2.

Right: minimal SM-game, e.g. 3.

Top: Both players have same learning rate.

Bottom: Second player has 1 8 learning rate of first (which is same as for top).

Reducing the learning rate of the second player destabilizes the dynamics in "half a game", whereas the SM-game is essentially unaffected.

Vectors are column-vectors.

The notations S 0 and v 0 refer to a positive-definite matrix and vector with all entries positive respectively.

Rather than losses, we work with profits.

Proofs are in the appendix.

We use economic terminology (firms, profits, forecasts, and sentiment) even though the examples of SM-games, such as GANs and adversarial training, are taken from mainstream machine learning.

We hope the economic terminology provides an invigorating change of perspective.

The underlying mathematics is no more than first and second-order derivatives.

Smooth games model interacting agents with differentiable objectives.

They are the kind of games that are played by neural nets.

In practice, the differentiability assumption can be relaxed by replacing gradients with gradient estimates.

Definition 1.

A smooth game (Letcher et al., 2019) consists in n players [n] = {1, . . .

, n}, equipped with twice continuously differentiable profit functions {π i :

Player i controls the parameters w i .

If players update their actions via simultaneous gradient ascent, then a smooth game yields a dynamical system specified by the differential equation

.

The setup can be recast in terms of minimizing losses by

Smooth games are too general to be tractable since they encompass all dynamical systems.

Lemma 1.

Every continuous dynamical system on R d , for any d, arises as simultaneous gradient ascent on the profit functions of a smooth game.

The next two sections illustrate some problems that arise in simple smooth games.

Definition 2.

We recall some solution concepts from dynamical systems and game theory:

• A stable fixed point 1 w * satisfies ξ(w * ) = 0 and v · J(w * ) · v < 0 for all vectors v = 0.

• A local Nash equilibrium w * has neighborhoods U i of w * i for all i, such that

for all w i and all players i.

Example 1 below shows that stable fixed points and local Nash equilibria do not necessarily coincide.

The notion of classical Nash equilibrium is ill-suited to nonconcave settings.

Intuitively, a fixed point is stable if all trajectories sufficiently nearby flow into it.

A joint strategy is a local Nash if each player is harmed if it makes a small unilateral deviation.

Local Nash differs from the classic definition in two ways.

It is weaker, because it only allows small unilateral deviations.

This is necessary since players are neural networks and profits are not usually concave.

It is also stronger, because unilateral deviations decrease (rather than not increase) profits.

A game is a potential game if ξ = ∇φ for some function φ, see Balduzzi et al. (2018) for details.

Example 1 (potential game).

Fix a small > 0.

Consider the two-player games with profit functions

The game has a unique local Nash equilibrium at w = (0, 0) with π 1 (0, 0) = 0 = π 2 (0, 0).

The game is chosen to be as nice as possible: π 1 and π 2 are strongly concave functions of w 1 and w 2 respectively.

The game is a potential game since ξ = (w 2 − w 1 , w 1 − w 2 ) = ∇φ for φ(w) = w 1 w 2 − 2 (w 2 1 + w 2 2 ).

Nevertheless, the game exhibits three related problems.

Firstly, the Nash equilibrium is unstable.

Players at the Nash equilibrium can increase their profits via the joint update w ← (0, 0) + η · (1, 1), so π 1 (w) = η(1 − 2 ) = π 2 (w) > 0.

The existence of a Nash equilibrium where players can improve their payoffs by coordinated action suggests the incentives are not well-designed.

Secondly, the dynamics can diverge to infinity.

Starting at w

(1) = (1, 1) and applying simultaneously gradient ascent causes the norm of vector w (t) 2 to increase without limit as t → ∞ -and at an accelerating rate -due to a positive feedback loop between the players' parameters and profits.

Finally, players impose externalities on each other.

The decisions of the first player affect the profits of the second, and vice versa.

Obviously players must interact for a game to be interesting.

However, positive feedback loops arise because the interactions are not properly accounted for.

In short, simultaneous gradient ascent does not converge to the Nash -and can diverge to infinity.

It is open to debate whether the fault lies with gradients, the concept of Nash, or the game structure.

In this paper, we take gradients and Nash equilibria as given and seek to design better games.

Gradient-based optimizers rarely follow the actual gradient.

For example RMSProp and Adam use adaptive, parameter-dependent learning rates.

This is not a problem when optimizing a function.

Suppose f (w) is optimized with reweighted gradient (∇f ) η := (η 1 ∇ 1 f, . . .

, η n ∇ n f ) where η 0 is a vector of learning rates.

Even though (∇f ) η is not necessarily the gradient of any function, it behaves like ∇f because they have positive inner product when ∇f = 0:

Parameter-dependent learning rates thus behave well in potential games where the dynamics derive from an implicit potential function ξ(w) = ∇φ(w).

Severe problems can arise in general games.

1 Berard et al. (2019) use a different notion of stable fixed point that requires J has positive eigenvalues.

Example 2 ("half a game").

Consider the following game, where the w 2 -player is indifferent to w 1 :

The dynamics are clear by inspection: the w 2 -player converges to w 2 = 0, and then the w 1 -player does the same.

It is hard to imagine that anything could go wrong.

In contrast, behavior in the next example should be worse because convergence is slowed down by cycling around the Nash: Example 3 (minimal SM-game).

A simple SM-game, see definition 3, is

.

Figure 1 shows the dynamics of the games, in discrete time, with small learning rates and small gradient noise.

In the top panel, both players have the same learning rate.

Both games converge.

Example 2 converges faster -as expected -without cycling around the Nash.

In the bottom panels, the learning rate of the second player is decreased by a factor of eight.

The SM-game's dynamics do not change significantly.

In contrast, the dynamics of example 2 become unstable: although player 1 is attracted to the Nash, it is extremely sensitive to noise and does not stay there for long.

One goal of the paper is to explain why SM-games are more robust, in general, to differences in relative learning rates.

Tools for automatic differentiation (AD) such as TensorFlow and PyTorch include stop gradient operators that stop gradients from being computed.

For example, let

).

The use of stop gradient means f is not strictly speaking a function and so we use ∇ AD to refer to its gradient under automatic differentiation.

which is the simultaneous gradient from example 2.

Any smooth vector field is the gradient of a function augmented with stop gradient operators, see appendix D. Stop gradient is often used in complex neural architectures (for example when one neural network is fed into another leading to multiplicative interactions), and is thought to be mostly harmless.

Section 2.2 shows that stop gradients can interact in unexpected ways with parameter-dependent learning rates.

It is natural to expect individually well-behaved agents to also behave well collectively.

Unfortunately, this basic requirement fails in even the simplest examples.

Maximizing a strongly concave function is well-behaved: there is a unique, finite global maximum.

However, example 1 shows that coupling concave functions can cause simultaneous gradient ascent to diverge to infinity.

The dynamics of the game differs in kind from the dynamics of the players in isolation.

Example 2 shows that reducing the learning rate of a well-behaved (strongly concave) player in a simple game destabilizes the dynamics.

How collectives behave is sensitive not only to profits, but also to relative learning rates.

Off-the-shelf optimizers such as Adam (Kingma and Ba, 2015) modify learning rates under the hood, which may destabilize some games.

Let us restrict to more structured games.

Take an accountant's view of the world, where the only thing we track is the flow of money.

Interactions are pairwise.

Money is neither created nor destroyed, so interactions are zero-sum.

If we model the interactions between players by differentiable functions g ij (w i , w j ) that depend on their respective strategies then we have an SM-game.

All interactions are explicitly tracked.

There are no externalities off the books.

Positive interactions, g ij > 0, are revenue, negative are costs, and the difference is profit.

The model prescribes that all firms are profit maximizers.

More formally:

Definition 3 (SM-game).

A smooth market is a smooth game where interactions between players are pairwise zero-sum.

The profits have the form

The functions f i can act as regularizers.

Alternatively, they can be interpreted as natural resources or dummy players that react too slowly to model as players.

Dummy players provide firms with easy (non-adversarial) sources of revenue.

Humans, unlike firms, are not profit-maximizers; humans typically buy goods because they value them more than the money they spend on them.

Appendix C briefly discusses extending the model.

SM-games codify a common design pattern:

1.

Optimizing a function.

A near-trivial case is where there is a single player with profit π 1 (w) = f 1 (w).

2.

Generative adversarial networks and related architectures like CycleGANs are zero or near zero sum (Goodfellow et al., 2014; Wu et al., 2019; Zhu et al., 2017) .

3.

Zero-sum polymatrix games are SM-games where f i (w i ) ≡ 0 and g ij (w i , w j ) = w i A ij w j for some matrices A ij .

Weights are constrained to probability simplices.

The games have nice properties including: Nash equilibria are computed via a linear program and correlated equilibria marginalize onto Nash equilibria (Cai et al., 2016) .

4.

Intrinsic curiosity modules use games to drive exploration.

One module is rewarded for predicting the environment and an adversary is rewarded for choosing actions whose outcomes are not predicted by the first module (Pathak et al., 2017) .

The modules share some weights, so the setup is nearly, but not exactly, an SM-game.

5.

Adversarial training is concerned with the minmax problem (Kurakin et al., 2017; Madry et al., 2018 )

, y i obtains a star-shaped SM-game with the neural net (player 0) at the center and n adversaries -one per datapoint (x i , y i )

-on the arms.

6. Task-suites where a population of agents are trained on a population of tasks, form a bipartite graph.

If the tasks are parametrized and adversarially rewarded based on their difficulty for agents, then the setup is an SM-game.

7.

Homogeneous games arise when all the coupling functions are equal up to sign (recall

An example is population self-play (Silver et al., 2016; Vinyals et al., 2019) which lives on a graph where g ij (w i , w j ) := P (w i beats w j ) − 1 2 comes from the probability that policy w i beats w j .

Monetary exchanges in SM-games are quite general.

The error signals traded between generators and discriminators and the wins and losses traded between agents in StarCraft are two very different special cases.

How to analyze the behavior of the market as a whole?

Adam Smith claimed that profit-maximizing leads firms to promote the interests of society, as if by an invisible hand (Smith, 1776) .

More formally, we can ask: Is there a measure that firms collectively increase or decrease?

It is easy to see that firms do not collectively maximize aggregate profit (AP) or aggregate revenue (AR): Maximizing aggregate profit would require firms to ignore interactions with other firms.

Maximizing aggregate revenue would require firms to ignore costs.

In short, SM-games are not potential games; there is no function that they optimize in general.

However, it turns out the dynamics of SM-games aggregates the dynamics of individual firms, in a sense made precise in section 4.3.

Give an objective function to an agent.

The agent is rational, relative to the objective, if it chooses actions because it forecasts they will lead to better outcomes as measured by the objective.

In SM-games, agents are firms, the objective is profit, and forecasts are computed using gradients.

Firms aim to increase their profit.

Applying the first-order Taylor approximation obtains

where {h.o.t.} refers to higher-order terms.

Firm i's forecast of how profits will change if it modifies production by

Forecasts encode how individual firms expect profits to change ceteris paribus 2 .

How does profit maximizing by individual firms look from the point of view of the market as a whole?

Summing over all firms obtains

where f v (w) = i f vi (w) is the aggregate forecast.

Unfortunately, the left-hand side of Eq. (3) is incoherent.

It sums the changes in profit that would be experienced by firms updating their production in isolation.

However, firms change their production simultaneously.

Updates are not ceteris paribus and so profit is not a meaningful macroeconomic concept.

The following minimal example illustrates the problem: Example 4.

Suppose π 1 (w) = w 1 w 2 and π 2 (w) = −w 1 w 2 .

Fix w = (w 1 , w 2 ) and let v = (w 2 , −w 1 ).

The sum of the changes in profit expected by the firms, reasoning in isolation, is

whereas the actual change in aggregate profit is zero because π 1 (x) + π 2 (x) = 0 for any x.

Tracking aggregate profits is therefore not useful.

The next section shows forecasts are better behaved.

Give a target function to every agent in a collective.

The collective is legible, relative to the targets, if it increases or decreases the aggregate target according to whether its members forecast, on aggregate, they will increase or decrease their targets.

We show that SM-games are legible.

The targets are profit forecasts (note: not profits).

Let us consider how forecasts change.

Define the sentiment as the directional derivative of the forecast D vi f vi (w) = v i ∇f vi (w).

The first-order Taylor expansion of the forecast shows that the sentiment is a forecast about the profit forecast:

The perspective of firms can be summarized as:

1. Choose an update direction v i that is forecast to increase profit.

2.

The firm is then in one of two main regimes: a. If sentiment is positive then forecasts increase as the firm modifies its productionforecasts become more optimistic.

The firm experiences increasing returns-to-scale.

b. If sentiment is negative then forecasts decrease as the firm modifies its productionforecasts become more pessimistic.

The firm experiences diminishing returns-to-scale.

Our main result is that sentiment is additive, which means that forecasts are legible: Proposition 2 (forecasts are legible in SM-games).

Sentiment is additive

Thus, the aggregrate profit forecast f v increases or decreases according to whether individual forecasts f vi are expected to increase or decrease in aggregate.

Section 5.1 works through an example that is not legible.

Finally, we study the dynamics of gradient-based learners in SM-games.

Suppose firms use gradient ascent.

Firm i's updates are, infinitesimally, in the direction v i = ξ i (w) so that dwi dt = ξ i (w).

Since updates are gradients, we can simplify our notation.

Define firm i's forecast as f i (w) := We allow firms to choose their learning rates; firms with higher learning rates are more responsive.

Define the η-weighted dynamics ξ η (w) := (η 1 ξ 1 , . . .

, η n ξ n ) and η-weighted forecast as

In this setting, proposition 2 implies that Proposition 3 (legibility under gradient dynamics).

Fix dynamics dw dt := ξ η (w).

Sentiment decomposes additively:

Thus, we can read off the aggregate dynamics from the dynamics of forecasts of individual firms.

The pairwise zero-sum structure is crucial to legibility.

It is instructive to take a closer look at example 1, where the forecasts are not legible.

Suppose π 1 (w) = w 1 w 2 − 2 w 2 1 and π 2 (w) = w 1 w 2 − 2 w 2 2 .

Then ξ(w) = (w 2 − w 1 , w 1 − w 2 ) and the firms' sentiments are df1 dt = − (w 2 − w 1 ) 2 and df2 dt = − (w 1 − w 2 ) 2 which are always non-positive.

However, the aggregate sentiment is df dt

which for small is dominated by w 1 w 2 , and so can be either positive or negative.

When w = (1, 1) we have

Each firm expects their forecasts to decrease, and yet the opposite happens due to a positive feedback loop that ultimately causes the dynamics to diverge to infinity.

We provide three fundamental results on the dynamics of smooth markets.

Firstly, we show that stability, from dynamical systems, and local Nash equilibrium, from game theory, coincide in SM-games: Theorem 4 (stability).

A fixed point in an SM-game is a local Nash equilibrium iff it is stable.

Thus, every local Nash equilibrium is contained in an open set that forms its basin of attraction.

Secondly, we consider convergence.

Lyapunov functions are tools for studying convergence.

Given dynamical system dw dt = ξ(w) with fixed point w * , recall that V (w) is a Lyapunov function if:

If a dynamical system has a Lyapunov function then the dynamics converge to the fixed point.

Aggregate forecasts share properties (i) and (ii) with Lyapunov functions.

(i) Shared global minima: f η (w) = 0 iff f η (w) = 0 for all η, η 0, which occurs iff w is a stationary point, ξ i (w) = 0 for all i.

(ii) Positivity: f η (w) > 0 for all points that are not fixed points, for all η 0.

We can therefore use forecasts to study convergence and divergence across all learning rates: Theorem 5.

In continuous time, for all positive learning rates η 0,

* is a stable fixed point (S ≺ 0), then there is an open neighborhood U w * where dfη dt (w) < 0 for all w ∈ U \ {w * }, so the dynamics converge to w * from anywhere in U .

* is an unstable fixed point (S 0), there is an open neighborhood U w * such that dfη dt (w) > 0 for all w ∈ U \ {w * }, so the dynamics within U are repelled by w * .

The theorem explains why SM-games are robust to relative differences in learning rates -in contrast to the sensitivity exhibited by the game in example 2.

If a fixed point is stable, then for any dynamics dw dt = ξ η (w), there is a corresponding aggregate forecast f η (w) that can be used to show convergence.

The aggregate forecasts provide a family of Lyapunov-like functions.

Finally, we consider the setting where firms experience diminishing returns-to-scale for sufficiently large production vectors.

The assumption is realistic for firms in a finite economy since revenues must eventually saturate whilst costs continue to increase with production.

Theorem 6 (boundedness).

Suppose all firms have negative sentiment for sufficiently large values of w i .

Then the dynamics are bounded for all η 0.

The theorem implies that the kind of positive feedback loops that caused example 1 to diverge to infinity, cannot occur in SM-games.

One of our themes is that legibility allows to read off the dynamics of games.

We make the claim visually explicit in this section.

Let us start with a concrete game.

Figure 3AB plots the dynamics of the SM-game in example 5, under two different learning rates for player 1.

There is an unstable fixed point at the origin and an ovoidal cycle.

Dynamics converge to the cycle from both inside and outside the ovoid.

Changing player 1's learning rate, panel B, squashes the ovoid.

Panels CD provide a cartoon map of the dynamics.

There are two regions, the interior and exterior of the ovoid and the boundary formed by the ovoid itself.

In general, the phase space of any SM-game is carved into regions where sentiment dfη dt (w) is positive and negative, with boundaries where sentiment is zero.

The dynamics can be visualized as operating on a landscape where height at each point w corresponds to the value of the aggregate forecast f η (w).

The dynamics does not always ascend or always descend the landscape.

Rather, sentiment determines whether the dynamics ascends, descends, or remains on a level-set.

Since sentiment is additive,

, the decision to ascend or descend comes down to a weighted sum of the sentiments of the firms.

3 Changing learning rates changes the emphasis given to different firms' opinions, and thus changes the shapes of the boundaries between regions in a relatively straightforward manner.

SM-games can thus express richer dynamics than potential games (cycles will not occur when performing gradient ascent on a fixed objective), which still admit a relatively simple visual description in terms of a landscape and decisions about which direction to go (upwards or downwards).

Computing the landscape for general SM-games, as for neural nets, is intractable.

Machine learning has got a lot of mileage out of treating differentiable modules like plug-and-play lego blocks.

This works when the modules optimize a single loss and the gradients chain together seamlessly.

Unfortunately, agents with differing objectives are far from plug-and-play.

Interacting agents form games, and games are intractable in general.

Worse, positive feedback loops can cause individually well-behaved agents to collectively spiral out of control.

It is therefore necessary to find organizing principles -constraints -on how agents interact that ensure their collective behavior is amenable to analysis and control.

The pairwise zero-sum condition that underpins SM-games is one such organizing principle, which happens to admit an economic interpretation.

Our main result is that SM-games are legible: changes in aggregate forecasts are the sum of how individual firms expect their forecasts to change.

It follows that we can translate properties of the individual firms into guarantees on collective convergence, stability and boundedness in SM-games, see theorems 4-6.

Legibility is a local-to-global principle, whereby we can draw qualitative conclusions about the behavior of collectives based on the nature of their individual members.

Identifying and exploiting games that embed local-to-global principles will become increasingly important as artificial agents become more common.

This section provides a physics-inspired perspective on smooth markets.

Consider a dynamical system with n particles moving according to the differential equations:

The kinetic energy of a particle is mass times velocity squared, mv 2 , or in our case energy of i th particle = η

where we interpret the learning rate squared η 2 i of particle i as its mass and ξ i as its velocity.

The total energy of the system is the sum over the kinetic energies of the particles:

For example, in a Hamiltonian game we have that energy is conserved: Balduzzi et al. (2018) ; Letcher et al. (2019) for details.

Energy is measured in joules (kg · m · s −2 ).

The rate of change of energy with respect to time is power, measured in joules per second or watts (kg · m · s −3 ).

Conservation of energy means that a (closed) Hamiltonian system, in aggregate, generates no power.

The existence of an invariant function makes Hamiltonian systems easy to reason about in many ways.

Smooth markets are more general than Hamiltonian games in that total energy is not necessarily conserved.

Nevertheless, they are much more constrained than general dynamical systems.

Legibility, proposition 3, says that the total power (total rate of energy generation) in smooth markets is the sum of the power (rate of energy generation) of the individual particles:

Example where legibility fails.

Once again, it is instructive to look at a concrete example where legibility fails.

Recall the potential game in example 1 with profits

and π 2 (w) = w 1 w 2 − 2 w 2 2 .

and sentiments

Physically, the negative sentiments df1 dt < 0 and df2 dt < 0 mean that that each "particle" in the system, considered in isolation, is always dissipating energy.

Nevertheless as shown in section 5.1 the system as a whole has df dt

which is positive for some values of w. Thus, the system as a whole can generate energy through interaction effects between the (dissipative) particles.

Proof of lemma 1.

Lemma 1.

Every continuous dynamical system on R d , for any d, arises as simultaneous gradient ascent on the profit functions of a smooth game.

Proof.

Specifically, we mean that every dynamical system of the form Proof of proposition 2.

Before proving proposition 2, we first prove a lemma.

Lemma 7 (generalized Helmholtz decomposition).

The Jacobian decomposes into J(w) = S(w) + A(w) where S(w) and A(w) are symmetric and antisymmetric, respectively, for all w ∈ R d .

Proof.

Follows immediately.

See Letcher et al. (2019) for details and explanation.

Proposition 2.

Sentiment is additive:

Proof.

For any collection of updates

, we need to show that

because A is antisymmetric and S is block-diagonal.

Proof of proposition 3.

First we prove a lemma.

Proof.

Observe by direct computation that

It is then easy to see that

where S = S since S is symmetric.

By antisymmetry of A, we have that v A v = 0 for all v. The expression thus simplifies to

by the block-diagonal structure of S.

Proposition 3 (legibility under gradient dynamics).

Fix dynamics

Proof.

Applying the chain rule obtains that

where the second equality follows by construction of the dynamical system as dw dt = ξ η (w).

Lemma 8 shows that

for all i as required.

Proof of theorem 4.

Theorem 4.

A fixed point in an SM-game is a local Nash equilibrium iff it is stable.

Proof.

Suppose that w * is a fixed point of the game, that is suppose ξ(w * ) = 0.

Recall from lemma 7 that the Jacobian of ξ decomposes uniquely into two components J(w) = S(w) + A(w) where S ≡ S is symmetric and A + A ≡ 0 is antisymmetric.

It follows that v Jv = v Sv + v Av = v Sv since A is antisymmetric.

Thus, w * is a stable fixed point iff S(w * ) 0 is negative definite.

In an SM-game, the antisymmetric component is arbitrary and the symmetric component is block diagonal -where blocks correspond to players' parameters.

That is, S ij = 0 for i = j because the interactions between players i and j are pairwise zero-sum -and are therefore necessarily confined to the antisymmetric component of the Jacobian.

Since S is block-diagonal, it follows that S is negative definite iff the submatrices S ii along the diagonal are negative definite for all players i.

is strictly concave in the parameters controlled by player i at w * .

The result follows.

Proof of theorem 5.

Theorem 5.

In continuous time, for all positive learning rates η 0, Proof.

We prove the first part.

The second follows by a symmetric argument.

First, strict concavity implies

ii π i is negative definite for all i. Second, since S is block-diagonal, with zeros in all blocks S ij for pairs of players i = j, it follows that S is also negative definite.

Observe that

for all ξ η = 0 since S is negative definite.

Thus, simultaneous gradient ascent on the profits acts to infinitesimally reduce the function f η (w).

Since ξ η reduces f η , it will converge to a stationary point satisfying ∇f η = 0.

Observe that ∇f η = 0 iff ξ η = 0 since ∇f η = J ξ η and the symmetric component S of the Jacobian is negative definite.

Finally, observe that all stationary points of f η , and hence ξ η , are stable fixed points of ξ η because S is negative definite, which implies that the fixed point is a Nash equilibrium.

Proof of theorem 6.

for the dynamical system defined by dw dt = ξ η .

Since we are operating in continuous time, all that is required is to show that f η (w (t) ) = g < g implies that f η (w (t+ ) ) < g for all sufficiently small > 0.

dt (w) < 0 for all w in a sufficiently small ball centered at w (t) .

In other words, the dynamics dw dt = ξ η reduce f η and the result follows.

Definition 3 proposes a model of monetary exchange in smooth markets.

It ignores some major aspects of actual markets.

For example, SM-games do not model inventories, investment, borrowing or interest rates.

Moreover, in practice money is typically exchanged in return for goods or serviceswhich are ignored by the model.

In this section, we sketch one way to extend SM-games to model the exchange of both money and goods -although still without accounting for inventories, which would more significantly complicate the model.

The proposed extension is extremely simplistic.

It is provided to indicate how the model's expressive power can be increased, and complications that results.

Suppose π i (w) = f i (w) + j =i α ij ω ij (w i , w j ) − g ij (w i , w j ) .

The functions ω ij measure the amount of goods (say, widgets) that are exchanged between firms i and j. We assume that ω ij + ω ji ≡ 0 since widgets are physically passed between the firms and therefore one firms increase must be the others decrease.

For two firms to enter into an exchange it must be that they subjectively value the widgets differently, hence we introduce the parameters α ij .

Note that if α ij = 1 for all ij then the model is equivalent to an SM-game.

The transaction between firms i and j is net beneficial to both firms i and j if

and, simultaneously α ji · ω ji (w i , w j ) > g ji (w i , w j ).

We can interpret the inequalities as follows.

First suppose that ω ij and g ij always have the same sign.

The assumption is reasonable so long as firms do not pay to give away widgets.

Further assume without loss of generality that ω ij and g ij are both greater than zero -in other words, firm i is buying widgets from firm j.

The above inequalities can then be rewritten as α ij · ω ij (w i , w j ) amount firm i values the widgets > g ij (w i , w j ) amount firm i pays and α ji · ω ij (w i , w j )

amount j values the widgets < g ij (w i , w j ) amount j is paid It follows that both firms benefit from the transaction.

Implications for dynamics.

The off-block-diagonal terms of the symmetric and anti-symmetric components of the game Jacobian are

and A ij = α ij + α ji 2 · ∇ 2 ij ω ij (w i , w j ) where it is easy to check that S ij = S ji and A ij + A ji = 0.

The off-block-diagonal terms of S has consequences for how forecasts behave:

When are near SM-games well-behaved?

If α ij = α ji for all i, j then the correction is zero; if α ij ∼ α ji then the corrections due to different valuations of goods will be negligible, and the game should be correspondingly well-behaved.

What can go wrong?

Eq (5) implies that the dynamics of near SM-games -specifically whether the dynamics are increasing or decreasing the aggregate forecast -cannot be explained in terms of the sum of sentiments of individual terms.

The correction terms involve interactions between dynamics of different firms and the (second-order) quantities of goods exchanged.

In principle, these terms could be arbitrarily large positive or negative numbers.

Concretely, the correction terms involving couplings between dynamics of different firms can lead to positive feedback loops, as in example 1, where the dynamics spiral off to infinity even though both players have strongly concave profit functions.

Lemma 9.

Any smooth vector field can be constructed as the gradient of a function augmented with stop gradient operators.

Proof.

Suppose ξ = ( ∂f1(w) ∂w1 , . . .

,

f w i , stop gradient(wî)

It follows that ∇ AD g(w) = ξ(w)

as required.

@highlight

We introduce a class of n-player games suited to gradient-based methods.