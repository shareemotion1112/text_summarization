A growing number of learning methods are actually differentiable games whose players optimise multiple, interdependent objectives in parallel – from GANs and intrinsic curiosity to multi-agent RL.

Opponent shaping is a powerful approach to improve learning dynamics in these games, accounting for player influence on others’ updates.

Learning with Opponent-Learning Awareness (LOLA) is a recent algorithm that exploits this response and leads to cooperation in settings like the Iterated Prisoner’s Dilemma.

Although experimentally successful, we show that LOLA agents can exhibit ‘arrogant’ behaviour directly at odds with convergence.

In fact, remarkably few algorithms have theoretical guarantees applying across all (n-player, non-convex) games.

In this paper we present Stable Opponent Shaping (SOS), a new method that interpolates between LOLA and a stable variant named LookAhead.

We prove that LookAhead converges locally to equilibria and avoids strict saddles in all differentiable games.

SOS inherits these essential guarantees, while also shaping the learning of opponents and consistently either matching or outperforming LOLA experimentally.

Problem Setting.

While machine learning has traditionally focused on optimising single objectives, generative adversarial nets (GANs) BID9 have showcased the potential of architectures dealing with multiple interacting goals.

They have since then proliferated substantially, including intrinsic curiosity BID19 , imaginative agents BID20 , synthetic gradients , hierarchical reinforcement learning (RL) BID23 BID22 and multi-agent RL in general BID2 .These can effectively be viewed as differentiable games played by cooperating and competing agents -which may simply be different internal components of a single system, like the generator and discriminator in GANs.

The difficulty is that each loss depends on all parameters, including those of other agents.

While gradient descent on single functions has been widely successful, converging to local minima under rather mild conditions BID13 , its simultaneous generalisation can fail even in simple two-player, two-parameter zero-sum games.

No algorithm has yet been shown to converge, even locally, in all differentiable games.

Related Work.

Convergence has widely been studied in convex n-player games, see especially BID21 ; BID5 .

However, the recent success of non-convex games exemplified by GANs calls for a better understanding of this general class where comparatively little is known.

BID14 recently prove local convergence of no-regreat learning to variationally stable equilibria, though under a number of regularity assumptions.

Conversely, a number of algorithms have been successful in the non-convex setting for restricted classes of games.

These include policy prediction in two-player two-action bimatrix games BID24 ; WoLF in two-player two-action games BID1 ; AWESOME in repeated games BID3 ; Optimistic Mirror Descent in two-player bilinear zero-sum games BID4 and Consensus Optimisation (CO) in two-player zerosum games BID15 ).

An important body of work including BID10 ; BID16 has also appeared for the specific case of GANs.

Working towards bridging this gap, some of the authors recently proposed Symplectic Gradient Adjustment (SGA), see BID0 .

This algorithm is provably 'attracted' to stable fixed points while 'repelled' from unstable ones in all differentiable games (n-player, non-convex).

Nonetheless, these results are weaker than strict convergence guarantees.

Moreover, SGA agents may act against their own self-interest by prioritising stability over individual loss.

SGA was also discovered independently by BID8 , drawing on variational inequalities.

In a different direction, Learning with Opponent-Learning Awareness (LOLA) modifies the learning objective by predicting and differentiating through opponent learning steps.

This is intuitively appealing and experimentally successful, encouraging cooperation in settings like the Iterated Prisoner's Dilemma (IPD) where more stable algorithms like SGA defect.

However, LOLA has no guarantees of converging or even preserving fixed points of the game.

Contribution.

We begin by constructing the first explicit tandem game where LOLA agents adopt 'arrogant' behaviour and converge to non-fixed points.

We pinpoint the cause of failure and show that a natural variant named LookAhead (LA), discovered before LOLA by BID24 , successfully preserves fixed points.

We then prove that LookAhead locally converges and avoids strict saddles in all differentiable games, filling a theoretical gap in multi-agent learning.

This is enabled through a unified approach based on fixed-point iterations and dynamical systems.

These techniques apply equally well to algorithms like CO and SGA, though this is not our present focus.

While LookAhead is theoretically robust, the shaping component endowing LOLA with a capacity to exploit opponent dynamics is lost.

We solve this dilemma with an algorithm named Stable Opponent Shaping (SOS), trading between stability and exploitation by interpolating between LookAhead and LOLA.

Using an intuitive and theoretically grounded criterion for this interpolation parameter, SOS inherits both strong convergence guarantees from LA and opponent shaping from LOLA.On the experimental side, we show that SOS plays tit-for-tat in the IPD on par with LOLA, while all other methods mostly defect.

We display the practical consequences of our theoretical guarantees in the tandem game, where SOS always outperforms LOLA.

Finally we implement a more involved GAN setup, testing for mode collapse and mode hopping when learning Gaussian mixture distributions.

SOS successfully spreads mass across all Gaussians, at least matching dedicated algorithms like CO, while LA is significantly slower and simultaneous gradient descent fails entirely.

We frame the problem of multi-agent learning as a game.

Adapted from BID0 , the following definition insists only on differentiability for gradient-based methods to apply.

This concept is strictly more general than stochastic games, whose parameters are usually restricted to action-state transition probabilities or functional approximations thereof.

Definition 1.

A differentiable game is a set of n players with parameters θ = (θ 1 , . . .

, θ n ) ∈ R d and twice continuously differentiable losses DISPLAYFORM0 Crucially, note that each loss is a function of all parameters.

From the viewpoint of player i, parameters can be written as θ = (θ i , θ −i ) where θ −i contains all other players' parameters.

We do not make the common assumption that each L i is convex as a function of θ i alone, for any fixed opponent parameters θ −i , nor do we restrict θ to the probability simplex -though this restriction can be recovered via projection or sigmoid functions σ : R → [0, 1].

If n = 1, the 'game' is simply to minimise a given loss function.

In this case one can reach local minima by (possibly stochastic) gradient descent (GD).

For arbitrary n, the standard solution concept is that of Nash equilibria.

Definition 2.

A pointθ ∈ R d is a (local) Nash equilibrium if for each i, there are neighbourhoods DISPLAYFORM1 In other words, each player's strategy is a local best response to current opponent strategies.

We write DISPLAYFORM2 Define the simultaneous gradient of the game as the concatenation of each player's gradient, DISPLAYFORM3 The ith component of ξ is the direction of greatest increase in L i with respect to θ i .

If each agent minimises their loss independently from others, they perform GD on their component ∇ i L i with learning rate α i .

Hence, the parameter update for all agents is given by θ ← θ − α ξ, where α = (α 1 , . . .

, α n ) and is element-wise multiplication.

This is also called naive learning (NL), reducing to θ ← θ − αξ if agents have the same learning rate.

This is assumed for notational simplicity, though irrelevant to our results.

The following example shows that NL can fail to converge.

Example 1.

Consider L 1/2 = ±xy, where players control the x and y parameters respectively.

The origin is a (global and unique) Nash equilibrium.

The simultaneous gradient is ξ = (y, −x) and cycles around the origin.

Explicitly, a gradient step from (x, y) yields (x, y) ← (x, y) − α(y, −x) = (x − αy, y + αx) which has distance from the origin (1 + α 2 )(x 2 + y 2 ) > (x 2 + y 2 ) for any α > 0 and (x, y) = 0.

It follows that agents diverge away from the origin for any α > 0.

The cause of failure is that ξ is not the gradient of a single function, implying that each agent's loss is inherently dependent on others.

This results in a contradiction between the non-stationarity of each agent, and the optimisation of each loss independently from others.

Failure of convergence in this simple two-player zero-sum game shows that gradient descent does not generalise well to differentiable games.

We consider an alternative solution concept to Nash equilibria before introducing LOLA.

Consider the game given by L 1 = L 2 = xy where players control the x and y parameters respectively.

The optimal solution is (x, y) DISPLAYFORM0 However the origin is a global Nash equilibrium, while also a saddle point of xy.

It is highly undesirable to converge to the origin in this game, since infinitely better losses can be reached in the anti-diagonal direction.

In this light, Nash equilibria cannot be the right solution concept to aim for in multi-agent learning.

To define stable fixed points, first introduce the 'Hessian' of the game as the block matrix DISPLAYFORM1 This can equivalently be viewed as the Jacobian of the vector field ξ.

Importantly, note that H is not symmetric in general unless n = 1, in which case we recover the usual Hessian H = ∇ 2 L.Definition 3.

A pointθ is a fixed point if ξ(θ) = 0.

It is stable if H(θ) 0, unstable if H(θ) ≺ 0 and a strict saddle if H(θ) has an eigenvalue with negative real part.

The name 'fixed point' is coherent with GD, since ξ(θ) = 0 implies a fixed updateθ ←θ − αξ(θ) = θ.

Though Nash equilibria were shown to be inadequate above, it is not obvious that stable fixed points (SFPs) are a better solution concept.

In Appendix A we provide intuition for why SFPs are both closer to local minima in the context of multi-loss optimisation, and more tractable for convergence proofs.

Moreover, this definition is an improved variant on that in BID0 , assuming positive semi-definiteness only atθ instead of holding in a neighbourhood.

This makes the class of SFPs as large as possible, while sufficient for all our theoretical results.

Assuming invertibility of H(θ) at SFPs is crucial to all convergence results in this paper.

The same assumption is present in related work including BID15 , and cannot be avoided.

Even for single losses, a fixed point with singular Hessian can be a local minimum, maximum, or saddle point.

Invertibility is thus necessary to ensure that SFPs really are 'local minima'.

This is omitted from now on.

Finally note that unstable fixed points are a subset of strict saddles, making Theorem 6 both stronger and more general than results for SGA by BID0 .

Accounting for nonstationarity, Learning with Opponent-Learning Awareness (LOLA) modifies the learning objective by predicting and differentiating through opponent learning steps .

For simplicity, if n = 2 then agent 1 optimises L 1 (θ 1 , θ 2 + ∆θ 2 ) with respect to θ 1 , where ∆θ 2 is the predicted learning step for agent 2. assume that opponents are naive learners, namely ∆θ 2 = −α 2 ∇ 2 L 2 .

After first-order Taylor expansion, the loss is approximately given by L 1 + ∇ 2 L 1 · ∆θ 2 .

By minimising this quantity, agent 1 learns parameters that align the opponent learning step ∆θ 2 with the direction of greatest decrease in L 1 , exploiting opponent dynamics to further reduce one's losses.

Differentiating with respect to θ 1 , the adjustment is DISPLAYFORM0 By explicitly differentiating through ∆θ 2 in the rightmost term, LOLA agents actively shape opponent learning.

This has proven effective in reaching cooperative equilibria in multi-agent learning, finding success in a number of games including tit-for-tat in the IPD.

The middle term above was originally dropped by the authors because "LOLA focuses on this shaping of the learning direction of the opponent".

We choose not to eliminate this term, as also inherent in LOLA-DiCE .

Preserving both terms will in fact be key to developing stable opponent shaping.

First we formulate n-player LOLA in vectorial form.

Let H d and H o be the matrices of diagonal and anti-diagonal blocks of H, so that DISPLAYFORM1 While experimentally successful, LOLA fails to preserve fixed pointsθ of the game since DISPLAYFORM2 in general.

Even ifθ is a Nash equilibrium, the updateθ ←θ − αLOLA =θ can push them away despite parameters being optimal.

This may worsen the losses for all agents, as in the game below.

Example 2 (Tandem).

Imagine a tandem controlled by agents facing opposite directions, who feed x and y force into their pedals respectively.

Negative numbers correspond to pedalling backwards.

Moving coherently requires x ≈ −y, embodied by a quadratic loss (x+y) 2 .

However it is easier for agents to pedal forwards, translated by linear losses −2x and −2y.

The game is thus given by L 1 (x, y) = (x + y) 2 − 2x and L 2 (x, y) = (x + y) 2 − 2y.

These sub-goals are incompatible, so agents cannot simply accelerate forwards.

The SFPs are given by {x + y = 1}. Computing χ (x, 1 − x) = (4, 4) = 0, none of these are preserved by LOLA.

Instead, we show in Appendix C that LOLA can only converge to sub-optimal scenarios with worse losses for both agents, for any α.

Intuitively, the root of failure is that LOLA agents try to shape opponent learning and enforce compliance by accelerating forwards, assuming a dynamic response from their opponent.

The other agent does the same, so they become 'arrogant' and suffer by pushing strongly in opposite directions.

The shaping term χ prevents LOLA from preserving fixed points.

Consider removing this component entirely, giving (I − αH o )ξ.

This variant preserves fixed points, but what does it mean from the perspective of each agent?

Note that LOLA optimises L 1 (θ 1 , θ 2 + ∆θ 2 ) with respect to θ 1 , while ∆θ 2 is a function of θ 1 .

In other words, we assume that our opponent's learning step depends on our current optimisation with respect to θ 1 .

This is inaccurate, since opponents cannot see our updated parameters until the next step.

Instead, assume we optimise DISPLAYFORM0 are the current parameters.

After Taylor expansion, the gradient with respect to θ 1 is given by DISPLAYFORM1 does not depend on θ 1 .

In vectorial form, we recover the variant (I −αH o )ξ since the shaping term corresponds precisely to differentiating through ∆θ 2 .

We name this LookAhead, which was discovered before LOLA by BID24 though not explicitly named.

Using the stop-gradient operator ⊥ 1 , this can be reformulated as optimising L 1 (θ 1 , θ 2 + ⊥∆θ 2 ) where ⊥ prevents gradient flowing from ∆θ 2 upon differentiation.

The main result of BID24 is that LookAhead converges to Nash equilibria in the small class of two-player, two-action bimatrix games.

We will prove local convergence to SFP and non-convergence to strict saddles in all differentiable games.

On the other hand, by discarding the problematic shaping term, we also eliminated LOLA's capacity to exploit opponent dynamics and encourage cooperation.

This will be witnessed in the IPD, where LookAhead agents mostly defect.

We propose Stable Opponent Shaping (SOS), an algorithm preserving both advantages at once.

Define the partial stop-gradient operator ⊥ p := p⊥ + (1 − p)I, where I is the identity and p stands for partial.

A p-LOLA agent optimises the modified objective DISPLAYFORM0 collapsing to LookAhead at p = 0 and LOLA at p = 1.

The resulting gradient is given by DISPLAYFORM1 We obtain an algorithm trading between shaping and stability as a function of p. Note however that preservation of fixed points only holds if p is infinitesimal, in which case p-LOLA is almost identical to LookAhead -losing the very purpose of interpolation.

Instead we propose a two-part criterion for p at each learning step, through which all guarantees descend.

First choose p such that ξ p points in the same direction as LookAhead.

This will not be enough to prove convergence itself, but prevents arrogant behaviour by ensuring convergence only to fixed points.

Formally, the first criterion is given by ξ p , ξ 0 ≥ 0.

If −α χ , ξ 0 ≥ 0 then ξ p , ξ 0 ≥ 0 automatically, so we choose p = 1 for maximal shaping.

Otherwise choose DISPLAYFORM2 −α χ , ξ 0 with any hyperparameter 0 < a < 1.

This guarantees a positive inner product DISPLAYFORM3 We complement this with a second criterion ensuring local convergence.

The idea is to scale p by a function of ξ if ξ is small enough, which certainly holds in neighbourhoods of fixed points.

Let 0 < b < 1 be a hyperparameter and take p = ξ 2 if ξ < b, otherwise p = 1.

Choosing p 1 and p 2 according to these criteria, the two-part criterion is p = min{p 1 , p 2 }.

SOS is obtained by combining p-LOLA with this criterion, as summarised in Algorithm 1.

Crucially, all theoretical results in the next section are independent from the choice of hyperparameters a and b.

Algorithm 1: Stable Opponent Shaping 1 Initialise θ randomly and fix hyperparameters a, b ∈ (0, 1).

2 while not done do DISPLAYFORM4

Our central theoretical contribution is that LookAhead and SOS converge locally to SFP and avoid strict saddles in all differentiable games.

Since the learning gradients involve second-order Hessian terms, our results assume thrice continuously differentiable losses (omitted hereafter).

Losses which are C 2 but not C 3 are very degenerate, so this is a mild assumption.

Statements made about SOS crucially hold for any hyperparameters a, b ∈ (0, 1).

See Appendices D and E for detailed proofs.

Convergence is proved using Ostrowski's Theorem.

This reduces convergence of a gradient adjustment g to positive stability (eigenvalues with positive real part) of ∇g at stable fixed points.

Theorem 2.

Let H 0 be invertible with symmetric diagonal blocks.

Then there exists > 0 such that (I − αH o )H is positive stable for all 0 < α < .This type of result would usually be proved either by analytical means showing positive definiteness and hence positive stability, or direct eigenvalue analysis.

We show in Appendix D that (I − αH o )H is not necessarily positive definite, while there is no necessary relationship between eigenpairs of H and H o .

This makes our theorem all the more interesting and non-trivial.

We use a similarity transformation trick to circumvent the dual obstacle, allowing for analysis of positive definiteness with respect to a new inner product.

We obtain positive stability by invariance under change of basis.

Corollary 3.

LookAhead converges locally to stable fixed points for α > 0 sufficiently small.

Using the second criterion for p, we prove local convergence of SOS in all differentiable games despite the presence of a shaping term (unlike LOLA).

Theorem 4.

SOS converges locally to stable fixed points for α > 0 sufficiently small.

Using the first criterion for p, we prove that SOS only converges to fixed points (unlike LOLA).

Proposition 5.

If SOS converges toθ and α > 0 is small thenθ is a fixed point of the game.

Now assume that θ is initialised randomly (or with arbitrarily small noise), as is standard in ML.

Let F (θ) = θ − αξ p (θ) be the SOS iteration.

Using both the second criterion and the Stable Manifold Theorem from dynamical systems, we can prove that every strict saddleθ has a neighbourhood U such that {θ ∈ U | F n (θ) →θ as n → ∞} has measure zero for α > 0 sufficiently small.

Since θ is initialised randomly, we obtain the following result.

Theorem 6.

SOS locally avoids strict saddles almost surely, for α > 0 sufficiently small.

This also holds for LookAhead, and could be strenghtened to global initialisations provided a strong boundedness assumption on H 2 .

This is trickier for SOS since p(θ) is not globally continuous.

Altogether, our results for LookAhead and the correct criterion for p-LOLA lead to some of the strongest theoretical guarantees in multi-agent learning.

Furthermore, SOS retains all of LOLA's opponent shaping capacity while LookAhead does not, as shown experimentally in the next section.

We evaluate the performance of SOS in three differentiable games.

We first showcase opponent shaping and superiority over LA/CO/SGA/NL in the Iterated Prisoner's Dilemma (IPD).

This leaves SOS and LOLA, which have differed only in theory up to now.

We bridge this gap by showing that SOS always outperforms LOLA in the tandem game, avoiding arrogant behaviour by decaying p while LOLA overshoots.

Finally we test SOS on a more involved GAN learning task, with results similar to dedicated methods like Consensus Optimisation.

IPD: This game is an infinite sequence of the well-known Prisoner's Dilemma, where the payoff is discounted by a factor γ ∈ [0, 1) at each iteration.

Agents are endowed with a memory of actions at the previous state.

Hence there are 5 parameters for each agent i: the probability P i (C | state) of cooperating at start state s 0 = ∅ or state s t = (a 1 t−1 , a 2 t−1 ) for t > 0.

One Nash equilibrium is to always defect (DD), with a normalised loss of 2.

A better equilibrium with loss 1 is named tit-for-tat (TFT), where each player begins by cooperating and then mimicks the opponent's previous action.

We run 300 training episodes for SOS, LA, CO, SGA and NL.

The parameters are initialised following a normal distribution around 1/2 probability of cooperation, with unit variance.

We fix α = 1 and γ = 0.96, following .

We choose a = 0.5 and b = 0.1 for SOS.

The first is a robust and arbitrary middle ground, while the latter is intentionally small to avoid poor SFP.

Tandem: Though local convergence is guaranteed for SOS, it is possible that SOS diverges from poor initialisations.

This turns out to be impossible in the tandem game since the Hessian is globally positive semi-definite.

We show this explicitly by running 300 training episodes for SOS and LOLA.

Parameters are initialised following a normal distribution around the origin.

We found performance to be robust to hyperparameters a, b. Here we fix a = b = 0.5 and α = 0.1.

We reproduce a setup from BID0 .

The game is to learn a Gaussian mixture distribution using GANs.

Data is sampled from a highly multimodal distribution designed to probe the tendency to collapse onto a subset of modes during training -see ground truth in Appendix F. The generator and discriminator networks each have 6 ReLU layers of 384 neurons, with 2 and 1 output neurons respectively.

Learning rates are chosen by grid search at iteration 8k, with a = 0.5 and b = 0.1 for SOS, following the same reasoning as the IPD.

IPD: Results are given in FIG1 .

Parameters in part (A) are the end-run probabilities of cooperating for each memory state, encoded in different colours.

Only 50 runs are shown for visibility.

Losses at each step are displayed in part (B), averaged across 300 episodes with shaded deviations.

SOS and LOLA mostly succeed in playing tit-for-tat, displayed by the accumulation of points in the correct corners of (A) plots.

For instance, CC and CD points are mostly in the top right and left corners so agent 2 responds to cooperation with cooperation.

Agents also cooperate at the start state, represented by ∅ points all hidden in the top right corner.

Tit-for-tat strategy is further indicated by the losses close to 1 in part (B).

On the other hand, most points for LA/CO/SGA/NL are accumulated at the bottom left, so agents mostly defect.

This results in poor losses, demonstrating the limited effectiveness of recent proposals like SGA and CO.

Finally note that trained parameters and losses for SOS are almost identical to those for LOLA, displaying equal capacity in opponent shaping while also inheriting convergence guarantees and outperforming LOLA in the next experiment.

Tandem: Results are given in Figure 3 .

SOS always succeeds in decreasing p to reach the correct equilibria, with losses averaging at 0.

LOLA fails to preserve fixed points, overshooting with losses averaging at 4/9.

The criterion for SOS is shown in action in part (B), decaying p to avoid overshooting.

This illustrates that purely theoretical guarantees descend into practical outperfor- mance.

Note that SOS even gets away from the LOLA fixed points if initialised there (not shown), converging to improved losses using the alignment criterion with LookAhead.

The generator distribution and KL divergence are given at {2k, 4k, 6k, 8k} iterations for NL, CO and SOS in Figure 4 .

Results for SGA, LOLA and LA are in Appendix F. SOS achieves convincing results by spreading mass across all Gaussians, as do CO/SGA/LOLA.

LookAhead is significantly slower, while NL fails through mode collapse and hopping.

Only visual inspection was used for comparison by BID0 , while KL divergence gives stronger numerical evidence here.

SOS and CO are slightly superior to others with reference to this metric.

However CO is aimed specifically toward two-player zero-sum GAN optimisation, while SOS is widely applicable with strong theoretical guarantees in all differentiable games.

Theoretical results in machine learning have significantly helped understand the causes of success and failure in applications, from optimisation to architecture.

While gradient descent on single losses has been studied extensively, algorithms dealing with interacting goals are proliferating, with little grasp of the underlying dynamics.

The analysis behind CO and SGA has been helpful in this respect, though lacking either in generality or convergence guarantees.

The first contribution of this paper is to provide a unified framework and fill this theoretical gap with robust convergence results for LookAhead in all differentiable games.

Capturing stable fixed points as the correct solution concept was essential for these techniques to apply.

Furthermore, we showed that opponent shaping is both a powerful approach leading to experimental success and cooperative behaviour -while at the same time preventing LOLA from preserving fixed points in general.

This conundrum is solved through a robust interpolation between LookAhead and LOLA, giving birth to SOS through a robust criterion.

This was partially enabled by choosing to preserve the 'middle' term in LOLA, and using it to inherit stability from LookAhead.

This results in convergence guarantees stronger than all previous algorithms, but also in practical superiority over LOLA in the tandem game.

Moreover, SOS fully preserves opponent shaping and outperforms SGA, CO, LA and NL in the IPD by encouraging tit-for-tat policy instead of defecting.

Finally, SOS convincingly learns Gaussian mixtures on par with the dedicated CO algorithm.

In the main text we showed that Nash equilibria are inadequate in multi-agent learning, exemplified by the simple game given by L 1 = L 2 = xy, where the origin is a global Nash equilibrium but a saddle point of the losses.

It is not however obvious that SFP are a better solution concept.

We begin by pointing out that for single losses, invertibility and symmetry of the Hessian imply positive definiteness at SFP.

These are exactly local minima of L detected by the second partial derivative test, namely those points provably attainable by gradient descent.

To emphasise this, note that gradient descent does not converge locally to all local minima.

This can be seen by considering the example L(x, y) = y 2 and the local (global) minimum (0, 0).

There is no neighbourhood for which gradient descent converges to (0, 0), since initialising at (x 0 , y 0 ) will always converge to (x 0 , 0) for appropriate learning rates, with x 0 = 0 almost surely.

This occurs precisely because the Hessian is singular at (0, 0).

Though a degenerate example, this suggests an important difference to make between the ideal solution concept (local minima) and that for which local convergence claims are possible to attain (local minima with invertible H 0).

Accordingly, the definition of SFP is the immediate generalisation of 'fixed points with positive semi-definite Hessian', or in other words, 'second-order-tractable local minima'.

It is important to impose only positive semi-definiteness to keep the class as large as possible, despite strict positive definiteness holding for single losses due to symmetry.

Imposing strict positivity would for instance exclude the origin in the cyclic game L 1 = xy = −L 2 , a point certainly worthy of convergence.

Note also that imposing a weaker condition than H 0 would be incorrect.

Invertibility aside, local convergence of gradient descent on single functions cannot be guaranteed if H 0, since such points are strict saddles.

These are almost always avoided by gradient descent, as proven by BID12 and .

It is thus necessary to impose H 0 as a minimal requirement in optimisation methods attempting to generalise gradient descent.

Remark A.1.

A matrix H is positive semi-definite iff the same holds for its symmetric part S = (H + H )/2, so SFP could equivalently be defined as S(θ) 0.

This is the original formulation given by part of the authors BID0 , who also imposed the extra requirement S(θ) 0 in a neighbourhood ofθ.

After discussion we decided to drop this assumption, pointing out that it is 1) more restrictive, 2) superficial to all theoretical results and 3) weakens the analogy with tractable local minima.

The only thing gained by imposing semi-positivity in a neighbourhood is that SFP become a subset of Nash equilibria.

Regarding unstable fixed points and strict saddles, note that H(θ) 0 implies H(θ) 0 in a neighbourhood, hence being equivalent to the definition in BID0 .

It follows also that unstable points are a subset of strict saddles: if H(θ) ≺ 0 then all eigenvalues are negative since any eigenpair (v, λ) satisfies DISPLAYFORM0 We introduced strict saddles in this paper as a generalisation of unstable FP, which are more difficult to handle but nonetheless tractable using dynamical systems.

The name is chosen by analogy to the definition in BID12 for single losses.

Proposition B.1.

The LOLA gradient adjustment is DISPLAYFORM0 in the usual assumption of equal learning rates.

Proof.

Recall the modified objective DISPLAYFORM1 for agent 1, and so on for each agent.

First-order Taylor expansion yields DISPLAYFORM2 and similarly for each agent.

Differentiating with respect to θ i , the adjustment for player i is DISPLAYFORM3 and thus DISPLAYFORM4 C TANDEM GAME We provide a more detailed exposition of the tandem game in this section, including computation of fixed points for NL/LOLA and corresponding losses.

Recall that the game is given by DISPLAYFORM5 Intuitively, agents wants to have x ≈ −y since (x + y) 2 is the leading loss, but would also prefer to have positive x and y. These are incompatible, so the agents must not be 'arrogant' and instead make concessions.

The fixed points are given by ξ = 2(x + y − 1) 1 1 = 0 , namely any pair (x, 1 − x).

The corresponding losses are L 1 = 1 − 2x = −L 2 , summing to 0 for any x. We have DISPLAYFORM6 everywhere, so all fixed points are SFP.

LOLA fails to preserve these, since DISPLAYFORM7 which is non-zero for any SFP (x, 1 − x).

Instead, LOLA can only converge to points such that DISPLAYFORM8 We solve this explicitly as follows: DISPLAYFORM9 The fixed points for LOLA are thus pairs (x, y) such that DISPLAYFORM10 This leads to worse losses DISPLAYFORM11 for agent 1 and similarly for agent 2.

In particular, losses always sum to something greater than 0.

This becomes negligible as the learning rate becomes smaller, but is always positive nonetheless Taking α arbitrarily small is not a viable solution since convergence will in turn be arbitrarily slow.

LOLA is thus not a strong algorithm candidate for all differentiable games.

We use Ostrowski's theorem as a unified framework for proving local convergence of gradient-based methods.

This is a standard result on fixed-point iterations, adapted from (Ortega & Rheinboldt, 2000, 10.1.3) .

We also invoke and prove a topological result of our own, Lemma D.9, at the end of this section.

This is useful in deducing local convergence, though not central to intuition.

DISPLAYFORM0 , and assumex ∈ Ω is a fixed point.

If all eigenvalues of ∇F (x) are strictly in the unit circle of C, then there is an open neighbourhood U ofx such that for all x 0 ∈ U , the sequence F (k) (x 0 ) converges tox.

Moreover, the rate of convergence is at least linear in k. Recall the simultaneous gradient ξ and the Hessian H defined for differentiable games.

Let X be any matrix with continuously differentiable entries.

Corollary D.3.

Assumex is a fixed point of a differentiable game such that XH(x) is positive stable.

Then the iterative procedure DISPLAYFORM1 converges locally tox for α > 0 sufficiently small.

Proof.

By definition of fixed points, ξ(x) = 0 and so DISPLAYFORM2 is positive stable by assumption, namely has eigenvalues a k + ib k with a k > 0.

It follows that DISPLAYFORM3 has eigenvalues 1 − αa k − iαb k , which are in the unit circle for small α.

More precisely, DISPLAYFORM4 which is always possible for a k > 0.

Hence ∇F (x) has eigenvalues in the unit circle for 0 DISPLAYFORM5 , and we are done by Ostrowski's Theorem sincex is a fixed point of F .We apply this corollary to LookAhead, which is given by DISPLAYFORM6 where X = (I − αH o ).

It is thus sufficient to prove the following result.

Theorem D.4.

Let H 0 invertible with symmetric diagonal blocks.

Then there exists > 0 such that (I − αH o )H is positive stable for all 0 < α < .Remark D.5.

Note that (I − αH o )H may fail to be positive definite, though true in the case of 2 × 2 matrices.

This no longer holds in higher dimensions, exemplified by the Hessian DISPLAYFORM7 By direct computation (symbolic in α), one can show that G = (I − αH o )H always has positive eigenvalues for small α > 0, whereas its symmetric part S always has a negative eigenvalue with magnitude in the order of α.

This implies that S and in turn G is not positive definite.

As such, an analytical proof of the theorem involving bounds on the corresponding bilinear form will fail.

This makes the result all the more interesting, but more involved.

Central to the proof is a similarity transformation proving positive definiteness with respect to a different inner product, a novel technique we have not found in the multi-agent learning literature.

Proof.

We cannot study the eigenvalues of G directly, since there is no necessary relationship between eigenpairs of H and H o .

In the aim of using analytical tools, the trick is to find a positive definite matrix which is similar to G, thus sharing the same positive eigenvalues.

First define DISPLAYFORM8 where H d is the sub-matrix of diagonal blocks,and rewrite DISPLAYFORM9 Note that H d is block diagonal with symmetric blocks ∇ ii L i 0, so (I + αH d ) is symmetric and positive definite for all α ≥ 0.

In particular its principal square root DISPLAYFORM10 is unique and invertible.

Now note that DISPLAYFORM11 which is positive semi-definite since DISPLAYFORM12 for all non-zero u. In particular M provides a similarity transformation which eliminates H d from G 1 while simultaneously delivering positive semi-definiteness.

We can now prove that First note that a Taylor expansion of M in α yields DISPLAYFORM13 DISPLAYFORM14 and DISPLAYFORM15

There are two cases to distinguish.

If u Hu > 0 then DISPLAYFORM0 for α sufficiently small.

Otherwise, u Hu = 0 and consider decomposing H into symmetric and antisymmetric parts S = (H + H )/2 and A = (H − H )/2, so that H = S + A. By antisymmetry of A we have u Au = 0 and hence u Hu = 0 = u Su.

Now H 0 implies S 0, so by Cholesky decomposition of S there exists a matrix T such that S = T T .

In particular 0 = u Su = T u 2 implies T u = 0, and in turn Su = 0.

Since H is invertible and u = 0, we have 0 = Hu = Au and so Au 2 > 0.

It follows in particular that DISPLAYFORM1 Using positive semi-definiteness of DISPLAYFORM2 for α > 0 small enough.

We conclude that for any u ∈ S m there is u > 0 such that DISPLAYFORM3 compact.

By Lemma D.9, this can be extended uniformly with some > 0 such that DISPLAYFORM4 for all u ∈ S m and 0 < α < .

It follows that M −1 GM is positive definite for all 0 < α < and thus G is positive stable for α in the same range, by similarity.

Corollary D.6.

LookAhead converges locally to stable fixed points for α > 0 sufficiently small.

0 invertible by definition, with diagonal blocks ∇ ii L i symmetric by twice continuous differentiability.

We are done by the result above and Corollary D.3.We now prove that local convergence results descend to SOS.

The following lemma establishes the crucial claim that our criterion for p is C 1 in neighbourhoods of fixed points.

This is necessary to invoke analytical arguments including Ostrowski's Theorem, and would be untrue globally.

Lemma D.7.

Ifθ is a fixed point and α is sufficiently small then p = ξ 2 in a neighbourhood ofθ.

Proof.

First note that ξ(θ) = 0, so there is a (bounded) neighbourhood V ofθ such that ξ(θ) < b for all θ ∈ V , for any choice of hyperparameter b ∈ (0, 1).

In particular p 2 (θ) = ξ(θ) 2 by definition of the second criterion.

We want to show that p(θ) = p 2 (θ) nearθ, or equivalently DISPLAYFORM0 in some neighbourhood U ⊆ V ofθ, for any choice of hyperparameter a ∈ (0, 1).

Now by boundedness of V and continuity of χ , there exists c > 0 such that −α χ (θ) = α 2 χ (θ) < c for all θ ∈ V and bounded α.

It follows by Cauchy-Schwartz that DISPLAYFORM1 in V , for some d > 0 and α sufficiently small, by boundedness of V and continuity of H o .

Finally there is a sub-neighbourhood U ⊂ V such that ξ(θ) < ad/c for all θ ∈ U , so that ad ξ /c > ξ(θ) 2 and hence DISPLAYFORM2 2 for all θ ∈ U , as required.

Theorem D.8.

SOS converges locally to stable fixed points for α > 0 sufficiently small.

Proof.

Though the criterion for p is dual, we will only use the second part.

More precisely, DISPLAYFORM3 The aim is to show that ifθ is an SFP then ∇ξ p (θ) is positive stable for small α, using Ostrowski to conclude as usual.

The first problem we face is that ∇ξ p does not exist everywhere, since p(θ) is not a continuous function.

However we know by Lemma D.7 that p = ξ 2 in a neighbourhood U ofθ, so ξ p is continuously differentiable in U .

Moreover, p(θ) = ξ(θ) 2 = 0 with gradient DISPLAYFORM4 by definition of fixed points.

It follows that DISPLAYFORM5 which is identical to LookAhead.

This is positive stable for all 0 < α < , andθ is a fixed point of the iteration since DISPLAYFORM6 We conclude by Corollary D.3 that SOS converges locally to SFP for any a, b ∈ (0, 1) and α sufficiently small.

Lemma D.9.

Let g : R + × Y → Z continuous with Y compact and Z ⊆ R. Assume that for any u ∈ Y there is u > 0 such that g(α, u) > 0 for all 0 < α < u .

Then there exists > 0 such that g(α, u) > 0 for all 0 < α < and u ∈ Y .Proof.

For any u ∈ Y there is u > 0 such that DISPLAYFORM7 We would like to extend this uniformly in u, namely prove that DISPLAYFORM8 is open by continuity of g, so each (0, u ) × {u} has a neighbourhood X u contained in g −1 (0, ∞).

Open sets in a product topology are unions of open products, so DISPLAYFORM9 In particular (0, u ) ⊆ x U x and at least one V x contains u, so we can take the open neighbourhood to be DISPLAYFORM10 for some neighbourhood V u of u. In particular Y ⊆ u∈Y V u , and by compactness there is a finite DISPLAYFORM11

Lemma E.1.

Let a k and b k be sequences of real numbers, and define DISPLAYFORM0 for all k ≥ M , k ≥ N .

Expanding the absolute value, this implies DISPLAYFORM1 which implies the contradiction L < L + δ .Proposition E.2.

If SOS converges toθ and α > 0 is small thenθ is a fixed point of the game.

Proof.

The iterative procedure is given by DISPLAYFORM2 If θ k →θ as k → ∞ then taking limits on both sides of the iteration yields DISPLAYFORM3 and so lim k ξ p (θ k ) = 0, omitting k → ∞ for convenience.

It follows by continuity that DISPLAYFORM4 noting that p(θ) is not a globally continuous function.

Assume for contradiction that ξ 0 (θ) = 0.

There are two cases to distinguish for clarity.(i) First assume −α χ , ξ 0 (θ) ≥ 0.

Note that lim k p(θ k ) ≥ 0 since p(θ) ≥ 0 for all θ, and so DISPLAYFORM5 In both cases a contradiction is obtained, hence ξ 0 (θ) = 0 = (I − αH o )ξ(θ).

Now note that (I − αH o )(θ) is singular iff H o (θ) has an eigenvalue 1/α, which is impossible for α sufficiently small.

Hence (I − αH o )ξ(θ) = 0 implies ξ(θ) = 0, as required.

Now assume that θ is initialised randomly (or with arbitrarily small noise around a point), as is standard in ML.

We prove that SOS locally avoids strict saddles using the Stable Manifold Theorem, inspired from BID13 .

Theorem E.3 (Stable Manifold Theorem).

Letx be a fixed point for the C 1 local diffeomorphism F : U → R d , where U is a neighbourhood ofx in R d .

Let E s ⊕ E u be the generalised eigenspaces of ∇F (x) corresponding to eigenvalues with |λ| ≤ 1 and |λ| > 1 respectively.

Then there exists a local stable center manifold W with tangent space E s atx and a neighbourhood B ofx such that DISPLAYFORM6 In particular, if ∇F (x) has at least one eigenvalue |λ| > 1 then E u has dimension at least 1.

Since W has tangent space E s atx, with codimension at least one, it follows that W has measure zero in R d .

This is central in proving that the set of initial points in a neighbourhood which converge through SOS to a given strict saddleθ has measure zero.

Theorem E.4.

SOS locally avoids strict saddles almost surely, for α > 0 sufficiently small.

Proof.

Letθ a strict saddle and recall that SOS is given by DISPLAYFORM7 Recall by Lemma D.7 that p(θ) = ξ(θ) 2 for all θ in a neighbourhood U ofθ.

Restricting F to U , all terms involved are continuously differentiable and DISPLAYFORM8 by assumption that ξ(θ) = 0.

Since all terms except I are of order at least α, ∇F (θ) is invertible for all α sufficiently small.

By the inverse function theorem, there exists a neighbourhood V ofθ such that F is has a continuously differentiable inverse on V .

Hence F restricted to U ∩ V is a C 1 diffeomorphism with fixed pointθ.

By definition of strict saddles, H(θ) has a negative eigenvalue.

It follows by continuity that (I − αH o )H(θ) also has a negative eigenvalue a + ib with a < 0 for α sufficiently small.

Finally, DISPLAYFORM9 has an eigenvalue λ = 1 − αa − iαb with |λ| = 1 − 2αa + α 2 (a 2 + b 2 ) ≥ 1 − 2αa > 1 .It follows that E s has codimension at least one, implying in turn that the local stable set W has measure zero.

We can now prove that DISPLAYFORM10 has measure zero, or in other words, that local convergence toθ occurs with zero probability.

Let B the neighbourhood guaranteed by the Stable Manifold Theorem, and take any θ ∈ Z. By definition of convergence there exists N ∈ N such that F N +n (θ) ∈ B for all n ≥ 0, so that DISPLAYFORM11 by the Stable Manifold Theorem.

This implies that θ ∈ F −N (W ), and finally θ ∈ ∪ n∈N F −n (W ).

Since θ was arbitrary, we obtain the inclusion Z ⊆ ∪ n∈N F −n (W ) .

Now F −1 is C 1 , hence locally Lipschitz and thus preserves sets of measure zero, so that F −n (W ) has measure zero for each n. Countable unions of measure zero sets are still measure zero, so we conclude that Z also has measure zero.

In other words, SOS converges toθ with zero probability upon random initialisation of θ in U .

In the Gaussian mixture experiment, data is sampled from a highly multimodal distribution designed to probe the tendency to collapse onto a subset of modes during training, given in FIG5 .The generator distribution and KL divergence are given at {2k, 4k, 6k, 8k} iterations for LA, LOLA and SGA in Figure 6 .

LOLA and SGA successfully spread mass across all Gaussians.

LookAhead displays mode collapse and hopping in early stages, but begins to incorporate further mixtures near 8k iterations.

We ran further iterations and discovered that LookAhead eventually spreads mass across all mixtures, though very slowly.

Comparing with results for NL/CO/SOS in the main text, we see that CO/SOS/LOLA/SGA are equally successful in qualitative terms.

Note that SOS/CO are slightly superior with respect to KL divergence after 6-8k iterations, though LOLA is initially faster.

This may be due only to random sampling.

We also noticed experimentally that LOLA often moves away from the correct distribution after 8-10k iterations (not shown), while SOS stays stable in the long run.

This may occur thanks to the two-part criterion encouraging convergence, while LOLA continually attempts to exploit opponent learning.

Finally we plot ξ at all iterations up to 12k for SOS, LA and NL in Figure 7 (other algorithms are omitted for visibility).

This gives further evidence of SOS converging quite rapidly to the correct distribution, while NL perpetually suffers from mode hopping and LA lags behind significantly.

@highlight

Opponent shaping is a powerful approach to multi-agent learning but can prevent convergence; our SOS algorithm fixes this with strong guarantees in all differentiable games.