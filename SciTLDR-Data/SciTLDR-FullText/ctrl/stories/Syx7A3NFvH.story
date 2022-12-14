This paper considers multi-agent reinforcement learning (MARL) in networked system control.

Specifically, each agent learns a decentralized control policy based on local observations and messages from connected neighbors.

We formulate such a networked MARL (NMARL) problem as a spatiotemporal Markov decision process and introduce a spatial discount factor to stabilize the training of each local agent.

Further, we propose a new differentiable communication protocol, called NeurComm, to reduce information loss and non-stationarity in NMARL.

Based on experiments in realistic NMARL scenarios of adaptive traffic signal control and cooperative adaptive cruise control, an appropriate spatial discount factor effectively enhances the learning curves of non-communicative MARL algorithms, while NeurComm outperforms existing communication protocols in both learning efficiency and control performance.

Reinforcement learning (RL), formulated as a Markov decision process (MDP), is a promising data-driven approach for learning adaptive control policies (Sutton & Barto, 1998) .

Recent advances in deep neural networks (DNNs) further enhance its learning capacity on complex tasks.

Successful algorithms include deep Q-network (DQN) (Mnih et al., 2015) , deep deterministic policy gradient (DDPG) (Lillicrap et al., 2015) , and advantage actor critic (A2C) (Mnih et al., 2016) .

However, RL is not scalable in many real-world control problems.

This scalability issue is addressed in multi-agent RL (MARL), where each agent learns its individual policy from only local observations.

However, MARL introduces new challenges in model training and execution, due to non-stationarity and partial observability in a decentralized MDP from the viewpoint of each agent.

To address these challenges, various learning methods and communication protocols are proposed to stabilize training and improve observability.

This paper considers networked MARL (NMARL) in the context of networked system control (NSC), where agents are connected via a communication network for a cooperative control objective.

Each agent performs decentralized control based on its local observations and messages from connected neighbors.

NSC is extensively studied and widely applied.

Examples include connected vehicle control (Jin & Orosz, 2014) , traffic signal control (Chu et al., 2019) , distributed sensing (Xu et al., 2018) , and networked storage operation (Qin et al., 2016) .

We expect an increasing trend of NMARL based controllers in the near future, after the development of advanced communication technologies such as 5G and Internet-of-Things.

Recent works studied decentralized NMARL under assumptions of global observations and local rewards (Zhang et al., 2018; Qu et al., 2019) , which are reasonable in multi-agent gaming but not suitable in NSC.

First, the control infrastructures are distributed in a wide region, so collecting global observations in execution increases communication delay and failure rate, and hurts the robustness.

Second, online learning is not common due to safety and efficiency concerns.

Rather, each model is trained offline and tested extensively before field deployment.

In online execution, the model only runs forward propagation, and its performance is constantly monitored for triggering re-training.

To reflect these practical constraints in NSC, we assume 1) each agent is connected to a limited number

The networked system is represented by a graph G(V, E) where i ??? V is each agent and ij ??? E is each communication link.

The corresponding MDP is characterized as (G, {S i , A i } i???V , p, r) where S i and A i are the local state space and action space of agent i. Let S := ?? i???V S i and A := ?? i???V A i be the global state space and action space, MDP transitions follow a stationary probability distribution p : S ?? A ?? S ??? [0, 1], and global step rewards be denoted by r : S ?? A ??? R. In a multi-agent MDP, each agent i follows a decentralized policy ?? i :

S i ?? A i ??? [0, 1] to chose its own action a i,t ??? ?? i (??|s i,t ) at time t. The MDP objective is to maximize E[R ?? 0 ], where R ?? t = T ?? =t ?? ?? ???t r ?? is the long-term global return with discount factor ??.

Here the expectation is taken over the global policy ?? : S ??A ??? [0, 1], the initial distribution s t ??? ??, and the transition s ?? +1 ??? p(??|s ?? , a ?? ), regarding the step reward r ?? = r(s ?? , a ?? ), ????? < T , and the terminal reward r T = r T (s T ) 2 .

The same system can be formulated as a centralized MDP.

Defining V ?? (s) = E[R ?? t |s t = s] as the state-value function and

MARL provides a scalable solution for controlling networked systems, but it introduces partial observability and non-stationarity in decentralized MDP of each agent, leading to inefficient and unstable learning performance.

To see this, note s i,t ??? S i ??? S does not provide sufficient information for ?? i .

Even assuming

is non-stationary if the behavior policies of other agents ?? ???i := {?? j } j???V\{i} are evolving over time.

In this paper, we enforce practical constraints and only allow local observations and neighborhood communications, which makes MARL even more challenging.

Definition 3.1 (Networked Multi-agent MDP with Neighborhood Communication).

In a networked cooperative multi-agent MDP (G,

with the message space M, the global reward is defined as r = 1 |V| i???V r i .

All local rewards are shared globally, whereas the communication is limited to neighborhoods, that is, each agent i observess i,t := s i,t ??? m Nii,t .

Here N i := {j ??? V|ji ??? E}, m Nii,t := {m ji,t } j???Ni , and each message m ji,t ??? M ji is derived from all the available information at that neighbor.

Definition 3.2 (Spatiotemporal MDP).

We assume local transitions are independent of other agents given the neighboring agents, that is,

where V i := N i ??? {i} is the closed neighborhood, and p is abused to denote any stationary transition.

Then from the viewpoint of each agent i, Definition 3.1 is equivalent to a decentralized spatiotemporal MDP, characterized as

, by optimizing the discounted return

where 0 ??? ?? ??? 1 is the spatial discount factor, and d ij is distance between agents i and j.

The major assumption in Definition 3.2 is that the Markovian property holds both temporally and spatially, so that the next local state depends on the neighborhood states and policies only.

This assumption is valid in most networked control systems such as traffic and wireless networks, as well as the power grid, where the impact of each agent is spread over the entire system via controlled flows, or chained local transitions.

Note in NSC, each agent is connected to a limited number of neighbors (the degree of G is low).

So spatiotemporal MDP is decentralized during model execution, and it naturally extends properties of MDP.

To reduce the learning difficulty of spatiotemporal MDP, a spatiotemporally discounted return is introduced in Eq. (2) to scale down reward signals further away (which are more difficult to fit using local information).

When ?? ??? 0,

each agent performs local greedy control; when ?? ??? 1, each agent performs global coordination and

, since the immediate local reward of each agent is only affected by controls within its closed neighborhood.

Now we assume each agent is A2C, with parametric models ?? ??i (s i ) and V ??i (s i , a Ni ) for fitting the optimal policy ?? * i and value function V ??i .

Note ifs i is able to provide global information through cascaded neighborhood communications, both ?? ??i and V ??i are able to fit return R ?? i,t .

Also, global and future information, such as R ?? i,?? and a Ni,?? , are always available from each rollout minibatch in offline training.

In contrast, only local informations i,t is allowed in online execution of policy ?? ??i .

Proposition 3.1 (Spatiotemporal RL with A2C).

Let {?? ??i } i???V and {V ??i } i???V be the decentralized actor-critics, and {(s i,?? , m Nii,?? , a i,?? , r i,?? )} i???V,?? ???B be the on-policy minibatch from spatiotemporal MDPs under stationary policies {?? ??i } i???V .

Then each actor and critic are updated by losses

is the estimated state-value, and ?? is the coefficient of the entropy loss.

For efficient and adaptive information sharing, we propose a new communication protocol called NeurComm.

To simplify the notation, we assume all messages sent from agent i are identical, i.e., m ij = m i , ???j ??? N i .

Then

where h i,t is the hidden state (or the belief ) of each agent and e ??i and g ??i are differentiable message encoding and extracting functions 3 .

To avoid dilution of state and policy information (the former is for improving observability while the later is for reducing non-stationarity), state and policy are explicitly included in the message besides agent belief, i.e., m i,t = s i,t ??? ?? i,t???1 ??? h i,t???1 , or s i,t := s Vi,t ????? Ni,t???1 ???h Ni,t???1 as in Eq. (5).

Note the communication phase is prior-decision, so only h i,t???1 and ?? i,t???1 are available.

This protocol can be easily extended for multi-pass communication:

Ni,t )), where h (0) i,t = h i,t???1 , and k denotes each of the communication passes.

The communication attentions can be integrated either at the sender as ?? i,t (m i,t ), or at the receiver as ?? i,t (m Ni,t ).

Replacing the input (s i,t ) of Eq. (3)(4) with the belief (h i,t ), the actor and critic become ?? ??i (??|h i,t ) and V ??i (h i,t , a Ni,t ), and the frozen estimations are ?? i,t and v i,t , respectively.

Proposition 4.1 (Neighborhood Neural Communication).

In spatiotemporal RL with neighborhood NeurComm, each agent utilizes the delayed global information to learn its belief, and it learns the message to optimize the control performance of all other agents.

NeurComm enabled MARL can be represented using a single meta-DNN since all agents are connected by differentiable communication links, ands i are the intermediate outputs after communication layers.

Fig. 1a illustrates the forward propagations inside each individual agent and Fig. 1b shows the broader multi-step spatiotemporal propagations.

Note the gradient propagation of this meta-DNN is decentralized based on each local loss signal.

As time advances, the involved parameters in each propagation expand spatially in the meta-DNN, due to the cascaded neighborhood communication.

To see this mathematically, ?? ??i,t (??|h i,t ) = ???? i,t (??|s Vi,t , ?? Ni,t???1 ), with

(??|s Vi,t+1 , ?? Ni,t , {s Nj ,t , ?? Nj ,t???1 } j???Ni ), with 3 Additional cell state needs to be maintained if LSTM is used.

In other words, {?? i , ?? i } will be updated for improving actors ?? ??j , ???j ??? V, as soon as they are included in?? j ; meanwhile, r i will be included in R ?? j .

In contrast, the policy is fully decentralized in execution, as g ??i depends ons i only.

(a) Intra-step propagations.

(b) Inter-step propagations.

NeurComm is general enough and has connections to other communication protocols.

CommNet performs a more lossy aggregation since the received messages are averaged before encoding, and all encoded inputs are summed up (Sukhbaatar et al., 2016) .

In DIAL, each DQN agent encodes the received messages instead of averaging them, but still it sums all encoded inputs (Foerster et al., 2016) .

Also, both CommNet and DIAL do not have policy fingerprints included in messages.

There are several benchmark MARL environments such as cooperative navigation and predator-prey, but few of them represent NSC.

Here we design two NSC environments: adaptive traffic signal control (ATSC) and cooperative adaptive cruise control (CACC).

Both ATSC and CACC are extensively studied in intelligent transportation systems, and they hold assumptions of a spatiotemporal MDP.

The objective of ATSC is to adaptively adjust signal phases to minimize traffic congestion based on real-time road-traffic measurements.

Here we implement two ATSC scenarios: a 5??5 synthetic traffic grid and a real-world 28-intersection traffic network from Monaco city, using standard microscopic traffic simulator SUMO (Krajzewicz et al., 2012) .

General settings.

For both scenarios, each episode simulates the peak-hour traffic, and a 5s control interval is applied to prevent traffic light from too frequent switches, based on RL control latency and driver response delay.

Thus, one MDP step corresponds to 5s simulation and the horizon is 720 steps.

Further, a 2s yellow time is inserted before switching to red light for safety purposes.

In ATSC, the real-time traffic flow, that is, the total number of approaching vehicles along each incoming lane, is measured by near-intersection induction-loop detectors (ILDs) (shown as the blue areas of example intersections in Fig. 2 ).

The cost of each agent is the sum of queue lengths along all incoming lanes.

Scenario settings.

Fig. 2a illustrates the traffic grid formed by two-lane arterial streets with speed limit 20m/s and one-lane avenues with speed limit 11m/s.

We simulate the peak-hour traffic dynamics through four collections of time-variant traffic flows, with both loading and recovering phases.

At beginning, three major flows F 1 are generated with origin-destination (O-D) pairs x 10 -x 4 , x 11 -x 5 , and x 12 -x 6 , meanwhile three minor flows f 1 are generated with O-D pairs x 1 -x 7 , x 2 -x 8 , and x 3 -x 9 .

After 15 minutes, F 1 and f 1 start to decay, while their opposite flows F 2 and f 2 start to dominate, as shown in Fig. 2b .

Note the flows define the high-level demand only, the particular route of each vehicle is randomly generated.

The grid is homogeneous and all agents have the same action space, which is a set of five pre-defined signal phases.

Fig. 2c illustrates the Monaco traffic network, with controlled intersections in blue.

NMARL in this scenario is more challenging since the network is heterogeneous with a variety of observation and action spaces.

Four traffic flow collections are generated to simulate the peak-hour traffic, and each flow is a multiple of a "unit" flow of 325veh/hr, with randomly sampled O-D pairs inside rectangle areas in Fig. 2c .

F 1 and F 2 are simulated during the first 40min, as [1, 2, 4, 4, 4, 4, 2, 1] unit flows with 5min intervals; F 3 and F 4 are generated in the same way, but with a delay of 15min.

See code for more details.

The objective of CACC is to adaptively coordinate a platoon of vehicles to minimize the car-following headway and speed perturbations based on real-time vehicle-to-vehicle communication.

Here we implement two CACC scenarios: "Catch-up" and "Slow-down", with physical vehicle dynamics.

General settings.

For both CACC tasks, we simulate a string of 8 vehicles for 60s, with a 0.1s control interval.

Each vehicle observes and shares its headway h, velocity v, and acceleration a to neighbors within two steps.

The safety constraints are: h ??? 1m, v ??? 30m/s, |a| ??? 2.5m/s 2 .

Safe RL is relevant here, but itself is a big topic and out of the scope of this paper.

So we adopt a simple heuristic optimal velocity model (OVM) (Bando et al., 1995) to perform longitudinal vehicle control under above constraints, whose behavior is affected by hyper-parameters: headway gain ??

??? , relative velocity gain ??

??? , stop headway h st = 5m and full-speed headway h go = 35m.

Usually (?? ??? , ?? ??? ) represent the human driver behavior, here we train NMARL to recommend appropriate (?? ??? , ?? ??? ) for each OVM controller, selected from four levels {(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)}. Assuming the target headway and velocity profile are h * = 20m and v * t , respectively, the cost of each agent is

Whenever a collision happens (h i,t < 1m), a large penalty of 1000 is assigned to each agent and the state becomes absorbing.

An additional cost 5(2h st ??? h i,t ) 2 + is provided in training for potential collisions.

Scenario settings.

Since exploring a collision-free CACC strategy itself is challenging for onpolicy RL, we consider simple scenarios.

In Catch-up scenario, v i,0 = v * t =15m/s and h i,0 = h * , ???i = 1, whereas h 1,0 = a ?? h * , with a ??? U [3, 4].

In Slow-down scenario, v i,0 = v * 0 = b??15m/s, b ??? U [1.5, 2.5], and h i,0 = h * , ???i, whereas v * t linearly decreases to 15m/s during the first 30s and then stays at constant.

For fair comparison, all MARL approaches are applied to A2C agents with learning methods in Eq. (3)(4), and only neighborhood observation and communication are allowed.

IA2C performs independent learning, which is an A2C implementation of MADDPG (Lowe et al., 2017) as the critic takes neighboring actions (see Eq. (4)).

ConseNet (Zhang et al., 2018) has the additional consensus update to overwrite parameters of each critic as the mean of those of all critics inside the closed neighborhood.

FPrint (Foerster et al., 2017) includes neighbor policies.

DIAL (Foerster et al., 2016) and CommNet (Sukhbaatar et al., 2016) are described in Section 4.

IA2C, ConseNet, and FPrint are non-communicative policies since they utilize only neighborhood information.

In contrast, DIAL, CommNet, and NeurComm are communicative policies.

Note communicative policies require more messages to be transferred and so higher communication bandwidth.

In particular, the local message sizes are O(|s i | + |?? i | + |h i |) for DIAL and NeurComm, O(|s i | + |h i |) for CommNet, O(|s i | + |?? i |) for FPrint, and O(|s i |) for IA2C and ConseNet.

The implementation details are in C.1.

All algorithms use the same DNN hidden layers: one fully-connected layer for message encoding e ?? , and one LSTM layer for message extracting g ?? .

All hidden layers have 64 units.

The encoding layer implicitly learns normalization across different input signal types.

We train each model over 1M steps, with ?? = 0.99, actor learning rate 5 ?? 10 ???4 , and critic learning rate 2.5 ?? 10 ???4 .

Also, each training episode has a different seed for generalization purposes.

In ATSC, ?? = 0.01, |B| = 120, while in CACC, ?? = 0.05, |B| = 60, to encourage the exploration of collision-free policies.

Each training takes about 30 hours on a 32GB memory, Intel Xeon CPU machine.

We perform ablation study in proposed scenarios, which are sorted as ATSC Monaco > ATSC Grid > CACC Slow-down > CACC Catch-up by task difficulty.

ATSC is more challenging than CACC due to larger scale (>=25 vs 8), more complex dynamics (stochastic traffic flow vs deterministic vehicle dynamics), and longer control interval (5s vs 0.1s).

ATSC Monaco > ATSC Grid due to more heterogenous network, while CACC Slow-down > CACC Catch-up due to more frequently changing leading vehicle profile.

To visualize the learning performance, we plot the learning curve, that is, average episode return (R = 1 T T ???1 t=0 i???V r i,t ) vs training step.

For better visualization, all learning curves are smoothened using moving average with a window size of 100 episodes.

First, we investigate the impact of spatial discount factor, by comparing the learning curves among ?? ??? {0.8, 0.9, 1} for IA2C and CommNet.

Fig. 3 reveals a few interesting facts.

First, ?? * CommNet is always higher than ?? * IA2C .

Indeed, ?? * CommNet = 1 in almost all scenarios (except for ATSC Monaco).

This is because communicative policies perform delayed global information sharing, whereas noncommunicative policies utilize neighborhood information only, causing difficulty to fit the global return.

Second, learning performance becomes much more sensitive to ?? when the task is more difficult.

Specifically, all ?? values lead to similar learning curves in CACC Catch-up, whereas appropriate ?? values help IA2C converge to much better policies more steadily in other scenarios.

Third, ?? * is high enough: ?? * IA2C = 0.9 except for CACC Slow-down where ?? * IA2C = 0.8.

This is because the discounted problem must be similar enough to the original problem in execution.

Next, we investigate the impact of NeurComm under ?? = 1.

We start with a baseline which is similar to existing differentiable protocols, i.e., h i,t = LSTM(h i,t???1 , relu(s Vi,t ) + relu(m Ni,t )).

We then evaluate two intermediate protocols "Concat Only" and "FPrint Only", in which encoded inputs are concatenated and neighbor policies are included, respectively.

Finally we evaluate their combination NeurComm.

As shown in Fig. 3 , all protocols have similar learning curves in easy CACC Catch-up scenario.

Otherwise, both "Concat" and "FPrint" are able to enhance the baseline learning curves in certain scenarios and their affects are additive in NeurComm.

Fig. 4 compares the learning curves of all MARL algorithms, after tuned ?? * ??? {0.6, 0.8, 0.9, 0.95, 1}. As expected, ?? * for non-communicative policies are lower than those for communicative policies.

Tab.

1 summarizes ?? * of controllers across different NMARL scenarios.

For challenging scenarios like ATSC Monaco, lower ?? is preferred by almost all policies (except NeurComm).

This demonstrates that ?? is an effective way to enhance MARL performance in general, especially for challenging tasks like ATSC Monaco.

From another view point, ?? serves as an informative indicator on problem difficulty and algorithm coordination level.

Based on Fig. 4 , NeurComm is at least competitive in CACC scenarios, and it clearly outperforms other policies on both sample efficiency and learning stability in more challenging ATSC scenarios.

Note in CACC a big penalty is assigned whenever a collision happens, so the standard deviation of episode returns is high.

We freeze and evaluate trained MARL policies in another 50 episodes, and summarize the results in Tab.

2.

In CACC scenarios, ?? enhanced FPrint policy achieves the best execution performance.

Note NeurComm still outperforms other communicative algorithms, so this result implies that delayed information sharing may not be helpful in easy but real-time and safety-critical CACC tasks.

In contrast, NeurComm achieves the best execution performance for ATSC tasks.

We also evaluate the execution performance of ATSC and CACC using domain-specific metrics in Tab.

3 and Tab.

4, respectively.

The results are consistent with the reward-defined ones in Tab.

2.

Further, we investigate the performance of top policies in ATSC scenarios.

For each ATSC scenario, we select the top two non-communicative and communicative policies and visualize their impact on network traffic by plotting the time series of network averaged queue length and intersection delay in Fig. 5 .

Note the line and shade show the mean and standard deviation of each metric across execution runs, respectively.

Based on Fig. 5a , NeurComm achieves the most sustainable traffic control in ATSC Grid, so that the congested grid starts recovering immediately after the loading phase ends at 3000s.

During the same unloading phase, CommNet prevents the queues from further increasing while non-communicative policies are failed to do so.

Also, FPrint is less robust than IA2C as it introduces a sudden congestion jump at 1000s.

Similarly, NeurComm achieves the lowest saturation rate in ATSC Monaco (Fig. 5b) .

Intersection delay is another key metric in ATSC.

Based on Fig. 5c , communicative policies are able to reduce intersection delay as well in ATSC Grid, though it is not explicitly included in the objective and so is not optimized by non-communicative policies.

In contrast, communicative policies have fast increase on intersection delay in ATSC Monaco.

This implies that communicative algorithms are able to capture the spatiotemporal traffic pattern in homogeneous networks whereas they still have the risk of overfitting on queue reduction in realistic and heterogenous networks.

For example, they block the short source edges on purpose to reduce on-road vehicles by paying a small cost of queue length.

Finally, we investigate the robustness (string stability) of top policies in CACC scenarios.

In particular, we plot the time series of headway and velocity for the first and the last vehicles in the platoon.

The profile of the first vehicle indicates how adaptively the controller pursues h * and v * , while that of the last vehicle indicates how stable the controlled platoon is.

Based on Tab.

1 and Tab.

4, the top communicative and non-communicative controllers are NeurComm and FPrint.

Fig. 6 shows the corresponding headway and velocity profiles for the selected controllers.

Interestingly, MARL controllers are able to achieve steady state v * and h * for the first vehicle of platoon, whereas they still have difficulty to eliminate the perturbation through the platoon.

This may be because of the heuristic low-level controller as well as the delayed information sharing.

We have formulated the spatiotemporal MDP for decentralized NSC under neighborhood communication.

Further, we have introduced the spatial discount factor to enhance non-communicative MARL algorithms, and proposed a neural communication protocol NeurComm to design adaptive and efficient communicative MARL algorithms.

We hope this paper provides a rethink on developing scalable and robust MARL controllers for NSC, by following practical engineering assumptions and combining appropriate learning and communication methods rather than reusing existing MARL algorithms.

One future direction is improving the recurrent units to naturally control spatiotemporal information flows within the meta-DNN in a decentralized way.

Kaiqing Zhang, Zhuoran Yang, Han Liu, Tong Zhang, and Tamer Ba??ar.

Fully decentralized multiagent reinforcement learning with networked agents.

arXiv preprint arXiv:1802.08757, 2018.

A.1 PROOF OF PROPOSITION 3.1

Proof.

The proof follows the learning method in A2C Mnih et al. (2016) , which shows that

, and v ?? = V ?? ??? (s ?? ), based on on-policy minibatch from a MDP {(s ?? , a ?? , r ?? )} ?? ???B .

Now we consider spatiotemporal MDP, which has transition in Eq. (1), optimizes return in Eq. (2), and collects experience (s i,t , m Nii,t , a i,t ,r i,t ), wherer i,t = j???V ?? dij r j,t .

In Theorem 3.1 of Zhang * i (??|s) and ai???Ai ?? i (a i |s)Q ??i (s, a) under global observations, respectively.

Now assuming the observations and communications are restricted to each neighborhood as in Definition 3.1, then the actor and critic become ?? ??i (s i ) ????? ??i (s) and V ??i (s i , a Ni ) ?????? ??i (s, a ???i ), with the best observability.

Hence, replacing ?? ?? (a|s), V ?? (s), r by ?? ??i (a i |s i ), V ??i (s i , a Ni ), andr i , respectively, we establish Eq. (3)(4) from Eq. (6)(7), which concludes the proof.

Note partial observability and non-stationarity are present in ?? ??i (a i |s i ) and V ??i (s i , a Ni ).

Fortunately, communication improves the observability.

Based on Definition 3.1, any information that agent j knows at time t can be included in m ji,t .

We assume s j,t ??? {m kj,t???1 } k???Nj ??? m ji,t .

The??

Thus,s i,t includes the delayed global observations.

On the other hand, Eq. (1)(2)

where x ??? y if information y is utilized to estimate x, and x 0:t := {x 0 , x 1 , . . .

, x t }.

Proof.

Based on the definition of NeurComm protocol (Eq. (5)), m i,t ??? h i,t???1 , and

= s i,t???1:t ??? {s j,t???1:t , ?? j,t???2:t???1 } j???Ni ??? {s j,t???1 , ?? j,t???2 } j???{V|dij =2}

??? {h j,t???2 } j???{V|dij ???2}

??? . . .

??? s i,0:t ??? {s j,0:t , ?? j,t???2:t???1 } j???Ni ??? {s j,0:t???1 , ?? j,0:t???2 } j???{V|dij =2}

??? . . .

??? {s j,0:t+1???dmax , ?? j,0:t???dmax } j???{V|dij =dmax} , which concludes the proof.

Lemma A.2 (Spatial Gradient Propagation).

In NeurComm, each message is learned to optimize the performance of other agents, that is,

Proof.

If we rewrite the required information for a given hidden state h i,t using intermediate messages instead of inputs, the result of Lemma A.1 becomes

Hence, m i,?? is included in the meta-DNN of agent j at time ?? + d ij ??? 1.

In other words, {?? i , ?? i } receive gradients from L(?? j ), L(?? j ), ???j ??? {V|j = i}, except for the first d ij ??? 1 experience samples.

Assuming d max |B|, {?? i , ?? i } receive almost all gradients from loss signals of all other agents, which concludes the proof.

Algo.

1 presents the algorithm of model training in a synchronous way, following descriptions in Section 3 and 4.

Four iterations are performed at each step: the first iteration (lines 3-5) updates and sends messages; the second iteration (lines 6-10) updates hidden state, policy, and action; the third iteration (lines 11-14) updates value estimation and executes action; the fourth iteration (lines 22-26) performs gradient updates on actor, critic, and neural communication.

On the other hand, Algo.

2 presents the algorithm of decentralized model execution in an asynchronous way.

It runs as a job that repeatedly measures traffic, sends message, receives messages, and performs control.

Algorithm 1: Multi-agent A2C with NeurComm (Training)

ConseNet: same as IA2C but with consensus critic update.

FPrint: h i,t = LSTM(h i,t???1 , concat(relu(s Vi,t ), relu(?? Ni,t???1 ))).

NeurComm: h i,t = LSTM(h i,t???1 , concat(relu(s Vi,t ), relu(?? Ni,t???1 ), relu(h Ni,t???1 ))).

DIAL: h i,t = LSTM(h i,t???1 , relu(s Vi,t ) + relu(relu(h i,t???1 )) + onehot(a i,t???1 )).

CommNet: h i,t = LSTM(h i,t???1 , tanh(s Vi,t ) + linear(mean(h Ni,t???1 ))).

For ConseNet, we only do consensus update on the LSTM layer, since the input and output layer sizes may not be fixed across agents.

Also, the actor and critic are ?? i,t = softmax(h i,t ), and v i,t = linear(concat(h i,t , onehot(a Ni,t ))) C.2 EXPERIMENTS IN ATSC ENVIRONMENT C.2.1 ACTION SPACE Fig. 7 illustrates the action space of five phases for each intersection in the ATSC Grid scenario.

The ATSC Monaco scenario has complex and heterogeneous action spaces, please see the code for more details.

To summarize, there are 11 two-phase intersections, 3 three-phase intersections, 10 four-phase intersections, 1 five-phase intersection, and 3 six-phase intersections.

Table 3 summarizes the key metrics in ATSC.

The spatial average is taken at each second, and then the temporal average is calculated for all metrics (except for trip delay, which is directly aggregated over all trips).

NeurComm outperforms all baselines on minimizing queue length and intersection delay.

Interestingly, even though IA2C is good at optimizing the given objective of queue length, it performs poorly on optimizing intersection and trip delays.

Fig. 8 and Fig. 9 show screenshots of traffic distributions in the grid at different simulation steps for each MARL controller.

The visualization is based on one execution episode with random seed 2000.

Clearly, communicative MARL controllers have better performance on reducing the intersection delay.

NeurComm and CommNet have the best overall performance.

Table 4 summarizes the key metrics in CACC.

The best headway and velocity averages are closest ones to h * = 20m, and v * = 15m/s.

Note the averages are only computed from safe execution episodes, and we use another metric "collision number" to count the number of episodes where an collision happens within the horizon.

Ideally, "collision-free" is the top priority.

However, safe RL is not the focus of this paper so trained MARL controllers cannot achieve this goal in the experiments of CACC.

<|TLDR|>

@highlight

This paper proposes a new formulation and a new communication protocol for networked multi-agent control problems

@highlight

Concerned with N-MARL's where agents update their policy based only on messages from neighboring nodes, showing that introducing a spatial discount factor stabilizes learning.