In recent years, the efficiency and even the feasibility of traditional load-balancing policies are challenged by the rapid growth of cloud infrastructure with increasing levels of server heterogeneity and increasing size of cloud services and applications.

In such many software-load-balancers heterogeneous systems, traditional solutions, such as JSQ, incur an increasing communication overhead, whereas low-communication alternatives, such as JSQ(d) and the recently proposed JIQ scheme are either unstable or provide poor performance.



We argue that a better low-communication load balancing scheme can be established by allowing each dispatcher to have a different view of the system and keep using JSQ, rather than greedily trying to avoid starvation on a per-decision basis.

accordingly, we introduce the Loosely-Shortest -Queue family of load balancing algorithms.

Roughly speaking, in Loosely-shortest -Queue, each dispatcher keeps a different approximation of the server queue lengths and routes jobs to the shortest among them.

Communication is used only to update the approximations and make sure that they are not too far from the real queue lengths in expectation.

We formally establish the strong stability of any Loosely-Shortest -Queue policy and provide an easy-to-verify sufficient condition for verifying that a policy is Loosely-Shortest -Queue.

We further demonstrate that the Loosely-Shortest -Queue approach allows constructing throughput optimal policies with an arbitrarily low communication budget.



Finally, using extensive simulations that consider homogeneous, heterogeneous and highly skewed heterogeneous systems in scenarios with a single dispatcher as well as with multiple dispatchers, we show that the examined Loosely-Shortest -Queue example policies are always stable as dictated by theory.

Moreover, it exhibits an appealing performance and significantly outperforms well-known low-communication policies, such as JSQ(d) and JIQ, while using a similar communication budget.

Background.

In recent years, due to the rapidly increasing size and heterogeneity of cloud services and applications BID3 BID7 BID12 BID19 , the design of load balancing algorithms for parallel server systems has become extremely challenging.

The goal of these algorithms is to efficiently load-balance incoming jobs to a large number of servers, even though these servers display large heterogeneity because of two reasons: First, current large-scale systems increasingly contain, in addition to multiple generations of CPUs (central processing units) BID11 , various types of accelerated devices such as GPUs (graphics processing units), FPGAs (field-programmable gate arrays) and ASICs (application-specific integrated circuit), with significantly higher processing speeds.

Second, VMs (virtual machines) and/or containers are commonly used to deploy different services that share resources on the same servers, potentially leading to significant and unpredictable heterogeneity.

In a traditional server farm, a centralized load-balancer (dispatcher) can rely on a full-state-information policy with strong theoretical guarantees for heterogeneous servers, such as join-theshortest-queue (JSQ), which routes emerging jobs to the server with the shortest queue BID4 BID5 BID12 BID28 BID29 .

This is because in such single-centralized-dispatcher scenarios, the dispatcher forms a single access point to the servers.

Therefore, by merely receiving a notification from each server upon the completion of each job, it can track all queue lengths, because it knows the exact arrival and departure patterns of each queue (neglecting propagation times) BID14 .

The communication overhead between the servers and the dispatcher is at most a single message per job, which is appealing and does not increase with the number of servers.

However, in current clouds, which keep growing in size and thus have to rely on multiple dispatchers BID9 , implementing a policy like JSQ may involve a prohibitive implementation overhead as the number m of dispatchers increases BID14 .

This is because each server needs to keep all m dispatchers updated as jobs arrive and complete, leading to up to O(m) communication messages per job.

This large communication overhead makes scaling the number of dispatchers difficult, and forces cloud dispatchers to rely on heuristics that do not provide any service guarantees with heterogeneous servers.

For instance, in L7 load-balancers, multi-dispatcher services are essentially decomposed into several fully-independent single-dispatcher services, where each dispatcher applies either round-robin or JSQ reduced to its own jobs only BID0 BID20 BID24 .

Unfortunately, such an approach suffers from lack of predictable guarantees, lack of a global view of the system, and communication bursts with potential incast issues.

Related work.

Despite their increasing importance, scalable policies for heterogeneous systems with multiple dispatchers have received little attention in the literature.

In fact, as we later discuss, the only suggested scalable policies that address the many-dispatcher scenario in an heterogeneous setting are based on join-the-idlequeue (JIQ), and none of them is stable BID31 .In the JSQ(d) (power-of-choice) policy, to make a routing decision, a dispatcher samples d ??? 2 queues uniformly at random and chooses the shortest among them BID1 BID2 BID8 BID17 BID30 .

JSQ(d) is stable in systems with homogeneous servers.

However, with heterogeneous servers, JSQ(d) leads to poor performance and even to instability, both with a single and multiple dispatchers BID6 .In the JSQ(d, m) (power-of-memory) policy, the dispatcher samples the m shortest queues from the previous decision in addition to d ??? m ??? 1 new queues chosen uniformly-at-random BID16 BID21 .

The job is then routed to the shortest among these d + m queues.

JSQ(d, m) has been shown to be stable in the case of a single dispatcher, even with heterogeneous servers.

However, it offers poor performance, and has not been considered with multiple dispatchers.

than W R. We complete the proof by using the fact that in W R, the routing decisions do not depend on the system state (unlike JSQ).

Sufficient stability condition.

It can be challenging to prove that a policy is Loosely-Shortest-Queue, i.e., that in expectation, the local dispatcher views are not too far from the real queue lengths.

Therefore, we develop a simple sufficiency condition to prove that a policy belongs to the Loosely-Shortest-Queue family, and exemplify its use.

Intuitively, the condition states that there is a non-zero probability that a server updates a dispatcher at each time-slot.

Example Loosely-Shortest-Queue policies.

Since Loosely-ShortestQueue is not restricted to work with either push (i.e., dispatchers sample the servers) or pull (i.e., servers update the dispatchers) based communication, we aim to achieve the same communication overhead as the lowest-overhead/best-known examples in each class.

Accordingly, we show how two of the newest existing low communication policies are in fact Loosely-Shortest-Queue and how to construct new Loosely-Shortest-Queue policies with communication patterns similar to that of other low-communication policies such as the push-based JSQ(2) and the pull-based JIQ, but with significantly stronger theoretical guarantees and empirical performance.

Extensive simulations.

Using extensive simulations considering homogeneous, heterogeneous, and highly-skewed heterogeneous systems, in scenarios of a single as well as multiple dispatchers, we show how simple Loosely-Shortest-Queue policies are always stable in practice, present appealing performance, and significantly outperform other low-communication policies using an equivalent communication budget.

We consider a system with a set M = {1, 2, . . .

, m} of dispatchers load-balancing incoming jobs among a set N = {1, 2, . . .

, n} of possibly-heterogeneous work-conserving servers.

Time slots.

We assume a time slotted system with the following order of events within each time slot: (1) jobs arrive at each dispatcher; (2) a routing decision is taken by each dispatcher and it immediately forwards its jobs to one of the servers; (3) each server performs its service for this time-slot.

Dispatchers.

As mentioned, each of the m dispatchers does not store incoming jobs, and instead immediately forwards them to one of the n servers.

We denote by a j (t) the number of exogenous job arrivals at dispatcher j at the beginning of time slot t. We make the following assumption: DISPLAYFORM0 That is, we only assume that the total job arrival process to the system is i.i.d.

over time slots and admits finite first and second moments.

The division of arriving jobs among the dispatchers is assumed to follow any arbitrary policy that does not depend on the system state (i.e., queue lengths).

We only assume that there is At each dispatcher, Loosely-Shortest-Queue relies on limited current and past information from the servers to build an approximated local view of all the server queue sizes.

For instance, dispatcher 1 believes that the queue length at server 3 is 1, while it is 2.

It then sends jobs to the shortest queue as dictated by its view (here, to server 3 rather than server n) .

Communication is used only to update the local views of the dispatchers, i.e., to improve their approximations.a positive probability of job arrivals at all dispatchers.

That is, we assume that there exists a strictly positive constant ?? 0 such that DISPLAYFORM1 This, for example, covers complex scenarios with time-varying arrival rates to the different dispatchers that are not necessarily independent.

We are not aware of previous work covering such scenarios.

We further denote a j i (t) as the number of jobs forwarded by dispatcher j to server i at the beginning of time slot t. Let DISPLAYFORM2 be the total number of jobs forwarded to server i at time slot t by all dispatchers.

Servers.

Each server has a FIFO queue for storing incoming jobs.

Let Q i (t) be the queue length of server i at the beginning of time slot t (before any job arrivals and departures at time slot t).

We denote by s i (t) the potential service offered to queue i at time slot t. That is, s i (t) is the maximum number of jobs that can be completed by server i at time slot t. We assume that, for all i ??? N , DISPLAYFORM3 Namely, we assume that the service process of each server is i.i.d.

over time slots and admits finite first and second moments.

We also assume that all service processes are mutually independent across the different servers and also independent of the arrival processes.3 Loosely-Shortest-Queue LOAD BALANCINGIn this section we formally introduce the Loosely-Shortest-Queue family of load balancing policies and then prove that any LooselyShortest-Queue policy is stable.

We assume that at each time slot t, each dispatcher j ??? M holds a local-view estimation of each server's i ??? N queue length, denoted byQ j i (t).

We begin by introducing the two following assumptions on these estimations and the dispatchers way of operation that define the Loosely-Shortest-Queue family of load balancing policies.

Assumption 1 (Local view proximity).

There exists a constant C > 0 such that at the beginning of each time slot (before any arrivals and departures), it holds that DISPLAYFORM0 As we later show, this assumption provides some appealing flexibility when designing a load balancing policy.

Assumption 2 (Local view based routing).

At each time slot, each dispatcher j follows the JSQ policy based on its local view of the queue lengths, i.e., {Q DISPLAYFORM1 .

That is, dispatcher j forwards all its incoming jobs at time slot t to server i * where i * ??? ar??min i {Q DISPLAYFORM2 (ties are broken randomly).Finally, we term a load balancing policy as an Loosely-ShortestQueue (Local Shortest Queue) policy if it respects Assumptions 1 and 2.

We now prove that any Loosely-Shortest-Queue load balancing policy is stable.

We begin by formally stating our considered form of stability.

Definition 1 (Strong stability).

We say that the system is strongly stable iff there exists a constant K ??? 0 such that DISPLAYFORM0 That is, the system is strongly stable when the expected time averaged sum of queue lengths admits a constant upper bound.

Strong stability is a strong form of stability that implies finite average backlog and (by Little's theorem) finite average delay.

Furthermore, under mild conditions, it implies other commonly considered forms of stability, such as steady state stability, rate stability, mean rate stability and more (see BID18 ).

Note that strong stability has been widely used in queueing systems (see BID10 and references therein) whose state does not necessarily admit an irreducible and aperiodic Markov chain representation (therefore positive-recurrence may not be considered).

Theorem 1.

Assume the system is sub-critical, i.e., assume that there exists ?? > 0 such that DISPLAYFORM1 Then, any Loosely-Shortest-Queue policy is strongly stable.

Proof.

A server can work on a job immediately upon its arrival.

Therefore, the queue dynamics at server i are given by DISPLAYFORM2 where [??] + ??? max {??, 0}. Squaring both sides of (10) yields DISPLAYFORM3 Rearranging FORMULA10 and omitting the last term yields DISPLAYFORM4 Summing over the servers yields DISPLAYFORM5 where DISPLAYFORM6 We would like to proceed in our analysis by taking the expectation of BID12 .

However, first we need to analyze the term DISPLAYFORM7 and {a i (t)} n i=1 are dependent.

Our plan is to use the following recipe.

We will introduce two additional policies into our analysis: (1) JSQ and (2) WeightedRandom (W R).

Roughly speaking, we will show that the routing decision that is taken by our policy at each dispatcher and each time slot t is sufficiently similar to the decision that would have been made by JSQ given the same system state, which, in turn, is no worse than the decision that W R would make at that time slot.

Since in W R the routing decisions taken at time slot t do not depend on the system state at time slot t, we will obtain the desired independence, which allows us to continue with the analysis.

We start by introducing the corresponding JSQ and W R notations.

DISPLAYFORM8 be the number of jobs that will be routed to server i at time slot t when using JSQ at time slot t. That is, each dispatcher forwards its incoming jobs to the server with the shortest queue (ties are broken randomly).

Formally, let i * ??? ar??min i {Q i (t)}, then ???j ??? M a j, J SQ i DISPLAYFORM9 be the number of jobs that will be routed to server i at time slot t when using W R at time slot t. That is, each dispatcher forwards its incoming jobs to a single randomly-chosen server, where the probability of choosing server i is DISPLAYFORM10 With these notations at hand, we continue our analysis by adding and subtracting the term 2 n i=1 a

(t)Q i (t) from the right hand side of BID12 .

This yields DISPLAYFORM0

We would like to take the expectation of BID16 .

However, as mentioned, since the actual queue lengths and the local views of the dispatchers and the routing decisions that are made both by our policy and JSQ are dependent, we shall rely on the W R policy and the expected distance of the local views from the actual queue lengths to evaluate the expected values.

To that end, we now introduce the following lemmas.

Proof.

See Section 7.1.

??? Lemma 2.

For all servers i ??? N and all time slots t, it holds that DISPLAYFORM0 Proof.

See Section 7.2.

??? Using Lemmas 1 and 2 in (17) yields DISPLAYFORM1 Taking the expectation of (18) yields DISPLAYFORM2 We now observe that both a(t) (according to (1) ) and DISPLAYFORM3 (according to the definition of the W R policy) are independent of DISPLAYFORM4 Applying this observation to BID18 and using the linearity of expectation yields DISPLAYFORM5 Next, since for any non-negative {x 1 , x 2 , . . .

, x n } such that (5) - FORMULA3 , the linearity of expectation and (1) - FORMULA0 , we obtain DISPLAYFORM6 DISPLAYFORM7 Additionally, using (1) , (2) and FORMULA4 yields DISPLAYFORM8 Finally, since the decisions taken by the W R policy are independent of the system state, we can introduce the following lemma.

Lemma 3.

For all i ??? N and t it holds that DISPLAYFORM9 Proof.

See Section 7.3.

???Using FORMULA27 , BID21 and Lemma 3 in (20) yields DISPLAYFORM10 For ease of exposition, denote the constants DISPLAYFORM11 DISPLAYFORM12 Rearranging FORMULA1 and using FORMULA31 and FORMULA32 yields DISPLAYFORM13 Summing FORMULA3 over time slots [0, 1, . . .

,T ???1], noticing the telescopic series at the right hand side of the inequality and dividing by 2??T yields DISPLAYFORM14 Taking limits of BID27 and making the standard assumption that the system starts its operation with finite queue lengths, i.e., DISPLAYFORM15 This implies strong stability and concludes the proof.

???

As mentioned, in order to establish that a policy is Loosely-ShortestQueue, Assumption 1 has to hold.

That is DISPLAYFORM0 Generally, it may be challenging to establish that this condition holds.

To that end, we now develop a simplified sufficient condition.

As we later demonstrate, this simplified condition captures a broad family of communication techniques among the servers and the dispatchers and allows for the design of stable policies with appealing performance and extremely low communication budgets.

DISPLAYFORM1 be an indicator function that obtains the value 1 iff server i updates dispatcher j (via the push-based sampling or the pull-based update message from the server) with its actual queue length at the end of time slot t (after arrivals and departures at time slot t).

Assume that there exists?? > 0 such that DISPLAYFORM2 Then, Assumption 1 holds.

Namely, we assume that there is a strictly positive probability of an update at the end of time slot t, given any gap between the actual queue lengths at the beginning of that time slot and the dispatcher local views.

Proof.

Fix server i and dispatcher j. Denote DISPLAYFORM3

Taking expectation of (31) yields DISPLAYFORM0 Next, using the law of total expectation DISPLAYFORM1 where the last inequality follows from the linearity of expectation and (30).

Now, using (33) in (32) yields DISPLAYFORM2 With this result at hand, we can finish our proof by an inductive claim.

Fix DISPLAYFORM3 We now show that for all t it holds that E[Z (t)] ??? C * .

Basis.

For t = 0 the claim trivially holds since DISPLAYFORM4 Inductive step.

Using the induction hypothesis in (34) yields DISPLAYFORM5 where we used the fact that by the definition of C * it holds that ?? (1) + max i ??(1) i ?????C * .

Note that the obtained bound is not dependent on (i, j, t).

This concludes the proof.

???

Since Loosely-Shortest-Queue is not restricted to work with either pull-or push-based communications, in this section we provide examples for both.

In a push-based policy, the dispatchers sample the servers for their queue length whereas in a pull-based policy the servers may update the dispatchers with their queue length.

While empirically, we will see that the pull-based approach can provide better performance in many scenarios, it may also incur additional implementation overhead because it requires the servers to actively update the dispatchers given some state conditions, rather than passively answer sample queries.

Therefore we are inclined to consider both the push and pull frameworks.

In both frameworks we consider policies with low communication overhead that can be unstable even with a single dispatcher, and provide an alternative Loosely-Shortest-Queue policy with similar communication budget that is strongly stable for any number of dispatchers.

The JSQ(d) policy forms a popular low-communication push-based load balancing approach.

As mentioned, JSQ(d) is not stable in heterogeneous systems even for a single dispatcher.

Instead, we will now extend a push-based Loosely-ShortestQueue policy that uses exactly the same communication pattern between the servers and the dispatchers.

Specifically, each dispatcher holds a local array of the server queue length approximations and sends jobs to the minimum one among them.

The approximations are updated as follows: (1) when a dispatcher sends jobs to a server, these jobs are added to the respective local approximation; (2) at each time slot, if new jobs arrive, the dispatcher randomly samples d distinct queues and uses this information only to update the respective d distinct entries in its local array to their actual value.

Algorithm 1 (termed Loosely-Shortest-Queue-Sample(d)) depicts the actions taken by each dispatcher at each time slot.

Remark 1.

Note that Loosely-Shortest-Queue-Sample(d) is not a new policy but is considered for a single dispatcher and homogenous servers in BID13 .The simplicity of Loosely-Shortest-Queue-Sample(d) may be surprising.

For instance, there is no attempt to guess or estimate how the other dispatchers send traffic or how the queue drains to get a better estimate, i.e., our estimate is based only on the jobs that the specific dispatcher sends and the last time it sampled a queue.

We also do not take the age of the information into account.

Furthermore, as we find below, the stability proof of LooselyShortest-Queue-Sample(d) only relies on the sample messages and not on the job increments.

We empirically find that these increments help improve the estimation quality and therefore the performance.

We now prove that using Loosely-Shortest-Queue-Sample(d) at each dispatcher results in strong stability in multi-dispatcher heterogeneous systems.

Remarkably, this result holds even for d = 1.

Proposition 1.

Assume that the system is sub-critical and each dispatcher uses Loosely-Shortest-Queue-Sample(d).

Then, the system is strongly stable.

Proof.

Fix dispatcher j and server i. Consider time slot t. By (4), with probability of at least ?? 0 , dispatcher j samples d out of n Algorithm 1: Loosely-Shortest-Queue-Sample(d) (push-based Loosely-Shortest-Queue example)Code for dispatcher j ??? M ; Route jobs:

foreach time slot t do Forward jobs to server i *

??? ar ??min i Q j i (t ) ; DISPLAYFORM0 DISPLAYFORM1 end end Update local state: foreach time slot t do if new jobs arrive at time slot t then Uniformly at random pickservers uniformly at random disregarding the system state at time slot t.

Therefore, we obtain DISPLAYFORM2 This respects the simplified sufficiency condition and thus concludes the proof.

???

JIQ is a popular, recently proposed, low-communication pull-based load balancing policy.

It offers a low communication overhead that is upper-bounded by a single message per job BID14 .

However, as mentioned, for heterogeneous systems, JIQ is not stable even for a single dispatcher.

We now propose a different pull-based Loosely-Shortest-Queue policy that conforms with the same communication upper bound, namely a single message per job, and leverages the important idleness signals from the servers.

Specifically, each server, upon the completion of one or several jobs at the end of a time slot, sends its queue length to a dispatcher, which is chosen uniformly at random, using the following rule: (1) if the server becomes idle, then the message is sent with probability 1; (2) otherwise, the message is sent with probability 0 < p ??? 1 where p is a fixed parameter.

Algorithm 2 (termed Loosely-Shortest-Queue-U pdate(p)) depicts the actions taken by each dispatcher at each time slot.

The intuition behind this approach is to always leverage the idleness signals in order to avoid immediate starvation as done by JIQ; yet, in contrast to JIQ, even when no servers are idle, we want to make sure that the local views are not too far from the approximations, which provides significant advantage at high loads.

Remark 2.

Note that Loosely-Shortest-Queue-U pdate(p) is not a new policy but, to the best of our knowledge, is similar to the policy considered in BID25 for a single dispatcher and homogeneous servers.

We now formally prove that using Loosely-Shortest-Queue-U pdate(p) results in strong stability in multi-dispatcher heterogeneous systems.

Remarkably, this result holds for any p > 0.

foreach arrived message ???i, q ??? at time slot t do UpdateQ j i (t ) ??? q; end end Code for server i ??? N ; Send update message: foreach time slot t do if completed jobs at time slot t then Uniformly at random pick j ??? M ; if idle then Send ???i, Q i (t )??? to dispatcher j; elseSend ???i, Q i (t )??? to dispatcher j w .p .

p; end end end Proposition 2.

Assume that the system is sub-critical and each dispatcher uses Loosely-Shortest-Queue-U pdate(p).

Then, the system is strongly stable.

Proof.

Fix dispatcher j, server i and time slot t. Our goal is to prove that (30) holds.

To do so, we examine two possible events at the beginning of time slot t: (1) Q i (t) = 0 and (2) Q i (t) > 0.(1) Since Q i (t) = 0, the server updated at least one dispatcher in a previous time slot, i.e., for at least one dispatcher j * we have that Q j * i (t) = 0.

This must hold since there is a dispatcher that received the update message after this queue got empty (that is, when a server becomes idle, a message is sent w.p.

1).

Now consider the DISPLAYFORM0 Since the tie breaking rule is random, by (1), FORMULA1 , (2) , (5) and (6) , there exists?? i > 0 such that P(A 1 ) >?? i .

Since a(t) and s i (t) are not dependent on any system information at the beginning of time slot t we obtain DISPLAYFORM1 there is a strictly positive probability that a job would be completed at this time slot.

That is, since s i (t) is not dependent on any system information at the beginning of time slot t we obtain DISPLAYFORM2 is a convex combination of the left hand sides of (36) and (37) we obtain that DISPLAYFORM3 This concludes the proof.

??? TAB1 summarizes the stability properties and the worst case communication requirements of the evaluated load balancing techniques as dictated by theory and verified by our evaluations.

Before moving to simulations, we discuss several practical considerations when considering different load balancing approaches.

Instantaneous routing.

An appealing property of the LooselyShortest-Queue policy, similarly to JIQ, is that a dispatcher can immediately take routing decisions upon a job arrival.

This is in contrast to common push-based policies that have to wait for a response from the sampled servers to be able to make a decision.

For example, when using the JSQ(2) policy, when a job arrives the dispatcher cannot immediately send the job to a server but must pay the additional delay of sampling two servers.

Note that Assumption 1 trivially applies for any additional constant time delay in update messages among the servers and the dispatchers.

Space requirements.

To implement the Loosely-Shortest-Queue policy, similarly to JSQ, each dispatcher has to hold an array of size n with all server queue length estimations.

It is important to note that such a space requirement incurs negligible overhead on a modern server.

For example, nowadays, any commodity server has tens to hundreds GB of DRAM.

But even a hypothetical cluster with 10 6 servers requires only a few MB of the dispatcher's memory, which is negligible in comparison to the DRAM size.

Computational complexity.

To implement the Loosely-ShortestQueue policy, similarly to JSQ, each dispatcher has to find the minimum (approximated) queue length and route the incoming jobs to it.

At first glance, it might seem that this requires O(n) operations for each decision making, which is a disadvantage in comparison to JIQ and JSQ(2) that require a constant number of operations per decision, irrespective of the size of the system.

However, by using a priority queue (e.g., min-heap), finding the minimum results in only a single operation (i.e., simply looking at the head of the priority queue).

For a queue length update operation, O(log n) operations are required in the worst case (e.g., decrease-key operation in a min-heap).

Even with n = 10 6 , just a few operations are required in the worst case per queue length update.

This results in a single commodity core being able to perform tens to hundreds of millions of such updates per second, hence resulting in negligible overhead, especially for a low-communication policy in which queue length updates are not too frequent.

In this section we conduct an extensive evaluation of the LooselyShortest-Queue approach, using both the Loosely-Shortest-QueueSample(d) (with d = 1 and d = 2) and Loosely-Shortest-Queue-U pdate(p) (with p = 1 and p = 0.01) families of LooselyShortest-Queue algorithms.

We compare them to the baseline fullinformation JSQ (for JSQ only, dispatchers take decisions sequentially to allow for a "water-filling" behaviour and avoid incast) and to the low-communication JSQ(2) and JIQ.

We consider heterogeneous systems with low and high skew, in small single-dispatcher and larger-scale multi-dispatcher scenarios.

For completeness, we also present the results for homogeneous systems in Appendix A.For each scenario and each policy, we measure the time-averaged number of jobs in the system.

This measure also translates to the mean delay by Little's Law, and reveals the stability region of each policy.

Additionally, even though we have theoretical guarantees for the worst-case communication overhead incurred by each policy, we also measure the total average communication overhead per time slot.

We start by considering a mix of weak and strong servers with a ratio of 2 between their service rates, thus exhibiting a moderate degree of heterogeneity.

In this subsection the job arrival process at a dispatcher is a Poisson process with parameter ?? and the server service processes are geometrically distributed with a parameter 2p for a weak server and a parameter p for a strong server.

In a simulation with n s strong servers, n w weak servers and m dispatchers we set p = n s +0.5n w 100mand sweep 0 ??? ?? < 100.

In this evaluation, we consider a small scale scenario with a single dispatcher and 10 heterogeneous servers.

The results are depicted in FIG4 .

As expected, Loosely-ShortestQueue-U pdate(1) performs identically to JSQ since in a single dispatcher scenario both are aware of all queue lengths at all times.

It is evident that, in all three scenarios, JIQ is not stable whereas our LSQ-U pdate(0.01) is stable with a similar communication overhead.

Additionally, in all three scenarios, Loosely-Shortest-QueueSample(2) performs better than JSQ(2), which appears to be stable in this scenario (we tested up to a normalized load of 0.995).

In this evaluation, we consider a larger scale scenario with 10 dispatchers and 100 heterogeneous servers.

The results are depicted in FIG6 and show similar trends.

LooselyShortest-Queue-U pdate(1) performs slightly worse than JSQ but with a significantly lower communication budget (by an order of magnitude).

Similarly to the previous scenario, it is evident that, in all three scenarios, JIQ is not stable again, and its stability region decreases as the proportion of weak servers decreases.

In contrast, our LSQ-U pdate(0.01) is always stable, with a similar communication overhead.

Again, in all three scenarios, Loosely-Shortest-QueueSample(2) performs better then JSQ(2), which appears to be stable in this scenario as well.

In this subsection, we consider systems with highly skewed heterogeneous servers.

That is, we test the different approaches with a mix of weak and strong servers, and assume a ratio of 10 between their service rates.

Again, in this subsection the job arrival process at a dispatcher is a Poisson process with parameter ?? but the server service processes are geometrically distributed with parameter 10p for a weak server and a parameter p for a strong server.

In a simulation with n s strong Throughput optimality Comm.

overhead

Per job arrival servers, n w weak servers and m dispatchers we set p = n s +0.1n w 100m DISPLAYFORM0 and sweep 0 ??? ?? < 100.

In this evaluation, we consider a small scale scenario with a single dispatcher and 10 highly skewed heterogeneous servers.

The results are depicted in FIG7 .

As expected, Loosely-Shortest-Queue-U pdate(1) performs identically to JSQ.

Again, in all three scenarios, JIQ is not stable, and its stability region has decreased dramatically for this larger skew in server service rates.

For example, for a mix of 5 strong and 5 weak servers, its stability region is only up to ???78%.

In contrast, our LSQ-U pdate(0.01) is always stable with a similar communication overhead.

JSQ (2) is not stable as well with a dramatic degradation, especially when the number of strong and weak servers is similar.

For example, in a scenario with 5 strong and 5 weak servers, the stability region of JSQ (2) is only up to ???44%.

On the other hand, Loosely-Shortest-Queue-Sample(2) is always stable with an identical communication overhead.

Note that, as dictated by theory, even Loosely-Shortest-Queue-Sample(1) is always stable with even less communication overhead.

In this evaluation, we consider a larger scale scenario with 10 dispatchers and 100 highly skewed heterogeneous servers.

The results are depicted in FIG8 .

The results show a similar trend to the results in the small-scale simulation.

It is evident that the stability regions of both JSQ(2) and JIQ suffer from a significant degradation that appears to be consistent with an increasing number of dispatchers.

On the other hand, all LSQ approaches are stable and keep performing well using similar communication overhead.

It is evident that LSQ-U pdate(1) is only slightly worse than JSQ but, again, with a significantly lower communication overhead (approximately by an order of magnitude).

Again LSQ-U pdate(0.01) and LSQ-Sample(2) use similar communication budgets as JIQ and JSQ(2), respectively, but are stable with good performance.

The Loosely-Shortest-Queue approach always guarantees stability and it achieves this using the same communication budget as other non-throughput-optimal low-communication techniques.

Additionally, the simulations indicate that, under identical low-communication requirements, Loosely-Shortest-Queue consistently exhibits good performance in different scenarios whereas other low-communication techniques are either unstable or offer poor performance.

Remark 3.

Interestingly, by allowing different dispatchers to have a different view of the system, Loosely-Shortest-Queue indirectly appears to solve the incast problem JSQ may incur in a parallel multidispatcher system.

Also, evaluation results present an interesting trade-off between the push-based LSQ-Sample(d) and the pull-based LSQ-U pdate(p) approaches for small d and p values (i.e., extremely low communication overhead).

LSQ-U pdate(p) appears to be consistently better at low loads (where servers keep getting idle frequently), whereas LSQSample(d) appears to be consistently better at high loads where idleness becomes rare and the approximations of LSQ-U pdate(p) are less effective than in the push approach.

We have shown, both formally and in simulations, that the LooselyShortest-Queue approach offers strong theoretical guarantees and appealing performance with low communication overhead.

Interestingly, by virtue of Theorem 2, we can construct various strongly stable Loosely-Shortest-Queue policies with any arbitrarily low communication budget, disregarding whether the system uses pull or push messages (or both).

This, in fact, appears to generalize a result from BID25 to multiple dispatchers and heterogeneous servers.

Let M(t) be the number of queue length updates performed by all dispatchers up to time t. Fix any arbitrary small r > 0.

Suppose that we want to achieve strong stability, such that the average message rate is at most r , i.e., for all t we have that E[M(t)]

??? rt.

Then, the two following per-time-slot dispatcher sampling rules trivially achieve strong stability (by Theorem 2) and respect the desired communication bound, i.e., E[M(t)]

??? rt.

Example 1 (Push-based communication example.).

Dispatcher sampling rule upon job(s) arrival: (1) pick a server i ??? N uniformly at random; (2) sample server i with probability r m .Example 2 (Pull-based communication example.).

Server messaging rule upon job(s) completion: (1) pick a dispatcher j ??? M uniformly at random; (2) update dispatcher j with probability r n .Theorem 2 also enables to design strongly stable LooselyShortest-Queue policies with hybrid communication (e.g., push and pull) that can attempt to maximize the benefits of both approaches (for example, as we have seen, for extremely low communication overhead requirements, at low loads LSQ-U pdate(p) appears to be more effective whereas at high loads LSQ-Sample(d) appears to be more effective).

For example, the following policy leverages both the advantages of JIQ (i.e., being immediately notified that a server becomes idle) and JSQ(d) (i.e., random exploration of shallow queues when no servers are idle).Example 3 (Hybrid communication example.).

Dispatcher sampling rule: (1) pick a server i ??? N uniformly at random; (2) sample server i with probability r m .

Server messaging rule: (1) if got idle, pick a dispatcher j ??? M uniformly at random and send it an update message.

The above three examples demonstrate the wide range of possibilities that the Loosely-Shortest-Queue approach offers to the design of stable, scalable policies with low communication overhead.

This section provides the proofs of the various lemmas that we employed towards establishing our main theoretical result, i.e., Theorem 1.

First, by definition DISPLAYFORM0 Therefore, both a DISPLAYFORM1 are feasible solutions to the optimization problem given by DISPLAYFORM2 The optimal solution to this problem is simply DISPLAYFORM3 which is exactly the way JSQ policy operates.

That is, DISPLAYFORM4 Clearly, any other feasible solution, e.g., DISPLAYFORM5 , cannot be better.

This concludes the proof.

???

Expanding the term DISPLAYFORM0 We now substitute DISPLAYFORM1 We now introduce the following lemma:Lemma 4.

For all t it holds that DISPLAYFORM2 Proof.

See Section 7.4.

??? Using Lemma 4 in (40) yields DISPLAYFORM3 Now, using the fact that xy ??? |x ||y| for all (x, y) ??? R 2 on (41) yields DISPLAYFORM4 Finally, it trivially holds that DISPLAYFORM5 Using (43) in (42) concludes the proof.

???

Each dispatcher applies the WR policy independently.

Therefore, by applying (5) , (6) , (1) and (2) we have that the expected number of jobs arriving at each server i is DISPLAYFORM0 Using (5), (6), BID8 and (44) yields DISPLAYFORM1 This concludes the proof.

???

Fix j = j * .

It is sufficient to show that DISPLAYFORM0 The proof now follows similar lines to the proof of Lemma 1.

By definition DISPLAYFORM1 Therefore, both a DISPLAYFORM2 and a DISPLAYFORM3 are feasible solutions to the optimization problem given by DISPLAYFORM4 The optimal solution to this problem is simply a j * (t) min i Q j * i (t) , which is exactly the way our policy operates since it performs JSQ considering Q j * i (t) instead of {Q i (t)}. That is DISPLAYFORM5 Any other feasible solution including a DISPLAYFORM6 cannot be better when considering Q j * i (t) instead of {Q i (t)}. This proves the inequality in (46) and thus concludes the proof.

???

In this paper, we introduced the Loosely-Shortest-Queue family of load balancing algorithms.

We formally established that any Loosely-Shortest-Queue policy is strongly stable and further developed a simplified sufficient condition for establishing that a policy is Loosely-Shortest-Queue.

We then demonstrated that the LooselyShortest-Queue approach allows to construct stable policies with arbitrary low communication budgets for system with multiple dispatchers and heterogeneous servers.

Using extensive simulations that consider homogeneous, heterogeneous and highly skewed heterogeneous systems in small single-dispatcher and larger-scale multi-dispatcher scenarios, we illustrated how simple low-communication Loosely-Shortest-Queue known policies are stable and at the same time exhibit appealing performance.

Our example policies significantly outperform wellknown low-communication policies such as JSQ(2) and JIQ, while obeying the same constraints on the communication overhead.

Given the strength of the Loosely-Shortest-Queue approach in large-scale multi-dispatcher heterogeneous systems, we believe that it has the potential to open a new thread in the research of scalable load balancing policies.

the best performance which is identical to the baseline, i.e., JSQ.

This is because in a single dispatcher scenario Loosely-ShortestQueue-U pdate(1) is always aware of the exact queue length of all queues.

Loosely-Shortest-Queue-U pdate(0.01) offers better performance than JIQ especially as the load increases.

This is achieved with similar average communication overhead.

It is notable that JIQ performs similarly to JSQ at low loads, but its performance quickly degrades as the load increases.

This complies with the latest theoretical results indicating that JIQ is asymptotically worse than JSQ at high loads (i.e., JIQ is not heavy traffic delay optimal) BID31 .Finally, Loosely-Shortest-Queue-U pdate(2) is always better than its JSQ(2) counterpart using exactly the same communication overhead.

Loosely-Shortest-Queue-U pdate(1) is slightly worse in this scenario but with a lesser communication overhead.

In this evaluation, we consider a larger scale scenario with 10 dispatcher and 100 homogeneous servers.

The results are depicted in FIG11 .

Similarly to the previous scenario, as dictated by theory, it is evident that all tested approaches are stable in this scenario as well.

Our Loosely-Shortest-Queue-U pdate(1) achieves good performance which is slightly worse than JSQ while using by an order of magnitude less communication.

The performance slightly degrades since in this multiple dispatchers scenario Loosely-Shortest-Queue-U pdate(1) cannot always be aware of the exact queue length of all queues.

Again, Loosely-Shortest-Queue-U pdate(0.01) is always better than JIQ with similar average communication overhead and Loosely-Shortest-Queue-U pdate(2) is always better than its JSQ(2) counterpart using exactly the same communication overhead.

<|TLDR|>

@highlight

Scalable and low communication load balancing solution for heterogeneous-server multi-dispatcher systems with strong theoretical guarantees and promising empirical results. 