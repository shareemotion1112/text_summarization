Reinforcement learning (RL) agents improve through trial-and-error, but when reward is sparse and the agent cannot discover successful action sequences, learning stagnates.

This has been a notable problem in training deep RL agents to perform web-based tasks, such as booking flights or replying to emails, where a single mistake can ruin the entire sequence of actions.

A common remedy is to "warm-start" the agent by pre-training it to mimic expert demonstrations, but this is prone to overfitting.

Instead, we propose to constrain exploration using demonstrations.

From each demonstration, we induce high-level "workflows" which constrain the allowable actions at each time step to be similar to those in the demonstration (e.g., "Step 1: click on a textbox; Step 2: enter some text").

Our exploration policy then learns to identify successful workflows and samples actions that satisfy these workflows.

Workflows prune out bad exploration directions and accelerate the agent’s ability to discover rewards.

We use our approach to train a novel neural policy designed to handle the semi-structured nature of websites, and evaluate on a suite of web tasks, including the recent World of Bits benchmark.

We achieve new state-of-the-art results, and show that workflow-guided exploration improves sample efficiency over behavioral cloning by more than 100x.

We are interested in training reinforcement learning (RL) agents to use the Internet (e.g., to book flights or reply to emails) by directly controlling a web browser.

Such systems could expand the capabilities of AI personal assistants BID42 , which are currently limited to interacting with machine-readable APIs, rather than the much larger world of human-readable web interfaces.

Reinforcement learning agents could learn to accomplish tasks using these human-readable web interfaces through trial-and-error BID44 .

But this learning process can be very slow in tasks with sparse reward, where the vast majority of naive action sequences lead to no reward signal BID46 BID30 .

This is the case for many web tasks, which involve a large action space (the agent can type or click anything) and require a well-coordinated sequence of actions to succeed.

A common countermeasure in RL is to pre-train the agent to mimic expert demonstrations via behavioral cloning BID37 BID23 , encouraging it to take similar actions in similar states.

But in environments with diverse and complex states such as websites, demonstrations may cover only a small slice of the state space, and it is difficult to generalize beyond these states (overfitting).

Indeed, previous work has found that warm-starting with behavioral cloning often fails to improve over pure RL BID41 .

At the same time, simple strategies to combat overfitting (e.g. using fewer parameters or regularization) cripple the policy's flexibility BID9 , which is required for complex spatial and structural reasoning in user interfaces.

In this work, we propose a different method for leveraging demonstrations.

Rather than training an agent to directly mimic them, we use demonstrations to constrain exploration.

By pruning away bad exploration directions, we can accelerate the agent's ability to discover sparse rewards.

Furthermore, for all demonstrations d do Induce workflow lattice from d

Observe an initial environment state πw samples a workflow from a lattice Roll out an episode e from the workflow Use e to update πw if e gets reward +1 then Add e to replay buffer Periodically: if replay buffer size > threshold then Sample episodes from replay buffer Update πn with sampled episodes Observe an initial environment state πn rolls out episode e Update πn and critic V with e if e gets reward +1 then Add e to replay bufferFigure 1: Workflow-guided exploration (WGE).

After inducing workflow lattices from demonstrations, the workflow policy π w performs exploration by sampling episodes from sampled workflows.

Successful episodes are saved to a replay buffer, which is used to train the neural policy π n .because the agent is not directly exposed to demonstrations, we are free to use a sophisticated neural policy with a reduced risk of overfitting.

To constrain exploration, we employ the notion of a "workflow" BID13 .

For instance, given an expert demonstration of how to forward an email, we might infer the following workflow:Click an email title → Click a "Forward" button → Type an email address into a textbox → Click a "Send" button This workflow is more high-level than an actual policy: it does not tell us exactly which email to click or which textbox to type into, but it helpfully constrains the set of actions at each time step.

Furthermore, unlike a policy, it does not depend on the environment state: it is just a sequence of steps that can be followed blindly.

In this sense, a workflow is environment-blind.

The actual policy certainly should not be environment-blind, but for exploration, we found environment-blindness to be a good inductive bias.

To leverage workflows, we propose the workflow-guided exploration (WGE) framework as illustrated in Figure 1: 1.

For each demonstration, we extract a lattice of workflows that are consistent with the actions observed in the demonstration (Section 3).2.

We then define a workflow exploration policy π w (Section 4), which explores by first selecting a workflow, and then sampling actions that fit the workflow.

This policy gradually learns which workflow to select through reinforcement learning.3.

Reward-earning episodes discovered during exploration enter a replay buffer, which we use to train a more powerful and expressive neural network policy π n (Section 5).A key difference between the web and traditional RL domains such as robotics BID5 or game-playing BID8 is that the state space involves a mix of structured (e.g. HTML) and unstructured inputs (e.g. natural language and images).

This motivates us to propose a novel neural network policy (DOMNET), specifically designed to perform flexible relational reasoning over the tree-structured HTML representation of websites.

We evaluate workflow-guided exploration and DOMNET on a suite of web interaction tasks, including the MiniWoB benchmark of BID41 , the flight booking interface for Alaska Airlines, and a new collection of tasks that we constructed to study additional challenges such as noisy environments, variation in natural language, and longer time horizons.

Compared to previous results on MiniWoB BID41 , which used 10 minutes of demonstrations per task (approximately 200 demonstrations on average), our system achieves much higher success rates and establishes new state-of-the-art results with only 3-10 demonstrations per task.

In the standard reinforcement learning setup, an agent learns a policy π(a|s) that maps a state s to a probability distribution over actions a. At each time step t, the agent observes an environment state s t and chooses an action a t , which leads to a new state s t+1 and a reward r t = r(s t , a t ).

The goal is to maximize the expected return E[R], where R = t γ t r t+1 and γ is a discount factor.

Typical reinforcement learning agents learn through trial-and-error: rolling out episodes (s 1 , a 1 , . . .

, s T , a T ) and adjusting their policy based on the results of those episodes.

We focus on settings where the reward is delayed and sparse.

Specifically, we assume that (1) the agent receives reward only at the end of the episode, and (2) the reward is high (e.g., +1) for only a small fraction of possible trajectories and is uniformly low (e.g., −1) otherwise.

With large state and action spaces, it is difficult for the exploration policy to find episodes with positive rewards, which prevents the policy from learning effectively.

We further assume that the agent is given a goal g, which can either be a structured key-value mapping (e.g., {task: forward, from: Bob, to: Alice}) or a natural language utterance (e.g., "Forward Bob's message to Alice").

The agent's state s consists of the goal g and the current state of the web page, represented as a tree of elements (henceforth DOM tree).

We restrict the action space to click actions Click(e) and type actions Type(e,t), where e is a leaf element of the DOM tree, and t is a string from the goal g (a value from a structured goal, or consecutive tokens from a natural language goal).

FIG2 shows an example episode for an email processing task.

The agent receives +1 reward if the task is completed correctly, and −1 reward otherwise.

Given a collection of expert demonstrations d = (s 1 ,ã 1 , . . .

,s T ,ã T ), we would like explore actions a t that are "similar" to the demonstrated actionsã t .

Workflows capture this notion of similarity by specifying a set of similar actions at each time step.

Formally, a workflow z 1:T is a sequence of workflow steps, where each step z t is a function that takes a state s t and returns a constrained set z t (s t ) of similar actions.

We use a simple compositional constraint language (Appendix A) to describe workflow steps.

For example, with z t = Click(Tag("img")), the set z t (s t ) contains click actions on any DOM element in s t with tag img.

We induce a set of workflows from each demonstration d = (s 1 ,ã 1 , . . .

,s T ,ã T ) as follows.

For each time step t, we enumerate a set Z t of all possible workflow steps z t such thatã t ∈ z t (s t ).

The set of workflows is then the cross product Z 1 × · · · × Z T of the steps.

We can represent the induced workflows as paths in a workflow lattice as illustrated in FIG2 .To handle noisy demonstrations where some actions are unnecessary (e.g., when the demonstrator accidentally clicks on the background), we add shortcut steps that skip certain time steps.

We also add shortcut steps for any consecutive actions that can be collapsed into a single equivalent action (e.g., collapsing two type actions on the same DOM element into a single Type step).

These shortcuts allow the lengths of the induced workflows to differ from the length of the demonstration.

We henceforth ignore these shortcut steps to simplify the notation.

The induced workflow steps are not equally effective.

For example in FIG2 , the workflow step Click(Near(Text("Bob"))) (Click an element near text "Bob") is too specific to the demonstration scenario, while Click(Tag("div")) (Click on any <div> element) is too general and covers too many irrelevant actions.

The next section describes how the workflow policy π w learns which workflow steps to use.

DISPLAYFORM0 Click(And(Tag("img"), Class("icon")))Type(SameRow( Like("to")), Field("to")) Type(Tag("input"),Field("to")) DISPLAYFORM1 Demonstration: goal = {task: forward, from: Bob, to: Alice} Workflow lattice: DISPLAYFORM2 Figure 2: From each demonstration, we induce a workflow lattice based on the actions in that demonstration.

Given a new environment, the workflow policy samples a workflow (a path in the lattice, as shown in bold) and then samples actions that fit the steps of the workflow.

Our workflow policy interacts with the environment to generate an episode in the following manner.

At the beginning of the episode, the policy conditions on the provided goal g, and selects a demonstration d that carried out a similar goal: DISPLAYFORM0 (1) where sim(g, g d ) measures the similarity between g and the goal g d of demonstration d. In our tasks, we simply let sim(g, g d ) be 1 if the structured goals share the same keys, and −∞ otherwise.

Then, at each time step t with environment state s t , we sample a workflow step z t according to the following distribution: DISPLAYFORM1 where each ψ z,t,d is a separate scalar parameter to be learned.

Finally, we sample an action a t uniformly from the set z t (s t ).

DISPLAYFORM2 The overall probability of exploring an episode e = (s 1 , a 1 , . . .

, s T , a T ) is then: DISPLAYFORM3 where p(s t |s t−1 , a t−1 ) is the (unknown) state transition probability.

Note that π w (z|d, t) is not a function of the environment states s t at all.

Its decisions only depend on the selected demonstration and the current time t. This environment-blindness means that the workflow policy uses far fewer parameters than a state-dependent policy, enabling it to learn more quickly and preventing overfitting.

Due to environment-blindness, the workflow policy cannot solve the task, but it quickly learns to certain good behaviors, which can help the neural policy learn.

To train the workflow policy, we use a variant of the REINFORCE algorithm BID49 BID44 .

In particular, after rolling out an episode e = (s 1 , a 1 , . . .

, s T , a T ), we approximate the gradient using the unbiased estimate DISPLAYFORM4 where G t is the return at time step t and v d,t is a baseline term for variance reduction.

Sampled episodes from the workflow policy that receive a positive reward are stored in a replay buffer, which will be used for training the neural policy π n .

As outlined in Figure 1 , the neural policy is learned using both on-policy and off-policy updates (where episodes are drawn from the replay buffer).

Both updates use A2C, the synchronous version of the advantage actor-critic algorithm .

Since only episodes with reward +1 enter the replay buffer, the off-policy updates behave similarly to supervised learning on optimal trajectories.

Furthermore, successful episodes discovered during on-policy exploration are also added to the replay buffer.

Model architecture.

We propose DOMNET, a neural architecture that captures the spatial and hierarchical structure of the DOM tree.

As illustrated in FIG4 , the model first embeds the DOM elements and the input goal, and then applies a series of attentions on the embeddings to finally produce a distribution over actions π n (a|s) and a value function V (s), the critic.

We highlight our novel DOM embedder, and defer other details to Appendix C.We design our DOM embedder to capture the various interactions between DOM elements, similar to recent work in graph embeddings BID24 BID35 BID16 .

In particular, DOM elements that are "related" (e.g., a checkbox and its associated label) should pass their information to each other.

To embed a DOM element e, we first compute the base embedding v e base by embedding and concatenating its attributes (tag, classes, text, etc.) .

In order to capture the relationships between DOM elements, we next compute two types of neighbor embeddings:1.

We define spatial neighbors of e to be any element e within 30 pixels from e, and then sum up their base embeddings to get the spatial neighbor embedding v e spatial .

2.

We define depth-k tree neighbors of e to be any element e such that the least common ancestor of e and e in the DOM tree has depth at most k. Intuitively, tree neighbors of a higher depth are more related.

For each depth k, we apply a learnable affine transformation f on the base embedding of each depth-k tree neighbor e , and then apply max pooling to get v

We evaluate our approach on three suites of interactive web tasks:1.

MiniWoB: the MiniWoB benchmark of BID41 2.

MiniWoB++: a new set of tasks that we constructed to incorporate additional challenges not present in MiniWoB, such as stochastic environments and variation in natural language.3.

Alaska: the mobile flight booking interface for Alaska Airlines, inspired by the FormWoB benchmark of BID41 .We describe the common task settings of the MiniWoB and MiniWoB++ benchmarks, and defer the description of the Alaska benchmark to Section 6.3.3.

Environment.

Each task contains a 160px × 210px environment and a goal specified in text.

The majority of the tasks return a single sparse reward at the end of the episode; either +1 (success) or −1 (failure).

For greater consistency among tasks, we disabled all partial rewards in our experiments.

The agent has access to the environment via a Selenium web driver interface.

The public MiniWoB benchmark 1 contains 80 tasks.

We filtered for the 40 tasks that only require actions in our action space, namely clicking on DOM elements and typing strings from the input goal.

Many of the excluded tasks involve somewhat specialized reasoning, such as being able to compute the angle between two lines, or solve algebra problems.

For each task, we used Amazon Mechanical Turk to collect 10 demonstrations, which record all mouse and keyboard events along with the state of the DOM when each event occurred.

Evaluation metric.

We report success rate: the percentage of test episodes with reward +1.

Since we have removed partial rewards, success rate is a linear scaling of the average reward, and is equivalent to the definition of success rate in BID41 .

We compare the success rates across the MiniWoB tasks of the following approaches:• SHI17: the system from BID41 , pre-trained with behavioral cloning on 10 minutes of demonstrations (approximately 200 demonstrations on average) and fine-tuned with RL.

Unlike DOMNET, this system primarily uses a pixel-based representation of the state.

• DOMNET+BC+RL: our proposed neural policy, DOMNET, but pre-trained with behavioral cloning on 10 demonstrations and fine-tuned with RL, like SHI17.

During behavioral cloning, we apply early stopping based on the reward on a validation set.• DOMNET+WGE: our proposed neural policy, DOMNET, trained with workflow-guided exploration on 10 demonstrations.

For DOMNET+BC+RL and DOMNET+WGE, we report the test success rate at the time step where the success rate on a validation set reaches its maximum.

The results are shown in Figure 3 .

By comparing SHI17 with DOMNET+BC+RL, we can roughly evaluate the contribution of our new neural architecture DOMNET, since the two share the same training procedure (BC+RL).

While SHI17 also uses the DOM tree to compute text alignment features in addition to the pixel-level input, our DOMNET uses the DOM structure more explicitly.

We find DOMNET+BC+RL to empirically improve the success rate over SHI17 on most tasks.

By comparing DOMNET+BC+RL and DOMNET+WGE, we find that workflow-guided exploration enables DOMNET to perform even better on the more difficult tasks, which we analyze in the next section.

Some of the workflows that the workflow policy π w learns are shown in Appendix B.6.3 ANALYSIS

We constructed and released the MiniWoB++ benchmark of tasks to study additional challenges a web agent might encounter, including: longer time horizons (click-checkboxes-large), "soft" reasoning about natural language (click-checkboxes-soft), and stochastically varying layouts (multiorderings, multi-layouts).

TAB1 lists the tasks and their time horizons (number of steps needed for a perfect policy to carry out the longest goal) as a crude measure of task complexity.

We first compare the performance of DOMNET trained with BC+RL (baseline) and DOMNET trained with WGE (our full approach).

The proposed WGE model outperforms the BC+RL model by an average of 42% absolute success rate.

We analyzed their behaviors and noticed two common failure modes of training with BC+RL that are mitigated by instead training with WGE:1.

The BC+RL model has a tendency to take actions that prematurely terminate the episode (e.g., hitting "Submit" in click-checkboxes-large before all required boxes are checked).

One likely cause is that these actions occur across all demonstrations, while other nonterminating actions (e.g., clicking different checkboxes) vary across demonstrations.

2.

The BC+RL model occasionally gets stuck in cyclic behavior such as repeatedly checking and unchecking the same checkbox.

These failure modes stem from overfitting to parts of the demonstrations, which WGE avoids.

Next, we analyze the workflow policy π w learned by WGE.

The workflow policy π w by itself is too simplistic to work well at test time for several reasons:1.

Workflows ignore environment state and therefore cannot respond to the differences in the environment, such as the different layouts in multi-layouts.

2.

The workflow constraint language lacks the expressivity to specify certain actions, such as clicking on synonyms of a particular word in click-checkboxes-soft.

3.

The workflow policy lacks expressivity to select the correct workflow for a given goal.

Nonetheless the workflow policy π w is sufficiently constrained to discover reward some of the time, and the neural policy π n is able to learn the right behavior from such episodes.

As such, the neural policy can achieve high success rates even when the workflow policy π w performs poorly.

While MiniWoB tasks provide structured goals, we can also apply our approach to natural language goals.

We collected a training dataset using the overnight data collection technique BID47 .

In the email-inbox-nl task, we collected natural language templates by asking annotators to paraphrase the task goals (e.g., "Forward Bob's message to Alice" →

"Email Alice the email I got from Bob") and then abstracting out the fields ("Email <TO> the email I got from <FROM>").

During training, the workflow policy π w receives states with both the structured goal and the natural language utterance generated from a random template, while the neural policy π n receives only the utterance.

At test time, the neural policy is evaluated on unseen utterances.

The results in TAB1 show that the WGE model can learn to understand natural language goals (93% success rate).Note that the workflow policy needs access to the structured inputs only because our constraint language for workflow steps operates on structured inputs.

The constraint language could potentially be modified to work with utterances directly (e.g., After("to") extracts the utterance word after "to"), but we leave this for future work.

We applied our approach on the Alaska benchmark, a more realistic flight search task on the Alaska Airlines mobile site inspired by the FormWoB task in BID41 .

In this task, the agent must complete the flight search form with the provided information (6-7 fields).

We ported the web page to the MiniWoB framework with a larger 375px × 667px screen, replaced the server backend with a surrogate JavaScript function, and clamped the environment date to March 1, 2017.Following BID41 , we give partial reward based on the fraction of correct fields in the submitted form if all required fields are filled in.

Despite this partial reward, the reward is still extremely sparse: there are over 200 DOM elements (compared to ≈ 10-50 in MiniWoB tasks), and a typical episode requires at least 11 actions involving various types of widgets such as autocompletes and date pickers.

The probability that a random agent gets positive reward is less than 10 −20 .We first performed experiments on Alaska-Shi17, a clone of the original Alaska Airlines task in BID41 , where the goal always specifies a roundtrip flight (two airports and two dates).

On their dataset, our approach, using only 1 demonstration, achieves an average reward of 0.97, compared to their best result of 0.57, which uses around 80 demonstrations.

Our success motivated us to test on a more difficult version of the task which additionally requires selecting flight type (a checkbox for one-way flight), number of passengers (an increment-decrement counter), and seat type (hidden under an accordion).

We achieve an average reward of 0.86 using 10 demonstrations.

This demonstrates our method can handle long horizons on real-world websites.

To evaluate the demonstration efficiency of our approach, we compare DOMNET+WGE with DOMNET+BC+RL trained on increased numbers of demonstrations.

We compare DOM-NET+WGE trained on 10 demonstrations with DOMNET+BC+RL on 10, 100, 300, and 1000 demonstrations.

The test rewards 3 on several of the hardest tasks are summarized in FIG3 .Increasing the number of demonstrations improves the performance of BC+RL, as it helps prevent overfitting.

However, on every evaluated task, WGE trained with only 10 demonstrations still achieves much higher test reward than BC+RL with 1000 demonstrations.

This corresponds to an over 100x sample efficiency improvement of our method over behavioral cloning in terms of the number of demonstrations.

Learning agents for the web.

Previous work on learning agents for web interactions falls into two main categories.

First, simple programs may be specified by the user BID50 or may be inferred from demonstrations BID1 .

Second, soft policies may be learned from scratch or "warm-started" from demonstrations BID41 .

Notably, sparse rewards prevented BID41 from successfully learning, even when using a moderate number of demonstrations.

While policies have proven to be more difficult to learn, they have the potential to be expressive and flexible.

Our work takes a step in this direction.

Sparse rewards without prior knowledge.

Numerous works attempt to address sparse rewards without incorporating any additional prior knowledge.

Exploration methods BID32 BID11 BID48 help the agent better explore the state space to encounter more reward; shaping rewards BID31 directly modify the reward function to encourage certain behaviors; and other works BID22 augment the reward signal with additional unsupervised reward.

However, without prior knowledge, helping the agent receive additional reward is difficult in general.

Imitation learning.

Various methods have been proposed to leverage additional signals from experts.

For instance, when an expert policy is available, methods such as DAGGER BID40 and AGGREVATE BID39 BID43 can query the expert policy to augment the dataset for training the agent.

When only expert demonstrations are available, inverse reinforcement learning methods BID0 Ziebart et al., 2008; BID15 BID19 BID7 infer a reward function from the demonstrations without using reinforcement signals from the environment.

The usual method for incorporating both demonstrations and reinforcement signals is to pre-train the agent with demonstrations before applying RL.

Recent work extends this technique by (1) introducing different objective functions and regularization during pre-training, and (2) mixing demonstrations and rolled-out episodes during RL updates BID20 BID18 BID46 BID30 .Instead of training the agent on demonstrations directly, our work uses demonstrations to guide exploration.

The core idea is to explore trajectories that lie in a "neighborhood" surrounding an expert demonstration.

In our case, the neighborhood is defined by a workflow, which only permits action sequences analogous to the demonstrated actions.

Several previous works also explore neighborhoods of demonstrations via reward shaping BID10 BID21 or off-policy sampling BID26 .

One key distinction of our work is that we define neighborhoods in terms of action similarity rather than state similarity.

This distinction is particularly important for the web tasks: we can easily and intuitively describe how two actions are analogous (e.g., "they both type a username into a textbox"), while it is harder to decide if two web page states are analogous (e.g., the email inboxes of two different users will have completely different emails, but they could still be analogous, depending on the task.)Hierarchical reinforcement learning.

Hierarchical reinforcement learning (HRL) methods decompose complex tasks into simpler subtasks that are easier to learn.

Main HRL frameworks include abstract actions BID45 BID25 BID17 , abstract partial policies BID33 , and abstract states BID38 BID14 BID27 .

These frameworks require varying amounts of prior knowledge.

The original formulations required programmers to manually specify the decomposition of the complex task, while BID3 only requires supervision to identify subtasks, and BID6 ; BID12 learn the decomposition fully automatically, at the cost of performance.

Within the HRL methods, our work is closest to BID33 and the line of work on constraints in robotics BID36 BID34 .

The work in BID33 specifies partial policies, which constrain the set of possible actions at each state, similar to our workflow items.

In contrast to previous instantiations of the HAM framework BID2 BID28 , which require programmers to specify these constraints manually, our work automatically induces constraints from user demonstrations, which do not require special skills to provide.

BID36 ; Perez-D'Arpino & Shah (2017) also resemble our work, in learning constraints from demonstrations, but differ in the way they use the demonstrations.

Whereas our work uses the learned constraints for exploration, BID36 only uses the constraints for planning and Perez-D'Arpino & Shah (2017) build a knowledge base of constraints to use at test time.

Summary.

Our workflow-guided framework represents a judicious combination of demonstrations, abstractions, and expressive neural policies.

We leverage the targeted information of demonstrations and the inductive bias of workflows.

But this is only used for exploration, protecting the expressive neural policy from overfitting.

As a result, we are able to learn rather complex policies from a very sparse reward signal and very few demonstrations.

Acknowledgments.

This work was supported by NSF CAREER Award IIS-1552635.

We try to keep the constraint language as minimal and general as possible.

The main part of the language is the object selector (elementSet) which selects either (1) objects that share a specified property, or (2) objects that align spatially.

These two types of constraints should be applicable in many typical RL domains such as game playing and robot navigation.

To avoid combinatorial explosion of relatively useless constraints, we limit the number of nested elementSet applications to 3, where the third application must be the Class filter.

When we induce workflow steps from a demonstration, the valid literal values for tag, string, and classes are extracted from the demonstration state.

login-user Enter the username "ashlea" and password "k0UQp" and press login.{username: ashlea, password: k0UQp} Type(Tag("input_text"),Field("username")) Type(Tag("input_password"),Field("password")) Click(Like("Login"))email-inbox Find the email by Ilka and forward it to Krista.

{task: forward, name:

Ilka, to: Krista} Click(Near(Field("by"))) Click(SameCol(Like("Forward"))) Type(And(Near("Subject"),Class("forward-sender")),Field("to")) Click(Tag("span"))

<|TLDR|>

@highlight

We solve the sparse rewards problem on web UI tasks using exploration guided by demonstrations