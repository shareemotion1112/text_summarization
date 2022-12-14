Neural programs are highly accurate and structured policies that perform algorithmic tasks by controlling the behavior of a computation mechanism.

Despite the potential to increase the interpretability and the compositionality of the behavior of artificial agents, it remains difficult to learn from demonstrations neural networks that represent computer programs.

The main challenges that set algorithmic domains apart from other imitation learning domains are the need for high accuracy, the involvement of specific structures of data, and the extremely limited observability.

To address these challenges, we propose to model programs as Parametrized Hierarchical Procedures (PHPs).

A PHP is a sequence of conditional operations, using a program counter along with the observation to select between taking an elementary action, invoking another PHP as a sub-procedure, and returning to the caller.

We develop an algorithm for training PHPs from a set of supervisor demonstrations, only some of which are annotated with the internal call structure, and apply it to efficient level-wise training of multi-level PHPs.

We show in two benchmarks, NanoCraft and long-hand addition, that PHPs can learn neural programs more accurately from smaller amounts of both annotated and unannotated demonstrations.

Representing the logic of a computer program with a parametrized model, such as a neural network, is a central challenge in AI with applications including reinforcement learning, robotics, natural language processing, and programming by example.

A salient feature of recently-proposed approaches for learning programs BID32 BID6 is their ability to leverage the hierarchical structure of procedure invocations present in well-designed programs.

Explicitly exposing this hierarchical structure enables learning neural programs with empirically superior generalization, compared to baseline methods that learn only from elementary computer operations, but requires training data that does not consists only of low-level computer operations but is annotated with the higher-level procedure calls BID32 BID6 . tackled the problem of learning hierarchical neural programs from a mixture of annotated training data (hereafter called strong supervision) and unannotated training data where only the elementary operations are given without their call-stack annotations (called weak supervision).

In this paper, we propose to learn hierarchical neural programs from a mixture of strongly supervised and weakly supervised data via the Expectation-Gradient method and an explicit program counter, in lieu of a high-dimensional real-valued state of a recurrent neural network.

Our approach is inspired by recent work in robot learning and control.

In Imitation Learning (IL), an agent learns to behave in its environment using supervisor demonstrations of the intended behavior.

However, existing approaches to IL are largely insufficient for addressing algorithmic domains, in which the target policy is program-like in its accurate and structured manipulation of inputs and data structures.

An example of such a domain is long-hand addition, where the computer loops over the digits to be added, from least to most significant, calculating the sum and carry.

In more complicated examples, the agent must correctly manipulate data structures to compute the right output.

Three main challenges set algorithmic domains apart from other IL domains.

First, the agent's policy must be highly accurate.

Algorithmic behavior is characterized by a hard constraint of output correctness, where any suboptimal actions are simply wrong and considered failures.

In contrast, many tasks in physical and simulated domains tolerate errors in the agent's actions, as long as some goal region in state-space is eventually reached, or some safety constraints are satisfied.

A second challenge is that algorithms often use specific data structures, which may require the algorithmic policies to have a particular structure.

A third challenge is that the environment in algorithmic domains, which consists of the program input and the data structures, is almost completely unobservable directly by the agent.

They can only be scanned using some limited reading apparatus, such as the read/write heads in a Turing Machine or the registers in a register machine.

Recently proposed methods can infer from demonstration data hierarchical control policies, where high-level behaviors are composed of low-level manipulation primitives BID8 .

In this paper, we take a similar approach to address the challenges of algorithmic domains, by introducing Parametrized Hierarchical Procedures (PHPs), a structured model of algorithmic policies inspired by the options framework BID38 , as well as the procedural programming paradigm.

A PHP is a sequence of statements, such that each statement branches conditionally on the observation, to either (1) perform an elementary operation, (2) invoke another PHP as a sub-procedure, or (3) terminate and return control to the caller PHP.

The index of each statement in the sequence serves as a program counter to accurately remember which statement was last executed and which one is next.

The conditional branching in each statement is implemented by a neural network mapping the program counter and the agent's observation into the elementary operation, sub-procedure, or termination to be executed.

The PHP model is detailed in Section 4.1.PHPs have the potential to address the challenges of algorithmic domains by strictly maintaining two internal structures: a call stack containing the current branch of caller PHPs, and the current program counter of each PHP in the stack.

When a statement invokes a PHP as a sub-procedure, this PHP is pushed into the call stack.

When a statement terminates the current PHP, it is popped from the stack, returning control to the calling PHP to execute its next statement (or, in the case of the root PHP, ending the entire episode).

The stack also keeps the program counter of each PHP, which starts at 0, and is incremented each time a non-terminating statement is executed.

PHPs impose a constraining structure on the learned policies.

The call stack arranges the policy into a hierarchical structure, where a higher-level PHP can solve a task by invoking lower-level PHPs that solve sub-tasks.

Since call stacks and program counters are widely useful in computer programs, they provide a strong inductive bias towards policy correctness in domains that conform to these constraints, while also being computationally tractable to learn.

To support a larger variety of algorithmic domains, PHPs should be extended in future work to more expressive structures, for example allowing procedures to take arguments.

We experiment with PHPs in two benchmarks, the NanoCraft domain introduced in , and long-hand addition.

We find that our algorithm is able to learn PHPs from a mixture of strongly and weakly supervised demonstrations with better sample complexity than previous algorithms: it achieves better test performance with fewer demonstrations.

In this paper we make three main contributions:??? We introduce the PHP model and show that it is easier to learn than the NPI model BID32 ).???

We propose an Expectation-Gradient algorithm for efficiently training PHPs from a mixture of annotated and unannotated demonstrations (strong and weak supervision).??? We demonstrate efficient training of multi-level PHPs on NanoCraft and long-hand addition BID32 , and achieve improved success rate.2 RELATED WORK BID32 Recursive NPI BID6 (recursive) NPL Mixed PHP (this work) Mixed BID18 , the Neural GPU BID19 , and End-to-End Memory Networks BID37 , have been proposed for learning neural programs from input-output examples, with components such as variable-sized memory and novel addressing mechanisms facilitating the training process.

In contrast, our work considers the setting where, along with the input-output examples, execution traces are available which describe the steps necessary to solve a given problem.

The Neural Programmer-Interpreter (NPI, BID32 ) learns hierarchical policies from execution traces which not only indicate the low-level actions to perform, but also a structure over them specified by higher-level abstractions.

BID6 showed that learning from an execution trace with recursive structure enables perfect generalization.

Neural Program Lattices work within the same setting as the NPI, but can learn from a dataset of execution traces where only a small fraction contains information about the higher-level hierarchy.

In demonstrations where the hierarchical structure along the trace is missing, this latent space grows exponentially in the trace length.

address this challenge via an approximation method that selectively averages latent variables on different computation paths to reduce the complexity of enumerating all paths.

In contrast, we compute exact gradients using dynamic programming, by considering a hierarchical structure that has small discrete latent variables in each time step.

Other works use neural networks as a tool for outputting programs written in a discrete programming language, rather than having the neural network itself represent a program.

BID3 learned to generate programs for solving competition-style problems.

BID9 and BID31 generate programs in a domain-specific language for manipulating strings in spreadsheets.

Automatic discovery of hierarchical structure has been well-studied, and successful approaches include action-sequence compression BID39 , identifying important transitional states BID28 BID29 ??im??ek & Barto, 2004; BID36 BID25 , learning from demonstrations BID5 BID22 BID7 BID23 , considering the set of initial states from which the MDP can be solved BID20 BID21 , policy gradients BID26 , information-theoretic considerations BID13 BID11 BID17 BID10 , active learning BID15 , and recently value-function approximation BID2 BID16 BID34 .Our approach is inspired by the Discovery of Deep Options (DDO) algorithm of .

Following the work of BID8 , who use Expectation-Maximization (EM) to train an Abstract Hidden Markov Model BID5 , DDO parametrizes the model with neural networks where complete maximization in the M-step is infeasible.

Instead, DDO uses Expectation-Gradient (EG) to take a single gradient step using the same forward-backward E-step as in the EM algorithm.

A variant of DDO for continuous action spaces (DDCO) has shown success in simulated and physical robot control .

This paper extends DDO by proposing an E-step that can infer a call-stack of procedures and their program counters.

Computation can be modeled as a deterministic dynamical system, where the computer is an agent interacting with its environment, which consists of the program input and its data structures.

Mathematically, the environment is a Deterministic Partially Observable Markov Decision Process (DET-POMDP BID4 ), which consists of a state space S, an observation space O, an action space A, the state-dependent observation o t ps t q, and the state transition s t`1 " f ps t , a t q. The initial state s 0 includes the program input, and is generated by some distribution p 0 ps 0 q.

This notation is general enough to model various computation processes.

In a Turing Machine, for example, s t is the machine's configuration, o t is the vector of tape symbols under the read/write heads, and a t contains writing and moving instructions for the heads.

In partially observable environments, the agent often benefits from maintaining memory m t of past observations, which reveals temporarily hidden aspects of the current state.

The agent has a parametrized stochastic policy ?? ?? , in some parametric family ?? P ??, where ?? ?? pm t , a t |m t??1 , o t q is the probability of updating the memory state from m t??1 to m t and taking action a t , when the observation is o t .

The policy can be rolled out to induce the stochastic process ps 0:T , o 0:T , m 0:T??1 , a 0:T??1 q, such that upon observing o T the agent chooses to terminate the process.

In a computation device, the memory m t stands for its internal state, such as the Finite State Machine of a Turing Machine.

We can scale computer programs by adding data structures to their internal state, such as a call stack, which we model in the next section as Parametrized Hierarchical Procedures.

In Imitation Learning (IL), the learner is provided with direct supervision of the correct actions to take.

The setting we use is Behavior Cloning (BC), where the supervisor rolls out its policy to generate a batch of demonstrations before learning begins, and the agent's policy is trained to minimize a loss on its own selection of actions in demonstrated states, with respect to the demonstrated actions.

In strong supervision, a demonstration contains not only the sequence of observable variables ?? " po 0:T , a 0:T??1 q, where a 0:T??1 is the sequence of supervisor actions during the demonstration, but also the sequence of the supervisor's memory states ?? " m 0:T??1 , which are ordinarily latent.

This allows the agent to directly imitate not just the actions, but also the memory updates of the supervisor, for example by maximizing the log-likelihood of the policy given the demonstrations arg max DISPLAYFORM0 the latter being the negative cross-entropy loss with respect to the demonstrations.

In weak supervision, on the other hand, only the observable trajectories ?? are given as demonstrations.

This makes it difficult to maximize the likelihood Pp??|??q " ?? ?? Pp??, ??|??q, due to the large space of possible memory trajectories ??.

We address this difficulty via the Expectation-Gradient algorithm described in Section 4.

The semantics of this definition are given by the following control policy.

The agent's memory maintains a stack rph 1 , ?? 1 q, . . .

, ph n , ?? n qs of the active procedures and their program counters.

Initially, this stack contains only the root procedure and the counter is 0.

Upon observing o t , the agent checks whether the top procedure should terminate, i.e. ?? ??n hn po t q " 1.

If the procedure h n terminates, it is popped from the stack, the next termination condition ?? ??n??1 hn??1 po t q is consulted, and so on.

For the first procedure h i that does not terminate, we select the operation ?? ??i`1 hi po t q, after incrementing the program counter ?? i .

If this operation is an invocation of procedure h 1 i`1 , we push ph 1 i`1 , 0q onto the stack, consult its operation statement ?? 0 h 1 i`1 po t q, and so on.

Upon the first procedure h 1 n 1 to select an elementary action a t , we save the new memory state m t " rph 1 , ?? 1 q, . . .

, ph i??1 , ?? i??1 q, ph i , ?? i`1 q, ph 1 i`1 , 0q, . . .

, ph 1 n 1 , 0qs, and take the action a t in the environment.

DISPLAYFORM1 The call stack and program counters act as memory for the agent, so that it can remember certain hidden aspects of the state that were observed before.

In principle, any finite memory structure can be implemented with sufficiently many PHPs, by having a distinct procedure for each memory state.

However, PHPs leverage the call stack and program counters to allow exponentially many memory states to be expressed with a relatively small set of PHPs.

We impose two practical limitations on the general definition of PHPs.

Our training algorithm in Section 4.2 does not support recursive procedures, i.e. cycles in the invocation graph.

In addition, for simplicity, we allow each procedure to either invoke other procedures or execute elementary actions, not both.

These two limitations are achieved by layering the procedures in levels, such that only the lowest-level procedures can execute elementary actions, and each higher-level procedure can only invoke procedures in the level directly below it.

This does not lose generality, since instead of invoking a procedure or action at a certain level, we can wrap it in a one-level-higher surrogate procedure that invokes it and terminates.

A Parametrized Hierarchical Procedure (PHP) is a representation of a hierarchical procedure by differentiable parametrization.

In this paper, we represent each PHP by two multi-layer perceptrons (MLPs) with ReLU activations, one for the PHP's operation statement and one for its termination statement.

The input is a concatenation of the observation o and the program counter ?? , where ?? is provided to the MLPs as a real number.

During training, we apply the soft-argmax activation function to the output of each MLP to obtain stochastic statements ?? ?? h p??|o t q and ?? ?? h p??|o t q. During testing, we replace the soft-argmax with argmax, to obtain deterministic statements as above.

In weak supervision, only the observable trajectory ?? " po 0:T , a 0:T??1 q is available in a demonstration, and the sequence of memory states ?? " m 0:T??1 is latent.

This poses a challenge, since the space of possible memory trajectories ?? grows exponentially in the length of the demonstration, which at first seems to prohibit the computation of the log-likelihood gradient ??? ?? log Pp??|?? ?? q, needed to maximize the log-likelihood via gradient ascent.

We use the Expectation-Gradient (EG) method to overcome this challenge BID33 .

This method has been previously used in dynamical settings to play Atari games and to control simulated and physical robots .

The EG trick expresses the gradient of the observable log-likelihood as the expected gradient of the full log-likelihood: DISPLAYFORM0 where the first and third equations follow from two applications of the identity ??? ?? x " x??? ?? log x.

In the E-step of the EG algorithm, we find the posterior distribution of ?? given the observed ?? and the current parameter ??.

In the G-step, we use this posterior to calculate and apply the exact gradient of the observable log-likelihood.

We start by assuming a shallow hierarchy, where the root PHP calls level-one PHPs that only perform elementary operations.

At any time t, the stack contains two PHPs, the root PHP and the PHP it invoked to select the elementary action.

The stack also contains the program counters of these two PHPs, however we ignore the root counter to reduce complexity, and bring it back when we discuss multi-level hierarchies in the next section.

Let us denote by ?? ?? h pa t |o t q and ?? ?? h pb t |o t q, respectively, the stochastic operation and termination statements of procedure h P H Y tKu, where K is the root PHP.

Let ph t , ?? t q be the top stack frame when action a t is selected.

Then the full likelihood Pp??, ??|??q of the policy given an annotated demonstration is a product of the terms that generate the demonstration, including ?? ??t ht pa t |o t q for the generation of each a t , as well as ?? ??t??1 ht??1 p1|o t q?? K ph t |o t q whenever h t??1 terminates and h t is pushed with ?? t " 0, and ?? ??t??1 ht??1 p0|o t q whenever h t??1 does not terminate (i.e. h t " h t??1 and ?? t " ?? t??1`1 ).

Crucially, the form of Pp??, ??|??q as a product implies that ??? ?? log Pp??, ??|??q decomposes into a sum of policy-gradient terms such as ??? ?? log ?? ??t ht pa t |o t q, and computing its expectation over Pp??|??, ??q only requires the marginal posterior distributions over single-step latent variables v t ph, ?? q " Pph t "h, ?? t "?? |??, ??q w t ph, ?? q " Pph t "h, ?? t "??, ?? t`1 "??`1|??, ??q.

The marginal posteriors v t and w t can be found via a forward-backward algorithm, as described in Appendix A, and used to compute the exact gradient DISPLAYFORM0 t ph, ?? q??? ?? log ?? ?? h pa t |o t q w t ph, ?? q??? ?? log ?? ?? h p0|o t`1 qq pv t ph, ?? q??w t ph, ?? qq??? ?? log ?? ?? h p1|o t`1 q????.

A naive attempt to generalize the same approach to multi-level PHPs would result in an exponential blow-up of the forward-backward state, which would need to include the entire stack.

Instead, we train each level separately, iterating over the PHP hierarchy from the lowest level to the highest.

Let us denote by d the number of levels in the hierarchy, with 0 being the root and d??1 the lowest level, then we train level i in the hierarchy after we have trained levels i`1, . . .

, d??1.Two components are required to allow this separation.

First, we need to use our trained levels i`1, . . .

, d??1 to abstract away from the elementary actions, and generate demonstrations where the level-pi`1q PHPs are treated as the new elementary operations.

In this way, we can view level-i PHPs as the new lowest-level PHPs, whose operations are elementary in the demonstrations.

This is easy to do in strongly supervised demonstrations, since we have the complete stack, and we only need to truncate the lowest d??i??1 levels.

In weakly supervised demonstrations, on the other hand, we need an algorithm for decoding the observable trajectories, and replacing the elementary actions with higher-level operations.

We present such an algorithm below.

The second component needed for level-wise training is approximate separation from higher levels that have not been trained yet.

When we train level i ?? 1 via the EG algorithm in the previous section, the "root PHP" would be at level i??1, had it corresponded to any real PHP.

In all but the simplest domains, we cannot expect a single PHP to perfectly match the behavior of the i-levels PHP hierarchy (levels 0, . . . , i??1) that actually selected the level-i PHPs that generated the demonstrations.

To facilitate better separation from higher levels, we augment the "root PHP" used for training with an LSTM that approximates the i-levels stack memory as ?? LST M K ph t |o 0 , . . .

, o t q.

As mentioned above, abstraction from lower levels is achieved by rewriting weakly supervised demonstrations to show level-pi`1q operations as elementary.

After level i`1 is trained, the levelpi`1q PHPs that generated the demonstrations are decoded using the trained parameters.

We considered three different decoding algorithms: (1) finding the most likely level-pi`1q PHP at each time step, by taking the maximum over v t ; (2) using a Viterbi-like algorithm to find the most likely latent trajectory of level-pi`1q PHPs; (3) sampling from the posterior distribution Pp??|??, ??q over latent trajectories.

In our current experiments we used latent trajectories sampled from the posterior distribution, given by DISPLAYFORM0 where the denominators can be computed via the same forward-backward algorithm used in the previous section to compute v t and w t , as detailed in Appendix A.

We evaluate our proposed method on the two settings studied by : NanoCraft, which involves an agent interacting in a grid world, and long-hand addition, which was also considered by BID32 and BID6 .

Task description.

The NanoCraft domain, introduced by , involves placing blocks in a two-dimensional grid world.

The goal of the task is to control an agent to build a rectangular building of a particular height and width, at a specified location within the grid, by moving around the grid and placing blocks in appropriate cells.

The state contains a 6??6 grid.

In our version, each grid cell can either be empty or contain a block.

The state also includes the current location of the agent, as well as the building's desired height, width, and location, expressed as the offset from the agent's initial location at the top left corner.

Initially, some of the blocks are already in place and must not be placed again.

The state-dependent observation o t ps t q reveals whether the grid cell at which the agent is located contains a block or not, and four numbers for the building's specifications.

We provide each observation to the MLPs as a 5-dimensional real-valued feature vector.

PHPs and elementary actions.

The top-level PHP nanocraft executes (moves_r, moves_d, builds_r, builds_d, builds_l, builds_u, return) .

moves_r calls move_r a number of times equal to the building's horizontal location, and similarly for moves_d w.r.t.

move_d and the vertical location; builds_r w.r.t.

build_r and the building's width; and so on for builds_d, builds_l, and builds_u.

At the lowest level, move_r takes the elementary action MOVE_RIGHT and terminates, and similarly for move_d taking MOVE_DOWN.

build_r executes (MOVE_RIGHT, if cell full: return, else: PLACE_BLOCK, return), and similarly for build_d, build_l, and build_u w.r.t.

MOVE_DOWN, MOVE_LEFT, and MOVE_UP.Experiment setup.

We trained our model on datasets of 16, 32, and 64 demonstrations, of which some are strongly supervised and the rest weakly supervised.

We trained each level for 2000 iterations, iteratively from the lowest level to the highest.

The results are averaged over 5 trials with independent datasets.

Results.

Our results summarized in FIG2 show that 32 strongly supervised demonstrations are sufficient for achieving perfect performance at the task, and that 16 such demonstrations approach the same success rate when used along with weakly supervised demonstrations, for a total of 16, 32, or 64 demonstrations.

An interesting question is whether these performance gains are due to the simplicity of the PHP model itself, the use of exact gradients in its optimization via the EG algorithm, or both.

The PHP and NPL/NPI experiments with 64 strongly supervised demonstrations FIG2 , blue curves at the 64 mark) directly compare the PHP model with the NPI model, since both algorithms use exact gradients in this case.

1 The accuracy is 1.0 for PHP; 0.724 for NPL/NPI, suggesting that the gains of PHP are at least in part due to the PHP model inducing an optimization landscape in which a good solution is easier to find.

In the experiments with 16 strongly supervised demonstrations of a total 64 (blue curves at the 16 mark), the success rate is 0.969 for PHP; 0.502 for NPL.

This 70% increase in the gain of PHP over NPL may be evidence that exact gradients are better at training the model than the approximate gradients of NPL, although the choice of an optimization method is conflated here with the choice of a model.

Task description.

The long-hand addition task was also considered by BID32 , , and BID6 .

In this task, our goal is to add two numbers represented in decimal, by starting at the rightmost column (least significant digit) and repeatedly summing each column to write the resulting digit and a carry if needed.

The state consists of 4 tapes, as in a Turing Machine, corresponding to the first number, the second number, the carries, and the output.

The state also includes the positions of 4 read/write heads, one for each tape.

Initially, each of the first two tapes contains the K digits of a number to be added, all other cells contain the empty symbol, and the heads point to the least significant digits.

The state-dependent observation o t ps t q reveals the value of the digits (or empty symbols) pointed to by the pointers.

The four values are provided to the MLPs in one-hot encoding, i.e., the input vector has 11??4 dimensions with exactly one 1-valued entry in each of the four group.

PHPs and elementary actions.

The top-level PHP add repeatedly calls add1 to add each column of digits.

add1 calls write, carry, and lshift in order to compute the sum of the column, write the carry in the next column, and move the pointers to the next column.

If the sum for a column is less than 10, then add1 does not call carry.

There are two kinds of elementary actions: one which moves a specified pointer in a specified direction (e.g. MOVE CARRY LEFT), and one which writes a specified digit to a specified tape (e.g. WRITE OUT 2).

?? write , ?? carry , and ?? lshift output the probability distribution over possible action and argument combinations as the product of 3 multinomial distributions, each with 2, 4, and 10 possibilities respectively.

Experiment setup.

Following , we trained our model on execution traces for inputs of each length 1 to 10.

We used 16 traces for each input length, for a total of 160 traces.

2 We experimented with providing 1, 2, 3, 5, and 10 strongly supervised traces, with the remainder containing only the elementary actions.

For training our model, we performed a search over two hyperparameters:??? Weight on loss from strongly supervised traces: When the number of weakly supervised demonstrations overwhelms the number of strongly supervised traces, the model can learn a hierarchy which does not match the supervisor.

By appropriately scaling up the loss contribution from the strongly supervised traces, we can ensure that the model learns to follow the hierarchy specified in them.??? Use of ?? in ??: The termination condition ?? ?? h pb t |o t q contains a dependence on ?? , the number of steps that the current procedure h has executed.

However, sometimes the underlying definition for ?? does not contain any dependence on ?? : ?? 1 h pb|oq " ?? 2 h pb|oq "??????. In such a case, the MLP for ?? h may learn a spurious dependency on ?? , and generalize poorly to values of ?? seen during test time.

Therefore, we searched over whether to use ?? for ?? at each level of the hierarchy.

Results.

Our results are summarized in Table 2 .

Previous work by learns a model which can generalize perfectly to input lengths of 500 but not 1000.

In our experiments with the same Accuracy for input length

Strongly supervised / total traces 500 1000NPI BID32 3 160 / 160 <100% <100% NPL 3 10 / 160 100% <100% PHP 3 / 160 100% 100% Table 2 : Empirical results for the long-hand addition task.

All models were trained with 16 traces per input length between 1 and 10, for a total of 160 traces, some of which strongly supervised.sample complexity, EG can train PHPs which generalize to length-1000 inputs with 100% empirical test accuracy.

Moreover, we successfully learn models with as few as 3 strongly supervised demonstrations, compared to the 10 used by .

However, we found that when the number of strongly supervised demonstrations was smaller than 10, early stopping of the training of the top-level policy was needed to learn a correct model.

To obtain our reported results, we evaluated different snapshots of the model generated dur reporteding training.

In this paper we introduced the Parametrized Hierarchical Procedures (PHP) model for hierarchical representation of neural programs.

We proposed an Expectation-Gradient algorithm for training PHPs from a mixture of strongly and weakly supervised demonstrations of an algorithmic behavior, showed how to perform level-wise training of multi-level PHPs, and demonstrated the benefits of our approach on two benchmarks.

PHPs alleviate the sample complexity required to train policies with unstructured memory architectures, such as LSTMs, by imposing the structure of a call stack augmented with program counters.

This structure may be limiting in that it requires the agent to also rely on observable information that could otherwise be memorized, such as the building specifications in the NanoCraft domain.

The benchmarks used so far in the field of neural programming are simple enough and observable enough to be solvable by PHPs, however we note that more complicated and less observable domains may require more expressive memory structures, such as passing arguments to sub-procedures.

Future work will explore such structures, as well as new benchmarks to further challenge the community.

Our results suggest that adding weakly supervised demonstrations to the training set can improve performance at the task, but only when the strongly supervised demonstrations already get decent performance.

Weak supervision could attract the optimization process to a different hierarchical structure than intended by the supervisor, and in such cases we found it necessary to limit the number of weakly supervised demonstrations, or weight them less than demonstrations annotated with the intended hierarchy.

An open question is whether the attractors strengthened by weak supervision are alternative but usable hierarchical structures, that are as accurate and interpretable as the supervisor's.

Future work will explore the quality of solutions obtained by training from only weakly supervised demonstrations.

In weak supervision, only the observable trajectory ?? " po 0:T , a 0:T??1 q is available in a demonstration, and the sequence of memory states ?? " m 0:T??1 is latent.

This poses a challenge, since the space of possible memory trajectories ?? grows exponentially in the length of the demonstration, which at first seems to prohibit the computation of the log-likelihood gradient ??? ?? log Pp??|?? ?? q, needed to maximize the log-likelihood via gradient ascent.

Our key insight is that the log-likelihood gradient can be computed precisely and efficiently using an instance of the Expectation-Gradient (EG) method BID33 , which we detail below: DISPLAYFORM0 where the first and third equations follow from the identity ??? ?? x " x??? ?? log x.

We start by assuming two-level PHPs, so that at any time t the stack contains the root PHP and the PHP it invoked to select the elementary action.

The stack also contains the program counters of these two PHPs, however we ignore the root counter to reduce complexity, and bring it back when we discuss multi-level hierarchies in Section 4.2.3 (and below).Let us denote by ?? ?? h pa t |o t q and ?? ?? h pb t |o t q, respectively, the stochastic operation and termination statements of procedure h P H Y tKu, where K is the root PHP.

Let ph t , ?? t q be the top stack frame when action a t is selected.

Then the full likelihood Pp??, ??|??q of the policy given an annotated demonstration is Pp??, ??|??q 9 ?? K ph 0 |o 0 q?? ??0"0 DISPLAYFORM1 where from the right-hand side we omitted the constant causal dynamics factor DISPLAYFORM2 Ppo t |o 0:t??1 , a 0:t??1 q, and with Pph t , ?? t |h t??1 , ?? t??1 , o t q " # ?? ??t??1 ht??1 p1|o t q?? K ph t |o t q if ?? t " 0 ?? ??t??1 ht??1 p0|o t q?? ht"ht??1 if ?? t " ?? t??1`1 .This formulation of the likelihood has the extremely useful property that ??? ?? log Pp??, ??|??q decomposes into a sum of gradients.

To find the expected gradient, as in (1), we do not need to represent the entire posterior distribution Pp??|??, ??q, which would be intractable.

Instead, we only need the marginal posteriors that correspond to the various terms, namely v t ph, ?? q " Pph t "h, ?? t "?? |??, ??q w t ph, ?? q " Pph t "h, ?? t "??, ?? t`1 "??`1|??, ??q.

With these, the EG trick gives us the gradient of the observable demonstration DISPLAYFORM3 t ph, 0q??? ?? log ?? K ph|o t q t ?? ?? "0??v t ph, ?? q??? ?? log ?? ?? h pa t |o t q w t ph, ?? q??? ?? log ?? ?? h p0|o t`1 qq pv t ph, ?? q??w t ph, ?? qq??? ?? log ?? ?? h p1|o t`1 q????.To allow the G-step (2), we take an E-step that calculates the marginal posteriors v and w with a forward-backward pass.

We first compute the likelihood of a trajectory prefix ?? t ph, ?? q 9 Ppo 0:t , a 0:t , h t "h, ?? t "?? q, up to the causal dynamics factor, via the forward recursion given by ?? 0 ph, 0q " ?? K ph|o 0 q, and for 0 ?? t ?? T??1 ?? t`1 ph 1 , 0q "???? hPH,0???? ??t ?? t ph, ?? q?? ?? h pa t |o t q?? ?? h p1|o t`1 q???? K ph 1 |o t`1 q ?? t`1 ph, ??`1q " ?? t ph, ?? q?? ?? h pa t |o t q?? ?? h p0|o t`1 qq.

We similarly compute the likelihood of a trajectory suffix ?? t ph, ?? q 9 Ppa t:T??1 , o t`1:T |o 0:t , h t "h, ?? t "?? q, via the backward recursion given by ?? T??1 ph, ?? q " ?? ?? h pa T??1 |o T??1 q?? ?? h p1|o T q, and for 0 ?? t ?? T??1 ?? t ph, ?? q " ?? ?? h pa t |o t q???? ?? h p1|o t`1 q ?? h 1 PH ?? K ph 1 |o t`1 q?? t`1 ph 1 , 0q`?? ?? h p0|o t`1 qq?? t`1 ph, ??`1q??.For efficiency considerations, note that this forward-backward graph has pt`1qk nodes in layer t, where k " |H|, but only pt`1qkpk`1q edges to the next layer, rather than the naive pt`1qpt`2qk 2 .We can calculate our target likelihood using any 0 ?? t ?? T , by takingPp??|??q " ?? hPH,0???? ??t Pp??, h t "h, ?? t "?? q 9 ?? hPH,0???? ??t ?? t ph, ?? q?? t ph, ?? q, so most efficient is to use t " 0 Pp??|??q " DISPLAYFORM4 Pp??, h 0 "h, ?? 0 "0q 9 ?? hPH ?? 0 ph, 0q?? 0 ph, 0q.

Finally, the marginal posteriors are given by v t ph, ?? q " 1 Pp??|??q ?? t ph, ?? q?? t p??, hq w T??1 ph, ?? q " 0, and for 0 ?? t ?? T??1 w t ph, ?? q " 1 Pp??|??q ?? t ph, ?? q?? ?? h pa t |o t q?? ?? h p0|o t`1 qq?? t`1 ph, ??`1q.

As mentioned in Section 4.2.3, level-wise training of multi-level PHPs requires abstraction from lower levels and separation from higher levels.

The former is achieved by rewriting weakly supervised demonstrations to show level-i operations as elementary, for the purpose of training the next-higher level i??1.After level i is trained, the level-i PHPs that generated the demonstrations are decoded using the trained parameters.

In our current experiments we used latent trajectories sampled from the posterior distribution, given by Pp??|??, ??q " v 0 ph 0 , ?? 0 q T??2 ?? t"0 Pph t , ?? t , h t`1 , ?? t`1 |??, ??q v t ph t , ?? t q ,where for each step 0 ?? t ?? T??1Pph t , ?? t , h t`1 , 0|??, ??q " 1 Pp??|??q ?? t ph t , ?? t q?? ??t ht pa t |o t q?? ??t ht po t`1 q?? K ph t`1 |o t`1 q?? t`1 ph t`1 , 0qPph t , ?? t , h t`1 , ?? t`1 |??, ??q " ?? ht`1"ht w t ph t , ?? t q.

<|TLDR|>

@highlight

We introduce the PHP model for hierarchical representation of neural programs, and an algorithm for learning PHPs from a mixture of strong and weak supervision.