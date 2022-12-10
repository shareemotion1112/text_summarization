Neural networks powered with external memory simulate computer behaviors.

These models, which use the memory to store data for a neural controller, can learn algorithms and other complex tasks.

In this paper, we introduce a new memory to store weights for the controller, analogous to the stored-program memory in modern computer architectures.

The proposed model, dubbed Neural Stored-program Memory, augments current memory-augmented neural networks, creating differentiable machines that can switch programs through time, adapt to variable contexts and thus fully resemble the Universal Turing Machine.

A wide range of experiments demonstrate that the resulting machines not only excel in classical algorithmic problems, but also have potential for compositional, continual, few-shot learning and question-answering tasks.

Recurrent Neural Networks (RNNs) are Turing-complete (Siegelmann & Sontag, 1995) .

However, in practice RNNs struggle to learn simple procedures as they lack explicit memory (Graves et al., 2014; Mozer & Das, 1993) .

These findings have sparked a new research direction called Memory Augmented Neural Networks (MANNs) that emulate modern computer behavior by detaching memorization from computation via memory and controller network, respectively.

MANNs have demonstrated significant improvements over memory-less RNNs in various sequential learning tasks Le et al., 2018a; Sukhbaatar et al., 2015) .

Nonetheless, MANNs have barely simulated general-purpose computers.

Current MANNs miss a key concept in computer design: stored-program memory.

The concept has emerged from the idea of Universal Turing Machine (UTM) (Turing, 1936) and further developed in Harvard Architecture (Broesch, 2009 ), Von Neumann Architecture (von Neumann, 1993 .

In UTM, both data and programs that manipulate the data are stored in memory.

A control unit then reads the programs from the memory and executes them with the data.

This mechanism allows flexibility to perform universal computations.

Unfortunately, current MANNs such as Neural Turing Machine (NTM) (Graves et al., 2014) , Differentiable Neural Computer (DNC) and Least Recently Used Access (LRUA) (Santoro et al., 2016) only support memory for data and embed a single program into the controller network, which goes against the stored-program memory principle.

Our goal is to advance a step further towards UTM by coupling a MANN with an external program memory.

The program memory co-exists with the data memory in the MANN, providing more flexibility, reuseability and modularity in learning complicated tasks.

The program memory stores the weights of the MANN's controller network, which are retrieved quickly via a key-value attention mechanism across timesteps yet updated slowly via backpropagation.

By introducing a meta network to moderate the operations of the program memory, our model, henceforth referred to as Neural Stored-program Memory (NSM), can learn to switch the programs/weights in the controller network appropriately, adapting to different functionalities aligning with different parts of a sequential task, or different tasks in continual and few-shot learning.

To validate our proposal, the NTM armed with NSM, namely Neural Universal Turing Machine (NUTM), is tested on a variety of synthetic tasks including algorithmic tasks from Graves et al. (2014) , composition of algorithmic tasks and continual procedure learning.

For these algorithmic problems, we demonstrate clear improvements of NUTM over NTM.

Further, we investigate NUTM in few-shot learning by using LRUA as the MANN and achieve notably better results.

Finally, we expand NUTM application to linguistic problems by equipping NUTM with DNC core and achieve competitive performances against stateof-the-arts in the bAbI task .

Taken together, our study advances neural network simulation of Turing Machines to neural architecture for Universal Turing Machines.

This develops a new class of MANNs that can store and query both the weights and data of their own controllers, thereby following the stored-program principle.

A set of five diverse experiments demonstrate the computational universality of the approach.

In this section, we briefly review MANN and its relations to Turing Machines.

A MANN consists of a controller network and an external memory M ∈ R N ×M , which is a collection of N M -dimensional vectors.

The controller network is responsible for accessing the memory, updating its state and optionally producing output at each timestep.

The first two functions are executed by an interface network and a state network 1 , respectively.

Usually, the interface network is a Feedforward neural network whose input is c t -the output of the state network implemented as RNNs.

Let W c denote the weight of the interface network, then the state update and memory control are as follows,

where x t and r t−1 are data from current input and the previous memory read, respectively.

The interface vector ξ t then is used to read from and write to the memory M. We use a generic notation memory (ξ t , M) to represent these memory operations that either update or retrieve read value r t from the memory.

To support multiple memory accesses per step, the interface network may produce multiple interfaces, also known as control heads.

Readers are referred to App.

F and Graves et al. (2014; ; Santoro et al. (2016) for details of memory read/write examples.

A deterministic one-tape Turing Machine can be defined by 4-tuple (Q, Γ, δ, q 0 ), in which Q is finite set of states, q 0 ∈ Q is an initial state, Γ is finite set of symbol stored in the tape (the data) and δ is the transition function (the program), δ : Q × Γ → Γ × {−1, 1} × Q. At each step, the machine performs the transition function, which takes the current state and the read value from the tape as inputs and outputs actions including writing new values, moving tape head to new location (left/right) and jumping to another state.

Roughly mapping to current MANNs, Q, Γ and δ map to the set of the controller states, the read values and the controller network, respectively.

Further, the function δ can be factorized into two sub functions: Q × Γ → Γ × {−1, 1} and Q × Γ → Q, which correspond to the interface and state networks, respectively.

By encoding a Turing Machine into the tape, one can build a UTM that simulates the encoded machine (Turing, 1936) .

The transition function δ u of the UTM queries the encoded Turing Machine that solves the considering task.

Amongst 4 tuples, δ is the most important and hence uses most of the encoding bits.

In other words, if we assume that the space of Q, Γ and q 0 are shared amongst Turing Machines, we can simulate any Turing Machine by encoding only its transition function δ.

Translating to neural language, if we can store the controller network into a queriable memory and make use of it, we can build a Neural Universal Turing Machine.

Using NSM is a simple way to achieve this goal, which we introduce in the subsequent section.

A Neural Stored-program Memory (NSM) is a key-value memory M p ∈ R P ×(K+S) , whose values are the basis weights of another neural network−the programs.

P , K, and S are the number of programs, the key space dimension and the program size, respectively.

This concept is a hybrid between the traditional slow-weight and fast-weight (Hinton & Plaut, 1987) .

Like slow-weight, the keys and values in NSM are updated gradually by backpropagation.

However, the values are dynamically interpolated to produce the working weight on-the-fly during the processing of a sequence, which resembles fast-weight computation.

Let us denote M p (i) .k and M p (i) .v as the key and the program of the i-th memory slot.

At timestep t, given a query key k p t , the working program is retrieved as follows,

where D (·) is cosine similarity and β p t is the scalar program strength parameter.

The vector working program p t is then reshaped to its matrix form and ready to be used as the weight of other neural networks.

The key-value design is essential for convenient memory access as the size of the program stored in M p can be millions of dimensions and thus, direct content-based addressing as in Graves et al. (2014; ; Santoro et al. (2016) is infeasible.

More importantly, we can inject external control on the behavior of the memory by imposing constraints on the key space.

For example, program collapse will happen when the keys stored in the memory stay close to each other.

When this happens, p t is a balanced mixture of all programs regardless of the query key and thus having multiple programs is useless.

We can avoid this phenomenon by minimizing a regularization loss defined as the following,

It turns out that the combination of MANN and NSM approximates a Universal Turing Machine (Sec. 2).

At each timestep, the controller in MANN reads its state and memory to generate control signal to the memory via the interface network W c , then updates its state using the state network RN N .

Since the parameters of RN N and W c represent the encoding of δ, we should store both into NSM to completely encode an MANN.

For simplicity, in this paper, we only use NSM to store W c , which is equivalent to the Universal Turing Machine that can simulate any one-state Turing Machine.

In traditional MANN, W c is constant across timesteps and only updated slowly during training, typically through backpropagation.

In our design, we compute W c t from NSM for every timestep and thus, we need a program interface network−the meta network P I −that generates an interface vector for the program memory: ξ

That is, only the meta-network learns the mapping from context c t to program.

When it falls into some local-minima (generating suboptimal w p t ), the metanetwork struggles to escape.

In our proposal, together with the meta-network, the memory keys are learnable.

When the memory keys are slowly updated, the meta-network will shift its query key generation to match the new memory keys and possibly escape from the local-minima.

For the case of multi-head NTM, we implement one NSM per control head and name this model Neural Universal Turing Machine (NUTM).

One NSM per head is to ensure programs for one head do not interfere with other heads and thus, encourage functionality separation amongst heads.

Each control head will read from (for read head) or write to (for write head) the data memory M via memory (ξ t , M) as described in Graves et al. (2014) .

It should be noted that using multiple heads is unlike using multiple controllers per head.

The former increases the number of accesses to the data memory at each timestep and employs a fixed controller to compute multiple heads, which may improve capacity yet does not enable adaptability.

On the contrary, the latter varies the property of each memory access across timesteps by switching the controllers and thus potential for adaptation.

Other MANNs such as DNC and LRUA (Santoro et al., 2016) can be armed with NSM in this manner.

We also employ the regularization loss l p to prevent the programs from collapsing, resulting in a final loss as follows,

where Loss pred is the prediction loss and η t is annealing factor, reducing as the training step increases.

The details of NUTM operations are presented in Algorithm 1.

Learning to access memory is a multi-dimensional regression problem.

Given the input c t , which is derived from the state h t of the controller, the aim is to generate a correct interface vector ξ t via optimizing the interface network.

Instead of searching for one transformation that maps the whole space of c t to the optimal space of ξ t , NSM first partitions the space of c t into subspaces, then finds multiple transformations, each of which covers subspace of

Require: a sequence x = {x t } T t=1 , a data memory M and R program memories {M p,n } R n=1

corresponding to R control heads 1: Initilize h 0 , r 0 2: for t = 1, T do 3:

RN N can be replaced by GRU/LSTM 4:

for n = 1, R do

Compute the program interface ξ p t,n ← P I,n (c t )

6:

Compute the data interface ξ t,n ← c t W c t,n 8:

Read r t,n from memory M (if read head) or update memory M (if write head) using memory n (ξ t,n , M) c t .

The program interface network P I is a meta learner that routes c t to the appropriate transformation, which then maps c t to the ξ t space.

This is analogous to multilevel regression in statistics (Andrew Gelman, 2006) .

Practical studies have shown that multilevel regression is better than ordinary regression if the input is clustered (Cohen et al., 2014; Huang, 2018) .

RNNs have the capacity to learn to perform finite state computations (Casey, 1996; Tiňo et al., 1998) .

The states of a RNN must be grouped into partitions representing the states of the generating automaton.

As Turing Machines are finite state automata augmented with an external memory tape, we expect MANN, if learnt well, will organize its state space clustered in a way to reflect the states of the emulated Turing Machine.

That is, h t as well as c t should be clustered.

We realize that NSM helps NTM learn better clusterization over this space (see App.

A), thereby improving NTM's performances.

In this section, we investigate the performance of NUTM on algorithmic tasks introduced in Graves et al. (2014) doubles the length of training sequences in the Copy task.

In these tasks, the model will be fed a sequence of input items and is required to infer a sequence of output items.

Each item is represented by a binary vector.

In the experiment, we compare two models: NTM 2 and NUTM with two programs.

Although the tasks are atomic, we argue that there should be at least two memory manipulation schemes across timesteps, one for encoding the inputs to the memory and another for decoding the output from the memory.

The two models are trained with cross-entropy objective function under the same setting as in Graves et al. (2014) .

For fair comparison, the controller hidden dimension of NUTM is set smaller to make the total number of parameters of NUTM equivalent to that of NTM.

The number of memory heads for both models are always equal and set to the same value as in the original paper (details in App.

C).

We run each experiments five times and report the mean with error bars of training losses for NTM tasks in Fig. 2 (a) .

Except for the Copy task, which is too simple, other tasks observe convergence speed improvement of NUTM over that of NTM, thereby validating the benefit of using two programs across timesteps even for the single task setting.

NUTM requires fewer training samples to converge and it generalizes better to unseen sequences that are longer than training sequences.

Table 1 reports the test results of the best models chosen after five runs and confirms the outperformance of NUTM over NTM for generalization.

To illustrate the program usage, we plot NUTM's program distributions across timesteps for Repeat Copy and Priority Sort in Fig. 3 (a) and (b), respectively.

Examining the read head for Repeat Copy, we observe two program usage patterns corresponding to the encoding and decoding phases.

As there is no reading in encoding, NUTM assigns the "no-read" strategy mainly to the "orange program".

In decoding, the sequential reading is mostly done by the "blue program" with some contributions from the "orange program" when resetting reading head.

Similar behaviors can be found in the write head for Priority Sort.

While the encoding "fitting writing" (see Graves et al. (2014) for explanation on the strategy) is often executed by the "blue program", the decoding writing is completely taken by the "orange" program (more visualizations in App.

B).

In this section, we conduct an ablation study on Associative Recall (AR) to validate the benefit of proposed components that constitute NSM.

We run the task with three additional baselines: NUTM using direct attention (DA), NUTM using key-value without regularization (KV), NUTM using fixed, uniform program distribution (UP) and a vanilla NTM with 2 memory heads (h = 2).

The meta-network P I in DA generates the attention weight w p t directly.

The KV employs key-value attention yet excludes the regularization loss presented in Eq. (6).

The training curves over 5 runs are plotted in Fig. 2 (b) .

The results demonstrate that DA exhibits fast yet shallow convergence.

It tends to fall into local minima, which finally fails to reach zero loss.

Key-value attention helps NUTM converge completely with fewer iterations.

The performance is further improved with the proposed regularization loss.

UP underperforms NUTM as it lacks dynamic programs.

The NTM with 2 heads shows slightly better convergence compared to the NTM, yet obviously underperforms NUTM (p = 2) with 1 head and fewer parameters.

This validates our argument on the difference between using multiple heads and multiple programs (Sec. 3.2).

In neuroscience, sequencing tasks test the ability to remember a series of tasks and switch tasks alternatively (Blumenfeld, 2010) .

A dysfunctional brain may have difficulty in changing from one task to the next and get stuck in its preferred task (perseveration phenomenon).

To analyze this problem in NTM, we propose a new set of experiments in which a task is generated by sequencing a list of subtasks.

The set of subtasks is chosen from the NTM single tasks (excluding Dynamic N-grams for format discrepancy) and the order of subtasks in the sequence is dictated by an indicator vector put at the beginning of the sequence.

Amongst possible combinations of subtasks, we choose {Copy, Repeat Copy}(C+RC), {Copy, Associative Recall} (C+AR), {Copy, Priority Sort} (C+PS) and all (C+RC+AC+PS) 3 .

The learner observes the order indicator followed by a sequence of subtasks' input items and is requested to consecutively produce the output items of each subtasks.

As shown in Fig. 4 , some tasks such as Copy and Associative Recall, which are easy to solve if trained separately, become unsolvable by NTM when sequenced together.

One reason is NTM fails to change the memory access behavior (perseveration).

For examples, NTM keeps following repeat copy reading strategy for all timesteps in C+RC task (Fig. 3 (d) ).

Meanwhile, NUTM can learn to change program distribution when a new subtask appears in the sequence and thus ensure different accessing strategy per subtask (Fig. 3 (c) ).

In continual learning, catastrophic forgetting happens when a neural network quickly forgets previously acquired skills upon learning new skills (French, 1999) .

In this section, we prove the versatility of NSM by showing that a naive application of NSM without much modification can help NTM to mitigate catastrophic forgetting.

We design an experiment similar to the Split MNIST (Zenke et al., 2017) to investigate whether NSM can improve NTM's performance.

In our experiment, we let the models see the training data from the while freezing others, we force "hard" attention over the programs by replacing the softmax function in Eq. 5 with the Gumbel-softmax (Jang et al., 2016) .

Also, to ignore catastrophic forgetting in the state network, we use Feedforward controllers in the two baselines.

After finishing one task, we evaluate the bit accuracy −measured by 1−(bit error per sequence/total bits per sequence) over 4 tasks.

As shown in in Fig. 5 , NUTM outperforms NTM by a moderate margin (10-40% per task).

Although NUTM also experiences catastrophic forgetting, it somehow preserves some memories of previous tasks.

Especially, NUTM keeps performing perfectly on Copy even after it learns Repeat Copy.

For other dissimilar task transitions, the performance drops significantly, which requires more effort to bring NSM to continual learning.

Few-shot learning or meta learning tests the ability to rapidly adapt within a task while gradually capturing the way the task structure varies (Thrun, 1998) .

By storing sampleclass bindings, MANNs are capable of classifying new data after seeing only few samples (Santoro et al., 2016) .

As NSM gives flexible memory controls, it makes MANN more adaptive to changes and thus perform better in this setting.

To verify that, we apply NSM to the LRUA memory and follow the experiments introduced in Santoro et al. (2016) , using the Omniglot dataset to measure few-shot classification accuracy.

The dataset includes images of 1623 characters, with 20 examples of each character.

During training, a sequence (episode) of images are randomly selected from C classes of characters in the training set (1200 characters), where C = 5, 10 corresponding to sequence length of 50, 75, respectively.

Each class is assigned a random label which shuffles between episodes and is revealed to the models after each prediction.

After 100,000 episodes of training, the models are tested with unseen images from the testing set (423 characters).

The two baselines are MANN and NUTM (both use LRUA core).

For NUTM, we only tune p and pick the best values: p = 2 and p = 3 for 5 classes and 10 classes, respectively.

Table 2 : Test-set classification accuracy (%) on the Omniglot dataset after 100,000 episodes of training.

* denotes available results from (Santoro et al., 2016) .

Model Error DNC 16.7 ± 7.6 SDNC (Rae et al., 2016) 6.4 ± 2.5 ADNC (Franke et al., 2018) 6.3 ± 2.7 DNC-MD (Csordas & Schmidhuber, 2019) 9.5 ± 1.6 NUTM (DNC core, p=1) 9.7 ± 3.5 NUTM (DNC core, p=2) 7.5 ± 1.6 NUTM (DNC core, p=4)

5.6 ± 1.9 persistent memory mode, which demands fast forgetting old experiences in previous episodes, NUTM 4 .

Readers are referred to App.

D for more details on learning curves and more results of the models.

Reading comprehension typically involves an iterative process of multiple actions such as reading the story, reading the question, outputting the answers and other implicit reasoning steps .

We apply NUTM to the question answering domain by replacing the NTM core with DNC .

Compared to NTM's sequential addressing, dynamic memory addressing in DNC is more powerful and thus suitable for NSM integration to solve non-algorithmic problems such as question answering.

Following previous works of DNC, we use bAbI dataset to measure the performance of the NUTM with DNC core (three variants p = 1, p = 2 and p = 4).

In the dataset, each story is followed by a series of questions and the network reads all word by word, then predicts the answers.

Although synthetically generated, bAbI is a good benchmark that tests 20 aspects of natural language reasoning including complex skills such as induction and counting, We found that increasing number of programs helps NUTM improve performance.

In particular, NUTM with 4 programs, after 50 epochs jointly trained on all 20 question types, can achieve a mean test error rate of 3.3% and manages to solve 19/20 tasks (a task is considered solved if its error <5%).

The mean and s.d.

across 10 runs are also compared with other results reported by recent works (see Table 3 ).

Excluding baselines under different setups, our result is the best reported mean result on bAbI that we are aware of.

More details are described in App.

E.

Previous investigations into MANNs mostly revolve around memory access mechanisms.

The works in Graves et al. (2014; introduce content-based, location-based and dynamic memory reading/writing.

Further, Rae et al. (2016) scales to bigger memory by sparse access; Le et al. (2019) optimizes memory operations with uniform writing; and MANNs with extra memory have been proposed (Le et al., 2018b) .

However, these works keep using memory for storing data rather than the weights of the network and thus parallel to our approach.

Other DNC modifications (Csordas & Schmidhuber, 2019; Franke et al., 2018) are also orthogonal to our work.

Another line of related work involves modularization of neural networks, which is designed for visual question answering.

In module networks (Andreas et al., 2016b; a) , the modules are manually aligned with predefined concepts and the order of execution is decided by the question.

Although the module in these works resembles the program in NSM, our model is more generic and flexible with soft-attention over programs and thus fully differentiable.

Further, the motivation of NSM does not limit to a specific application.

Rather, NSM aims to help MANN reach general-purpose computability.

If we view NSM network as a dynamic weight generator, the program in NSM can be linked to fast weight (von der Malsburg, 1981; Hinton & Plaut, 1987; Schmidhuber, 1993b) .

These papers share the idea of using different weights across timesteps to enable dynamic adaptation.

Using outer-product is a common way to implement fast-weight (Schmidhuber, 1993a; Schlag & Schmidhuber, 2017) .

These fast weights are directly generated and thus different from our programs, which are interpolated from a set of slow weights.

Tensor/Multiplicative RNN (Sutskever et al., 2011) and Hypernetwork (Ha et al., 2016) are also relevant related works.

These methods attempt to make the working weight of RNNs dependent on the input to enable quick adaption through time.

Nevertheless, they do not support modularity.

In particular, Hypernetwork generates scaling factors for the single weight of the main RNN.

It does not aim to use multiple slow-weights (programs) and thus, different from our approach.

Tensor RNN is closer to our idea when the authors propose to store M slow-weights, where M is the number of input dimension, which is acknowledged impractical.

Unlike our approach, they do not use a meta-network to generate convex combinations amongst weights.

Instead, they propose Multiplicative RNN that factorizes the working weight to product of three matrices, which looses modularity.

On the contrary, we explicitly model the working weight as an interpolation of multiple programs and use a meta-network to generate the coefficients.

This design facilitates modularity because each program is trained towards some functionality and can be switched or combined with each other to perform the current task.

Last but not least, while the related works focus on improving RNN with fast-weight, we aim to reach a neural simulation of Universal Turing Machine, in which fast-weight is a way to implement stored-program principle.

This paper introduces the Neural Stored-program Memory (NSM), a new type of external memory for neural networks.

The memory, which takes inspirations from the stored-program memory in computer architecture, gives memory-augmented neural networks (MANNs) flexibility to change their control programs through time while maintaining differentiability.

The mechanism simulates modern computer behavior, potential making MANNs truly neural computers.

Our experiments demonstrated that when coupled with our model, the Neural Turing Machine learns algorithms better and adapts faster to new tasks at both sequence and sample levels.

When used in few-shot learning, our method helps MANN as well.

We also applied the NSM to the Differentiable Neural Computer and observed a significant improvement, reaching the state-of-the-arts in the bAbI task.

Although this paper limits to MANN integration, other neural networks can also reap benefits from our proposed model, which will be explored in future works.

Table 9 : Task settings (continual procedure learning tasks).

We use similar hyper-parameters as in Santoro et al. (2016) , which are reported in Tab Table 11 : Test-set classification accuracy (%) on the Omniglot dataset after 100,000 episodes of training.

* denotes available results from Santoro et al. (2016) (some are estimated from plotted figures).

We train the models using RMSprop optimizer with fixed learning rate of 10 −4 and momentum of 0.9.

The batch size is 32 and we adopt layer normalization (Lei Ba et al., 2016 ) to DNC's layers.

Following Franke et al. (2018) practice, we also remove temporal linkage for faster training.

The details of hyper-parameters are listed in Table 12 .

Full NUTM (p = 4) results are reported in 3.3 5.6 ± 1.9 Failed (Err.

>5%) 1 3 ± 1.2 Table 13 : NUTM (p = 4) bAbI best and mean errors (%).

9 When p = 1, the model converges to layer-normed DNC For all tasks, η t is fixed to 0.1, reducing with decay rate of 0.9.

Ablation study's learning losses with mean and error bar are plotted in Fig. 23 .

<|TLDR|>

@highlight

A neural simulation of Universal Turing Machine