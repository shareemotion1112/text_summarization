We present a novel approach for training neural abstract architectures which in- corporates (partial) supervision over the machine’s interpretable components.

To cleanly capture the set of neural architectures to which our method applies, we introduce the concept of a differential neural computational machine (∂NCM) and show that several existing architectures (e.g., NTMs, NRAMs) can be instantiated as a ∂NCM and can thus benefit from any amount of additional supervision over their interpretable components.

Based on our method, we performed a detailed experimental evaluation with both, the NTM and NRAM architectures, and showed that the approach leads to significantly better convergence and generalization capabilities of the learning phase than when training using only input-output examples.

Recently, there has been substantial interest in neural abstract machines that can induce programs from examples BID2 ; BID4 ; ; BID7 ; BID11 ; BID14 ; BID18 ; BID20 ; BID23 ; BID24 .

While significant progress has been made towards learning interesting algorithms BID8 , ensuring the training of these machines converges to the desired solution can be very challenging.

Interestingly however, even though these machines differ architecturally, they tend to rely on components (e.g., neural memory) that are more interpretable than a typical neural network (e.g., an LSTM).

A key question then is:Can we somehow provide additional amounts of supervision for these interpretable components during training so to bias the learning towards the desired solution?In this work we investigate this question in depth.

We refer to the type of supervision mentioned above as partial trace supervision, capturing the intuition that more detailed information, beyond inputoutput examples, is provided during learning.

To study the question systematically, we introduce the notion of a differential neural computational machine (∂NCM), a formalism which allows for clean characterization of the neural abstract machines that fall inside our class and that can benefit from any amount of partial trace information.

We show that common architectures such as Neural Turing Machines (NTMs) and Neural Random Access Machines (NRAMs) can be phrased as ∂NCMs, useful also because these architectures form the basis for many recent extensions, e.g., BID8 ; BID9 ; BID11 .

We also explain why other machines such as the Neural Program Interpreter (NPI) BID18 or its recent extensions (e.g., the Neural Program Lattice BID15 ) cannot be instantiated as an ∂NCM and are thus restricted to require large (and potentially prohibitive) amounts of supervision.

We believe the ∂NCM abstraction is a useful step in better understanding how different neural abstract machines compare when it comes to additional supervision.

We then present ∂NCM loss functions which abstractly capture the concept of partial trace information and show how to instantiate these for both the NTM and the NRAM.

We also performed an extensive evaluation for how partial trace information affects training in both architectures.

Overall, our experimental results indicate that the additional supervision can substantially improve convergence while leading to better generalization and interpretability.

To provide an intuition for the problem we study in this work, consider the simple task of training an NTM to flip the third bit in a bit stream (called Flip3rd) -such bitstream tasks have been extensively studied in the area of program synthesis (e.g., BID10 ; BID17 ).

An example input-output pair for this task could be examples, our goal is to train an NTM that solves this task.

An example NTM that generalizes well and is understandable is shown in FIG0 .

Here, the y-axis is time (descending), the x-axis is the accessed memory location, the white squares represent the write head of the NTM, and the orange squares represent the read head.

As we can see, the model writes the input sequence to the tape and then reads from the tape in the same order.

However, in the absence of richer supervision, the NTM (and other neural architectures) can easily overfit to the training set -an example of an overfitting NTM is shown in FIG0 .

Here, the traces are chaotic and difficult to interpret.

Further, even if the NTM generalizes, it can do so with erratic reads and writes, an example of which is shown in FIG0 .

Here, the NTM learns to read from the third bit (circled) with a smaller weight than from other locations, and also reads and writes erratically near the end of the sequence.

This model is less interpretable than the one in FIG0 because it is unclear how the model knows which the third bit actually is, or why a different read weight would help flip that bit.

In this work we will develop principled ways for guiding the training of a neural abstract machine towards the behavior shown in FIG0 .

For instance, for Flip3rd, providing partial trace information on the NTM's read heads for 10% of the input-output examples is sufficient to bias the learning towards the NTM shown in FIG0 100% of the time.

To capture the essence of our method and illustrate its applicability, we now define the abstract notion of a neural computational machine (NCM).

NCMs mimic classic computational machines with a controller and a memory, and generalize multiple existing architectures.

Our approach for supervision with partial trace information applies to all neural architectures expressible as NCMs.

A useful feature of the NCM abstraction is that it clearly delineates end-to-end differentiable architectures BID7 's NTM, BID14 's NRAM), which can train with little to no trace supervision, from architectures that are not end-to-end differentiable BID18 's NPI) and hence require a certain minimum amount of trace information.

In the follow-up section, we show how to phrase two existing neural architectures (NTMs and NRAMs) as an NCM.An NCM is a triple of functions: a processor, a controller, and a loss:Processor The processor π : W × C × M → B × M performs a pre-defined set of commands C, which might involve manipulating memories in M .

The commands may produce additional feedback in B. Also, the processor's operation may depend on parameters in W .Controller The controller κ : W × B × Q × I → C × Q × O decides which operations the machine performs at each step.

It receives external inputs from I and returns external outputs in O. It can also receive feedback from the processor and command it to do certain operations (e.g., memory read).

The decisions the controller takes may depend on its internal state (from Q).

The controller can also depend on parameters in W .

For instance, if the controller is a neural network, then the network's weights will range over W .

Loss Function The loss function L e : Trace × E → R indicates how close a trace τ ∈ Trace of an execution of the machine (defined below) is to a behavior from a set E. The loss function provides a criterion for training a machine to follow a prescribed set of behaviors, and hence we impose certain differentiability conditions.

We require that the loss surface is continuous and piecewise differentiable with respect to the weights w ∈ W for all examples e and inputs x with traces τ (w, x): DISPLAYFORM0 Execution The execution of the machine begins with an input sequence x = {x t } n 1 and initial values of the controller state q 0 , memory m 0 , and processor feedback b 0 .

At each time step t = 1 . . .

n, controller and processor take turns executing according to the following equations: DISPLAYFORM1 A trace τ (w, x, b 0 , m 0 , q 0 ) = {(c t , b t , q t , y t , m t )} n 1 records these quantities' values at each time step.

We will occasionally write τ C , τ B , . . .

for the trace projected onto one of its components c, b, . . . .

∂NCMs Note that the differentiability conditions that we impose on the loss do not imply that any of the NCM functions π, κ and L e are continuous or differentiable.

They indeed can be highly discontinuous as in NCMs like BID21 's memory networks with a hard attention mechanism, or as in BID18 's neural programmer-interpreters.

In order to fix these discontinuities and recover a differentiable loss surface, these architectures train with strong supervision only: the training examples e ∈ E must provide a value for every traced quantity that comes from a discontinuous parameter.

In contrast, what we call differentiable neural computational machines (∂NCM), have κ, π and L e continuous and piecewise differentiable.

In this case, the loss surface is differentiable with respect to every parameter.

Thus, there is no need to specify corresponding values in the examples, and so we can train with as much trace information as available.

We now show how NTMs and NRAMs can be instantiated as ∂NCMs.

NTM as ∂NCM An Neural Turing Machine (NTM) BID7 FIG1 ) has access to a memory M ∈ R c×n of c cells of n real numbers each.

We suppose the machine has one read head and one write head, whose addresses are, respectively, the probability vectors r, w ∈ [0, 1] {1...c} .

At every time step, the read head computes the expected value m ∈ R n of a random cell at index i ∼ r. This value together with the current input are fed into a controller neural network, which then decides on several commands.

It decides what fraction e ∈ R n to erase and how much a ∈ R n to add to the cells underneath the write head.

The write head stores the tape expected after a random modification at index i ∼

w.

Then the controller indicates the head movement with two probability vectors ∆r, ∆w ∈ [0, 1] {−1,0,+1} which are convolved with the respective head addresses (the actual addressing mechanism is more involved, but we omit it for brevity) Finally, the controller produces the current output value.

In terms of NCMs, the NTM's variables fall into the following classes: DISPLAYFORM0 Each of these variables change over time according to certain equations (see Appendix A for details).The processor π and the controller κ functions for each time step satisfy: DISPLAYFORM1 The standard loss function L e for the NTM simply includes a term, such as cross-entropy or L 2 distance, for the machine output at every time step.

Each of these compare the machine output to the respective values contained in the examples e ∈ E.NRAM as ∂NCM A Neural Random Access Machine (NRAM) BID14 is a neural machine designed for ease of pointer (de-) referencing.

An NRAM has a variable sized memory M ∈ R c×c whose size varies between runs.

It also has access to a register file r ∈ R n×c with a constant number n of registers.

Both the memory and the registers store probability vectors over {1 . . .

c}.

The controller receives no external inputs, but at each time step reads the probability that a register assigns to 0.

It also produces no external output, except a probability f ∈ [0, 1] for termination at the current time step.

The output of the run is considered to be the final memory state.

Unlike the NTM, computation in the NRAM is performed by a fixed sequence of modules.

Each module implements a simple integer operation/memory manipulation lifted to probability vectors.

For example, addition lifts to convolution, while memory access is like that of the NTM.

At every time step the controller organizes the sequence of modules into a circuit, which is then executed.

The circuit is encoded by a pair of probability distributions per module, as shown in FIG1 .

These distributions specify respectively which previous modules or registers will provide a given module first/second arguments.

The distributions are stacked in the matrices a and b .

A similar matrix c is responsible for specifying what values should be written to the registers at the end of the time step.

The NCM instantiation of an NRAM is the following: DISPLAYFORM2 The equations that determine these quantities can be found in Appendix B. The processor function π and the controller function κ expressed in terms of these quantities are: DISPLAYFORM3 The loss of the NRAM is more complex than the NTM loss: it is an expectation with respect to the probability distribution p of termination time, as determined by the termination probabilities f t (see Appendix B).

For every t = 1 . . .

k, the loss considers the negative log likelihood that the i-th memory cell at that time step equals the value e i provided in the example, independently for each i: DISPLAYFORM4

Incorporating supervision during NCM training can be helpful with: (i) convergence: additional bias may steer the minimization of the NCM's loss function L e , as much as possible, away from local minima that do not correspond to good solutions, (ii) interpretability: the bias can also be useful in guiding the NCM towards learning a model that is more intuitive/explainable to a user (especially if the user already has an intuition on what it is that parts of the model should do), and (iii) generalization: the bias can steer the NCM towards solutions which minimize not just the loss on example of difficulties it has seen, but on significantly more difficult examples.

The way we provide additional supervision to NCMs is, by encoding, for example, specific commands issued to the processor, into extra loss terms.

Let us illustrate how we can bias the learning with an NTM.

Consider the task of copying the first half of an input sequence {x t } 2l 1 into the second half of the machine's output {y t } 2l 1 , where the last input x l from the first half is a special value indicating that the first half ended.

Starting with both heads at position 1, the most direct solution is to consecutively store the input to the tape during the first half of the execution, and then recall the stored values during the second half.

In such a solution, we expect the head positions to be: DISPLAYFORM0 To incorporate this information into the training, we add loss terms that measure the cross-entropy (H) between p(t) and w t as well as between q(t) and r t .

Importantly, we need not add terms for every time-step, but instead we can consider only the corner cases where heads change direction: DISPLAYFORM1 H(p(t), w t ) + H(q(t), r t ).

We now describe the general shape of the extra loss terms for arbitrary NCMs.

Since, typically, we can interpret only the memory and the processor in terms of well-understood operations, we will consider loss terms only for the memory state and the communication flow between the controller and the processor.

We leave the controller's hidden state unconstrained -this also permits us to use the same training procedure with different controllers.

The generic loss is expressed with four loss functions for the different components of an NCM trace: DISPLAYFORM0 For each part α ∈ {C, B, O, M }, we provide hints (t, v, µ) ∈ σ α that indicate a time step t at which the hint applies, an example v ∈ E α for the relevant component, and a weight w ∈ R of the hint.

The weight is included to account for hints having a different importance at different time-steps, but also to express our confidence in the hint, e.g., hints coming from noisy sources would get less weight.

A subtrace σ is a collection of hints used for a particular input-output example e.

We call it a subtrace because, typically, it contains hints for a proper subset of the states traced by the NCM during execution.

The net loss for a given input-output example and subtrace equals the original loss L e added to the weighted losses for all the hints, scaled by a constant factor λ: DISPLAYFORM1

For NTMs, we allow hints on the output y, the addresses r and w, and the tape M. We include extra loss terms for the memory state only (all other loss terms are zero): DISPLAYFORM0 Unlike the output and addresses, values on the tape are interpreted according to an encoding internal to the controller (which emerges only during training).

Forcing the controller to use a specific encoding for the tape, as we do with NTM output, can have a negative effect on training (in our experiments, training diverged consistently).

To remedy this, we do not apply loss to the tape directly but to a decoded version of a cell on the tape.

While a decoder might find multiple representations and overfit, we found that it forced just enough consistency to improve the convergence rate.

The decoder itself is an auxiliary network φ trained together with the NTM, which takes a single cell from memory as input.

The output of the decoder is compared against the expected value which should be in that cell: DISPLAYFORM1 For all subtraces we provide in our experiments with NTMs, the hints have the same unit weight.

For NRAMs, we hint which connections should be present in the circuit the controller constructs at each step, including the ones for register updates.

An example circuit is shown in FIG2 .

In terms of an NCM, this amounts to providing loss for commands and no loss for anything else.

We set the loss to the negative log likelihood of the controller choosing specific connections revealed in the hint: DISPLAYFORM0 In our experiments, we observed that assigning higher weight to hints at earlier timesteps is crucial for convergence of the training process.

For a hint at time-step t, we use the weight µ = (t + 1) −2 .

A possible reason for why this helps is that the machine's behavior at later time-steps is highly dependent on its behavior at the early time-steps.

Thus, the machine cannot reach a later behavior that is right before it fixes its early behavior.

Unless the behavior is correct early on, the loss feedback from later time-steps will be mostly noise, masking the feedback from early time-steps.

Other Architectures The NCM can be instantiated to architectures as diverse as a common LSTM network or End-To-End Differentiable Memory Networks.

Any programming inducing neural network with at least partially interpretable intermediate states for which the dataset contains additional hints could be considered a good candidate for application of this abstraction.

We evaluated our NCM supervision method on the NTM and NRAM architectures.

For each of the two architectures we implemented a variety of tasks and experimented with different setups of trace supervision.

The main questions that we address are: (i) does trace supervision help convergence, interpretability, and generalization? (ii) how much supervision is needed to train such models?

Below, we summarize our findings -further details are provided in the appendix.

Figure 5: The number of initial runs which generalized for Flip3rd.

The first dimension listed in the rows controls the execution details revealed in a subtrace, while the second dimension (the density column) controls the proportion of examples that receive extra subtrace supervision.

We measured how often we successfully trained an NTM that achieves strong generalization.

We consider a model to generalize if relative to the training size limit n, it achieves perfect accuracy on all of tests of size ≤ 1.5n, and perfect accuracy on 90% of the tests of size ≤ 2n.

FIG3 reports the average improvement compared to a baseline using only I/O examples.

We ran experiments with four different tasks and various types of hints (cf.

Appendices C, E).

Some of the hint types are: read and write specify respective head addresses for all time steps; address combines the previous two; corner reveals the head addresses, but only when the heads change direction; value gives value for a single cell.

Except for three cases, trace supervision helped improve generalization.

Here, RepeatFlip3d is most challenging, with baseline generalizing only 5% of the time (cf.

Appendix I).

Here we have the largest improvement with extra supervision: corner type of hints achieve eight-fold increase in success rate, reaching 40%.

Another task with an even larger ratio is RepeatCopyTwice (cf.

Appendix), where success increases from 15.5% to 100%.In addition to this experiment, we performed an extensive evaluation of different setups, varying the global λ parameter of the loss Eq. 8, and providing hints for just a fraction of the examples.

The full results are in Appendix I; here we provide those for RepeatFlip3d in Table 5 .

The table reveals that the efficacy of our method heavily depends on these two parameters.

The best results in this case are for the read/corner type of hints 1 2 / 1 10 of the time, with λ ∈ {0.1, 1}. The best results for other tasks are achieved with different setups.

Generally, our conclusion is that training with traces 50% of the time usually improves performance (or does not lower it much) when compared to the best method.

This observation raises the interesting question of what the best type and amount of hints are for a given task.

Finally, we observed that in all cases where training with trace supervision converged, it successfully learned the head movements/tape values we had intended.

This show that trace supervision can bias the architecture towards more interpretable behaviors.

In those cases, the NTM learned consistently sharper head positions/tape values than the baseline, as FIG4 shows for Flip3rd.

BID16 reporting that ListK for example generalizes poorly, even when trained with noise in the gradient, curriculum learning, and an entropy bonus.

We observed that when run on an indefinite number of examples with the correct number of timesteps and a correct module sequence, Swap and Increment would in fact occasionally generalize perfectly, but did not have the resources to run such indefinite tests with Permute, ListK, and Merge.

FIG5 demonstrates that when training had finished, either because it had ended early or had reached 5000 training examples (our upper bound), generalization would in fact be on average significantly better than the baseline the more hints that were used for all tasks.

Here, number of hints used seemed to be a sufficient predictor for the quality of the trained model.

The effect of increasing supervision on the quality of the trained model was so strong that not even noise in the input was able to significantly hinder generalization.

In FIG5 , we corrupted a single character in the output examples for the Permute problem in 10% of the examples.

We found that without any extra hints, no convergence was seen after training was complete, whereas with just corner subtraces, the generalization was nearly optimal.

Furthermore, we found that noise in the trace does not seriously harm performance.

We corrupted a single hint for 20% of the traces of the Increment task using otherwise full supervision, as can be seen in the NoisyFull line of FIG0 .

We presented a method for incorporating (any amount of) additional supervision into the training of neural abstract machines.

The basic idea was to provide this supervision (called partial trace information) over the interpretable components of the machine and to thus more effectively guide the learning towards the desired solution.

We introduced the ∂NCM architecture in order to precisely capture the neural abstract machines to which our method applies.

We showed how to formulate partial trace information as abstract loss functions, how to instantiate common neural architectures such as NTMs and NRAMs as ∂NCMs and concretize the ∂NCM loss functions.

Our experimental results indicate that partial trace information is effective in biasing the learning of both NTM's and NRAM's towards better converge, generalization and interpretability of the resulting models.

The controller for the NTM consists of the networks ϕ, ψ y , ψ e , ψ a , χ r , χ w , which operate on the variables:x -in q -controller state r -read address ∆r -change in r e -erase M -tape y -out m -read value w -write address ∆w -change in w a -addThe equations that describe NTM executions are: DISPLAYFORM0

The controller of the NRAM consists of the networks ϕ, ψ a , ψ b , ψ c , ψ f , which operate on the variables:a -lhs circuit b -rhs circuit c -register inputs o -module outputs r -register state M -memory tape h -controller state f -stop probability.

The equations that describe the NRAM execution are: DISPLAYFORM0 DISPLAYFORM1 For all of our NTM experiments we use a densely connected feed-forward controller.

There are two architectural differences from the original NTM BID7 that helped our baseline performance: (1) the feed-forward controller, the erase and the add gates use tanh activation; (2) the output layer uses softmax.

In the original architecture these are all logistic sigmoids.

For the newly introduced tape decoder (active only during training) we used two alternative implementations: a tanh-softmax network, and a single affine transformation.

We tested the NTM's learning ability on five different tasks for sequence manipulation, two of which have not been previously investigated in this domain.

These tasks can be found in Appendix E.We performed experiments using several combination of losses as summarized in Appendix F. The observed training performance per task is shown in Appendix I, with rows corresponding to the different loss setups.

The corner setup differs from the address setup in that the example subtraces were defined only for a few important corner cases.

For example in RepeatCopyTwice, the write head was provided once at the beginning of the input sequence, and once at the end.

Similarly, the read head was revealed at the beginning and at the end of every output repetition.

In all other setups we provide full subtraces (defined for all time steps).The supervision amount can be tuned by adjusting the λ weight from Equation 8.

Further, we can also control the fraction of examples which get extra subtrace supervision (the density row in Figure I ).

The performance metric we use is the percentage of runs that do generalize after 100k iterations for the given task and supervision type.

By generalize we mean that the NTM has perfect accuracy on all testing examples up to 1.5× the size of the max training length, and also perfect accuracy on 90% of the testing examples up to 2× the maximum training length.

We used a feed-forwad controller with 2 × 50 units, except for RepeatCopyTwice, which uses 2 × 100 units.

For training we used the Adam optimizer BID12 , a learning rate of 10 −3 for all tasks except RepeatFlip3d and Flip3rd which use 5 · 10 −4 .

The lengths of the training sequences for the first four tasks are from 1 to 5, whereas the generalization of the model was tested with sequences of lengths up to 20.

For Flip3rd and RepeatFlip3d, the training sequence length was up to 16, whereas the testing sequences have maximum length of 32.

Like in the NTM, we use a densely connected two layer feed forward controller for our experiments, and use ReLU as the activation function.

We make no modifications to the original architecture, and use noise with the parameter η = 0.3 as suggested by BID16 , and curriculum learning as described by BID22 .

We stop training once we get to a difficulty specified by the task, and increase the difficulty once 0 errors were found on a new testing batch of 10 samples.

Each training iteration trains with 50 examples of the currently randomly sampled difficulty.

Regardless of whether the model had converged, training is stopped after 5000 samples were used.

Such a low number is used to replicate the potential conditions under which such a model might be used.

As with the NTM, the Adam optimizer was used.

The specific tasks we use are described in Appendix G, and the specific kinds of supervision we give are described in Appendix H. The λ we used here was 40.

The system was implemented using PyTorch.

Every input sequence ends with a special delimiter x E not occurring elsewhere in the sequence Copy -The input consists of generic elements, x 1 . . .

x n x E .

The desired output is x 1 . . .

x n x E .RepeatCopyTwice -The input is again a sequence of generic elements, x 1 . . .

x n x E .

The desired output is the input copied twice x 1 . . .

x n x 1 . . .

x n x E .

Placing the delimiter only at the end of the output ensures that the machine learns to keep track of the number of copies.

Otherwise, it could simply learn to cycle through the tape reproducing the given input indefinitely.

We kept the number of repetitions fixed in order to increase baseline task performance for the benefit of comparison.

DyckWords -The input is a sequence of open and closed parentheses, x 1 . . .

x n x E .

The desired output is a sequence of bits y 1 . . .

y n x E such that y i = 1 iff the prefix x 1 . . .

x i is a balanced string of parentheses (a Dyck word).

Both positive and negative examples were given.

Flip3rd -The input is a sequence of bits, x 1 x 2 x 3 . . .

x n x E .

The desired output is the same sequence of bits but with the 3rd bit flipped: x 1 x 2x3 . . .

x n x E .

Such a task with a specific index to be updated (e.g., 3rd) still requires handling data dependence on the contents of the index (unlike say the Copy task).RepeatFlip3d -The input is a sequence of bits, x 1 x 2 x 3 x 4 x 5 x 5 . . .

x E .

The desired output is the same sequence of bits but with every 3rd bit flipped: DISPLAYFORM0 F NTM SUBTRACES value traces provide hints for the memory at every timestep as explained in Equation FORMULA0 .read -provides a hint for the address of the read head at every timestep.write -provides a hint for the address of the write head at every timestep.address -provides hints for the address of both the read and the write head at every timestep.addr+val -provides value, read and write hints for every timestep.corner -provides hints for the address of both the read and the write head at every "important" timestep -we decided what important means here depends on which task we are referring to.

In general, we consider the first and last timesteps important, and also any timestep where a head should change direction.

For example, in RepeatCopyTwice for an example of size n with e repeats, we'd provide the heads at timesteps 0, n, 2n, 3n . . .

, en.

Below we describe all the tasks we experimented with.

We predominantly picked tasks that the NRAM is known to have trouble generalizing on.

We did not introduce any new tasks, and more detailed descriptions of these tasks can be found in BID14 .Swap -Provided two numbers, a and b and an array p, swap p[a] and p [b] .

All elements but that in the last memory cell are not zero.

Increment -Given an array p, return the array with one added to each element.

All elements but that in the last cell for the input are not zero.

Elements can be zero in the output.

Permute -Given two arrays p and q return a new array s such that DISPLAYFORM0 The arrays p and q are preceded by a pointer, a, to array q. The output is expected to be a, DISPLAYFORM1 ListK -Given a linked list in array form, and an index k return the value at node k.

Merge -given arrays p and q, and three pointers a, b, c to array p, q, and the output sequence (given as zeros initially), place the sorted combination of p and q into the output location.

The following table describes the specific NRAM instantiation used for each task.

The default sequence (def) is the one described by BID14 .

The number of timesteps is usually dependent on the length of the problem instance, M (equivalently the word size or difficulty), and in the case of ListKwas given with respect to the argument k. The difficulty (D) was simply the length of the sequence used.

H NRAM SUBTRACES For each of the tasks listed Appendix G, we hand coded a complete circuit for every module and every timestep we would provide.

The following subtraces types describe how we provide hints based on this circuit.

None -provides no hints.

Full -provides the entire circuit.

SingleHint -provides a random hint at a random timestep.

SingleTimestep -provides the entire circuit at a random timestep.

Corners -provides the entire circuit at the first and last timesteps.

Registers -provides hints for the registers at every timestep.

Modules -provides hints for the modules at every timestep.

Which Details to Reveal for NTM?

The first dimension listed in the rows of the tables of Figure I controls the execution details revealed in a Subtrace.

We use subtraces showing either the addresses without the tape values, only the read heads or the write heads, or even weaker supervision in a few corner cases.

In tasks Copy FIG7 ), RepeatCopyTwice ( FIG7 ) and DyckWords FIG7 , it is frequently the case that when the NTM generalizes without supervision, it converges to an algorithm which we are able to interpret.

For them, we designed the addr+val traces to match this algorithm, and saw increases in generalization frequency of at least 45%.

It can be concluded that when additionally provided supervision reflects the interpretable "natural" behavior of the NTM, the learning becomes significantly more robust to changes in initial weights.

Additionally, for tasks Flip3rd FIG7 ) and RepeatFlip3d FIG7 ), both the baseline and other supervision types are outperformed by training with read supervision.

It is also notable that corner supervision in RepeatFlip3d achieves highest improvement over the baseline, 60% over 5%.

In essence, this means that providing only a small part of the trace can diminish the occurrence of local minima in the loss function.

How Often to Reveal for NTM?

The second dimension controls the proportion of examples that receive extra subtrace supervision (the density columns in Figure I ).

For Flip3rd, RepeatCopyTwice and DyckWords we observed that having only a small number of examples with extra supervision leads to models which are more robust to initial weight changes than the baseline, although not necessarily always as robust as providing supervision all the time.

A couple of interesting cases stand out.

For Flip3rd with 10% corner subtraces and λ = 1, we find a surprisingly high rate of generalization.

Providing address traces 10% of the time when training RepeatCopyTwice leads to better performance all the time.

For RepeatFlip3d, write traces at 1% frequency and λ = 0.1 generalize 30% of the time vs. 5% for baseline.

While the type of trace which works best varies per task, for each task there exists a trace which can be provided only 1% of the time and still greatly improve the performance over the baseline.

This suggests that a small amount of extra supervision can improve performance significantly, but the kind of supervision may differ.

It is an interesting research question to find out how the task at hand relates to the optimal kind of supervision.

FIG0 : The average number of errors on the test set for each task and subtrace type once trained.

FORMULA0 with that of the NRAM using the full tracer for Merge.

For this experiment, a maximum of 10000 samples were used for the DNGPU and 5000 for the NRAM.

The DNGPU was run out of the box from the code supplied by the authors.

20 runs were averaged for the DNGPU and 38 runs for the NRAM.

One can deduce that while neither is able to generalize this task perfectly, the simpler and easier to understand architecture, NRAM, does generalize better with fewer examples when those examples come with richer supervision.

The NRAM is parametrized by one or more straight-line partial programs, i.e., programs with no branching and no loops, chosen by register states.

The machine runs in a loop, repeatedly selecting the program for that register state then executing it.

The programs are expressend in a simple single-assignment imperative language.

Each program statement i invokes one of the modules of the architecture and assigns the result of the invocation to a local variable x i .

That variable cannot be changed later.

The final program statement is a parallel-asignment that modifies the machine registers r 1 . . .

r k .

The values that appear in assignments/invocations can be: variables in scope, machine registers, or holes ?.

These values are not used directly during execution: the actual values needs to be supplied by the NRAM controller.

The values are only used as hints for the controller during training, with the whole ? denoting no hint.

We can describe the language in an EBNF-style grammar: P 1 ::= S 1 P i ::= P i−1 ; S i P ::= P 1 ; R 1 | P 2 ; R 2 | . .

.An example program for the Increment task would be the following: DISPLAYFORM0 x 2 ← READ(r 1 ); x 3 ← ADD(x 2 , x 1 ); x 4 ← WRITE(r 1 , x 3 ); x 5 ← ADD(r 1 , x 1 ); DISPLAYFORM1 Here, the controller is encouraged to read the memory at the location stored in the first register r 1 , add one to it, then store it back, and then increment the the first register.

An alternative to the trace-based approach is to make the controller produce values only for the holes, and use directly the specified variable/register arguments.

This way, only the unspecified parts of the program are learned.

This is, for example, the approach taken by ∂Forth BID0 .

There, programs are expressed in a suitably adapted variant of the Forth programming language, which is as expressive as the language discussed above, but less syntactically constrained.

The drawback of this alternative is that whenever an argument other than a whole is specified, one must also specify the time steps to which it applies in all possible executions and not just the training ones.

That is why, typically, these values are specified either for all or for none of the time steps.

In the following examples, we will describe the register states using "0", "!

0" and "-" meaning respectively that a register has 0, that it contains anything but zero, or that it can contain anything.

For any register pattern.x 1 ← READ(r 0 ); x 2 ← W RIT E(0, x 1 ); x 3 ← READ(r 1 ); x 4 ← ADD(x 3 , x 1 ); x 5 ← READ(x 4 ); x 6 ← W RIT E(r 1 , x 5 ); x 7 ← IN C(r 1 ); x 8 ← DEC(x 1 ); x 9 ← LT (x 7 , x 8 ); r 0 ← 0; r 1 ← x 7 ; r 2 ← x 9 ; r 3 ← 0;

<|TLDR|>

@highlight

We increase the amount of trace supervision possible to utilize when training fully differentiable neural machine architectures.