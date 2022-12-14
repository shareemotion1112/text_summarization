The concepts of unitary evolution matrices and associative memory have boosted the field of Recurrent Neural Networks (RNN) to state-of-the-art performance in a variety of sequential tasks.

However, RNN still has a limited capacity to manipulate long-term memory.

To bypass this weakness the most successful applications of RNN use external techniques such as attention mechanisms.

In this paper we propose a novel RNN model that unifies the state-of-the-art approaches: Rotational Unit of Memory (RUM).

The core of RUM is its rotational operation, which is,  naturally,  a unitary matrix, providing architectures with the power to learn long-term dependencies by overcoming the vanishing and exploding gradients problem.

Moreover,  the rotational unit also serves as associative memory.

We evaluate our model on synthetic memorization, question answering and language modeling tasks.

RUM learns the Copying Memory task completely and improves the state-of-the-art result in the Recall task.

RUM’s performance in the bAbI Question Answering task is comparable to that of models with attention mechanism.

We also improve the state-of-the-art result to 1.189 bits-per-character (BPC) loss in the Character Level Penn Treebank (PTB) task, which is to signify the applications of RUM to real-world sequential data.

The universality of our construction, at the core of RNN, establishes RUM as a promising approach to language modeling, speech recognition and machine translation.

Recurrent neural networks are widely used in a variety of machine learning applications such as language modeling BID7 ), machine translation BID5 ) and speech recognition BID11 ).

Their flexibility of taking inputs of dynamic length makes RNN particularly useful for these tasks.

However, the traditional RNN models such as Long Short-Term Memory (LSTM, BID12 ) and Gated Recurrent Unit (GRU, BID5 ) exhibit some weaknesses that prevent them from achieving human level performance: 1) limited memory-they can only remember a hidden state, which usually occupies a small part of a model; 2) gradient vanishing/explosion BID4 ) during training-trained with backpropagation through time the models fail to learn long-term dependencies.

Several ways to address those problems are known.

One solution is to use soft and local attention mechanisms BID5 ), which is crucial for most modern applications of RNN.

Nevertheless, researchers are still interested in improving basic RNN cell models to process sequential data better.

Numerous works BID7 ; BID2 ) use associative memory to span a large memory space.

For example, a practical way to implement associative memory is to set weight matrices as trainable structures that change according to input instances for training.

Furthermore, the recent concept of unitary or orthogonal evolution matrices BID0 ; BID14 ) also provides a theoretical and empirical solution to the problem of memorizing long-term dependencies.

Here, we propose a novel RNN cell that resolves simultaneously those weaknesses of basic RNN.

The Rotational Unit of Memory is a modified gated model whose rotational operation acts as associative memory and is strictly an orthogonal matrix.

We tested our model on several benchmarks.

RUM is able to solve the synthetic Copying Memory task while traditional LSTM and GRU fail.

For synthetic Recall task, RUM exhibits a stronger ability to remember sequences, hence outperforming state-of-the-art RNN models such as Fastweight RNN BID2 ) and WeiNet (Zhang & Zhou (2017) ).

By using RUM we achieve the state-of-the-art result in the real-world Character Level Penn Treebank task.

RUM also outperforms all basic RNN models in the bAbI question answering task.

This performance is competitive with that of memory networks, which take advantage of attention mechanisms.

Our contributions are as follows:1.

We develop the concept of the Rotational Unit that combines the memorization advantage of unitary/orthogonal matrices with the dynamic structure of associative memory; 2.

The Rotational Unit of Memory serves as the first phase-encoded model for Recurrent Neural Networks, which improves the state-of-the-art performance of the current frontier of models in a diverse collection of sequential task.

The problem of the gradient vanishing and exploding problem is well-known to obstruct the learning of long-term dependencies BID4 ).We will give a brief mathematical motivation of the problem.

Let's assume the cost function is C. In order to evaluate ∂C/∂W ij , one computes the derivative gradient using the chain rule: DISPLAYFORM0 where DISPLAYFORM1 } is the Jacobian matrix of the point-wise nonlinearity.

As long as the eigenvalues of D (k) are of order unity, then if W has eigenvalues λ i 1, they will cause gradient explosion ∂C/∂h (T ) → ∞, while if W has eigenvalues λ i 1, they can cause gradient vanishing, ∂C/∂h (T ) → 0.

Either situation hampers the efficiency of RNN.LSTM is designed to solve this problem, but gradient clipping BID22 ) is still required for training.

Recently, by restraining the hidden-to-hidden matrix to be orthogonal or unitary, many models have overcome the problem of exploding and vanishing gradients.

Theoretically, unitary and orthogonal matrices will keep the norm of the gradient because the absolute value of their eigenvalues equals one.

Several approaches have successfully developed the applications of unitary and orthogonal matrix to recurrent neural networks.

BID0 BID14 use parameterizations to form the unitary spaces.

Wisdom et al. (2016) applies gradient projection onto a unitary manifold.

BID28 uses penalty terms as a regularization to restrain matrices to be unitary, hence accessing long-term memorization.

Only learning long-term dependencies is not sufficient for a powerful RNN.

BID13 finds that the combination of unitary/orthogonal matrices with a gated mechanism improves the performance of RNN because of the benefits of a forgetting ability.

BID13 also points out the optimal way of such a unitary/gated combination: the unitary/orthogonal matrix should appear before the reset gate, which can then be followed by a modReLU activation.

In RUM we implement an orthogonal operation in the same place, but the construction of that matrix is completely different: instead of parameterizing the kernel, we encode a natural rotation, generated by the inputs and the hidden state.

Limited memory in RNN is truly a shortage.

Adding an external associative memory is a natural solution.

For instance, the Neural Turing Machine BID7 ) and many other models have shown the power of using this technique.

While it expands the accessible memory space, the technique significantly increases the size of the model, therefore making the process of learning so many parameters harder.

Now, we will briefly describe the concept of associative memory.

In basic RNN, h t = σ(W x t + Ah t−1 + b) where h t is the hidden state at time step t and x is the input data at each step.

Here W and A are trainable parameters that are fixed in the model.

A recent approach replaces A with a dynamic A t (as a function of time) so that this matrix can serve as a memory state.

Thus, the memory size increases from DISPLAYFORM0 , where N h is the hidden size.

In particular, A t is determined by A t−1 , h t−1 and x t which can be a part of a multi-layer or a Hopfiled net.

By treating the RNN weights as memory determined by the current input data, a larger memory size is provided and less trainable parameters are required.

This significantly increases the memorization ability of RNN.

Our model also falls into this category of associative memory through its rotational design of an orthogonal A t matrix.

Recently, BID23 proposed a novel neural network architecture that uses vectors instead of conventional single neurons to represent concepts in hidden states.

These vectors are called capsules.

Special connections are also designed to connect capsules through a process, called dynamic routing.

This work shows promising performance of phase-encoded models in Convolutional Neural Networks.

The Rotational Unit of Memory model, which we introduce below, serves as the first successful phase-encoded model in the RNN domain.

We give a detailed comparison of these two models in section 5.3.

Rotations are well-studied mathematical structures that have various fundamental applications in the theory of Lie groups (Artin (2011); BID9 ), quantum physics BID24 ), etc.

In computer vision BID25 ) the position and orientation of an object form a pose, which contains valuable information about the object.

A feasible way of estimating poses is through rotational matrices and quaternions BID15 ; BID18 ).The conventional way of representing memory in RNNs is by encoding the information in a hidden state, which is a vector of a certain finite dimension N .

To the best of our knowledge, the frontier of RNN models utilizes mostly the norm of the elements of the hidden state during the learning process.

Experiments and theory point, however, that representational advantages can be achieved, by using capsules as vectors in the Euclidean R N space and thereby allowing the model to manipulate the pose of these capsules BID23 ).Here, we equip the hidden state in an RNN with a pose by viewing it as a vector with position and orientation in R N .

We then propose an efficient method for manipulating the pose of the orientation by the means of rotations in an N -dimensional space.

Our particular parameterization for the rotation is a natural way to define a differentiable orthogonal operation within the RNN cell.

For the remainder of this section we suggest ways of engineering models that incorporate rotations as units of memory.

In the following discussion N x is the input size and N h is the hidden size.

The operation Rotation is an efficient encoder of an orthogonal operation, which acts as a unit of memory.

Rotation computes an orthogonal operator R(a, b) in R N h ×N h that represents the rotation between two non-collinear vectors a and b in the two-dimensional subspace span(a, b) of the Euclidean space R N h with distance · .

As a consequence, R can act as a kernel on a hidden state h. More formally, what we propose is a function DISPLAYFORM0 Other ways to extract an orthogonal operation from elements in the RNN cell are still possible.

Some approaches are as follows: 1.

Use a skew-symmetric matrix A to define the orthogonal operator e A ; 2.

Use a permutation operator.

However, those constructions are difficult to implement and do not offer a natural intuition about encoding memory.

We recognize that other constructions are also feasible and potentially interesting for research.

such that after ortho-normalizing a and b to DISPLAYFORM1 we encode the following matrix in DISPLAYFORM2 Figure 1 (a) demonstrates the projection to the plane span(a, b) in the brackets of equation FORMULA5 .

The A practical advantage of Rotation is that it is both orthogonal and differentiable.

On one hand, it is a composition of differentiable sub-operations, which enables learning via backpropagation.

On the other hand, it preserves the norm of the hidden state, hence it can yield more stable gradients.

We were motivated to find differentiable implementations of unitary (orthogonal in particular) operations in existing toolkits for deep learning.

Our conclusion is that Rotation can be implemented in various frameworks that are utilized for RNN and other deep learning architectures.

Indeed, Rotation is not constrained to parameterize a unitary structure, but instead it produces an orthogonal matrix from simple components in the cell, which makes it useful for experimentation.

DISPLAYFORM3 We implement Rotation together with its action on a hidden state efficiently.

We do not need to compute the matrix R t before we rotate.

Instead we can directly apply the RHS of equation FORMULA5 to the hidden state.

Hence, the memory complexity of our algorithm is O(N b ·N h ), which is determined by the RHS of (1).

Note that we only use two trainable vectors in R N h to generate orthogonal weights in R N h ×N h , which means the model has O(N 2 h ) degrees of freedom for a single unit of memory.

Likewise, the time complexity is O(N b · N 2 h ).

Thus, Rotation is a universal operation that enables implementations suitable to any neural network model with backpropagation.

We propose the Recurrent Unit of Memory as the first example of an application of Rotation to a recurrent cell.

FIG0 (b) is a sketch of the connections in the cell.

RUM consists of an update gate u ∈ R N h that has the same function as in GRU.

Instead of a reset gate, however, the model learns a memory target variable τ ∈ R N h .

RUM also learns to embed the input vector DISPLAYFORM0 Hence Rotation encodes the rotation between the embedded input and the target, which is accumulated to the associative memory unit R t ∈ R N h ×N h (originally initialized to the identity matrix).

Here λ is a non-negative integer that is a hyper-parameter of the model.

From here, the orthogonal R t acts on the state h to produce an evolved hidden stateh.

Finally RUM obtains the new hidden state via u, just as in GRU.

The RUM equations are as follows DISPLAYFORM1 σ activation of the update gate; ε t =W xh · x t +b t embedded input for Rotation; DISPLAYFORM2 rotational associative memory; DISPLAYFORM3 unbounded evolution of hidden state; DISPLAYFORM4 DISPLAYFORM5 The norm η is a scalar hyper-parameter of the RUM model.

The orthogonal matrix R(ε t , τ ) conceptually takes the place of a kernel acting on the hidden state in GRU.

This is the most efficient place to introduce an orthogonal operation, as the Gated Orthogonal Recurrent Unit (GORU, BID13 ) experiments suggest.

The difference with the GORU cell is that GORU parameterizes and learns the kernel as an orthogonal matrix, while RUM does not parameterize the rotation R. Instead, RUM learns τ , which together with x, determines R. The orthogonal matrix keeps the norm of the vectors, so we experiment with a ReLU activation instead of the conventional tanh in gated mechanisms.

Even though R is an orthogonal element of RUM, the norm of h t is not stable because of the ReLU activation.

Therefore, we suggest normalizing the hidden state h t to a have norm η.

We call this technique time normalization as we usually feed mini-batches to the RNN during learning that have the shape (N b , N T ), where N b is the size of the batch and N T is the length of the sequence that we feed in.

Time normalization happens along the sequence dimension as opposed to the batch dimension in batch normalization.

Choosing appropriate η for the RUM model stabilizes learning and ensures the eigenvalues of the kernels are bounded from above.

This in turn means that the smaller η is, the more we reduce the effect of exploding gradients.

Finally, even though RUM uses an update gate, it is not a standard gated mechanism, as it does not have a reset gate.

Instead we suggest utilizing additional memory via the target vector τ .

By feeding inputs to RUM, τ adapts to encode rotations, which align the hidden states in desired locations in R N h , without changing the norm of h. We believe that the unit of memory R t gives advantage to RUM over other gated mechanisms, such as LSTM and GRU.

Firstly, we test RUM's memorization capacity on the Copying Memory Task.

Secondly, we signify the superiority of RUM by obtaining a state-of-the-art result in the Associative Recall Task.

Thirdly, we show that even without external memory, RUM achieves comparable to state-of-the-art results in the bAbI Question Answering data set.

Finally, we utilize RUM's rotational memory to reach 1.189 BPC in the Character Level Penn Treebank.

We experiment with λ = 0 RUM and λ = 1 RUM, the latter model corresponding to tuning in the rotational associative memory.

A standard way to evaluate the memory capacity of a neural network is to test its performance in the Copying Memory Task BID12 , BID10 BID0 ).

We follow the setup in BID14 .

The objective of the RNN is to remember (copy) information received T time steps earlier (see section A for details about the data).Our results in this task demonstrate: 1.

RUM utilizes a different representation of memory that outperforms those of LSTM and GRU; 2.

RUM solves the task completely, despite its update gate, which does not allow all of the information encoded in the hidden stay to pass through.

The only other gated RNN model successful at copying is GORU.

FIG1 reveals that LSTM and GRU hit a predictable baseline, which is equivalent to random guessing.

RUM falls bellow the baseline, and subsequently learns the task by achieving zero loss after a few thousands iterations.

The spikes on the learning curves for RUM are arising from the fact that we are using a ReLU activation for RUM without gradient clipping.

With the help of figure 2 we will explain how the additional hyper-parameters for RUM affect its training.

We observe that when we remove the normalization (η = N/A) then RUM learns more quickly than the case of requiring a norm η = 1.0.

At the same time, though, the training entails more fluctuations.

Hence we believe that choosing a finite η to normalize the hidden state is an important tool for stable learning.

Moreover, it is necessary for the NLP task in this paper (see section 4.4): for our character level predictions we use large hidden sizes, which if left unnormalized, can make the cross entropy loss blow up.

We also observe the benefits of tuning in the associative rotational memory.

Indeed, a λ = 1 RUM has a smaller hidden size, N h = 100, yet it learns much more quickly than a λ = 0 RUM.

It is possible that the accumulation of phase via λ = 1 to enable faster long-term dependence learning than the λ = 0 case.

Either way, both models overcome the vanishing/exploding gradients, and eventually learn the task completely.

Another important synthetic task to test the memory ability of recurrent neural network is the Associative Recall.

This task requires RNN to remember the whole sequence of the data and perform extra logic on the sequence.

We follow the same setting as in BID2 and Zhang & Zhou (2017) and modify the original task so that it can test for longer sequences.

In detail, the RNN is fed into a sequence of characters, e.g. "a1s2d3f4g5??d".

The RNN is supposed to output the character based on the "key" which is located at the end of the sequence.

The RNN needs to look back into the sequence and find the "key" and then to retrieve the next character.

In this example, the correct answer is "3".

See section B for further details about the data.

In this experiment, we compare RUM to an LSTM, , a Fast-weight RNN BID2 ) and a recent successful RNN WeiNet (Zhang & Zhou (2017) ).

All the models have the same hidden state N h = 50 for different lengths T .

We use a batch size 128.

The optimizer is RMSProp with a learning rate 0.001.

We find that LSTM fails to learn the task, because of its lack of sufficient memory capacity.

NTM and Fast-weight RNN fail longer tasks, which means they cannot learn to manipulate their memory efficiently.

TAB2 gives a numerical summary of the results and figure 4, in the appendix, compares graphically RUM to LSTM.

Length

Question answering remains one of the most important applicable tasks in NLP.

Almost all stateof-the-art performance is achieved by the means of attention mechanisms.

Few works have been done to improve the performance by developing stronger RNN.

Here, we tested RUM on the bAbI Question Answering data set ) to demonstrate its ability to memorize and reason without any attention.

In this task, we train 20 sub-tasks jointly for each model, using a 10k training sets.

See section C for detailed experimental settings and results on each sub-task.

We compare our model with several baselines: a simple LSTM, an End-to-end Memory Network BID27 ) and a GORU.

We find that RUM outperforms significantly LSTM and GORU and achieves competitive result with those of MemN2N, which has an attention mechanism.

We summarize the results in Table 2 .

We emphasize that for some sub-tasks in the table, which require large memory, RUM outperforms models with attention mechanisms (MemN2N).

Test Accuracy (%) LSTM ) 49 GORU BID13 ) 60 MemN2N BID27 ) 86 RUM (ours) 73.2 Table 2 : Question Answering task on bAbI dataset.

Test accuracy (%) on LSTM, MemN2N, GORU and RUM.

RUM outperforms LSTM/GORU and is outperformed only by MemN2N, which uses an attention mechanism.

The Rotational Unit of Memory is a natural architecture that can learn long-term structure in data while avoiding significant overfitting.

Perhaps, the best way to demonstrate this unique property, among other RNN models, is to test RUM on real world character level NLP tasks.

The corpus is a collection of articles in The Wall Street Journal BID19 ).

The text is in English and its vocabulary consists of 10000 words.

We split the data into train, validation and test sets according to .

We train by feeding mini-batches of size N b that consist of sequences of T consecutive characters.

We incorporate RUM into the state-of-the-art high-level model: Fast-Slow RNN (FS-RNN, BID21 ).

The FS-RNN-k architecture consists of two hierarchical layers: one of them is a "fast" layer that connects k RNN cells F 1 , . . .

, F k in series; the other is a "slow" layer that consists of a single RNN cell S. The organization is roughly as follows: F 1 receives the input from the mini-batch and feeds its state into S; S feeds its state into F 2 ; the output of F k is the probability distribution of the predicted character.

FORMULA5 1.24 -HyperLSTM (Ha et al. FORMULA5 1.219 14.4M NASCell (Zoph & V. Le FORMULA5 1.214 16.3M FS-LSTM-4 (Mujika et al. FORMULA5 1.193 6.5M FS-LSTM-2 (Mujika et al. FORMULA5 1.190 7.2M FS-RUM-2 (ours)1.189 11.2M TAB3 : With FS-RUM-2 we achieve the state-of-the-art test result on the Penn Treebank task.

FS-RUM-2 generalizes better than other gated models, such as GRU and LSTM, because it learns efficient patterns for activation in its kernels.

Such a skill is useful for the large Penn Treebank data set, as with its special diagonal structure, the RUM cell in FS-RUM-2 activates the hidden state effectively.

We discuss this representational advantage in section 5.1.

One advantage of the Rotational Unit of Memory is that it allows the model to encode information in the phase of the hidden state.

In order to demonstrate the structure behind such learning, we look at the kernels that generate the target memory τ in the RUM model.

FIG2 (a) is a visualization for the Recall task that demonstrates the diagonal structure of W FORMULA5 hh which generates τ (a diagonal structure is also present W (2) hh , but it is contrasted less).

One way to interpret the importance of the diagonal contrast is that each neuron in the hidden state plays an important role for learning since each element on the diagonal activates a distinct neuron.

Moreover, the diagonal structure is not task specific.

For example, in FIG2 (b) we observe a particular W (2) hh for the target τ on the Penn Treebank task.

The way we interpret the meaning of the diagonal structure, combined with the off-diagonal activations, is that probably they encode grammar and vocabulary, as well as the links between various components of language.

It is natural to view the Rotational Unit of Memory and many other approaches using orthogonal matrices to fall into the category of phase-encoding architectures: R = R(θ), where θ is a phase information matrix.

For instance, we can parameterize any orthogonal matrix according to the Efficient Unitary Neural Networks (EUNN, BID14 DISPLAYFORM0 , where U 0 is a block diagonal matrix containing N/2 numbers of 2-by-2 rotations.

The component θ i is an one-by-(N/2) parameter vector.

Therefore, the rotational memory equation in our model can be represented as where θ t are rotational memory phase vectors at time t and φ represents the phases generated by the operation Rotation correspondingly.

Note that each element of the matrix multiplication U 0 (θ i ) · U 0 (φ i ) only depends on one element from θ i and φ i each.

This means that, to cancel out one element θ i , the model only needs to learn to express φ i as the negation of θ i .

DISPLAYFORM1 As a result, our RNN implementation does not require a reset gate, as in GRU or GORU, because the forgetting mechanism is automatically embedded into the representation (2) of phase-encoding.

Thus, the concept of phase-encoding is simply a special sampling on manifolds generated by the special orthogonal Lie group SO(N ).

Now, let N = N h be the hidden size.

One way to extend the current RUM model is to allow for λ to be any real number in the associative memory equation DISPLAYFORM2 This will expand the representational power of the rotational unit.

The difficulty is to mathematically define the raising of a matrix to a real power, which is equivalent to defining a logarithm of a matrix.

Again, rotations prove to be a natural choice since they are elements of SO(N h ), and their logarithms correspond to elements of the vector space of the Lie algebra so(N h ), associatied to SO(N h ).

We want to clarify that RUM and Capsule Net are not equivalent in terms of learning representations, but they share notable spiritual similarities.

A parallel between RUMs state and Capsules representation.

The hidden state in our model is viewed as a vector in an Euclidean space R n -it has an orientation and a magnitude.

In a similar fashion, a capsule is a vector that has an orientation and a magnitude.

Both RUM and Capsule Net learn to manipulate the orientation and magnitude of their respective components.

The Rotation operation and the Routing mechanism.

Both mechanisms are ways of manipulating orientations and magnitudes.

In the routing mechanism we start from priors (linearly generated from the input to the given layer of capsules), then generate outputs, and finally measure the dot product between the priors and the output.

This dot product essentially measures the similarity between the two vectors through the cosine of the angle between them.

This relative position between the two vectors is used for effective routing, so that the orientations of the capsules can be manipulated iteratively.

For rotation mechanism, we start with the embedded input vector (an alternative of the priors) and then generate the target memory (an alternative of the outputs).

Then we measure (encode) the rotation between the embedded input and the target memory (an alternative of taking the dot product).

And finally we use that encoded rotation to change the orientation of the hidden state (the iterative process of the routing mechanism).Main Difference.

The hidden state in RUM usually has a much larger dimensionality than the capsules that are used in BID23 .

Hence, effectively, we demonstrate how to manipulate orientations and magnitudes of a much higher dimensionality (for example, we have experimented with hidden sizes of 1000 and 2000 for language modeling).

For future work, the RUM model can be applied to other higher-level RNN structures.

For instance, in section 4.4 we already showed how to successfully embed RUM into FS-RNN to achieve stateof-the-art results.

Other examples may include Recurrent Highway Networks (Zilly et al. (2017) ), HyperNetwork BID8 ) structures, etc.

The fusion of RUM with such architectures could lead to more state-of-the-art results in sequential tasks.

We proposed a novel RNN architecture: Rotational Unit of Memory.

The model takes advantage of the unitary and associative memory concepts.

RUM outperforms many previous state-of-the-art models, including LSTM, GRU, GORU and NTM in synthetic benchmarks: Copying Memory and Associative Recall tasks.

Additionally, RUM's performance in real-world tasks, such as question answering and language modeling, is competetive with that of advanced architectures, some of which include attention mechanisms.

We claim the Rotational Unit of Memory can serve as the new benchmark model that absorbs all advantages of existing models in a scalable way.

Indeed, the rotational operation can be applied to many other fields, not limited only to RNN, such as Convolutional and Generative Adversarial Neural Networks.

The alphabet of the input consists of symbols {a i }, i ∈ {0, 1, · · · , n − 1, n, n + 1}, the first n of which represent data for copying, and the remaining two form "blank" and "marker" symbols, respectively.

In our experiment n = 8 and the data for copying is the first 10 symbols of the input.

The expectation from the RNN model is to output "blank" and, after the "marker" appears in the input, to output (copy) sequentially the initial data of 10 steps.

The sequences for training are randomly generated, and consist of pairs of "character" and "number" elements.

We set the key to always be a "character".

We fix the size of the "character" set equal to half of the length of the sequence and the size of the "number" set equal to 10.

Therefore, the total category has a size of T /2 + 10 + 1.

The associative memory provided by rotational operation Rotation enables RUM to solve the Associative Recall Task.

The input sequences is 50 .

For all models N h = 50.

For the training of all models we use RMSProp optimization with a learning rate of 0.001 and a decay rate of 0.9; the batch size N b is 128.

We observe that it is necessary to tune in the associative memory via λ = 1 since λ = 0 RUM does not learn the task.

In this task, we train 20 models jointly on each sub-task.

All of them use a 10k data set, which is divided into 90% of training and 10% of validation.

We first tokenize all the words in the data set and combine the story and question by simply concatenating two sequences.

Different length sequences are filled with "blank" at the beginning and the end.

Words in the sequence are embedded into dense vectors and then fed into RNN in a sequential manner.

The RNN model outputs the answer prediction at the end of the question through a softmax layer.

We use batch size of 32 for all 20 subsets.

The model is trained with Adam Optimizer with a learning rate 0.001.

Each subset is trained with 20 epochs and no other regularization is applied.

For all RNN cells we apply layer normalization BID3 ) to the cells and to the LSTM gates and RUM's update gate and target memory, zoneout BID17 ) to the recurrent connections, and dropout BID26 ) to the FS-RNN.

For training we use Adam optimization BID16 ).

We apply gradient clipping with maximal norm of the gradients equal to 1.0.

TAB9 lists the hyper-parameters we use for our models.

We embed the inputs into a higher-dimensional space.

The output of each models passes through a softmax layer; then the probabilities are evaluated by a standard cross entropy loss function.

The bits-per-character (BPC) loss is simply the cross entropy with a binary logarithm.

Table 7 outlines the performance of all variances of RUM models.

BID21 achieve their record with FS-LSTM-2, by setting F 1,2 and S to LSTM.

The authors in the same paper suggest that the "slow" cell has the function of capturing long-term dependencies from the data.

Hence, it is natural to set S to be a RUM, given its memorization advantages.

In particular, we experiment with FS-RUM-2, for which S is a RUM and F 1,2 are LSTM, as shown in figure 5 .

Additionally, we compare the validation performance of derivative models of our baseline FS-RUM-2 model in figure 6 .Empirically, we discovered that a time normalization η = 1.0 works best for RUM when gradient clipping norm is also 1.0.

1.189 14.2M FS-RUM-2 (ours)

1.189 11.2M Table 7 : FS-RUM-2(B)+LR is the baseline FS-RUM-2 except that the learning rate equals 0.003.

FS-RUM-2(B)+(S)800-1100, 900-1200, 1000-1000 and 1000-1200 are derivative models of FS-RUM-2, which are defined in figure 6 , and improve the validation performance of the baseline FS-LSTM-2 model.

In figure 6 we also show the 0.002 improvement to the validation BPC loss, achieved by FS-RUM-2(B)+(S)800-1100.

2).

FS-RUM-2(B)+Norm is the same as FS-RUM-2(B) except that the time normalization norm η is 1.3.

FS-RUM-2(B)+(S)800-1100 is the same as FS-RUM-2(B) except that the fast cells' size is 800 and the slow cell's size is 1100.

FS-RUM-2(B)+(S)1000-1000 is the same as FS-RUM-2(B) except that the fast cells' size is 1000 and the slow cell's size is 1000.

FS-RUM-2(B)+(S)900-1200 is the same as FS-RUM-2(B) except that the fast cells' size is 900 and the slow cell's size is 1200.

@highlight

A novel RNN model which outperforms significantly the current frontier of models in a variety of sequential tasks.