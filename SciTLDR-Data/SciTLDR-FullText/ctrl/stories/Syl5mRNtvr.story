In this paper, we propose a differentiable adversarial grammar model for future prediction.

The objective is to model a formal grammar in terms of differentiable functions and latent representations, so that their learning is possible through standard backpropagation.

Learning a formal grammar represented with latent terminals, non-terminals, and productions rules allows capturing sequential structures with multiple possibilities from data.



The adversarial grammar is designed so that it can learn stochastic production rules from the data distribution.

Being able to select multiple production rules leads to different predicted outcomes, thus efficiently modeling many plausible futures.

We confirm the benefit of the adversarial grammar on two diverse tasks: future 3D human pose prediction and future activity prediction.

For all settings, the proposed adversarial grammar outperforms the state-of-the-art approaches, being able to predict much more accurately and further in the future, than prior work.

Future prediction in videos is one of the most challenging visual tasks.

Being able to accurately predict future activities, human or object pose has many important implications, most notably for robot action planning.

Prediction is particularly hard because it is not a deterministic process as multiple potential 'futures' are possible, and in the case of human pose, predicting real-valued output vectors is further challenging.

Given these challenges, we address the long standing questions: how should the sequential dependencies in the data be modeled and how can multiple possible long-term future outcomes be predicted at any given time.

To address these challenges, we propose an adversarial grammar model for future prediction.

The model is a differentiable form of a regular grammar trained with adversarial sampling of various possible futures, which is able to output real-valued predictions (e.g., 3D human pose) or semantic prediction (e.g., activity classes).

Learning sequences of actions or other sequential processes with the imposed rules of a grammar is valuable, as it imposes temporal structural dependencies and captures relationships between states (e.g., activities).

At the same time, the use of adversarial sampling when learning the grammar rules is essential, as this adversarial process is able to produce multiple candidate future sequences that follow a similar distribution to sequences seen in the data.

More importantly, a traditional grammar will need to enumerate all possible rules (exponential growth in time) to learn multiple futures.

This adversarial stochastic sampling process allows for much more memory-efficient learning without enumeration.

Additionally, unlike other techniques for future generation (e.g., autoregressive RNNs), we show the adversarial grammar is able to learn long sequences, can handle multi-label settings, and predict much further into the future.

The proposed approach is driven entirely by the structure imposed from learning grammar rules and their relationships to the terminal symbols of the data and by the adversarial losses which help model the data distribution over long sequences.

To our knowledge this is the first approach of adversarial grammar learning and the first to be able to successfully produce multiple feasible long-term future predictions for high dimensional outputs.

The approach outperforms previous state-of-the-art methods, including RNN/LSTM and memory based methods.

We evaluate future prediction on high dimensional data and are able to predict much further in the future than prior work.

The proposed approach is also general -it is applied to diverse future prediction tasks: 3D human pose prediction and multi-class and multi-label activity forecasting, and on three challenging datasets: Charades, MultiTHUMOS, and Human3.6M.

Grammar models for visual data.

The notion of grammars in computational science was introduced by Chomsky (1956) for description of language, and has found a widespread use in natural language understanding.

In the domain of visual data, grammars are used to parse images of scenes (Zhu & Mumford, 2007; Zhao & Zhu, 2011; Han & Zhu, 2008) .

In their position paper, Zhu & Mumford (2007) present a comprehensive grammar-based language to describe images, and propose MCMC-based inference.

More recently, a recursive neural net based approach was applied to parse scenes by Socher et al. (2011) .

However, this work has no explicit representation of grammar.

In the context of temporal visual data, grammars have been applied to activity recognition and parsing (Moore & Essa, 2002; Ryoo & Aggarwal, 2006; Vo & Bobick, 2014; Pirsiavash & Ramanan, 2014) but not to prediction or generation.

Qi et al. (2017) used used traditional stochastic grammar to predict activities, but only within 3 seconds.

Generative models for sequences.

Generative Adversarial Networks (GANs) are a very powerful mechanism for data generation by an underlying learning of the data distribution through adversarial sampling (Goodfellow et al., 2014) .

GANs have been very popular for image generation tasks (Emily L Denton, 2015; Isola et al., 2017; Brock et al., 2019) .

Prior work on using GANs for improved sequences generation (Yu et al., 2017; Fedus et al., 2018; Hu et al., 2017) has also been successful.

Fraccaro et al. (2016) proposed a stochastic RNN which enables generation of different sequences from a given state.

Differentiable Rule Learning Previous approaches that address differentiable rule or grammar learning are most aligned to our work .

However, they can only handle rules with very small branching factors and have not been demonstrated in high dimensional output spaces.

Future pose prediction.

Previous approaches for human pose prediction (Fragkiadaki et al., 2015; Ionescu et al., 2014; Tang et al., 2018) are relatively scarce.

The dominant theme is the use of recurrent models (RNNs or GRUs/LSTMs) (Fragkiadaki et al., 2015; Martinez et al., 2017) .

Tang et al. (2018) use attention models specifically to target long-term predictions, up to 1 second in the future.

Jain et al. (2016) propose a structural RNN which learns the spatio-temporal relationship of pose joints.

The above models, contrary to ours, cannot deal with multi-modality and ambiguity in the predictions, and do not produce multiple futures.

These results are also only within short-term horizons and the produced sequences often 'interpolate' actual data examples.

Video Prediction.

Without providing an exhaustive survey on video prediction, we note that our approach is related to the video prediction literature (Finn et al., 2016; Denton & Fergus, 2018; Babaeizadeh et al., 2017) where adversarial formulations are also common (Lee et al., 2018) .

Overview and main insights.

Our approach is driven by learning the production rules of a grammar, with which we can learn the transitions between continuous events in time, for example 3D human pose or activity.

While an activity or action may be continuous, it can also spawn into many possible futures at different points, similarly to switching between rules in a grammar.

For example, an activity corresponding to 'walking' can turn into 'running' or continuing the 'walking' behaviour or change to 'stopping'.

These production rules are learned in a differentiable fashion with an adversarial mechanism which allows learning multiple candidate future sequences.

This enables robust future prediction, which, more importantly, can easily generate multiple realistic futures.

A formal regular grammar is represented as the tuple (N, ??, P, N 0 ) where N is a finite non-empty set of non-terminals, ?? is a finite set of terminals (or output symbols), P is a set of production rules, and N 0 is the starting non-terminal symbol, N 0 ??? N .

Productions rules in a regular grammar are of the form A ??? aB, A ??? b, and A ??? , where A, B ??? N , a, b ??? ??, and is the empty Figure 1 : Overview of the adversarial grammar model.

The initial non-terminal is produced by an encoder based on some observations.

The grammar then generates multiple possible sequences from the non-terminal.

The generated and real sequences are used to train the discriminator.

string.

Applying multiple productions rules to the starting non-terminal generates a sequence of terminals.

Note that we only implement rules of form A ??? aB in our grammar, allowing it to generate sequences infinitely.

Our objective is to learn such non-terminals (e.g., A) and terminals (e.g., a) as latent representations directly from training data, and model the production rules P as a (differentiable) generative neural network function.

That is, at the heart of the proposed method is learning nonlinear function G : N ??? {(N, ??)} that maps a non-terminal to a set of (non-terminal, terminal) pairs.

We denote each element (i.e., each production rule) derived from the input non-terminal as {(A i , t i )}.

Note that this mapping to multiple possible elements enables modeling of multiple, different sequences, and is not done by existing models (e.g., RNNs).

For any latent non-terminal A ??? N , the grammar production rules are generated by applying the function G, to A as (here G is a neural network with learnable parameters):

Each pair corresponds to a particular production rule for this non-terminal.

More specifically,

This function is applied recursively to obtain a number of output sequences, similar to prior recurrent methods (e.g., RNNs and LSTMs).

However, in RNNs, the learned state/memory is required to abstract multiple potential possibilities into a single representation, as the mapping from the state/memory representation to the next representation is deterministic.

As a result, when learning from sequential data with multiple possibilities, standard RNNs tend to learn states as a mixture of multiple sequences instead of learning more discriminative states.

By learning explicit production rules, our states lead to more salient and distinct predictions which can be exploited for learning long-term, complex output tasks with multiple possibilities, as shown later in the paper.

For example, suppose A is the non-terminal that encodes the activity for 'walking'.

An output of the rule A ??? walkingA will be able to generate a sequence of continual 'walking' behavior.

Additional rules, e.g., A ??? stoppingV , A ??? runningU , can be learned, allowing for the activity to switch to 'stopping' or 'running' (with the non-terminals V, U respectively learning to generate their corresponding potential futures).

Clearly, for high dimensional outputs, such as 3D human pose, the number and dimensionality of the non-terminals required will be larger.

We also note that the non-terminals act as a form of memory, capturing the current state with the Markov property.

To accomplish the above task, G has a special structure.

The model contains a number of nonterminals and terminals which are learned: |N | non-terminals of dimensionality D, and |??| terminals of dimensionality C (the latter naturally correspond to the number and dimensionality of the desired outputs).

G takes input of A ??? N , then using several nonlinear transformations (e.g., fully connected layers), maps A to a vector r corresponding to a set of rules: r = f R (A).

Here, r is a vector with the size |P | whose elements specify the probability of each rule given input non-terminal.

We learn |P | rules which are shared globally, but only a (learned) subset are selected for each non-terminal as the other rule probabilities would become zero.

This is conceptually similar to using memory with recurrent neural network methods (Yogatama et al., 2018) , but the main difference is that the rule vectors are used to build grammar-like rule structures which are more advantageous in explicitly modeling of temporal dependencies.

In order to generate multiple outputs, the candidate rules, r are followed by the Gumbel-Softmax function (Jang et al., 2017; Maddison et al., 2017) , which allows for stochastic selection of a rule.

This function is differentiable and samples a single rule from the candidate rules based on the learned rule probabilities.

These probabilities model the likelihood of each generated sequence.

Two nonlinear functions f T and f N are additionally learned, such that, given a rule r, output the resulting terminal and non-terminal: B = f N (r), t = f T (r).

These functions are both a sequence of fully-connected layers followed by a non-linear activation function (e.g., softmax or sigmoid depending on the task).

As a result,

The schematic of G is visualized in Figure 1 , more details on the functions are provided in the later sections.

The non-terminals and terminals are modeled as sets of high dimensional vectors with pre-specified size and are learned jointly with the rules (all are tunable parameters and naturally more complex datasets require larger capacity).

For example, for a simple C-class classification problem, the terminals are represented as C-dimensional vectors matching the one-hot encoding for each class.

Difference to stochastic RNNs Standard recurrent models have a deterministic state, given some input, while the grammar is able to generate multiple potential next non-terminals (i.e., states).

Stochastic RNNs (Fraccaro et al., 2016) address this by allowing the next state to be stochastically generated, but this is difficult to control, as the next state now depends on a random value.

In the grammar model, the next non-terminal is sampled randomly, but from a set of deterministic candidates.

By maintaining a set of deterministic candidates, the next state can be selected randomly or by some other method, giving more control over the generated sequences.

Learning the starting non-terminal.

Given an initial input data sequence (e.g., a short video or pose sequences), we learn to generate its corresponding starting non-terminal (i.e., root node).

This is used as input to G to generate a sequence of terminal symbols starting from the given nonterminal.

Concretely, given the initial input sequence X, a function s is learned which gives the predicted starting non-terminal N 0 = s(X).

Then the function G is applied recursively to obtain the possible sequences where j is an index in the sequence and i is one of the possible rules:

The function G generates a set of (non-terminal, terminal) pairs, which is applied recursively to the non-terminals, resulting in new rules and the next set of (non-terminal, terminal) pairs.

Note that in most cases, each rule generates a different non-terminal, thus sampling G many times will lead to a variety of generated sequences.

As a result, an exponential number of sequences will need to be generated during training, to cover the possible sequences.

For example, consider a branching factor of k rules per non-terminal with a sequence of length L. This results in k L terminals and nonterminals (e.g., for k = 2 we have ??? 1000 and for k = 3 ??? 60, 000).

Thus, enumerating all possible sequences is computationally prohibitive beyond k = 2.

Furthermore, this restricts the tasks that can be addressed to ones with lower dimensional outputs because of memory limits.

With k = 1 (i.e., no branching), this reduces to a standard RNN during training, unable to generate multiple possible future sequences (i.e., we observed that the rules for each non-terminals become the same).

We address this problem by using stochastic adversarial rule sampling.

Given the non-terminals, which effectively contain a number of potential 'futures', we learn an adversarial-based sampling, similar to GAN approaches (Goodfellow et al., 2014) , which learns to sample the most likely rules for the given input.

The use of a discriminator network allows the model to generate realistic sequences that may not match the ground truth without being penalized.

We use the function G, which is the function modeling the learned grammar described above, as the generator function and build an additional discriminator function D. Following standard GAN training, the discriminator function returns a binary prediction which discriminates examples from the data distribution vs. generated ones.

Note that the adversarial process is designed to ultimately generate terminals, i.e., the final output sequence for the model.

D is defined as:

More specifically, D is tasked with the prediction of p ??? {T rue, F alse} based on if the input sequence of terminals, t = t 0 t 1 t 2 . . .

t L , is from the data or not (L is the length of the sequence).

Note that our discriminator is also conditioned on the non-terminal sequence (n = n 0 n 1 n 2 . . .

n L ), thus the distribution on non-terminals is learned implicitly, as well.

The discriminator function D is implemented as follows: given an input non-terminal and terminal sequence, we apply several 1D convolutional layers to the terminals and non-terminals, then concatenate their representations followed by a fully-connected layer to produce the binary prediction.

(Note that we also tried a GRU/LSTM instead of 1D conv, and it did not making a difference).

The discriminator and generator (grammar) functions are trained to work jointly, as is standard in GANs training.

This constitutes the adversarial grammar.

The optimization objective is defined as:

where p data (x) is the real data distribution (i.e., sequences of actions or human pose) and G(z) is the generated sequence from an initial state based on a sequence of frames (X).

The sequences generated by G could be compared to the ground truth to compute a loss during training (e.g., maximum likelihood estimation), however, doing so requires enumerating many possibilities in order learn multiple, distinct possible sequences.

Without enumeration, the model converges to a mixture representing all possible sequences.

By using the adversarial training of G, the model is able to generate sequences that match the distribution observed in the dataset.

This allows for computationally feasible learning of longer, higher-dimensional sequences.

Architectures and implementation details.

The functions G, f N and f t , f R , mentioned above, are networks using several fully-connected layers, which depend on the task and dataset (specific details are provided in the supplemental material).

For pose, the function s is implemented as a two-layer GRU module (Cho et al., 2014) followed by a 1x1 convolutional layer with D N outputs to produce the starting non-terminal.

For activity prediction, s is implemented as two sequential temporal convolutional layers which produce the starting non-terminal.

The model is trained for 5000 iterations using gradient descent with momentum of 0.9 and the initial learning rate set to 0.1.

We follow the cosine learning rate decay schedule.

Our models were trained on a single V100 GPU.

We conduct experiments on two sets of problems for future prediction: future activity prediction and future 3D human pose prediction and three datasets.

Our experiments demonstrate strong performance of the proposed approach over the state-of-the-art and the ability to produce multiple future outcomes, to handle multi-label datasets, and to predict further in the future than prior work.

We first test the method for video activity anticipation, where the goal is to predict future activities at various time-horizons, using an initial video sequence as input.

We predict future activities up to 45 seconds in the future on well-established video understanding datasets MultiTHUMOS (Yeung et al., 2015) for multi-class prediction and Charades (Sigurdsson et al., 2016) which is a multi-class and multi-label prediction task.

We note that we predict much further into the future than prior approaches, that reported results within a second or several seconds (Yeung et al., 2015) .

Evaluation metric To evaluate the approaches, we use a standard evaluation metric: we predict the activities occurring T seconds in the future and compute the mean average precision (mAP) between the predictions and ground truth.

As the grammar model is able to generate multiple, different future sequences, we also report the maximum mAP over 10 different future predictions.

We compare the (Fragkiadaki et al., 2015; Martinez et al., 2017; Tang et al., 2018) predictions at 1, 2, 5, 10, 20, 30 and 45 seconds into the future.

As little work has explored longterm future activity prediction, we compare against four different baseline methods: (i) repeating the activity prediction of the last seen frame, (ii) using a fully connected layer to predict the next second (applied autoregressively), (iii) using a fully-connected layer to directly predict activities at various future times, and (iv) an LSTM applied autoregressively to future activity predictions.

MultiTHUMOS dataset.

The MultiTHUMOS dataset (Yeung et al., 2015) is a popular video understanding benchmark with 400 videos spanning about 30 hours of video and 65 action classes.

Table 1 shows activity prediction accuracy for the MultiTHUMOS dataset.

In the table, we denote our approach as 'Adversarial Grammar -max' but also report our approach when limited to generating a single outcome ('Adversarial Grammar'), to be consistent to previous methods which are not able to generate more than one outcome.

We also compare to grammar without adversarial learning.

As seen, our approach outperforms alternative methods including LSTMs.

We observe that the gap to other approaches widens further in the future:

3.9 mean accuracy for the LSTM vs 11.2 of ours at 45 seconds in the future, as these autoregressive approaches become noisy.

Due to the structure of the grammar model, we are able to generate better long-term predictions.

We also find that by predicting multiple futures and taking the max improves performance, confirming that the grammar model is generating different sequences, some of which more closely match the ground truth.

Charades dataset.

Charades (Sigurdsson et al., 2016 ) is a challenging video dataset containing longer-duration activities recorded in home environments.

Charades is a multi-label dataset in which multiple activities often co-occur.

We use it to demonstrate the ability to handle this complex data.

It consists of 9858 videos (7990 training, 1868 test) over 157 activity classes.

Table 2 shows the future activity prediction results for Charades.

Similarly, we observe that the adversarial grammar model provides more accurate future prediction than previous work, slightly outperformed by grammar only.

We note that Charades is more challenging than others on both recognition and prediction tasks, and that grammar only, while performing well here, is not feasible for high dimensional tasks.

Figure 2 shows a true sequence and several other sequences generated by the adversarial grammar.

As Charades contains many different possible sequences, generating multiple futures is beneficial.

We further evaluate the approach on forecasting 3D human pose, a high dimensional structuredoutput problem.

This is a challenging task (Jain et al., 2016; Fragkiadaki et al., 2015) but is of high importance, e.g., for motion planning in robotics.

It also showcases the use of the Adversarial Grammar, as using the standard grammar is not feasible.

Human 3.6M dataset.

We conduct experiments on a well established future pose prediction benchmark, the Human 3.6M dataset (Ionescu et al., 2014; Catalin Ionescu, 2011) , which has 3.6 million 3D human poses of 15 activities.

The goal is to predict the future 3D locations of 32 joints in the human body.

We use quaternions to represent each joint location, allowing for a more continuous joint representation space.

We also predict differences, rather than absolute positions, which we found leads to more stable learning.

Previous work demonstrated prediction results up to a second on this dataset.

This work can generate future sequences for longer horizons, 4 seconds in the future.

We compare against the state-of-the-art methods on the Human 3.6M benchmark (Fragkiadaki et al., 2015; Jain et al., 2016; Ionescu et al., 2014; Martinez et al., 2017; Tang et al., 2018) using the Mean Angle Error (MAE) metric as introduced by Jain et al. (2016) .

Table 3 shows average MAE for all (Fragkiadaki et al., 2015; Martinez et al., 2017; Tang et al., 2018 activities compared to the state-of-the-art methods and Table 4 shows results on several activities, consistent with the protocol in prior work.

As seen from the tables, our work outperforms all prior methods.

Furthermore, we are able to generate results at larger time horizons of four seconds in the future.

In Fig 3, we show some predicted future poses for several different activities, confirming the results reflect the characteristics of the actual behaviors.

In Fig. 4 , we show the grammar's ability to generate different sequences from a given starting state.

Here, given a starting state, we select different rules, which lead to different sequences corresponding to walking, eating or sitting.

We propose a novel differentiable adversarial grammar and apply it to several diverse future prediction and generation tasks.

Because of the structure we impose for learning grammar-like rules for sequences and learning in adversarial fashion, we are able to generate multiple sequences that follow the distribution seen in data.

Our work outperforms prior approaches on all tasks and is able to generate sequences much further in the future.

We plan to release the code.

Activity Prediction For activity prediction, the number of non-terminals (|N |) was set to 64, the number of terminals (|??|) was set to the number of classes in the dataset (e.g., 65 in MultiTHUMOS and 157 in Charades).

We used 4 rules for each non-terminal (a total of 256 rules).

G, f N and f t each used one fully connected layer with sizes matching the desired inputs/outputs.

s is implemented as a two sequential temporal convolutional layers with 512 channels.

3D Pose estimation For 3D pose, the number of non-terminals (|N |) was set to 1024, the number of terminals (|??|) was set to 1024, where each terminal has size of 128 (32 joints in 4D quaternion representation).

The number of rules was set to 2 per non-terminal (a total of 2048 rules).

G was composed of 2 fully connected layers, f N and f t each used three fully connected layers with sizes matching the desired inputs/outputs.

s was implemented as a 2-layer GRU using a representation size of 1024.

A.2 SUPPLEMENTAL RESULTS Table 5 provides results of our approach for future 3D human pose prediction for all activities in the Human3.6M dataset.

Figure 5 shows more examples of future predicted 3D pose at different timesteps.

<|TLDR|>

@highlight

We design a grammar that is learned in an adversarial setting and apply it to future prediction in video.