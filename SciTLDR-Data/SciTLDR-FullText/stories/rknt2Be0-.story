One of the distinguishing aspects of human language is its compositionality, which allows us to describe complex environments with limited vocabulary.

Previously, it has been shown that neural network agents can learn to communicate in a highly structured, possibly compositional language based on disentangled input (e.g. hand- engineered features).

Humans, however, do not learn to communicate based on well-summarized features.

In this work, we train neural agents to simultaneously develop visual perception from raw image pixels, and learn to communicate with a sequence of discrete symbols.

The agents play an image description game where the image contains factors such as colors and shapes.

We train the agents using the obverter technique where an agent introspects to generate messages that maximize its own understanding.

Through qualitative analysis, visualization and a zero-shot test, we show that the agents can develop, out of raw image pixels, a language with compositional properties, given a proper pressure from the environment.

One of the key requirements for artificial general intelligence (AGI) to thrive in the real world is its ability to communicate with humans in natural language.

Natural language processing (NLP) has been an active field of research for a long time, and the introduction of deep learning BID18 enabled great progress in NLP tasks such as translation, image captioning, text generation and visual question answering Vinyals et al., 2015; BID13 BID10 Serban et al., 2016; BID19 BID0 .

However, training machines in a supervised manner with a large dataset has its limits when it comes to communication.

Supervised methods are effective for capturing statistical associations between discrete symbols (i.e. words, letters).

The essence of communication is more than just predicting the most likely word to come next; it is a means to coordinate with others and potentially achieve a common goal BID1 BID7 Wittgenstein, 1953 ).An alternative path to teaching machines the art of communication is to give them a specific task and encourage them to learn how to communicate on their own.

This approach will encourage the agents to use languages grounded to task-related entities as well as communicate with other agents, which is one of the ways humans learn to communicate BID5 .

Recently, there have been several notable works that demonstrated the emergence of communication between neural network agents.

Even though each work produced very interesting results of its own, in all cases, communication was either achieved with a single discrete symbol (as opposed to a sequence of discrete symbols) BID8 BID17 or via a continuous value (Sukhbaatar et al., 2016; BID12 .

Not only is human communication un-differentiable, but also using a single discrete symbol is quite far from natural language communication.

One of the key features of human language is its compositional nature; the meaning of a complex expression is determined by its structure and the meanings of its constituents BID9 .

More recently, BID22 and BID16 trained the agents to communicate in grounded, compositional language.

In both studies, however, inputs given to the agents were hand-engineered features (disentangled input) rather than raw perceptual signals that we receive as humans.

In this work, we train neural agents to simultaneously develop visual perception from raw image pixels, and learn to communicate with a sequence of discrete symbols.

Unlike previous works, our setup poses greater challenges to the agents since visual understanding and discrete communication have to be induced from scratch in parallel.

We place the agents in a two-person image description game, where images contain objects of various color and shape.

Inspired by the pioneering work of BID3 , we employ a communication philosophy named obverter to train the agents.

Having its root in the theory of mind (Premack & Woodruff, 1978) and human language development BID21 , the obverter technique motivates an agent to search over messages and generate the ones that maximize their own understanding.

The contribution of our work can be summarized as follows:??? We train artificial agents to learn to disentangle raw image pixels and communicate in compositional language at the same time.??? We describe how the obverter technique, a differentiable learning algorithm for discrete communication, could be employed in a communication game with raw visual input.??? We visualize how the agents are perceiving the images and show that they learn to disentangle color and shape without any explicit supervision other than the communication one.??? Experiment results suggest that the agents could develop, out of raw image input, a language with compositional properties, given a proper pressure from the environment (i.e. the image description game).Finally, while our exposition follows a multi-agent perspective, it is also possible to interpret our results in the single-agent setting.

We are effectively learning a neural network that is able to learn disentangled compositional representations of visual scenes, without any supervision.

Subject to the constraints imposed by their environment, our agents learn disentangled concepts, and how to compose these to form new concepts.

This is an important milestone in the path to AGI.

2.1 THE TWO-PERSON IMAGE DESCRIPTION GAME Speaker Listener ABACD 1Speaker Listener ABACD 0Figure 1:

The two-person image description game.

Speaker observes an image and generates a message (i.e. a sequence of discrete symbols).

The listener, after observing a separate image and the message, must correctly decide whether it is seeing the same object as the speaker (left side; output 1) or not (right side; output 0). (blue, red, white, gray, yellow, green, cyan, magenta) , and five shapes (box, sphere, cylinder, capsule, ellipsoid), giving us total 40 combinations.

We choose a straightforward image description game with two factors (color and shape) so that we can perform extensive analysis on the outcome confidently, based on full control of the experiment.

In a single round of the two-person image description game, one agent becomes the speaker and the other the listener.

The speaker is given a random image, and generates a message to describe it.

The listener is also given a random image, possibly the same image as the speaker's.

After hearing the message from the speaker, the listener must decide if it is seeing the same object as the speaker (Figure 1 ).

Note that an image is the raw pixels given to the agents, and an object is the thing described by the image.

Therefore two different images can depict the same object.

In each round the agents change roles of being the speaker and the listener.

We generated synthetic images using Mujoco physics simulator 1 .

The example images are shown in FIG0 .

Each image depicts a single object with a specific color and shape in 128??128 resolution.

There are eight colors (blue, red, white, gray, yellow, green, cyan, magenta) and five shapes (box, sphere, cylinder, capsule, ellipsoid), giving us 40 combinations.

We generated 100 variations for each of the 40 object type.

Note that the position of the object varies in each image, changing the object size and the orientation.

Therefore even if the speaker and the listener are given the same object type, the actual images are very likely to be different, preventing the agents from using pixel-specific information, rather than object-related information to win the game.

Figure 3: Agent model architecture.

The visual module processes the image, and the language module generates or consumes messages.

The decision module accepts embeddings from both modules and produces the output.

The solid arrows indicate modifying the output from the previous layer.

The dotted arrows indicate copying the output from the previous layer.

Aside from using disentangled input, another strong assumption made in previous works BID3 BID22 was that the agents had access to the true intention of the speaker.

In BID3 , the listener was trained to modify its RNN hidden vector as closely to the speaker's intention (meaning vector; please see TAB7 in Appendix A) as possible.

In BID22 , each agent had an auxiliary task to predict the goals of all other agents.

In both cases, the true meaning/goal vector was used to update the model parameters, exposing the disentangled information to the agents.

In order to relax this assumption and encourage the agents to develop communication with minimal guidance, our model uses no other signal than whether the listener made a correct decision.

Figure 3 depicts the agent model architecture.

We use a convolutional neural network followed by a fully-connected layer to process the image.

A single RNN, specifically the gated recurrent units (GRU) , is used for both generating and consuming messages (the message generation using the obverter strategy is described in the next section).

When consuming a message, the image embedding from the visual module and the message embedding from the language module are concatenated and processed by another fully-connected layers (i.e. decision module) with the sigmoid output??, 0 being "My (listener) image is different from the speaker's" and 1 being "My image is the same as the speaker's".

Further details of the model architecture (e.g. number of layers) are described in Appendix C.

Although our work is inspired by Batali (1998) (see Appendix A for the description of BID3 ), obverter technique is a general message generation philosophy used/discussed in a number of communication and language evolution studies BID11 Oliphant & Batali, 1997; Smith, 2001; BID14 , which has its root in the theory of mind.

Theory of mind (Premack & Woodruff, 1978) observes that a human has direct access only to one's own mind and not to the others'.

Therefore we typically assume that the mind of others is analogous to ours, and such assumption is reflected in the functional use of language BID5 .

For example, if we want to convey a piece of information to the listener 2 , it is best to speak in a way that maximizes the listener's understanding.

However, since we cannot directly observe the listener's state of mind, we cannot exactly solve this optimization problem.

Therefore we posit that the listener's mind operates in a similar manner as ours, and speak in a way that maximizes our understanding, thus approximately solving the optimization problem.

This is exactly what the obverter technique tries to achieve.

When an agent becomes the teacher (i.e. speaker), the model parameters are fixed.

The image is converted to an embedding via the visual module.

After initializing its RNN hidden layer to zeros, the teacher at each timestep evaluates?? for all possible symbols and selects the one that maximizes??.

The RNN hidden vector induced by the chosen symbol is used in the next timestep.

This is repeated until?? becomes bigger than the predefined threshold, or the maximum message length is reached (see Appendix D for algorithm).

Therefore the teacher, through introspection, greedily selects characters at each timestep to generate a message such that the consistency between the image and the message is as clear to itself as possible.

When an agent becomes the learner (i.e. listener), its parameters are updated via back-propagating the cross entropy loss between its output?? and the true label y. Therefore the agents must learn to communicate only from the true label indicating whether the teacher and the learner are seeing the same object.

We remind the reader that only the learner's RNN parameters are updated, and the teacher uses its fixed RNN.

Therefore an agent uses only one RNN for both speaking and listening, guaranteeing self-consistency (see Appendix B for a detailed comparison between the obverter technique and the RL-based approach).

Furthermore, because the teacher's parameters are fixed, message generation can easily be extended to be more exploratory.

Although in this work we deterministically selected a character in each timestep, one can, for example, sample characters proportionally to?? and still use gradient descent for training the agents.

Using a more exploratory message generation strategy could help us discover a more optimal communication language when dealing with complex tasks.

Another feature of the obverter technique is that it observes the principle of least effort (Zipf, 1949) .

Because the teacher stops generating symbols as soon as?? reaches the threshold, it does not waste any more effort trying to perfect the message.

The same principle was implemented in one way or another in previous works, such as choosing the shortest among the generated strings BID14 or imposing a small cost for generating a message BID22 .

During the early stages of research, we noticed that randomly sampling object pairs (one for the teacher, one for the learner) lead to agents focusing only on colors and ignoring shapes.

When the teacher's object is fixed, there are 40 (8 colors ?? 5 shapes) possibilities on the learner's side.

If the teacher only talks about the color of the object, the learner can correctly decide for 36 out of 40 possible object types.

The learner makes incorrect decisions only when the teacher and the learner are given objects of the same color but different shapes, resulting in 90% accuracy on average.

This is actually what we observed; the accuracy plateaued between 0.9 and 0.92 during the training, and the messages were more or less the same for objects with the same color.

Therefore when constructing a mini-batch of images, we set 25% to be the object pairs of the same color and shape, 30% the same shape but different colors, 20% the same color but different shapes.

The remaining 25% object pairs were picked randomly 3 .Vocabulary size (i.e. number of unique symbols) and the maximum message length were also influential to the final outcome.

We noticed that a larger vocabulary and a longer message length helped the agents achieve a high communication accuracy more easily.

But the resulting messages were more challenging to analyze for compositional patterns.

In all our experiments we used 5 and 20 respectively for the vocabulary size and the maximum message length, similar to what BID3 used.

This suggests that the environment plays as important, if not more, role as the model architecture in the emergence of complex communication as discussed by previous studies BID15 BID4 BID16 and should be a main consideration for future efforts.

Further details regarding hyperparameters are described in Appendix E.

In this section, we first study the convergence behavior during the training phase.

Then we analyze the language developed by the agents in terms of compositionality.

As stated in the introduction, in compositional language, the meaning of a complex expression is determined by its structure and the meanings of its constituents.

With this definition in mind, we focus on two aspects of the inter-agent communication to evaluate its compositional properties: the structure (i.e. grammar) of the communication, and zero-shot performance (i.e. generalizing to novel stimuli).

These two aspects, which are both necessary conditions for any language to be considered compositional, have been used by previous works to study the compositional nature of artificial communication BID3 BID22 BID16 .To evaluate the structure of the messages, we study the evolution of the communication as training proceeds, and try to derive a grammar for expressing colors and shapes.

To evaluate the zero-shot capabilities, we test if the agents can compose consistent messages for objects they have not seen during the training.

Moreover, we visualize the image embeddings from the visual modules of both agents to understand how they are recognizing colors and shapes, the results of which, for a better view of the figures, are provided in Appendix H.

Figure 4: Progress during the training (best seen in color). (Top) We plot the training accuracy, training loss, average message length and average message distinctness in each round. (Bottom) We plot the perplexities and the Jaccard similarity of the messages spoken by both agents in each round.

Note that the average message length and the perplexities are divided by 20 to match the y-axis range with other metrics.

Figure 4 shows the convergence behavior during the training.

Training accuracy was calculated by rounding the learner's sigmoid output by 0.5.

Message distinctness was calculated by dividing the number of unique messages in the mini-batch by the size of the mini-batch.

Ideally there should be, on average, 40 distinct messages in the mini-batch of 50 images, therefore giving us 0.8 distinctness.

Every 10 round, both agents were given the same 1, 000 randomly sampled images to generate 1, 000 message pairs.

Then perplexity was calculated for each object type and averaged, thus indicating the average number of distinct messages used by the agents to describe a single object type (note that perplexities in the plot was divided by 20).

Jaccard similarity between both agents' messages was also calculated for each object type and averaged.

At the beginning, the listener (i.e. learner) always decides it is not seeing the same object as the speaker, giving us 0.75 accuracy 4 .

But after 7, 000 rounds, accuracy starts to go beyond 0.9.

Loss is negatively correlated with accuracy until round 15, 000, where it starts to fluctuate.

Accuracy, however, remains high due to how accuracy is measured; by rounding the learner's output by 0.5.

Although we could occasionally observe some patterns in the messages when both accuracy and loss were high, a lower loss generally resulted in a clearer communication structure (i.e. grammar) and better zero-shot performance.

The loss fluctuation also indicates some instability in the training process, which is a potential direction for future work.

Message distinctness starts at near 0, indicating the agents are generating the same message for all object types.

After round 7, 000, where both message distinctness and message length reach their maximum, both start to decrease.

But message distinctness never goes as high as the ideal 0.8, meaning that the agents are occasionally using the same message for different object types, as will be shown in the following section.

Both perplexities and Jaccard similarity show seemingly meaningless fluctuation at early rounds.

After round 7, 000, perplexities and Jaccard similarity show negatively correlated behavior, meaning that not only is each agent using consistent messages to describe each object type, but also both agents are using very similar messages to describe each object type.

We found perplexity and Jaccard similarity to be an important indicator of the degree of the communication structure.

During rounds 7, 000 ??? 8, 000, performance was excellent in terms of loss and accuracy, but perplexity was high and Jaccard similarity low, indicating the agents were assigning incoherent strings to each object type just to win the game.

Similar behavior was observed in the early stages of language evolution simulation in BID14 where words represented some meanings but had no structure (i.e. protolanguage).

It seems that artificial communication acquires compositional properties after the emergence of protolanguage regardless of whether the input is entangled or disentangled.

We choose agents from different training rounds to highlight how the language becomes more structured over time.

TAB1 shows agents' messages in the beginning (round 40), when the training accuracy starts pushing beyond 90% (round 6, 940), when agents settle on a common language (round 16, 760).In round 40, both agents are respectively producing the same message for all object types as mentioned in section 3.1.

We might say the messages are structured, but considering that the listener always answers 0 in early rounds, we cannot say the agents are communicating.

In round 6, 940, which is roughly when the agents begin to communicate more efficiently, training accuracy is significantly higher than round 40.

However, perplexities show that both agents are assigning many names to a single object type (40-80 names depending on the object type), indicating that the agents are focusing on pixel-level differences between images of the same object type.

TAB1 shows, as an example, the messages used by both agents to describe the red sphere.

Due to high perplexity, it is difficult to capture the underlying grammar of the messages even with regular expression.

Furthermore, as Jaccard similarity indicates, both agents are generating completely different messages for the same object type.

In round 16, 760, as the perplexities and Jaccard similarity tell us, the agents came to share a very narrow set of names for each object type (1-4 names depending on the object type).

Moreover, the names of the same-colored objects and same-shaped objects clearly seem to follow a pattern.

Overall, each of the three phases (round 40, round 6,940, round 16,760) seem to represent the development of visual perception, learning to communicate, and emergence of structure.

We found the messages in round 16, 760 could be decomposed in a similar manner as Table 6 in Appendix A. The top of TAB2 shows a possible decomposition of the messages from round 16, 760 and the bottom shows the rules for each color and shape derived from the decomposition.

According to our analysis, the agents use the first part of the message (i.e. prefix) to specify a shape, and the second part (i.e. suffix) to specify a color.

However, they use two different strings to specify a shape.

For example, the agents use either aaaa or bbbbb to describe a box.

The strings used for specifying colors show slightly weaker regularity.

For example, red is always described by either the suffix c or suffix e, but magenta is described by the suffix bb, bd, and sometimes b or bc.?? used for gray objects represents deletion of the prefix a. Note that removing prefixes beyond their length causes the pattern to break.

For example, gray box, gray sphere and gray cylinder use the same????a to express the color, but gray capsule and gray ellipsoid use irregular suffixes.

Despite some irregularities and one exceptional case (cyan box), the messages provide strong evidence that the agents learned to properly recognize color and shape from raw pixel input (see Appendix H for studying what the visual module learned), mapped each color and shape to prefixes and suffixes, and are able to compose meaningful messages to describe a given image to one another.

Communication accuracy for each object type is described in Appendix F. Communication examples and their analysis are given in Appendix G.

If the agents have truly learned to compose a message that can be divided into a color part and a shape part, then they should be able to accurately describe an object they have not seen before, which is another necessary condition for a compositional language.

Therefore, we hold out five objects (the shaded cells in TAB4 ) from the dataset during the training and observe how agents describe five novel objects during the test phase.

The agents were chosen from round 19, 980, which showed a high accuracy (97.8%), low perplexities (1.48, 1.65) and a high Jaccard similarity (0.75).

TAB4 shows a potential decomposition of the messages used by the agents (original messages are described by TAB12 in Appendix I).

We can observe that there is clearly a structure in the communication, although some messages show somewhat weaker patterns compared to when the agents were trained with all object types ( even when we consider the effects ofb and??.

However, the messages describing the held-out object types show clear structure with the exception of yellow ellipsoid.

In order to assess the communication accuracy when held-out objects are involved, we conducted another test with the agents from round 19, 980.

Each held-out object was given to the speaker, the listener, or both.

In the first two cases, the held-out object was paired with all 40 object types and each pair was tested 10 times.

In the last case, the held-out object was tested against itself 10 times.

In all cases, the agents switched roles after 5 times.

Table 4 shows communication accuracies for each case.

We can see the agents can successfully communicate most of the time even when given novel objects.

The last column shows that the listener is not simply producing 0 to maximize its chance to win the game.

It is also notable that the objects described withoutb or?? show better performance in general.

We noticed the communication accuracy for held-out objects seems relatively weak considering the messages used to describe them strongly showed structure.

TAB4 .

This, however, results from the grammar (i.e. structure) being not as straightforward as TAB2 , especially with short messages (i.e. frequent use ofb and??).

The same tendency can be observed for non-held-out objects as described by the per-object communication accuracy Table 9 in Appendix J. Table 4 : Communication accuracy when agents were given objects not seen during the training.

From the grammar analysis in the previous section, we have shown that the emerged language strongly follows a well-defined grammar.

In the zero-shot test, the agents demonstrated that they can successfully describe novel object, although not perfectly, by also following a similar grammar.

Both are, as stated in the beginning of section 3, necessary conditions for any communication to be considered compositional.

Therefore we can safely conclude that the emerged language in this work possesses some qualifications to be considered compositional.

In this work, we used the obverter technique to train neural network agents to communicate in a two-person image description game.

Through qualitative analysis, visualization and the zero-shot test, we have shown that even though the agents receive raw perception in the form of image pixels, under the right environment pressures, the emerged language had properties consistent with the ones found in compositional languages.

As an evaluation strategy, we followed previous works and focused on assessing the necessary conditions of compositional languages.

However, the exact definition of compositional language is still somewhat debatable, and, to the best of our knowledge, there is no reliable way to mathematically quantify the degree of compositionality of an arbitrary language.

Therefore, in order to encourage active research and discussion among researchers in this domain, we propose for future work, a quantitatively measurable definition of compositionality.

We believe compositionality of a language is not binary (e.g. language A is compositional/not compositional), but a spectrum.

For example, human language has some aspects that are compositional (e.g., syntactic constructions, most morphological combinations) and some that are not (e.g., irregular verb tenses in English, character-level word composition).

It is also important to clearly define grounded language and compositional language.

If one agent says abc (eat red apple) and another says cba (apple red eat), and they both understand each other, are they speaking compositional language?

We believe such questions should be asked and addressed to shape the definition of compositionality.

In addition to the definition/evaluation of compositional languages, there are numerous directions of future work.

Observing the emergence of a compositional language among more than two agents is an apparent next step.

Designing an environment to motivate the agents to disentangle more than two factors is also an interesting direction.

Training agents to consider the context (i.e. pragmatics), such as giving each agent several images instead of one, is another exciting future work.

A EMERGENCE OF GRAMMAR, BID3 In BID3 , the author successfully trained neural agents to develop a structured (i.e. grammatical) language using disentangled meaning vectors as the input.

Using 10 subject vectors and 10 predicate vectors, all represented as explicit binary vectors, total 100 meaning vectors could be composed TAB7 ).

Each digit in the subject vector 5a serves a clear role, respectively representing speaker(sp), hearer(hr), other(ot), and plural(pl).

The predicate vector values, on the other hand, are randomly chosen so that each predicate vector will have three 1's and three 0's.

The combination of ten subject vectors and ten predicate vectors allows 100 meaning vectors.

The author used twenty neural agents for the experiment.

Each agent was implemented with the vanilla recurrent neural networks (RNN), where the hidden vector h's size was 10, same as the size of the meaning vector m in order to treat h as the agent's understanding of m. In each training round a single learner (i.e. listener) and ten teachers (i.e. speaker) were randomly chosen.

Each teacher, given all 100 m's in random order, generates a message s 5 for each m and sends it to the learner.

The messages are generated using the obverter techinque, which is described in Algorithm 1.

The learner is trained to minimize the mean squared error (MSE) between h (after consuming the s) and m. After the learner has learned from all ten teachers, the next round begins, repeating the process until the error goes below some threshold.

Algorithm 1: Message generation process used in BID3 .

DISPLAYFORM0 9 Append i to s; DISPLAYFORM1 Terminate;When the training was complete, the author was able to find strong patterns in the messages used by the agents ( Table 6 ).

Note that the messages using predicates tired, scared, sick and happy especially follow a very clear pattern.

Batali also conducted a zero-shot test where the agents were trained without the diagonal elements in Table 6 and tested with all 100 meaning vectors.

The agents were able to successfully communicate even when held-out meaning vectors were used, but the Table 6 : (Top) Messages used by a majority of the population for each of the given meanings.(Bottom) A potential analysis of the system in terms of a root plus modifications.

Italic symbols are used to specify predicates and roman symbols are used to specify subjects.

Messages in parentheses cannot be made to fit into this analysis.messages used for the held-out meaning vectors did not show as strong compositional patterns as the non-zero-shot case.

The obverter technique allows us to generate messages that encourage the agents to use a shared language, even a highly structured one, via using a single RNN for both speaking and listening.

This is quite different from other RL-based related works BID17 BID22 BID8 BID12 BID16 where each agent has separate components (e.g. two RNNs) for generating messages and consuming messages.

This is necessary typically because the message generation module and the message consumption module have different input/output requirements.

The message generation module accepts some input related to the task (e.g. goal description vector, question embedding, or image embedding) and generates discrete symbols.

The message consumption module, on the other hand, accepts discrete symbols (i.e. the message) and generates some output related to the task (e.g. some prediction or some action to take).

Therefore, when a neural agent speaks in the RL-based approach, its message generation process is completely separated from its own listening process, but tied to the listening process of another agent (i.e. listener) 6 .

This means an agent may not have internal consistency; what an agent speaks may not make sense to itself.

However, agents in the RL-based setting do converge on a common language because, during the training, the error signal flows from the listener to the speaker directly.

Obverter approach, on the other hand, requires that each agent has a single component for both message generation and message consumption.

This single component accepts discrete symbols and generates some output related to the task.

This guarantees internal consistency because an agent's message generation process is tied to its own message consumption process; it will only generate messages that make sense to itself.

In the obverter setting, the error signal does not flow between agents directly, but agents converge on a common language by taking turns to be the listener; the listener tries to understand what the speaker says, so that when the listener becomes the speaker, it can generate messages that make sense to itself and, at the same time, will be understood by the former speaker (now listener).The advantage of obverter approach over RL-based approach is that it is motivated by the theory of mind and more resembles the acquisition/development process of human language.

Having a single mechanism for both speaking and listening, and training oneself to be a good listener leads to the emergence of self-consistent, shared language.

However, obverter technique requires that all agents perform the same task, which means all agents must have identical model architectures.

This is because, during the message generation process, the speaker internally simulates what the listener will go though when it hears the message.

Therefore we cannot play an asymmetrical game such as where the speaker sees only one image and generates a message but the listener is given multiple images and must choose one after hearing the message.

RL-based approaches do not have this problem since there are separate modules for speaking and listening.

We believe obverter technique could be the better choice for certain tasks regarding human mind emulation.

But it certainly is not the tool for every occasion.

The RL-based approach is a robust tool for any general task that may or may not involve human-like communication.

We conclude this section with a possible future research direction that combines the strengths of both approaches to enable communication in more interesting and complicated tasks.

We used TensorFlow and the Sonnet library for all implementation.

We used an eight-layer convolutional neural network.

We used 32 filters with the kernel size 3 for every layer.

The strides were [2, 1, 1, 2, 1, 2, 1, 2] for each layer.

We used rectified linear unit (ReLU) as the activation function for every layer.

Batch normalization was used for every layer.

We did not use the bias parameters since we used Batch normalization.

For padding, we used the TensorFlow VALID padding option for every layer.

The fully connected layer that follows the convolutional neural network was of 256 dimensions, with ReLU as the activation function.

We used a single layer Gated Recurrent Units (GRU) to implement the language module.

The size of the hidden layer was 64.

We used a two-layer feedforward neural network.

The first layer reduces the dimensionality to 128 with ReLU as the activation function, then the second layer generates a scalar value with sigmoid as the activation function.

Both agents' model parameters are randomly initialized.

The training process consists of rounds where teacher/learner roles are changed, and each round consists of multiple games where learner's model parameters are updated.

In each game, the teacher, given a mini-batch of images, generates corresponding messages.

The learner, given a separate mini-batch of images and the messages from the teacher, decides whether it is seeing the same object type as the teacher.

Learner's model parameters are updated to minimize the cross entropy loss.

After playing a predefined number of games, we move on to the next round where two agents change their roles.

Algorithm 2: Message generation process used in our work.

Table 7 : Accuracy when each object type is given to the speaker.

DISPLAYFORM0 We conducted a separate test with the agents from round 16, 760 to assess the communication accuracy for each object type.

The agents were given 1, 600 total object pairs (40 ?? 40).

Each object pair was tested 10 times, where after 5 times the agents switched speaker/listener roles.

The average accuracy was 95.4%, and only 88 out of 1, 600 object pairs were communicated with accuracy lower than 0.8.

Table 7 describes the accuracy when each object type was given to the speaker.

We can observe that the accuracy is higher for objects that are described with less overlapping messages.

For example, yellow box is communicated with the accuracy of 98%, and it is described with aaaaaa, which is not used for any other object types.

Gray box, on the other hand, is communicated with accuracy 93%.

It is described with aaa, which is also used for yellow capsule and green sphere, both of which are communicated with low accuracies as well.

Figure 5 provides ten examples of communication when the speaker is given a blue box and the listener is given various object types.

The listener's belief (i.e. score) that it is seeing the same image as the speaker changes each time it consumes a symbol.

It is notable that most of the time the score jumps between 0 and 1, rather than gradually changing in between.

This is natural given that messages that differ by only a single character can mean different objects (e.g. blue box and blue cylinder).

This phenomenon can also be seen in human language.

For example, blue can and blue cat differ by a single alphabet, but the semantics are completely different.

Object types that are described by similar messages as blue box, such as blue cylinder and magenta box cause marginal confusion to the listener such that prediction scores for both objects are not complete zeros.

There are also cases where two completely different objects are described by the same message as mentioned in Section 3.1.

From TAB2 we can see that blue box and cyan cylinder are described by the same message bbbbbbb{b,d}, although the messages were composed using different rules.

Therefore the listener generates high scores for both objects, occasionally losing the game when the agents are given this specific object pair (1 out of 40 chance).

This can be seen as a side effect coming from the principle of least effort which motivates the agents to win the game most of the time while minimizing the effort to generate messages.

Section 3.2 provides strong evidence that the agents are properly recognizing the color and shape of an object.

In this section, we study the visual module of both agents to study how they are processing Table 9 : Accuracy when each object type is given to the speaker.

Shaded cells indicate the objects not seen during the training.

In the same manner as Appendix F, we conducted a separate test with the agents from round 19, 980 to assess the communication accuracy for each object type.

The agents were given 1, 600 total object pairs (40 ?? 40).

Each object pair was tested 10 times, where after 5 times the agents switched speaker/listener roles.

The average accuracy was 94.73%, and 103 out of 1, 600 object pairs were communicated with accuracy lower than 0.8.

Table 9 describes the accuracy when each object type was given the speaker.

Shaded cells indicate objects not seen during the training.

Here we can observe the same tendency as the one seen in Appendix F; the accuracy is higher for objects that are described with less overlapping messages.

Lets assume agent0 is aware of red circle, blue square and green triangle.

If agent0 came upon a blue circle for the first time and had to describe it to agent1, the efficient way would be to say blue circle.

But it could also say blue not square not triangle.

If agent1 had a similar knowledge as agent0 did, then both agents would have a successful communication.

However, it is debatable whether saying blue not square not triangle is as compositional as blue circle.

@highlight

We train neural network agents to develop a language with compositional properties from raw pixel input.