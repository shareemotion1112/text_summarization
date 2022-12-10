Memory Network based models have shown a remarkable progress on the task of relational reasoning.

Recently, a simpler yet powerful neural network module called Relation Network (RN) has been introduced.

Despite its architectural simplicity, the time complexity of relation network grows quadratically with data, hence limiting its application to tasks with a large-scaled memory.

We introduce Related Memory Network, an end-to-end neural network architecture exploiting both memory network and relation network structures.

We follow memory network's four components while each component operates similar to the relation network without taking a pair of objects.

As a result, our model is as simple as RN but the computational complexity is reduced to linear time.

It achieves the state-of-the-art results in jointly trained bAbI-10k story-based question answering and  bAbI dialog dataset.

Neural network has made an enormous progress on the two major challenges in artificial intelligence: seeing and reading.

In both areas, embedding methods have served as the main vehicle to process and analyze text and image data for solving classification problems.

As for the task of logical reasoning, however, more complex and careful handling of features is called for.

A reasoning task requires the machine to answer a simple question upon the delivery of a series of sequential information.

For example, imagine that the machine is given the following three sentences: "Mary got the milk there.", "John moved to the bedroom.", and "Mary traveled to the hallway."

Once prompted with the question, "Where is the milk?", the machine then needs to sequentially focus on the two supporting sentences, "Mary got the milk there." and "Mary traveled to the hallway." in order to successfully determine that the milk is located in the hallway.

Inspired by this reasoning mechanism, J. has introduced the memory network (MemNN), which consists of an external memory and four components: input feature map (I), generalization (G), output feature map (O), and response (R).

The external memory enables the model to deal with a knowledge base without loss of information.

Input feature map embeds the incoming sentences.

Generalization updates old memories given the new input and output feature map finds relevant information from the memory.

Finally, response produces the final output.

Based on the memory network architecture, neural network based models like end-to-end memory network (MemN2N) BID11 , gated end-to-end memory network (GMemN2N) BID7 , dynamic memory network (DMN) BID6 , and dynamic memory network + (DMN+) BID13 are proposed.

Since strong reasoning ability depends on whether the model is able to sequentially catching the right supporting sentences that lead to the answer, the most important thing that discriminates those models is the way of constructing the output feature map.

As the output feature map becomes more complex, it is able to learn patterns for more complicate relations.

For example, MemN2N, which has the lowest performance among the four models, measures the relatedness between question and sentence by the inner product, while the best performing DMN+ uses inner product and absolute difference with two embedding matrices.

Recently, a new architecture called Relation Network (RN) BID9 has been proposed as a general solution to relational reasoning.

The design philosophy behind it is to directly capture the supporting relation between the sentences through the multi-layer perceptron (MLP).

Despite its simplicity, RN achieves better performance than previous models without any catastrophic failure.

The interesting thing we found is that RN can also be interpreted in terms of MemNN.

It is composed of O and R where each corresponds to MLP which focuses on the related pair and another MLP which infers the answer.

RN does not need to have G because it directly finds all the supporting sentences at once.

In this point of view, the significant component would be MLP-based output feature map.

As MLP is enough to recognize highly non-linear pattern, RN could find the proper relation better than previous models to answer the given question.

However, as RN considers a pair at a time unlike MemNN, the number of relations that RN learns is n 2 when the number of input sentence is n. When n is small, the cost of learning relation is reduced by n times compared to MemNN based models, which enables more data-efficient learning BID9 .

However, when n increases, the performance becomes worse than the previous models.

In this case, the pair-wise operation increases the number of non-related sentence pairs more than the related sentence pair, thereby confuses RN's learning.

BID9 has suggested attention mechanisms as a solution to filter out unimportant relations; however, since it interrupts the reasoning operation, it may not be the most optimal solution to the problem.

Our proposed model, "Relation Memory Network" (RMN), is able to find complex relation even when a lot of information is given.

It uses MLP to find out relevant information with a new generalization which simply erase the information already used.

In other words, RMN inherits RN's MLP-based output feature map on Memory Network architecture.

Experiments show its state-ofthe-art result on the text-based question answering tasks.

Relation Memory Network (RMN) is composed of four components -embedding, attention, updating, and reasoning.

It takes as the inputs a set of sentences x 1 , x 2 , ..., x n and its related question u, and outputs an answer a. Each of the x i , u, and a is made up of one-hot representation of words, for example, DISPLAYFORM0 .., n i ), V = vocabulary size, n i = number of words in sentence i).

We first embed words in each x i = {x i1 , x i2 , x i3 , ..., x ini } and u to a continuous space multiplying an embedding matrix A ∈ R d×V .

Then, the embedded sentence is stored and represented as a memory object m i while question is represented as q. Any of the following methods are available for embedding component: simple sum (equation 1), position encoding (J. (equation 2), concatenation (equation 3), LSTM, and GRU.

In case of LSTM or GRU, m i is the final hidden state of it.

DISPLAYFORM0 As the following attention component takes the concatenation of m i and q, it is not necessarily the case that sentence and question have the same dimensional embedding vectors unlike previous memory-augmented neural networks.

Attention component can be applied more than once depending on the problem; Figure 1 illustrates 2 hop version of RMN.

We refer to the i th embedded sentence on the t th hop as m DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2

To forget the information already used, we use intuitive updating component to renew the memory.

It is replaced by the amount of unconsumed from the old one: DISPLAYFORM0 Contrary to other components, updating is not a mandatory component.

When it is considered to have 1 hop, there is no need to use this.

Similar to attention component, reasoning component is also made up of MLP, represented as f φ .

It receives both q and the final result of attention component r f and then takes a softmax to produce the model answerâ: DISPLAYFORM0 3 RELATED WORK

To answer the question from a given set of facts, the model needs to memorize these facts from the past.

Long short term memory (LSTM) BID2 , one of the variants of recurrent neural network (RNN), is inept at remembering past stories because of their small internal memory BID11 .

To cope with this problem, J. has DISPLAYFORM0 proposed a new class of memory-augmented model called Memory Network (MemNN).

MemNN comprises an external memory m and four components: input feature map (I), generalization (G), output feature map (O), and response (R).

I encodes the sentences which are stored in memory m. G updates the memory, whereas O reads output feature o from the memory.

Finally, R infers an answer from o.

MemN2N, GMemN2N, DMN, and DMN+ all follow the same structure of MemNN from a broad perspective, however, output feature map is composed in slightly different way.

The relation between question and supporting sentences is realized from its cooperation.

MemN2N first calculates the relatedness of sentences in the question and memory by taking the inner product, and the sentence with the highest relatedness is selected as the first supporting sentence for the given question.

The first supporting sentence is then added with the question and repeat the same operation with the updated memory to find the second supporting sentence.

GMemN2N selects the supporting sentence in the same way as MemN2N, but uses the gate to selectively add the the question to control the influence of the question information in finding the supporting sentence in the next step.

DMN and DMN + use output feature map based on various relatedness such as absolute difference, as well as inner product, to understand the relation between sentence and question at various points.

The more difficult the task, the more complex the output feature map and the generalization component to get the correct answer.

For a dataset experimenting the text-based reasoning ability of the model, the overall accuracy could be increased in order of MemN2N, GMemN2N, DMN, and DMN+, where the complexity of the component increases.

Relation Network (RN) has emerged as a new and simpler framework for solving the general reasoning problem.

RN takes in a pair of objects as its input and simply learns from the compositions of two MLPs represented as g θ and f φ .

The role of each MLP is not clearly defined in the original paper, but from the view of MemNN, it can be understood that g θ corresponds to O and f φ corresponds to R. TAB1 summarizes the interpretation of RN compared to MemN2N and our model, RMN.To verify the role of g θ , we compare the output when pairs are made with supporting sentences and when made with unrelated sentences.

FIG2 shows the visualization result of each output.

When we focus on whether the value is activated or not, we can see that g θ distinguishes supporting sentence pair from non-supporting sentence pair as output feature map examines how relevant the sentence is to the question.

Therefore, we can comprehend the output of g θ reveals the relation between the object pair and the question and f φ aggregates all these outputs to infer the answer.

bAbI story-based QA dataset bAbI story-based QA dataset is composed of 20 different types of tasks for testing natural language reasoning ability.

Each task requires different methods to infer the answer.

The dataset includes a set of statements comprised of multiple sentences, a question and answer.

A statement can be as short as two sentences and as long as 320 sentences.

To answer the question, it is necessary to find relevant one or more sentences to a given question and derive answer from them.

Answer is typically a single word but in a few tasks, answers are a set of words.

Each task is regarded as success when the accuracy is greater than 95%.

There are two versions of this dataset, one that has 1k training examples and the other with 10k examples.

Most of the previous models test their accuracy on 10k dataset with trained jointly.bAbI dialog dataset bAbI dialog dataset BID0 ) is a set of 5 tasks within the goal-oriented context of restaurant reservation.

It is designed to test if model can learn various abilities such as performing dialog management, querying knowledge bases (KBs), and interpreting the output of such queries.

The KB can be queried using API calls and 4 fields (a type of cuisine, a location, a price range, and a party size).

They should be filled to issue an API call.

Task 1 tests the capacity of interpreting a request and asking the right questions to issue an API call.

Task 2 checks the ability to modify an API call.

Task 3 and 4 test the capacity of using outputs from an API call to propose options in the order of rating and to provide extra-information of what user asks for.

Task 5 combines everything.

The maximum length of the dialog for each task is different: 14 for task 1, 20 for task 2, 78 for task 3, 13 for task 4, and 96 for task 5.

As restaurant name, locations, and cuisine types always face new entities, there are normal and OOV test sets to assess model's generalization ability.

Training sets consist fo 1k examples, which is not a large amount of creating realistic learning conditions.

bAbI story-based QA dataset We trained 2 hop RMN jointly on all tasks using 10k dataset for model to infer the solution suited to each type of tasks.

We limited the input to the last 70 stories for all tasks except task 3 for which we limited input to the last 130 stories, similar to BID13 which is the hardest condition among previous models.

Then, we labeled each sentence with its relative position.

Embedding component is similar to BID9 , where story and question are embedded through different LSTMs; 32 unit word-lookup embeddings; 32 unit LSTM for story and question.

For attention component, as we use 2 hop RMN, there are g 1 θ and g 2 θ ; both are three-layer MLP consisting of 256, 128, 1 unit with ReLU activation function BID8 .

f φ is composed of 512, 512, and 159 units (the number of words appearing in bAbI dataset is 159) of three-layer MLP with ReLU non-linearities where the final layer was a linear that produced logits for a softmax over the answer vocabulary.

For regularization, we use batch normalization BID3 for all MLPs.

The softmax output was optimized with a cross-entropy loss function using the Adam optimizer BID5 ) with a learning rate of 2e −4 .bAbI dialog dataset We trained on full dialog scripts with every model response as answer, all previous dialog history as sentences to be memorized, and the last user utterance as question.

Model selects the most probable response from 4,212 candidates which are ranked from a set of all bot utterances appearing in training, validation and test sets (plain and OOV) for all tasks combined.

We also report results when we use match type features for dialog.

Match type feature is an additional label on the candidates indicating if word is found on the dialog history.

For example, if the world 'Seoul' is found, then the 'location' field is checked to hint model this word is important and should be used in API call.

This feature can alleviate OOV problem.

Training was done with Adam optimizer and a learning rate of 1e −4 for all tasks.

Additional model details are given in Appendix A. concentrate on the same sentences which are all critical to answer the question, and sometimes g 1 θ θ chooses the key fact to answer.

While trained jointly, RMN learns these different solutions for each task.

For the task 3, the only failed task, attention component still functions well; it focuses sequentially on the supporting sentences.

However, the reasoning component, f φ , had difficulty catching the word 'before'.

We could easily figure out 'before' implies 'just before' the certain situation, whereas RMN confused its meaning.

As shown in table 3c, our model found all previous locations before the garden.

Still, it is remarkable that the simple MLP carried out all of these various roles.

The results in the TAB6 show that the RMN has the best results in any conditions.

Without any match type, RN and RMN outperform previous memory-augmented models on both normal and OOV tasks.

This is mainly attributed to the impressive result on task 4 which can be interpreted as an effect of MLP based output feature map.

To solve task 4, it is critical to understand the relation between 'phone number' of user input and 'r phone' of previous dialog as shown in TAB10 .

We assumed that inner product was not sufficient to capture their implicit similarity and performed an supporting experiment.

We converted RMN's attention component to inner product based attention, and the results revealed the error rate increased to 11.3%.For the task 3 and task 5 where the maximum length is especially longer than the others, RN performs worse than MemN2N, GMemN2N and RMN.

The number of unnecessary object pairs created by the RN not only increases the processing time but also decreases the accuracy.

With the match type feature, all models other than RMN have significantly improved their performance except for task 3 compared to the plain condition.

RMN was helped by the match type only on the OOV tasks and this implies RMN is able to find relation in the With Match condition for the normal tasks.

When we look at the OOV tasks more precisely, RMN failed to perform well on the OOV task 1 and 2 even though g 1 θ properly focused on the related object as shown in TAB10 .

We state that this originated from the fact that the number of keywords in task 1 and 2 is bigger than that in task 4.

In task 1 and 2, all four keywords (cuisine, location, number and price) must be correctly aligned from the supporting sentence in order to make the correct API call which is harder than task 4.

Consider the example in TAB10 .

Supporting sentence of task 4 have one keyword out of three words, whereas supporting sentences of task 1 and 2 consist of four keywords (cuisine, location, number and price) out of sixteen words.

Different from other tasks, RMN yields the same error rate 25.1% with MemN2N and GMemN2N on the task 3.

The main goal of task 3 is to recommend restaurant from knowledge base in the order of rating.

All failed cases are displaying restaurant where the user input is <silence>which is somewhat an ambiguous trigger to find the input relevant previous utterance.

As shown in TAB10 , there are two different types of response to the same user input.

One is to check whether all the required fields are given from the previous utterances and then ask user for the missing fields or send a "Ok let me look into some options for you." message.

The other type is to recommend restaurant starting from the highest rating.

All models show lack of ability to discriminate these two types of silences so that concluded to the same results.

To verify our statement, we performed an additional experiment on task 3 and checked the performance gain (extra result is given in TAB1 of Appendix B).

Effectiveness of the MLP-based output feature map The most important feature that distinguishes MemNN based models is the output feature map.

TAB7 summarizes the experimental results for the bAbI story-based QA dataset when replacing the RMN's MLP-based output feature map with the idea of the previous models.

inner product was used in MemN2N, inner product with gate was used in GMemN2N, and inner product and absolute difference with two embedding matrices was used in DMN and DMN+.

From the TAB7 , the more complex the output feature map, the better the overall performance.

In this point of view, MLP is the effective output feature map.

Performance of RN and RMN according to memory size Additional experiments were conducted with the bAbI story-based QA dataset to see how memory size affects both performance and training time of RN and RMN.

Test errors with training time written in parentheses are summarized in TAB8 .When memory size is small, we could observe the data-effeciency of RN.

It shows similar performance to RMN in less time.

However, when the memory size increases, performance is significantly reduced compared to RMN, even though it has been learned for a longer time.

It is even lower than itself when the memory size is 20.

On the other hand, RMN maintains high performance even when the memory size increases.

Effectiveness of the number of hops bAbI story based QA dataset differs in the number of supporting sentences by each task that need to be referenced to solve problems.

For example, task 1, 2, and 3 require single, two, and three supporting facts, respectively.

The result of the mean error rate for each task according to the number of hops is in TAB9 .Overall, the number of hops is correlated with the number of supporting sentences.

In this respect, when the number of relations increases, RMN could reason across increasing the number of hops to 3, 4 or more.

Our work, RMN, is a simple and powerful architecture that effectively handles text-based question answering tasks when large size of memory and high reasoning ability is required.

Multiple access to the external memory to find out necessary information through a multi-hop approach is similar to most existing approaches.

However, by using a MLP that can effectively deal with complex relatedness when searching for the right supporting sentences among a lot of sentences, RMN raised the state-of-the-art performance on the story-based QA and goal-oriented dialog dataset.

When comparing RN which also used MLP to understand relations, RMN was more effective in the case of large memory.

Future work will apply RMN to image based reasoning task (e.g., CLEVR, DAQUAR, VQA etc.).To extract features from the image, VGG net BID10 ) is used in convention and outputs 196 objects of 512 dimensional vectors which also require large sized memory.

An important direction will be to find an appropriate way to focus sequentially on related object which was rather easy in text-based reasoning.

A MODEL DETAILS We modify the user input from <silence>to <silence><silence>when looking for restaurant recommendations.

This makes model to distinguish two different situations whether to ask for additional fields or to recommend restaurant.

<|TLDR|>

@highlight

A simple reasoning architecture based on the memory network (MemNN) and relation network (RN), reducing the time complexity compared to the RN and achieving state-of-the-are result on bAbI story based QA and bAbI dialog.

@highlight

Introduces Related Memory Network (RMN), an improvement over Relationship Networks (RN).