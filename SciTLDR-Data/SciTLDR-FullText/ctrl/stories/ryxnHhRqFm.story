End-to-end task-oriented dialogue is challenging since knowledge bases are usually large, dynamic and hard to incorporate into a learning framework.

We propose the global-to-local memory pointer (GLMP) networks to address this issue.

In our model, a global memory encoder and a local memory decoder are proposed to share external knowledge.

The encoder encodes dialogue history, modifies global contextual representation, and generates a global memory pointer.

The decoder first generates a sketch response with unfilled slots.

Next, it passes the global memory pointer to filter the external knowledge for relevant information, then instantiates the slots via the local memory pointers.

We empirically show that our model can improve copy accuracy and mitigate the common out-of-vocabulary problem.

As a result, GLMP is able to improve over the previous state-of-the-art models in both simulated bAbI Dialogue dataset and human-human Stanford Multi-domain Dialogue dataset on automatic and human evaluation.

Task-oriented dialogue systems aim to achieve specific user goals such as restaurant reservation or navigation inquiry within a limited dialogue turns via natural language.

Traditional pipeline solutions are composed of natural language understanding, dialogue management and natural language generation BID32 BID28 , where each module is designed separately and expensively.

In order to reduce human effort and scale up between domains, end-to-end dialogue systems, which input plain text and directly output system responses, have shown promising results based on recurrent neural networks BID10 BID14 and memory networks BID24 .

These approaches have the advantages that the dialogue states are latent without hand-crafted labels and eliminate the needs to model the dependencies between modules and interpret knowledge bases (KB) manually.

However, despite the improvement by modeling KB with memory network BID0 , end-to-end systems usually suffer from effectively incorporating external KB into the system response generation.

The main reason is that a large, dynamic KB is equal to a noisy input and hard to encode and decode, which makes the generation unstable.

Different from chit-chat scenario, this problem is especially harmful in task-oriented one, since the information in KB is usually the expected entities in the response.

For example, in TAB0 the driver will expect to get the correct address to the gas station other than a random place such as a hospital.

Therefore, pointer networks BID26 or copy mechanism BID8 ) is crucial to successfully generate system responses because directly copying essential words from the input source to the output not only reduces the generation difficulty, but it is also more like a human behavior.

For example, in TAB0 , when human want to reply others the Valero's address, they will need to "copy" the information from the table to their response as well.

Therefore, in the paper, we propose the global-to-local memory pointer (GLMP) networks, which is composed of a global memory encoder, a local memory decoder, and a shared external knowledge.

Unlike existing approaches with copy ability BID9 BID8 , which the only information passed to decoder is the encoder hidden states, our model shares the external knowledge and leverages the encoder and the external knowledge to learn a global memory pointer and global contextual representation.

Global memory pointer modifies the external knowledge by softly filtering words that are not necessary for copying.

Afterward, instead of generating system responses directly, the local memory decoder first uses a sketch RNN to obtain sketch responses without slot values but sketch tags, which can be considered as learning a latent dialogue management to generate dialogue action template.

Then the decoder generates local memory pointers to copy words from external knowledge and instantiate sketch tags.

We empirically show that GLMP can achieve superior performance using the combination of global and local memory pointers.

In simulated out-of-vocabulary (OOV) tasks in the bAbI dialogue dataset BID0 , GLMP achieves 92.0% per-response accuracy and surpasses existing end-to-end approaches by 7.5% in full dialogue.

In the human-human dialogue dataset , GLMP is able to surpass the previous state of the art on both automatic and human evaluation, which further confirms the effectiveness of our double pointers usage.

Our model 1 is composed of three parts: global memory encoder, external knowledge, and local memory decoder, as shown in FIG0 (a).

The dialogue history X = (x 1 , . . . , x n ) and the KB information B = (b 1 , . . . , b l ) are the input, and the system response Y = (y 1 , . . .

, y m ) is the expected output, where n, l, m are the corresponding lengths.

First, the global memory encoder uses a context RNN to encode dialogue history and writes its hidden states into the external knowledge.

Then the last hidden state is used to read the external knowledge and generate the global memory pointer at the same time.

On the other hand, during the decoding stage, the local memory decoder first generates sketch responses by a sketch RNN.

Then the global memory pointer and the sketch RNN hidden state are passed to the external knowledge as a filter and a query.

The local memory pointer returns from the external knowledge can copy text from the external knowledge to replace the sketch tags and obtain the final system response.

Our external knowledge contains the global contextual representation that is shared with the encoder and the decoder.

To incorporate external knowledge into a learning framework, end-to-end memory networks (MN) are used to store word-level information for both structural KB (KB memory) and temporal-dependent dialogue history (dialogue memory), as shown in FIG0 .

In addition, the MN is well-known for its multiple hop reasoning ability BID24 , which is appealing to strengthen copy mechanism.

Global contextual representation.

In the KB memory module, each element b i ∈ B is represented in the triplet format as (Subject, Relation, Object) structure, which is a common format used to represent KB nodes BID18 .

For example, the KB in the TAB0 will be denoted as {(Tom's house, distance, 3 miles), ..., (Starbucks, address, 792 Bedoin St)}. On the other hand, the dialogue context X is stored in the dialogue memory module, where the speaker and temporal encoding are included as in BID0 like a triplet format.

For instance, the first utterance from the driver in the TAB0 will be denoted as {($user, turn1, I), ($user, turn1, need) , ($user, turn1, gas)}. For the two memory modules, a bag-of-word representation is used as the memory embeddings.

During the inference time, we copy the object word once a memory position is pointed to, for example, 3 miles will be copied if the triplet (Toms house, distance, 3 miles) is selected.

We denote Object(.) function as getting the object word from a triplet.

Knowledge read and write.

Our external knowledge is composed of a set of trainable embedding matrices C = (C 1 , . . .

, C K+1 ), where C k ∈ R |V |×d emb , K is the maximum memory hop in the MN, |V | is the vocabulary size and d emb is the embedding dimension.

We denote memory in the external knowledge as M = [B; X] = (m 1 , . . .

, m n+l ), where m i is one of the triplet components mentioned.

To read the memory, the external knowledge needs a initial query vector q 1 .

Moreover, it can loop over K hops and computes the attention weights at each hop k using DISPLAYFORM0 where DISPLAYFORM1 is the embedding in i th memory position using the embedding matrix C k , q k is the query vector for hop k, and B(.) is the bag-of-word function.

Note that p k ∈ R n+l is a soft memory attention that decides the memory relevance with respect to the query vector.

Then, the model reads out the memory o k by the weighted sum over c k+1 and update the query vector q k+1 .

Formally, DISPLAYFORM2

In FIG1 (a), a context RNN is used to model the sequential dependency and encode the context X. Then the hidden states are written into the external knowledge as shown in FIG0 (b).

Afterward, the last encoder hidden state serves as the query to read the external knowledge and get two outputs, the global memory pointer and the memory readout.

Intuitively, since it is hard for MN architectures to model the dependencies between memories , which is a serious drawback especially in conversational related tasks, writing the hidden states to the external knowledge can provide sequential and contextualized information.

With meaningful representation, our pointers can correctly copy out words from external knowledge, and the common OOV challenge can be mitigated.

In addition, using the encoded dialogue context as a query can encourage our external knowledge to read out memory information related to the hidden dialogue states or user intention.

Moreover, the global memory pointer that learns a global memory distribution is passed to the decoder along with the encoded dialogue history and KB information.

Context RNN.

A bi-directional gated recurrent unit (GRU) BID2 ) is used to encode dialogue history into the hidden states H = (h e 1 , . . .

, h e 1 ), and the last hidden state h e n is used to query the external knowledge as the encoded dialogue history.

In addition, the hidden states H are written into the dialogue memory module in the external knowledge by summing up the original memory representation with the corresponding hidden states.

In formula, c DISPLAYFORM0 Global memory pointer.

Global memory pointer G = (g 1 , . . . , g n+l ) is a vector containing real values between 0 and 1.

Unlike conventional attention mechanism that all the weights sum to one, each element in G is an independent probability.

We first query the external knowledge using h e n until the last hop, and instead of applying the Softmax function as in FORMULA0 , we perform an inner product followed by the Sigmoid function.

The memory distribution we obtained is the global memory pointer G, which is passed to the decoder.

To further strengthen the global pointing ability, we add an auxiliary loss to train the global memory pointer as a multi-label classification task.

We show in the ablation study that adding this additional supervision does improve the performance.

Lastly, the memory readout q K+1 is used as the encoded KB information.

In the auxiliary task, we define the label DISPLAYFORM1 by checking whether the object words in the memory exists in the expected system response Y .

Then the global memory pointer is trained using binary cross-entropy loss Loss g between G and G label .

In formula, DISPLAYFORM2

Given the encoded dialogue history h e n , the encoded KB information q K+1 , and the global memory pointer G, our local memory decoder first initializes its sketch RNN using the concatenation of h e n and q K+1 , and generates a sketch response that excludes slot values but includes the sketch tags.

For example, sketch RNN will generate "@poi is @distance away", instead of "Starbucks is 1 mile away."

At each decoding time step, the hidden state of the sketch RNN is used for two purposes: 1) predict the next token in vocabulary, which is the same as standard sequence-to-sequence (S2S) learning; 2) serve as the vector to query the external knowledge.

If a sketch tag is generated, the global memory pointer is passed to the external knowledge, and the expected output word will be picked up from the local memory pointer.

Otherwise, the output word is the word that generated by the sketch RNN.

For example in FIG1 (b), a @poi tag is generated at the first time step, therefore, the word Starbucks is picked up from the local memory pointer as the system output word.

DISPLAYFORM0 We use the standard cross-entropy loss to train the sketch RNN, we define Loss v as.

DISPLAYFORM1 We replace the slot values in Y into sketch tags based on the provided entity table.

The sketch tags ST are all the possible slot types that start with a special token, for example, @address stands for all the addresses and @distance stands for all the distance information.

Local memory pointer.

Local memory pointer L = (L 1 , . . . , L m ) contains a sequence of pointers.

At each time step t, the global memory pointer G first modify the global contextual representation using its attention weights, DISPLAYFORM2 and then the sketch RNN hidden state h d t queries the external knowledge.

The memory attention in the last hop is the corresponding local memory pointer L t , which is represented as the memory distribution at time step t. To train the local memory pointer, a supervision on top of the last hop memory attention in the external knowledge is added.

We first define the position label of local memory pointer L label at the decoding time step t as DISPLAYFORM3 The position n+l+1 is a null token in the memory that allows us to calculate loss function even if y t does not exist in the external knowledge.

Then, the loss between L and L label is defined as DISPLAYFORM4 Furthermore, a record R ∈ R n+l is utilized to prevent from copying same entities multiple times.

All the elements in R are initialized as 1 in the beginning.

During the decoding stage, if a memory position has been pointed to, its corresponding position in R will be masked out.

During the inference time,ŷ t is defined aŝ DISPLAYFORM5 where is the element-wise multiplication.

Lastly, all the parameters are jointly trained by minimizing the weighted-sum of three losses (α, β, γ are hyper-parameters): DISPLAYFORM6 3 EXPERIMENTS

We use two public multi-turn task-oriented dialogue datasets to evaluate our model: the bAbI dialogue BID0 and Stanford multi-domain dialogue (SMD) .

The bAbI dialogue includes five simulated tasks in the restaurant domain.

Task 1 to 4 are about calling API calls, modifying API calls, recommending options, and providing additional information, respectively.

Task 5 is the union of tasks 1-4.

There are two test sets for each task: one follows the same distribution as the training set and the other has OOV entity values.

On the other hand, SMD is a human-human, multi-domain dialogue dataset.

It has three distinct domains: calendar scheduling, weather information retrieval, and point-of-interest navigation.

The key difference between these two datasets is, the former has longer dialogue turns but the regular user and system behaviors, the latter has few conversational turns but variant responses, and the KB information is much more complicated.

The model is trained end-to-end using Adam optimizer BID12 , and learning rate annealing starts from 1e −3 to 1e −4 .

The number of hop K is set to 1,3,6 to compare the performance difference.

The weights α, β, γ summing up the three losses are set to 1.

All the embeddings are initialized randomly, and a simple greedy strategy is used without beam-search during the decoding stage.

The hyper-parameters such as hidden size and dropout rate are tuned with grid-search over Table 2 : Per-response accuracy and completion rate (in the parentheses) on bAbI dialogues.

GLMP achieves the least out-of-vocabulary performance drop.

Baselines are reported from Query Reduction Network BID20 , End-to-end Memory Network BID0 , Gated Memory Network BID15 , Point to Unknown Word BID9 , and Memory-to-Sequence .

DISPLAYFORM0 Ptr-Unk Mem2Seq GLMP K1 GLMP K3 GLMP K6 T1 99.4 (-) 99.9 (99.6) 100 FORMULA0 100 FORMULA0 100 FORMULA0 100 FORMULA0 100 FORMULA0 100 FORMULA0 100 (100) T2 99.5 (-) 100 FORMULA0 100 FORMULA0 100 FORMULA0 100 FORMULA0 100 FORMULA0 100 FORMULA0 100 FORMULA0 100 FORMULA0 the development set (per-response accuracy for bAbI Dialogue and BLEU score for the SMD).

In addition, to increase model generalization and simulate OOV setting, we randomly mask a small number of input source tokens into an unknown token.

The model is implemented in PyTorch and the hyper-parameters used for each task and the dataset statistics are reported in the Appendix.

bAbI Dialogue.

In Table 2 , we follow BID0 to compare the performance based on per-response accuracy and task-completion rate.

Note that for utterance retrieval methods, such as QRN, MN, and GMN, cannot correctly recommend options (T3) and provide additional information (T4), and a poor generalization ability is observed in OOV setting, which has around 30% performance difference in Task 5.

Although previous generation-based approaches (Ptr-Unk, Mem2Seq) have mitigated the gap by incorporating copy mechanism, the simplest cases such as generating and modifying API calls (T1, T2) still face a 6-17% OOV performance drop.

On the other hand, GLMP achieves a highest 92.0% task-completion rate in full dialogue task and surpasses other baselines by a big margin especially in the OOV setting.

No per-response accuracy loss for T1, T2, T4 using only the single hop, and only decreases 7-9% in task 5.Stanford Multi-domain Dialogue.

For human-human dialogue scenario, we follow previous dialogue works BID10 to evaluate our system on two automatic evaluation metrics, BLEU and entity F1 score 2 .

As shown in TAB3 , GLMP achieves a highest 14.79 BLEU and 59.97% entity F1 score, which is a slight improvement in BLEU but a huge gain in entity F1.

In fact, for unsupervised evaluation metrics in task-oriented dialogues, we argue that the entity F1 might be a more comprehensive evaluation metric than per-response accuracy or BLEU, as shown in that humans are able to choose the right entities but have very diversified responses.

Note that the results of rule-based and KVR are not directly comparable because they simplified the task by mapping the expression of entities to a canonical form using named entity recognition and linking 3 .

Moreover, human evaluation of the generated responses is reported.

We compare our work with previous state-of-the-art model Mem2Seq 4 and the original dataset responses as well.

We randomly select 200 different dialogue scenarios from the test set to evaluate three different responses.

Amazon Mechanical Turk is used to evaluate system appropriateness and human-likeness on a scale from 1 to 5.

As the results shown in TAB3 , we see that GLMP outperforms Mem2Seq in both measures, which is coherent to previous observation.

We also see that human performance on this assessment sets the upper bound on scores, as expected.

More details about the human evaluation are reported in the Appendix.

Ablation Study.

The contributions of the global memory pointer G and the memory writing of dialogue history H are shown in TAB4 .

We compare the results using GLMP with K = 1 in bAbI OOV setting and SMD.

GLMP without H means that the context RNN in the global memory encoder does not write the hidden states into the external knowledge.

As one can observe, our model without H has 5.3% more loss in the full dialogue task.

On the other hand, GLMP without G means that we do not use the global memory pointer to modify the external knowledge, and an 11.47% entity F1 drop can be observed in SMD dataset.

Note that a 0.4% increase can be observed in task 5, it suggests that the use of global memory pointer may impose too strong prior entity probability.

Even if we only report one experiment in the table, this OOV generalization problem can be mitigated by increasing the dropout ratio during training.

Visualization and Qualitative Evaluation.

Analyzing the attention weights has been frequently used to interpret deep learning models.

In Figure 3 , we show the attention vector in the last hop for each generation time step.

Y-axis is the external knowledge that we can copy, including the KB information and the dialogue history.

Based on the question "what is the address?" asked by the driver in the last turn, the gold answer and our generated response are on the top, and the global memory pointer G is shown in the left column.

One can observe that in the right column, the final memory pointer successfully copy the entity chevron in step 0 and its address 783 Arcadia Pl in step 3 to fill in the sketch utterance.

On the other hand, the memory attention without global weighting is reported in the middle column.

One can find that even if the attention weights focus on several point of interests and addresses in step 0 and step 3, the global memory pointer can mitigate the issue as expected.

More dialogue visualization and generated results including several negative examples and error analysis are reported in the Appendix.

Task-oriented dialogue systems.

Machine learning based dialogue systems are mainly explored by following two different approaches: modularized and end-to-end.

For the modularized systems BID29 BID28 , a set of modules for natural language understanding BID32 BID1 , dialogue state tracking BID13 BID35 , dialogue management BID23 , and natural language generation BID22 are used.

These approaches achieve good stability via combining domain-specific knowledge and slot-filling techniques, but additional human labels are needed.

On the other hand, end-to-end approaches have shown promising results recently.

Some works view the task as a next utterance retrieval problem, for examples, recurrent entity networks share parameters between RNN BID30 , query reduction networks modify query between layers BID20 , and memory networks BID0 BID15 perform multi-hop design to strengthen reasoning ability.

In addition, some approaches treat the task as a sequence generation problem.

BID14 Delexicalized Generation: @poi is at @address Final Generation: chevron is at 783_arcadia_pl Gold: 783_arcadia_pl is the address for chevron gas_station Figure 3 : Memory attention visualization in the SMD navigation domain.

Left column is the global memory pointer G, middle column is the memory pointer without global weighting, and the right column is the final memory pointer.

these approaches can encourage more flexible and diverse system responses by generating utterances token-by-token.

Pointer network.

BID26 uses attention as a pointer to select a member of the input source as the output.

Such copy mechanisms have also been used in other natural language processing tasks, such as question answering BID3 BID10 , neural machine translation BID9 BID8 , language modeling BID17 , and text summarization BID19 .

In task-oriented dialogue tasks, first demonstrated the potential of the copy-augmented Seq2Seq model, which shows that generationbased methods with simple copy strategy can surpass retrieval-based ones.

Later, augmented the vocabulary distribution by concatenating KB attention, which at the same time increases the output dimension.

Recently, combines end-to-end memory network into sequence generation, which shows that the multi-hop mechanism in MN can be utilized to improve copy attention.

These models outperform utterance retrieval methods by copying relevant entities from the KBs.

Others.

BID10 proposes entity indexing and introduces recorded delexicalization to simplify the problem by record entity tables manually.

In addition, our approach utilized recurrent structures to query external memory can be viewed as the memory controller in Memory augmented neural networks (MANN) BID6 .

Similarly, memory encoders have been used in neural machine translation BID27 and meta-learning applications .

However, different from other models that use a single matrix representation for reading and writing, GLMP leverages end-to-end memory networks to perform multiple hop attention, which is similar to the stacking self-attention strategy in the Transformer BID25 .

In the work, we present an end-to-end trainable model called global-to-local memory pointer networks for task-oriented dialogues.

The global memory encoder and the local memory decoder are designed to incorporate the shared external knowledge into the learning framework.

We empirically show that the global and the local memory pointer are able to effectively produce system responses even in the out-of-vocabulary scenario, and visualize how global memory pointer helps as well.

As a result, our model achieves state-of-the-art results in both the simulated and the human-human dialogue datasets, and holds potential for extending to other tasks such as question answering and text summarization.

A.1 TRAINING PARAMETERS Table 5 : Selected hyper-parameters in each dataset for different hops.

The values is the embedding dimension and the GRU hidden size, and the values between parenthesis is the dropout rate.

For all the models we used learning rate equal to 0.001, with a decay rate of 0.5.

T2 T3 T4 T5 SMD GLMP K1 64 FORMULA0 2) GLMP K3 64 FORMULA0 2) GLMP K6 64 FORMULA0 A.2 DATASET STATISTICS

For bAbI dialogues, the mistakes are mainly from task 3, which is recommending restaurants based on their rating from high to low.

We found that sometimes the system will keep sending those restaurants with the higher score even if the user rejected them in the previous turns.

On the other hand, SMD is more challenging for response generation.

First, we found that the model makes mistakes when the KB has several options corresponding to the user intention.

For example, once the user has more than one doctor appointment in the table, the model can barely recognize.

In addition, since we do not include the domain specific and user intention supervision, wrong delexicalized responses may be generated, which results in an incorrect entity copy.

Lastly, we found that the copied entities may not be matched to the generated sketch tags.

For example, an address tag may result in a distance entity copy.

We leave the space of improvement to future works.

One of the reviewers suggested us to compare our work to some existing dialogue framework such as PyDial 5 .

To the best of our knowledge, in the PyDial framework, it requires to have the dialogue acts labels for the NLU module and the belief states labels for the belief tracker module.

The biggest challenge is we do not have such labels in the SMD and bAbI datasets.

Moreover, the semi tracker in PyDial is rule-based, which need to re-write rules whenever it encounters a new domain or new datasets.

Even its dialogue management module could be a learning solution like policy networks, the input of the policy network is still the hand-crafted state features and labels.

Therefore, without the rules and labels predefined in the NLU and belief tracker modules, PyDial could not learn a good policy network.

Truly speaking, based on the data we have (not very big size) and the current state-of-the-art machine learning algorithms and models, we believe that a well and carefully constructed task-oriented dialogue system using PyDial in a known domain using human rules (in NLU and Belief Tracker) with policy networks may outperform the end-to-end systems (more robust).

However, in this paper, without additional human labels and human rules, we want to explore the potential and the advantage of end-to-end systems.

Besides easy to train, for multi-domain cases, or even zero-shot domain cases, we believe end-to-end approaches will have better adaptability compared to any rule-based systems.

Delexicalized Generation: the nearest @poi_type is @poi , @distance away at @address Final Generation: the nearest grocery_store is willows_market , 3_miles away at 409_bollard_st

Gold:

we are 3_miles away from willows_market but there is a car_collision_nearby Figure 5 : Memory attention visualization from the SMD navigation domain.

Delexicalized Generation: the nearest @poi_type is @poi , @distance away at @address Final Generation: the nearest grocery_store is sigona_farmers_market , 4_miles away at 819_alma_st Gold: there are whole_foods 2_miles away and sigona_farmers_market 4_miles away where do we go ?

Delexicalized Generation: okay , setting a reminder for your @event with @party @date at @time Final Generation: okay , setting a reminder for your dinner with marie 7pm at the_6th Gold: i will set a reminder for dinner at 7pm , for the_6th of this month with marie Figure 9 : Memory attention visualization from the SMD schedule domain.

Delexicalized Generation: your @event is on @date at @time Final Generation: your tennis_activity is on the_4th at 5pm Gold: your tennis_activity is on the_4th at 5pm and your sister will be attending Delexicalized Generation:

your next @event is on @date at @time Final Generation: your next tennis_activity is on the_13th at 6pm Gold: your next tennis_activity is on the_4th at 5pm with your mother Delexicalized Generation: your @event is on @date at @time with @party Final Generation: your doctor is on the_5th at 6pm with alex Gold: are you talking about the doctor_appointment on wednesday or the one on the_5th ?

Delexicalized Generation: it will not be @weather_attribute in @location @weekly_time Final Generation: it will not be drizzle in redwood_city weekend Gold: there will be no drizzle in redwood_city this weekend Delexicalized Generation: there will be @weather_attribute in @location on @date Final Generation: there will be clear_skies in danville on thursday Gold: dew is predicted in danville on thursday FIG0 : Memory attention visualization from the SMD weather domain.

<|TLDR|>

@highlight

GLMP: Global memory encoder (context RNN, global pointer) and local memory decoder (sketch RNN, local pointer) that share external knowledge (MemNN) are proposed to strengthen response generation in task-oriented dialogue.