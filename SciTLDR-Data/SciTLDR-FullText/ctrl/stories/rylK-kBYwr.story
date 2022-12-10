It has been an open research challenge for developing an end-to-end multi-domain task-oriented dialogue system, in which a human can converse with the dialogue agent to complete tasks in more than one domain.

First, tracking belief states of multi-domain dialogues is difficult as the dialogue agent must obtain the complete belief states from all relevant domains, each of which can have shared slots common among domains as well as unique slots specifically for the domain only.

Second, the dialogue agent must also process various types of information, including contextual information from dialogue context, decoded dialogue states of current dialogue turn, and queried results from a knowledge base, to semantically shape context-aware and task-specific responses to human.

To address these challenges, we propose an end-to-end neural architecture for task-oriented dialogues in multiple domains.

We propose a novel Multi-level Neural Belief Tracker which tracks the dialogue belief states by learning signals at both slot and domain level independently.

The representations are combined in a Late Fusion approach to form joint feature vectors of (domain, slot) pairs.

Following recent work in end-to-end dialogue systems, we incorporate the belief tracker with generation components to address end-to-end dialogue tasks.

We achieve state-of-the-art performance on the MultiWOZ2.1 benchmark with 50.91% joint goal accuracy and competitive measures in task-completion and response generation.

In a task-oriented dialogue system, the Dialogue State Tracking (DST) module is responsible for updating dialogue states (essentially, what the user wants) at each dialogue turn.

The DST supports the dialogue agent to steer the conversation towards task completion.

As defined by Henderson et al. (2014a) , a dialogue belief state consists of inform slots -information to query a given knowledge base or database (DB), and request slots -information to be returned to the users.

Task-oriented dialogues can be categorized as either single-domain or multi-domain dialogues.

In single-domain dialogues, humans converse with the dialogue agent to complete tasks of one domain.

In contrast, in multi-domain dialogues, the tasks of interest can come from different domains.

A dialogue state in a multi-domain dialogue should include all inform and request slots of corresponding domains up to the current turn.

Examples of a single-domain dialogue and a multi-domain dialogue with annotated states after each turn can be seen in Figure 1 .

Despite there being several efforts in developing task-oriented dialogue systems in a single domain (Wen et al., 2016a; Lei et al., 2018) , there have been limited contributions for multi-domain task-oriented dialogues.

Developing end-to-end systems for multi-domain dialogues faces several challenges: (1) Belief states in multi-domain dialogues are usually larger and more complex than in single-domain, because of the diverse information from multiple domains.

Each domain can have shared slots that are common among domains or unique slots that are not shared with any.

(2) In an end-to-end system, the dialogue agent must incorporate information from source sequences, e.g. dialogue context and human utterances, as well as tracked belief states and extracted information from knowledge base, to semantically shape a relevant response with accurate information for task completion.

Directly applying methods for single-domain dialogues to multi-domain dialogues is not straightforward because the belief states extend across multiple domains.

A possible solution is to process a multi-domain dialogue for N D times for N D domains, each time obtaining a belief state of one domain.

However, this approach does not allow learning co-references in dialogues whereby users can switch from one domain to another turn by turn.

We propose an end-to-end dialogue system approach which explicitly track the dialogue states in multiple domains altogether.

Specifically, (1) we propose Multi-level Neural Belief Tracker to process contextual information for both slot-level and domain-level signals independently.

The two levels are subsequently combined to learn multi-domain dialogue states.

Our dialogue state tracker enables shared learning of slots common among domains as well as learning of unique slots in each domain.

(2) we utilize multi-head attention layers (Vaswani et al., 2017) to comprehensively process various types of information: dialogue context, user utterances, belief states of both inform and request slots, and DB query results.

The multi-head structure allows the model to independently attend to the features over multiple representation sub-spaces; and (3) we combine all components to create a dialogue system from state tracking to response generation.

The system can be jointly learned in an end-to-end manner.

Our end-to-end dialogue system utilizes supervision signals of dialogue states and output responses without using system action annotation.

To comprehensively validate our method, we compare our models with baselines in end-to-end, DST, and context-to-text generation settings.

We achieve the state-of-the-art performance in DST, task-completion, and response generation in the MultiWOZ2.1 corpus Eric et al., 2019 ) as compared to other baselines in similar settings.

In context-to-text generation setting that allows supervision of dialogue acts, our models can achieve competitive measures of Inform and BLEU metric.

Our work is related to 2 main bodies of research: DST and end-to-end dialogue systems.

Prior DST work focuses on single-domain dialogues using WOZ (Wen et al., 2017) and DSTC2 (Henderson et al., 2014a) corpus. (Mrkšić et al., 2015; Wen et al., 2016b; Rastogi et al., 2017) address transfer learning in dialogues from one domain to another rather than multiple domains in a single dialogue.

Our work is more related to recent effort for multi-domain DST such as Lee et al., 2019; Wu et al., 2019a; .

These models can be categorized into two main categories of DST: fixed-vocabulary and open-vocabulary approach.

Fixed vocabulary models (Zhong et al., 2018; Lee et al., 2019) assume known slot ontology with a fixed candidate set for each slot.

Open-vocabulary models (Lei et al., 2018; Wu et al., 2019a; Gao et al., 2019) derive the candidate set based on the source sequence i.e. dialogue history, itself.

Our approach is more related to open-vocabulary approach as we aim to dynamically generate dialogue state based on input dialogue history.

Different from most prior work, our Multi-level Neural Belief Tracker can learn domain-level and slot-level signals independently and both are combined in a Late Fusion manner to obtain contextual representations of all (domain, slot) pairs.

Conventionally, an end-to-end dialogue system is composed of separate modules for Natural Language Understanding (NLU) (Hashemi et al., 2016; Gupta et al., 2018) , DST (Henderson et al., 2014b; Zhong et al., 2018) , Dialogue Policy (Peng et al., 2017; , and Natural Language Generator (NLG) (Wen et al., 2016a; Su et al., 2018) .

These components can be learned independently and combined into a pipeline architecture for end-to-end system (Wen et al., 2017; Liu & Lane, 2017; .

Another line of research aims to develop a dialogue agent without modularizing these components but incorporating them into a single network (Eric & Manning, 2017; Lei et al., 2018; Madotto et al., 2018; Wu et al., 2019b) .

Our work is more related to the latter approach whereby we incorporate conventional components into an integrate network architecture and jointly train all parameters.

However, following (Lei et al., 2018) , we consider a separate module that combines NLU and DST together.

The module utilizes additional supervision for more fine-grained tracking of user goals.

This strategy is also suitable for large-scale knowledge base with large number of entities. (Madotto et al., 2018; Wu et al., 2019b; Gangi Reddy et al., 2019) completely omit the DST component by formulating entity attributes into memory form based on (Subject, Relation, Object) tuples.

These models achieve good performance in small-scale corpus such as In-Car Assistant (Eric & Manning, 2017) and WOZ2.0 (Wen et al., 2017) but will become extremely hard to scale to large knowledge base in multi-domain setting such as MultiWOZ corpus.

Given a dialogue with dialogue history of t − 1 turns, each including a pair of user utterance and system response, (U 1 , S 1 ), ..., (U t−1 , S t−1 ), the user utterance at current dialogue turn U t , and a knowledge base in form of entity data tables, the goal of a task-oriented dialogue system is to generate a response S t that is not only appropriate to the dialogue context, but also task-related with the correct entity results for the users.

In the multi-domain dialogue setting, turns in the dialogue history and the current user utterance could come from different domains.

Therefore, the generated response in this setting should also be domain-related with the correct domain-specific information for the users.

We propose a novel Multi-level Neural Belief Tracker to track belief states at both domain level and slot level to address multi-domain dialogues.

Following (Lei et al., 2018) , we utilize the previous belief states B t−1 as an input to the model.

This allows the model to rely on the dialogue states detected from the previous dialogue step t − 1 to update the state of the current step t. In addition, we adopt the attention-based principle of Transformer network (Vaswani et al., 2017) and propose an end-to-end architecture for task-oriented dialogues.

Our model allows comprehensive information processing from different input sources, incorporating contextual features from dialogue context and user utterance as well as learning signals from domain-level and slot-level dialogue states.

Our solution consists of 3 major components: (i) Encoders encode sequences of dialogue history, current user utterances, target system responses, domain and slot names, and previous dialogue belief states, into continuous representations. (ii) Multi-level Neural Belief Tracker includes 2 modules, one for processing slot-level information and one for domain-level information.

Each module comprises attention layers to project domain or slot token representations and attend on relevant features for state tracking.

The outputs of the two modules are combined to create domain-slot joint feature representations.

Each feature representation is used as a context-aware vector to decode the corresponding inform or request slots in each domain. (iii) Response Generator projects the target system responses and incorporates contextual information from dialogue context as well as intermediate variables from the state tracker and DB query results.

Employing attention mechanisms with feed-forward and residual connections allows our models to focus on relevant parts of the inputs and pass on the relevant information to decode appropriate system responses.

We combine all the modules into an end-to-end architecture and jointly train all components.

An overview of the proposed approach can be seen in Figure 2 .

An encoder encodes a text sequence of tokens (x 1 , ..., x n ) to a sequence of continuous representation z = (z 1 , ..., z n ) ∈ R n×d .

Each encoder includes a token-level trainable embedding layer and layer normalization (Ba et al., 2016) .

Depending on the type of text sequences, we inject sequential characteristics of the tokens (i.e. their positions in the sequence) using a sine and cosine positional encoding functions (Vaswani et al., 2017) .

Element-wise summation is used to combine the token- level embedding with positional embedding, each has the same embedding dimension d. The current user utterance U t is tokenized, prepended and appended with sos and eos token respectively.

In the dialogue history, each human utterance and system response up to dialogue step t − 1 is processed similarly.

The tokenized past utterances and system responses are concatenated sequentially by the dialogue step.

For target system response S t , during training, the sequence is offset by one position to ensure that token prediction in generation step i is based on the previous positions only i.e. 1, ..., i − 1.

Denoting name sloti and value sloti as the slot name and slot value of slot i, we create sequences of dialogue belief state from previous turn by following the template: value sloti inf _name sloti ... req_name slotj ...

domain d ...

A req_name slotj is only included in the sequence if slot j is predicted as in the previous turn.

As a slot such as area can be both request or inform type, the 2 slot types are differentiated by the prefixes inf and req.

Our belief sequences can be used to encode past dialogue states of multiple domains, each separated by the domain d token.

To learn slot-level and domain-level signals for state tracking, we construct set of slot and domain tokens as input to the state tracker.

Each input set is created by concatenating slot names or domains:

respectively.

Both sequences are kept fixed in all dialogue samples to factor in all possible domains and slots for multi-domain state tracking.

Positional encoding is used in all sequences except for input sets of slot and domain tokens as these sets do not contain sequential characteristic.

Embedding weights are shared among all the encoders of source sequences.

Embedding weights of the target system responses are not shared to allow the models to learn the semantics of input and output sequences differently.

The DST module processes slot-level and domain-level information independently, and integrates the two for multi-domain state tracking.

We adopt a Late Fusion approach to combine domain and slot representations in deeper network layers.

Slot-level Processing.

Given the encoded features from the source sequences, including dialogue history z his , previous belief state z bs , and the current user utterance z utt , the slot-level signals are learned by projecting the encoded slot token sequence z S through N S dst identical layers.

Each layer contains 4 attention blocks, each of which employ the multi-head attention mechanism (Vaswani et al., 2017) to attend on the inputs at different representation sub-spaces.

Each attention block is coupled with a position-wise feed-forward layer, including 2 linear transformations with ReLU activation in between.

Residual connection (He et al., 2016) and layer normalization (Ba et al., 2016) are employed in each attention block.

Specifically, given the current feature vector z out S as output from previous attention block (or z S itself in the first attention block of the first processing layer) and the encoded features z seq of a source sequence, the multi-head attention is defined as:

where

, and seq = {S, his, bs, utt} (for simplicity, the subscripts of S and seq are omitted in each W ).

The first attention block is a self-attention, i.e. seq = S, which enables learning the relation between slots independently from domains.

Subsequent attention layers on dialogue context, previous belief state, and user utterance of current turn, inject each slot token representation with dialogue contextual information up to current user utterance in turn t. Through residual connection, the contextual information are passed forward in each z out S .

Using different attention blocks allows flexible processing of information from various input sources.

Domain-level Processing.

The input to the domain-level processing module includes the encoded domain token sequence z D , the encoded dialogue history up to turn t − 1 z his , and the encoded user utterance of current turn z utt .

The domain features are passed through N D dst identical layers, each of which include 3 multi-head attention blocks to obtain important contextual information from dialogue context and user utterance.

Similarly to slot-level processing, a self-attention block is leveraged to allow reasoning among domains independently from slots.

Attending on dialogue history and current user utterance separately enables learning domain signals from the contextual information of past dialogue turns and current turns differently.

Therefore, the models can potentially detect changes of dialogue domains from past turns to the current turn.

Especially in multi-domain dialogues, users can switch from one domain to another and the generated responses should address the latest domain.

d is used to decode the corresponding domain-specific slot i. The vector is used as initial hidden state for an RNN decoder to decode an inform slot token by token or passed through a linear transformation layer for binary classification for a request slot.

The decoded dialogue states are used to query the DBs of all domains and obtain the number of the result entities in each domain.

We then create a fixed-dimensional one-hot pointer vector for each domain d: z We embed the pointer vector with the learned embedding and positional embedding as similarly described in Section 3.1, resulting in z db ∈ R 6N D ×d .

The DB pointer vector z db , context-aware domain-slot joint features z out DS , encoded dialogue history z his , and user utterance of current turn z utt , are used as inputs to incorporate relevant signals to decode system responses.

The generator includes N gen identical layers, each includes 5 multi-head attention blocks, including a self-attention block at the beginning.

Adopting attention with residual connection in each block allows the models to comprehensively obtain contextual cues, either through text sequences or domain-slot joint features, and knowledge base signals from DB pointer vectors.

The final output z out gen is passed to a linear transformation with softmax activation to decode system responses.

The objective function is a combination of belief state objectives, including the log-likelihood of all inform slot sequences S inf , and the binary cross entropy of request slots S req , and the system response objective, including the log-likelihood of the target response sequence T , as follows:

where

The above objectives are conditioned on the input features, including dialogue context C, current user utterance U , previous and current belief state B t−1 and B t , and DB queries Q.

4.1 DATA We used the MultiWOZ 2.1 dataset Eric et al., 2019) which consists of both single-domain and multi-domain dialogues.

Compared to version 2.0, MultiWOZ 2.1 is improved with some correction of DST labels, including about 40% changes across training samples.

We pre-processed the dialogues by tokenizing, lower-casing, and delexicalizing all system responses.

From the belief state annotation of the training data, we identified all possible domains and slots.

We identified N D = 7 domains and N S = 35 unique inform slots in total.

We followed the preprocessing scripts as provided by Wu et al., 2019b) .

The result corpus includes 8,438 dialogues in the training with an average of 1.8 domains per dialogue.

Each dialogue has more than 13 turns.

There are 1,000 in each validation and test set, each including an average of 1.9 domains per dialogue.

Other details of data pre-processing procedures, domains, slots, and entity DBs, are included in Appendix A.1.

The model parameters are:

We employed dropout (Srivastava et al., 2014) of 0.3 at all network layers except the linear layers in the generative components.

Label smoothing (Szegedy et al., 2016) for target system responses is applied during training.

During training, we utilize teacher-forcing learning strategy by simply using the ground-truth inputs of dialogue state from previous turn and the gold DB pointer.

During inference, in each dialogue, we decode system responses sequentially turn by turn, using the previously decoded belief state as input in the current turn, and at each turn, using the decoded belief state to query DBs for pointer vectors.

We train all networks in an end-to-end manner with Adam optimizer (Kingma & Ba, 2015) and the learning rate schedule similarly adopted by Vaswani et al. (2017) .

We used batch size 32 and tuned the warmup_steps from 9K to 15K training steps.

All models are trained up to 30 epochs and best models are selected based on validation loss.

We used a greedy approach to decode all slots and beam search with beam size 5 and a length penalty 1.0 to decode responses.

The maximum length is set to 10 tokens for each slot and 20 for system responses.

Our models are implemented using PyTorch (Paszke et al., 2017) .

To evaluate the models, we use the following metrics: (1) DST metrics: Joint Accuracy and Slot Accuracy (Henderson et al., 2014b) .

Joint Accuracy compares the predicted dialogue states to the ground truth in each dialogue turn.

All slot values must match the ground truth labels to be counted as a correct prediction.

Slot Accuracy considers individual slot-level accuracy across the topology.

(2) Task-completion metrics: Inform and Success (Wen et al., 2017) .

Inform refers to system ability to provide an appropriate entity while Success is the system ability to answer all requested attributes.

(3) Generation metrics: BLEU score (Papineni et al., 2002) .

We ran all experiments 3 times and reported the average results.

We report results in 2 different settings: end-to-end dialogues and DST.

In end-to-end setting, we train a dialogue agent that is responsible for both DST and text generation without assuming access to ground-truth labels.

End-to-End.

In this setting, we compare our model performance on the joint task of DST and context-to-text generation.

For fair comparision, we select TSCP (Lei et al., 2018) as the baseline as TSCP does not use additional supervision signals of system action as input.

This is the current state-of-the-art for end-to-end dialogue task in the single-domain WOZ (Wen et al., 2017) .

TSCP applies pointer network to develop a two-stage decoding process to decode belief states, in a form of text sequence, and subsequently decode system responses.

We adapt the method to the multi-domain dialogue setting.

We experiment with 2 cases of TSCP in which the maximum length of the output state sequence L bspan is set to 8 and 20 tokens.

As can be seen in Table 1 , our model outperforms in all metrics, except for the Slot Acc metric in one case.

Overall, our model performs well in both multi-domain and single-domain dialogues, especially with higher performance gain in multi-domain dialogues.

The performance gain in multi-domain dialogues can be explained by the separate network structure between domain and slot processing modules in our models.

This allows our models to learn domain-dependent and slot-dependent signals separately before the two are fused into a joint feature vectors for downstream process.

For TSCP, increasing the L bspan from 8 to 20 tokens helps to improve the performance, but also increases the training time to convergence significantly.

In our approach, all inform and request slots are decoded independently and the training time is less affected by the size of the target dialogue states, especially in cases of extensive belief states (e.g. 4 or 5 domains in a dialogue).

Additional results by individual domains are described in Appendix A.3.

DST.

We isolate the DST components (i.e. training models only with L(B t )) and report the DST performance.

We compare the performance with the baseline models on the MultiWOZ 2.1 in Table  2 (Refer to Appendix A.2 for more description of DST baselines).

Our model outperforms existing baselines and achieves the state-of-the-art performance in MultiWOZ2.1 corpus.

By leveraging on dialogue context signals through independent attention modules at domain level and slot level, our DST can generate slot values more accurately.

DST approaches that try to separate domain and slot signals such as TRADE (Wu et al., 2019a ) reveal competitive performance.

However, our approach has better performance as we enable deeper interaction of context-related signals in each domain and slot representation.

Compared to TRADE, our approach can be considered as Late Fusion approach that combines representations in deeper network layers for better joint features of domains and slots.

We also noted that DST performance improves when our models are trained as an end-to-end system.

This can be explained as additional supervision from system responses not only contributes to learn a good response generation network but also positively impact DST network.

Additional DST results of individual domains can be seen in Appendix A.3.

For completion, we also conduct experiment for context-to-text generation setting and compare with baseline models in Appendix A.3.

We experiment with different model variants in Table 3 .

First, we noted that removing selfattention on the joint feature domain-slot vectors (N DS dst = 0) reduces the joint accuracy performance.

This self-attention is important because it allows our models to learn signals across (domain, slot) joint features rather than just at independently domain level and slot level.

Second, ranging the number of attention layers in domain-level processing and slot-level processing from 3 to 1 gradually reduces the model performance.

This shows the efficacy of our Late Fusion approach.

Combining the

Joint Accuracy HJST (Eric et al., 2019) 35.55% DST Reader (Gao et al., 2019) 36.40% TSCP (Lei et al., 2018) 37.12% FJST (Eric et al., 2019) 38.00% HyST 38.10% TRADE (Wu et al., 2019a) 45.60% Ours 49.55%

features at deeper network layers results in better joint feature representation and hence, increases the model performance.

Lastly, we observed that our models can efficiently detect contextual signals from the dialogue states of previous turn PrevBS as the performance of our models with or without using the full dialogue history is very similar.

This will benefit as the dialogue history evolves over time and our models only need to process the latest dialogue turn in combination with the predicted dialogue state in previous turn as an input.

Qualitative Analysis.

We examine an example dialogue in the test data and compare our predicted outputs with the baseline TSCP (L bspan = 20) (Lei et al., 2018) and the ground truth.

From the table in the left of Figure 3 , we observe that both our predicted dialogue state and system response are more correct than the baseline.

Specifically, our dialogue state can detect the correct type slot in the attraction domain.

As our dialogue state is correctly predicted, the queried results from DB is also more correct, resulting in better response with the right information (i.e. 'no attraction available').

From visualization of domain-level and slot-level attention on the user utterance, we notice important tokens of the text sequences, i.e. 'entertainment' and 'close to', are attended with higher attention scores.

In addition, at domain-level attention, we find a potential additional signal from the token 'restaurant', which is also the domain from the previous dialogue turn.

We also observe that attention is more refined along the neural network layers.

For example, in the domain-level processing, compared to the 2 nd layer, the 4 th layer attention is more clustered around specific tokens of the user utterance.

The complete predicted output for this example dialogue and other qualitative analysis can be seen in Appendix A.4.

In this work, we proposed an end-to-end dialogue system with a novel Multi-level Neural Belief Tracker.

Our DST module can track complex belief states of multiple domains and output more accurate dialogue states.

The DST is combined with attention-based generation module to generate dialogue responses.

Evaluated on the large-scale multi-domain dialogue benchmark MultiWOZ2.1, our models achieve the state-of-the-art performance in DST and competitive measures in taskcompletion and response generation.

Figure 3 : Example dialogue with the input system response St−1 and current user utterance Ut, and the output belief state BSt and system response St. Compared with TSCP (Row 3), our dialogue state and response (Last Row) are more correct and closer to the ground truth (Row 2).

Visualization of attention to the user utterance sequence at slot-level (lower right) and domain-level (upper right) is also included.

More red denotes higher attention score between domain or slot representation and token representation.

Best viewed in color.

A.1 DATA PRE-PROCESSING First, we delexicalize each target system response sequence by replacing matched entity attribute appeared in the sequence to the canonical tag domain_attribute .

For example, the original target response 'the train id is tr8259 departing from cambridge' is delexicalized into 'the train id is train_id departing from train_departure'.

We use the provided entity databases (DBs) to match potential attributes in all target system responses.

For dialogue history, we keep the original version of all text, including system responses of previous turns, rather than the delexicalized form.

We split all sequences of dialogue history, user utterances of current turn, previous belief states, and delexicalized target responses, into case-insensitive tokens.

We share the embedding weights of all source sequences.

For source sequences, in total there are 5,491 unique tokens, including slot and domain tokens as well as eos , sos , pad , and unk tokens.

For target sequences, there are 2,648 unique tokens in total, including all canonical tags as well as eos , sos , pad , and unk tokens.

As can be seen in Table 4 , in source sequences, the overlapping rates of unique tokens to the training embedding vocabulary are about 64% and 65% in validation and test set respectively.

For target sequences, the overlapping rates are about 83% and 82% in validation and test set respectively.

As we analyze the data, we summarize the number of dialogues in each domain in Table 5 .

For each domain, a dialogue is selected as long as the whole dialogue (i.e. single-domain dialogue) or parts of the dialogue (i.e. in multi-domain dialogue) is involved with the domain.

For each domain, we also build a set of possible inform and request slots using the belief state annotation in the training data.

The details of slots, entity attributes, and DB size, in each domain, can be seen in Table 6 .

The DBs of 3 domains taxi, police, and hospital were not provided in the benchmark.

We describe a list of baseline models in DST setting and context-to-text generation setting.

FJST and HJST (Eric et al., 2019) .

FJST and HJST follow a fixed-vocabulary approach for state tracking.

Both models include encoder modules (either bidirectional LSTM or hierarchical LSTM) to encode the dialogue history.

The models pass the context hidden states to separate linear transforma- tion to obtain final vectors to predict individual state slots separately.

The output vector is used to measure a score of a predefined candidate set for each slot.

TSCP (Lei et al., 2018) .

TSCP is an end-to-end dialogue system that can do both DST and NLG.

The model utilize pointer network to generate both dialogue states and responses.

To compare with TSCP in DST setting, we adapt the model to multi-domain dialogues and report the results only on DST components.

For DST experiment, we reported the performance when the maximum length of dialogue state sequence in the state decoder L is set to 20 tokens.

DST Reader (Gao et al., 2019) .

This model considers the DST task as a reading comprehension task.

The model predicts each slot as a span over tokens within dialogue history.

DST Reader utilizes attention-based neural network with additional modules to predict slot type and slot carryover probability.

HyST .

This baseline combines the advantage of both fixed-vocabulary and open-vocabulary approaches.

In open-vocabulary, the set of candidates of each slot is constructed based on all word n-grams in dialogue history.

Both approaches are applied in all slots and depending on their performance in validation set, the better approach is applied to predict individual slots.

TRADE (Wu et al., 2019a) .

This is the current state-of-the-art model on the MultiWOZ2.1 dataset.

The model combines pointer network to generate individual slot token-by-token.

The prediction is additional supported by a slot gating component that decides whether the slot is "none", "dontcare", or "pointer" (generated).

provides a baseline for this setting by following the sequence-to-sequence model with additional signals from the belief tracker and discrete data pointer vector.

TokenMoE (Pei et al., 2019) .

TokenMoE refers to Token-level Mixture-of-Expert model.

The model follows a modularized approach by separating different components known as expert bots for different dialogue scenarios.

A dialogue scenario can be dependent on a domain, a type of dialogue act, etc.

A chair bot is responsible for controlling expert bots to dynamically generate dialogue responses.

HDSA (Chen et al., 2019) .

This is the current state-of-the-art for context-to-text generation setting in MultiWOZ2.0.

HDSA leverages the structure of dialogue acts to build a multi-layer hierarhical graph.

The graph is incorporated as an inductive bias in self-attention network to improve the semantic quality of generated dialogue responses.

Structured Fusion (Mehri et al., 2019) .

This approach follows a traditional modularized dialogue system architecture, including separate components for NLU, DM, and NLG.

These components are pretrained and combined into an end-to-end system.

Each component output is used as a structured input to other components.

LaRL (Zhao et al., 2019) .

This model uses a latent dialogue action framework instead of traditional handcrafted dialogue acts.

The latent variables are learned using unsupervised learning with stochastic variational inference.

The model are trained in a reinforcement learning framework whereby the parameters are trained to yield better Success rate.

Domain-Specific Results.

In Table 7 and 8, we presented additional results of our model and the baselines TSCP (Lei et al., 2018) .

For state tracking, the metrics are calculated for domain-specific slots of the corresponding domain at each dialogue turn.

For task completion and response generation, we calculated the metrics for single-domain dialogues of the corresponding domain.

We do not report the Inform metric for the taxi domain because no DB was provided in the benchmark for this domain.

From Table 7 , in each domain, our approach outperforms TSCP across most of the metrics, except the Success and BLEU metric in the taxi domain.

In term of task-completion, our model performs better significant improvement in domains with large DB sizes such as train and restaurant.

In term of response generation, our results are consistently higher than the baselines as the model can return more appropriate responses from better decoded dialogue states and DB queried results.

For state tracking task alone, in Table 8 , our models perform consistently in the 3 domains attraction, restaurant, and train domains.

However, the performance significantly drops in the taxi domain.

This performance drop negatively impacts the overall performance across all domains.

We plan to investigate further to identify and address challenges in this particular domain in future work.

Context-to-Text Generation.

Following Eric et al. (2019) , to compare with baselines in this setting, we assumes access to the ground-truth labels of dialogue belief states and data pointer during inference.

We compare with existing baselines in Table 2 (Refer to Appendix A.2 for more description of the baselines).

Our model achieves the state-of-the-art in the Inform metric but do not perform as well in terms of Success metric.

We achieve a competitive BLEU score, only behind the current state-of-the-art HDSA model.

An explanation for our model not able to achieve a high Success metric is that we did not utilize the dialogue act information.

The current state-of-the-art HDSA leverages the graph structure of dialogue acts into dialogue models.

Furthermore, compared to approaches such Table 9 : Performance for context-to-text generation setting on MultiWOZ2.0.

The baseline results are as reported in the benchmark leaderboard.

Inform Success BLEU Baseline 71.29% 60.96% 18.80 TokenMoE (Pei et al., 2019) 75.30% 59.70% 16.81 HDSA (Chen et al., 2019) 82.90% 68.90% 23.60 Structured Fusion (Mehri et al., 2019) 82.70% 72.10% 16.34 LaRL (Zhao et al., 2019) 82.78% 79.20% 12.80 Ours 83.83% 67.36% 19.88

as (Pei et al., 2019; Mehri et al., 2019) , our model does not use pretrained network modules such as NLU and DST.

Our end-to-end setting is more related along the line of research work for end-to-end dialogue systems without relying on system action annotation (Lei et al., 2018; Madotto et al., 2018; Wu et al., 2019b) .

To improve the Success metric, we plan to extend our work in the future that can derive better dialogue policy for higher Success rate.

In Table 10 , we reported the complete output of an example multi-domain dialogue.

Overall, our dialogue agent can carry a proper dialogue with the user throughout the dialogue steps.

Specifically, we observed that our model can detect new domains at dialogue steps where the domains are introduced e.g. attraction domain at the 5 th turn and taxi domain at the 8 th turn.

The dialogue agent can also detect some of the co-references among the domains.

For example, at the 5 th turn, the dialogue agent can infer the slot area for the new domain attraction as the user mentioned 'close the restaurant'.

We noticed that that at later dialogue steps such as the 6 th turn, our decoded dialogue state is not correct possibly due to the incorrect decoded dialogue state in the previous turn, i.e. 5 th turn.

In Figure 4 and 5, we reported the Joint Goal Accuracy and BLEU metrics of our model by dialogue turn.

As we expected, the Joint Accuracy metric tends to decrease as the dialogue history extends over time.

The dialogue agent achieves the highest accuracy in state tracking at the 1 st turn and gradually reduces to zero accuracy at later dialogue steps, i.e. 15 th to 18 th turns.

For response generation performance, the trend of BLEU score is less obvious.

The dialogue agent obtains the highest BLEU scores at the 3 rd turn and fluctuates between the 2 nd and 13 th turn.

<|TLDR|>

@highlight

We proposed an end-to-end dialogue system with a novel multi-level dialogue state tracker and achieved consistent performance on MultiWOZ2.1 in state tracking, task completion, and response generation performance.