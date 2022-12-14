Semantic parsing which maps a natural language sentence into a formal machine-readable representation of its meaning, is highly constrained by the limited annotated training data.

Inspired by the idea of coarse-to-fine, we propose a general-to-detailed neural network(GDNN) by incorporating cross-domain sketch(CDS) among utterances and their logic forms.

For utterances in different domains, the General Network will extract CDS using an encoder-decoder model in a multi-task learning setup.

Then for some utterances in a specific domain, the Detailed Network will generate the detailed target parts using sequence-to-sequence architecture with advanced attention to both utterance and generated CDS.

Our experiments show that compared to direct multi-task learning, CDS has improved the performance in semantic parsing task which converts users' requests into meaning representation language(MRL).

We also use experiments to illustrate that CDS works by adding some constraints to the target decoding process, which further proves the effectiveness and rationality of CDS.

Recently many natural language processing (NLP) tasks based on the neural network have shown promising results and gained much attention because these studies are purely data-driven without linguistic prior knowledge.

Semantic parsing task which maps a natural language sentence into a machine-readable representation BID6 ), as a particular translation task, can be treated as a sequence-to-sequence problem BID3 ).

Lately, a compositional graph-based semantic meaning representation language (MRL) has been introduced BID14 ), which converts utterance into logic form (action-object-attribute), increasing the ability to represent complex requests.

This work is based on MRL format for semantic parsing task.

Semantic parsing highly depends on the amount of annotated data and it is hard to annotate the data in logic forms such as Alexa MRL.

Several researchers have focused on the area of multi-task learning and transfer learning BID10 , BID6 , BID15 ) with the observation that while these tasks differ in their domains and forms, the structure of language composition repeats across domains BID12 ).

Compared to the model trained on a single domain only, a multi-task model that shares information across domains can improve both performance and generalization.

However, there is still a lack of interpretations why the multi-task learning setting works BID26 ) and what the tasks have shared.

Some NLP studies around language modeling BID18 , BID29 , BID2 ) indicate that implicit commonalities of the sentences including syntax and morphology exist and can share among domains, but these commonalities have not been fully discussed and quantified.

To address this problem, in this work, compared to multi-task learning mentioned above which directly use neural networks to learn shared features in an implicit way, we try to define these cross-domain commonalities explicitly as cross-domain sketch (CDS).

E.g., Search weather in 10 days in domain Weather and Find schedule for films at night in domain ScreeningEvent both have action SearchAction and Attribute time, so that they share a same MRL structure like SearchAction(Type(time@?)), where Type indicates domain and ? indicates attribute value which is copying from the original utterance.

We extract this domain general MRL structure as CDS.

Inspired by the research of coarse-to-fine BID4 ), we construct a two-level encoder-decoder by using CDS as a middle coarse layer.

We firstly use General Network to get the CDS for every utterance in all domains.

Then for a single specific domain, based on both utterance and extracted CDS, we decode the final target with advanced attention while CDS can be seen as adding some constraints to this process.

The first utterance-CDS process can be regarded as a multi-task learning setup since it is suitable for all utterances across the domains.

This work mainly introducing CDS using multi-task learning has some contributions listed below: 1) We make an assumption that there exist cross-domain commonalities including syntactic and phrasal similarity for utterances and extract these commonalities as cross-domain sketch (CDS) which for our knowledge is the first time.

We then define CDS on two different levels (action-level and attribute-level) trying to seek the most appropriate definition of CDS.2) We propose a general-to-detailed neural network by incorporating CDS as a middle coarse layer.

CDS is not only a high-level extraction of commonalities across all the domains, but also a prior information for fine process helping the final decoding.3) Since CDS is cross-domain, our first-level network General Network which encodes the utterance and decodes CDS can be seen as a multi-task learning setup, capturing the commonalities among utterances expressions from different domains which is exactly the goal of multi-task learning.

Traditional spoken language understanding (SLU) factors language understanding into domain classification, intent prediction, and slot filling, which proves to be effective in some domains BID9 ).

Representations of SLU use pre-defined fixed and flat structures, which limit its expression skills like that it is hard to capture the similarity among utterances when the utterances are from different domains BID15 ).

Due to SLU's limited representation skills, meaning representation language (MRL) has been introduced which is a compositional graph-based semantic representation, increasing the ability to represent more complex requests BID14 ).

There are several different logic forms including lambda-calculus expression BID17 ), SQL BID33 ), Alexa MRL BID14 ).

Compared to fixed and flat SLU representations, MRL BID14 ) based on a large-scale ontology, is much stronger in expression in several aspects like cross-domain and complex utterances.

Mapping a natural language utterance into machine interpreted logic form (such as MRL) can be regarded as a special translation task, which is treated as a sequence-to-sequence problem BID28 ).

Then BID1 and BID22 advance the sequence-tosequence network with attention mechanism learning the alignments between target and input words, making great progress in the performance.

BID23 explore the attention mechanism with some improvements by replacing attention function with attention sparsity.

Besides, to deal with the rare words, BID8 incorporate the copy mechanism into the encoder-decoder model by directly copying words from inputs.

Lately, many researchers have been around improving sequence-to-sequence model itself, in interpreting the sentence syntax information.

BID5 encode the input sentence recursively in a bottom-up fashion.

BID30 generate the target sequence and syntax tree through actions simultaneously.

Another aspect which has caught much attention is constrained decoding.

BID16 and BID25 add some constraints into decoding process, making it more controllable.

BID3 use the recurrent network as encoder which proves effective in sequence representation, and respectively use the recurrent network as decoder and tree-decoder.

BID16 employ the grammar to constrain the decoding process.

BID4 , believe utterance understanding is from high-level to low-level and by employing sketch, improve the performance.

For semantic parsing task especially in MRL format, it is expensive and time-consuming to annotate the data, and it is challenging to train semantic parsing neural models.

Multi-task learning aims to use other related tasks to improve target task performance.

BID20 deal with traditional SLU piper-line network by jointly detecting intent and doing slot filling.

BID27 share parameters among various tasks, according to the low-level and high-level difference.

BID11 divide the representation network into task-specific and general which is shared during multi-task learning.

BID6 and BID12 directly share the encoder or decoder neural layers (model params) through different semantic parsing tasks.

In BID15 , multi-task learning also mainly acts sharing the params of the network.

For human language expressions, especially task-oriented requests, there exist commonalities across sentences including linguistic syntax and phrase similarity.

They can be seen with general sentence templates.

TAB1 Since sentence templates are too many, we try to leverage these common regularities more abstractly.

We extract these invariant commonalities which are implicit across domains, and call them as crossdomain sketch (CDS) in a canonical way.

We define CDS in meaning representation language (MRL) format (action-object-attribute) and on two levels (action-level and attribute-level).

Action-level CDS means to acquire the same action for utterances from different domains while the attribute-level CDS means to extract more detailed information.

See examples in TAB1 .

Instead of extracting CDS from utterance directly, we try converting from logic form into CDS reversely, because it is easier to deal with structural logic form than utterance in natural language form.

We analyze the dataset Snips and use a rule-based method to obtain CDS.

We strip logic forms from domain-specific components and preserve domainindependent parts including general actions and attributes.

We do some statistics on the dataset Snips BID7 ) used in this paper.

We convert attributes [object type, movie type, restaurant type] into {object type}, [object name, restaurant name, movie name] into {object name}, [year, timeRange] into {time}, [location name, current location] into {object location}, [country, city] into {place}. All those attributes account for 55% of all attributes which indicate the existence and feasibility of CDS.

Figure 1 shows our overall network, which contains a two-level encoder-decoder.

The General Network encodes utterance and decodes cross-domain sketch (CDS).

Since this process is domaingeneral, it can be done to all domains, which is a multi-task setup.

The Detailed Network firstly encodes the CDS and the utterance, then it decodes the target result based on both utterance and CDS.

This process is domain-dependent, so that it is a fine-tuning process in a specific domain.

For an input utterance u = u 1 , u 2 , ...u |u| , and its middle result cross-domain sketch (CDS) c = c 1 , c 2 , ...c |c| , and its final result logic form y = y 1 , y 2 , ...y |y| , the conditional probability is: Figure 1 : Overall Network.

General Network (red dashed box below) encodes the utterance with bi-directional LSTM and decodes cross-domain sketch (CDS) using unidirectional LSTM with attention to utterance in all domains.

For identical encoding, general utterance encoding and specific utterance encoding share the same encoder while for separate encoding, they are not (see Section 3.2.2).

Then Detailed Network, in one specific domain, encodes CDS and utterance using bi-directional LSTM, decodes the final target with advanced attention to both utterance and CDS.

DISPLAYFORM0 where y <t = y 1 , y 2 , ...y |t???1| , and c <t = c 1 , c 2 , ...c |t???1| .

The neural encoder of our model is similar to neural machine translation (NMT) model, which uses a bi-directional recurrent neural network.

Firstly each word of utterance is mapped into a vector u t ??? R d via embedding layer and we get a word sequence u = (u 1 , ..., u |u| ).

Then we use a bi-directional recurrent neural network with long short-term memory units (LSTM) BID13 ) to learn the representation of word sequence.

We generate forward hidden state DISPLAYFORM0 The t-th word will be h DISPLAYFORM1 We construct two kinds of utterance encoders, general utterance encoder for General Network and specific utterance encoder for Detailed Network (see in Figure 1 ), so as to extract different information for different purposes.

The general utterance encoder, meant to pay more attention to cross-domain commonalities of utterances, is used by all utterances from all domains.

The specific utterance encoder, which is domain-dependent, belongs to one specific domain and is more sensitive to details.

We call encoder outputs h ug t from general utterance encoder and h us t from specific utterance encoder.

When the two encoders share the same parameters that is h ug t = h us t , we call it identical encoding and when they are not, we call it separate encoding, inspired by BID27 ; BID16 ; BID21 ; BID0 ) which explore the sharing mechanisms of multi-task learning and propose some improvements.

The General Network is meant to obtain cross-domain sketch (CDS) c conditioned on utterance u, using an encoder-decoder network.

After encoding utterance by general utterance encoder for all domains, we obtain h ug t (see Section 3.2.2).

Then we start to decode CDS.The decoder is based on a unidirectional recurrent neural network, and the output vector is used to predict the word.

The cd represents CDS decoder.

DISPLAYFORM0 where c t is the previously predicted word embedding.

The LuongAttention BID22 ) between decoding hidden state d t and encoding sequence e i (i = 1, 2, ...|e|) at time step t is computed as: DISPLAYFORM1 DISPLAYFORM2 Based on equations FORMULA4 and FORMULA5 , we compute the attention a

The t-th predicted output token will be: DISPLAYFORM0 DISPLAYFORM1 where W, b are parameters.

After decoding CDS words c = (c 1 , ..., c |c| ), we use an encoder to represent its meaning and due to words' relation with forward and backward contexts, we choose to use a bi-directional LSTM.

We generate forward hidden state ??? ??? h

Through specific utterance encoder and cross-domain sketch (CDS) encoder, we acquired t-th word representation h us t and h ce t .

Finally with advanced attention to both encoded utterance u and CDS c, we decode the final target y.

The decoder is based on a unidirectional recurrent neural network, and the output vector is used to predict the word.

The y represents target decoder.

DISPLAYFORM0 where y t is the previously predicted word embedding.

During target decoding process and at time step t, we not only compute the attention to utterance encoding outputs h us but also compute the attention to CDS encoding outputs h ce .

The attention between target hidden state and utterance is a 1, 2 , ...|c|) in the same way.

Then the t-th predicted output token will be based on the advanced two-aspect attention: DISPLAYFORM1

For training process, the objective is: DISPLAYFORM0 T is the training corpus.

For inference process, we firstly obtain cross-domain sketch (CDS) via c = argmax p(c|u) then we get the final target logic form via y = argmax p(y|u, c).

For both decoding processes, we use greedy search to generate words one by one.

Existed semantic parsing datasets, e.g., GEO BID32 ), ATIS BID31 ), collect data only from one domain and have a very limited amount, which can not fully interpret the effectiveness of cross-domain sketch (CDS) since it needs large dataset among different domains.

In this case, we mainly consider the semantic parsing task Snips (Goo et al. FORMULA0 ) based on MRL format (action-object-attribute).

Snips collects users' requests from a personal voice assistant.

The original dataset is annotated in spoken language understanding (SLU) format (intent-slot).

It has 7 intent types and 72 slot labels, and more statistics are shown in TAB4 .

Based on the format (intent-slot), we pre-process this dataset into MRL format by some pre-defined rules, then we regard the intent as domain/task and share CDS among them.

The details are shown in Target SearchAction ( ScreeningEventType ( object type @ 2 , movie type @ 4 , timeRange @ 6 , location name @ 8 9 10 ) ) TAB3 : Several examples of Snips.

Utterance is the user's request which is a natural language expression.

Intent and slots are in formats from original dataset.

Cross-domain sketch (CDS) has two levels (action-level and attribute-level).

Target is the final logic form with numbers indicating copying words from utterance (index starting from 0).

We use Tensorflow in all our experiments, with LuongAttention BID22 ) and copy mechanism.

The embedding dimension is set to 100 and initialized with GloVe embeddings (Pennington et al. FORMULA0 ).

The encoder and decoder both use one-layer LSTM with hidden size 50.

We apply the dropout selected in {0.3,0.5}. Learning rate is initialized with 0.001 and is decaying during training.

Early stopping is applied.

The mini-batch size is set to 16.

We use the logic form accuracy as the evaluation metric.

Firstly, in order to prove the role of the cross-domain sketch (CDS) in helping to guide decoding process with multi-tasking learning setup, we do several experiments, and the results are shown in Table 4 .

For joint learning, we apply several multi-task architectures from BID6 ), including one-to-one, one-to-many and one-to-shareMany.

One-to-one architecture applies a single sequence-to-sequence model across all the tasks.

One-to-many only shares the encoder across all the tasks while the decoder including the attention parameters is not shared.

In one-to-shareMany model, tasks share encoder and decoder (including attention) params, but the output layer of decoder is task-independent.

From the Table 4 , in general, joint learning performs better than single task learning.

In joint learning, one-to-one is the best and performs way better than one-to-many and one-to-shareMany, probably limited by the dataset's size and similarity among tasks.

By incorporating CDS, our GDNN (general-to-detailed neural network) models have all improved the performance to different degrees.

The CDS is defined on two levels (action-level and attribute-level, see examples in TAB3 ) and attribute-level CDS improves greater than action-level CDS, which is in our expectation since it offers more information for tasks to share.

We also experiment on different utterance encoding setups with identical encoding and separate encoding (see Section 3.2.2).

The separate encoding setup performs better than sharing the same encoder for utterance, which integrates the fact that different encoders pay different attention to the utterances due to different purposes which means one is more general and the other is more specific detailed.

Method Snips Accuracy Single Seq2Seq BID28 62.3 Joint Seq2Seq (one-to-many) BID6 62.0 Joint Seq2Seq (one-to-shareMany) BID6 64.2 Joint Seq2Seq (one-to-one) BID6 71.4 GDNN with Action-level CDS (identical encoding) 74.9 GDNN with Action-level CDS (separate encoding) 75.1 GDNN with Attribute-level CDS (identical encoding) 76.7 GDNN with Attribute-level CDS (separate encoding) 78.1 Table 4 : Multi-task Results.

Single Seq2Seq means each task has a sequenece-to-sequence model.

Joint Seq2Seq show results with three multi-task mechanisms.

Our results include GDNN (generalto-detailed neural network) models with different levels of CDS (action-level/attribute level) and different utterance encoding mechanisms (identical encoding/separate encoding).We also list the full results of GDNN in TAB6 Moreover, we compare our experiments with traditional models which regard the task as intent classification and slot filling (IC SF).

The results are shown in TAB7 below.

From TAB7 , we can see compared to IC SF models (based on sequence labeling format), Seq2Seq perform worse (71.4% compared to 73.2%) due to its fewer assumptions and larger decode size as well as its difficulty of training, which is usual in comparing IC SF models and sequence-tosequence models.

Through using CDS, the performance has significantly improved.

On the one Method Snips Accuracy Joint Seq.

BID10 73.2 Atten.-Based BID19 74.1 Slot.-Gated (Intent Atten.)

BID7 74.6 Slot.-Gated (Full Atten.)

BID7 75.5 Joint Seq2Seq BID6 71.4 GDNN with Action-level CDS 73.2 GDNN with Attribute-level CDS 74.6 hand, CDS extract the cross-domain commonalities among tasks helping to make the multi-task learning more specific, which can be seen as an advance to multi-task learning.

On the other hand, CDS can be seen adding some constraints to the final target decoding process which has offered more information for the decoding process, compared to direct joint Seq2Seq.

To better prove and explain this idea, we do some experiments according to constraint decoding aspect.

We try to compare the sub-process of converting utterance to CDS through different models, e.g., IC SF, Seq2Seq.

From the Table 7 , we can see that Seq2Seq achieve the comparable results (87.7%) to IC SF model (84.9%) for generating CDS from utterance, which further explains that, the fact joint seq2seq performs worse (71.4%, see TAB7 ) than IC SF model (73.2%) is owing to the lack of guidance and constraints during the follow-up decoding process.

By incorporating CDS, we add some constraints to this decoding process thus obtaining better performance.

Table 7 : Results of CDS generation in dataset Snips by two methods.

IC SF is using intent classification and slot filling with evaluation metric (intent accuracy, slot labelling accuracy and final accuracy).

Seq2Seq generates CDS based on utterance using an encoder-decoder.

In this paper, we propose the concept of cross-domain sketch (CDS) which extracts some shared information across domains, trying to fully utilize the cross-domain commonalities such as syntactic and phrasal similarity in human expressions.

We try to define CDS on two levels and give some examples to illustrate our idea.

We also present a general-to-detailed neural network (GDNN) for converting an utterance into a logic form based on meaning representation language (MRL) form.

The general network, which is meant to extract cross-domain commonalities, uses an encoderdecoder model to obtain CDS in a multi-task setup.

Then the detailed network generates the final domain-specific target by exploiting utterance and CDS simultaneously via attention mechanism.

Our experiments demonstrate the effectiveness of CDS and multi-task learning.

CDS is able to generalize over a wide range of tasks since it is an extraction to language expressions.

Therefore, in the future, we would like to perfect the CDS definition and extend its' ontology to other domains and tasks.

Besides, in this paper, we use attention mechanism to make use of CDS which is still a indirect way.

We would like to explore more effective ways such as constraint decoding to further enhance the role of CDS.

@highlight

General-to-detailed neural network(GDNN)  with Multi-Task Learning by incorporating cross-domain sketch(CDS) for semantic parsing