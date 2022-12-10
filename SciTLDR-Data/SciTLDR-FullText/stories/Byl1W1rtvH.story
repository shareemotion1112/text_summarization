To simultaneously capture syntax and semantics from a text corpus, we propose a new larger-context language model that extracts recurrent hierarchical semantic structure via a dynamic deep topic model to guide natural language generation.

Moving beyond a conventional language model that ignores long-range word dependencies and sentence order, the proposed model captures not only intra-sentence word dependencies, but also temporal transitions between sentences and inter-sentence topic dependences.

For inference, we develop a hybrid of stochastic-gradient MCMC and recurrent autoencoding variational Bayes.

Experimental results on a variety of real-world text corpora demonstrate that the proposed model not only outperforms state-of-the-art larger-context language models, but also learns interpretable recurrent multilayer topics and generates diverse sentences and paragraphs that are syntactically correct and semantically coherent.

Both topic and language models are widely used for text analysis.

Topic models, such as latent Dirichlet allocation (LDA) (Blei et al., 2003; Griffiths & Steyvers, 2004; Hoffman et al., 2013) and its nonparametric Bayesian generalizations (Teh et al., 2006; Zhou & Carin, 2015) , are well suited to extract document-level word concurrence patterns into latent topics from a text corpus.

Their modeling power has been further enhanced by introducing multilayer deep representation (Srivastava et al., 2013; Mnih & Gregor, 2014; Gan et al., 2015; Zhou et al., 2016; Zhao et al., 2018; .

While having semantically meaningful latent representation, they typically treat each document as a bag of words (BoW), ignoring word order (Griffiths et al., 2004; Wallach, 2006) .

Language models have become key components of various natural language processing (NLP) tasks, such as text summarization (Rush et al., 2015; Gehrmann et al., 2018) , speech recognition (Mikolov et al., 2010; Graves et al., 2013) , machine translation (Sutskever et al., 2014; Cho et al., 2014) , and image captioning (Vinyals et al., 2015; Mao et al., 2015; Xu et al., 2015; Gan et al., 2017; Rennie et al., 2017) .

The primary purpose of a language model is to capture the distribution of a word sequence, commonly with a recurrent neural network (RNN) (Mikolov et al., 2011; Graves, 2013) or a Transformer based neural network (Vaswani et al., 2017; Dai et al., 2019; Devlin et al., 2019; Radford et al., 2018; 2019) .

In this paper, we focus on improving RNN-based language models that often have much fewer parameters and are easier to perform end-to-end training.

While RNN-based language models do not ignore word order, they often assume that the sentences of a document are independent to each other.

This simplifies the modeling task to independently assigning probabilities to individual sentences, ignoring their orders and document context (Tian & Cho, 2016) .

Such language models may consequently fail to capture the long-range dependencies and global semantic meaning of a document (Dieng et al., 2017; .

To relax the sentence independence assumption in language modeling, Tian & Cho (2016) propose larger-context language models that model the context of a sentence by representing its preceding sentences as either a single or a sequence of BoW vectors, which are then fed directly into the sentence modeling RNN.

An alternative approach attracting significant recent interest is leveraging topic models to improve RNN-based language models.

Mikolov & Zweig (2012) use pre-trained topic model features as an additional input to the RNN hidden states and/or output.

Dieng et al. (2017) ; Ahn et al. (2017) combine the predicted word distributions, given by both a topic model and a language model, under variational autoencoder (Kingma & Welling, 2013) .

Lau et al. (2017) introduce an attention based convolutional neural network to extract semantic topics, which are used to extend the RNN cell.

learn the global semantic coherence of a document via a neural topic model and use the learned latent topics to build a mixture-of-experts language model.

Wang et al. (2019) further specify a Gaussian mixture model as the prior of the latent code in variational autoencoder, where each mixture component corresponds to a topic.

While clearly improving the performance of the end task, these existing topic-guided methods still have clear limitations.

For example, they only utilize shallow topic models with only a single stochastic hidden layer in their data generation process.

Note several neural topic models use deep neural networks to construct their variational encoders, but still use shallow generative models (decoders) (Miao et al., 2017; Srivastava & Sutton, 2017) .

Another key limitation lies in ignoring the sentence order, as they treat each document as a bag of sentences.

Thus once the topic weight vector learned from the document context is given, the task is often reduced to independently assigning probabilities to individual sentences (Lau et al., 2017; 2019) .

In this paper, as depicted in Fig. 1 , we propose to use recurrent gamma belief network (rGBN) to guide a stacked RNN for language modeling.

We refer to the model as rGBN-RNN, which integrates rGBN , a deep recurrent topic model, and stacked RNN (Graves, 2013; Chung et al., 2017) , a neural language model, into a novel larger-context RNN-based language model.

It simultaneously learns a deep recurrent topic model, extracting document-level multi-layer word concurrence patterns and sequential topic weight vectors for sentences, and an expressive language model, capturing both short-and long-range word sequential dependencies.

For inference, we equip rGBN-RNN (decoder) with a novel variational recurrent inference network (encoder), and train it end-to-end by maximizing the evidence lower bound (ELBO).

Different from the stacked RNN based language model in Chung et al. (2017) , which relies on three types of customized training operations (UPDATE, COPY, FLUSH) to extract multi-scale structures, the language model in rGBN-RNN learns such structures purely under the guidance of the temporally and hierarchically connected stochastic layers of rGBN.

The effectiveness of rGBN-RNN as a new larger-context language model is demonstrated both quantitatively, with perplexity and BLEU scores, and qualitatively, with interpretable latent structures and randomly generated sentences and paragraphs.

Notably, rGBN-RNN can generate a paragraph consisting of a sequence of semantically coherent sentences.

Denote a document of J sentences as D = (S 1 , S 2 , . . .

, S J ), where S j = (y j,1 , . . .

, y j,Tj ) consists of T j words from a vocabulary of size V .

Conventional statistical language models often only focus on the word sequence within a sentence.

Assuming that the sentences of a document are independent to each other, they often define

Tj t=2 p (y j,t | y j,<t ) p (y j,1 ) .

RNN based neural language models define the conditional probability of each word y j,t given all the previous words y j,<t within the sentence S j , through the softmax function of a hidden state h j,t , as

where f (·) is a non-linear function typically defined as an RNN cell, such as long short-term memory (LSTM) (Hochreiter & Schmidhuber, 1997) and gated recurrent unit (GRU) (Cho et al., 2014) .

These RNN-based statistical language models are typically applied only at the word level, without exploiting the document context, and hence often fail to capture long-range dependencies.

While Dieng et al. (2017); Lau et al. (2017); 2019) remedy the issue by guiding the language model with a topic model, they still treat a document as a bag of sentences, ignoring the order of sentences, and lack the ability to extract hierarchical and recurrent topic structures.

We introduce rGBN-RNN, as depicted in Fig. 1(a) , as a new larger-context language model.

It consists of two key components: (i) a hierarchical recurrent topic model (rGBN), and (ii) a stacked RNN based language model.

We use rGBN to capture both global semantics across documents and long-range inter-sentence dependencies within a document, and use the language model to learn the local syntactic relationships between the words within a sentence.

Similar to Lau et al. (2017); , we represent a document as a sequence of sentence-context pairs as

+ summarizes the document excluding S j , specifically (S 1 , ..., S j−1 , S j+1 , ..., S J ), into a BoW count vector, with V c as the size of the vocabulary excluding stop words.

Note a naive way is to treat each sentence as a document, use a dynamic topic model ( Blei & Lafferty, 2006) to capture the temporal dependencies of the latent topic-weight vectors, which is fed to the RNN to model the word sequence of the corresponding sentence.

However, the sentences are often too short to be well modeled by a topic model.

In our setting, as d j summarizes the document-level context of S j , it is in general sufficiently long for topic modeling.

Note during testing, we redefine d j as the BoW vector summarizing only the preceding sentences, i.e., S 1:j−1 , which will be further clarified when presenting experimental results.

Fig. 1 (a) , to model the time-varying sentence-context count vectors d j in document D, the generative process of the rGBN component, from the top to bottom hidden layers, is expressed as

where θ l j ∈ R K l + denotes the gamma distributed topic weight vectors of sentence j at layer l,

the transition matrix of layer l that captures cross-topic temporal dependencies,

the loading matrix at layer l, K l the number of topics of layer l, and τ 0 ∈ R + a scaling hyperparameter.

At j = 1, θ + can be factorized into the sum of Φ l+1 θ l+1 j , capturing inter-layer hierarchical dependence, and Π l θ l j−1 , capturing intra-layer temporal dependence.

rGBN not only captures the document-level word occurrence patterns inside the training text corpus, but also the sequential dependencies of the sentences inside a document.

Note ignoring the recurrent structure, rGBN will reduce to the gamma belief network (GBN) of Zhou et al. (2016) , which can be considered as a multi-stochastic-layer deep generalization of LDA (Cong et al., 2017a) .

If ignoring its hierarchical structure (i.e., L = 1), rGBN reduces to Poisson-gamma dynamical systems (Schein et al., 2016) .

We refer to the rGBN-RNN without its recurrent structure as GBN-RNN, which no longer models sequential sentence dependencies; see Appendix A for more details.

Different from a conventional RNN-based language model, which predicts the next word only using the preceding words within the sentence, we integrate the hierarchical recurrent topic weight vectors θ l j into the language model to predict the word sequence in the jth sentence.

Our proposed language model is built upon the stacked RNN proposed in Graves (2013); Chung et al. (2017), but with the help of rGBN, it no longer requires specialized training heuristics to extract multi-scale structures.

As shown in Fig. 1 (b) , to generate y j,t , the t th token of sentence j in a document, we construct the hidden states h l j,t of the language model, from the bottom to top layers, as

where LSTM l word denotes the word-level LSTM at layer l, W e ∈ R V are word embeddings to be learned, and x j,t = y j,t−1 .

Note a (1), the conditional probability of y j,t becomes

There are two main reasons for combining all the latent representations a 1:L j,t for language modeling.

First, the latent representations exhibit different statistical properties at different stochastic layers of rGBN-RNN, and hence are combined together to enhance their representation power.

Second, having "skip connections" from all hidden layers to the output one makes it easier to train the proposed network, reducing the number of processing steps between the bottom of the network and the top and hence mitigating the "vanishing gradient" problem (Graves, 2013).

To sum up, as depicted in Fig. 1 (a) , the topic weight vector θ l j of sentence j quantifies the topic usage of its document context d j at layer l. It is further used as an additional feature of the language model to guide the word generation inside sentence j, as shown in Fig. 1 (b) .

It is clear that rGBN-RNN has two temporal structures: a deep recurrent topic model to extract the temporal topic weight vectors from the sequential document contexts, and a language model to estimate the probability of each sentence given its corresponding hierarchical topic weight vector.

Characterizing the word-sentencedocument hierarchy to incorporate both intra-and inter-sentence information, rGBN-RNN learns more coherent and interpretable topics and increases the generative power of the language model.

Distinct from existing topic-guided language models, the temporally related hierarchical topics of rGBN exhibit different statistical properties across layers, which better guides language model to improve its language generation ability.

For rGBN-RNN, given

, the marginal likelihood of the sequence of sentence-context

where e

The inference task is to learn the parameters of both the topic model and language model components.

One naive solution is to alternate the training between these two components in each iteration: First, the topic model is trained using a sampling based iterative algorithm provided in ; Second, the language model is trained with maximum likelihood estimation under a standard cross-entropy loss.

While this naive solution can utilize readily available inference algorithms for both rGBN and the language model, it may suffer from stability and convergence issues.

Moreover, the need to perform a sampling based iterative algorithm for rGBN inside each iteration limits the scalability of the model for both training and testing.

Algorithm 1 Hybrid SG-MCMC and recurrent autoencoding variational inference for rGBN-RNN.

Set mini-batch size m and the number of layer L Initialize encoder and neural language model parameter parameter Ω, and topic model parameter

Randomly select a mini-batch of m documents consisting of J sentences to form a subset X = {di,1:

,j according to (6), and update Ω; Sample θ l i,j from (7) and (8) via

and {Φ l } L l=1 , will be described in Appendix C; end for

To this end, we introduce a variational recurrent inference network (encoder) to learn the latent temporal topic weight vectors θ

, the ELBO of the log marginal likelihood shown in (5) can be constructed as

which unites both the terms that are primarily responsible for training the recurrent hierarchical topic model component, and terms for training the neural language model component.

Similar to , we define q(θ

, a random sample from which can be obtained by transforming standard uniform variables

To capture the temporal dependencies between the topic weight vectors, both k l j and λ l j , from the bottom to top layers, can be expressed as

where h Rather than finding a point estimate of the global parameters

of the rGBN, we adopt a hybrid inference algorithm by combining TLASGR-MCMC described in Cong et al. (2017a); and our proposed recurrent variational inference network.

In other words, the global parameters

can be sampled with TLASGR-MCMC, while the parameters of the language model and variational recurrent inference network, denoted by Ω, can be updated via stochastic gradient descent (SGD) by maximizing the ELBO in (6).

We describe a hybrid variational/sampling inference for rGBN-RNN in Algorithm 1 and provide more details about sampling

with TLASGR-MCMC in Appendix C. We defer the details on model complexity to Appendix E.

To sum up, as shown in Fig. 1(c) , the proposed rGBN-RNN works with a recurrent variational autoencoder inference framework, which takes the document context of the jth sentence within a document as input and learns hierarchical topic weight vectors θ 1:L j that evolve sequentially with j. The learned topic vectors in different layer are then used to reconstruct the document context input and as an additional feature for the language model to generate the jth sentence.

We consider three publicly available corpora, including APNEWS, IMDB, and BNC.

The links, preprocessing steps, and summary statistics for them are deferred to Appendix D. We consider a recurrent variational inference network for rGBN-RNN to infer θ l j , as shown in Fig. 1(c) , whose number of hidden units in (8) are set the same as the number of topics in the corresponding layer. , which extracts the global semantic coherence of a document via a neural topic model, with the probability of each learned latent topic further adopted to build a mixture-of-experts language model; (vii) TGVAE (Wang et al., 2019) , combining a variational auto-encoder based neural sequence model with a neural topic model; (viii) GBN-RNN, a simplified rGBN-RNN that removes the recurrent structure of its rGBN component.

For rGBN-RNN, to ensure the information about the words in the jth sentence to be predicted is not leaking through the sequential document context vectors at the testing stage, the input d j in (8) only summarizes the preceding sentences S <j .

For GBN-RNN, following TDLM (Lau et al., 2017) and TCNLM , all the sentences in a document, excluding the one being predicted, are used to obtain the BoW document context.

As shown in Table 1 , rGBN-RNN outperforms all baselines, and the trend of improvement continues as its number of layers increases, indicating the effectiveness of assimilating recurrent hierarchical topic information.

rGBN-RNN consistently outperforms GBN-RNN, suggesting the benefits of exploiting the sequential dependencies of the sentence-contexts for language modeling.

Moreover, comparing Table 1 and Table 4 of Appendix E suggests rGBN-RNN, with its hierarchical and temporal topical guidance, achieves better performance with fewer parameters than comparable RNN-based baselines.

Note that for language modeling, there has been significant recent interest in replacing RNNs with Transformer (Vaswani et al., 2017) , which consists of stacked multi-head self-attention modules, and its variants (Dai et al., 2019; Devlin et al., 2019; Radford et al., 2018; 2019) .

While Transformer based language models have been shown to be powerful in various natural language processing tasks, they often have significantly more parameters, require much more training data, and take much longer to train than RNN-based language models.

For example, Transformer-XL with 12L and that with 24L (Dai et al., 2019), which improve Transformer to capture longer-range dependencies, have 41M and 277M parameters, respectively, while the proposed rGBN-RNN with three stochastic hidden layers has as few as 7.3M parameters, as shown in Table 4 , when used for language modeling.

From a structural point-of-view, we consider the proposed rGBN-RNN as complementary to rather than competing with Transformer based language models, and consider replacing RNN with Transformer to construct rGBN guided Transformer as a promising future extension.

, we use test-BLEU to evaluate the quality of generated sentences with a set of real test sentences as the reference, and self-BLEU to evaluate the diversity of the generated sentences (Zhu et al., 2018) .

Given the global parameters of the deep recurrent topic model (rGBN) and language model, we can generate the sentences by following the data generation process of rGBN-RNN: we first generate topic weight vectors θ L j randomly and then downward propagate it through the rGBN as in (2) to generate θ <L j .

By assimilating the random draw topic weight vectors with the hidden states of the language model in each layer depicted in (3), we generate a corresponding sentence, where we start from a zero hidden state at each layer in the language model, and sample words sequentially until the end-of-the-sentence symbol is generated.

Comparisons of the BLEU scores between different methods are shown in Fig. 2 , using the benchmark tool in Texygen (Zhu et al., 2018) ; We show below BLEU-3 and BLEU-4 for BNC and defer the analogous plots for IMDB and APNEWS to Appendix G and H. Note we set the validation dataset as the ground-truth.

For all datasets, it is clear that rGBN-RNN yields both higher test-BLEU and lower self-BLEU scores than related methods do, indicating the stacked-RNN based language model in rGBN-RNN generalizes well and does not suffer from mode collapse (i.e., low diversity).

Hierarchical structure of language model: In Fig. 3 , we visualize the hierarchical multi-scale structures learned with the language model of rGBN-RNN and that of GBN-RNN, by visualizing the L 2 -norm of the hidden states in each layer, while reading a sentence from the APNEWS validation set as "the service employee international union asked why cme group needs tax relief when it is making huge amounts of money?"

As shown in Fig. 3(a) , in the bottom hidden layer (h1), the L 2 norm sequence varies quickly from word to word, except within short phrases such "service employee", "international union," and "tax relief," suggesting layer h1 is in charge of capturing short-term local dependencies.

By contrast, in the top hidden layer (h3), the L 2 norm sequence varies slowly and exhibits semantic/syntactic meaningful long segments, such as "service employee international union," "asked why cme group needs tax relief," "when it is," and "making huge amounts of," suggesting that layer h3 is in charge of capturing long-range dependencies.

Therefore, the language model in 48 budget lawmakers gov.

revenue vote proposal community legislation 57 lawmakers pay proposal legislation credit session meeting gambling 60 budget gov.

revenue vote costs mayor california conservative 57 generated sentence: the last of the four companies and the mississippi inter national speedway was voted to accept the proposal .

60 generated sentence: adrian on thursday issued an officer a news release saying the two groups will take more than $ 40,000 for contacts with the private nonprofit .

48 generated sentence: looming monday , the assembly added a proposal to balance the budget medicaid plan for raising the rate to $ 142 million , whereas years later , to $ 200 million .

lawmaker proposal legislation approval raising audit senate 75 generated sentence: the state senate would give lawmakers time to accept the retirement payment .

48-57-75 generated sentence: the proposal would give them a pathway to citizenship for the year before , but they don't have a chance to participate in the elections .

inc gambling credit assets medicaid investment 62 generated sentence: the gambling and voting department says it was a chance of the game .

48-57-62 generated sentence: the a r k a n s a s s e n a t e h a s purchased a $ 500 million state bond for a proposed medicaid expansion for a new york city .

11 budget revenue loan gains treasury incentives profits 11 generated sentence: the office of north dakota has been offering a $ 22 million bond to a $ 68 m i l l i o n b u d g e t w i t h t h e proceeds from a escrow account .

48-60-11 generated sentence: a new report shows the state has been hit by a number of shortcomings in jindal 's budget proposal for the past decade .

84 gov.

vote months conservation ballot reform fundraising 84 generated sentence: the u.s. sen.

joe mccoy in the democratic party says russ of the district , must take the rest of the vote on the issues in the first half of the year .

it was partially the other in the republican caucus that raised significant amounts of spending last year and ended with cuts from a previous government shutdown .

Figure 4 : Topics and their temporal trajectories inferred by a three-hidden-layer rGBN-RNN from the APNEWS dataset, and the generated sentences under topic guidance (best viewed in color).

Top words of each topic at layer 3, 2, 1 are shown in orange, yellow and blue boxes respectively, and each sentence is shown in a dotted line box labeled with the corresponding topic index.

Sentences generated with a combination of topics in different layers are at the bottom of the figure.

rGBN-RNN can allow more specific information to transmit through lower layers, while allowing more general higher level information to transmit through higher layers.

Our proposed model have the ability to learn hierarchical structure of the sequence, despite without designing the multiscale RNNs on purpose like Chung et al. (2017) .

We also visualize the language model of GBN-RNN in Fig. 3(b) ; with much less smoothly time-evolved deeper layers, GBN-RNN fails to utilize its stacked RNN structure as effectively as rGBN-RNN does.

This suggests that the language model is much better trained in rGBN-RNN than in GBN-RNN for capturing long-range temporal dependencies, which helps explain why rGBN-RNN exhibits clearly boosted BLEU scores in comparison to GBN-RNN.

We present an example topic hierarchy inferred by a three-layer rGBN-RNN from APNEWS.

In Fig. 4 , we select a large-weighted topic at the top hidden layer and move down the network to include any lower-layer topics connected to their ancestors with sufficiently large weights.

Horizontal arrows link temporally related topics at the same layer, while top-down arrows link hierarchically related topics across layers.

For example, topic 48 of layer 3 on "budget, lawmakers, gov., revenue," is related not only in hierarchy to topic 57 on "lawmakers, pay, proposal, legislation" and topic 60 of the lower layer on "budget, gov., revenue, vote, costs, mayor," but also in time to topic 35 of the same layer on "democratic, taxes, proposed, future, state."

Highly interpretable hierarchical relationships between the topics at different layers, and temporal relationships between the topics at the same layer are captured by rGBN-RNN, and the topics are often quite specific semantically at the bottom layer while becoming increasingly more general when moving upwards.

Sentence generation under topic guidance: Given the learned rGBN-RNN, we can sample the sentences both conditioning on a single topic of a certain layer and on a combination of the topics from different layers.

Shown in the dotted-line boxes in Fig. 4 , most of the generated sentences conditioned on a single topic or a combination of topics are highly related to the given topics in terms of their semantical meanings but not necessarily in key words, indicating the language model is successfully guided by the recurrent hierarchical topics.

These observations suggest that rGBN-RNN has successfully captured syntax and global semantics simultaneously for natural language generation.

Sentence/paragraph generation conditioning on a paragraph: Given the GBN-RNN and rGBN-RNN learned on APNEWS, we further present the generated sentences conditioning on a paragraph, as shown in Fig. 5 .

To randomly generate sentences, we encode the paragraph into a hierarchical latent representation and then feed it into the stacked-RNN.

Besides, we can generate a paragraph with rGBN-RNN, using its recurrent inference network to encode the paragraph into a dynamic hierarchical latent representation, which is fed into the language model to predict the word sequence Document Generated Sentences with GBN-RNN Generated Sentences with rGBN-RNN the proposal would also give lawmakers with more money to protect public safety , he said .

the proposal , which was introduced in the house on a vote on wednesday , has already passed the senate floor to the house .

Generated temporal Sentences with rGBN-RNN (Paragraph) the senate sponsor (…) , a house committee last week removed photo ids issued by public colleges and universities from the measure sponsored by republican rep.

susan lynn , who said she agreed with the change .

the house approved the bill on a 65-30 vote on monday evening .

but republican sen.

bill ketron in a statement noted that the upper chamber overwhelmingly rejected efforts to take student ids out of the bill when it passed 21-8 earlier this month .

ketron said he would take the bill to conference committee if needed .

if the house and senate agree , it will be the first time they 'll have to seek their first meeting .

the city commission voted last week to approve the law , which would have allowed the council to approve the new bill .

senate president pro tem joe scarnati said the governor 's office has never resolved the deadline for a vote in the house .

the proposal is a new measure version of the bill to enact a senate committee to approve the emergency manager 's emergency license .

the house gave the bill to six weeks of testimony , but the vote now goes to the full house for consideration .

jackson signed his paperwork wednesday with the legislature .the proposal would also give lawmakers with more money to protect public safety , he said .

"a spokesman for the federal department of public safety says it has been selected for a special meeting for the state senate to investigate his proposed law .

a new state house committee has voted to approve a measure to let idaho join a national plan to ban private school systems at public schools .

the campaign also launched a website at the university of california , irvine , which are studying the current proposal .

in each sentence of the input paragraph.

It is clear that both the proposed GBN-RNN and rGBN-RNN can successfully capture the key textual information of the input paragraph, and generate diverse realistic sentences.

Interestingly, the rGBN-RNN can generate semantically coherent paragraphs, incorporating contextual information both within and beyond the sentences.

Note that with the topics that extract the document-level word cooccurrence patterns, our proposed models can generate semantically-meaningful words, which may not exist in the original document.

We propose a recurrent gamma belief network (rGBN) guided neural language modeling framework, a novel method to learn a language model and a deep recurrent topic model simultaneously.

For scalable inference, we develop hybrid SG-MCMC and recurrent autoencoding variational inference, allowing efficient end-to-end training.

Experiments results conducted on real world corpora demonstrate that the proposed models outperform a variety of shallow-topic-model-guided neural language models, and effectively generate the sentences from the designated multi-level topics or noise, while inferring interpretable hierarchical latent topic structure of document and hierarchical multiscale structures of sequences.

For future work, we plan to extend the proposed models to specific natural language processing tasks, such as machine translation, image paragraph captioning, and text summarization.

Another promising extension is to replace the stacked-RNN in rGBN-RNN with Transformer, i.e., constructing an rGBN guided Transformer as a new larger-context neural language model.

GBN-RNN: {y 1:T , d} denotes a sentence-context pair, where d ∈ Z Vc + represents the document-level context as a word frequency count vector, the vth element of which counts the number of times the vth word in the vocabulary appears in the document excluding sentence y 1:T .

The hierarchical model of a L-hidden-layer GBN, from top to bottom, is expressed as

The stacked-RNN based language model described in (3) is also used in GBN-RNN.

Statistical inference: To infer GBN-RNN, we consider a hybrid of stochastic gradient MCMC (Welling & Teh, 2011; Patterson & Teh, 2013; Li et al., 2015; Ma et al., 2015; Cong et al., 2017a) , used for the GBN topics φ l k , and auto-encoding variational inference (Kingma & Welling, 2013; Rezende et al., 2014) , used for the parameters of both the inference network (encoder) and RNN.

More specifically, GBN-RNN generalizes Weibull hybrid auto-encoding inference (WHAI) of : it uses a deterministic-downward-stochastic-upward inference network to encode the bag-of-words representation of d into the latent topic-weight variables θ l across all hidden layers, which are fed into not only GBN to reconstruct d, but also a stacked RNN in language model, as shown in (3), to predict the word sequence in y 1:T .

The topics φ l k can be sampled with topic-layeradaptive stochastic gradient Riemannian (TLASGR) MCMC, whose details can be found in Cong et al. (2017a); , omitted here for brevity.

Given the sampled topics φ l k , the joint marginal likelihood of {y 1:T , d} is defined as

) is used to provide a ELBO of the log joint marginal likelihood as

and the training is performed by maximizing E pdata(

, where both k l and λ l are deterministically transformed from d using neural networks.

Distinct from a usual variational auto-encoder whose inference network has a pure bottom-up structure, the inference network here has a determisticupward-stoachstic-downward ladder structure .

is the written portion of the British National Corpus (Consortium, 2007) .

Following the preprocessing steps in Lau et al. (2017) , we tokenize words and sentences using Stanford CoreNLP (Klein & Manning, 2003) , lowercase all word tokens, and filter out word tokens that occur less than 10 times.

For the topic model, we additionally exclude stopwords 2 and the top 0.1% most frequent words.

All these corpora are partitioned into training, validation, and testing sets, whose summary statistics are provided in Table 2 of the Appendix.

in (2) , and the parameters of the variational recurrent inference network (encoder), consisting of RNN

The language model component is parameterized by LSTM l word in (3) and the coupling vectors g l described in Appendix B. We summarize in Table 3 the complexity of rGBN-RNN (ignoring all bias terms), where V denotes the vocabulary size of the language model, E the dimension of word embedding vectors, V c the size of the vocabulary of the topic model that excludes stop words, H w l the number of hidden units of the word-level LSTM at layer l (stacked-RNN language model), H s l the number of hidden units of the sentence-level RNN at layer l (variational recurrent inference network), and K l the number of topics at layer l. Table 4 further compares the number of parameters between various RNN-based language models, where we follow the convention to ignore the word embedding layers.

Some models in Table 1 are not included here, because we could not find sufficient information from their corresponding papers or code to accurately calculate the number of model parameters.

Note when used for language generation at the testing stage, rGBN-RNN no longer needs its topics {Φ l }, whose parameters are hence not counted.

Note the number of parameters of the topic model component is often dominated by that of the language model component.

Left panel is BLEU-3 and right is BLEU-4, and a better BLEU score would fall within the lower right corner, where black point represents mean value and circles with different colors denote the elliptical surface of probability of BLEU in a two-dimensional space.

@highlight

We introduce a novel larger-context language model to simultaneously captures syntax and semantics, making it capable of generating highly interpretable sentences and paragraphs