Learning knowledge graph embeddings (KGEs) is an efficient approach to knowledge graph completion.

Conventional KGEs often suffer from limited knowledge representation, which causes less accuracy especially when training on sparse knowledge graphs.

To remedy this, we present Pretrain-KGEs, a training framework for learning better knowledgeable entity and relation embeddings, leveraging the abundant linguistic knowledge from pretrained language models.

Specifically, we propose a unified approach in which we first learn entity and relation representations via pretrained language models and use the representations to initialize entity and relation embeddings for training KGE models.

Our proposed method is model agnostic in the sense that it can be applied to any variant of KGE models.

Experimental results show that our method can consistently improve results and achieve state-of-the-art performance using different KGE models such as TransE and QuatE, across four benchmark KG datasets in link prediction and triplet classification tasks.

Knowledge graphs (KGs) constitute an effective access to world knowledge for a wide variety of NLP tasks, such as question-answering, entity linking and information retrieval.

A typical KG such as Freebase (Bollacker et al., 2008) and WordNet (Miller, 1995) consists of a set of triplets in the form of (h, r, t) with the head entity h and the tail entity t as nodes and relations r as edges in the graph.

A triplet represents the relation between two entities, e.g., (Steve Jobs, founded, Apple Inc.).

Despite their effectiveness, KGs in real applications suffer from incompleteness and there have been several attempts for knowledge graph completion among which knowledge graph embedding is one of prominent approaches.

Knowledge graph embedding (KGE) models have been designed extensively in recent years (Bordes et al., 2013; Ji et al., 2015; Lin et al., 2015; Sun et al., 2019; Ebisu and Ichise, 2018; Nickel et al., 2011; Kazemi and Poole, 2018; Trouillon et al., 2016; Zhang et al., 2019) .

The general methodology of these models is to model entities and relations in vector spaces based on a score function for triplets (h, r, t).

The score function measures the plausibility of each candidate triplet (h, r, t) compared to corrupted false triplets (h , r, t) or (h, r, t ).

However, traditional KGE models often suffer from limited knowledge representation due to the simply symbolic representation of entities and relations.

Some recent works take advantages of both fact triplets and textual description to enrich knowledge representation (Socher et al., 2013a; Xu et al., 2017; Xiao et al., 2017; Xie et al., 2016; , but without exploitation of contextual information of the textual descriptions.

Moreover, much of this research effort has been dedicated to developing novel architectures for knowledge representation without applications to KGE models.

Unlike many existing works which try to propose new architectures for KGEs or knowledge representation, we focus on model-agnostic pretraining technique for KGE models.

We present a unified training framework named as PretrainKGEs which consists of three phases: fine-tuning phase, initializing phase and training phase (see Fig. 1 ).

During the fine-tuning phase, we learn better knowledgeable entity and relation representations via pretrained language models using textual descriptions as input sequence.

Different from previous works incorporating textual information into knowledge representation, we use pretrained langauge models such as BERT (Devlin et al., 2019) to better understand textual description by making full use of syntactic and semantic information in large- scale corpora on which BERT is pretrained.

Thus, we enable to incorporate rich linguistic knowledge learned by BERT into entity and relation representations.

Then during the initializing phase, we use knowledgeable entity and relation representations to initialize entity and relation embeddings so that the initialized KGEs inherit the rich knowledge.

Finally, during the training phase, we train a KGE model the same way as a traditional KGE model to learn entity and relation embeddings.

Extensive experiments using six public KGE models across four benchmark KG datasets show that our proposed training framework can consistently improve results and achieve state-of-the-art performance in link prediction and triplet classification tasks.

Our contributions are as follows:

• We propose a model-agnostic training framework for learning knowledge graph embeddings by first learning knowledge representation via pretrained language models.

• Results on several benchmark datasets show that our method can improve results and achieve state-of-the-art performance over variants of knowledge graph embedding models in link prediction and triplet classification tasks.

• Further analysis demonstrates the effects of knowledge incorporation in our method and shows that our Pretrain-KGEs outperforms baselines especially in the case of fewer training triplets, low-frequency and the out-ofknowledge-base (OOKB) entities.

2 Background and Related Work

For each head entity h and tail entity t with their corresponding entity embeddings E h , E t , and each relation r with its relation embeddings R r , we formulate KGE models as follows:

where v h , v r , v t ∈ F d are the learnt vectors for each head entity, relation, and tail entity respectively, The model is then optimized to calculate a higher score for true triplets than corrupted false ones.

According to the score function, KGE models can be roughly divided into translational models and semantic matching models (Wang et al., 2017) .

Translational models popularized by TransE (Bordes et al., 2013) learn vector embeddings of the entities and the relations, and consider the relation between the head and tail entity as a translation between the two entity embeddings, i.e., in the form of v h + v r ≈ v t when the candidate triplet (h, r, t) holds.

Since TransE has problems when dealing with 1-to-N, N-to-1 and N-to-N relations, different translational models are proposed subsequently to define various relational patterns, such as TransH (Wang et al., 2014) , TransR (Lin et al., 2015) , TransD (Ji et al., 2015) , RotatE (Sun et al., 2019) , and TorusE (Ebisu and Ichise, 2018) .

On the other hand, semantic matching models define a score function to match latent semantics of the head, tail entity and the relation.

For instance, RESCAL (Nickel et al., 2011) , DistMult , SimplE (Kazemi and Poole, 2018) , and ComplEx (Trouillon et al., 2016 ) adopt a bilinear approach to model entities and relations for KGEs.

Specifically, ComplEx learns complexvalued representations of entities and relations in complex space, while DistMult, SimplE, and RESCAL embed entities and relations in the traditional real number field.

The recent state-of-the-art, QuatE (Zhang et al., 2019) represents entities as hypercomplex-valued embeddings and models relations as rotations in the quaternion space.

Both translational models and semantic matching models learn entity and relation embeddings in spite of different embedding spaces.

However, these KGE models only use structural information observed in triplets without incorporating external knowledge resources into KGEs, such as textual description of entities and relations.

Thus, the embeddings of entities and relations suffer from limited knowledge representation.

We instead propose a unified approach to introduce rich linguistic knowledge into KGEs via pretrained language models.

In a knowledge graph dataset, names of each entity and relation are provided as textual description of entities and relations.

Socher et al. (2013a) first utilize textual information to represent entities by averaging word embeddings of entity names.

Following the word averaging method, Li et al. (2016) improve the coverage of commonsense resources in ConceptNet (Speer and Havasi, 2012) by mining candidate triplets from Wikipedia.

They leverage a word averaging model to convert entity and relation names into name vectors.

Other recent works also leverage textual description to enrich knowledge representation but ignore contextual information of the textual descriptions (Socher et al., 2013a; Xu et al., 2017; Xiao et al., 2017; Xie et al., 2016; .

Instead, our method exploits rich contextual information via pretrained models.

Recent approaches to modeling language representations offer significant improvements over embeddings, especially pretrained deep contextualized lanaguge representation models such as ELMo (Peters et al., 2018) , BERT (Devlin et al., 2019) , GPT-2 (Radford et al., 2019) , and T5 (Raffel et al., 2019) .

These deep language models learn better contextualized word presentations, since they are pretrained on large-scale free text data, which make full use of syntactic and semantic information in the large corpora.

In this work, we use BERT, a bidirectional Transformer encoder to learn entity and relation representation given textual description.

Therefore, by incorporating the plentiful linguistic knowledge learned by pretrained language models, our proposed method can learn better knowledgeable entity and relation representations for subsequent KGE learning.

In this section, we will introduce our unified training framework Pretrain-KGEs and provide details of learning knowledgeable entity and relation representations via BERT.

An overview of Pretrain-KGEs is shown in Fig. 1 .

The framework consists of three phases: finetuning phase, initializing phase, and training phase.

Our major contribution is the fine-tuning phase with the initializing phase, which incorporates rich knowledge into KGEs via pretained language models, i.e., BERT that enables to exploit contextual information of textual description for entities and relations.

By initializing embeddings with knowledgeable entity and relation representations, our training framework improves KGE models to learn better entity and relation embeddings.

Fine-tuning Phase Given textual description of entities and relations such as entity names and relation names, we first encode the textual descriptions into vectors via pretrained language models to represent entities and relations respectively.

We then project the entity and relation representations into two separate vector spaces to get the entity encoder Enc e (·) for each entity e and the relation encoder Enc r (·) for each relation r. Formally, Enc e (·) and Enc r (·) output entity and relation representations as:

where v h , v r , and v t represents encoding vectors of the head entity, the relation, and the tail entity in a triplet (h, r, t) respectively.

For details of Enc e (·) and Enc r (·), see section 3.2.

Given the entity and relation representations, we then calculate the score of a triplet to measure its plausibility in Eq. 2.

For instance, if TransE is adopted, the score function is v h + v r − v t .

After fine-tuning, the knowledge representation is used in the following initializing phase.

Initializing Phase Given the knowledgeable entity and relation representation, we initialize entity embeddings E and relation embeddings R for a KGE model instead of random initialization.

Specifically, E = [E 1 ; E 2 ; · · · ; E k ] ∈ F k×d and R = [R 1 ; R 2 ; · · · ; R l ] ∈ F l×d in which ";" denotes concatenating column vectors into a matrix.

k and l denote the total number of entities and relations respectively.

F satisfies R ⊆ F and d denotes the embedding dimension.

Then E i ∈ F d represents the embedding of entity with index i and R j ∈ F d represents the embedding of relation with index j.

During the initializing phase, we use the representation vector of entity with index i encoded by the entity encoder Enc e (·) as the initialized embedding E i for training KGE models to learn entity embeddings.

Likewise, the representation vector of relation with index j encoded by the relation encoder Enc r (·) is considered as the initialized embedding R j for training KGE models to learn relation embeddings.

Training Phase After initializing entity and relation embeddings with knowledgeable entity and relation representations, we train a KGE model in the same way as a traditional KGE model.

We calculate the score of each training triplet in Eq. 1 and Eq. 2 with the same score function in the finetuning phase.

Finally, we optimize the entity embedding E and the relation embedding R using the same loss function of the corresponding KGE model.

For example, if TransE and the max-margin loss function with negative sampling are adopted, the loss in the training phase is calculated as:

where (h, r, t) and (h , r , t ) represent a candidate and a corrupted false triplet respectively, γ denotes the margin, · + = max(·, 0), and f (·) denotes score function of TransE (Bordes et al., 2013) .

To learn better knowledge representation of entities and relations given textual description, we first encode the textual description through Bert (Devlin et al., 2019) , a bidirectional Transformer encoder which is pretrained on large-scale corpora and thus learns rich contextual information of texts by making full use of syntactic and semantic information in the large corpora.

We define T (e) and T (r) as the textual description of entities and relations respectively.

The textual description can be words, phrases, or sentences providing information about entities and relations such as names of entities and relations or definitions of word senses.

For example, the definition of entity e = Nyala.n.1 in WordNet is city in Sudan.

Then T (Nyala.n.1) = Nyala : city in Sudan.

Given the textual descriptions of entities and relations T (e) and T (r), Bert(·) converts T (e) and T (r) into entity representation and relation representation respectively in a vector space R n (n denotes the vector size).

We then project the entity and relation representations into two separate vector spaces F d through linear transformations.

Formally, we get the entity encoder Enc e (·) for each entity e and the relation encoder Enc r (·) for each relation r as:

Enc r (r) = σ(W r Bert(T (r)) + b r )

where W e , W r ∈ F d×n , b e , b r ∈ F d , and σ :

The entity and relation representation encoded by Enc e (·) and Enc r (·) are then used to initialize entity and relation embeddings for a KGE model.

We evaluate our proposed training framework on four benchmark KG datasets: WN18 (Bordes et al., 2013) In our experiments, we perform link prediction task (filtered setting) mainly with triplet classification task.

The link prediction task aims to predict either the head entity h given the relation r and the tail entity t or the tail entity given the head entity and the relation, while triplet classification aims to judge whether a candidate triplet is correct or not.

For the link prediction task, we generate corrupted false triplets (h , r, t) and (h, r, t ) using negative sampling.

For n test triplets, we get their ranks r = (r 1 , r 2 , · · · , r n ) and calculate standard evaluation metrics: Mean Rank (MR), Mean Reciprocal Rank (MRR) and Hits at N (H@N).

For triplet classification, we follow the evaluation protocol in Socher et al. (2013b) and adopt the accuracy metric (Acc) to evaluate our training method.

To evaluate the universality of our training framework Pretrain-KGEs, we select multiple public KGE models as baselines including translational models:

• TransE (Bordes et al., 2013) , the translationalbased model which models the relation as translations between entities;

2 Detailed statistics of datasets are in Appendix.

A. Table 3 : Link prediction and Triplet classification ("Class") results using QuatE. "Name" means using names of entities and relations as textual description.

"Definition" means using names of entities and relations as well as definitions of word senses as textual description.

• RotatE (Sun et al., 2019) , the extension of translational-based models which introduces complex-valued embeddings to model the relations as rotations in complex vector space;

• pRotatE (Sun et al., 2019) , a variant of RotatE where the modulus of complex entity embeddings are constrained and only phase information is involved;

and semantic matching models:

• DistMult , a semantic matching model where each relation is represented with a diagonal matrix;

• ComplEx (Trouillon et al., 2016) , the extension of semantic matching model which embedds entities and relations in complex space.

• QuatE (Zhang et al., 2019) , the recent stateof-the-art KGE model which learns entity and relation embeddings in the quaternion space.

We present results for the Pretrain-KGEs algorithm in Table 1, Table 2 and Table 3 .

Table 1 shows the link prediction results on four benchmark KG datasets using six public KGE models.

Table 2 compares the results on WordNet of using entity names and relation names to the results of adding definitions of word senses as additional textual description for entities.

Table 3 demonstrates the state-of-the-art performance of our proposed method in both link prediction and triplet classification tasks 3 .

From the results, we can observe that:

(1) Our unified training framework can be applied to multiple variants of KGE models in spite of different embedding spaces, and achieves improvements over TransE, DistMult, ComplEx, RotatE, pRotatE and QuatE on most evaluation metrics, especially on MR but still being competitive on MRR (see detailed analysis of MR and MRR in section 5.2.1).

Yet, it verifies the universality of our training framework.

The reason is that our method incorporates rich linguistic knowledge into entity and relation representation via pretrained language models to learn better knowledgeable representation for the embedding initialization in KGE models.

For the effects of knowledge incorporation, see detailed analysis in section 5.2.

(2) Our training framework can also facilitate in improving the recent state-of-the-art even further over QuatE on most evaluation metrics in link prediction and triplet classification tasks.

It verifies the effectiveness of our proposed training framework.

In this section, we provide further analysis of Pretrain-KGEs' performance in the case of fewer training triplets, low-frequency entities and the out-of-knowledge-base (OOKB) entities which are particularly hard to handle due to lack of knowledge representation.

We also evaluate the effects of knowledge incorporation into entity and relation embeddings by demonstrating the sensitivity of MR and MRR metrics and visualizing the process of knowledge incorporation.

We also evaluate our training framework in the case of fewer training triplets on WordNet and test its performance on entities of varying frequency in test triplets on FB15K as well as the performance on the OOKB entities in test triplets on WordNet as shown in Fig. 2a-2e .

To test the performance of our training framework given fewer training triplets, we conduct experiments on WN18 and WN18RR by feeding varying number of training triplets to a KGE model.

We use traditional TransE as one of the baselines.

Baseline-TransE does not utilize any textual description and randomly initializes entity and relation embeddings before the training phase.

Thus, it suffers from learn knowledgeable KGEs when training triplets become fewer.

In contrast, our Pretrain-TransE first learns knowledgeable entity and relation representations by encoding textual description through BERT, and uses the learned representations to initialize KGEs for TransE. In this way, we enable to incorporate rich linguistic knowledge from BERT into initizalized entity and relation embeddings so that TransE can perform better given fewer training triplets.

On the other hand, to verify the effectiveness of BERT during the fine-tuning phase, we also set the word averaging model following Li et al. (2016) to be the entity encoder Enc e (·) in Eq. 3 for comparison 4 .

From the results, we can observe that although the word averaging model contributes to better performance of TransE on fewer training triplets compared to Baseline-TransE, it does not learn knowledgeable entity and relation representations as well as BERT because BERT can better understand textual descriptions of entities and relations by exploiting rich contextual information of the textual descriptions.

Moreover, by utilizing definitions of word senses as additional textual description of entities, the results show that our training method achieves the best performance in the case of fewer training triplets.

Besides, we also evaluate our training framework for its performance on entities of varying frequency in training triplets on FB15K.

From the results in Fig. 2c , we can observe that our training framework outperforms Baseline-TransE especially on infrequent entities.

The reason is that traditional TransE method cannot learn good representation of infrequent entities due to inadquate dataset information and lack of textual description of entities.

When training triplets becomes fewer, there can be increasing OOKB entities in test triplets not observed at training time.

Traditional training method of KGE models cannot address the OOKB entity problem since it randomly gives scores of test triplets containing OOKB entities due to random initialization of entity embeddings before training.

In contrast, our training method initializes entity embeddings with knowledgeable entity representation.

Thus, we also evaluate our training method in the case of OOKB entities.

From the results in Fig. 2d-2e , we can observe that our training framework can solve the OOKB entity problem on WordNet dataset and performs best when using BERT to encode textual description of entities and In (a)-(e), "TransE" means TransE baseline with random initialization; "Avg" means a word averaging model using entity names and definitions provided in WordNet as textual description; "Name" refers to our proposed Pretrain-TransE method using entity names and relation names as textual description; "Definition" refers to our proposed Pretrain-TransE method using names of entities and relations as well as definitions in WordNet as textual description.

In (d)-(e), "Random" means randomly giving scores of triplets.

In (f), "1"-"5" denotes the number of iterations during the training phase are 10000-50000 updates.

relations including their names and definitions of word senses.

Our training framework has natural advantages over traditional training method of KGE models since we learn better knowledgeable entity and relation representation via BERT before training a KGE model.

This section verifies the effectiveness of knowledge incorporation during the fine-tuning phase.

We show the performance of Baseline-TransE and Pretrain-TransE on WN18RR as iteration increases during the training phase in Fig. 2f .

We analyze the changing trend of MR and MRR in Theorem 2.

Formally, for n test triplets, we get corresponding ranks in link prediction task r = (r 1 , r 2 , · · · , r n ), and MR(r) = n i=1 r i n;

Theorem 1.

5 Sensitivity of MR and MRR metrics MR is more sensitive to tricky triplets than MRR.

Formally, for r = (r 1 , r 2 , · · · , r n ) and r i > r j (triplet i is worse-learnt than triplet j):

where f k (r) denotes ∂f ∂r k

(f ∈ {MR, MRR}) and means the sensitivity of metric f to triplet k.

In Figure 2c , we can observe that there is better performance on high-frequency triplets than low-frequency ones which are more tricky to handle, since there is less information in datasets provided for low-frequency triplets.

According to The- orem 2, we can thus suggest that MR is more sensitive to low-frequency triplets while MRR is more sensitive to high-frequency triplets.

Reasons for the increasing MR of Pretrain-TransE in Fig. 2f are illustrated in the following.

We visualize the knowledge learning process of Baseline-TransE and our Pretrain-TransE in Fig. 3a 3c.

We select top five common supersenses in WN18: plant, animal, act, person and artifact, among which the last three supersenses are all relevant to the concept of human beings and thus can be considered to constitute one common supersense.

In Fig. 3a , we can observe that Baseline-TransE learns entity and relation embeddings for triplets containing the five supersenses but does not distinguish embeddings between plant, animal and the other three supersenses.

In contrast, Fig. 3b shows that our Pretrain-TransE can further distinguish embeddings between different supersenses, especially separating supersenses related to human beings from others.

The main reason is that we can learn better knowledgeable entity and relation representation via BERT by incorporating rich linguistic knowledge into entity and relation embeddings during the initializing phase.

However, during the training phase, our PretrainTransE gradually learns different KGEs from those in the initializing phase.

Fig. 3c shows that it is due to the oblivion of partial linguistic knowledge incorporated into entity and relation embeddings as the KGEs learn more information contained in datasets at training time.

This process can account for the increasing MR results of Pretrain-TransE during the training phase in Fig. 2f .

But the absolute values of MR and MRR for our Pretrain-TransE are overtly lower than those for TransE baseline, which demonstrates that our training framework enables to learn better knowledgeable entity and relation representation and there still remains incorporated knowledge in entity and relation embeddings during the training phase.

To conclude, during the training phase, TransE baseline learns original knowledge contained in datasets.

Instead, our proposed method first learns rich linguistic knowledge from BERT, and continues to learn knowledge from datasets while losing partial knowledge learned from BERT.

Yet finally, there still remains knowledge from BERT incorporated in entity and relation embeddings during the training phase.

We present Pretrain-KGEs, a simple and efficient pretraining technique for knowledge graph embedding models.

Pretrain-KGEs is a general technique that can be applied to any KGE model.

It contributes to learn better knowledgeable entity and relation representations from pretrained language models, which are leveraged during the initializing and the training phases for a KGE model to learn entity and relation embeddings.

Through extensive experiments, we demonstrate state-of-the-art performances using this effective pretraining technique on various benchmark datasets.

Further, we verify the effectiveness of our method by demonstrating promising results in the case of fewer training triplets, infrequent and OOKB entities which are particularly hard to handle due to lack of knowledge representation.

We finally analyze the effects of knowledge incorporation by demonstrating the sensitivity of MR and MRR metrics and visualizing the process of knowledge incorporation.

A Detailed Implementation A.1 Implementation Our implementations of TransE (Bordes et al., 2013) , DistMult , ComplEx (Trouillon et al., 2016) , RotatE (Sun et al., 2019) , pRotatE (Sun et al., 2019) are based on the framework provided by Sun et al. (2019) 6 .

Our implementation of QuatE is based on on the framework provided by Zhang et al. (2019) 7 .

In fine-tuning phase, we adopt the following non-linear pointwise function σ(·):

x i e i ∈ F (where F can be real number filed R, complex number filed C or quaternion number ring H):

where x i ∈ R and e i is the K-dimension hypercomplex-value unit.

For instance, when K = 1, F = R; when K = 2, F = C, e 1 = i (the imaginary unit); when K = 4, F = H, e 1,2,3 = i, j, k (the quaternion units).

The score functions of baselines are listed in Table 4 .

TransE (Bordes et al., 2013) v h + vr − vt R DistMult v h , vr, vt R ComplEx (Trouillon et al., 2016) Re( v h , vr,vt ) C RotatE (Sun et al., 2019) v h vr − vt C pRotatE (Sun et al., 2019) 2C sin θ h +θr −θ t 2 C QuatE (Zhang et al., 2019) v h ⊗vr vt H Table 4 : Score functions and corresponding F of previous work.

v h , v r , v t denote head, tail and relation embeddings respectively.

R, C, H denote real number field, complex number field and quaternion number division ring respectively.

· denotes L1 norm.

· denotes generalized dot product.

Re(·) denotes the real part of complex number.

· denotes the conjugate for complex vectors.

⊗ denotes circular correlation, denotes Hadamard product.

C denotes a constraint on the pRotatE model: v h 2 = v t 2 = C.· denotes the normalized operator.

θ h , θ r , θ t denote the angle of complex vectors v h , v r , v t respectively.

We also implement the word-averaging baseline to utilize the entitiy names and entity definition in WordNet to represent the entity embedding better.

Formally, for entitiy e and its textual description T (e) = w 1 w 2 · · · w L , where w i denotes the i-th token in sentence T (e) and T (e) here together utilizing the entitiy names and entity definition in WordNet.

where u i denotes the word embedding of token w i , which is a trainable randomly initialized parameter and will be trained in the pretraining phase.

We also adopt our three-phase training method to train word-averaging baseline.

Similarly, E = [E 1 ; E 2 ; · · · ; E k ] ∈ F k×d and R = [R 1 ; R 2 ; · · · ; R l ] ∈ F l×d denote entity and relation embeddings.

In pretraining phase, for head entity h, tail entity t and relation r, the score function is calculated as: v h , v r , v t = Avg(h), R r , Avg(t)

where R r denotes the relation embedding of relation r. In initializing phase, similar to our proposed model, we initialize E i with Avg(e i ).

In training phase, we optimize E and R with the same training method to TransE baseline.

We evaluate our proposed training framework on four benchmark KG datasets: WN18 (Bordes et al., 2013) , WN18RR (Dettmers et al., 2018) , FB15K (Bordes et al., 2013) and FB15K-237 (Toutanova and Chen, 2015) .

We list detailed statistics of datasets are in Table 5 .

The hyper-parameters of are listed in Table 6 .

@highlight

We propose to learn knowledgeable entity and relation representations from Bert for knowledge graph embeddings.