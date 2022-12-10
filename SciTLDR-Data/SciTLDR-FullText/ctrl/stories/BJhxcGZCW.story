Online healthcare services can provide the general public with ubiquitous access to medical knowledge and reduce the information access cost for both individuals and societies.

To promote these benefits, it is desired to effectively expand the scale of high-quality yet novel relational medical entity pairs that embody rich medical knowledge in a structured form.

To fulfill this goal, we introduce a generative model called Conditional Relationship Variational Autoencoder (CRVAE), which can discover meaningful and novel relational medical entity pairs without the requirement of additional external knowledge.

Rather than discriminatively identifying the relationship between two given medical entities in a free-text corpus, we directly model and understand medical relationships from diversely expressed medical entity pairs.

The proposed model introduces the generative modeling capacity of variational autoencoder to entity pairs, and has the ability to discover new relational medical entity pairs solely based on the existing entity pairs.

Beside entity pairs, relationship-enhanced entity representations are obtained as another appealing benefit of the proposed method.

Both quantitative and qualitative evaluations on real-world medical datasets demonstrate the effectiveness of the proposed method in generating relational medical entity pairs that are meaningful and novel.

Increasingly, people engage in health services on the Internet BID11 .

The healthcare services can provide the general public with ubiquitous access to medical knowledge and reduce the information access cost significantly.

The relational medical entity pair, which consists of two medical entities with a semantic connection between them, is an intuitive representation that distills human medical reasoning processes in a structured form.

The medical relationships discussed in this paper are binary ones.

For example, the Disease Cause − −−− →Symptom relationship indicates a "Cause" relationship from a disease entity to a symptom entity that is caused by this disease, such as the medical entity pairs <Synovitis, Joint Pain>. For the relationship Symptom Belongto − −−−−− →Department, we may have a relational medical entity pair such as <Stiffness of a Joint, Orthopedics>.The ability to understand, reason and generalize is central to human intelligence BID27 .

However, it possesses significant challenges for machines to understand and reason about the relationships between two entities BID31 .

Real-world relational medical entity pairs possess certain challenging properties to deal with: First, as the medical research develops, many medical relationships among medical entities that were once neglected due to the underdeveloped medical knowledge now need to be discovered.

An increasing number of relationships will be formed among a large number of medical entities.

Also, various linguistic expressions can be used for the same medical entity.

For example, Nose Plugged, Blocked Nose and Sinus Congestion are symptom entities that share the same meaning but expressed very differently.

Moreover, one medical relationship may instantiate entity pairs with varying granularities or relationship strength.

For instance, Disease Cause − −−− →Symptom may include entity pairs like <Rhinitis, Nose Plugged> as a coarse-grained entity pair, while < Acute Rhinitis, Nose Plugged>, <Chronic Rhinitis, Nose Plugged> are considered fine-grained entity pairs.

As for the relationship strength, <Cold, Fatigue> has greater relationship strength than <Cold, Ear Infections> as cold rarely cause serious complications such as ear infections.

To effectively expand the scale of high-quality yet novel relational medical entity pairs, relation extraction methods BID8 BID2 are proposed to examine whether or not a semantic relationship exists between two given entities given a context.

Although the existing relation extraction methods BID1 BID3 BID30 BID39 BID6 BID37 achieve decent performance in identifying the relationship for given entity pairs, those methods require contexts such as sentences retrieved from a large free-text corpus, from existing domain-specific knowledge graphs BID0 , or from web tables and links BID21 .

As medical relationships in the real-world are becoming more and more complex and diversely expressed, existing relation extraction methods suffer from the data sparsity problem where it is hard to obtain additional external knowledge that covers all possible entity pairs, e.g. free-text corpus where two entities co-occur in the same sentence with a relationship between them.

Therefore, it is crucial and appealing for us to discover meaningful relational medical entity pairs solely based on existing medical entity pairs, without the requirement of a well-maintained context as an additional external knowledge.

Furthermore, most relation extraction methods adopt a discriminative approach that learns to distinguish entity pairs of one relationship from the other BID41 BID22 , or to identify meaningful entity pairs from randomly sampled negative entity pairs with no relationships BID4 BID32 .

Those methods need to iterate over the combination of all possible entity pairs and check each of them to discover new entity pairs.

Such discriminative approach is tedious and labor-intensive.

It is challenging yet rewarding for us to understand medical relationships intrinsically from the existing entity pairs.

Specifically, in the medical domain, the diversely expressed medical entity pairs offer great advantages for us to ultimately understand medical relationships and discover high-quality relational medical entity pairs solely from existing meaningful medical entity pairs.

Problem Studied: We propose a novel research problem called RElational Medical Entity-pair DiscoverY (REMEDY), which aims at modeling relational medical entity pairs solely from the existing entity pairs.

Also, it aims to discover meaningful and novel entity pairs pertaining to a certain medical relationship in a generative fashion, without sophisticated feature engineering and the requirement of external knowledge such as free-text corpora.

Proposed Model: A generative model named Conditional Relationship Variational Autoencoder (CRVAE) is introduced for relational medical entity pair discovery.

It is unlikely to create meaningful, novel relational medical entity pairs without intrinsically understanding each medical relationship, more specifically, understanding the relationships between every two medical entities that instantiate a particular relationship.

CRVAE fully explores the generative modeling capacity which roots in Bayesian inference while incorporating deep learning for powerful hands-free feature engineering.

CRVAE is trained to encode each relational medical entity pair into a latent space conditioned on the relationship type.

The encoding process addresses relationship-enhanced entity representations, interactions between entities as well as expressive latent variables.

The latent variables are decoded to reconstruct entity pairs.

Once the model is trained, we can sample directly from the distribution of latent variables and decode them into high-quality and novel relational medical entity pairs.

Overall, CRVAE has three notable strengths:CRVAE models the intrinsic relations between medical entity pairs directly based on the existing meaningful relational medical entity pairs, without the requirement of additional external contexts for entity pair extraction.

Existing relation extraction methods usually rely on the free-text corpus to decide whether a candidate entity pair it mentions is meaningful or not.

The CRVAE only utilizes the existing entity pairs and pre-trained word vector as initial entity representations which are trained separately.

CRVAE is able to generate entity pairs for a particular relationship, even if it observes existing entity pairs only for that particular relationship.

Unlike most discriminative methods which harness discrepancies among different relationships to distinguish the relationship of an entity pair from the other, or from randomly constructed negative entity pairs with no relations.

The CRVAE understands the intrinsic medical relation from diversely expressed medical entity pairs and discovers meaningful, novel entity pairs of a particular relationship that we specified.

CRVAE generates novel entity pairs by a density-based sampling strategy in the generator.

The generator samples directly from the latent space based on the density of hidden parameters.

With the hands-free feature engineering by deep neural networks, the model is able to discover meaningful and novel entity pairs which does not exist in the training data.

The contributions of this paper can be summarized as follows:• We study the Relational Medical Entity-pair Discovery (REMEDY) problem, which aims to expand the scale of high-quality yet novel relational medical entity pairs without maintaining large-scale context information such as the free-text corpus.• We propose a generative model named Conditional Relationship Variational Autoencoder (CRVAE) that discovers relational medical entity pairs for a given relationship, solely from the diversely expressed entity pairs without sophisticated feature engineering.• We obtain relationship-enhanced entity representations as an appealing benefit of the proposed model.

In this section, we introduce the Conditional Relationship Variational Autoencoder (CRVAE) model for the REMEDY problem.

The proposed model consists of three modules: encoder, decoder, and generator.

The encoder module takes relational medical entity pairs and a relationship indicator as the input, trained to intrinsically understand each relationship by translating and mapping the entity pair to a latent space as Q φ .

The decoder is jointly trained to reconstruct the entity pairs as P θ .

The generator model shares the same structure with the decoder, and it directly samples from the learned latent variable distribution to creatively generate meaningful medical relational entity pairs for a particular relationship.

Figure 1 gives an overview of the proposed model.

The model takes a tuple <e h , e t > and a relationship indicator r as the input, where e h and e t are head and tail medical entity of a relationship r. For example, e h ="Synovitis" and e t ="Joint Pain", while the corresponding r is an indicator for Disease DISPLAYFORM0 To effectively represent medical entities, pre-trained word embeddings that embody rich semantic information can be obtained as initial entity representations for e h and e t .

For simplicity, Skip-gram BID24 ) is adopted to obtain 200-dimensional word embeddings trained separately and unsupervisely on a publicly accessible medical corpus.

After a table lookup on the pre-trained word vector matrix W embed ∈ R V ×D E where V is the vocabulary size (usually tens of thousands) and D E is the dimension of the initial entity representation (usually tens or hundreds), embed h ∈ R 1×D E and embed t ∈ R 1×D E are derived as the initial embedding of medical entities.

With the initial entity representation embed h and embed t and their relationship indicator r, the encoder first translates and then maps entity pairs to a latent space as Q φ (z|embed h , embed t , r).

The initial embedding obtained from word embedding reflects semantic and categorical information.

However, it is not specifically designed to model the medical relationship among medical entities (See observations in Section 3.4.3).

To get entity representations that address relationship information, the encoder learns to translate each medical entity from its initial embedding to a relationshipenhanced embedding that distills relationship information.

For example, a non-linear transformation can be used: translate(x) = f (x·W trans +b trans ) where f can be an non-linear activation function such as the Exponential Linear Unit (ELU) BID7 .

W trans ∈ R D E ×D R is the weight variable and b trans ∈ R 1×D R is the bias where D R is the dimension for relationship-enhanced embeddings.trans h = translate(embed h), trans t = translate(embed t)( 1) are obtained as relationship-enhanced embeddings for e h and e t .

The relationship-enhanced entity representation trans h and trans t are concatenated trans ht = [trans h , trans t ] and mapped to the latent space by multiple fully connected layers.

For example, we can obtain a variable l ht that addresses the relationship information, as well as entity interactions from two medical entities, by applying three consecutive non-linear fully connected layers on trans ht .

As a variational inference model, we assume a simple Gaussian distribution of Q φ (z|embed h , embed t , r) for the relational medical entity pair <e h , e t > with a relationship r.

Therefore, for each relational medical entity pair <e h , e t > and a relationship indicator r, a mean vector µ and a variance vector σ 2 can be learned as latent variables to model Q φ (z|embed h , embed t , r): DISPLAYFORM0 where a one-hot indicator r ∈ R 1×|R| is used for the medical relationship r and |R| is the number of all relationships.

W µ , W σ ∈ R (Dl ht +|R|)×D L are weight terms and b µ , b σ ∈ R 1×D L are bias terms.

D L is the dimension for latent variables and D l ht is the dimension for l ht .

To stabilize the training, we model the variation vector σ 2 by its log form log σ 2 (to be explained in Equation 15).

Once we obtain latent variables µ, σ 2 for an input tuple <e h , e t > which has the relationship r, the decoder uses latent variables and the relationship indicator r to reconstruct the relational medical entity pair.

The decoder implements the P θ (embed h , embed t |z, r).Given µ, σ 2 , it is intuitive to sample the latent value z from the distribution N (µ, σ 2 ) directly.

However, such operator is not differentiable thus optimization methods failed to calculate its gradient.

To solve this problem, a reparameterization trick is introduced in to divert the non-differentiable part out of the network.

Instead of directly sampling from N (µ, σ 2 ), we sample from a standard normal distribution ∼ N (0, I) and then convert it back to z by z = µ + σ .

In this way, sampling from does not depend on the network.

Similarly as the use of multiple non-linear fully connected layers for the mapping in the encoder, multiple non-linear fully connected layers are used for an inverse mapping in the decoder.

After the inverse mapping we obtain trans ht ∈ R 1×2D R .

The first D R dimensions of trans ht are considered as a decoded relationship-enhanced embedding for e h , while the last D R dimensions are for e t : DISPLAYFORM0 where trans h , trans t ∈ R 1×D R .

trans h and trans t are further inversely translated back to the initial embedding space R D E : DISPLAYFORM1 where embed h , embed t ∈ R 1×D E are considered as reconstructed representations for embed h and embed t .

Inspired by the loss function of the conditional variational autoencoder (CVAE) BID33 , the loss function of CRVAE is formulated to minimize the variational lower bound: DISPLAYFORM0 where Q φ (z|embed h , embed t , r) is a simple Gaussian distribution used to approximate the unknown true distribution P θ (z|embed h , embed t , r).

P θ (z|r) describes the true latent distribution z given a certain relationship r and E [log (P θ (embed h , embed t |z, r))] estimates the maximum likelihood.

A closed-form solution for the first term can be derived as: DISPLAYFORM1 where µ is the mean vector and σ 2 is the variance vector.

l in the subscript indicates the l-th dimension of the vector.

Details for obtaining the closed-form solution are given in Appendix AThe second term penalizes the maximum likelihood, which is the conditional probability P θ (embed h , embed t |z, r) of a certain entity pair <e h , e t > given the latent variable z and the relationship indicator r. The mean squared error (MSE) is adopted to calculate the difference between <embed h , embed t > and <embed h , embed t >: DISPLAYFORM2 where · 2 is the vector 2 norm.

To minimize the L CRVAE , existing optimizers such as Adadelta (Zeiler, 2012) can be used.

Furthermore, a warm-up technique introduced in BID34 can let the training start with deterministic and gradually switch to variational, by multiplying β to the first term.

The final loss function used for training is formulated as: DISPLAYFORM3 where β is initialized as 0 and increase by 0.1 at the end of each training epoch, until it reaches 1.0 as its maximum.

When we have a certain relationship r in our mind that the generated relational medical entity pairs should belong to, a density-based sampling method is introduced for the generator to sampleẑ from the latent space given a certain relationship r.ˆe The generator that generate meaningful, novel relational medical entity pairs from the latent space.

Instead of using the latent variable z provided by certain µ and log σ 2 in the encoding process from a certain e h , e t and r, the generator tries to sampleẑ directly from P θ (ẑ|r) to get the latent space valueẑ for a particular relationship r. Onceẑ is obtained, the decoder structure is used to decode the relational medical entity pair.

FIG0 illustrates the generative process.

The denser region in the latent space P θ (ẑ|r) indicates that more densely entity pairs are located in the manifold.

Therefore, a sampling method that considers the density distribution of P θ (ẑ|r) samples more often from the denser regions in the latent space so as to preserve the true latent space distribution of the sampled values.

Specifically, for each relationship r, the densitybased sampling samplesẑ directly from P θ (ẑ|r) ∼ N (0, I), when trained properly.

The resulting vectorsêmbed h andêmbed t are mapped back to their entities in the initial embedding space R 1×D E , namelyê h andê t , by finding the nearest neighbor of the initial entity representation using W embed .

The -2 distance measure is used for the nearest neighbor search.

The dataset consists of 46.02k real-world relational medical entity pairs in Chinese from a Chinese online healthcare forum www.xywy.com.

The data set covers six different types of medical relationships.

TAB1 shows the collection of relational medical entity pairs used in this study.

70% data are used for training and 30% for validation.

We use 200-dimensional word embeddings learned with the Skip-gram algorithm in BID24 , trained from 6 million text corpus on the Chinese online healthcare forum as the initial entity representation.

The vocabulary covers 126,270 words.

We use Xavier initialization BID12 for weight variables and zeros for biases.

A wide range of hyperparameter configurations are tested with the proposed model.

See Appendix B for detailed hyperparameter analysis.

For each medical relationship, 1000 entity pairs are generated.

Three evaluation metrics are introduced to quantitatively measure the generated relational medical entity pairs: quality, support, and novelty.

Quality Since it is hard for the machine to evaluate whether a relational medical entity pair is meaningful or not, human annotation is involved in assessing the quality of the generated relational medical entity pairs.

A human annotation task is deployed on Amazon Mechanical Turk for annotation (Task shown in Appendix C).

Similar as the precision metric adopted in BID2 , the quality 1 is measured by: quality = # of entity pairs that are meaningful # of all the generated entity pairs .Support Besides the quality metric, a support metric is developed to quantitatively measure the degree of belongingness of a generated entity pair to a relationship.

For each generated relational medical entity pair <ê h ,ê t > and a candidate relationship r c , the support score is calculated by: DISPLAYFORM0 where distance(êmbed h ,êmbed t ) calculates the distance between the vectorêmbed h −êmbed t and N N rc (êmbed h −êmbed t ) using distance measure such as cosine distance.

The N N rc implements the nearest neighbor search over the embed h − embed t space on all the training data which has the relationship r c .

For each generated medical entity pair, the support scores for all the candidate relationships are normalized so that they sum up to one: DISPLAYFORM1 The relationship having the highest score is considered as the estimated relationship for <ê h ,ê t > while the relationship r given during the generating process is considered as the ground truth for <ê h ,ê t >.

The final support value is based on the accuracy of the estimated relationship and the ground truth relationship.

Novelty The ability to generate novel relational medical entity pairs is one of our key contributions.

Due to different scope of medical knowledge among individuals, human annotators are not able to precisely evaluate the novelty.

We measure the novelty of the generation process by: novelty = # of entity pairs that do not exist in the dataset # of all the generated entity pairs .

Considering that no known methods are currently available for the REMEDY problem, we compare the performance of the following models:• CRVAE-MONO: The proposed model which only takes one single type of relational medical entity pairs in both training and generation.

For each type of relationship, we train a separate CRVAE only with entity pairs having that relationship.• RVAE: The unconditional version of the model CRVAE where the relationship indicator r is not provided during model training and generation.• CRVAE-RAND: The proposed model CRVAE with a random sampling based generator.

Unlike the density-based sampling adopted in CRVAE, the generator of CRVAE-RAND samples randomly from the latent space.• CRVAE: The proposed method where relational medical entity pairs that belong to all types of relationships are used to train the model altogether.

The training is conditioned on relationships and density-based sampling is used.• CRVAE-WA: The proposed method with the warm-up strategy introduced in Section 2.3.

We summarize the performance of the proposed method, along with other alternatives, in TAB2 CRVAE-MONO demonstrates the power of generative models in terms of learning the intrinsic representation and generating new entity pairs only given one type of relationship during the training (Quality: 0.6698, Support: 0.9550, Novelty: 0.5118).

For CRVAE-RAND, although it generates highly novel (0.9952) entity pairs that are not seen in the training data, the generated entity pairs are of low quality (0.2550).

By comparing CRVAE and CRVAE-RAND, we can see that the densitybased sampling enables the generation of high-quality entity pairs that results in +47.58% in quality and +52.84% in support.

The warm up technique adopted in CRVAE-WA is able to give CRVAE a further performance boost, where all measures improve consistently (+4.09% in quality, +2.43% in support and +5.11% in novelty).

As a qualitative measure, we also provide relational medical entity pairs generated by the proposed model.

For example, the entity pair <痢疾(dysentery), 肠(intestine)> is generated given the medical relationship DiseaseCause − −−− →Body Part, while entity pairs such as <阿米巴痢疾(amebic dysentery）, 肠(intestine)> and <细菌性痢疾(bacterial dysentery), 胸部(chest)> are found in the training data.

More entity pairs generated by the proposed method can be found in Appendix D.

Unlike discriminative models which utilize the difference between instances of different classes to discriminate instances from one class to another, the proposed method purely learns from the existing relational medical entity pairs to generate new entity pairs.

To validate such appealing property, Table 3 compares the fine-grained quality, support and novelty of entity pairs generated by CRVAE-MONO and CRVAE on each relationship.

Table 3 : Quality, support and novelty metrics of the generated relational medical entity pairs by CRVAE-MONO and CRVAE.

As shown in Table 3 , the CRVAE-MONO on each relationship achieves a reasonable performance, which shows the capability of generative models in understanding every single medical relationship individually.

Furthermore, when all types of entity pairs are trained and generated altogether in CRVAE, we observe a consistent improvement in not only quality but also novelty.

To validate the effectiveness of the density-based sampling for the generator, we compare the proposed method with CRVAE-RAND where a random sampling strategy is adopted.

From TAB2 we can see that the random sampling strategy in CRVAE-RAND tends to generate more entity pairs that are not seen in the existing dataset.

However, we observe a significant reduction in the quality and support of the generated entity pairs when compared with CRVAE which adopts a density-based sampling.

The dense region in the latent space indicates that more densely entity pairs are located.

Therefore, in CRVAE, the quality and support of the generated entity pairs benefit from sampling more often at denser regions in the latent space, resulting in less novel but higher quality entity pairs.

As mentioned in Section 2.1.1, the translating layer adjusts the original embedding to get relationship-enhanced entity representations.

In the experiments, we study the embedding spaces before/after translation and observe that in the original embedding space, the Skip-gram tends to put entities that share similar context (e.g. muscle strain and pull-up) in proximity.

While after relationship-enhancing, entities with similar functionalities in the same medical relationship are nearby with each other (e.g. heart malformations and chromosome abnormalities).

See Appendix E for details.

One of our key contributions is that with proper training, the proposed method can generate relational medical entity pairs given a certain relationship.

That is, the ability to infer new entity pairs for a particular relationship.

Besides seamlessly incorporating this idea in the model design, we also visualize latent space of CRVAE and RVAE in order to show the conditional inference ability.

See Appendix F for details.

Generative Models: Recent years have witnessed an increasing interests in the research topic of generative models, which aims to generate observable data values based on some hidden parameters.

Various generative models have been developed, such as Generative Adversarial Networks (GANs) BID13 BID29 and Variational Autoencoders (VAEs) BID18 BID33 BID15 BID26 .

Unlike GANs which generate data based on arbitrary noises, the VAE setting adopted in this paper is more expressive for our task since it tries to model the underlying probability distribution of the data by latent variables so that new data from that distribution can be sampled accordingly.

There are some generative models and applications considering data in different modalities, such as generating images BID28 BID14 BID9 or natural language texts BID5 BID23 BID16 BID38 .

As far as we know, the relational medical entity pair discovery problem we studied in this paper, which suits the generative purpose, has not been studied in a generative perspective.

Relationship Extraction:

There is another related research area that studies relation extraction, which usually amounts to examining whether or not a relation exists between two given entities BID8 .

Most relationship extraction methods require large amounts of high-quality external information, such as a large text corpus BID3 BID1 BID30 BID39 and knowledge graphs BID37 BID6 BID35 .

However, it is tedious and time-consuming to check each possible pair over all combinations of entities in the entity space.

Thus, we propose an effective generative method that generates meaningful and novel relational medical entity pairs directly.

Also, it is time consuming to collect and prepare a large corpus that covers all the mentions of those entity pairs, which makes it difficult to apply those methods.

In this work, our model does not rely on additional external corpus for entity pair discovery.

Moreover, previous discriminative models usually need negative samples for supervised training.

For example, BID32 trains the model to distinguish entity pairs with a relationship from randomly generated entity pairs as negative samples, while our model is can understand the medical relationship only from rational relational medical entity pairs thus even works when being fed with entity pairs having the same relationship type.

To effectively expand the scale of high-quality relational medical entity pairs which store the medical knowledge, a novel generative model named Conditional Relationship Variational Autoencoder (CRVAE) is introduced for Relational Medical Entity-pair Discovery (REMEDY).

The proposed model fully explores the generative modeling ability while incorporates deep learning for powerful hands-free feature engineering.

Unlike traditional relation extraction tasks which require additional contexts for extraction and need negative samples for discriminative training, the proposed method learns to intrinsically understand the medical relations from diversely expressed medical entity pairs, without the requirement of external context information.

Moreover, it is able to generate meaningful, novel entity pairs for a given type of medical relationship.

The relationshipenhanced entity representations have the potential to improve other NLP tasks.

The performance of the proposed method is evaluated on real-world medical data both quantitatively and qualitatively.

Inspired by the loss function of the conditional variational autoencoder (CVAE) ; BID33 , the loss function of CRVAE is formulated to minimize the variational lower bound: DISPLAYFORM0 where the first term minimizes the KL divergence loss between the unknown true distribution P θ (z|embed h , embed t , r) which is hard to sample from and a simple distribution Q φ (z|embed h , embed t , r).The second term models the entity pairs by log (P θ (embed h , embed t |r)).

The above equation can be reformulated as: DISPLAYFORM1 where P θ (z|r) describes the true latent distribution z given a certain relationship r and E [log (P θ (embed h , embed t |z, r))] estimates the maximum likelihood.

Since we want to sample from P θ (z|r) in the generator, the first term aims to let to let Q φ (z|embed h , embed t , r) to be as close as possible to P θ (z|r) which has a simple distribution N (0, I) so that it is easy to sample from.

Furthermore, if P θ (z|r) ∼ N (0, I) and Q(z|embed h , embed t , r) ∼ N (µ, σ 2 ), then a close-form solution for the first term can be derived as: DISPLAYFORM2 where l in the subscript indicates the l-th dimension of the vector.

Since it is more stable to have exponential term than a log term, we model log σ 2 as σ 2 which results in the final closed-form of Equation 15: DISPLAYFORM3

We train the proposed model with a wide range of hyperparameter configurations, which are listed in TAB3 .

We vary the batch size from 64 to 256.

The dimension D R for translating the initial entity embeddings is set from 64 to 2048.

We try two to seven hidden layers from trans ht to l ht and from [z, r] to trans ht , with different non-linear activation functions.

For each hidden layer, the hidden unit number D H is set from 2 to 1024.

The latent dimension D L is set from 2 to 200. , 128, 256, 512, 640, 768, 1024 , 1280 , 1536 , 1792 , 2048 D H 2, 4, 8, 16, 32, 64, 128, 256, 512, 640, 768, 1024 D L 2, 3, 4, 5, 10, 20, 50, 100, 200 Activation ELU (Clevert et al., 2015 , ReLU BID25 , Sigmoid, Tanh Optimizer Adadelta (Zeiler, 2012), Adagrad BID10 , Adam BID17 , RMSProp Tieleman & Hinton (2012) D GENERATED RELATIONAL MEDICAL ENTITY PAIRS TAB6 shows meaningful entity pairs generated by the proposed method.

To show the effectiveness of relationship-enhancement, TAB7 shows the nearest neighbors of a disease entity 生殖道畸形 (genital tract malformation) and a symptom entity 肌肉拉伤 (muscle strain) in their original embedding space, as well as the space after relationship-enhancing.

From these cases we can see that the original entity representations trained with Skip-gram BID24 tend to put entities in proximity when they appear in similar contexts.

In the first case, the entity 生殖道畸形(genital tract malformation) is in close proximity to 不孕 (infertility), 不孕 症 (acyesis).

In the second case, entities that have similar context like 引体向上 (pull-up) and 运动 量 (amount of exercise) are found near by the entity 肌肉拉伤(muscle strain).

The translation layer adjusts the original entity representation so that they are more suitable for the Relational Medical Entity-pair Discovery task.

The nearest neighbors in the adjusted space are not necessarily entities that co-occur in the same context, but more relation-wise similar with the given entity.

For example, 心脏畸形 (heart malformations) and 染色体异常 (chromosome abnormalities) may not be semantically similar with the given word 生殖道畸形(genital tract malformation), but they may serve similar functionalities in a Disease Cause − −−− →Symptom relationship.

Figure 4 shows the values of validation data after being mapped into the µ space using RVAE (left) and CRVAE (right), respectively.

The values are colored based on their ground truth relationship indicators.

The left figure indicates that when the relationship indicator r is not given during the training/validation, RVAE is still able to map different relationships into various regions in the latent space, while a single distribution models all types of relationships.

Such property is appealing for an unsupervised model, but since the relationship indicator r is not given, RVAE fails to generate entity pairs having a particular relationship, unless we manually assign a boundary for each relationship in the latent space.

The right figure shows that when the relationship indicator r is incorporated during the training, CRVAE learns to let each relationship have a unified latent representation.

A separate but nearly identical distribution is used to model each medical relationship.

Such property enables the generator of CRVAE to sample from a relationship-independent, unified latent space for diversities regarding the generation, while the relationship indicator r given in CRVAE's generator provides categorical information on the type of relationship to generate.

运动量 (amount of exercise) Figure 4 : The latent variable µ of RVAE (left) and CRVAE (right) on the validation data, presented in a two-dimensional space after dimension reduction using Primary Component Analysis.

<|TLDR|>

@highlight

Generatively discover meaningful, novel entity pairs with a certain medical relationship by purely learning from the existing meaningful entity pairs, without the requirement of additional text corpus for discriminative extraction.

@highlight

Presents a variational autoencoder for generating entity pairs given a relation in a medical setting.

@highlight

In the medical context, this paper describes the classic problem of "knowledge base completion" from structured data only.