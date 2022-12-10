We present a novel method to precisely impose tree-structured category information onto word-embeddings, resulting in ball embeddings in higher dimensional spaces (N-balls for short).

Inclusion relations among N-balls implicitly encode subordinate relations among categories.

The similarity measurement in terms of the cosine function is enriched by category information.

Using a geometric construction method instead of back-propagation, we create large N-ball embeddings that satisfy two conditions: (1) category trees are precisely imposed onto word embeddings at zero energy cost; (2) pre-trained word embeddings are well preserved.

A new benchmark data set is created for validating the category of unknown words.

Experiments show that N-ball embeddings, carrying category information, significantly outperform word embeddings in the test of nearest neighborhoods, and demonstrate surprisingly good performance in validating categories of unknown words.

Source codes and data-sets are free for public access \url{https://github.com/gnodisnait/nball4tree.git} and \url{https://github.com/gnodisnait/bp94nball.git}.

Words in similar contexts have similar semantic and syntactic information.

Word embeddings are vector representations of words that reflect this characteristic BID16 BID19 and have been widely used in AI applications such as question-answering BID24 , text classification BID22 , information retrieval BID14 , or even as a building-block for a unified NLP system to process common NLP tasks BID2 .

To enhance semantic reasoning, researchers proposed to represent words in terms of regions instead of vectors.

For example, BID5 extended a word vector into a region by estimating the log-linear probability of weighted feature distances and found that hyponym regions often do not fall inside of their hypernym regions.

By using external hyponym relations, she obtained 95.2% precision and 43.4% recall in hypernym prediction for a small scale data set.

Her experiments suggest that regions structured by hyponym relations may not be located within the same dimension as the space of word embeddings.

Yet, how to construct strict inclusion relations among regions is still an open problem when representing hypernym relations.

In this paper, we restrict regions to be n dimensional balls (N -ball for short) and propose a novel geometrical construction approach to impose tree-structured category information onto word embeddings.

This is guided by two criteria: (1) Subordinate relations among categories shall be implicitly and precisely represented by inclusion relations among corresponding N -balls.

This way, the energy costs of imposing structure will be zero; (2) Pre-trained word embeddings shall be well-preserved.

Our particular contributions are as follows: (1) The proposed novel geometric approach achieves zero energy costs of imposing tree structures onto word-embeddings.

(2) By considering category information in terms of the boundary of an N -ball, we propose a new similarity measurement that is We create a large data set of N -ball embeddings using the pre-trained GloVe embeddings and a large category tree of word senses extracted from Word-Net 3.0.The remainder of our presentation is structured as follows: Section 2 presents the structure of N -ball embeddings; Section 3 describes the geometric approach to construct N -ball embeddings; Section 4 presents experiment results; Section 5 briefly reviews related work; Section 6 concludes the presented work, and lists on-going research.

FIG0 (a), so they are not RCC regions that can be either open or closed, or even a mixture, thus avoiding a number of problems BID4 BID3 .

We distinguish two topological relations between N -balls: being inside and being disconnected from as illustrated in FIG0 (b, c) .

DISPLAYFORM0 can be measured by the result of subtracting the sum of radius r u and the distance between their central vectors from radius r w .

Formally, we define DISPLAYFORM1 Similarly, → u disconnecting from → w can be measured by the result of subtracting the distance between their center vectors from the sum of their radii.

Formally, we define DISPLAYFORM2

Similarity is normally measured by the cosine value of two vectors, e.g., BID16 .

For N -balls, the similarity between two balls can be approximated by the cosine value of their central vectors.

Formally, given two N -balls can be defined as cos( ⃗ A u , ⃗ A w ).

One weakness of the method is that we do not know the boundary of the lowest cos value below which two word senses are not similar.

Using category information, we can define that two word senses are not similar, if they have different direct hypernyms.

Formally, DISPLAYFORM0 N -ball embeddings encode two types of information: (1) word embeddings and (2) tree structures of hyponym relations among word senses.

A word can have several word senses.

We need to create a unique vector to describe the location of a word sense in hypernym trees.

We introduce a virtual root (*root*) to be the parent of all tree roots, fix word senses in alphabetic order in each layer, and number each word sense based on the fixed order.

A fragment tree structure of word senses is illustrated in FIG1 (a).

The path of a word sense to *root* can be uniquely identified by the numbers along the path.

For example, the path from *root* to flower.n.03 is [entity.n.01, abstraction.n.06, measure.n.02, flower.n As a word and its hypernym may not co-occur in the same context, their co-occurrence relations can be weak, and the cosine similarity of their word embeddings could even be less than zero.

For example, in GloVe embeddings, cos(ice cream, dessert)= −0.1998, cos(tuberose, plant)= −0.2191.

It follows that the hypernym ball must contain the origin point of the n-dimensional space, if this hypernym ball contains its hyponym, as illustrated in Figure 3 (a).

When this happens to two semantically unrelated hypernyms, N -balls of the two unrelated hypernyms shall partially overlap, as they both contain the original point.

For example, the ball of desert shall partially overlap with the ball of plant.

This violates our first criterion.

To avoid such cases, we require that no N -ball shall contain the origin O. This can be computationally achieved by adding dimensions and realized by introducing a spatial extension code, a constant non-zero vector, as illustrated in FIG1 (b).

To intuitively understand this, imagine that you stand in the middle of two objects A and B, and cannot see both of them.

To see both of them without turning the head or the eyes, you would have to walk several steps away so that the angle between A and B is less than some degree, as illustrated in Figure 3 (c).

DISPLAYFORM1 , not containing O, the sum of α and β is less than 90• ; (c) The angle ∠AOB = 180• .

If we shift A and B one unit into the new dimension, we have A ′ (0, 1, 1) and DISPLAYFORM2 , inclusion relations among them are preserved.

Following BID28 and BID27 , we structure the central vector of an N -ball by concatenating three vectors: (1) the pre-trained word-embedding, (2) the PLC (if the code is shorter than the max length, we append 0s till it reaches the fixed length), (3) the spatial extension code.

Our first criterion is to precisely encode subordinate relations among categories into inclusion relations among N -balls.

This is a considerable challenge, as the widely adopted back-propagation training process BID20 quickly reaches non-zero local minima and terminates 2 .

The problem is that when the location or the size of an N -ball is updated to improve its relation with a second ball, its relation to a third ball will deteriorate very easily.

We propose the classic depth-first recursion process, listed in Algorithm 1, to traverse the category tree and update sizes and locations of N -balls using three geometric transformations as follows.

Homothetic transformation (H-tran) which keeps the direction of the central vector and enlarges lengths of ⃗ A and r with the same rate k. DISPLAYFORM0 Shift transformation (S-tran) which keeps the length of the radius r and adds a new vector ⃗ s to ⃗ A. DISPLAYFORM1 Rotation transformation (R-tran) which keeps the length of the radius r and rotates angle α of ⃗ A inside the plane spanned by the i-th and the j-th dimensions of DISPLAYFORM2 To satisfy our second criterion, we do not choose rotation dimensions among the dimensions of pre-trained word embeddings.

Rather, to prevent the deterioration of already improved relations, we use the principle of family action: if a transformation is applied for one ball, the same transformation will be applied to all its child balls.

Among the three transformations, only H-tran preserves inclusion and disconnectedness relations among the family of N -balls as illustrated in Figure 3(d) , therefore H-tran has the priority to be used.

In the process of adjusting sibling N -balls to be disconnected from each other, we apply H-tran obeying the principle of family action.

When an N -ball is too close to the origin of the space, a S-tran plus R-tran will be applied which may change the pre-trained word embeddings.

Following the depth-first procedure, a parent ball is constructed after all its child balls.

Given a child N -ball B( ⃗ B, r 2 ), a candidate parent ball B( ⃗ A, r 1 ) is constructed as the minimal cover of B( ⃗ B, r 2 ), illustrated in Figure 3(b) .

The final parent ball B( ⃗ P , r p ) is the minimal ball which covers these already constructed candidate parent balls B( ⃗ P i , r pi ).Algorithm 1: training one f amily(root): Depth first algorithm to construct N -balls of all nodes of a tree input : a tree pointed by root; each node stores a word sense and its word embedding output: the N -ball embedding of each node children ←− get all children of (root) if number of (children) > 0 then foreach child ∈ children do // depth first training one f amily(child) end if number of (children) > 1 then // adjusting siblings to be disconnected from each other adjust to be disconnected(children) end // create parent ball for all children root = create parent ball of (children) else initialize ball(root) end

We use the GloVe word embeddings of BID19 as the pre-trained vector embedding of words and extract trees of hyponym relations among word senses from Word-Net 3.0 of BID17 .

The data set has 54, 310 word senses and 291 trees, among which root entity.n.01 is the largest tree with 43, 669 word senses.

Source code and input data sets are publically available at https://github.com/GnodIsNait/nball4tree.git.

We proved that all subordinate relations in the category tree are preserved in N -ball embeddings.

That is, zero energy cost is achieved by utilizing the proposed geometric approach.

Therefore, the first criterion is satisfied.

We apply homothetic, shifting, and rotating transformations in the construction/adjusting process of N -ball embeddings.

Shifting and rotating transformations may change pre-trained word embeddings.

The aim of this experiment is to examine the effect of the geometric process on pre-trained word embeddings, and check whether the second criterion is satisfied.

Method 1 We examine the standard deviation (std) of the pre-trained word embedding in N -ball embeddings of its word senses.

The less their std, the better they are preserved.

The N -ball embeddings have 32,503 word stems.

For each word stem, we extract the word embedding parts from N -ball embeddings, normalize them, minus pre-train word embeddings, and compute standard deviation.

The maximum std is 0.7666.

There are 417 stds greater than 0.2, 6 stds in the range of (0.1, 0.2], 9 stds in the range of (10 −12 , 0.1], 9699 stds in the range of (0, 10 −12 ], 22,372 stds equals 0.

With this statistics we conclude that only a tiny fraction (1.3%) of pre-trained word embeddings have a small change (std ∈ (0.1, 0.7666]).Method 2 The quality of word embeddings can be evaluated by computing the consistency (Spearman's co-relation) of similarities between human-judged word relations and vector-based word similarity relations.

The standard data sets in the literature are the WordSim353 data set BID6 ) which consists of 353 pairs of words, each pair associated with a human-judged value about the co-relation between two words, and Stanford's Contextual Word Similarities (SCWS) data set BID10 which contains 2003 word pairs, each with 10 human judgments on the similarity.

Given a word w, we extract the word embedding part from its word senses' N -ball embeddings and use the average value as the word-embedding of w in the experiment.

Unfortunately, both data sets cannot be used directly within our experimental setting as some words do not appear in the ball-embedding due to (1) words whose word senses have neither hypernym, nor hyponym in Word-Net 3.0, e.g. holy; (2) words whose word senses have different word stems, e.g. laboratory, midday, graveyard, percent, zoo, FBI, . . . ; (3) words which have no word senses, e.g. Maradona; (4) words whose word senses use their basic form as word stems, e.g. clothes, troops, earning, fighting, children.

After removing all the missing words, we have 318 paired words from WordSim353 and 1719 pairs from SCWS dataset for the evaluation.

We get exactly the same Spearman's co-relation values in all 11 testing cases: the Spearman's corelation on WordSim318 is 76.08%; each test on Spearman's co-relations using SCWS1719 is also the same.

We conclude that N -ball embeddings are a "loyal" extension to word embeddings, therefore, the second criterion is satisfied.

Following BID13 , we do qualitative evaluations.

We manually inspect nearest neighbors and compare results with pre-trained GloVe embeddings.

A sample is listed in Table 1 -2 with interesting observations as follows.

Precise neighborhoods N -ball embeddings precisely separate word senses of a polysemy.

For example, the nearest neighbors of berlin.n.01 are all cities, the nearest neighbors of berlin.n.02 are all names as listed in Table 1 .Typed cosine similarity function better than the normal cosine function Sim 0 enriched by category information produces much better neighborhood word senses than the normal cosine measurement.

For example, the top-5 nearest neighbors of beijing in GloVe using normal cosine measurement are: china, taiwan, seoul, taipei, chinese, among which only seoul and taipei are cities.

The top-10 nearest neighbors of berlin in GloVe using normal cosine measurement are : vienna, warsaw, munich, prague, germany, moscow, hamburg, bonn, copenhagen, cologne , among which germany is a country.

A worse problem is that neighbors of the word sense berlin.n.02 as the family name do not appear.

Without structural constraints, word embeddings are severely biased by a training corpus.

Category information contributes to the sparse data problem Due to sparse data, some words with similar meanings have negative cosine similarity value.

For example, tiger as a fierce or audacious person (tiger.n.01) and linguist as a specialist in linguistics (linguist.n.02) seldom appear in the same context, leading to −0.1 cosine similarity value using GloVe word embeddings.

However, they are hyponyms of person.n.01, using this constraint our geometrical process transforms the Nballs of tiger.n.01 and linguist.n.02 inside the N -ball of person.n.01, leading to high similarity value measured by the typed cosine function.

Upper category identification Using N -ball embeddings, we can find upper-categories of a word sense.

Given word sense ws, we collect all those cats satisfying D inside (ws, cat) > 0.

These cats shall be upper-categories of ws.

If we sort them in increasing order and mark the first cat with + 1 , the second with + 2 . . .

, the cat marked with + 1 is the direct upper-category of ws, the cat marked with + 2 is the direct upper-category of the cat with + 1 . . .

, as listed in Table 1 : Top-6 nearest neighbors based on Sim 0 , the cos value of word stems are listed, e.g. cos(beijing, london)= 0.47, tiger.n.01 refers to a fierce or audacious person.word sense 1 word sense 2 beijing.n.01 city.n.01+ 1 , municipality.n.01+ 2 , region.n.03+ 3 , location.n.01+ 4 , object.n.01+ 5 , entity.n.01+ 6 berlin.n.02 songwriter.n.01+ 1 , composer.n.01+ 2 , musician.n.02+ 3 , artist.n.01+ 4 , creator.n.02+ 5 tiger.n.01person.n.01+ 1 , organism.n.01+ 2 , whole.n.02+ 3 , object.n.01+ 4 , entity.n.01+ 5 france.n.02writer.n.01+ 1 , communicator.n.01+ 2 , person.n.01+ 3 , organism.n.01+ 4 cat.n.01 wildcat.n.03+ 1 , lynx.n.02+ 2 , cougar.n.01+ 3 , bobcat.n.01+ 4 , caracal.n.01+ 5 , ocelot.n.01+ 6 , feline.n.01+ 7 ,jaguarundi.n.01+ 8 y.n.02 letter.n.02+ 1 character.n.08+ 2 symbol.n.01+ 3 , signal.n.01+ 4 communication.n.02+ 5

The fourth experiment is to validate the category of an unknown word, with the aim to demonstrate the predictive power of the embedding approach BID1 .

We describe the task as follows:

Given pre-trained word embeddings E N with vocabulary size N , and a tree structure of hyponym relations T K on vocabulary W K , (K < N ).

Given w x / ∈ W K and c ∈ W K , we need to decide whether w x ∈ c. For example, when we read mwanza is also experiencing, we may guess mwanza a person, if we continue to read mwanza is also experiencing major infrastructural development, we would say mwanza is a city.

If mwanza is not in the current taxonomy of hypernym structure, we only have its word embedding from the text.

Should mwanza be a city, or a person?Dataset From 54,310 word senses, we randomly selected 1,000 word senses of nouns and verbs as target categories, with the condition that each of them has at least 10 direct hyponyms; For each target category, we randomly select p% (p ∈ [5, 10, 20, 30, 40, 50, 60, 70, 80, 90] ) from its direct hyponyms as training data.

The test data sets are generated from three sources: (1) the rest 1 − p% from the direct hyponyms as true values, (2) randomly choose 1,000 false values from W K ; (3) 1,000 words from W N which do not exist in W K .

In total, we created 118,938 hyponymy relations in the training set, and 17,975,042 hyponymy relations in the testing set.

Method We develop an N -ball solution to solve the membership validation task as follows: Suppose c has a hypernym path [c, h 1 , h 2 , . . .

, h m ] and has several known members (direct hyponyms) t 1 , . . . , t s .

For example, city.n.01 has a hypernym path [city.n.01, municipality.n.01, urban area.n.01, . . .

, entity.n.01 ] and a number of known members oxford.n.

01, banff.n.01, chicago.n.01 .

We construct N -ball embeddings for this small tree with the stem [c, h 1 , h 2 , . . .

, h m ] and leaves t 1 , . . . , t s , and record the history of the geometric transformations of each ball.

Suppose that w x be a member of c, we initialize the N -ball of w x using the same parameter as the N -ball of t 1 , and apply the recorded history of the geometric transformations of t 1 's N -ball for w x 's N -ball.

If the final N -ball of w x is located inside the N -ball of c, we will decide that w x is the member of c, otherwise not.

This method can be explained in terms of Support Vector Machines (Shawe-

In the experiment, results show that the N -ball method is very precise and robust as shown in FIG2 (a): the precision is always 100%, even only 5% from all members is selected as training set.

The method is quite robust: If we select 5% as training set, the recall reaches 76.8%; if we select 50% as training set, the recall reaches 96.7%.

Theoretically, the Nball method can not guarantee 100% recall as shown in FIG2 (b).

If p < 70%, the population standard deviation (pstdev) decreases with the increasing percentage of selected training data.

When p > 70%, there is a slight increase of pstdev.

The reason is that in the experiment setting, if more than 80% of the children are selected, it can happen that only one unknown member is left for validating.

If this single member is excluded outside of the category's N -ball, the recall drops to 0, which increases pstdev.

The experiment result can be downloaded at https://figshare.

com/articles/membership_validation_results/7571297.In the literature of representational learning, margin-based score functions are the state-of-the-art approach BID7 : The score of a positive sample shall be larger than the score of a negative sample plus a margin.

This can be understood as a simple use of categorizationno chained subordinate relations, no clear membership relations of negative samples, no requirement on zero energy loss.

However, when category information is fully and strictly used, the precision will increase significantly, and surprisingly reach 100% in this experiment.5 RELATED WORK BID12 explored the possibility of identifying hypernyms in distributional semantic model; BID21 presented an entropy-based model to identify hypernyms in an unsupervised manner; BID11 induced mappings from words/sentences embeddings into Boolean structure, with the aim to narrow the gap between co-occurrence based embeddings and logic-based structures.

There are some works on word embedding and knowledge graph embedding using regions to represent words or entities.

BID0 used multi-modal Gaussian distribution to represent words; BID9 embedded entities using Gaussian distributions; BID27 used manifolds to represent entities and relations; BID18 used Poincaré balls to embed tree structures.

Mirzazadeh et al. (2015) share certain common interest with the presented work in embedding constraints.

However, in none of these works, structural imposition at zero-energy cost is targeted.

We proposed a novel geometric method to precisely impose external tree-structured category information onto word embeddings, resulting in region-based (N -ball embeddings) word sense embeddings.

They can be viewed as Venn diagrams (Venn, 1880) of the tree structure, if zero energy cost is achieved.

Our N -ball method has demonstrated great performance in validating the category of unknown words, the reason for this being under further investigation.

Our on-going work also includes multi-lingual region-based knowledge graph embedding where multiple relations on directed acyclic graphs need to be considered.

N -balls carry both vector information from deep learning and region information from symbolic structures.

Therefore, N -balls establish a harmony between Connectionism and Symbolicism, as discussed by BID15 , and thus may serve as a novel building block for the commonsense representation and reasoning in Artificial Intelligence.

N -balls, in particular, contribute a new topic to Qualitative Spatial Reasoning (QSR) that dates back to BID26 .

<|TLDR|>

@highlight

we show a geometric method to perfectly encode categroy tree information into pre-trained word-embeddings.

@highlight

The paper proposes N-ball embedding for taxonomic data where an N-ball is a pair of a centroid vector and the radius from the center.

@highlight

The paper presents a method for tweaking existing vector embeddings of categorical objects (such as words), to convert them to ball embeddings that follow hierarchies.

@highlight

Focuses on adjusting the pretrained word embeddings so that they respect the hypernymy/hyponymy relationship by appropriate n-ball encapsulation.