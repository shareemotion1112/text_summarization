Estimating the frequencies of elements in a data stream is a fundamental task in data analysis and machine learning.

The problem is typically addressed using streaming algorithms which can process very large data using limited storage.

Today's streaming algorithms, however, cannot exploit patterns in their input to improve performance.

We propose a new class of algorithms that automatically learn relevant patterns in the input data and use them to improve its frequency estimates.

The proposed algorithms combine the benefits of machine learning with the formal guarantees available through algorithm theory.

We prove that our learning-based algorithms have lower estimation errors than their non-learning counterparts.

We also evaluate our algorithms on two real-world datasets and demonstrate empirically their performance gains.

Classical algorithms provide formal guarantees over their performance, but often fail to leverage useful patterns in their input data to improve their output.

On the other hand, deep learning models are highly successful at capturing and utilizing complex data patterns, but often lack formal error bounds.

The last few years have witnessed a growing effort to bridge this gap and introduce algorithms that can adapt to data properties while delivering worst case guarantees.

Deep learning modules have been integrated into the design of Bloom filters (Kraska et al., 2018; BID18 , caching algorithms (Lykouris & Vassilvitskii, 2018) , graph optimization BID12 , similarity search BID22 BID29 ) and compressive sensing BID3 .

This paper makes a significant step toward this vision by introducing frequency estimation streaming algorithms that automatically learn to leverage the properties of the input data.

Estimating the frequencies of elements in a data stream is one of the most fundamental subroutines in data analysis.

It has applications in many areas of machine learning, including feature selection BID0 , ranking (Dzogang et al., 2015) , semi-supervised learning BID27 and natural language processing (Goyal et al., 2012) .

It has been also used for network measurements (Estan & Varghese, 2003; BID30 BID28 and security BID23 .

Frequency estimation algorithms have been implemented in popular data processing libraries, such as Algebird at Twitter BID4 .

They can answer practical questions like: what are the most searched words on the Internet?

or how much traffic is sent between any two machines in a network?The frequency estimation problem is formalized as follows: given a sequence S of elements from some universe U , for any element i ∈ U , estimate f i , the number of times i occurs in S. If one could store all arrivals from the stream S, one could sort the elements and compute their frequencies.

However, in big data applications, the stream is too large (and may be infinite) and cannot be stored.

This challenge has motivated the development of streaming algorithms, which read the elements of S in a single pass and compute a good estimate of the frequencies using a limited amount of space.1 Over the last two decades, many such streaming algorithms have been developed, including Count-Sketch BID7 , Count-Min BID11 ) and multistage filters (Estan & Varghese, 2003) .

The performance guarantees of these algorithms are wellunderstood, with upper and lower bounds matching up to O(·) factors (Jowhari et al., 2011) .However, such streaming algorithms typically assume generic data and do not leverage useful patterns or properties of their input.

For example, in text data, the word frequency is known to be inversely correlated with the length of the word.

Analogously, in network data, certain applications tend to generate more traffic than others.

If such properties can be harnessed, one could design frequency estimation algorithms that are much more efficient than the existing ones.

Yet, it is important to do so in a general framework that can harness various useful properties, instead of using handcrafted methods specific to a particular pattern or structure (e.g., word length, application type).In this paper, we introduce learning-based frequency estimation streaming algorithms.

Our algorithms are equipped with a learning model that enables them to exploit data properties without being specific to a particular pattern or knowing the useful property a priori.

We further provide theoretical analysis of the guarantees associated with such learning-based algorithms.

We focus on the important class of "hashing-based" algorithms, which includes some of the most used algorithms such as Count-Min, Count-Median and Count-Sketch.

Informally, these algorithms hash data items into B buckets, count the number of items hashed into each bucket, and use the bucket value as an estimate of item frequency.

The process can be repeated using multiple hash functions to improve accuracy.

Hashing-based algorithms have several useful properties.

In particular, they can handle item deletions, which are implemented by decrementing the respective counters.

Furthermore, some of them (notably Count-Min) never underestimate the true frequencies, i.e., f i ≥ f i holds always.

However, hashing algorithms lead to estimation errors due to collisions: when two elements are mapped to the same bucket, they affect each others' estimates.

Although collisions are unavoidable given the space constraints, the overall error significantly depends on the pattern of collisions.

For example, collisions between high-frequency elements ("heavy hitters") result in a large estimation error, and ideally should be minimized.

The existing algorithms, however, use random hash functions, which means that collisions are controlled only probabilistically.

Our idea is to use a small subset of S, call it S , to learn the heavy hitters.

We can then assign heavy hitters their own buckets to avoid the more costly collisions.

It is important to emphasize that we are learning the properties that identify heavy hitters as opposed to the identities of the heavy hitters themselves.

For example, in the word frequency case, shorter words tend to be more popular.

The subset S itself may miss many of the popular words, but whichever words popular in S are likely to be short.

Our objective is not to learn the identity of high frequency words using S .

Rather, we hope that a learning model trained on S learns that short words are more frequent, so that it can identify popular words even if they did not appear in S .Our main contributions are as follows:• We introduce learning-based frequency estimation streaming algorithms, which learn the properties of heavy hitters in their input and exploit this information to reduce errors• We provide performance guarantees showing that our algorithms can deliver a logarithmic factor improvement in the error bound over their non-learning counterparts.

Furthermore, we show that our learning-based instantiation of Count-Min, a widely used algorithm, is asymptotically optimal among all instantiations of that algorithm.

See Table 4 .1 in section 4.1 for the details.• We evaluate our learning-based algorithms using two real-world datasets: traffic load on an Internet backbone link and search query popularity.

In comparison to their non-learning counterparts, our algorithms yield performance gains that range from 18% to 71%.

Frequency estimation in data streams.

Frequency estimation, and the closely related problem of finding frequent elements in a data stream, are some of the most fundamental and well-studied problems in streaming algorithms, see BID9 for an overview.

Hashingbased algorithms such as Count-Sketch BID7 , Count-Min BID11 ) and multi-stage filters (Estan & Varghese, 2003) are widely used solutions for these problems.

These algorithms also have close connections to sparse recovery and compressed sens-ing BID6 Donoho, 2006) , where the hashing output can be considered as a compressed representation of the input data (Gilbert & Indyk, 2010) .Several "non-hashing" algorithms for frequency estimation have been also proposed BID17 BID13 Karp et al., 2003; BID15 .

These algorithms do not possess many of the properties of hashing-based methods listed in the introduction (such as the ability to handle deletions), but they often have better accuracy/space tradeoffs.

For a fair comparison, our evaluation focuses only on hashing algorithms.

However, our approach for learning heavy hitters should be useful for non-hashing algorithms as well.

Some papers have proposed or analyzed frequency estimation algorithms customized to data that follows Zipf Law BID7 BID10 BID15 BID16 BID21 ; the last algorithm is somewhat similar to the "lookup table" implementation of the heavy hitter oracle that we use as a baseline in our experiments.

Those algorithms need to know the data distribution a priori, and apply only to one distribution.

In contrast, our learning-based approach applies to any data property or distribution, and does not need to know that property or distribution a priori.

Learning-based algorithms.

Recently, researchers have begun exploring the idea of integrating machine learning models into algorithm design.

In particular, researchers have proposed improving compressed sensing algorithms, either by using neural networks to improve sparse recovery algorithms BID20 BID3 , or by designing linear measurements that are optimized for a particular class of vectors BID2 BID19 , or both.

The latter methods can be viewed as solving a problem similar to ours, as our goal is to design "measurements" of the frequency vector (f 1 , f 2 . . .

, f |U | ) tailored to a particular class of vectors.

However, the aforementioned methods need to explicitly represent a matrix of size B × |U |, where B is the number of buckets.

Hence, they are unsuitable for streaming algorithms which, by definition, have space limitations much smaller than the input size.

Another class of problems that benefited from machine learning is distance estimation, i.e., compression of high-dimensional vectors into compact representations from which one can estimate distances between the original vectors.

Early solutions to this problem, such as Locality-Sensitive Hashing, have been designed for worst case vectors.

Over the last decade, numerous methods for learning such representations have been developed BID22 BID29 Jegou et al., 2011; BID28 .

Although the objective of those papers is similar to ours, their techniques are not usable in our applications, as they involve a different set of tools and solve different problems.

More broadly, there have been several recent papers that leverage machine learning to design more efficient algorithms.

The authors of BID12 show how to use reinforcement learning and graph embedding to design algorithms for graph optimization (e.g., TSP).

Other learning-augmented combinatorial optimization problems are studied in (He et al., 2014; BID1 Lykouris & Vassilvitskii, 2018) .

More recently, (Kraska et al., 2018; BID18 have used machine learning to improve indexing data structures, including Bloom filters that (probabilistically) answer queries of the form "is a given element in the data set?"

As in those papers, our algorithms use neural networks to learn certain properties of the input.

However, we differ from those papers both in our design and theoretical analysis.

Our algorithms are designed to reduce collisions between heavy items, as such collisions greatly increase errors.

In contrast, in existence indices, all collisions count equally.

This also leads to our theoretical analysis being very different from that in BID18 3.

PRELIMINARIES

We will use e i := |f i − f i | to denote the estimation error for f i .

To measure the overall estimation error between the frequencies F = {f 1 , f 2 , · · · , f |U | } and their estimatesF = {f 1 ,f 2 , · · · ,f |U | }, we will use the expected error E i∼D [e i ], where D models the distribution over the queries to the data structure.

Similar to past work BID21 , we assume the query distribution D is the same as the distribution of the input stream, i.e., for any j we have where N is the sum of all frequencies.

This leads to the estimation error ofF with respect to F: DISPLAYFORM0 DISPLAYFORM1 We note that the theoretical guarantees of frequency estimation algorithms are typically phrased in the "( , δ)-form", e.g., Pr[|f i − f i | > N ] < δ for every i (see e.g., BID11 ).

However, this formulation involves two objectives ( and δ).

We believe that the (single objective) expected error in Equation 3.1 is more natural from the machine learning perspective.

In this section, we recap three variants of hashing-based algorithms for frequency estimation.

Single Hash Function.

DISPLAYFORM0 Note that it is always the case DISPLAYFORM1 and an array C of size k × B. The algorithm maintains C, such that at the end of the stream we have C[ , b] = j:h (j)=b f j .

For each i ∈ U , the frequency estimatef i is equal to min ≤k C[ , h (i)], and always satisfiesf i ≥ f i .Count-Sketch.

Similarly to Count-Min, we have k distinct hash functions DISPLAYFORM2 and an array C of size k × B. Additionally, in Count-Sketch, we have k sign functions g i : U → {−1, 1}, and the algorithm maintains C such that DISPLAYFORM3 .

For each i ∈ U , the frequency estimatef i is equal to the median of {g (i)·C[ , h (i)]} ≤k .

Note that unlike the previous two methods, here we may havef i < f i .

In our theoretical analysis we assume that the item frequencies follow the Zipf Law.

That is, if we re-order the items so that their frequencies appear in a sorted order f i1 ≥ f i2 ≥ . . .

≥ f in , then f ij ∝ 1/j.

To simplify the notation we assume that f i = 1/i.

We aim to develop frequency estimation algorithms that exploit data properties for better performance.

To do so, we learn an oracle that identifies heavy hitters, and use the oracle to assign each heavy hitter its unique bucket to avoid collisions.

Other items are simply hashed using any classic frequency estimation algorithm (e.g., Count-Min, or Count-Sketch), as shown in the block-diagram in Figure 4 .1.

This design has two useful properties: First, it allows us to augment a classic frequency estimation algorithm with learning capabilities, producing a learning-based counterpart that inherits the original guarantees of the classic algorithm.

For example, if the classic algorithm is Count-Min, the resulting learning-based algorithm never underestimates the frequencies.

Second, it provably reduces the estimation errors, and for the case of Count-Min it is (asymptotically) optimal.

Algorithm 1 provides pseudo code for our design.

The design assumes an oracle HH(i) that attempts to determine whether an item i is a "heavy hitter" or not.

All items classified as heavy hitters are assigned to one of the B r unique buckets reserved for heavy items.

All other items are fed to the remaining B − B r buckets using a conventional frequency estimation algorithm SketchAlg (e.g., Count-Min or Count-Sketch).The estimation procedure is analogous.

To computef i , the algorithm first checks whether i is stored in a unique bucket, and if so, reports its count.

Otherwise, it queries the SketchAlg procedure.

Note that if the element is stored in a unique bucket, its reported count is exact, i.e.,f i = f i .The oracle is constructed using machine learning and trained with a small subset of S, call it S .

Note that the oracle learns the properties that identify heavy hitters as opposed to the identities of the heavy hitters themselves.

For example, in the case of word frequency, the oracle would learn that shorter words are more frequent, so that it can identify popular words even if they did not appear in the training set S .

Our algorithms combine simplicity with strong error bounds.

Below, we summarize our theoretical results, and leave all theorems, lemmas, and proofs to the appendix.

In particular, Table 4 .1 lists the results proven in this paper, where each row refers to a specific streaming algorithm, its corresponding error bound, and the theorem/lemma that proves the bound.

First, we show (Theorem 9.11 and Theorem 9.14) that if the heavy hitter oracle is accurate, then the error of the learned variant of Count-Min is up to a logarithmic factor smaller than that of its nonlearning counterpart.

The improvement is maximized when B is of the same order as n (a common scenario 2 ).

Furthermore, we prove that this result continues to hold even if the learned oracle makes prediction errors with probability δ, as long as δ = O(1/ ln n) (Lemma 9.15).Second, we show that, asymptotically, our learned Count-Min algorithm cannot be improved any further by designing a better hashing scheme.

Specifically, for the case of Learned Count-Min with a perfect oracle, our design achieves the same asymptotic error as the "Ideal Count-Min", which optimizes its hash function for the given input (Theorem 10.4).Finally, we note that the learning-augmented algorithm inherits any ( , δ)-guarantees of the original version.

Specifically, its error is not larger than that of SketchAlg with space B − B r , for any input.

Expected Error Analysis DISPLAYFORM0 Theorem 10.4 Table 4 .1: Our performance bounds for different algorithms on streams with frequencies obeying Zipf Law.

k is a constant (≥ 2) that refers to the number of hash functions, B is the number of buckets, and n is the number of distinct elements.

The space complexity of all algorithms is the same, Θ(B).

See section 9.4 for non-asymptotic versions of the some of the above bounds

Baselines.

We compare our learning-based algorithms with their non-learning counterparts.

Specifically, we augment Count-Min with a learned oracle using Algorithm 1, and call the learningaugmented algorithm "Learned Count-Min".

We then compare Learned Count-Min with traditional Count-Min.

We also compare it with "Learned Count-Min with Ideal Oracle" where the neuralnetwork oracle is replaced with an ideal oracle that knows the identities of the heavy hitters in the test data, and " Table Lookup for each stream element i do 3:if DISPLAYFORM0 if i is already stored in a unique bucket then 5: increment the count of i 6:else create a new unique bucket for i and 7:initialize the count to 1 8: end if 9: else 10:feed i to SketchAlg with B − Br buckets 11:end if 12: end for 13: end procedure us to show the ability of Learned Count-Min to generalize and detect heavy items unseen in the training set.

We repeat the evaluation where we replace Count-Min (CM) with Count-Sketch (CS) and the corresponding variants.

We use validation data to select the best k for all algorithms.

Training a Heavy Hitter Oracle.

We construct the heavy hitter oracle by training a neural network to predict the heaviness of an item.

Note that the prediction of the network is not the final estimation.

It is used in Algorithm 1 to decide whether to assign an item to a unique bucket.

We train the network to predict the item counts (or the log of the counts) and minimize the squared loss of the prediction.

Empirically, we found that when the counts of heavy items are few orders of magnitude larger than the average counts (as is the case for the Internet traffic data set), predicting the log of the counts leads to more stable training and better results.

Once the model is trained, we select the optimal cutoff threshold using validation data, and use the model as the oracle described in Algorithm 1.

For our first experiment, the goal is to estimate the number of packets for each network flow.

A flow is a sequence of packets between two machines on the Internet.

It is identified by the IP addresses of its source and destination and the application ports.

Estimating the size of each flow i -i.e., the number of its packets f i -is a basic task in network management BID24 .

Model: The patterns of the Internet traffic are very dynamic, i.e., the flows with heavy traffic change frequently from one minute to the next.

However, we hypothesize that the space of IP addresses should be smooth in terms of traffic load.

For example, data centers at large companies and university campuses with many students tend to generate heavy traffic.

Thus, though the individual flows from these sites change frequently, we could still discover regions of IP addresses with heavy traffic through a learning approach.

We trained a neural network to predict the log of the packet counts for each flow.

The model takes as input the IP addresses and ports in each packet.

We use two RNNs to encode the source and destination IP addresses separately.

The RNN takes one bit of the IP address at each step, starting from the most significant bit.

We use the final states of the RNN as the feature vector for an IP address.

The reason to use RNN is that the patterns in the bits are hierarchical, i.e., the more significant bits govern larger regions in the IP space.

Additionally, we use two-layer fully-connected networks to encode the source and destination ports.

We then concatenate the encoded IP vectors, encoded port vectors, and the protocol type as the final features to predict the packet counts 3 .

The inference time takes 2.8 microseconds per item on a single GPU without any optimizations 4 .Results: We plot the results of two representative test minutes (the 20th and 50th) in FIG3 .2.

All plots in the figure refer to the estimation error (Equation 3.1) as a function of the used space.

The space includes space for storing the buckets and the model.

Since we use the same model for all test minutes, the model space is amortized over the 50-minute testing period.

Second, the figure also shows that our neural-network oracle performs better than memorizing the heavy hitters in a lookup table.

This is likely due to the dynamic nature of Internet traffic -i.e., the heavy flows in the training set are significantly different from those in the test data.

Hence, memorization does not work well.

On the other hand, our model is able to extract structures in the input that generalize to unseen test data.

Third, the figure shows that our model's performance stays roughly the same from the 20th to the 50th minute ( FIG3 .2b and FIG3 .2d), showing that it learns properties of the heavy items that generalize over time.

Lastly, although we achieve significant improvement over Count-Min and Count-Sketch, our scheme can potentially achieve even better results with an ideal oracle, as shown by the dashed green line in FIG3 .2.

This indicates potential gains from further optimizing the neural network model.

For our second experiment, the goal is to estimate the number of times a search query appears.

We use the AOL query log dataset, which consists of 21 million search queries collected from 650 thousand users over 90 days.

The users are anonymized in the dataset.

There are 3.8 million unique queries.

Each query is a search phrase with multiple words (e.g., "periodic table element poster").

We use the first 5 days for training, the following day for validation, and estimate the number of times different search queries appear in subsequent days.

The distribution of search query frequency follows the Zipfian law, as shown in FIG3 Model: Unlike traffic data, popular search queries tend to appear more consistently across multiple days.

For example, "google" is the most popular search phrase in most of the days in the dataset.

Simply storing the most popular words can easily construct a reasonable heavy hitter predictor.

However, beyond remembering the popular words, other factors also contribute to the popularity of a search phrase that we can learn.

For example, popular search phrases appearing in slightly different forms may be related to similar topics.

Though not included in the AOL dataset, in general, metadata of a search query (e.g., the location of the search) can provide useful context of its popularity.

To construct the heavy hitter oracle, we trained a neural network to predict the number of times a search phrase appears.

To process the search phrase, we train an RNN with LSTM cells that takes characters of a search phrase as input.

The final states encoded by the RNN are fed to a fully-connected layer to predict the query frequency.

Our character vocabulary includes lower-case English alphabets, numbers, punctuation marks, and a token for unknown characters.

We map the character IDs to embedding vectors before feeding them to the RNN 6 .

We choose RNN due to its effectiveness in processing sequence data BID25 Graves, 2013; Kraska et al., 2018) .

We plot the estimation error vs. space for two representative test days (the 50th and 80th day) in FIG3 .4.

As before, the space includes both the bucket space and the space used by the model.

The model space is amortized over the test days since the same model is used for all days.

Similarly, our learned sketches outperforms their conventional counterparts.

For Learned CountMin, compared to Count-Min, it reduces the loss by 18% at 0.5 MB and 52% at 1.0 MB FIG3 .4a).

For Learned Count-Sketch, compared to Count-Sketch, it reduces the loss by 24% at 0.5 MB and 71% at 1.0 MB FIG3 .

Further, our algorithm performs similarly for the 50th and the 80th day ( FIG3 .4b and FIG3 .4d), showing that the properties it learns generalize over a long period.

The figures also show an interesting difference from the Internet traffic data: memorizing the heavy hitters in a lookup table is quite effective in the low space region.

This is likely because the search queries are less dynamic compared to Internet traffic (i.e., top queries in the training set are also popular on later days).

However, as the algorithm is allowed more space, memorization becomes ineffective.

We analyze the accuracy of the neural network heavy hitter models to better understand the results on the two datasets.

Specifically, we use the models to predict whether an item is a heavy hitter (top 1% in counts) or not, and plot the ROC curves in FIG3 .5.

The figures show that the model for the Internet traffic data has learned to predict heavy items more effectively, with an AUC score of 0.9.

As for the model for search query data, the AUC score is 0.8.

This also explains why we see larger improvements over non-learning algorithms in FIG3 .2.

In this section, we visualize the embedding spaces learned by our heavy hitter models to shed light on the properties or structures the models learned.

Specifically, we take the neural network activations before the final fully-connected layer, and visualize them in a 2-dimensional space using t-SNE BID14 .

To illustrate the differences between heavy hitters (top 1% in counts) and the rest ("light" items), we randomly sample an equal amount of examples from both classes.

We visualize the embedding space for both the Internet traffic and search query datasets.

We show the embedding space learned by the model on the Internet traffic data in Figure 6 .1.

Each point in the scatter plot represents one Internet traffic flow.

By coloring each flow with its number of packets in Figure 6 .1a, we see that the model separate flows with more packets (green and yellow clusters) from flows with fewer packets (blue clusters).

To understand what structure the model learns to separate these flows, we color each flow with its destination IP address in Figure 6 .1b.

We found that clusters with more packets are often formed by flows sharing similar destination address prefixes.

Interestingly, the model learns to group flows with similar IP prefixes closer in the embedding space.

For example, the dark blue cluster at the upper left of Figure 6 .1b shares a destination IP address prefix "1.96.*.*".

Learning this "structure" from the Internet traffic data allows the model to generalize to packets unseen in the training set.

We show the embedding space learned by the model on the search query data in Figure 6 .2.

Each point in the scatter plot represents one search query.

Similarly, the model learns to separate frequent search queries from the rest in Figure 6 .2a.

By coloring the queries with the number of characters in Figure 6 .2b, we have multiple interesting findings.

First, queries with similar length are closer in the embedding space, and the y-axis forms the dimension representing query length.

Second, if we simply use the query length to predict heavy hitters, many light queries will be misclassified.

The model must have learned other structures to separate heavy hitters in Figure 6 .2a.

We have presented a new approach for designing frequency estimation streaming algorithms by augmenting them with a learning model that exploits data properties.

We have demonstrated the benefits of our design both analytically and empirically.

We envision that our work will motivate a deeper integration of learning in algorithm design, leading to more efficient algorithms.

In this section, we analyze the performance of three different approaches, single (uniformly random) hash function, Count-Min sketch, and Learned Count-Min sketch when the frequency of items is from Zipfian distribution.

For simplicity, we assume that the number of distinct elements n is equal to the size of the universe |U |, and f i = 1/i.

We use [n] to denote the set {1 . . .

n}.

We also drop the normalization factor 1/N in the definition of estimation error.

The following observation is useful throughout this section (in particular, in the section on nonasymptotic analysis).

Observation 9.1.

For sufficiently large values of n (i.e., n > 250), DISPLAYFORM0 DISPLAYFORM1 Moreover, since each bucket maintains the frequency of items that are mapped to it under h, the space complexity of this approach is proportional to the number of buckets which is Θ(B).

Here, we provide an upper bound and lower bound for the expected estimation error of CountMin sketch with k hash functions and B buckets per row.

In the rest of this section, for each j ∈ [n], ≤ k, we use e j, and e j respectively to denote the estimation error of f j by h and Count-Min sketch.

Recall that the expected error of Count-Min sketch is defined as follows: DISPLAYFORM0 Our high-level approach is to partition the interval [0, B ln n] into m + 1 smaller intervals by a sequence of thresholds Θ(ln 1+γ ( n B )) = r 0 ≤ · · · ≤ r m = B ln n where γ is a parameter to be determined later.

Formally, we define the sequence of r i s to satisfy the following property: DISPLAYFORM1 Proof: By (9.3) and assuming ln r i+1 ≥ ln( DISPLAYFORM2 1+γ > 2 ln r i for sufficiently large values of r i 7 assuming γ ≤ 3.Note that as long as ln r i+1 ≥ ln( Then, to compute (9.2), we rewrite E[e j ] using the thresholds r 0 , · · · , r m as follows: DISPLAYFORM3 .

Proof: First we prove the following useful observation.

DISPLAYFORM4 Thus, by Markov's inequality, for each item j and hash function h , DISPLAYFORM5 Now, for each h in Count-Min sketch, we bound the value of Pr(e j, ≥ t B ) where t ∈ [r i , r i+1 ): DISPLAYFORM6 ) by (9.5) and Corollary 9.4 (9.6) Hence, for k ≥ 2, DISPLAYFORM7 by (9.5) and (9.6) Next, for each item j, we bound the contribution of each interval ( DISPLAYFORM8 ).Proof: DISPLAYFORM9 Similarly, r i+1 B r i B m−1 q=i+1 B q dx is at most: DISPLAYFORM10 Now, we complete the error analysis of (9.4): DISPLAYFORM11 Note that (9.10) requires γ(k − 1) − 2 ≥ 1 which is satisfied by setting γ = 3/(k − 1) and k ≥ 2.

Thus, for each item j, DISPLAYFORM12 Lemma 9.8.

The expected error of Count-Min sketch of size k × B (with k ≥ 2) for estimating items whose frequency distribution is Zipfian is O( DISPLAYFORM13 Proof: By plugging in our upper bound on the estimation error of each item computed in (9.11) in the definition of expected estimation error of Count-Min (9.2), we have the following.

DISPLAYFORM14 Next, we show a lower bound on the expected error of Count-Min sketch with B buckets (more precisely, of size (k × B/k)) for estimating the frequency of items that follow Zipf Law.

Observation 9.9.

For each item j, Pr[e j ≥ 1/(2( DISPLAYFORM15 For each item j, the probability that none of the first 2( ).

DISPLAYFORM16 In particular, for the case B = Θ(n) and k = O(1), the expected error of Count-Min sketch is Θ( ln n B ).

Proof: The proof follows from Lemma 9.8 and 9.10.

We remark that the bound in Lemma 9.8 is for the expected estimation error of Count-Min sketch of size k × B. Hence, to get the bound on the expected error of Count-Min of size k × ( B k ), we must replace B with B/k.

Definition 9.12 (φ-HeavyHitter).

Given a set of items I = {i 1 , · · · , i n } with frequencies f = f 1 , · · · , f n , an item j is a φ-HeavyHitter of I if f j ≥ φ|| f || 1 .

Remark 9.13.

If the frequency distribution of items I is Zipfian, then the number of φ-HeavyHitters is at most 1/(φ ln n).

In other words, B r ≤ (φ ln n) −1 .To recall, in our Learned Count-Min sketch with parameters (B r , B), B r buckets are reserved for the frequent items returned by HH and the rest of items are fed to a Count-Min sketch of size k × (

) where k is a parameter to be determined.

We emphasize that the space complexity of Learned Count-Min sketch with parameter (B r , B) is B r + B = O(B).

Theorem 9.14.

The optimal expected error of Learned Count-Min sketches with parameters (B r , B) is at most DISPLAYFORM0 Proof:

Since, the count of top B r frequent items are stored in their own buckets, for each j ≤ B r , e j = 0.

Hence, DISPLAYFORM1 Note that the last inequality follows from the guarantee of single hash functions; in other words, setting k = 1 in the Count-Min sketch.

Unlike the previous part, here we assume that we are given a noisy HeavyHitters oracle HH δ such that for each item j, Pr HH δ (j, DISPLAYFORM0 Br ln n ) ≤ δ where HH 0 is an ideal HeavyHitter oracle that detects heavy items with no error.

Lemma 9.15.

In an optimal Learned Count-Min sketch with parameters (B r , B) and a noisy HeavyHitters oracle DISPLAYFORM1 Proof: The key observation is that each heavy item, any of B r most frequent items, may only misclassify with probability δ.

Hence, for each item j classified as "not heavy", (9.12) where the first term denotes the expected contribution of the misclassified heavy items and the second term denotes the expected contribution of non-heavy items.

DISPLAYFORM2 The rest of analysis is similar to the proof of Theorem 9.14.

DISPLAYFORM3 Corollary 9.16.

Assuming B r = Θ(B) = Θ(n) and DISPLAYFORM4 Space analysis.

Here, we compute the amount of space that is required by this approach. ) with cutoff (B r reserved buckets) for estimating the frequency of items whose distribution is Zipfian is O(B).Proof: The amount of space required to store the counters corresponding to functions DISPLAYFORM5 Here, we also need to keep a mapping from the heavy items (top B r frequent items according to HH δ ) to the reserved buckets B r which requires extra O(B r ) space; each reserved buckets stores both the hashmap of its corresponding item and its count.

In this section, we compare the non-asymptotic expected error of Count-Min sketch and out Learned Count-Min sketch with ideal HeavyHitters oracle.

Throughout this section, we assume that the amount of available space to the frequency estimation algorithms is (1+α)B words.

More precisely, we compare the expected error of Count-Min sketch with k hash functions and our Learned CountMin sketch with B r = αB reserved buckets.

Recall that we computed the following bounds on the expected error of these approaches (Lemma 9.10 and Theorem 9.14): DISPLAYFORM0 In the rest of this section, we assume that B ≥ γn and then compute the minimum value of γ that guarantees DISPLAYFORM1 In other words, we compute the minimum amount of space that is required so that our Learned Count-Min sketch performs better than Count-Min sketch by a factor of at least (1 + ε).

DISPLAYFORM2 Hence, we must have (0.58 DISPLAYFORM3 · ln n. By solving the corresponding quadratic equation, DISPLAYFORM4 This implies that ln γ = − ln α + 0.58 DISPLAYFORM5 Published as a conference paper at ICLR 2019 Next, we consider different values of k and show that in each case what is the minimum amount of space in which Learned CM outperforms CM by a factor of 1.06 (setting ε = 0.06).• k = 1.

In this case, we are basically comparing the expected error of a single hash function and Learned Count-Min.

In particular, in order to get a gap of at least (1 + ε), by a more careful analysis of Lemma 9.2, γ must satisfy the following condition: DISPLAYFORM6 To simplify it further, we require that (ln( 2 γ )+0.58) 2 ≤ 3.18·(ln 2 n−1.65) which implies that γ = Θ(1/ ln n).• k = 2.

In this case, γ ≤ • k ∈ {3, 4}: In this case, γ ≤ 2 e √ (ln n)/3.5for sufficiently large values of B. Hence, we require that the total amount of available space is at least DISPLAYFORM7 for sufficiently large values of B. Hence, we require that the total amount of available space is at least DISPLAYFORM8 We also note that settings where the number of buckets is close to n are quite common in practice.

Recall that the estimation error of a hash function h is defined as Err(F(I),F h (I)) := i∈I f i · (f (h(i)) − i).

Note that we can rewrite Err(F(I),F h (I)) as DISPLAYFORM9 Note that in (10.1) the second term is independent of h and is a constant.

Hence, an optimal hash function minimizes the first term, b∈B I f (b) 2 .Suppose that an item i * with frequency at least DISPLAYFORM10 collides with a (non-empty) set of items I * ⊆

I \ {i * } under an optimal hash function h * .

Since the total frequency of the items mapped to the bucket b * containing i * is greater than .

Next, we define a new hash function h with smaller estimation error compared to h * which contradicts the optimality of h * : DISPLAYFORM11 Formally, Err(F(I),F h * (I)) − Err(F(I),F h (I)) = f h * (b DISPLAYFORM12 Next, we show that in any optimal hash function h * :[n] → [B] and assuming Zipfian input distribution, Θ(B) most frequent items do not collide with any other items under h * .Lemma 10.2.

Suppose that B = n/γ where γ ≥ e 4.2 is a constant and lets assume that items follow Zipfian distribution.

In any hash function h * :[n] → [B] with minimum estimation error, none of the B 2 ln γ most frequent items collide with any other items (i.e., they are mapped to a singleton bucket).

Proof: Let i j * be the most frequent item that is not mapped to a singleton bucket under h * .

If j * > B 2 ln γ then the statement holds.

Suppose it is not the case and j * ≤ B 2 ln γ .

Let I denote the set of items with frequency at most f j * = 1/j * (i.e., I = {i j | j ≥ j * }) and let B I denote the number of buckets that the items with index at least j * mapped to; B I = B − j * + 1.

Also note that by Observation 9.1, f (I) < ln( n j * ) + 1.

Next, by Claim 10.1, we show that h * does not hash the items {j * , · · · , n} to B I optimally.

In particular, we show that the frequency of item j * is more than DISPLAYFORM13 .

To prove this, first we observe that the function g(j) := j · (ln(n/j) + 1) is strictly increasing in Proof: By Lemma 10.2, in any hash function with minimum estimation error, the ( B 2 ln γ ) most frequent items do not collide with any other items (i.e., they are mapped into a singleton bucket) where γ = n/B > e 4.2 .Hence, the goal is to minimize (10.1) for the set of items I which consist of all items other than the ( B 2 ln γ ) most frequent items.

Since the sum of squares of m items that summed to S is at least S 2 /m, the multi-set loss of any optimal hash function is at least: ) as well.

DISPLAYFORM14

@highlight

Data stream algorithms can be improved using deep learning, while retaining performance guarantees.