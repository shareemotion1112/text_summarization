Knowledge bases, massive collections of facts (RDF triples) on diverse topics, support vital modern applications.

However, existing knowledge bases contain very little data compared to the wealth of information on the Web.

This is because the industry standard in knowledge base creation and augmentation suffers from a serious bottleneck: they rely on domain experts to identify appropriate web sources to extract data from.

Efforts to fully automate knowledge extraction have failed to improve this standard: these automated systems are able to retrieve much more data and from a broader range of sources, but they suffer from very low precision and recall.

As a result, these large-scale extractions remain unexploited.

In this paper, we present MIDAS, a system that harnesses the results of automated knowledge extraction pipelines to repair the bottleneck in industrial knowledge creation and augmentation processes.

MIDAS automates the suggestion of good-quality web sources and describes what to extract with respect to augmenting an existing knowledge base.

We make three major contributions.

First, we introduce a novel concept, web source slices, to describe the contents of a web source.

Second, we define a profit function to quantify the value of a web source slice with respect to augmenting an existing knowledge base.

Third, we develop effective and highly-scalable algorithms to derive high-profit web source slices.

We demonstrate that MIDAS produces high-profit results and outperforms the baselines significantly on both real-word and synthetic datasets.

Knowledge bases support a wide range of applications and enhance search results for multiple major search engines, such as Google and Bing BID1 .The coverage and correctness of knowledge bases are crucial for the applications that use them, and for the quality of the user experience.

However, there exists a gap between facts on the Web and in knowledge bases: compared to the wealth of information on the Web, most knowledge bases are largely incomplete, with many facts missing.

For example, one of the largest knowledge bases, Freebase BID7 BID0 , does not provide sufficient facts for different types of cocktails such as the ingredients of Margarita.

Yet, such information is explicitly profiled and described by many web sources, such as Wikipedia.

Figure 1 : Two knowledge extraction procedures and Midas.

The output of the automated process (b) is often discarded in industrial production due to low accuracy.

Midas uses the the automatically-extracted facts to identify the right web sources for the semi-automated process under the industry standard and therefore resolves a major bottleneck.

Industry standard.

Industry typically follows a semi-automated knowledge extraction process to create or augment a knowledge base with facts that are new to an existing knowledge base (or new facts) from the Web.

This process ( Figure 1a ) first relies on domain experts to select web sources; it then uses crowdsourcing to annotate a fraction of entities and facts and treats them as the training data; finally, it applies wrapper induction BID19 BID21 and learns Xpath patterns to extract facts from the selected web sources.

Since source selection and training data preparation are carefully curated, this process achieves high precision and recall with respect to each selected web source.

However, it can only produce a small volume of facts overall and cannot scale, as the source-selection step is a severe bottleneck, relying on manual curation by domain experts.

Automated process.

To conquer the scalability limitation in the industry standard, automated knowledge extraction BID13 BID30 attempts to extract facts with little or no human intervention.

Instead of manually selecting a small set of web sources, automated extraction (Figure 1b) often takes a wide variety of web sources, e.g., ClueWeb09 BID10 , as input and uses facts in an existing knowledge base, or a small portion of labeled input web sources, as training data.

This automated extraction process is able to produce a vast number of facts.

However, because of the limited training data (per source), especially for uncommon facts, e.g., the ingredients of Margarita, this process suffers from low accuracy.

The TAC-KBP competition showed that automated processes BID33 BID3 BID34 BID12 can hardly achieve above 0.3 recall, leaving a lot of the wealth of web information unexploited.

Due to this limitation, such automatically extracted facts are often abandoned for knowledge bases in industrial production.

In this paper, we propose Midas 1 , a system that harnesses the correct 2 extractions of the automated process to automatically identify suitable web sources and repair the bottleneck in the industry standard.

The core insight of Midas is that the automatically extracted facts, even though they may not be of high overall accuracy and coverage, give clues about which web sources contain a large amount of valuable information, allow for easy annotation, and are worthwhile for extraction.

We demonstrate this through an example.1.

Our system is named after King Midas, known in Greek mythology for his ability to turn what he touched into gold.

2.

We refer to correct facts as facts with confidence value ≥ 0.7 as true.

Example 1.

FIG1 shows a snapshot of high-confidence facts (subject, predicate, object) extracted from 5 web pages under web domain http://space.skyrocket.de.

Automated extraction systems may not be able to obtain high precision and recall in extracting facts from this website due to lack of effective training data.

However, the few correct extracted facts give important clues on what one could extract from this site.

For each fact, the subject indicates an entity; the predicate and object values further describe properties associated with the entity.

For example, fact t 1 specifies that the category property of the entity Project Mercury is space program.

Entities can form groups based on their common properties.

For example, entity "Project Mercury" and entity "Project Gemini" are both "space programs that are sponsored by NASA".The facts labeled "Y" in the "new?

" column are absent from Freebase.

All of these new facts are under the same sub-domain and are all "rocket families sponsored by the NASA.

"

This observation provides a critical insight: one can augment Freebase by extracting facts pertaining to "rocket families sponsored by NASA" from http://space.skyrocket.de/doc_lau_fam.Example 1 shows that one can abstract the contents of a web source through extracted facts: A web source often includes facts of multiple groups of homogeneous entities.

Each group of entities forms a particular subset of content in the web source, which we call a web source slice (or slice).

The common properties shared by the group of entities not only define, but also describe the slice of facts.

For example, it is easy to tell that a slice describes "rocket families sponsored by NASA" through its common properties, "category = rocket family" and "sponsor = NASA".

Moreover, entities in a single web source slice often belong to the same type, e.g., "rocket families sponsored by NASA", and thus share similar predicates.

The limited number of predicates in a web source slice simplifies annotation.

Our objective is to discover web source slices that (1) contain a sufficient number of facts that are absent from the knowledge base we wish to augment, and (2) their extraction effort does not outweigh the benefit.

However, evaluating and quantifying the suitability of a web source slice with respect to these two desired properties is not straightforward.

In addition, the number of slices in a single web source often grows exponentially with the number of facts, posing a significant scalability challenge.

This challenge is amplified by the massive number of sources on the Web, in various genres, languages, and domains.

Even a single web domain Midas derived slides using facts extracted from a real-world, large-scale, automated knowledge extraction pipeline (name hidden for anonymity) that operates on billions of web pages.

New facts refer to extracted facts that are absent from Freebase.may contain an extensive amount of knowledge.

For example, as of July 2018, there are more than 45 million entries in Wikipedia [3] .Midas addresses these challenges through (1) efficient and scalable algorithms for producing web source slices, and (2) an effective profit function for measuring the utility of slices.

In this paper, we first formalize the problem of identifying and describing "good" web sources as an optimization problem and then quantify the quality of web source slices through a profit function (Section 2).

We then propose an algorithm to generate the high-profit slices in a single web source and design a scalable framework to extend this algorithm for multiple web sources (Section 3).

Finally, we evaluate our proposed algorithm on both real-word and synthetic datasets and illustrate that our proposed system, Midas, is able to identify interesting web sources slices in an efficient and scalable manner (Section 4).

Example 2.

We applied Midas on AnonSys, a dataset extracted by a comprehensive knowledge extraction system, which includes 810M facts extracted from 218M web sources.

Midas is able to identify and customize "good" web sources for an existing knowledge base.

In FIG2 , we demonstrate the 5 highest-profit slices that Midas derived to augment Freebase.

The web source slices provide new and valuable information for augmenting the existing knowledge base; in addition, many of these web sources contain semi-structured data with respect to entities in the reported web source slice.

Therefore, they are easy for annotation.

In this section, we first define web source slices (Section 2.1); we then use these abstractions to formalize the problem of slice discovery for knowledge base augmentation (Section 2.2).

Web source.

URL hierarchies offer access to web sources at different granularities, such as a web domain (https://www.cdc.gov), a sub-domain (https://www.cdc.gov/niosh), or a web page (https://www.cdc.gov/niosh/ipcsneng/neng0363.html).

Web domains often use URL hierarchies to classify their contents.

For example, the web domain https: //www.golfadvisor.com classifies facts for "golf course in Jamaica" under the finer-grained URL https://www.golfadvisor.com/course-directory/8545-jamaica.

The URL hierarchies in these web domains divide their contents into smaller, coherent subsets, providing opportunities to reduce unnecessary extraction effort.

For example, the web domain https: //www.cdc.gov requires significant extraction effort as its contents are varied and spread across Contents of a web source.

Facts extracted from a web source typically correspond to many different entities.

However, they can share common properties: for example, the entities "Atlas" and "Castor-4" FIG1 ) have the common property of being rocket families sponsored by NASA.

We abstract and formalize the content represented by a group of entities as a web source slice and define it by the entities' common properties.

The abstraction of web source slices achieves two goals: (1) it offers a representation of the content of a web source that is easily understandable by humans, and (2) it allows for the efficient retrieval of all facts relevant to that content.

As described in Example 1, an extracted fact corresponds to an entity and describes properties of that entity.

Web source slices, in turn, are defined over a group of entities with common properties.

To facilitate this exposition, we organize facts of a web source W in a fact table F W (Figure 4) .

A row in the fact table contains facts that correspond to the same entity (denoted by the subject).

Definition 3 (Fact table) .

Let T W = {(s, p, o)} be a set of facts, in the form of (subject, predicate, object), extracted from a web source W , and n be the number of distinct predicates in T W (n = |{t.p | t ∈ T W }|).

We define the fact table F W (subject, pred 1 , . . . , pred n ), which has a primary key (subject) and one attribute for each of the n distinct predicates.

Each fact t ∈ T W maps to a single, non-empty cell in F W : DISPLAYFORM0 where Π and σ are the Projection and Selection operators in relational algebra.

Note that cells in F W may contain a set of values, corresponding to facts with the same subject and predicate.

For ease of exposition, we use single values in our examples.

We now define properties and web source slices over the fact table F W .

Definition 4 (Property).

A property c = (pred, v) is a pair derived from a fact table F W , such that pred is an attribute in F W and v ∈ Π pred (F W ).

We further denote with C W the set of all properties in a web source W : Figure 4 lists all the properties derived from the fact table of our running example.

Midas considers properties where the value is strictly derived from the domain of pred: v ∈ Π pred (F W ).

Our method can be easily extended to more general properties, e.g., "year > 2000"; however, we decided against this generalization, as it increases the complexity of the algorithms significantly, without observable improvement in the results.

In addition, Midas does not consider properties on the subject attribute since in most real-word datasets subjects are typically identification numbers.

Definition 5 (Web Source Slice).

Given a set of facts T W extracted from web source W , the corresponding fact table F W , and the collection of properties C W , a web source slice (or slice), denoted by S(W ) (or S for short), is a triplet S(W ) = (C, Π, Π * ), where, DISPLAYFORM1 DISPLAYFORM2 is a non-empty set of entities, each of which includes all of the properties in C; DISPLAYFORM3 Π} is a non-empty set of facts that are associated with entities in Π. Example 6.

Figure 4 demonstrates the fact table (upper-left), properties (upper-right), and slices (bottom) derived from the facts of FIG1 .

For example, slice S 6 on property {c 6 } represents facts for projects sponsored by NASA; slice S 4 on properties {c 1 , c 6 } represents facts for space programs sponsored by NASA.

Canonical slice.

Different slices may correspond to the same set of entities.

For example, in Figure 4 , the slice defined by {c 5 , c 6 } corresponds to entity e 5 , the same as slice S 3 , but it has a different semantic interpretation: projects sponsored by NASA and started in 1957.

Based on the extracted knowledge, it is impossible to tell which slice is more precise; reporting and exploring all of them introduces redundancy to the results and also significantly increases the overall problem complexity.

In Midas, we choose to report canonical slices: among all slices that correspond to the same set of entities and facts, the one with the maximum number of properties is a canonical slice.

DISPLAYFORM4 Focusing on canonical slices does not sacrifice generality.

The canonical slice is always unique, and one can infer the unreported slices from the canonical slices by taking any subset of a canonical slice's properties and validating the corresponding entities.

All six slices in Figure 4 are canonical slices that select at least one fact.

Definition 8 (Problem Definition).

Let E be an existing knowledge base, W = {W 1 , ...} be a collection of web sources, T W be the facts extracted from web source W ∈ W, and f (S) be an objective function evaluating the profit of a set of slices on the given existing knowledge base E. The web source suggestion problem finds a list of web source slices, S = {S 1 , ...}, such that the objective function f (S) is maximized.

Inspired by solutions in BID15 BID29 , we quantify the value of a set of slices as the profit (i.e., gain−cost) of using the set of slices to augment an existing knowledge base.

We measure the gain as a function of the number of unique new facts presented in the slices, showing the potential benefit of these facts in downstream applications.

We estimate the cost based on common knowledge-base augmentation procedures BID13 BID25 BID30 , which contain three steps: crawling the web source to extract the facts, de-duplicating facts that already exist in the knowledge base, and validating the correctness of the newly-added facts.

In our implementation, we assume that the gain and cost are linear with respect to the number of (new) facts in all slices.

This assumption is not inherent to our methodology, and one can adjust the gain and cost functions.

Definition 9.

Let S be the set of slices derived from web source W and let E be a knowledge base.

We compute the gain and the cost of S with respect to E as G(S) = | ∪ S∈S S \ E| and C(S) = C crawl (S) + C de-dup (S) + C validate (S), respectively.

The profit of S is the difference: DISPLAYFORM0 In this paper, we measure the crawling cost as C crawl (S) = |S| · f p + W ∈W f c · |T W |, which includes a unit cost f p for training and an extra cost for crawling; de-duplication cost as C de-dup (S) = f d · | S∈S S|, which is proportional to the number of facts in the slices; and validation cost as C validate (S) = f v · | ∪ S∈S S \ E|, which is proportional to the number of new facts in the slices.

For our experiments, we use the default values f p = 10, f c = 0.001, f d = 0.01, and f v = 0.1 (we switch to f p = 1 for the running examples in the paper).

Appendix A includes more details on the gain and cost functions.

Midas uses this profit function as the objective function in Definition 8 to identify the set of web source slices that are best-suited for augmenting a given knowledge base.

The objective of the slice discovery problem is to identify the collection of web source slices with the maximum total profit.

Through a reduction from the set cover problem, we can show that this optimization problem is NP-complete.

In addition, because it is a Polynomial Programming problem with a non-linear objective function, the problem is also APX-complete, which means that no constant-factor polynomial approximation algorithm exists.

Theorem 10 (Complexity of slice discovery).

The optimal slice discovery problem is NPcomplete and it is also APX-complete BID4 .In this section, we first present an algorithm, Midas alg , that solves a simpler problem: identifying the good slices in a single web source (Section 3.1).

We then extend the Midas alg algorithm to the general form of the slice discovery problem and propose a highly-parallelizable framework, Midas, that detects good slices from multiple web sources (Section 3.2).

The problem of identifying high-profit slices in a single web-source is in itself challenging.

As per Definition 5, given a web source and its extracted facts, any combination of properties, which are derived from the facts, may form a web source slice.

Therefore, the number of slices in a single web source can be exponential in the number of extracted facts in the web source.

This factor renders most set cover algorithms, as well as existing source selection algorithms BID15 BID29 , inefficient and unsuitable for solving the slice discovery problem since they often need to perform multiple iterations over all slices in a web source.

Our approach, Midas alg , avoids property combinations that fail to match any extracted fact by constructing the slice hierarchy in a bottom-up fashion and guarantees the result quality by further traversing the trimmed slice hierarchy.

We demonstrate the two steps algorithm for facts in Example 1 through FIG9 in Appendix C.

A key to Midas alg 's efficiency is that it constructs slices only as needed, building a slice hierarchy in a bottom-up fashion, and smartly pruning slices during construction.

The hierarchy is implied by the properties of slices.

For example, slice S 4 ( Figure 4 ) has a subset of the properties of slice S 1 , and thus corresponds to a superset of entities compared to S 1 .

As a result, S 4 is more general and thus an ancestor to S 1 in the slice hierarchy.

Midas alg first generates slices at the finest granularity (least general) and then iteratively generates, evaluates, and potentially prunes slices in the coarser levels.

Midas alg creates a set of initial slices from the entities in the fact table F W .

Each entity e is associated with the facts (s, p, o) ∈ T W that correspond to that entity (s = e).

Each such fact maps to one property (p, o).

Thus, the set of all properties that relate to entity e are: DISPLAYFORM0 For each entity e, Midas alg creates one slice for each combination of properties in C e , such that each property is on a different predicate; if e has a single value for each predicate, there will be a single slice created for e.

The algorithm assigns a level to each slice, corresponding to the number of properties that define the slice.

These initial slices contain a maximal number of properties and are, thus, canonical slices (Definition 2.1).

For example, based on entities in Figure 4 , Midas alg creates three slices, S 1 , S 2 , and S 3 , at level 3 from entities e 1 , e 3 , and e 5 , respectively, and one slice, S 4 , at level 2 from entities e 2 and e 4 .Bottom-up hierarchy construction and pruning.

Starting with the initial slices, Midas alg constructs and prunes the slice hierarchy in a bottom-up fashion.

At each level, Midas alg follows three steps: (1) it constructs the parent slices for each slice in the current level; (2) for each new slice, it evaluates whether it is canonical and prunes it if it is not; (3) if the slice is canonical, it evaluates its profit and prunes the slice if the profit is low compared to other available slices.

Slices pruned during construction are marked as invalid : (1) Constructing parent slices.

At each level, Midas alg constructs the next level of the slice hierarchy by generating the parent slices for each slice in the current level.

To generate the parent slices for a slice, Midas alg uses a process similar to that of building the candidate itemset lattice structure in the Apriori algorithm BID2 .

Given a slice S = σ C (F W ) with properties C = {c 1 , ..., c k }, Midas alg generates k parent slices for S, by removing one property from C at a time.

For example, Midas alg generates three parent slices for slice S 2 : {c 2 , c 4 }, {c 2 , c 6 }, and {c 4 , c 6 }.

For each slice we record its children slices; this will be important for removing non-canonical slices safely, as we proceed to discuss.

(2) Pruning non-canonical slices.

Midas only reports canonical slices, which are slices with a maximal number of properties (Section 2.1).

To identify the canonical slices efficiently, Midas alg relies on the following property.

Proposition 11.

A slice S is canonical if and only if it satisfies one of the following two conditions:(1) slice S is an initial slice defined from an entity; or (2) slice S has at least two children slices that are canonical.

This proposition, proved by contradiction, formalizes a critical insight: the determination of whether a slice is canonical relies on two easily verifiable conditions.

For example, in Figure 4 , there are two slices, S 4 and S 5 , at level 2 and both of them are canonical slices (depicted with solid lines) because 1).

S 4 is one of the initial slices, defined by entities e 2 and e 4 ; and 2).

S 5 has two canonical children, S 2 and S 3 .In order to record children slices correctly after pruning, Midas alg works at two levels of the hierarchy at a time: it constructs the parent slices at level l − 1 before pruning slices at level l.

The removal of a non-canonical slice S, also updates the children list of the slice's parent, S p .

Each child S c of the removed slice S becomes a child of S p if S c is not already a descendant of S p through another node.

For slices in Figure 4 , Midas alg prunes the non-canonical slice ({c 1 , c 3 }, ..., ...) and makes its child slice S 1 a direct child of the parent slice ({c 3 }, ..., ...).

However, it does not make S 1 a child of ({c 1 }, ..., ...) since S 1 is a descendant of ({c 1 }, ..., ...) through slice node S 4 .

(3) Pruning low-profit slices.

For the remaining canonical slices, Midas alg calculates the statistics to identify and prune slices that may lead to lower profit.

This pruning step significantly reduces the number of slices that the traversal (Section 3.1.2) will need to examine.

The pruning logic follows a simple heuristic: the ancestors of a slice are likely to be low-profit if the slice's profit is either negative or lower than that of its descendants.

For a slice S, we maintain a set of slices from the subtree of S, denoted by S LB (S).

This set is selected to provide a lower bound of the (maximum) profit that can be achieved by the subtree rooted at S; we denote the corresponding profit as f LB (S).

f LB (S) is always non-negative, as the lowest profit, achieved by S LB (S) = ∅, is zero.

Let C S be the set of children of slice S. We compute f LB (S) and update S LB (S) by comparing the profit of S itself with the profit of the slices in the lower bound sets (S LB ) of S's children: DISPLAYFORM1 Midas alg marks a slice S as low-profit if its current profit is negative or if it is lower than the total profit that can be obtained from the lower bound slices in its subtree (f LB (S)).

This is because reporting S LB (S) instead of {S} is more likely to lead to a higher profit.

For example, among two canonical slices S 4 and S 5 at level 2 in Figure 4 , Midas alg prunes slice S 4 due to its negative profit.

After pruning non-canonical and low-profit slices, Midas alg significantly reduces the number of slices at level 2.Constructing the hierarchy of slices is related to agglomerative clustering BID31 BID24 , which builds the hierarchy of clusters by merging two clusters that are most similar at each iteration.

However, Midas alg is much more efficient than agglomerative clustering, as we show in our experiments (Section 4).

The hierarchy construction is effective at pruning a large portion of slices in advance, reducing the number of slices we need to consider by several orders of magnitude (Section 4).

However, redundancies, or heavily overlapped slices, may still present in the trimmed slice hierarchy, especially for slices that belong to the same subtree.

The second step of Midas alg traverses the hierarchy top-down to select a final set of slices (Algorithm 1).

In this top-down traversal, DISPLAYFORM0 S ← S ∪ S

S.covered = true 7: if S.covered then 8: for S c in C S do 9:S c .covered = true 10: Return S Midas alg prioritizes valid (unpruned) slices at higher levels of the hierarchy, since they are more likely to produce higher profit and cover a larger number of facts than their descendants.

We initialize unpruned slices as valid (S.valid =true) but not covered in the result set (S.covered =false).

Proposition 12.

Midas alg has O(m |P| ) time complexity, where m is the maximum number of distinct (subject, predicate) pairs, and |P| is the number of distinct predicates in the web source W .According to Theorem 10, the optimal slice discovery problem is APX-complete.

Therefore, it is impossible to derive a polynomial time algorithm with constant-factor approximation guarantees for this problem.

However, as we demonstrate in our evaluation, Midas alg is efficient and effective at identifying multiple slices for a single web source in practice (Section 4).

To detect slices from a large web source corpus, a naïve approach is to apply Midas alg on every web source.

However, this approach leads to low efficiency and low accuracy, as it ignores the hierarchical relationship among web sources from the same web domain, e.g., http://space.skyrocket.de/doc_sat/apollo-history.htm is a child of http://space.

skyrocket.de/doc_sat in the hierarchy.

The naïve approach repeats computation on the same set of facts from multiple web sources and returns redundant results.

For example, given the facts and web sources in Figure 1 , the naïve approach will perform Midas alg on 7 web sources, including 5 web pages, 2 sub-domains, and 1 web domain, and report three slices, "rocket families sponsored by NASA" on web source http://space.skyrocket.de/doc_lau_fam, "rocket families sponsored by NASA and started in 1957" on web source http://space.skyrocket.de/ .../atlas.htm, and "rocket families sponsored by NASA and started in 1971" on web source http://space.skyrocket.de/.../castor-4.htm.

Even though these three slices achieve the highest profit in their respective web sources, they are as a set redundant and lead to a reduction in the total profit: since the web sources are in the same domain, reporting the latter two slices is redundant and hurts the total profit since the first one already covers all their facts.

In this section, we introduce a highly-parallelizable framework that relies on the natural hierarchy of web sources and explores web source slices in an efficient manner.

This framework starts from the finest grained web sources and reuses the derived slices to form the initial slices while processing their parent web source.

This framework not only improves the execution efficiency, but also avoids reporting redundant slices over different web sources in the same web domain.

Here we highlight three core components in the framework.

Sharding.

At each iteration, we take a finer-grained child web source and a list of slices as the input.

We generate a one-level-coarser web domain as parent web source (if any) and use it as the key to shard the inputs.

Detecting.

After sharding, Midas first collects a set of slices for each coarser web source (current) from its finer-grained children, then uses the collected slices to form the initial hierarchy, and applies Midas alg to detect slices for the current web source.

Consolidating.

To avoid hurting the total profit caused by overlapping slices in the parent and children web sources, Midas prunes the slices in the parent web source when there exists a set of slices in the children web sources that cover the same set of facts with higher profit.

Midas delivers the remaining slices in the parent web source as the input for the next round.

In this section, we present an extensive evaluation of the efficiency and effectiveness of Midas over real-world and synthetic datasets.

Our experiments show that Midas is significantly better than the baseline algorithms at identifying the best sources for knowledge base augmentation.

Due to space limit, in this section, we only present experiment results on real-world dataset and we demonstrate the results on synthetic datasets in Appendix D.2.We ran our evaluation on a ProLiant DL160 G6 server with 16GB RAM, two 2.66GHZ CPUs with 12 cores each, running CentOS release 6.6.

ReVerb/NELL: empty initial KB.We evaluate our algorithms over two real-world datasets.

For our experiments on these datasets, we use an empty initial knowledge base and evaluate the precision of returned slices.

ReVerb.

The ReVerb ClueWeb extraction dataset BID17 samples sentences from the Web using Yahoo's random link service and uses 6 OpenIE extractors to extract facts from these sentences.

The dataset includes facts extracted with confidence score above 0.75.

Entities and predicates in ReVerb are presented in unlexicalized format; for example, the fact ("Boston", "be a city in", "USA") is extracted from https://en.wikipedia.org.

The ReVerb dataset contains 15M facts extracted from 20M URLs.

NELL.

The Never-Ending Language Learner project BID11 is a system that continuously extracts facts from text in webpages and maintains those with confidence score above 0.75.

Unlike ReVerb, NELL is a ClosedIE system and the types of entities follow a pre-defined ontology; for example, in the fact ("concept/athlete/MichaelPhelps", "generalizations", "concept/athlete"), extracted from Wikipedia, the subject "concept/athlete/MichaelPhelps" and object "concept/athlete" are both defined in the ontology.

The NELL dataset includes 2.9M facts extracted from 340K URLs.

Evaluation Setup.

Due to the scale of the ReVerb and NELL datasets, we report the precision of the returned slices.

We manually labeled the correctness of the top-K returned web source slice.

Appendix D.1 includes detailed criteria for assigning the labels.

ReVerb-Slim/NELL-Slim: existing KB with adjustable coverage.

The ReVerb and NELL datasets provide the input of the slice discovery problem, but they do not contain the optimal output that suggests "what to extract and from which web source".

To better evaluate different methods, we generate two smaller datasets, ReVerb-Slim and NELL-Slim, over a 100 sampled web sources in the ReVerb and NELL datasets respectively.

We manually label the content of these sources to create an Initial Silver Standard of their optimal slices with respect to an empty existing knowledge base.

We consider that this optimal, manually-curated set of slices forms a complete knowledge base (100% coverage).

We then create knowledge bases of varied coverage, by selecting a subset of the Initial Silver Standard: to create a knowledge base of x% coverage, we (1) randomly select x% of the slices from the Initial Silver Standard; (2) build a knowledge base with the facts in the selected slices; (3) use the remaining slices (those not selected in step 1) to form the optimal output for the new knowledge base.

The ReVerb-Slim and NELL-Slim datasets contain 859K and 508K facts respectively.

Evaluation Setup.

For ReVerb-Slim and NELL-Slim datasets, we select the web sources and manually generate the Silver Standard.

Appendix D.1 includes more details on the detailed steps for generating the Silver Standard.

Naïve.

There are no baselines that produce web source slices, as this is a novel concept.

We compare our techniques with a naïve baseline that selects entire web sources (rather than a slice of their content) based on the number of new facts extracted from each source.

Greedy.

Our second comparison is a greedy algorithm that focuses on deriving a single slice with the maximum profit from a web source.

It relies on our proposed profit function and generates the slice in a web source by iteratively selecting conditions that improve the profit of the slice the most.

AggCluster.

We compare our techniques with agglomerative clustering BID31 , using our proposed objective function as the distance metric.

This algorithm initializes a cluster for each individual entity, and it merges two clusters that lead to the highest non-negative profit gain at each iteration.

The time complexity of this algorithm is O(|E| 2 log(|E|), where |E| is the number of entities in a web source.

Midas (Section 3.1).

Our Midas alg algorithm organizes candidate slices in a hierarchy to derive a set of slices from a single source.

Used as the slice detection module in the parallelizable framework of Midas (Section 3.2), it derives slices across multiple sources.

Note that our parallelizable framework in Section 3.2 also supports the alternative algorithms, including Greedy and AggCluster, by adjusting the slice detection algorithm in the Detecting phase.

Therefore, with the support of our framework, all of these algorithms can easily run in parallel.

For all alternative algorithms, we compare their effectiveness based on their precision, recall, and f-measure; and compare their efficiency based on their total execution time.

Our evaluation on the real-world datasets includes two components.

First, we focus on a smaller version of the datasets, where we can apply our silver standard to better evaluate the result quality using precision, recall, and f-measure across knowledge bases of different cov- • AggCluster Naive erage.

Second, we study the performance of all methods on ReVerb and NELL, reporting the precision of the methods' top-k results, for varying values of k, and their execution efficiency.

Slice quality vs. Knowledge Base coverage.

For this experiment, we evaluate the four methods on the ReVerb-Slim and NELL-Slim datasets, each with the 100 web sources with labeled silver standard and we run the four methods using input knowledge bases of coverage varying from 0 to 80%.

We show the precision-recall curves for three coverage ratios: 0, 0.4, and 0.8 and the precision, recall, and f-measure with increasing coverage ratio from 0 to 0.8.

Due to space limit, we only present the result on ReVerb-Slim dataset in FIG5 and we highlight the major observations of results on the NELL-Slim dataset.

The full result can be found in FIG11 in Appendix D. As shown, Midas performs significantly better than the alternative algorithms, especially on the ReVerb-Slim dataset, but there is a noticeable decline in performance with increased coverage.

This decline is partially an artifact of our silver standard: since the silver standard was generated against an empty knowledge base, the profit of some of its slices drops as the slices now have increased overlap with existing facts.

Midas tends to favor alternative slices to cover new facts, and may return slices that are not included in the silver standard but are, in fact, better.

Greedy performs poorly on both datasets (well under 0.5 for all measures).

Its effectiveness is dominated by its recall, which increases with coverage.

This is expected since in knowledge bases of higher coverage, there are fewer remaining slices for each source in the silver standard.

AggCluster performs poorly for ReVerb-Slim.

This is because AggCluster is more likely to make mistakes for datasets with more entities and predicates.

In addition, AggCluster requires significantly longer execution time compared to Midas (as demonstrated in FIG7 ).

Naïve ranks web sources according to the number of new facts, thus its accuracy heavily relies on the portion of web sources that contain only one high-profit slice.

Thus, it achieves similar recall in all different scenarios.

Overall, the performance of this baseline is low across the board.

Due to the limited size of these two datasets, the execution time of the four methods does not differ significantly.

We evaluate the execution efficiency of the methods through our next experiment on the full datasets, ReVerb and NELL.

Precision and efficiency.

We further study the quality of the results of all four methods by looking at their top-k returned slices, ordered by their profit, when the algorithms operate on an empty knowledge base.

Figures 6a and 6c report the precision for varied values of k up to k = 100, for ReVerb and NELL, respectively.

We observe that the Naïve baseline performs poorly, with precision below 0.25 and 0.4, respectively.

This is expected, as Naïve considers the number of facts that are new in a source, but does not consider possible correlations among them.

Thus, Naïve may consider a forum or a news website, which contains a large number of loosely related extractions, as a good web source slice.

In contrast, Midas outperforms Naïve by a large margin, maintaining precision above 0.75 for both datasets.

The major disadvantage of Greedy is that it may miss many high-profit slices as it only derives a single slice per web source.

However, since we only evaluate the top-100 returns, the precision of Greedy remains high on both datasets.

AggCluster performs well on the NELL dataset, but not as well on ReVerb, which includes a higher number of entities and predicates.

This is because AggCluster is more likely to reach a local optimum for datasets with more entities and predicates.

While AggCluster is comparable to our methods with respect to precision, it does not scale over web sources with larger input, and its running time is an order of magnitude (or more) slower than our methods in most cases.

In particular, its efficiency drops significantly on sources with a large number of facts.

The NELL dataset contains one source that is disproportionally larger, and dominates the running time of AggCluster FIG7 ).

In ReVerb, most sources have a large number of facts, so the increase is more gradual FIG7 ).

In contrast, the execution time of Greedy, and Midas increases linearly.

Naïve is the fastest of the methods, as it simply counts the number of new facts that a web source contributes.

Knowledge extraction systems extract facts from diverse data sources and generate facts in either fixed ontologies for their subjects/predicate categories, or in unlexicalized format: ClosedIE extraction systems, including KnowledgeVault BID13 , NELL BID11 , PROSPERA BID26 , DeepDive/Elementary BID30 BID27 , and extraction systems in the TAC-KBP competition BID12 , often generate facts of the first type; whereas OpenIE extraction system BID17 BID18 normally extract facts of the latter type.

In addition, there are many data cleaning and data fusion tools BID14 BID6 to improve extraction quality of such extraction systems.

Midas solves the problem of identifying web source slices for augmenting the content of knowledge bases by leveraging on the extracted and cleaned facts.

Therefore, the quality of web source slices Midas derives significantly relies on the performance of the above systems.

Similar to source selection techniques BID15 for data integration tasks, Midas also uses customized gain and cost functions to evaluate the profit of a web source slice.

However, the slice discovery problem is fundamentally different from source selection problems since the candidate web source slices are unknown.

Collection Selection BID9 BID8 has been long recognized as an important problem in distributed information retrieval.

Given a query and a set of document collections stored in different servers or databases, collection selection techniques retrieve a ranked list of relevant documents: They first perform the selection algorithm on each collection, based on the pregenerated collection descriptions and a similarity metric, and then integrate and consolidate the results into a single coherent ranked list.

The slice discovery problem is correlated with the collection selection problem: web sources under the same web domain form a collection, which is further described by the extracted facts; our goal, finding the right web sources for knowledge gaps, can also be considered as a query operate on the collections of web sources.

However, instead of a query of keywords, our query is an existing knowledge base.

Other than the difference on the queries, there are several additional properties that render these two problems fundamentally different: first, the similarity metrics, which focus on measuring the semantic similarity, in collection selection, do not apply to the slice discovery problem; second, the web sources in a collection in the slice discovery problem form a hierarchy; third, the slice discovery problem not only targets retrieving relevant web sources, but also generating descriptions for the web sources with respect to our query on the fly.

Finally, the slice discovery problem in this paper is related to clustering of entities in a web source BID23 .

However, it is unclear how to form features for entities.

In addition, existing clustering techniques BID32 , fail to provide any high level description of the content in a cluster, thus they are ill-suited for solving the slice discovery problem.

In this paper, we presented Midas, an effective and highly-parallelizable system, that leverages extracted facts in web sources, for detecting high-profit web source slices to fill knowledge gaps.

In particular, we defined a web source slice as a selection query that indicates what to extract and from which web source.

We designed an algorithm, Midas alg , to detect high-quality slices in a web source and we proposed a highly-parallelizable framework to scale Midas to million of web sources.

We analyzed the performance of our techniques in synthetic data scenarios, and we demonstrated that Midas is effective and efficient in real-world settings.

However, there are still many challenges towards solving this problem due to the quality of current extraction systems.

There is a substantial number of missing extractions due to the lack of training data and one cannot infer the quality of web sources with respect to such missing extractions.

In our future work, we plan to extend our techniques to conquer the limitations of extractions and improve the quality of the derived web source slices.

Crawling.

The first step of the augmentation process is to crawl and extract the facts in a given web source.

This requires training the crawler for the facts in each slice.

We use a unit cost f p to model the cost of training, which includes annotating and schema matching, for each slice.

The cost for the rest of the crawling process is proportional to the size of the web source BID16 .

Measuring the size of web sources is hard due to their diverse design and format; instead, we estimate it based on the total number of facts extracted from the web sources, scaled proportional to an adjustable normalization factor f c : DISPLAYFORM0 De-duplication.

A typical step in the augmentation process is to identify and purge redundant facts before adding them to the knowledge base.

This de-duplication is often performed through linkage BID5 BID22 BID20 between the facts of the slice and those of the knowledge base.

Thus, the de-duplication cost is proportional to the number of facts selected by the web source slice, subject to an adjustable normalization factor (f d ): DISPLAYFORM1 Before adding facts to a knowledge base, it is essential to verify their validity.

The cost of this step is proportional to the new facts that the slice contributes, and subject to an adjustable normalization factor (f v ) that depends on the employed validation technique BID35 BID28 : DISPLAYFORM2 Finally, we compute the cost of slices in the same web domain C(S) as the sum of the respective costs of the crawling, de-duplication, and validation steps.

DISPLAYFORM3 The four adjustable normalization factors included in the computation of each of the three costs relate to the particular techniques used for the corresponding steps (e.g., different de-duplication methods may result in different values for f d ).

In this paper, we set these factors such that they are roughly proportional to the actual execution time of such techniques.

However, one can always adjust the setting of these factors.

For our experiments, we use the default values f p = 10, f c = 0.001, f d = 0.01, and f v = 0.1 (we switch to f p = 1 for the running examples in the paper).

Thus, de-duplication is more costly than crawling, and validation is proportionally the most expensive operation except training.

We measure the suitability of a collection of slices S under the same web domain for augmenting a given knowledge base as the profit of the slice, namely, the difference between the gain and the cost.

Definition 14.

Let S be the web source slices derived from web source W , we denote the gain and the cost of S with respect to knowledge base E as G(S, E) (or G(S)) and C(S, E) (or C(S)), respectively.

DISPLAYFORM0 The profit function underlines three important properties for web source slices.

Productivity.

Midas prioritizes slices that can contribute a larger number of new facts: if S 1 contributes more new facts than S 2 , then G({S 1 }) > G({S 2 }).Specificity.

Midas prioritizes slices with fewer irrelevant facts: if S 1 on W 1 contributes the same number of new facts as DISPLAYFORM1 Dissimilarity.

Midas prioritizes slices with fewer facts overlapping with E: if S 1 contributes the same new facts and is extracted from the same web source as S 2 , but S 1 has more facts already appearing in E, then DISPLAYFORM2 In our objective function f (S), we follow the state-of-the-art procedure and further simplify it with several assumptions: we assume the gain and cost are linear with respect to the number of (new) facts in all slices.

However, such assumptions are not inherent in Midas; one can adjust the gain and cost functions and use the same methodology to derive high-profit web source slices.

Midas uses the above profit function as the objective function f (S) in Definition 8 to identify the set of web source slices that are best-suited for augmenting a given knowledge base.

Note that although we define our gain and cost functions as linear functions over the number of (new) facts in all slices, they are non-linear to the input S since slices in S may overlap with each other.

Example 15.

In Figure 4 , there are three set of slices, {S 2 , S 3 }, {S 5 }, and {S 6 }, that cover all the new facts in the web source.

Among these slices, reporting S 5 is intuitively the most effective option, since S 5 selects all new facts in the web source and covers zero existing one.

We reflect this intuition in our profit function (f (S)): slice {S 5 } has the same gain, but lower de-duplication cost (6f d vs. 13f d ), compared to slice {S 6 } as it contains fewer facts; slice {S 5 } and slices {S 2 , S 3 } also has the same gain, but {S 5 } has lower crawling cost (f p vs. 2f p ) as it avoids the unit cost for training an additional slice.⇒ The optimal solution of the set cover problem is the optimal solution for the constructed slice discovery problem.

Let I as the optimal solution for the set cover problem with |I| sets, the corresponding slices J is the optimal solution for the slice discovery problem with profit |U | − |J|/(|S| + 1).

This is because removing any of the slices in J will hurt the gain by at least 1, but save less than 1 in cost as ∀k > 0, (|J| − k)/(|S| + 1) < 1.

Replacing or adding slices in J will also hurt the gain without improving the cost.

⇐ The optimal solution for the constructed slice discovery problem is the optimal solution of the set cover problem.

Let J as the optimal solution for the slice discovery problem, the corresponding sets I is the optimal solution for the set cover problem.

First, J must cover all facts in the problem.

We may prove this through contradiction: let us assume J does not cover all facts, then any collection of slices, e.g., J , that cover all facts will have a higher profit than J since |J|/(|S| + 1) − |J |/(|S| + 1) < 1.

In addition, among all slice collections that cover all facts, |J| is minimum because otherwise it will not be the optimal solution.

As a result, the corresponding collection of sets, I, is also optimal.

Therefore, the slice discovery problem is NP-Complete.

Proposition 12 BID1 .

A slice S is canonical if slice S has at least two children slices that are canonical.

DISPLAYFORM0 as two children slices of slice S = (C, Π, Π * ).

We say S is also canonical if both S i and S j are canonical.

We prove this through contradiction: Assume S is not canonical, it means that there must exist another slice S = (C , Π, Π * ) such that C ⊂ C .

Since S is the parent of S i and S j , we know that Π i ⊂ Π , Π j ⊂ Π, Π i * ⊂ Π * , and Π j * ⊂ Π * .

As S and S cover the same set of entities and facts, the above conclusion also holds for slice S .

However, since C ⊂ C , S cannot be the parent of S i and S j as there is at least another slice, S , that is between S i and S (or S j and S).

This contradicts with our initial assumption, therefore S must also be canonical.

FIG9 demonstrates the two steps algorithm for identifying slices from a single web source (Section 3.1).

During the slice hierarchy construction step, Midas alg first creates three slices, S 1 , S 2 , and S 3 , at level 3 from entities e 1 , e 3 , and e 5 , respectively, and one slice, S 4 , at level 2 from entities e 2 and e 4 .

Midas alg then generates parent slices for slices at the lowest level.

For example, Midas alg generates three parent slices for slice S 2 : {c 2 , c 4 }, {c 2 , c 6 }, and {c 4 , c 6 }.

While constructing the slice hierarchy, Midas alg prunes non-canonical slices.

For example, at level 2 in FIG9 , slices S 4 and S 5 are canonical slices (depicted with solid lines) because S 4 is one of the initial slices, defined by entities e 2 and e 4 , and S 5 has two canonical children, S 2 and S 3 .In order to record children slices correctly after pruning, Midas alg works at two levels of the hierarchy at a time: it constructs the parent slices at level l − 1 before pruning slices FIG1 .

LB is short for the profit lower bound (f LB (S)), and Cur is short for current profit (f (S)).

The initial slices, identified by extracted entities, are highlighted in light gray, and identified canonical slices in each step are depicted with solid lines.

If the current profit of a slice is lower than the lower bound, we highlight it in red; these slices are low-profit and are eliminated during the pruning stage.

The remaining, desired slices are depicted in bold black lines, and have current profit greater or equal to the lower bound.at level l.

For example, in FIG9 , Midas alg has constructed the parent slices at level 1, as it is pruning slices at level 2.

The removal of a non-canonical slice S, also updates the children list of the slice's parent, S p .

Each child S c of the removed slice S becomes a child of S p if S c is not already a descendant of S p through another node.

In FIG8 , Midas alg prunes the non-canonical slice ({c 1 , c 3 }, ..., ...) and makes its child slice S 1 a direct child of the parent slice ({c 3 }, ..., ...).

However, it does not make S 1 a child of ({c 1 }, ..., ...) since S 1 is a descendant of ({c 1 }, ..., ...) through slice node S 4 .

Besides non-canonical slices, Midas alg also prunes low-profit slices.

For example, in FIG9 there are two canonical slices, S 4 and S 5 , remaining at level 2.

To prune low-profit slices, Midas alg first calculates the statistics of these two slices and then prunes S 4 since its profit is negative.

After pruning non-canonical and low-profit slices FIG8 ), Midas alg significantly reduces the number of slices at level 2 from 8 to 1.

Figure 9 : A snapshot of selected web sources in the silver standard: Among 100 selected web sources, 50 of them contain at least one high-profit slice."correct" if it satisfies two criteria: (1), whether it provides information that is absent from the existing knowledge base; and (2), whether it allows for easy annotation.

We implement these two criteria based on two statistics: (a) The ratio (R new ) of new facts for the covered entities; (b) The ratio (R anno ) of entities that provide homogeneous information.

To evaluate a given web source slice, we first randomly select K or fewer entities and their web pages; then, we display them to human workers, together with the slice description and existing facts associated with the entity; finally, we ask human workers to label the above two statistics.

For this set of experiments on ReVerb and NELL, since the initial knowledge base is empty, the first ratio R new becomes binary: it equals to 1.0 when there exist facts of the associated entities, or 0.0 otherwise.

In our experiment, we set K = 20 and mark a slice as "correct" if both statistics are above 0.5.

ReVerb-Slim/NELL-Slim Datasets Evaluation Setup.

For ReVerb-Slim and NELLSlim datasets, we select the web sources and generate the Initial Silver Standard as follows:(1) we manually select 100 web sources, such that 50 of them contain at least one high-profit slice, with respect to an empty knowledge base; (2) we apply all algorithms on the selected web sources with an empty knowledge base; (3) we manually label slices and web sources returned by the algorithms, and add those labeled as correct to the Initial Silver Standard.

We demonstrate a snapshot of the selected web sources and the description of the labeled silver standard slices for the ReVerb-Slim dataset in Figure 9 .

As described earlier, the initial silver standard allows us to adjust the coverage of the existing knowledge base and the optimal output.

In our experiment, we evaluate the performance of the different methods against knowledge bases of varied coverage, ranging from 0% (empty KB) to 80%.

We use synthetic data to perform a deeper analysis of the tradeoffs between the three algorithms, Greedy, Midas, and AggCluster, that use our objective function and to study the effectiveness of the pruning strategies of our proposed algorithm, Midas.

We Greedy and Naïve perform poorly.

AggCluster competes with Midas, but is significantly slower FIG7 ).•

AggCluster Naive MIDAS Greedy create synthetic data by randomly generating facts in a web source based on user-specified parameters: the number of slices k, the number of optimal slices m ≤ k (output size), and the number of facts n (input size): For each slice, we first generate its selection rule that consists 5 conditions and then creates n · 1% entities in this slice.

To better simulate the real-world scenario, we also introduce some randomness while generating the facts in the optimal slice: for each entity, the probability of having a condition in the corresponding selection rule is above 0.95 and the probability of having a condition absent from the selection rule is below 0.05.

Among k slices, we select m of them as optimal slices and construct the existing knowledge base accordingly: for non-optimal slices, we randomly select 0.95 of their facts and add them in the existing knowledge base.

In addition, we ensure that each optimal web source slice covers at least 5% of the total input facts.

We compare the Greedy, Midas, and AggCluster in terms of their total running times and their f-measure scores FIG13 ).

In our first experiment, we fix b = 20, m = 10 (10 optimal slices out of 20 slices in a web source), and range the number of facts from 1,000 to 10,000.

Midas remains highly accurate in detecting web source slices in all these settings.

However, due to its time complexity, the execution time of Midas grows linearly with the number of facts.

AggCluster tends to make more mistakes when there are more facts and its execution time grows at a significantly higher rate than Midas.

The greedy algorithm, Greedy, runs much faster than the other algorithms, but it can only detect one out of ten optimal slices.

In our second experiment, we use a web source with 5000 facts (n = 5000) on 20 slices (b = 20), and vary the number of optimal slices in the web source from 1 to 10.

We report the execution time and f-measure in FIG8 , respectively.

AggCluster is much slower than Midas and it fails to identify the optimal slices under several settings.

This is expected as AggCluster only combines two slices at a time, thus it needs more iterations to finish and the probability of reaching a local optimum is much higher than Midas.

Notably, Midas achieves perfect f-measure across the board.

Greedy is three times faster than Midas, but its f-measure score declines quickly as the number of slices increases.

This is expected, as Greedy can only retrieve a single high-profit slice.

At the same time, Greedy is able to find the optimal slice when there is only one.

Midas prunes non-canonical slices and low-profit slices while constructing the hierarchy (Section 3.1.1).

Here, we further study the effectiveness of these two pruning strategies by comparing the number of slices in the constructed hierarchy.

More specifically, using synthetic data, we compare Midas with both non-canonical and low-profit slice pruning (Midas-PruneAll), Midas with the pruning of non-canonical slices strategy only (MidasPruneNonCan), and Midas with no pruning strategy (Midas-NoPrune).

FIG1 shows the number of slices with increasing number of facts (n = 1000 ∼ 10000) and a fixed number of optimal slices (m = 10).

Midas-PruneAll generates significantly fewer slices than Midas-PruneNonCan.

Midas-NoPrune needs to examine every non-empty slice in the web source, thus produces several orders of magnitude more slices than Midas-PruneAll and Midas-PruneNonCan.

FIG1 demonstrates the number of slices with fixed number of facts (n = 5000) and an increasing number of optimal slices (m = 1 ∼ 10).

Similar to our observation in the previous experiment, Midas-PruneNonCan and Midas-NoPrune generate significantly more slices than Midas-PruneAll across all settings.

<|TLDR|>

@highlight

This paper focuses on identifying high quality web sources for industrial knowledge base augmentation pipeline.