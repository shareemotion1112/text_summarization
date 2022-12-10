The goal of network representation learning is to learn low-dimensional node embeddings that capture the graph structure and are useful for solving downstream tasks.

However, despite the proliferation of such methods there is currently no study of their robustness to adversarial attacks.

We provide the first adversarial vulnerability analysis on the widely used family of methods based on random walks.

We derive efficient adversarial perturbations that poison the network structure and have a negative effect on both the quality of the embeddings and the downstream tasks.

We further show that our attacks are transferable since they generalize to many models, and are successful even when the attacker is restricted.

Unsupervised node embedding (network representation learning) approaches are becoming increasingly popular and achieve state-of-the-art performance on many network learning tasks BID5 .

The goal is to embed each node in a low-dimensional feature space such that the graph's structure is captured.

The learned embeddings are subsequently used for downstream tasks such as link prediction, node classification, community detection, and visualization.

Among the variety of proposed approaches, techniques based on random walks (RWs) (Perozzi et al.; Grover & Leskovec) are highly successful since they incorporate higher-order relational information.

Given the increasing popularity of these method, there is a strong need for an analysis of their robustness.

In particular, we aim to study the existence and effects of adversarial perturbations.

A large body of research shows that traditional (deep) learning methods can easily be fooled/attacked: even slight deliberate data perturbations can lead to wrong results BID17 BID28 BID6 BID12 BID26 BID10 .So far, however, the question of adversarial perturbations for node embeddings has not been addressed.

This is highly critical, since especially in domains where graph embeddings are used (e.g. the web) adversaries are common and false data is easy to inject: e.g. spammers might create fake followers on social media or fraudsters might manipulate friendship relations in social networks.

Can node embedding approaches be easily fooled?

The answer to this question is not immediately obvious.

On one hand, the relational (non-i.i.d.) nature of the data might improve robustness since the embeddings are computed for all nodes jointly rather than for individual nodes in isolation.

On the other hand, the propagation of information might also lead to cascading effects, where perturbations in one part of the graph might affect many other nodes in another part of the graph.

Compared to the existing works on adversarial attacks our work significantly differs in various aspects.

First, by operating on plain graph data, we do not perturb the features of individual instances but rather their interaction/dependency structure.

Manipulating the structure (the graph) is a highly realistic scenario.

For example, one can easily add or remove fake friendship relations on a social network, or write fake reviews to influence graph-based recommendation engines.

Second, the node embedding works are typically trained in an unsupervised and transductive fashion.

This means that we cannot rely on a single end-task that our attack might exploit to find appropriate perturbations, and we have to handle a challenging poisoning attack where the model is learned after the attack.

That is, the model cannot be assumed to be static as in most other adversarial attack works.

Lastly, since graphs are discrete classical gradient-based approaches BID28 for finding adversarial perturbations that were designed for continuous data are not well suited.

Particularly for RW-based methods, the gradient computation is not directly possible since they are based on a non-differentiable sampling procedure.

How to design efficient algorithms that are able to find adversarial perturbations in such a challenging -discrete and combinatorial -graph domain?We propose a principled strategy for adversarial attacks on unsupervised node embeddings.

Exploiting results from eigenvalue perturbation theory BID35 we are able to efficiently solve a challenging bi-level optimization problem associated with the poisoning attack.

We assume an attacker with full knowledge about the data and the model, thus, ensuring reliable vulnerability analysis in the worst case.

Nonetheless, our experiments on transferability demonstrate that our strategy generalizes -attacks learned based on one model successfully fool other models as well.

Overall, we shed light on an important problem that has not been studied so far.

We show that node embeddings are sensitive to adversarial attacks.

Relatively few changes are needed to significantly damage the quality of the embeddings even in the scenario where the attacker is restricted.

Furthermore, our work highlights that more work is needed to make node embeddings robust to adversarial perturbations and thus readily applicable in production systems.

We focus on adversarial attacks on unsupervised node embedding approaches based on random walks (RWs), and further show how one can easily apply a similar analysis to attack other node embeddings based on factorization.

For a recent extensive survey, also of other non-RW based approaches, we refer to BID5 .

Moreover, while many (semi-)supervised learning methods BID22 BID15 have been introduced, we focus on unsupervised methods since they are often used in practice due to their flexibility in solving various downstream tasks.

Adversarial attacks.

Attacking machine learning models has a long history, with seminal works on SVMs and logistic regression BID4 BID28 .

Deep neural networks were also shown to be highly sensitive to small adversarial perturbations to the input BID36 BID17 .

While most works focus on image classification, recent works have shown the existence of adversarial examples also in other domains BID18 .Different taxonomies exist characterizing the attacks/adversaries based on their goals, knowledge, and capabilities BID30 .

The two dominant attacks types are poisoning attacks that target the training data (the model is trained after the attack) and evasion attacks that target the test data/application phase (the learned model is assumed fixed).

Compared to evasion attacks, poisoning attacks are far less studied BID23 BID30 BID28 BID10 since they require solving a challenging bi-level optimization problem.

Attacks on semi-supervised graph models.

The robustness of semi-supervised graph classification methods to adversarial attacks has recently been analyzed (Zügner et al., 2018; BID13 .

The first work, introduced by Zügner et al. (2018) , linearizes a graph convolutional network (GCN) BID22 to derive a closed-form expression for the change in class probabilities for a given edge/feature perturbation.

They calculate a score for each possible edge flip based on the classification margin and greedily pick the top edge flips with highest scores.

Later, BID13 proposed a reinforcement (Q-)learning formulation where they decompose the selection of relevant edge flips into selecting the two end-points.

Both approaches focus on targeted attacks (misclassify a given node) for the semi-supervised graph classification task.

In contrast, our work focuses on general attacks (decrease the overall quality) on unsupervised node embeddings.

Manipulating graphs.

In the context of graph clustering, BID11 measure the changes in the result when injecting noise to a bi-partite graph of DNS queries, but do not focus on automatically generating attacks.

There is an extensive literature on works that optimize the graph structure to manipulate e.g. information spread in a network (Chen et al.; Khalil et al.) , user opinions BID1 BID8 , shortest paths (Phillips; Israeli & Wood) , page rank scores and other metrics (Avrachenkov & Litvak; Chan et al.) .

Remotely related are poisoning attacks on multi-task relationship learning (Zhao et al., 2018) .

While they exploit the relations between different tasks, they still deal with the classical scenario of i.i.d.

instances within each task.

Robustness and adversarial training.

The robustification of machine learning models has also been studied -known as adversarial machine learning or robust machine learning.

Such approaches are out of scope for this paper and we do not discuss them.

The goal of adversarial training (e.g. via GANs BID14 ) is to improve the embeddings, while our goal is to damage the embeddings produced by existing models by perturbing the graph structure.

Here we explore poisoning attacks on the graph structure -the attacker is capable of adding or removing (flipping) edges in the original graph within a given budget.

We focus mainly on approaches based on random walks and extend the analysis to spectral approaches (Sec. 6.2 in the appendix).

Let G = (V, E) be an undirected unweighted graph where V is the set of nodes, E is the set of edges, and A ∈ {0, 1} |V |×|V | is the adjacency matrix.

The goal of network representation learning is to find a low-dimensional embedding z v ∈ R K for each node with K |V |.

This dense lowdimensional representation should preserve information about the network structure -nodes similar in the original network should be close in the embedding space.

DeepWalk (Perozzi et al.) and node2vec (Grover & Leskovec) learn an embedding based on RWs by extending and adapting the skip-gram architecture BID29 for learning word embeddings.

They sample finite (biased) RWs and use the co-occurrence of node-context pairs in a given window in each RW as a measure of similarity.

To learn z v they maximize the probability of observing v's neighborhood.

We denote withÂ the adjacency matrix of the graph obtained after the attacker has modified certain entries in A. We assume the attacker has a given, fixed budget and is only capable of modifying f entries, i.e. ||Â − A|| 0 = 2f (we have 2f since G is undirected).

The goal of the attacker is to damage the quality of the learned embeddings, which in turn harms subsequent learning tasks such as node classification or link prediction that use the embeddings as features.

We consider both a general attack that aims to degrade the embeddings of the network as a whole, as well as a targeted attack that aims to damage the embedding regarding a specific target or specific task.

The quality of the embeddings is measured by the loss L(A, Z) of the model under attack, with lower loss corresponding to higher quality, where Z ∈ R N ×K is the matrix containing the embeddings of all nodes.

Thus, the goal of the attacker is to maximize the loss.

We can formalize this as the following bi-level optimization problem: DISPLAYFORM0 Here, Z * is always the 'optimal' embedding resulting from the (to be optimized) graphÂ, i.e. it minimizes the loss, while the attacker tries to maximize the loss.

Solving such a problem is highly challenging given its discrete and combinatorial nature, thus we derive efficient approximations.

Since the first step in the embedding approaches is to generate a set of random walks that serve as a training corpus for the skip-gram model, the bi-level optimization problem is even more complicated.

We have DISPLAYFORM0 where RW l is an intermediate stochastic procedure that generates RWs of length l given the graphÂ which we are optimizing.

By flipping (even a few) edges in the original graph, the attacker necessarily changes the set of possible RWs, thus changing the training corpus.

Therefore, this RW generation process precludes any gradient-based methods.

To tackle this challenge we leverage recent results that show that (given certain assumptions) RW based node embedding approaches are implicitly factorizing the Pointwise Mutual Information (PMI) matrix (Yang & Liu, 2015; BID34 .

We study DeepWalk as an RW-based representative approach since it's one of the most popular methods and has many extensions.

Specifically, we use the results from BID34 to sidestep the RW stochasticity.

Lemma 1 BID34 ).

DeepWalk is equivalent to factorizingM = log(max(M, 1)) with DISPLAYFORM1 where the embedding Z * is obtained by the Singular Value Decomposition ofM = U ΣV T using the top-K largest singular values / vectors, i.e. DISPLAYFORM2 Here, D is the diagonal degree matrix with D ii = j A ij , T is the window size, b is the number of negative samples and vol(A) = i,j A ij is the volume.

Since M is sparse and has many zero entries the matrix log(M ) where the log is elementwise is ill-defined and dense.

To cope with this, similar to the Shifted Positive PMI (PPMI) approach the elementwise maximum is introduced to formM .

Using this insight, we see that DeepWalk is equivalent to optimizing minM DISPLAYFORM3 F wherẽ M K is the best rank-K approximation toM .

This in turn means that the loss for DeepWalk when using the optimal embedding Z * for a given graph A is L DW1 (A, Z * ) = |V | p=K+1 σ 2 p where σ p are the singular values ofM (A) ordered decreasingly σ 1 ≥ σ 2 · · · ≥ σ |V | .

This result shows that we do not need to construct random walks, nor do we have to (explicitly) learn the embedding Z * -it is implicitly considered via the singular values ofM (A).

Accordingly, we have transformed the bi-level problem into a single-level optimization problem.

However, maximizing L DW1 is still challenging due to the singular value decomposition and the discrete nature of the problem.

Gradient based approach.

Maximizing L DW1 with a gradient-based approach is not straightforward since we cannot easily backpropagate through the SVD.

To tackle this challenge we exploit ideas from eigenvalue perturbation theory BID35 ) to approximate L DW1 (A) in closed-form without needing to recompute the SVD.

This enables us to efficiently calculate the gradient.

Theorem 1.

Let A be the initial adjacency matrix andM (A) be the respective co-occurrence matrix.

Let u p be the p-th eigenvector corresponding to the p-th largest eigenvalue ofM .

Given a perturbed matrix A , with A = A + ∆A, and the respective change ∆M .

We can approximately compute the loss: DISPLAYFORM4 The proof is given in the appendix.

For a small ∆A and thus small ∆M we obtain a very good approximation, and if ∆A = ∆M = 0 then the loss is exact.

Intuitively, we can think of using eigenvalue perturbation as analogous to taking the gradient of the loss w.r.t.

M (A).

Now, gradient-based optimization is efficient since ∇ A L DW2 (A) avoids recomputing the eigenvalue decomposition.

The gradient provides useful information for a small change, however, here we are considering discrete flips, i.e. = ±1 so its usefulness is limited.

Furthermore, using gradient-based optimization requires a dense instantiation of the adjacency matrix, which has complexity O(N 2 ) in both runtime and memory (infeasible for large graphs).

This motivates the need for our more advanced approach.

Sparse closed-form approach.

Our goal is to efficiently compute the change in the loss L DW1 (A) given a set of flipped edges.

To do so we will analyze the change in the spectrum of some of the intermediate matrices and then derivate a bound on the change in the spectrum of the co-occurrence matrix, which in turn will give an estimate of the loss.

First, we need some results.

Lemma 2.

The matrix S in Eq. 2 is equal to S = U ( T r=1 Λ r )U T where the matrices U and Λ contain the eigenvectors and eigenvalues solving the generalized eigen-problem Au = λDu.

The proof is given in the appendix.

We see that the spectrum of S (and, thus, the one of M by taking scalars into account) is obtainable from the generalized spectrum of A. The difference to BID34 's derivation where a factorization of S using A norm := D −1/2 AD −1/2 is important.

As we will show, our formulation using the generalized spectrum of A is key for an efficient approximation.

Let A = A + ∆A be the adjacency matrix after the attacker performed some edge flips.

As above, by computing the generalized spectrum of A , we can estimate the spectrum of the resulting S and M .

However, recomputing the eigenvalues λ of A for every possible set of edge flips is still not efficient for large graphs, preventing an effective application.

Thus, we derive our first main result: an efficient approximation bounding the change in the singular values of M for any edge flip.

Theorem 2.

Let ∆A be a matrix with only 2 non-zero elements, namely ∆A ij = ∆A ji = 1 − 2A ij corresponding to a single edge flip (i, j), and ∆D the respective change in the degree matrix, i.e. A = A + ∆A and D = D + ∆D. Let u y be the y-th generalized eigenvector of A with generalized eigenvalue λ y .

Then the generalized eigenvalue λ y of A solving λ y A = λ y D u y is approximately: DISPLAYFORM5 where u yi is the i-th entry of the vector u y , and ∆w ij = (1 − 2A ij ) indicates the edge flip, i.e ±1.The proof is provided in the appendix.

By working with the generalized eigenvalue problem in Theorem 2 we were able to express A and D after flipping an edge as additive changes to A and D, this in turn enabled us to leverage results from eigenvalue perturbation theory to efficiently approximate the change in the spectrum.

If we used A norm instead, the change to A norm would be multiplicative preventing efficient approximations.

Using Eq. 3, instead of recomputing λ we only need to compute ∆λ, significantly reducing the complexity when evaluating different edge flips (i, j).

Using this result, we can now efficiently bound the change in the singular values of S .

Lemma 3.

Let A be defined as before and S be the resulting matrix.

The singular values of S are bounded: DISPLAYFORM6 r where π is a permutation simply ensuring that the finalσ p (i, j) are sorted decreasingly, where d min is the smallest degree in A .We provide the proof in the appendix.

Using this result, we can efficiently compute the loss for a rank-K approximation/factorization of M , which we would obtain when performing the edge flip DISPLAYFORM7 based on the matrixM = log(max(M, 1)), there are unfortunately currently no tools available to analyze the spectrum ofM given the spectrum of M .

Therefore, we use L DW3 as a surrogate loss for L DW1 (Yang et al. similarly exclude the element-wise logarithm).

As our experimental analysis shows, the surrogate loss is effective and we are able to successfully attack the node embeddings that factorize the actual co-occurrence matrixM , as well as the original skip-gram model.

Similarly, methods based on spectral embedding, factorize the graph Laplacian and have a strong connection to the RW based approaches.

We provide a similar detailed analysis in the appendix (Sec. 6.2).The overall algorithm.

Our goal is to maximize L DW3 by performing f edge flips.

While Eq. 3 enables us to efficiently compute the loss for a single edge, there are still O(n 2 ) possible flips.

To reduce the complexity when adding edges (see Sec. 4.2 for removing) we instead form a candidate set by randomly sampling C candidate flips.

This introduces a further approximation that nonetheless works well in practice.

For every candidate we compute its impact on the loss via L DW3 and greedily choose the top f flips.1 The runtime complexity of our overall approach is: O(N ·|E|+C ·N log N ).

First, we can compute the generalized eigenvectors of A in a sparse fashion in O(N · |E|).

Then we sample C candidate edges, and for each we can compute the approximate eigenvalues in constant time (Theorem 2).

To obtain the final loss, we sort the values leading to the overall complexity.

The approach is easily parallelizable since every candidate edge flip can be evaluated in parallel.

If the goal of the attacker is to attack a specific node t ∈ V , called the target, or a specific downstream task, it is suboptimal to maximize the overall loss via L DW * .

Rather, we should define some other target specific loss that depends on t's embedding -replacing the loss function of the outer optimization in Eq. 1 by another one operating on t's embedding.

Thus, for any edge flip (i, j) we now need the change in t's embedding -meaning changes in the eigenvectors -which is inherently more difficult to compute compared to changes in eigen/singular-values.

We study two cases: misclassifying a target node and manipulating the similarity of node pairs (i.e. link prediction task).Surrogate embeddings.

To efficiently compute the change in eigenvectors, we define surrogate embeddingsZ * .

Specifically, instead of performing an SVD decomposition on M (or equivalently S with upscaling) and using the results from Lemma 2 we defineZ DISPLAYFORM0 Experimentally, usingZ * instead of Z * as the embedding showed no significant change in the performance on downstream tasks (even on the clean graph; suggesting its general use since it is more efficient to compute).

Now, we can approximate the generalized eigenvectors, and thusZ * (A ), in closed-form: Theorem 3.

Let ∆A, ∆D and ∆w ij be defined as before, and ∆λ y be the change in the y-th generalized eigenvalue λ y as derived in Theorem 2.

Then, the y-th generalized eigenvector u y of A after performing the edge flip (i, j) can be approximated with: DISPLAYFORM1 where E i (x) returns a vector of zeros except at position i where the value is x, d is a vector of the node degrees, • is the Hadamard product, and (·) + is the pseudo inverse.

We provide the proof in the appendix.

Computing Eq. 4 seems expensive at first due to the pseudo inverse term.

However, note that this term does not depend on the particular edge flip we perform.

Thus, we can pre-compute it once and furthermore, parallelize the computation for each y. Similarly, we can pre-compute u y d, while the rest of the terms are all computable in O(1).

For any edge flip we can now efficiently compute the optimal embeddingZ * (A ) using Eqs. 3 and 4.

The t-th row of Z * (A ) is the desired embedding for a target node t after the attack.

Targeting node classification.

The goal is to enforce misclassification of the target t for the downstream task of node classification (i.e. node labels are partially given).

To fully specify the targeted attack we need to define the candidate flips and the target-specific loss responsible for scoring the candidates.

As candidates we use {(v, t)|v = t}. For the loss, we first pre-train a classifier C on the clean embeddingZ * .

Then we predict the class probabilities p t of the target t using the compromisedZ * t,· and we calculate the classification margin m(t) = p t,c(t) − max c =c(t) p t,c , where c(t) is the ground-truth class for t. That is, our loss is the difference between the probability of the ground truth and the next most probable class after the attack.

Finally, we select the top f flips with smallest margin m (note when m(t) < 0 node t is misclassified).

In practice, we average over 10 randomly trained classifiers.

Another (future work) approach is to treat this as a tri-level optimization problem.

Targeting link prediction.

The goal of the attack is: given a set of target node pairs T ⊂ V × V , decrease the similarity between the nodes that have an edge, and increase the similarity between nodes that do not have an edge, by modifying other parts of the graph -i.e.

it is not allowed to directly flip pairs in T .

For example, in an e-commerce graph representing users and items, the goal might be to increase the similarity between a certain item and user, by adding/removing connections between other users/items.

To achieve this, we first train the initial clean embedding without the target edges.

Then, for a candidate set of flips, we estimateZ * using Eqs. 3 and 4 and use them to calculate the average precision score (AP score) on the target set T , withZ * DISPLAYFORM2 T as a similarity measure.

Finally, we pick the top f flips with lowest AP scores and use them to poison the network.

Since this is the first work considering adversarial attacks on node embeddings there are no known baselines.

Similar to works that optimize the graph structure (Chen et al.) we compare with several strong baselines.

B rnd randomly flips edges (we report averages over ten seeds), B eig removes edges based on their eigencentrality in the line graph L(A), and B deg removes edges based on their degree centrality in L(A)

-or equivalently sum of degrees in the original graph.

When adding edges we use the same baselines as above, now calculated on the complement graph, except for B eig since it is infeasible to compute even for medium size graphs.

A DW2 denotes our gradient based attack, A DW3 our closed-form attack, A link our link prediction attack, A class our node classification attack.

The size of the sampled candidate set for adding edges is 20K (for removing edges see Sec. 4.2).We aim to answer the following questions: (Q1) how good are our approximations of the loss; (Q2) how much damage is caused to the embedding quality by our attacks/baselines; (Q3) can we still perform a successful attack when restricted; FORMULA10 FORMULA2 ).

In all experiments, after choosing the top f flips we retrain the embeddings and report the final performance since this is a poisoning attack.

Note, for the general attack, the downstream node classification task is only a proxy for estimating the quality of the embeddings after the attack, it is not our goal to damage this task, but rather to attack the unsupervised embeddings in general.

To estimate the approximation quality we randomly select a subset of 20K candidate flips and compute the correlation between the actual loss and our approximation as measured by Pearson's R score.

For example, for K = 32 we have R(L DW2 , L DW1 ) = 0.11 and R(L DW3 , L DW1 ) = 0.90, clearly showing that our closed-form strategy approximates the loss significantly better compared to the gradient-based one.

Similarly, L DW 3 is a better approximation than L DW2 for K = 16, 64, 128.

To obtain a better understanding we investigate the effect of removing and adding edges separately.

Since real graphs are usually sparse, for removing we set the candidate set to be the set of all edges, with one edge set aside for each node to ensure we do not have singleton nodes.

To obtain candidate edges for adding we randomly sample a set of edges.

We then simply select the top f edges from the candidate set according to our scoring function.

For adding edges, we also implemented an alternative add-by-remove strategy denoted as A abr .

Here, we first add cf -many edges randomly sampled from the candidate set to the graph and subsequently remove (c − 1)f -many of them.

This strategy performed better empirically.

Since the graph is undirected, for each (i, j) we also flip (j, i).

.

Removed/added edges are denoted on the x-axis with negative/positive values respectively.

On FIG2 we see that our strategies achieve a significantly higher loss compared to the baselines when removing edges.

To analyze the change in the embedding quality we consider the node classification task (i.e. using it as a proxy to evaluate quality; this is not our targeted attack).

Interestingly, B deg is the strongest baseline w.r.t.

to the loss, but this is not true for the downstream task.

As shown in FIG2 , our strategies significantly outperform the baselines.

As expected, A DW3 and A abr perform better than A DW2 .

On Cora our attack can cause up to around 5% more damage compared to the strongest baseline.

On PolBlogs, by adding only 6% edges we can decrease the classification performance by more than 23%, while being more robust to removing edges.

Restricted attacks.

In the real world, attackers cannot attack any node, but rather only specific nodes under their control, which translates to restricting the candidate set.

To evaluate the restricted scenario, we first initialize the candidate sets as before, then we randomly choose a given percentage p r of nodes as restricted and discard every candidate that includes them.

As expected, the results in FIG2 show that for increasingly restrictive sets with p r = 10%, 25%, 50%, our attack is able to do less damage.

However, we always outperform the baselines (not plotted), and even in the case when half of the nodes are restricted (p r = 50%) we are still able to damage the embeddings.

With this we are can answer question (Q3) affirmatively -the attacks are successful even when restricted.

Analysis of selected adversarial edges.

In Fig. 2a we analyze the top 1K edges on Cora-ML.

For each edge we consider its source node degree (destination node, resp.) and plot it on the x-axis (yaxis).

The heatmap shows adversarial edge counts divided by total edge counts for each bin.

We see that low, medium and high degree nodes are all represented.

In Fig. 2b we plot the edge centrality distribution for the top 1K adversarial edges and compare it with the distribution of the remaining edges.

There is no clear distinction.

The findings highlight the need for a principled method such as ours since using intuitive heuristics such as degree/edge centrality cannot identify adversarial edges.

To obtain a better understanding of the performance we study the margin m(t) before and after the attack considering every node t as a potential target.

We allow only (d t + 3) flips for attacking each node ensuring the degrees stay similar.

Each dot in Fig. 4 represents one node grouped by its degree in the clean graph (logarithmic bins).

We see that low-degree nodes are easier to misclassify (m(t) < 0), and that high degree nodes are more robust in general -the baselines have 0% success.

Our method, however, can successfully attack even high degree nodes.

In general, our attack is significantly more effective across all bins -as shown by the numbers on top of each box -with 77.89% nodes successfully misclassified on average compared to e.g. only 33.64% for B rnd .

For the link prediction task (Fig. 3) we are similarly able to cause significant damage -e.g.

A link achieves almost 10% decrease in performance by flipping around 12.5% of edges on Cora, significantly better than all other baselines.

Here again, compared to adding edges, removing has a stronger effect.

Overall, answering (Q5), both experiments confirm that our attacks hinder the downstream tasks.

The question of transferability -do attacks learned for one model generalize to other models -is important since in practice the attacker might not know the model used by the system under attack.

However, if transferability holds, such knowledge is not required.

To obtain the perturbed graph, we remove the top f adversarial edges with the A DW3 attack.

The same perturbed graph is then used to learn node embeddings using several other state-of-the-art approaches.

TAB0 shows the change in node classification performance compared to the embeddings learned on the clean graph for each method respectively.

We tune the key hyperparameters for each method (e.g. p and q for node2vec).

Answering (Q6), the results show that our attack generalizes: the adversarial edges have a noticeable impact on other models as well.

We can damage DeepWalk trained with the skip-gram objective with negative sampling (SGNS) showing that the factorization analysis is successful.

We can even damage the performance of semi-supervised approaches such as GCN and Label Propagation.

Compared to the transferability of the baselines (Sec. 6.3) our attack causes significantly more damage.

We demonstrate that node embeddings are vulnerable to adversarial attacks which can be efficiently computed and have a significant negative effect on node classification and link prediction.

Furthermore, successfully poisoning the system is possible with relatively small perturbations and under restriction.

More importantly, our attacks generalize -the adversarial edges are transferable across different models.

Future work includes modeling the knowledge of the attacker, attacking other network representation learning methods, and developing effective defenses against such attacks.

Attacking spectral embedding.

Finding the spectral embedding is equivalent to the following trace minimization problem: DISPLAYFORM0 subject to orthogonality constraints, where L xy is the graph Laplacian.

The solution is obtained via the eigen-decomposition of L, with Z * = U K where U K are the K-first eigen-vectors corresponding to the K-smallest eigenvalues λ i .

The Laplacian is typically defined in three different ways: the unnormalized Laplacian L = D − A, the normalized random walk Laplacian From Lemma 5 we see that we can attack both normalized versions of the graph Laplacian with a single attack strategy since they have the same eigenvalues.

It also helps us to do that efficiently similar to our previous analysis (Theorem.

3).

DISPLAYFORM1 Theorem 4.

Let L rw (or equivalently L sym ) be the initial graph Laplacian before performing a flip and λ y and u y be any eigenvalue and eigenvector of L rw .

The eigenvalue λ y of L rw obtained after flipping a single edge (i, j) is DISPLAYFORM2 where u yi is the i-th entry of the vector u y .Proof.

From Lemma 5 we can estimate the change in L rw (or equivalently L sym ) by estimating the eigenvalues solving the generalized eigen-problem Lu = λDu.

Let ∆L = L − L be the change in the unnormalized graph Laplacian after performing a single edge flip (i, j) and ∆D be the corresponding change in the degree matrix.

Let e i be defined as before.

Then ∆L = (1 − 2A ij )(e i − e j )(e i − e j ) T and ∆D = (1 − 2A ij )(e i e T i + e j e T j ).

Based on the theory of eigenvalue perturbation we have λ y ≈

λ y + u T y (∆L − λ y ∆D)u y .

Substituting ∆L and ∆D are re-arranging we get the above results.

Using now Theorem 4 and Eq. 5 we finally estimate the loss of the spectral embedding after flipping an edge L SC (L rw , Z) ≈ K p=1 λ p .

Note that here we are summing over the K-first smallest eigenvalues.

We see that spectral embedding and the random walk based approaches are indeed very similar.

We provide similar analysis for the the unnormalized Laplacian:Theorem 5.

Let L be the initial unnormalized graph Laplacian before performing a flip and λ y and u y be any eigenvalue and eigenvector of L. The eigenvalue λ y of L obtained after flipping a single edge (i, j) can be approximated by: DISPLAYFORM3 Proof.

Let ∆A = A − A be the change in the adjacency matrix after performing a single edge flip (i, j) and ∆D be the corresponding change in the degree matrix.

Let e i be defined as before.

Then

Upper bound on singular values.

From Lemma 3 we have that L DW 3 is an upper bound on L DW 1 (excluding the elementwise logarithm) so maximizing L DW 3 is principled.

To gain a better understanding of the tightness of the bound we visualize the singular values of S and their respective upper-bound for all datasets.

As we can see in FIG6 , the gap is different for different datasets and relatively small.

Furthermore we can notice that the gap tends to increase for larger singular values.

Transferability of the baselines.

To further support the transferability of our proposed attack we also examine the transferability of the baseline attacks.

Specifically, we examine the transferability of B eig since it is the strongest baseline when removing edges as shown in FIG2 .

We use the same experimental setup as in Sec. 4.4 and show the results in TAB1 .

We can see that compared to our proposed attack the baseline can do a significantly smaller amount of damage (compare to results in TAB0 ).

Interestingly, it can do significant damage to GCN when removing 250 edges on Cora, but not when removing 500 edges.

We plan on exploring this counterintuitive finding in future work.

<|TLDR|>

@highlight

Adversarial attacks on unsupervised node embeddings based on eigenvalue perturbation theory.