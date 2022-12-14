Deep Neutral Networks(DNNs) require huge GPU memory when training on modern image/video databases.

Unfortunately, the GPU memory as a hardware resource is always finite, which limits the image resolution, batch size, and learning rate that could be used for better DNN performance.

In this paper, we propose a novel training approach, called Re-forwarding, that substantially reduces memory usage in training.

Our approach automatically finds a subset of vertices in a DNN computation graph, and stores tensors only at these vertices during the first forward.

During backward, extra local forwards (called the Re-forwarding process) are conducted to compute the missing tensors between the subset of vertices.

The total memory cost becomes the sum of (1) the memory cost at the subset of vertices and (2) the maximum memory cost among local re-forwards.

Re-forwarding trades training time overheads for memory and does not compromise any performance in testing.

We propose theories and algorithms that achieve the optimal memory solutions for DNNs with either linear or arbitrary computation graphs.

Experiments show that Re-forwarding cuts down up-to 80% of training memory on popular DNNs such as Alexnet, VGG, ResNet, Densenet and Inception net.

The standard DNN training process consists of two alternated stages: forward and backward.

FIG0 (a) illustrates an example of feed-forward neural networks.

In the forward stage, the network takes an input tensor, [BatchSize × Channel × W idth × Height], and computes the tensors at each layer until producing the output.

In the backward stage, difference between the output and ground truth is passed back along the network to compute the gradients at each layer.

The regular training approach saves tensors at all layers during forward, because they are all needed to compute gradients during backward.

The total memory cost is the sum of cost over all layers.

In popular backbone DNNs for feature extraction of images, such as AlexNet BID13 ), VGG BID22 ) and ResNet BID10 ), the memory cost increases quadratically with the input image resolution and network depth.

For example, given an median size input tensor of (32, 3, 224, 224) , ResNet101 requires around 5000 MB.

In more challenging tasks, DNNs that detect small objects and large number of object categories require input image resolution of more than 600 × 600 BID18 ; BID23 ; BID17 ).

The memory issue is worse for video-based DNNs, such as CDC BID21 ), C3D BID12 ) and 3D-ResNet BID9 ).

To model complex activities in video, the input tensor may contain 64 frames.

Moreover, DNN training takes much more memory than testing.

In order to train DNNs with large databases and big learning rate, the batch size can be up to 64.

In training DNN compositions, such as Generative adversarial networks (GANs), multiple generator and discriminator networks are simultaneously stored in GPU memory.

Existing efforts to address memory issues presented three main approaches: (1) Better single GPUs.

Recent GPUs provide larger memory at the expense of exponentially growing price and power consumption.

For instance, from TitanXp, Quadro P6000 to Tesla V100, for 1-2.7 times increase in memory, the prices increase 2.8-8.5 times.

(2) Parallelization among multiple GPUs BID8 ; BID20 ; ; BID15 BID16 ; BID27 ; BID2 ; BID1 ), which requires expensive The regular approach saves all tensors during forward, and uses these tensors to compute gradients during backward.

(b) Reforwarding (our) saves a subset of tensors during the first forward, and conducts "Re-forward" to compute tensors for gradients during backward.clusters, introduces substantial I/O cost, and does not reduce the total memory cost.

(3) Low-level heuristic techniques.

Optimization of computation graphs BID3 ), which merges inplace operations into non-inplace operations to cut down memory.

Liveness analysis BID3 ), which dynamically recycles garbage tensors in training epochs.

These approaches are specific to certain DNN structures, data and tasks.

To address above issues, we propose a fundamental approach that explores trade-off between memory and computation power of GPUs.

Note that recent affordable GPUs, although limited in memory ( 12GB), provide exceptional improvement in GPU cores and FLOPS.

Trading computational time for memory is a very attractive solution that make it possible to train very heavy DNNs with finite GPU memory.

Our approach only saves tensors at a subset of layers during the first forward, and conduct only extra local forwards to compute the missing tensors needed during backward.

We call the extra forward process as Re-forwarding.

The total memory cost is the sum of (1) the cost at the subset of layers and (2) the maximum memory cost among local re-forwards.

Training with Reforwarding, see FIG0 (b), leads to substantial memory reduction.

We propose sophisticate theories and efficient algorithms that achieve the optimal memory solution of arbitrary computation graphs.

To alleviate the memory pressure from a single GPU processor, many researchers utilized the wellestablished techniques for distributed computation BID8 ; BID20 BID15 BID16 BID27 ; BID2 ; BID1 ).

These techniques distribute memory pressure to possibly infinite GPUs or server clusters, but do not reduce the total memory cost of DNNs.

Other researchers reduced the memory on finite hardware by optimizing computation graph of DNN and performing liveness analysis.

The computation graph of DNNs describes the dependencies of tensors among layers.

Liveness analysis recycles garbage to manage memory.

These ideas were originated from compiler optimization BID3 ) and has been widely adopted by deep learning frameworks: Theano BID4 ; BID5 ), MXNet BID6 ), Tensorflow BID0 ) and CNTK BID26 ).

Some other techniques efficiently swap data between CPU and GPU BID25 ; BID19 ).

These techniques usually cost extra I/O time and still do not actually reduce the total memory cost.

The closest work to our approach, Chen et al. ), uses the gradient checkpoints (similar to the subset of layers in Re-forwarding).

However, ) only worked on linear computation graph via a heuristic algorithm.

Our approach generates optimal solutions for both linear and arbitrary computation graphs.

Our algorithm reduces training memory by manipulating high-level tensors, therefore is generalizable to any DNNs and their compositions.

All previous techniques are compatible to our approach and can further improve the memory efficiency of DNN training.

Denote a computation graph as G = E, V .

E = {e i } and V = {v i } are the edges and vertices in the computation graph, respectively.

In deep neural networks, the vertices represent the tensors and the edges represent operations.

Denote function l(·) as a measure of memory cost.

V R is the subset of vertices saved during the first forward.

l(v i ) is defined as the memory cost of storing vertex v i .

For two adjacent vertices v i and v j in set V R , the memory cost during re-forwarding from v i to v j is defined as l(v i , v j ) = j−1 t=i+1 l(v t ), which is the sum of cost over all the vertices between v i and v j .

Using these notations, the memory cost of training with re-forwarding is formulated as DISPLAYFORM0 where the first term is the sum of the memory cost of all the stored tensors, and the second term is the maximal cost among the re-forwards.

For easy illustration, we start by formulating Re-forwarding on Linear Computation Graphs (LCG) ( FIG1 ).

For LCGs, Eqn.

1 can be solved in two cases.

Case (2) LCG with Non-identical Vertex Cost: When the assumption of identical cost does not hold, the solution to Eqn.

1 does not have an analytic form.

Denote the maximal Re-forward cost max j l(v j , v j+1 ) as a constant C, and the solution to Eqn.

1 is reduced to solving for min DISPLAYFORM1 Set the maximal term as l(v i , v j )

Construct Accessibility Graph 4:Find the shortest path in the Accessibility Graph as the solution 5:Compute the actual total cost of the solution 6:Save the solution if it's better.

Suppose the actual max term of this solution is B, and l(v i , v j ) = C, skip the loops where DISPLAYFORM0 All the Re-forward costs in an optimal solution satisfy the constraint l(v j , v j+1 ) ≤ C. We solve Eqn.

1 by constructing a new graph, called Accessibility Graph DISPLAYFORM1 DISPLAYFORM2 is equivalent to finding the shortest path from the source vertex and the target vertex in the Accessibility Graph.

Notice that in the optimal solution, the max term equal the one maximal term among all l(v i , v i+1 ) terms.

To traverse all possible max terms, we can simply compute the loss of every vertex pair and use it as a possible max term.

Given a max term C, suppose the actual max term of the solution under C is B and B < C.

It's obvious that for all the max terms B ≤ max < C, the solution would be the same solution.

Therefore, these max terms can be skipped.

Algorithm 1 summarizes the process for searching an optimal solution for LCG.

Suppose there are N vertices in the computation graph, the time complexity of Algorithm 1 is O(N 4 ) 1 .

As generalization of DNNs with LCG, we present theory 2 and algorithms for DNNs with Arbitrary Computation Graphs (ACG), in particular the acyclic directed graphs FIG1 ).

The optimal solution of Re-forwarding corresponds to an optimal division of ACG, such that memory cost (Eqn.

1) is minimum.

We denote that an ACG is divided into end-to-end segments by a set of vertices.

These end-to-end segments can have multiple endpoint vertices, for example, multiple source vertices and multiple target vertices.

In this paper, as an assumption and also for simplification, these end-to-end segments are narrowed down to those with only one source vertex and one target vertex.

Another assumption in the case of ACG is imposed on the operation that has multiple inputs: one can compute the gradients of output with respect to the gradients of inputs without using the current value of inputs.

Examples of operations that meet this assumption are: concatenation (the gradient of output is also the concatenation of the gradient of input), add (the gradient of output equals the gradient of input), etc.

An example that breaks this assumption is multiplication (the gradient of input depends on the input).

Fortunately, most of the popular networks meet this assumption.

A simple way to remove this assumption is to store all the input tensors of this multi-input operation.

However, this is not modeled by our loss function and may lead to sub-optimal solution.

In summary, there are only two assumptions in our approach: (1) the segment in a solution only has two endpoints (source and target).

(2) the multi-input operation can compute the gradients of output without using the current value of input.

Under these two assumptions, our approach is optimal for ACGs. .

Definition 1.

Closed Set: A set s containing vertices and edges is a closed set if and only if it satisfies the following three properties: 1.

All the vertices of s have a common ancestor v i and a common descendent v j ; 2.

Denote the vertex subset of s as V , edge subset as E, and the set of edges between two arbitrary vertices of V ∪ {v i , v j } is E , the edge from v i to v j (if exists) as e ij .

E must either be E or E − {e ij }; 3.

An arbitrary v 1 ∈ V doesn't have edge with another arbitrary v 2 / ∈ V ∪ {v i , v j }.

For multiple valid closed sets between v i and v j , we denote the largest one as DISPLAYFORM0 In the definition of Closed Set, property 1 corresponds to the two endpoint assumption in section 4.1 where the two endpoints become v i and v j in the definition.

Property 2 confines the edge subsets of 1 More detailed complexity analysis is in the appendix due to space limitation 2 All proofs are in the appendix due to space limitation.s to be one of two cases: E or E − {e ij }.

Both cases are valid although they have different edges.

Property 3 guarantees the independence of such a set s, meaning that the vertices within s have no connections with other vertices outside s ∪ {v i , v j }.

As there might be multiple valid closed sets between v i and v j , which corresponds to the Branched Closed Set in Definition 5, we denote the largest closed set between v i and v j as s ij and denote smaller closed set with an extra superscript, such as s 1 ij .

Definition 3.

Splitting Vertex:

A vertex v t ∈ s ij is a splitting vertex of s ij if and only if s it exists, s tj exists and s ij = s it ∪ s tj ∪ {v t } and s it ∩ s tj = ∅ Definition 4.

Splittable Closed Set (Type 1): closed set with at least 1 splitting vertex.

The definition of Splitting Vertex is to describe whether a closed set can be divided into two linearly arranged closed set.

A closed set is splittable if it has at least 1 splitting vertex and is defined as Closed Set Type 1.

Definition 5.

Branched Closed Set (Type 2): A closed set is branched if it has 0 splitting vertex and can be divided into branches: s ij = s DISPLAYFORM1 Definition 8.

Division of Closed Set: For type 1, its division is the linear segments separated by all its splitting vertices; for type 2, its division is all its branches, any of which cannot be divided into more branches; for type 3, its division is its maximal split.

For closed set type 1, it can be divided into linearly arranged segments.

For closed set type 2, it can be divided into branches.

So here we investigate the division of closed set type 3.

As we don't want trivial division, for example, division that is formed by every edge in the closed set, we define Maximal Split to describe the split such that each member of the split is as large as possible.

An example of maximal split is shown in FIG4 .

In the definition of maximal split, the term maximal is implied by saying that any subset of this split cannot be combined into a single closed set.

If it can, then the maximal split will be formed by this larger closed set and all the rest of the previous split.

For closed set type 3, we use its maximal split as its division.

Definition 9.

Division Tree: Division tree is a representation of a computation graph, where the root node is the whole computation graph, the leaf nodes are all the single tensors in the computation graph, and for a non-leaf node, its children is the members of its division.

With the division of 3 types of closed sets, the computation graph can be reorganized into a division tree ( Figure 5 ) where a non-leaf node would be a closed set and its children would be its corresponding division.

The root node is the whole computation graph, the largest closed set, and the leaf nodes would be single tensors in the computation graph.

With division tree, we can apply divide-and-conquer to search for optimal solution.

Figure 5: In this tree, the root node is the whole computation graph.

All the leaf nodes are single tensors.

Every other node except root and leaves is a member of the division of its parent.

Theorem 1.

The division tree of a computation graph is unique and complete.

The uniqueness of the division tree indicates that the optimal solution of the division tree would also be the optimal solution of the whole computation graph.

The completeness indicates that the division tree has included all the possible members of solution and represents the whole search space for the optimal solution.

Theorem 1 is proved in the appendix.

We search optimal solutions for ACGs by solving several sub-problems using Algorithm 2-4 respectively.

Based on these components, we present our final solver as Algorithm 5.Algorithm 2 judges whether a vertex is a splitting vertex of a closed set.

This algorithm mainly follows the Definition 3 and uses vertex set to check the property of a splitting vertex.

With this algorithm, we can judge whether a closed set is type 1 and get its division if it is.

Suppose there are N vertices in s ij , the time complexity of Algorithm 2 is O(N 2 ).

Algorithm 3 examines whether a closed set is branched.

It uses a growing algorithm to check whether an independent subpart of this closed set can form a closed set.

If a non-trivial closed set s ij has an edge from v i to v j , then it's branched because this edge itself can be treated as a closed set.

Combined with Algorithm 2, we can know the type of a closed set and get its division if it's type 2.

Suppose there are N vertices in s ij , the time complexity of Algorithm 3 is O(N 2 ).Algorithm 4 addresses the problem of finding the maximal split, the division of a closed set type 3 s ij .

First get all the possible closed sets within s ij and use a property of maximal split to judge whether this closed set is a member of the maximal split.

The property is: there cannot exist another closed set s ab s ij but contains any member of this maximal split.

This property is proved in Lemma 6 of the appendix.

Suppose there are N vertices in s ij , the time complexity of Algorithm 4 is O(N 4 ).Algorithm 5 is the solver for ACGs.

First, the division tree of the computation graph is built.

Similar to the linear solver, a max term list is formed by the cost of all the possible closed sets for traverse.

Given a max term, we propose a greedy idea: for a closed set, never expand it unless the its cost exceed the max term.

In other word, if the max term doesn't allow a leap over this closed set, we expand it, otherwise, do not expand it.

Because once expanded, some cost of other vertices inside this closed set might be introduced, and the cost will never be smaller than unexpanded.

If some children of the closed set type 1 are expanded, the rest reforms a few linear segments and still can be solved by the linear solver.

If some children of the closed set type 2 or 3 are expanded, the other Initialize a vertex set s = {v k }.

v k ∈ s ij is a randomly chosen vertex.

while True do 7:For any v t ∈ s ij , v t ∈ s that has connection to any v k ∈ s, add v t to s.

if No more vertex can be added to s then 9:Break 10:if s = {v ∈ s ij } then Return false Algorithm 4 Find the maximal split of a non-branched s ij with 0 splitting vertex DISPLAYFORM0 For all the vertices {v} that have paths from v k and have paths to v t .

if ∃v 2 ∈ {v} and v 2 = v k , v t , v 2 has connection to a v 1 ∈ {v} then 4:Form a closed set s kt with all these vertices.

5: for each formed closed set s kt do

If there doesn't exist a s ab such that s kt s ab s ij , put s kt into the maximal split.

For all the children that have cost larger than current max term.

Expand them and solve the next level.

All the expanded children have separated the current closed set to linear segments.

Solve all the linear segments with current max term.

For all the children that have cost larger than current max term.

Expand them and solve the next level.

All the other members remain unexpanded.

Summarize the total loss, save the current solution if it's better.

We evaluated Re-forwarding on two main groups of neural networks (1) networks with linear structures, such as Alexnet BID13 ) and vgg series BID22 ).

(2) networks with non-linear structures, such as Resnet series BID10 ), Densenet series BID11 ) and Inception net BID24 ).

For each network in TAB4 , an computation graph is built such that every vertex is a Float32 tensor, every edge is an operation, and the memory cost of a vertex is its tensor size (measured in MB).

We compared Re-forwarding with Chen ) and the regular training approach.

Note that only worked on linear computation graphs.

To compare with ) on non-linear networks, we manually re-organized all the non-linear computation graphs into linear computation graphs with their splitting vertices, and fed them to (see TAB4 (MB)").

Our Re-forwarding approach directly works on arbitrary computation graphs.

We have also included a customized network ("CustomNet"), on which even the manual version of Chen's approach is not applicable.

Our approach directly works on all networks.

The computation graph of this network is visualized in the appendix.

All experiments were conducted in Pytorch. [BatchSize, 3, 224, 224] .

We also measure the training time (time of 1 training iteration) for the regular approach and our approach.

Each time is measured as the average of 20 iterations.

Our approach has the same training time as Chen's approach and its manual version, see "Space Efficient Training Time" in TAB4 .

Table.

1 shows that Re-forwarding cuts down huge amount of memory from the regular approach at reasonable time overheads: 26% space off and 40% time overhead for Alexnet, around 40% space off and 40% time overhead for Vgg series.

For Resnet series, the deeper network, the more memory was cut down.

On the deepest Resnet152, 80% space off was achieved with only 39% time overhead.

For Densenet series, more than 80% space off was achieved with around 40% time overhead.

Notice that, only works on linear networks.

Its results on non-linear networks were manually synthesized.

Re-forwarding directly works on non-linear networks and constantly outperformed and its "manual" version.

This supports our claim that Re-forwarding is optimal.

Re-forwarding is a fundamental approach that explores trade-off between memory and computation power of GPUs.

By saving tensors at a subset of layers during forward, and conducting extra local forwards for backward, Re-forwarding makes it possible to train very heavy DNNs with finite GPU memory.

To our knowledge, our theoretical and algorithmic results are the first top-down work that achieve an optimal memory solution for arbitrary computation graphs in DNNs.

Re-forwarding can be further embedded and optimized with any low-level techniques such as distributed computing, GPU/CPU swapping, computation graph optimization and liveness analysis.

Same on v q , v q must be v j or v t .

As s ⊂ [s ij ), ∀v 1 ∈ s, v 1 has no edge with v 2 ∈ [s ij ).

As s kj is close, ∀v 1 ∈ s, v 1 has no edge with v 2 ∈ s kj .

∀v 1 ∈ s, v 1 can only have edge with v 2 ∈ [s].

Thus the independence of s is guaranteed.

Therefore, s is closed set, v k is the splitting vertex of s ij .

DISPLAYFORM0 Same on v j , v j is the splitting vertex of s kt Lemma 4.

If s ij has n splitting vertices {v 1 , v 2 , ..., v n }, then s ij = s i1 ∪ s 12 ∪ ... ∪ s nj ∪ {v 1 , v 2 , ..., v n } Proof.

If n = 2, the splitting vertices are DISPLAYFORM1 According to Lemma 3, v 1 is splitting vertex of s i2 and v 2 is splitting vertex of s 1j .

Therefore, DISPLAYFORM2 For n > 2, the lemma can be proved by repetitively using the conclusion in n = 2.

Lemma 6.

Any member of a maximal split can not be the subset of another closed set s s ij .Proof.

Suppose the source vertex of s is v 1 and target vertex is v 2 , a member s xy of the maximal split is inside s. Suppose a member s ab of the maximal split has its source vertex v a inside s and target vertex v b outside s.

Then the boundary vertex (the vertex that has edges to the non-overlapping parts of both sets) must be v 2 , otherwise the independence of s will be violated.

Notice that v 2 is inside s ab and the independence of s ab needs to be guaranteed, for ∀v p ∈ s, v p / ∈ s ∩ s ab , v q ∈ s ∩ s ab , v p has no edge with v q .

Therefore, v a is a splitting vertex of s.

Similarly, if s ba has its target vertex v a inside s and source vertex v b outside s, the boundary vertex must be v 1 and v a is a splitting vertex of s.

For the closed set s, from the discussion above, we know that there are at most 2 members of the maximal split that can overlap with s. Other members must be either completely inside s or completely outside s. Let's discuss the number of members that overlaps with s.

If there are 0 member that overlaps with s, s is the union of a subset of members of the maximal split, which violates the definition of maximal split.

If there is 1 member that overlaps with s, suppose the corresponding splitting vertex is v b , and the boundary vertex is actually v 2 .

Then s 1b is a closed set containing s xy and corresponds to the situation of 0 member overlapping.

s 1b is the union of a subset of members of the maximal split, and violates the definition of maximal split.

If there are 2 members that overlaps with s, suppose they generate two different splitting vertex v a and v b .

Then s ab is a closed set containing s xy and corresponds to the situation of 0 member overlapping.

s ab is the union of a subset of members of the maximal split, and violates the definition of maximal split.

If they generate the same splitting vertex v b , from lemma 5, v b is also the endpoint vertex of at least 1 other member s ab which has to be inside s. Suppose the two overlapping members are s cb that contains v 1 , and s bd that contains v 2 .

As the source vertex of s, v 1 has path to v b and v 1 has path to v a , which implies v b has path to v a .

As the target vertex of s, v 2 has path from v b and v 2 has path from v a , which implies v b has path from v a .

This conflicts with the fact that s is acyclic.

Therefore, this case is not possible.

Therefore, this lemma is proved.

Lemma 7.

If non-branched s ij has at least 1 vertex but has 0 splitting vertex, then its maximal split has length > 2 Proof.

As s ij is not branched, the members of its maximal split cannot have the starting vertex as v i and the ending vertex as v j at the same time.

If s ij has at least 1 vertex, and its maximal split has length 2, then its maximal split must be {[s ik ], [s kj ]}, and v k will be the splitting vertex of s ij , which violates that s ij has no splitting vertex.

If s ij has at least 1 vertex without splitting vertex, it has at least 2 edges and cannot have a trivial length 1 maximal split.

Therefore, its maximal split has length > 2

To prove this uniqueness, we simply discuss the division uniqueness of closed set type 1, 2 and 3.

Proof.

By the definition of this division and Lemma 4, the uniqueness of the division is equivalent to the uniqueness of the splitting vertex set of a closed set type 1.

The splitting vertex set is obviously unique.

Proof.

If there exists another division, there must be a branch member s

Proof.

As the closed set in the division tree has at least 1 vertex, with Lemma 7, we know that the division, i.e. maximal split of a closed set type 3 s ij within the division tree will have length > 2.

Denote this maximal split as {[s pq ]}, we only need to prove this maximal split is unique.

In all the cases, there cannot exist another different maximal split.

Therefore, the maximal split is unique.

Similar with the uniqueness, the completeness of division tree is equivalent to the completeness of the division of a closed set.

To prove this completeness, we simply discuss the division completeness of closed set type 1, 2 and 3.An equivalent statement of the division completeness is: there doesn't exist a closed set whose head is in one member of the division and whose tail is in another member of the division.

Proof.

Suppose there exists a closed set s whose head v p is in one member s 1 and whose tail v q is in another member s 2 .If v p is not an endpoint of s 1 , then according to Lemma 3, v p is also a splitting vertex in s 1 and can break s 1 into smaller segments, which makes v p also the splitting vertex of the whole closed set.

However, v p is not the splitting vertex of the whole closed set s ij .

This also applies to v q .

Therefore, the division of closed set type 1 is complete.

Proof.

Suppose there exists a closed set s whose head v p is in one member s 1 and whose tail v q is in another member s 2 .

Same with closed set type 2, the boundary vertex v has to be the endpoint vertex of s 1 or the independence of s 1 will be violated.

According to Lemma 5, v is the endpoint vertex of at least 3 members, meaning that v will at least have 1 connection with another closed set s 3 .

To maintain the independence of s, s has to include s 3 as well.

However, s 3 also has its endpoints.

This will propagate until s becomes the whole closed set.

Therefore, there cannot exist such a closed set s. The division of closed set type 3 is complete.

DISPLAYFORM0 .

These steps will cost O(N 2 ).

Then we traverse each (v i , v j ) pair to form the edges of the accessibility graph, which also cost O(N 2 ).

Solving the shortest path problem in accessibility graph will also cost O(N 2 ) as the accessibility graph has N vertices.

Therefore, the overall time complexity of Algorithm 1 would be O(N 4 ).The space complexity would be O(N 2 ) for the table of l(v i , v j ) and the accessibility graph itself.

Suppose there are N vertices in the closed set s ij .

In step 1, getting {v in } and {v out } will cost O(N ) time for traversing the ancestors and descendents of v t .

In our implementation, an array a of length N is used to represent {v in } and {v out }: a i = 1 indicates v i ∈ {v in }, a i = 2 indicates v i ∈ {v out } and a i = 0 indicates v i / ∈ {v in } ∪ {v out }.

Then the union check and intersection check in step 2 can be done in O(N ).

The connection check in step 2 traverses the edges and costs O(N 2 ).

Other steps are O(1).

Therefore, the overall time complexity of Algorithm 2 would be O(N 2 ).The space complexity would be O(N ) for the array to represent {v in } and {v out }.

Suppose there are N vertices in the closed set s ij .

The most time consuming part will be from step 5 to step 13.

Other steps are O(1).

In step 5 to step 13, every edge between two vertices in s ij is at most visited once and there are O(N 2 ) edges.

Therefore, the overall time complexity of Algorithm 3 would be O(N 2 ).In our implementation, an array of length N is used to represent the vertex set s = {v k }.

Therefore, the space complexity would be O(N ).

Suppose there are N vertices in the closed set s ij and there are O(N 2 ) vertex pairs.

For each vertex pair, the connection check in step 2-4 will cost O(N 2 ), similar to the connection check in Algorithm 2.

Thus step 1-4 will cost O(N 4 ).

In our implementation, for each vertex in the closed set s ij , we select the largest formed closed set s kt that contains this vertex.

The closed set number is then reduced to O(N ) and step 5-6 can be done in O(N 3 ).

Therefore, the overall time complexity of Algorithm 4 would be O(N 4 )As O(N 2 ) closed sets can be formed in step 1-4 and each closed set is a smaller DAG with O(N ) vertices and cost O(N 2 ) space, the space complexity would be O(N 4 ) for all these closed sets.

B.5 ALGORITHM 5Step 1 is similar to step 1-4 in Algorithm 4 with s ij being the whole computation graph.

Therefore, the overall time complexity for step 1 is O(N 4 ).In step 2, the complexity of building division tree is related to the complexity of getting the division of a closed set.

For closed set type 1, Algorithm 2 is called for each vertex to get all splitting vertices.

Thus getting the division of closed set type 1 cost O(N 3 ) time.

For closed set type 2, Algorithm 3 is used to solve for its division and costs O(N 2 ) time.

For type 3, Algorithm 4 is called to solve for its division.

Notice that we have already stored all possible closed sets in step 1, step 1-4 in Algorithm 4 can be skipped and thus the time complexity of getting the division of closed set type 3 is reduced to O(N 3 ).

Therefore, getting the division of an arbitrary closed set costs O(N 3 ) time.

In depth i of the division tree, suppose there are k closed sets, and the number of vertices of jth closed sets is a j .

To build depth i + 1 of the division tree, we need to get the division of all these closed sets, which will cost j O(a DISPLAYFORM0 As the depth of division tree is at most N , the overall time complexity of step 2 would be O(N 4 ).For step 3-10, if the computation graph is linear, the ACG solver will reduce to LCG solver and has complexity O(N 4 ).

If the computation graph is non-linear, the length of {m} would be O(N 2 ) for there are O(N 2 ) vertex pair.

For a max term m, from step 4-10, the actual time costing part will be step 6 which calls the LCG solver, and other steps would be O(1).

Suppose the LCG solver is called k times, solving problems of a 1 , a 2 , ..., a k vertices.

The total complexity of this would be O(a Step 1 would cost O(N 4 ) space to store all the possible closed sets.

Step 2 would cost O(N 2 ) space for the division tree.

Step 3-10 would cost O(N 2 ) space for calling LCG solver.

In conclusion, the overall time complexity of Algorithm 5 is O(N 4 ) and the overall space complexity of Algorithm 5 is O(N 4 ).

C RUNTIME AND THEORETICAL ANALYSISThe number of vertices in the computation graph and the runtime of ACG Solver (Algorithm 5) for each network are listed in TAB10 .

All the runtimes were measured on a single core of CPU i7-8700.Notice that the runtime is measured on only 1 cpu core, it can be massively reduced by parallelization on multiple cpu cores.

The runtime can also be further reduced through a better implementation as our implementation is a prototype.

Although it might be concerning that the runtime is too much for some deep networks, it is still relatively small compared to training processes which might cost days or even weeks.

More importantly, solving the optimal solution for a network is an one-time effort.

The optimal solutions for all popular networks will be released online for people to use without taking the time to run ACG solver.

To see how well the reality matches with the theory, we also compare the measured memory cut off and theoretical memory cut off (given by Algorithm 5) in TAB10 .

Observe that all the measured memory cut off are slightly lower than theoretical memory cut off.

This is because, in implementation, we assume that the whole input tensors of each operation are always stored for backward.

In reality, some operations only need to store small tensors for backward.

For example, batchnormalization only needs a few statistics for backward and doesn't need the whole input tensor.

We visualize the computation graph of Alexnet, vgg11, vgg13, vgg16 , vgg19 and CustomNet and the solution of our approach (in green) and the solution of Chen's approach (in red).

In the computation graphs, the cost of each vertex and the actual operation of each edge are also marked.

The cost of each vertex is the size of this tensor during forward given the input as [1, 3, 224, 224] ([1, 3, 300, 300] for inception v3).

For example, in Alexnet, the input is [1, 3, 224, 224] and thus the source vertex has the cost 150528 = 1 × 3 × 224 × 224.

After 2D convolution and relu, the tensor becomes [1, 64, 55, 55] and thus the second vertex has the cost 193600 = 1×64×55×55.

@highlight

This paper proposes fundamental theory and optimal algorithms for DNN training, which reduce up to 80% of training memory for popular DNNs.