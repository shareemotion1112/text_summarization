We introduce CGNN, a framework to learn functional causal models as generative neural networks.

These networks are trained using backpropagation to minimize the maximum mean discrepancy to the observed data.

Unlike previous approaches, CGNN leverages both conditional independences and distributional asymmetries to seamlessly discover bivariate and multivariate   causal structures, with or without hidden variables.

CGNN does not only estimate the causal structure, but a full and differentiable generative model of the data.

Throughout an extensive variety of experiments, we illustrate the competitive  esults of CGNN w.r.t state-of-the-art alternatives in observational causal discovery on both simulated and real data, in the tasks of cause-effect inference, v-structure identification, and multivariate causal discovery.

Deep learning models have shown extraordinary predictive abilities, breaking records in image classification BID19 , speech recognition , language translation BID1 , and reinforcement learning BID33 .

However, the predictive focus of black-box deep learning models leaves little room for explanatory power.

In particular, current machine learning paradigms offer no protection to avoid mistaking correlation by causation.

For example, consider that we are interested in predicting a target variable Y given a feature vector (X 1 , X 2 ).

Assume that the generative process underlying (X 1 , X 2 , Y ) is described by the equations: DISPLAYFORM0 , where (E X2 , E y ) are additive noise variables.

These equations tell that the values of Y are computed as a function of the values of X 1 , and that the values of X 2 are computed as a function of the values of Y .

The "assignment arrows" emphasize the asymmetric relations between the three random variables: we say that "X 1 causes Y ", and that "Y causes X 2 ".

However, since X 2 provides a stronger signal-to-noise ratio for the prediction of Y , the least-squares solution to this problem iŝ Y = 0.25X 1 + 0.5X 2 , a typical case of inverse regression BID7 .

Such least-squares prediction would explain some changes in Y as a function of changes in X 2 .

This is a wrong explanation, since X 2 does not cause the computation of Y .

Even though there exists the necessary machinery to detect all the cause-effect relations in this example BID15 , common machine learning solutions will misunderstand how manipulating the values and distributions of (X 1 , X 2 ), or how changing the mapping from Y to X 2 , affect the values of Y .

Mistaking correlation by causation can be catastrophic for agents who must plan, reason, and decide based on observation.

Thus, discovering causal structures is of crucial importance.

The gold standard to discover causal relations is to perform experiments BID27 .

However, experiments are in many cases expensive, unethical, or impossible to realize.

In these situations, there is a need for observational causal discovery, that is, the estimation of causal relations from observation alone BID35 BID28 .

The literature in observational causal discovery is vast (see Appendix B for a brief survey), but lacks a unified solution.

For instance, some approaches rely on distributional asymmetries to discover bivariate causal relations BID15 BID40 BID4 BID37 BID6 , while others rely on conditional independence to discover structures on three or more variables BID35 BID0 .

Furthermore, different algorithms X 5 DISPLAYFORM1 Figure 1: Example of causal graph and associated functional model for X = (X 1 , . . .

, X 5 ).FCMs are generative models.

We can draw a sample x = (x 1 , . . .

, x d ) from the distribution P := P (X) by observing the FCM at play.

First, draw e i ∼ Q for all i = 1, . . . , d. Second, construct Pa(i;G) , e i ) in the topological order of G. Since this process observes but does not manipulate the equations of the FCM, we call x one observational sample from P , the observational distribution of X. However, one FCM contains more information than the observational distribution alone, since we can decide to manipulate any of its equations and obtain a new distribution.

For instance, we could decide to set and hold constant X j = 0.1, hereby removing all the causal influences X k → X j , for all k ∈ Pa(j; G).

We denote by P do(Xj =0.1) (X) the corresponding interventional distribution.

Importantly, intervening is different from conditioning (correlation does not imply causation).

Understanding the effect of interventions requires the (partial) knowledge of the FCM.

This is why this work focuses on discovering such causal structures from data.

DISPLAYFORM2 Formal definitions and assumptions Two random variables (X, Y ) are conditionally independent given Z if P (X, Y |Z) = P (X|Z)P (Y |Z).

Three of random variables (X, Y, Z) form a v-structure iff their causal structure is X → Z ← Y .

The random variable Z is a confounder (or common cause) of the pair of random variables (X, Y ) if (X, Y, Z) have causal structure X ← Z → Y .

The skeleton U of a DAG G is obtained by replacing all the directed edges in G by undirected edges.

Discovering the causal structure of a random vector is a difficult task when considered in full generality.

Because of this reason, the literature in causal inference relies on a set of common assumptions BID27 .

The causal sufficiency assumption states that there are no unobserved confounders.

The causal Markov assumption states that all the d-separations in the causal graph G imply conditional independences in the observational distribution P .

The causal faithfulness assumption states that all the conditional independences in the observational distribution P imply d-separations in the causal graph G.

We call Markov equivalence class to the set of graphs containing the same set of d-separations.

When using the causal faithfulness assumption and conditional independence information, we are able to recover the Markov equivalence class of the causal structure underlying a random vector -which, in some cases contains one graph, the causal structure itself.

Markov equivalence classes are DAGs where some of the edges remain undirected.

Learning FCMs from data using score methods Consider a random vector X = (X 1 , . . .

, X d ) following the FCM C = (G, f, Q) with associated observational distribution P .

Furthermore, assume access to n samples drawn from P , denoted by DISPLAYFORM3 , where DISPLAYFORM4 for all i = 1, . . .

, n. Given these data, the goal of observational causal discovery is to estimate the underlying causal DAG G and the causal mechanisms f .One family of methods for observational causal discovery are score-based methods BID0 .

In essence, score-based methods rely on some score-function S(G, D) to measure the fit between a candidate set {G, f } and the observed data D. Then, we select the DAG on d variables achieving the maximum score as measured by S. As an example of score-function, consider the Bayesian Information Criterion (BIC): DISPLAYFORM5 where pθ j the maximum-likelihood estimate of a simple parametric family of conditional distributions p θ∈Θ allowing efficient density evaluation.

The term λ ∈ [0, ∞) penalizes the number of edges (that is, the model complexity assuming equal number of parameters per edge) in the graph.

Finally, we may associate each edge X i → X j in G to an importance or confidence score proportional to its contribution to the overal loss: as DISPLAYFORM6 A naïve score-based method would enumerate all the DAGs of d variables and select the one maximizing S. Unfortunately, the number of DAGs over d nodes is super-exponential in d. Thus, the brute-force search of the best DAG is intractable, even for moderate d. Inspired by BID38 ; BID25 , we assume in this paper known graph skeletons.

Such a skeleton may arise from expert knowledge or a feature selection algorithm algorithm BID39 under standard assumptions such as causal Markov, faithfulness, and sufficiency.

Given a skeleton with k edges, causal discovery reduces to selecting one out of the O(2 k ) possible edge orientations.

This section proposes a framework to learn FCMs from data by leveraging the representational power of generative neural networks.

In particular, we propose to estimate FCMs C asĈ = (Ĝ,f ,Q), witĥ DISPLAYFORM0 for all i = 1, . . . , d. Here,Ĝ is the estimated causal graph of X, the functionsf = (f 1 , . . .

,f d ) are the estimated causal mechanisms of X producing the estimated observed variablesX = (X 1 , . . .

,X d ), and the estimated noise variablesÊ = (Ê 1 , . . .

,Ê d ) are sampled from a fixed distributionQ. Given the estimated FCM (1), we can draw n samples from its observational distributionP (see Section 2) and construct the estimated observational samplesD DISPLAYFORM1 .

We parametrize the equations (1) as generative neural networks, also known as conditional generators BID8 .

Without loss of generality, we assume that the independent noise variableŝ E are sampled from an univariate Normal distribution BID37 .

Then, we propose the following score-function to measure the fit between a candidate structureĜ and data D: DISPLAYFORM2 ( 2) where MMD is the Maximum Mean Discrepancy statistic BID10 : DISPLAYFORM3 The MMD statistic scores a graphĜ by measuring the discrepancy between the data observational distribution P and the estimated observational distributionP , on the basis of their samples.

When using a characteristic kernel k such as the Gaussian kernel k(x, x ) = exp(−γ x − x 2 2 ), MMD is an well-defined score-function: it is zero if and only if P =P as n → ∞ BID10 .

Since the computation of MMD k takes O(n 2 ) time, our experiments will also consider an approximation based on m random features (Lopez-Paz, 2016), denoted by MMD m k .

Appendix A offers a brief exposition on MMD.

In a nutshell, CGNN implements Occam's razor to prefer simpler models as causal.

Unlike previous methods, CGNN can seamlessly leverage both distributional asymmetries (due to the representational power of generative networks) and conditional independences (due to the joint minimization of those networks using MMD) to score both bivariate and multivariate graphs.

For a differentiable kernel k such as the Gaussian kernel, the score function (2) is differentiable and therefore CGNN is trainable using backpropagation.

CGNN is a directed acyclic graph of conditional generator networks that result in a flexible generative model of the data causal structure.

Searching causal graphs with CGNN Using the CGNN score (2), we propose the following greedy approach to orient a given skeleton: 1.

Orient each X i − X j as X i → X j or X j → X i by selecting the 2-variable CGNN with the best score.2.

Remove all cycles: all paths starting from a random set of nodes are followed iteratively until all nodes are reached; an edge pointing towards an already visited node forms a cycle, so is reversed.3.

For a number of iterations, reverse the edge that leads to the maximum improvement over a d-variable CGNN, without creating a cycle .Dealing with hidden confounders The search method above assumes the causal sufficiency assumptions: or, the non-existence of hidden confounders.

We address this issue in a variant of our algorithm as follows.

When assuming the existence of confounders, each edge X i − X j in the skeleton is due to one out of three possibilities: either X i → X j , X j ← X i , or there exists an unobserved variable E i,j such that X i ← E i,j → X j .

Therefore, each equation in the FCM is extended to: DISPLAYFORM4 where Ne(i; S) ⊂ {1, . . .

d} is the set of indices of the variables adjacent to X i in the skeleton.

Here each E i,j ∼ Q and denotes the hypothetical unobserved common causes of X i and X j .

For instance, if we hide X 1 from the FCM described in Figure 1 , this would require considering a confounder E 2,3 .

Finally, when considering hidden confounders, the third step above considers three possible mutations of the graph: reverse, add, or remove an edge.

Here, the term λ|Ĝ| takes an active role and promotes simple graphs.

DISPLAYFORM5

We evaluate the performance of CGNN at discovering different types of causal structures.

We study the problems of discovering cause-effect relations (Section 4.1), v-structures (Section 4.2), and multivariate causal structures without (Section 4.3) or with (Section 4.4) hidden variables.

Our experiments run at an Intel Xeon 2.7GHz CPU, and an NVIDIA 1080Ti GPU.

MMD uses a sum of Gaussian kernels with bandwidths γ ∈ {0.005, 0.05, 0.25, 0.5, 1, 5, 50}. CGNN uses onehidden-layer neural networks with n h ReLU units, trained with the Adam optimizer BID18 and initial learning rate 0.01.

According to preliminary experiments, using all data for both training and evaluating models produces good results, since resampling noise variables conbats overfitting.

Also, our best results follow when using the whole data as a minibatch.

We train CGNN during n train = 1000 epochs and evaluate it on n eval = 500 generated samples.

We ensemble CGNN training over n run = 32 random initializations for MMD k and n run = 64 for MMD m k .

Regarding CGNN model selection, the number of hidden units n h is the most sensitive hyperparameter, and should be cross-validated for every application.

The number of hidden units n h relates to the flexibility of the CGNN to model each of the causal mechanisms.

For small n h , we may miss some of the patterns in the data.

For a large n h , we may find over-complicated explanations from effects to causes.

Therefore, our interest is to find the smallest n h explaining the data well.

We illustrate such Occam's razor principle in FIG0 , where we learn two bivariate CGNNs of different complexity (n h = 2, 5, 20, 100) using data from the FCM: FIG0 shows the associated MMDs (averaged on 32 runs), confirming the importance of capacity control BID40 .

On this illustrative case the most discriminative value appears for n h = 2.

DISPLAYFORM0

Under the causal sufficiency assumption, the statistical dependence between two random variables X and Y is due to a causal relation X → Y or due to a causal relation X ← Y BID27 .

Given data from the observational distribution P (X, Y ), this section evaluates the performance of CGNN to decide whether X → Y or X ← Y .In the following, we use five cause-effect inference datasets, covering a wide range of associations.

CE-Cha contains 300 cause-effect pairs from the challenge of BID11 .

CE-Net contains 300 artificial cause-effect pairs generated using random distributions as causes, and neural networks as causal mechanisms.

CE-Gauss contains 300 artificial cause-effect pairs as generated by BID43 , using random mixtures of Gaussians as causes, and Gaussian process priors as causal mechanisms.

CE-Multi contains 300 artificial cause-effect pairs built with random linear and polynomial causal mechanisms.

In this dataset, we simulate additive or multiplicative noise, applied before or after the causal mechanism.

CE-Tueb contains the 99 real-world scalar cause-effect pairs from the Tübingen dataset , concerning domains such as climatology, finance, and medicine.

We set n ≤ 1, 500.

See our implementation for details.

CGNN is compared to following algorithms: The Additive Noise Model, or ANM , with Gaussian process regression and HSIC independence test.

The Linear Non-Gaussian Additive Model, or LiNGAM BID32 , a variation of Independent Component Analysis to identify linear causal relations.

The Information Geometric Causal Inference, or IGCI BID4 , with entropy estimator and Gaussian reference measure.

The Post-Non-Linear model, or PNL BID40 , with HSIC test.

The GPI method BID37 , where the Gaussian process regression with higher marginal likelihood is preferred as the causal direction.

The Conditional Distribution Similarity statistic, or CDS BID6 , which prefers the causal direction with lowest variance of conditional distribution variances.

The award-winning method Jarfo BID6 , a random forest classifier trained on the ChaLearn Cause-effect pairs and hand-crafted to extract 150 features, including the methods ANM, IGCI, CDS, and LiNGAM.The code for ANM, IGCI, PNL, GPI, and LiNGAM is available at https://github.com/ ssamot/causality.

We follow a leave-one-dataset-out scheme to select the best hyperparameters for each method.

For CGNN, we search for the number of hidden neurons n h ∈ {5, 10, 15, 20, 25, 30, 35, 40, 50, 100}. The leave-one-dataset-out hyperparameter selection chooses n h equal to 35, 35, 40, 30, 40 for the CE-Cha, CE-Net, CE-Gauss, CE-Multi and CE-Tueb datasets respectively.

For ANM, we search the Gaussian kernel bandwidth γ used in the Gaussian process regression in {0.01, 0.1, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 2, 5, 10}. For LiNGAM and IGCI, there are no parameters to set.

For PNL, we search for the significance level of the independence test α ∈ {0.0005, 0.005, 0.01, 0.025, 0.04, 0.05, 0.06, 0.075, 0.1, 0.25, 0.5}. For GPI, we use the default parameters from the original implementation.

For CDS, we search for the best discretization of the cause variable into {1, . . .

, 10} levels.

For Jarfo, we train the random forest using 4, 000 cause-effect pairs generated in the same way as the proposed datasets, except the one used for testing.

TAB1 reports the Area Under the Precission/Recall Curve (AUPRC) associated to the binary classification problem of deciding "X → Y " or "X ← Y " for each cause-effect pair, for all methods and datasets.

The table also shows the computational time (in both CPU and GPU), and computational complexity for methods.

The least performing methods are those based on linear regression.

The methods CDS and IGCI perform well on a few datasets.

This indicates the existence of certain biases (such as causes having always higher entropy than effects) on such datasets.

ANM performs well when the additive noise assumption holds (for instance, CE-Gauss), but badly otherwise.

PNL, a generalization of ANM, compares favorably to these methods.

Jarfo, the method using thousands of training cause-effect pairs to learn from data, performs well on artificial data but badly on real examples.

The generative methods GPI and CGNN show a good performance on most datasets, including the real-world cause-effect pairs CE-Tueb.

In terms of computation, generative methods are the most expensive alternatives.

Fortunately for CGNN, the approximation of MMD with random features (see Appendix A) does not degrade performance, but reduces the computation time.

Overall, these results suggest that CGNN is competitive compared to the state-of-the-art on the cause-effect inference problem, where it is necessary to discover distributional asymmetries.

This section studies the performance of CGNN on the task of identifying the causal structure of three random variables (A, B, C) with skeleton A − B − C. The four possible structures are the chain A → B → C, the reverse chain A ← B ← C, the v-structure A → B ← C, and the reverse v-structure A ← B → C. Other skeletons are not of interest, since the absence of an edge (statistical independence) is easier to discover, and the remaining edge could be oriented using the bivariate methods described in the previous section.

Three of the possible structures (the chain, the reverse chain, and the reverse v-structure) are Markov equivalent, and therefore indistinguishable from each other using statistics alone.

Therefore, the goal of this section is to use CGNN to determine whether P (A, B, C) follows or not an FCM with causal graph: A → B ← C. This section considers an FCM with identity causal mechanisms and Normal noise variables; for instance, B ← A + E B , where E B ∼ N (0, 1).

Therefore, the joint distribution of one cause and its effect is symmetrical (a two-dimensional Gaussian), and the bivariate methods used in the previous section all fail to apply.

To succeed at this task, a causal discovery method must reason about the conditional independences between the three random variables at play.

Our protocol fits one CGNN for each of the four possible causal graphs with skeleton A − B − C. Then, we evaluate MMD of each of the four CGNN models, and prefer the one achieving the lowest MMD.

Table 2 summarizes our results: CGNN assigns the lowest MMD to the v-structure hypothesis on those datasets generated by v-structures, and assigns the largest MMD to the v-structure hypothesis on those datasets not generated by v-structures.

Sections 4.1 and 4.2 show the two complementary properties of the CGNN: leveraging distributional asymmetries and conditional independences.

Consider a random vector X = (X 1 , ..., X d ).

Our goal is to find the FCM of X under the causal Markov, faithfulness and causal sufficiency assumptions.

At this point, we will assume known skeleton, so the problem reduces to orienting every edge.

To that end, all experiments provide all algorithms the true graph skeleton, so their ability to orient edges is compared in a fair way.

This allows us to separate the task of orienting the graph from that of uncovering the skeleton.

We draw 500 samples from four artificial causal graphs G 2 , G 3 , G 4 , and G 5 on 20 variables.

For i = {2, . . . , 5}, the variables in the graph G i have a random number of parents between 1 and i. We build the graphs with polynomial mechanisms, and additive/multiplicative noise.

We compare CGNN to the PC algorithm BID35 , the score-based method GES (Chickering, 2002), ANM, LiNGAM, and Jarfo.

For PC, we employ the better-performing, order-independent version of the PC algorithm proposed by BID2 .

PC needs the specification of a conditional independence test.

We compare PC-Gaussian, which employs a Gaussian conditional independence test on Fisher z-transformations, and PC-HSIC, which uses the HSIC conditional independence test with the Gamma approximation BID9 .

For both conditional independence tests, the significance level achieving best results is α = 0.1.

For GES, the best penalization parameter is λ = 3.11.

PC and GES are implemented in the pcalg package .

For CGNN, n h is set to 20.

We also compare to pairwise methods presented in the last section : ANM, LiNGAM, and Jarfo.

TAB3 displays the performance of all algorithms measured from the area under the precision/recall curve.

Overall, the best performing method is PC-HSIC, followed closely by CGNN.

The performance of PC-HSIC is best for denser graphs.

This is because the PC algorithm uses a majority voting rule to decide each orientation, one strategy well suited to dense known skeletons, since one edge belongs to multiple v-structures.

However, CGNN offers the advantage to orient all the edges (while some edges remain undirected by PC-HSIC) and to deliver a full generative model useful for simulation (while PC-HSIC only gives the graph).

To explore the scalability of our method, we were able to extend the experiment on 5 graphs G 3 with 100 variables, achieving an AUPRC of 85.5 ± 4, in 30 hours of computation on four NVIDIA 1080Ti GPUs.

In real applications, some confounding variables may be unobserved.

We propose to use the same data from the previous section, but hide some of the 20 observed variables in the graph.

More specifically, we hide three random variables that cause at least two others in the same graph.

Consequently, the skeleton now includes additional edges X − Y for all pairs of variables (X, Y ) that are consequences of the same hidden cause (confounder).

The goal in this section is to orient the edges due to direct causal relations, and to remove those edges due to confounding.

We compare CGNN to the RFCI algorithm , which is a modification of the PC algorithm that accounts for hidden variables.

As done in the previous section, we compare variants of RFCI based on Gaussian or HSIC conditional independence tests.

We also evaluate the performance of the data-driven method Jarfo, this time trained on the whole Kaggle data of BID11 , in order to classify relations into X → Y , X ← Y , or X ← Z → Y (confounder).

For CGNN, we penalize the objective function (2) with λ = 5 × 10−5.

TAB4 shows that CGNN is robust to the existence of hidden confounders, achieving state-of-the-art performance in this task.

Interestingly, the true causal relations exhibit a high confidence score, while edges due to confounding effects are removed or have low confidence scores.

Overall, CGNN performs best on the graphs G 2 , G 3 , and G 4 , and is slightly outperformed by RFCI-HSIC on the denser graph G 5 .

However, CGNN is the only approach providing a generative model of the data.

We introduced a new framework to learn functional causal models based on generative neural networks.

We train these networks by minimizing the discrepancy between their generated samples and the observed data.

Such models are instances of the bigger family of FCMs for which each function is a shallow neural network with n h hidden units.

We believe that our approach opens new avenues of research, both from the point of view of leveraging the power of deep learning in causal discovery and from the point of view of building deep networks with better structure interpretability.

Once the model is learned, the CGNNs present the advantage to be fully parametrized and may be used to simulate interventions on one or more variables of the model and evaluate their impact on a set of target variables.

This usage is relevant in a wide variety of domains, typically among medical and sociological domains.

Five directions for future work are to i) lower the computational cost of CGNN, ii) extend CGNN to deal with categorical data, iii) explore better heuristics for causal graph search, iv) adapt our methods for temporal data and v) obtain theoretical guarantees for basic use cases.

The Maximum Mean Discrepancy (MMD) statistic BID10 measures the distance between two probability distributions P andP , defined over R d , as the real-valued quantity DISPLAYFORM0 Here, µ k = k(x, ·)dP (x) is the kernel mean embedding of the distribution P , according to the real-valued symmetric kernel function k(x, x ) = k(x, ·), k(x , ·) H k with associated reproducing kernel Hilbert space H k .

Therefore, µ k summarizes P as the expected value of the features computed by k over samples drawn from P .In practical applications, we do not have access to the distributions P andP , but to their respective sets of samples D andD, defined in Section 3.

In this case, we approximate the kernel mean embedding DISPLAYFORM1 , and respectively forP .

Then, the empirical MMD statistic is DISPLAYFORM2 Importantly, the empirical MMD tends to zero as n → ∞ if and only if P =P , as long as k is a characteristic kernel BID10 .

This property makes the MMD an excellent choice to model how close the observational distribution P is to the estimated observational distribution P .

Throughout this paper, we will employ a particular characteristic kernel: the Gaussian kernel k(x, x ) = exp(−γ x − x 2 2 ), where γ > 0 is a hyperparameter controlling the smoothness of the features.

In terms of computation, the evaluation of MMD k (D,D) takes O(n 2 ) time, which is prohibitive for large n. When using a shift-invariant kernel, such as the Gaussian kernel, one can invoke Bochner's theorem BID30

The literature about learning FCMs from data is vast.

We recommend the books BID35 BID27 BID28 and surveys BID16 BID13 BID5 .

FCM learning methods can be classified into bivariate and multivariate algorithms.

On the one, pairwise algorithms aim at orienting the cause-effect relation between two random variables (X, Y ) by searching for asymmetries in the distribution P (X, Y ).

The Additive Noise Model, or ANM BID15 , assumes an FCM with form Y ← f (X) + E, where the cause X is statistically independent from the noise E. Following these assumptions, the ANM performs one nonlinear regression in each direction, and prefers the one that produces residuals statistically independent from the alleged cause.

The Post Non-Linear (PNL) model BID40 extends the ANM by allowing FCMs with form Y ← g(f (X) + E), where g is a monotone function.

The IGCI method BID4 prefers the causal direction producing a cause distribution independent from the derivative of the causal mechanism.

The LiNGAM method BID32 leverages independent component analysis to orient linear cause-effect relations.

The CURE

@highlight

Discover the structure of functional causal models with generative neural networks