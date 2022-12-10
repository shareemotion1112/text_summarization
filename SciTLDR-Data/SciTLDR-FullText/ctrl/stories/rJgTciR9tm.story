Extracting relevant information, causally inferring and predicting the future states with high accuracy is a crucial task for modeling complex systems.

The endeavor to address these tasks is made even more challenging when we have to deal with high-dimensional heterogeneous data streams.

Such data streams often have higher-order inter-dependencies across spatial and temporal dimensions.

We propose to perform a soft-clustering of the data and learn its dynamics to produce a compact dynamical model while still ensuring the original objectives of causal inference and accurate predictions.

To efficiently and rigorously process the dynamics of soft-clustering, we advocate for an information theory inspired approach that incorporates stochastic calculus and seeks to determine a trade-off between the predictive accuracy and compactness of the mathematical representation.

We cast the model construction as a maximization of the compression of the state variables such that the predictive ability and causal interdependence (relatedness) constraints between the original data streams and the compact model are closely bounded.

We provide theoretical guarantees concerning the convergence of the proposed learning algorithm.

To further test the proposed framework, we consider a high-dimensional Gaussian case study and describe an iterative scheme for updating the new model parameters.

Using numerical experiments, we demonstrate the benefits on compression and prediction accuracy for a class of dynamical systems.

Finally, we apply the proposed algorithm to the real-world dataset of multimodal sentiment intensity and show improvements in prediction with reduced dimensions.

The use of machine learning for making inference and prediction from the real-world data has shown unprecedented growth.

There exist a plethora of approaches for complex system (CS) modeling (e.g., multi-input multi-output state space identification (Stoica & Jansson, 2000) , expectation maximization (EM) BID17 , regularization BID6 , graphical models (Meinshausen & Buhlmann, 2006) , combined regularization and Bayesian learning BID11 BID10 BID3 , kernel-based regularization (Pillonetto & Chiuso, 2015) ).

With the increase in the size of data, the complexity of the accurate models also increases, making inference and predictions slower.

The major challenges of the upcoming era, hence, are likely to deal with the massive and diverse data sources, and still making quick decisions.

Therefore, the compact modeling of time-varying complex systems 1 is a challenging task and appealing for more investigation.

The real-world data has complex inter-dependencies across spatial and temporal dimensions.

We aim to identify such dependencies and carefully construct a compact representation of the given CS model while still ensuring accurate predictions.

We do so by performing a soft-clustering of such inter-dependencies to preserve only the relevant information.

For the CS model in the form of a dynamical system, we additionally argue that similar to the data, the most relevant information also gets transformed at each hop in an alternate dynamical system.

From a bird's-eye view, we track how the most relevant information propagates across the given dynamical system.

We represent this propagation via an alternate dynamical system (compact model) and develop an unsupervised learning technique of such process.

The most relevant work in this regard is information bottleneck (IB) principle (Tishby et al., 2000) .

For fixed two random variables, it performs a soft-clustering to compress one variable while predicting another, given the joint probability distribution.

The IB has been successfully applied to speech recognition (Hecht & Tishby, 2005) , document classification (Slonim & Tishby, 2000) , gene expression BID12 ) and deep learning (Tishby & Zaslavsky, 2015) , etc., and it has shown good performance.

In contrast, we aim to learn a dynamics of the soft-clustering across the given dynamical system, and propose a general optimization framework to study the trade-offs between compactness and the resulting accuracies.

The problem statement addressed in this work is: Given a dynamical system, we aim to develop a compact model by learning the dynamics of the soft-clustering in an unsupervised manner, or alternate dynamical process, through Information Bottleneck hierarchy (IBH).

The main contributions of the present work are as follows: (i) By learning the dynamics of the soft-clustering, we propose an alternate compact dynamical system of the given process, with emphasis on the prediction accuracies. (ii) We formulate a novel optimization setup, compact perception problem, and characterize general solution to the information theoretic problem. (iii) We quantify how most relevant information about future gets transformed at each hop in the alternatively designed dynamical system.

A brief mention of the mathematical notations is provided in the next part.

In this manuscript, we use capital letters to denote random variables (RVs), and lowercase letters are used for the realizations.

The bold letters are used for multi-variate RVs.

For a RV X, with little abuse of notation, we denote the probability mass function p X (x) as p(X), unless specified otherwise.

The expectation operator is denoted as E [.] .

A Gaussian distributed multi-variate RV is denoted as X ∼ N (µ X , Σ X ), where µ X and Σ X are the mean vector and covariance matrix, respectively.

Next, we present a few information theoretic definitions relevant to this work.

Definition 1.

The Kullback-Leibler (KL) divergence BID7 between two probability mass functions p(·) and q(·) is written as DISPLAYFORM0 The KL divergence is in general not symmetric, and D KL ≥ 0 with D KL (p||q) = 0 if and only if p = q. Using (1), the mutual information between two RVs X and Y is defined as DISPLAYFORM1 .

Next, we state the problem statement addressed in this work.

Complex systems consist of a large number of interacting dynamic components.

In practical situations, we have high-dimensional time-series (node activities) having complex spatial and temporal correlations.

The challenge lies in identifying such complicated inter-dependencies and recover the true dimensionality of the system in an unsupervised manner.

We equivalently call it as identifying the compact model while preserving the relevant information across spatio-temporal states.

Under stationary assumptions, (Tishby et al., 2000) proposed the IB to process information compactly.

For the sake of completeness, we provide a brief overview of the IB principle.

The IB compresses a variable X into a new stochastic variable B via soft-clustering, while maintaining as much information as possible about another variable of interest Y .

The variable B operates to minimize the information compression task and to maximize the relevant information.

Provided that the relevance of one RV to another is measurable by an information-theoretic metric (i.e., mutual information), the trade-off can be written as the following variational problem DISPLAYFORM0 where β controls the trade-off between the tasks mentioned above.

Hence, the variable B, solving the minimization problem (2), encodes the most informative part from the input X about output Y .Inspired by the IB concept and its internal predictive coding, we propose a causal inference framework for discrete-time stochastic dynamical systems.

More precisely, we label two consecutive dynamic states X k and X k+1 to be the input and the output of the bottleneck B k , respectively.

Hence, B k carries the most informative part (relevant information) from the past about the future.

Next, we formalize this idea to create a sequence of bottlenecks (or dynamics of soft-clustering) to compactly and accurately represent the given dynamical system.

We consider a stochastic dynamical system, involving large number n of random processes X = [X 1 , X 2 , ....

X n ], interacting and evolving in time.

A fundamental problem for decision making or prediction is to learn a compressed representation of the system dynamics from high-dimensional data (i.e., system identification).

The higher-order dependencies hiding behind the high-dimensions make this task challenging, and it needs sophisticated techniques to extract meaningful information for an appropriate application.

In this paper, we propose a framework that focuses only on the dynamics of the relevant information, by learning an alternate representation of the given process.

Figure 1 : A N -length stochastic dynamical system with corresponding IBH in parallel.

We study the shaded 3-hop process in isolation, i.e., given three consecutive states X k−1 , X k and X k+1 of the input process, the stochastic variables B k , B k+1 represent dynamics of the alternatively designed process to capture the relevant information.

DISPLAYFORM0 Adopting an information theoretic representation, we aim to determine B k and B k+1 jointly since they provide compressed predictive information about the system dynamics.

Figure 1 summarizes our objectives, and we study the shaded region (3 states) of the dynamical system in isolation, i.e. without the influence of any other RVs in the complete system.

An argument to generalize this study to any N -length is provided later in Section 3.1.

We determine the stochastic variable B k that not only compresses X k−1 as much as possible while preserving the relevant information about X k , but also delivers this information to B k+1 .

The variable B k+1 quantifies the meaningful information about the state of the system at time k + 1, building upon the compressed information received via B k , thus forming a dynamics of the relevant information.

By construction, the new mapping B k is designed from X k−1 to preserve the consistent information about X k , therefore, given X k−1 , B k and X k are independent.

Similarly, the mapping B k+1 is independent of X k+1 given B k .

Since, B k+1 carries information from B k which is compressed representation of X k−1 , then given B k , X k−1 and B k+1 are also independent.

With this framework, the following Markov chains are considered: DISPLAYFORM1 The trade-off between the compression and preservation of relevant information is defined as the minimum achievable rate I(X k−1 ; B k ) subject to constraints on the information processing.

We call it a compact perception problem which determines a trade-off between compression representation and predictive characteristics.

Formally, this can be written as the following optimization.

DISPLAYFORM2 where the constraints characterize the bounds on the desired prediction/compression at each step.

We wish to lower bound (to guarantee) the prediction accuracies via lower bounding I(B k ; X k ) and I(X k+1 ; B k+1 ), and upper bound (to limit) the compression level across hop via I(B k ; B k+1 ).

For example, 1 bounds the accuracy of the prediction of X k by B k .

The information flow across alternate dynamical process is controlled by 2 .

Lastly, 3 defines closeness of prediction of X k+1 by B k+1 .

We show in the Section 3.1, and results in Section 4, that such trade-off can be alternatively studied by choice of the Lagrange parameters.

This section provides the main results concerning solving the compact perception optimization problem in equation (3) under the most general case.

Next, this general result is used to study a highdimensional continuous Gaussian distribution.

Lastly, we describe an iterative method to update the corresponding parameters.

Finding the alternate dynamical representation, or B k and B k+1 , as stated in (3) requires to solve a variational problem.

To solve this problem, we introduce the Lagrange multipliers β, λ and γ for the information processing constraints.

Hence, we find the alternative representation (or IBH) by minimizing the following functional written using (3) : DISPLAYFORM0 (4) We can argue the following regarding the optimal solution which minimizes the equation (4) .

Theorem 1.

The optimal solution that minimizes functional F in (4) satisfy the following selfconsistent equations: DISPLAYFORM1 DISPLAYFORM2 where Z 1 and Z 2 are normalizing partition functions.

The functional F in (4) may not be convex in the product space of the associated probability simplexes.

Hence, it is difficult to obtain a global optimum, however, a stationary point (and locally optimal solution, in most of the cases) can be obtained using the following result.

Corollary 1.

The self-consistent equations in Theorem 1 can be used to write an iterative procedure to update the associated probabilities as equations FORMULA0 - FORMULA2 .The iterative approach of Corollary 1 is detailed in the Appendix B. The idea is similar to the BlahutArimoto algorithm BID0 BID2 , also observed in (Tishby et al., 2000) .

Finally, it remains to show the existence of a stationary point of the functional in (4).

The convergence of the iterations in Corollary 1 is established through the following result.

Lemma 1.

The iterative procedure in Corollary 1 to minimize the functional F in (4) is convergent to a stationary point.

The idea of proof (in Appendix C) is also somewhat similar to the EM BID18 but with minimization of the functional.

As in standard EM, in most of the cases, the stationary convergence point is local minimum (maximum in EM) of the functional.

Corollary 2.

The IBH solution in Theorem 1 reduces to IB in (2) upon setting X k−1 = X k , and β → ∞. Proof.

With X k−1 = X k , the functional of IBH in equation (4) can be written as DISPLAYFORM3 , and the problem reduces to minimize the followinĝ DISPLAYFORM4 where λ can be dropped, as λ ≥ 0.An advantage of using the optimization framework in (3) is that it can be generalized to any length of the given input dynamical process by properly repeating the second and third constraint.

In this work, we have studied the case of length three of the input process.

However, the same principles can be used to write the solution for any length N of the dynamical system (the complete Figure 1 ), as will be presented in the future work.

Many applications in machine learning involve time-series with multi-dimensional observations where each dimension is very likely to be correlated with others, and the state of observations evolves in time.

Assuming that these observations are corrupted by Gaussian noise, then a linear dynamical system is a promising model for analyzing the provided time-series data.

Thus, the system dynamics can be modeled similarly as the evolution of a stochastic time-invariant linear system.

Finally, if we have Gaussian distributed initial-state of the linear dynamical system with additive Gaussian noise at each time step, then all future states are jointly Gaussian distributed.

In this case, our framework learns the IBH through Gaussian random vectors.

Recall that the states X k of the dynamical system under study are jointly Gaussian, and without loss of generality we assume that they are centered.

As explained previously, we aim to design B k and B k+1 to define an alternate representation which captures the dynamics of the relevant information.

It is shown in the prior works of BID14 BID5 that for the problem setup of IB in which the input and output variables are jointly Gaussian, the optimum solution of the IB Lagrangian obtained by a stochastic transformation is also jointly Gaussian with the bottleneck's input.

Consequently, B k is jointly Gaussian with X k−1 and B k+1 .

Since RVs in consideration are mean centered and X k−1 , B k , B k+1 are jointly Gaussian; the IBH variables can be very well represented as linear transformations of each other.

Additionally, using the MC conditions from Section 2.2 we write the following linear relations.

DISPLAYFORM0 where ξ k and ξ k+1 are centered Gaussian random vectors independent of X k−1 and B k , respectively.

Given the aforementioned settings, the solution of the minimization problem in equation (3) is determined by finding the matrices Φ and ∆, and the covariance matrices Σ ξ k and Σ ξ k+1 .

An iterative procedure using Corollary 1 to update the concerned parameters in FORMULA11 is presented as the following result.

Theorem 2.

Given the parameters β, λ and γ, the Gaussian bottlenecks B k = ΦX k−1 + ξ k and B k+1 = ∆B k + ξ k+1 are obtained by performing the following iterations over the parameters DISPLAYFORM1 where t is the iteration index.

The detailed proof of the Theorem 2 is provided in the Appendix.

In the next section, we show some numerical results generated using synthetic data.

We numerically evaluate the results of Theorem 1 and Theorem 2 using synthetically generated Gaussian distribution.

Specifically, the covariance matrices for the input dynamical process 40, 30, 20) .

The rank variation is presented by fixing one parameter in each scenario: (β = 100, λ, γ) in (3a), (β, γ = 10, λ) in (3b), and (β, γ, λ = 0.01) in (3c).

DISPLAYFORM0 X k−1 -X k -X k+1 are generated numerically for a given size tuple, and we compute the parameters of (7) using (8).

The quantities of interest are I(B k+1 ; X k+1 ) and I(B k+1 ; X k−1 ) which are indicators of prediction information and inverse of compression, respectively.

We compare the prediction/compression behavior of the proposed approach of the alternate design of the dynamical system vs. designing local IB's between each hop.

The local IB's are designed between two consecutive RVs in the dynamical system independently while the IBH is designed jointly.

We show the distinction upon varying the dimensions of the input process (X k−1 , X k , X k+1 ) in Figure 2 .

It is observed that the gap between prediction (for a fixed level of compression) grows with an increase in the input dimensions.

The IBH by design takes into account the entire input dynamical system, and construct an alternate representation which provides better prediction at each step, by appropriate choice of the Lagrange parameters β, λ, γ.

The Lagrange parameters (β, λ, γ) control the trade-off between compression and prediction at each step of the alternatively designed dynamical process.

In the optimization problem (3), β corresponds to the first constraint, and hence plays a deciding role in prediction accuracy of B k .

The λ corresponds to the second constraint, and will control the flow of relevant information across B k .

Finally, λ and γ together tune the accuracy of prediction using B k+1 .

For example, in (7), the information tapping behavior can be visualized by inspecting the ranks of Φ and ∆ matrices upon varying (β, λ, γ).

In Figure 3a , the rank(∆) increases upon increasing γ for each choice of λ.

Since λ appears in front of both prediction and compression expression in (4), it has little effect on the rank(∆) for a fixed γ and β.

Next, in Figure 3b , we observe dynamical effects of the information flow.

By fixing λ, we limit the information acceptance of B k+1 , hence the parameter β can only increase the rank(∆) up to a certain limit by allowing maximum information through B k .

We witness in Figure 3b , that with higher λ, the parameter β quickly increase the rank(∆).

Now, for a fixed λ, both I thought they were average Figure 4 : The alignment of three modalities (text, visual and audio) with averaging of visual and audio features corresponding to the time boundaries obtained from text.β and γ intertwine with each other to decide rank(∆).

By fixing β, we limit the input information through B k to B k+1 , or availability, and hence γ can only increase the rank(∆) up to certain extent.

Similarly, fixing γ limits the maximum information that B k+1 can process, and therefore β can do best narrowly up to some extent.

We witness this hyperbolic behavior in Figure 3c .

In this section, we apply the ideas of IBH to extract the features from the challenging multimodal datasets available in the form of time-series.

Particularly, we have used the CMU Multimodal Sentiment Analysis (CMU-MOSI) dataset (Zadeh et al., 2016) which consists of a total of 2198 videos.

Each video comprises one speaker expressing their opinion in front of the camera.

The available modalities from the dataset are text, visual and audio.

The goal of the dataset is to perform a discriminative task of predicting the speaker sentiment using the available modalities.

From the three modalities of text, visual and audio, the corresponding features are extracted using GloVe word embeddings (Pennington et al., 2014 ), Facet (iMotions, 2017 and COVAREP BID8 , respectively.

The extracted feature size are 300 for text, 74 for audio, and 46 for visual component.

The extracted features in the form of time-series are aligned across modalities as shown in Figure 4 .

Specifically, multiple features of visual and audio modality (due to high frame/sampling rate) are time averaged with boundaries corresponding to the text component.

For each speaker, we have a maximum of 20 words with three modalities and the corresponding sentiment intensity being a real number ranging from −3 to 3, with negative values representing negative sentiments and vice-versa.

Some of the prior work in the multimodal representation learning include Discriminative representation learning (Zadeh et al., 2018a; 2018b; BID4 and Generative representation learning (Sohn et al., 2014; Srivastava & Salakhutdinov, 2014; Suzuki et al., 2016) .

Interestingly, recent work Tsai et al. (2018) proposed Multimodal Factorized Model (MFM) that exploits the fusion of both these techniques.

The work factorizes the data into discriminative factors and modalityspecific generative factors.

We note that, at the very core, the challenge in learning patterns from multiple modalities is to address the complex inter and well as intra dependencies across them.

Since IBH is capable of compressing the given dynamical model into an alternate version stochastically, we propose to, (i) first map the multiple modalities into a time-varying linear dynamical system; and then (ii) use IBH to identify the complex inter and well as intra dependencies across them in a reduced dimensional model for better discrimination using simple machine learning classifiers.

The IBH is applied to various modalities to capture the information flow patterns across them and in the compressed fashion.

We have taken three modalities to be in the following Markov Chain, text-audio-visual.

The intuitive reason behind this assumption is that text is the most informative modality for sentiment and hence the first state, while audio and visual follows text in the Markov chain assuming the speaker is being honest in speaking and making visual expressions for a particular sentiment.

The data is mean centered and the covariances of three modalities are estimated for each speaker in the training as well as the testing dataset.

However, we have only 20 words and hence 20 samples to estimate the covariance matrix of much larger dimensions.

To remedy this fewer samples problem, we have resorted to something called pooling of the covariance matrix as follows.

The covariance matrix of any modality for the ith speaker is written as which is estimated by taking all of the training data.

The parameter α is used to make a trade-off between these two matrices and is usually chosen close to 1.

The covariance matrices are fed to the algorithm in Theorem 2 to estimate Φ and ∆ matrices.

The Φ matrix has components for interactions across text and audio, while ∆ has entries representing interactions with the compressed version of text-audio inter-dependencies as well as with the visual modality.

Therefore, the entries of ∆ matrix are good candidates for representing inter as well as intra dependencies across all three modalities.

Hence, we use ∆ matrix as a feature to predict the sentiment intensity, as this matrix is central to information propagation from the text, through audio, to visual component.

The low rank of ∆ will be key in reducing the number of features.

The values of parameters (β, λ, γ) are chosen such that maximum information flow to first bottleneck, by setting high value for β, and λ close to 1, and low values for γ for better compression.

DISPLAYFORM0 The results are reported for various evaluation methods, namely binary (with the positive and negative sentiment) and 7− class classification, and Mean Average Error (MAE) for regression.

We compare our performance with naive early fusion of features in raw format, most recent results in the multimodal neural networks (Tsai et al., 2018) vs. features processed by IBH, and the numerical results are presented in Table 1 .

Support Vector Machines (SVM) is used in the cases of early fusion and IBH.

The processing of features by IBH has a two-fold advantage: First, due to compression, the dimensionality of the input can be reduced from 8400 to 1150.

Second, the performance with respect to various metrics is better.

In this paper, we have introduced a novel information-theoretic inspired approach to learn the compact dynamics of a time-varying complex system.

The trade-off between the predictive accuracy and the compactness of the mathematical representation is formulated as a multi-hop compact perception optimization problem.

A key ingredient to solve the aforementioned problem is to exploit variational calculus in order to derive the general solution expressions.

Additionally, we have investigated the guaranteed convergence of the proposed iterative algorithm.

Moreover, considering a specific class of distributions (Gaussian), we have provided closed-form expressions for the model parameters' update in our algorithm.

Interestingly, the proposed compact perception shows improvements in prediction with reduced dimension on challenging real-world problems.

The quantification of information flow across a dynamical system can have an enormous impact on understanding and improving the current state-of-the-art in neural networks as realized in (Tishby & Zaslavsky, 2015) .

Moreover, modeling with dynamical systems is a standard approach, and by using the proposed framework, we can make a better compact representation of the system.

The driving force of a dynamical system can enforce different behaviors of information flow, as realized in defining dynamical entropy by (Sinai, 1959) .

Therefore, measuring the information flow can help in estimating/differentiating the actual driving component behind the observed activities.

Such concepts are useful in predicting brain imagined tasks from observed electroencephalogram activities.

The appendix is arranged as follows: In the Section A, we provide the proof of Theorem 1.

In the Section B, we provide the iterative procedure (mentioned as Corollary 1) to minimize the functional in (4).

Next, in Section C, we present the detailed proof of the Lemma 1, and finally, in Section D, a detailed proof of Theorem 2 is presented.

A PROOF OF THEOREM 1Proof.

For the sake of simplicity, a sketch of the proof is given for discrete variables.

The Lagrangian associated with the minimization problem is the following DISPLAYFORM0 where α 1 (X k−1 ) and α 2 (B k ) are Lagrange multipliers for the normalization of the distributions p(B k |X k−1 ) and p(B k+1 |B k ), respectively.

Taking the derivative of each term of the Lagrangian L with respect to p(B k |X k−1 ), we have DISPLAYFORM1 Setting the derivative of the Lagrangian equal to zero and arranging the terms we obtain the self consistent equation FORMULA7 .

Note that all the constant terms in the derivative independent of B k will be captured by the Lagrange multiplier α 1 (X k−1 ).

The derivative of the Lagrangian L with respect to p(B k+1 |B k ) involves only the two last terms from the functional F and the term that ensures the normalization condition.

Then, we have DISPLAYFORM2 DISPLAYFORM3 Thus, the variational condition is written as follows DISPLAYFORM4 where α 2 (B k ) is the summation of the Lagrange multiplier α 2 (B k ) and the terms independent of B k+1 , and hence the equation FORMULA8 follows.

The self-consistent equations derived in Theorem 1 to minimize the functional F in (4) can be used to write the following set of iterative equations.

DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 where in equations FORMULA0 - FORMULA0 , FORMULA0 and FORMULA2 we have used the Markov assumption as stated in Section 2.2.C PROOF OF LEMMA 1Proof.

The proof of the Lemma 1 can be divided into two parts.

First, we show that the F in (4) is lower-bounded.

Next, we show that each iteration using the equations FORMULA0 - FORMULA2 monotonically decrease the functional.

Lower bound: Let us consider the following alternate functional.

DISPLAYFORM7 It can be readily verified that the functionalF ≥ 0 for given non-negative constants β, λ and γ.

Also, theF in (21) can be expanded as DISPLAYFORM8 Since the functionalF in the equation FORMULA2 differs from the functional F in (4) only in constants, therefore, F is lower bounded as well.

Monotonicity: For proving monotonic decrement of the functional F, we will use the formulation similar to BID0 .

First, let us consider the following observation made in (Tishby et al., 2000) .For a given joint distribution p(X, Y ), we can write the following.

DISPLAYFORM9 where the minimization is performed over the probability simplex of φ(Y ) such that the joint distribution is p(X, Y ).

The functionalF in (22) can be written in its most general form as DISPLAYFORM10 where DISPLAYFORM11 therefore,F reduces to the form in (22) upon setting φ 1 = p(X k |B k ) and φ 2 = p(X k+1 |B k+1 ).With the objective of minimizing the functionalF, we can write its value at iteration t asF (t) = F(p1 , p2 , φ1 , φ2 ).

The iterations to minimizeF will involve the successive choice of tuple (p 1 , p 2 , φ 1 , φ 2 ).

At iteration t, let us assume that we have chosen p Using equation FORMULA2 , it easily follows that φ (t) 1 = p (t) (X k |B k ) and φ (t) 2 = p (t) (X k+1 |B k+1 ), and hence G(t, t) =F(t).

Now, for fixed p2 , φ1 , φ2 , it can be easily realized that theF is convex in p 1 , therefore, minimizingF with respect to (w.r.t.)

p 1 will involve writing Lagrangian, and then differentiation, and setting to zero.

This step is similar to the Theorem 1, and we can write that DISPLAYFORM12 where the resulting solution is (5), and hence p (t+1) 1will have the expression as in FORMULA0 .

Similarly, for fixed p DISPLAYFORM13 2 , theF is convex in p 2 , and the same steps follow to obtain p (t+1) 2 as written in (18).

It should be noted that the choice to perform minimization w.r.t.

p 1 before p 2 is arbitrary, and the reverse can also be performed.

This will change the order of iteration index in equations FORMULA0 - FORMULA2 accordingly.

Using equations FORMULA2 and FORMULA2 , we can conclude that F (t+1) = G(t + 1, t + 1) ≤ G(t + 1, t) ≤ G(t, t) =F (t) , and therefore, iterating equations FORMULA0 - FORMULA2 written using the self-consistent equations of Theorem 1 minimizesF (t) monotonically.

SinceF and F differs only in constant, it reduces F as well monotonically from above.

Proof.

For a multivariate random variable X, X ∈ R n with Gaussian distribution, i.e. X ∼ N (µ, Σ), the entropy can be written as DISPLAYFORM0 where c is constant for a given dimension n. Using equation FORMULA2 , the KL-divergence between two Gaussian distributed random variables, X 1 ∼ N (µ 1 , Σ 1 ) and X 2 ∼ N (µ 2 , Σ 2 ) of the same dimensions, is written as We have assumed that the given data is centered, hence all considered random variables will have zero mean, i.e., for characterizing each random variable, we only need the corresponding covariance matrix.

Let us revisit the linear transformation model for the IBH.B k = ΦX k−1 + ξ k , B k+1 = ∆B k + ξ k+1 .

Now, to completely specifying the model, we have to determine the constant matrices Φ, ∆ and the covariances of ξ k , ξ k+1 .

Since entropy is well defined for Gaussian distribution, as in (27), due to their nice tail distribution, we can use Theorem 1 and equation FORMULA2 to write the self-consistent

<|TLDR|>

@highlight

Compact perception of dynamical process

@highlight

Studies the problem of compactly representing the model of a complex dynamic system while preserving information by using an information bottleneck method.

@highlight

This paper studied the Gaussian linear dynamic and proposed an algorithm for computing the Information Bottleneck Hierarchy (IBH).