Decentralized training of deep learning models is a key element for enabling data privacy and on-device learning over networks, as well as for efficient scaling to large compute clusters.

As current approaches are limited by network bandwidth, we propose the use of communication compression in the decentralized training context.

We show that Choco-SGD achieves linear speedup in the number of workers for arbitrary high compression ratios on general non-convex functions, and non-IID training data.

We demonstrate the practical performance of the algorithm in two key scenarios: the training of deep learning models (i) over decentralized user devices, connected by a peer-to-peer network and (ii) in a datacenter.

Distributed machine learning-i.e.

the training of machine learning models using distributed optimization algorithms-has enabled many recent successful applications in research and industry.

Such methods offer two of the key success factors: 1) computational scalability by leveraging the simultaneous computational power of many devices, and 2) data-locality, the ability to perform joint training while keeping each part of the training data local to each participating device.

Recent theoretical results indicate that decentralized schemes can be as efficient as the centralized approaches, at least when considering convergence of training loss vs. iterations (Scaman et al., 2017; Lian et al., 2017; Tang et al., 2018; Koloskova et al., 2019; Assran et al., 2019) .

Gradient compression techniques have been proposed for the standard distributed training case (Alistarh et al., 2017; Wen et al., 2017; Lin et al., 2018b; Wangni et al., 2018; Stich et al., 2018) , to reduce the amount of data that has to be sent over each communication link in the network.

For decentralized training of deep neural networks, Tang et al. (2018) introduce two algorithms (DCD, ECD) which allow for communication compression.

However, both these algorithms are restrictive with respect to the used compression operators, only allowing for unbiased compressors and-more significantlyso far not supporting arbitrarily high compression ratios.

We here study CHOCO-SGD-recently introduced for convex problems only (Koloskova et al., 2019 )-which overcomes these constraints.

For the evaluation of our algorithm we in particular focus on the generalization performance (on the test-set) on standard machine learning benchmarks, hereby departing from previous work such as e.g. (Tang et al., 2018; Wang et al., 2019; Tang et al., 2019b; Reisizadeh et al., 2019 ) that mostly considered training performance (on the train-set).

We study two different scenarios: firstly, (i) training on a challenging peer-to-peer setting, where the training data is distributed over the training devices (and not allowed to move), similar to the federated learning setting (McMahan et al., 2017) .

We are again able to show speed-ups for CHOCO-SGD over the decentralized baseline (Lian et al., 2017) with much less communication overhead.

Secondly, (ii) training in a datacenter setting, where decentralized communication patterns allow better scalability than centralized approaches.

For this setting we show that communication efficient CHOCO-SGD can improve time-to-accuracy on large tasks, such as e.g. ImageNet training.

However, when investigating the scaling of decentralized algorithms to larger number of nodes we observe that (all) decentralized schemes encounter difficulties and often do not reach the same (test and train) performance as centralized schemes.

As these findings do point out some deficiencies of current decentralized training schemes (and are not particular to our scheme) we think that reporting these results is a helpful contribution to the community to spur further research on decentralized training schemes that scale to large number of peers.

??? On the theory side, we are the first to show that CHOCO-SGD converges at rate O 1 / ??? nT + n /(?? 4 ?? 2 T ) on non-convex smooth functions, where n denotes the number of nodes, T the number of iterations, ?? the spectral gap of the mixing matrix and ?? the compression ratio.

The main term, O 1 / ??? nT , matches with the centralized baselines with exact communication and shows a linear speedup in the number of workers n. Both ?? and ?? only affect the asymptotically smaller second term.

??? On the practical side, we present a version of CHOCO-SGD with momentum and analyze its practical performance on two relevant scenarios:

??? for on-device training over a realistic peer-to-peer social network, where lowering the bandwidth requirements of joint training is especially impactful ??? in a datacenter setting for computational scalability of training deep learning models for resource efficiency and improved time-to-accuracy ??? Lastly, we systematically investigate performance of the decentralized schemes when scaling to larger number of nodes and we point out some (shared) difficulties encountered by current decentralized learning approaches.

For the training in communication restricted settings a variety of methods have been proposed.

For instance, decentralized schemes (Lian et al., 2017; Nedi?? et al., 2018; Koloskova et al., 2019) , gradient compression (Seide et al., 2014; Strom, 2015; Alistarh et al., 2017; Wen et al., 2017; Lin et al., 2018b; Wangni et al., 2018; Bernstein et al., 2018; Lin et al., 2018b; Alistarh et al., 2018; Stich et al., 2018; , asynchronous methods (Recht et al., 2011; Assran et al., 2019) or performing multiple local SGD steps before averaging McMahan et al., 2017; Lin et al., 2018a) .

This especially covers learning over decentralized data, as extensively studied in the federated Learning literature for the centralized algorithms (McMahan et al., 2016) .

In this paper we advocate for combining decentralized SGD schemes with gradient compression.

Decentralized SGD.

We in particular focus on approaches based on gossip averaging (Kempe et al., 2003; Xiao & Boyd, 2004; Boyd et al., 2006) whose convergence rate typically depends on the spectral gap ?? ??? 0 of the mixing matrix (Xiao & Boyd, 2004) .

Lian et al. (2017) combine SGD with gossip averaging and show convergence at the rate O 1 / ??? nT + n /(?? 2 T ) .

The leading term in the rate, O 1 / ??? nT , is consistent with the convergence of the centralized mini-batch SGD (Dekel et al., 2012) and the spectral gap only affects the asymptotically smaller terms.

Similar results have been observed very recently for related schemes (Scaman et al., 2017; Koloskova et al., 2019; Yu et al., 2019) .

Quantization.

Communication compression with quantization has been popularized in the deep learning community by the reported successes in (Seide et al., 2014; Strom, 2015) .

Theoretical guarantees were first established for schemes with unbiased compression (Alistarh et al., 2017; Wen et al., 2017; Wangni et al., 2018) but soon extended to biased compression (Bernstein et al., 2018) as well.

Schemes with error correction work often best in practice and give the best theoretical gurantees (Lin et al., 2018b; Alistarh et al., 2018; Stich et al., 2018; .

Recently, also proximal updates and variance reduction have been studied in combination with quantized updates Horv??th et al., 2019) .

Decentralized Optimization with Quantization.

It has been observed that gossip averaging can diverge (or not converge to the correct solution) in the presence of quantization noise (Xiao et al., 2005; Carli et al., 2007; Nedi?? et al., 2008; Dimakis et al., 2010; Carli et al., 2010b; Yuan et al., 2012) .

Reisizadeh et al. (2018) propose an algorithm that can still converge, though at a slower rate than the exact scheme.

Another line of work proposed adaptive schemes (with increasing compression accuracy) that converge at the expense of higher communication cost (Carli et al., 2010a; Doan et al., 2018; Berahas et al., 2019) .

For deep learning applications, Tang et al. (2018) proposed the DCD and ECD algorithms that converge at the same rate as the centralized baseline though only for constant compression ratio.

The CHOCO-SGD algorithm that we consider in this work can deal with arbitrary high compression, and has been introduced in (Koloskova et al., 2019) but only been analyzed for convex functions.

For non-convex functions we show a rate of

, where ?? > 0 measures the compression quality.

Simultaneous work of Tang et al. (2019a) introduced DeepSqueeze, an alternative method which also converges with arbitrary compression ratio.

In our experiments, under the same amount of tuning, CHOCO-SGD achieves higher test accuracy.

Algorithm 1 CHOCO-SGD (Koloskova et al., 2019) input:

, E) and mixing matrix W , initializex

1: for t in 0 . . .

T ??? 1 do {in parallel for all workers i ??? [n]} 2:

for neighbors j : {i, j} ??? E (including {i} ??? E) do 5:

end for 8:

9: In this section we formally introduce the decentralized optimization problem, compression operators, and the gossip-based stochastic optimization algorithm CHOCO-SGD from (Koloskova et al., 2019) .

Distributed Setup.

We consider optimization problems distributed across n nodes of the form

where D 1 , . . .

D n are local distributions for sampling data which can be different on every node, Communication.

Every device is only allowed to communicate with its local neighbours defined by the network topology, given as a weighted graph G = ([n], E), with edges E representing the communication links along which messages (e.g. model updates) can be exchanged.

We assign a positive weight w ij to every edge (w ij = 0 for disconnected nodes {i, j} / ??? E).

Assumption 1 (Mixing matrix).

We assume that

In our experiments we set the weights based on the local node degrees: w ij = max{deg(i), deg(j)} ???1 for {i, j} ??? E.

This will not only guarantee ?? > 0 but these weights can easily be computed in a local fashion on each node (Xiao & Boyd, 2004) .

Compression.

We aim to only transmit compressed (e.g. quantized or sparsified) messages.

We formalized this through the notion of compression operators that was e.g. also used in (Tang et al., 2018; Stich et al., 2018) .

for a parameter ?? > 0.

Here E Q denotes the expectation over the internal randomness of operator Q.

In contrast to the quantization operators used in e.g. (Alistarh et al., 2017; Horv??th et al., 2019) , compression operators defined as in (2) are not required to be unbiased and therefore supports a larger class of compression operators.

Some examples can be found in (Koloskova et al., 2019) and we further discuss specific compression schemes in Section 5.

Algorithm.

CHOCO-SGD is summarized in Algorithm 1.

Every worker i stores its own private variable x i ??? R d that is updated by a stochastic gradient step in part 2 and a modified gossip averaging step on line 2.

This step is a key element of the algorithm as it preserves the averages of the iterates even in presence of quantization noise (the compression errors are not discarded, but aggregated in the local variables x i , see also (Koloskova et al., 2019) ).

The nodes communicate with their neighbors in part 1 and update the variablesx j ??? R d for all their neighbors {i, j} ??? E only using compressed updates.

Thesex i are available to all the neighbours of the node i and represent the 'publicly available' copies of the private x i , in general x i =x i , due to the communication restrictions.

From an implementation aspect, it is worth highlighting that the communication part 1 and the gradient computation part 2 can both be executed in parallel because they are independent.

Moreover, each node only needs to store 3 vectors at most, independent of the number of neighbors (this might not be obvious from the notation used here for additinal clarity, for further details c.f. (Koloskova et al., 2019) ).

We further propose a momentum-version of CHOCO-SGD in Algorithm 2 (see also Section D for further details).

As the first main contribution, we here extend the analysis of CHOCO-SGD to non-convex problems.

For this we make the following technical assumptions:

and the variance of the stochastic gradients is bounded on each worker:

where 2 , the averaged iterates

i of Algorithm 1 satisfy:

where c := ?? 2 ?? 82 denotes the convergence rate of the underlying consensus averaging scheme of (Koloskova et al., 2019) .

This result shows that CHOCO-SGD converges asymptotically as

The first term shows a linear speed-up compared to SGD on a single node, while compression and graph topology affect only the higher order second term.

For slightly more general statements than Theorem 4.1 (with improved constants) as well as for the proofs and convergence of the individual iterates x i we refer to Appendix A.

In this section we experimentally compare CHOCO-SGD to the relevant baselines for a selection of commonly used compression operators.

For the experiments we further leverage momentum in all implemented algorithms.

The newly developed momentum version of CHOCO-SGD is given as Algorithm 2.

Algorithm 2 CHOCO-SGD with Momentum input: Same as for Algorithm 1, additionally: weight decay factor ??, momentum factor ??, local momentum memory v

i := 0 ???i ??? [n] Lines 1-8 in Algorithm 1 are left unmodified Line 9 in Algorithm 1 is replaced with the following two lines

local momentum with weight decay 10:

Setup.

In order to match the setting in (Tang et al., 2018) for our first set of experiments, we use a ring topology with n = 8 nodes and train the ResNet20 architecture (He et al., 2016) on the Cifar10 dataset (50K/10K training/test samples) (Krizhevsky, 2012) .

We randomly split the training data between workers and shuffle it after every epoch, following standard procedure as e.g. in (Goyal et al., 2017) .

We implement DCD and ECD with momentum (Tang et al., 2018) , DeepSqueeze with momentum (Tang et al., 2019a) , CHOCO-SGD with momentum (Algorithm 2) and standard (all-reduce) mini-batch SGD with momentum and without compression (Dekel et al., 2012) .

The momentum factor is set to 0.9 without dampening.

For all algorithms we fine-tune the initial learning rate and gradually warm it up from a relative small value (0.1) (Goyal et al., 2017) for the first 5 epochs.

The learning rate is decayed by 10 twice, at 150 and 225 epochs, and stop training at 300 epochs.

For CHOCO-SGD and DeepSqueeze the consensus learning rate ?? is also tuned.

The detailed hyper-parameter tuning procedure refers to Appendix F. Every compression scheme is applied to every layer of ResNet20 separately.

We evaluate the top-1 test accuracy on every node separately over the whole dataset and report the average performance over all nodes.

Compression Schemes.

We implement two unbiased compression schemes: (i) gsgd b quantization that randomly rounds the weights to b-bit representations (Alistarh et al., 2017) , and (ii) random a sparsification, which preserves a randomly chosen a fraction of the weights and sets the other ones to zero (Wangni et al., 2018) .

Further two biased compression schemes: (iii) top a , which selects the a fraction of weights with the largest magnitude and sets the other ones to zero (Alistarh et al., 2018; Stich et al., 2018) , and (iv) sign compression, which compresses each weight to its sign scaled by the norm of the full vector (Bernstein et al., 2018; .

We refer to Appendix C for exact definitions of the schemes.

DCD and ECD have been analyzed only for unbiased quantization schemes, thus the combination with the two biased schemes is not supported by theory.

In converse, CHOCO-SGD and DeepSqueeze has been studied only for biased schemes according to Definition 2.

However, both unbiased compression schemes can be scaled down in order to meet the specification (cf.

discussions in (Stich et al., 2018; Koloskova et al., 2019) ) and we adopt this for the experiments.

Results.

The results are summarized in Table 1 .

For unbiased compression schemes, ECD and DCD only achieve good performance when the compression ratio is small, and sometimes even diverge when the compression ratio is high.

This is consistent 1 with the theoretical and experimental results in (Tang et al., 2018) .

We further observe that the performance of DCD with the biased top a sparsification is much better than with the unbiased random a counterpart, though this operator is not yet supported by theory.

CHOCO-SGD can generalize reasonably well in all scenarios (at most 1.65% accuracy drop) for fixed training budget.

The sign compression achieves state-of-the-art accuracy and requires approximately 32?? less bits per weight than the full precision baseline.

We now shift our focus to challenging real-world scenarios which are intrinsically decentralized, i.e. each part of the training data remains local to each device, and thus centralized methods either fail or are inefficient to implement.

Typical scenarios comprise e.g. sensor networks, or mobile devices or hospitals which jointly train a machine learning model.

Common to these applications is that i) each device has only access to locally stored or acquired data, ii) communication bandwidth is limited (either physically, or artificially for e.g. metered connections), iii) the global network topology is typically unknown to a single device, and iv) the number of connected devices is typically large.

Additionally, this fully decentralized setting is also strongly motivated by privacy aspects, enabling to keep the training data private on each device at all times.

Modeling.

To simulate this scenario, we permanently split the training data between the nodes, i.e. the data is never shuffled between workers during training, and every node has distinct part of the dataset.

To the best of our knowledge, no prior works studied this scenario for decentralized Figure 1: Scaling of CHOCO-SGD with sign compression to large number of devices on Cifar10 dataset.

Left: best testing accuracy of the algorithms reached after 300 epochs.

Right: best testing accuracy reached after communicating 1000 MB.

deep learning.

For the centralized approach, gathering methods such as all-reduce are not efficiently implementable in this setting, hence we compare to the centralized baseline where all nodes route their updates to a central coordinator for aggregation.

For the comparison we consider CHOCO-SGD with sign compression (this combination achieved the compromise between accuracy and compression level in Table 1 )), decentralized SGD without compression, and centralized SGD without compression.

Scaling to Large Number of Nodes.

To study the scaling properties of CHOCO-SGD, we train on 4, 16, 36 and 64 number of nodes.

We compare decentralized algorithms on two different topologies: ring as the worst possible topology, and on the torus with much larger spectral gap.

Their parameters are listed in the table 2. (Krizhevsky, 2012) .

For the simplicity, we keep the learning rate constant and separately tune it for all methods.

We tune consensus learning rate for CHOCO-SGD.

The results are summarized in Figure 1 .

First we compare the testing accuracy reached after 300 epochs (Fig. 1, Left) .

CentralizedSGD has a good performance for all the considered number of nodes.

CHOCO-SGD slows down due to the influence of graph topology (Decentralized curve), which is consistent with the spectral gaps order (see Tab.

2), and also influenced by the communication compression (CHOCO curve), which slows down training uniformly for both topologies.

We observed that the train performance is similar to the test on Fig. 1 , therefore the performance degradation is explained by the slower convergence (Theorem 4.1) and is not a generalization issue.

Increasing the number of epochs improves the performance of the decentralized schemes.

However, even using 10 times more epochs, we were not able to perfectly close the gap between centralized and decentralized algorithms for both train and test performance.

In the real decentralized scenario, the interest is not to minimize the epochs number, but the amount of communication to reduce the cost of the user's mobile data.

We therefore fix the number of transmitted bits to 1000 MB and compare the best testing accuracy reached (Fig. 1, Right) .

CHOCO-SGD performs the best while having slight degradation due to increasing number of nodes.

It is beneficial to use torus topology when the number of nodes is large because it has good mixing properties, for small networks there is not much difference between these two topologies-the benefit of large spectral gap is canceled by the increased communication due larger node degree for torus topology.

Both Decentralized and Centralized SGD requires significantly larger number of bits to reach reasonable accuracy.

Experiments on a Real Social Network Graph.

We simulate training models on user devices (e.g. mobile phones), connected by a real social network.

We chosen Davis Southern women social network (Davis et al., 1941) with 32 nodes.

We train ResNet20 (0.27 million parameters) model on the Cifar10 dataset (50K/10K training/test samples) (Krizhevsky, 2012) for image classification and a three-layer LSTM architecture (Hochreiter & Schmidhuber, 1997 ) (28.95 million parameters) for a language modeling task on WikiText-2 (600 training and 60 validation articles with a total of 2 088 628 and 217 646 tokens respectively) (Merity et al., 2016) .

We use exponentially decaying learning rate schedule.

For more detailed experimental setup we refer to Appendix F.

The results are summarized in Figures 2-3 and in Table 3 .

For the image classification task, when comparing the training accuracy reached after the same number of epochs, we observe that the decentralized algorithm performs best, follows by the centralized and lastly the quantized decentralized.

However, the test accuracy is highest for the centralized scheme.

When comparing the test accuracy reached for the same transmitted data 2 , CHOCO-SGD significantly outperforms the exact decentralized scheme, with the centralized performing worst.

We note a slight accuracy drop, i.e. after the same number of epochs (but much less transmitted data), CHOCO-SGD does not reach the same level of test accuracy than the baselines.

For the language modeling task, both decentralized schemes suffer a drop in the training loss when the evaluation reaching the epoch budget; while our CHOCO-SGD outperforms the centralized SGD in test perplexity.

When considering perplexity for a fixed data volume (middle and right subfigure of Figure 3 ), CHOCO-SGD performs best, followed by the exact decentralized and centralized algorithms.

Figure 4 : Large-scale training: Resnet-50 on ImageNet-1k in the datacenter setting.

The topology has 8 nodes (each accesses 4 GPUs).

We use "Sign+Norm" as the quantization scheme of CHOCO-SGD.

The benefits of CHOCO-SGD can be further pronounced when scaling to more nodes.

Decentralized optimization methods offer a way to address scaling issues even for well connected devices, such as e.g. in datacenter with fast InfiniBand (100Gbps) or Ethernet (10Gbps) connections.

Lian et al. (2017) describe scenarios when decentralized schemes can outperform centralized ones, and recently, Assran et al. (2019) presented impressive speedups for training on 256 GPUs, for the setting when all nodes can access all training data.

The main differences of their algorithm to CHOCO-SGD are the asynchronous gossip updates, time-varying communication topology and most importantly exact communication, making their setup not directly comparable to ours.

We note that these properties of asynchronous communication and changing topology for faster mixing are orthogonal to our contribution, and offer promise to be combined.

Setup.

We train ImageNet-1k (1.28M/50K training/validation) (Deng et al., 2009 ) with Resnet-50 (He et al., 2016) .

We perform our experiments on 8 machines (n1-standard-32 from Google Cloud), where each of machines has 4 Tesla P100 GPUs.

Within one machine communication is fast and we perform all-reduce with the full model.

Between different machines we use decentralized communication with compressed communication (sign-CHOCO-SGD) in a ring topology.

The mini-batch size on each GPU is 128, and we follow the general SGD training scheme in (Goyal et al., 2017) and directly use all their hyperparameters for CHOCO-SGD.

Due to the limitation of the computational resource, we did not heavily tune the consensus stepsize for CHOCO-SGD 3 .

Results.

We depict the training loss and top-1 test accuracy in terms of epochs and time in Figure 4 .

CHOCO-SGD benefits from its decentralized and parallel structure and takes less time than all-reduce to perform the same number of epochs, while having only a slight 1.5% accuracy loss. (All-reduce with full precision gradients achieved test accuracy of 76.37%, vs. 75.15% for CHOCO-SGD).

In terms of time per epoch, our speedup does not match that of (Assran et al., 2019) , as the used hardware is very different.

Their scheme is orthogonal to our approach and could be integrated for better training efficiency.

Nevertheless, we still demonstrate a time-wise 20% gain over the common all-reduce baseline, on our used commodity hardware cluster.

We propose the use of CHOCO-SGD (and its momentum version) for enabling decentralized deep learning training in bandwidth-constrained environments.

We provide theoretical convergence guarantees for the non-convex setting and show that the algorithm enjoys the a linear speedup in the number of nodes.

We empirically study the performance of the algorithm in a variety of settings on image classification (ImageNet-1k, Cifar10) and on a language modeling task (WikiText-2).

Whilst previous work successfully demonstrated that decentralized methods can be a competitive alternative to centralized training schemes when no communication constraints are present (Lian et al., 2017; Assran et al., 2019) , our main contribution is to enable training in strongly communication-restricted environments, and while respecting the challenging constraint of locality of the training data.

We theoretically and practically demonstrate the performance of decentralized schemes for arbitrary high communication compression, and under data-locality, and thus significantly expand the reach of potential applications of fully decentralized deep learning.

In this section we present the proof of Theorem 4.1.

For this, we will first derive a slightly more general statement: in Theorem A.2 we analyze CHOCO-SGD for arbitrary stepsizes ??, and then derive Theorem 4.1 as a special case.

The structure of the proof follows Koloskova et al. (2019) .

That is, we first show that Algorithm 1 is a special case of a more general class of algorithms (given in Algorithm 3): Observe that Algorithm 1 consists of two main components: 2 the stochastic gradient update, performed locally on each node, and 1 the (quantized) averaging among the nodes.

We can show convergence of all algorithms of this type-i.e.

stochastic gradient updates 2 followed by an arbitrary averaging step 1 -as long as the averaging scheme exhibits linear convergence.

For the specific averaging used in CHOCO-SGD, linear convergence has been shown in (Koloskova et al., 2019 ) and we will use their estimate of the convergence rate of the averaging scheme.

For convenience, we use the following matrix notation in this subsection.

Decentralized SGD with arbitrary averaging is given in Algorithm 3.

Algorithm 3 DECENTRALIZED SGD WITH ARBITRARY AVERAGING SCHEME input:

blackbox averaging/gossip 4: end for 2 { 1 { Assumption 3.

For an averaging scheme h :

Assume that h preserves the average of iterates:

and that it converges with linear rate for a parameter 0 < c ??? 1

and Laypunov function ??(X,

, where E h denotes the expectation over internal randomness of averaging scheme h.

Example: Exact Averaging.

Setting X + = XW and Y + = X + gives an exact consensus averaging algorithm with mixing matrix W (Xiao & Boyd, 2004) .

It converges at the rate c = ??, where ?? is an eigengap of mixing matrix W , defined in Assumption 1.

Substituting it into the Algorithm 3 we recover D-PSGD algorithm, analyzed in Lian et al. (2017) .

CHOCO-SGD.

To recover CHOCO-SGD, we need to choose CHOCO-GOSSIP (Koloskova et al., 2019) as consensus averaging scheme, which is defined as 82 in the more general results below.

It is important to note that for Algorithm 1 given in the main text, the order of the communication part 1 and the gradient computation part 2 is exchanged.

We did this to better illustrate that both these parts are independent and that they can be executed in parallel.

The effect of this change can be captured by changing the initial values but does not affect the convergence rate.

A.2 PROOF OF THEOREM 4.1 Lemma A.1.

Under Assumptions 1-2 the iterates of the Algorithm 3 with constant stepsize ?? satisfy

Proof of Lemma A.1.

These are exactly the same calculations as the first 9 lines in the proof of Lemma 21 from Koloskova et al. (2019) .

We got a recursion

Verifying that r t ??? ?? 2 4A c 2 satisfy recursion completes the proof as E X

Theorem A.2.

Under Assumptions 1-3 with constant stepsize ?? = n T +1 for T ??? 64nL 2 , the averaged iterates

i of Algorithm 3 satisfy:

where c denotes convergence rate of underlying averaging scheme.

The first term shows a linear speed up compared to SGD on one node, whereas the underlying averaging scheme affects only the second-order term.

Substituting the convergence rate for exact averaging with W gives the rate O( 1 / ??? nT + n /(T ?? 2 )), which recovers the rate of D-PSGD (Lian et al., 2017) .

CHOCO-SGD with the underlying CHOCO-GOSSIP averaging scheme converges at the rate

2 )).

The dependence on ?? (eigengap of the mixing matrix W ) is worse than in the exact case.

This might either just be an artifact of our proof technique or a consequence of supporting arbitrary high compression.

Proof of Theorem A.2.

By L-smoothness

To estimate the second term, we add and subtract ???f (

For the last term, we add and subtract ???f (x (t) ) and the sum of ???f j (x

Combining this together and using L-smoothness to estimate f (

Using Lemma A.1 to bound the third term and using that ?? ???

Rearranging terms and averaging over t

Substituting ?? = n T +1 and using that T ??? 64nL 2 we get the statement of the theorem.

The theorem gives guarantees for the averaged vector of parameters x, however in a decentralized setting it is very expensive and sometimes impossible to average all the parameters distributed across several machines, especially when the number of machines and the model size is large.

We can get similar guarantees on the individual iterates x i as e.g. in (Assran et al., 2019) .

We summarize these briefly below.

Corollary A.3 (Convergence of local weights).

Under the same setting as in Theorem 4.1,

Proof of Corollary A.3.

where we used L-smoothness of f .

Using Theorem 4.1 and Lemma A.1

The previous result holds only for T larger than 64nL 2 .

This is not necessary and can be relaxed.

Theorem A.4.

Under Assumptions 2, 1 with constant stepsize ?? and the consensus stepsize ?? := ?? 2 ?? 16??+?? 2 +4?? 2 +2???? 2 ???8???? where ?? = I ??? W 2 ??? [0, 2]

Algorithm 3 converges at the speed

where

i and c is convergence rate of underlying averaging scheme.

In contrast to Theorem A.2, this rate holds for any T , however the first term is worse than in Theorem A.2 because ?? 2 is usually much smaller than G 2 .

Proof of Theorem A.4.

By L-smoothness

Using Lemma A.1 we can bound the last term

Rearranging terms, taking ?? = 1 and averaging over t we are getting statement of the theorem

Corollary A.5 (Convergence of local weights x (t) i ).

Under Assumtions 2, 1, algorithm 1 with ?? = n T +1 converges at the speed

c 2 (T + 1) .

Proof.

where we used L-smoothness of f .

This holds for ????? > 0.

Using Theorem A.4 and Lemma A.1 and setting ?? = 1

where we set ?? = n T +1 .

Lemma B.1.

For arbitrary set of n vectors

Lemma B.3.

For given two vectors a,

This inequality also holds for the sum of two matrices A, B ??? R n??d in Frobenius norm.

Algorithm 4 CHOCO-SGD (Koloskova et al., 2019) as Error Feedback

, E) and mixing matrix W , initializex

1: for t in 0 . . .

T ??? 1 do {in parallel for all workers i ??? [n]} 2:

for neighbors j : {i, j} ??? E (including {i} ??? E) do end for 10:

11:

stochastic gradient update 12: end for D CHOCO-SGD WITH MOMENTUM Algorithm 2 demonstrates how to combine CHOCO-SGD with weight decay and momentum.

Nesterov momentum can be analogously adapted for our decentralized setting.

To better understand how does CHOCO-SGD work, we can interpret it as an error feedback algorithm (Stich et al., 2018; .

We can equivalently rewrite CHOCO-SGD (Algorithm 1) as Algorithm 4.

The common feature of error feedback algorithms is that quantization errors are saved into the internal memory, which is added to the compressed value at the next iteration.

In CHOCO-SGD the value we want to transmit is the difference x (t) i ??? x (t???1) i , which represents the evolution of local variable x i at step t. Before compressing this value on line 4, the internal memory is added on line 3 to correct for the errors.

Then, on line 5 internal memory is updated.

Note that m

i in the old notation.

We precise the procedure of model training as well as the hyper-parameter tuning in this section.

Social Network Setup.

For the comparison we consider CHOCO-SGD with sign compression (this combination achieved the compromise between accuracy and compression level in Table 1 )), decentralized SGD without compression, and centralized SGD without compression.

We train two models, firstly ResNet20 (He et al., 2016) (0.27 million parameters) for image classification on the Cifar10 dataset (50K/10K training/test samples) (Krizhevsky, 2012) and secondly, a three-layer LSTM architecture (Hochreiter & Schmidhuber, 1997 ) (28.95 million parameters) for a language modeling task on WikiText-2 (600 training and 60 validation articles with a total of 2 088 628 and 217 646 tokens respectively) (Merity et al., 2016) .

For the language modeling task, we borrowed and adapted the general experimental setup of Merity et al. (2017) , where we use a three-layer LSTM with hidden dimension of size 650.

The loss is averaged over all examples and timesteps.

The BPTT length is set to 30.

We fine-tune the value of gradient clipping (0.4), and the dropout (0.4) is only applied on the output of LSTM.

We train both of ResNet20 and LSTM for 300 epochs, unless mentioned specifically.

The per node mini-batch size is 32 for both datasets.

The learning rate of CHOCO-SGD follows a linear scaling rule, which is proportional to the node degree.

The momentum (with factor 0.9) is only applied on the ResNet20 training.

Social Network and a Datacenter details.

For all algorithms, we gradually warmup (Goyal et al., 2017 ) the learning rate from a relative small value (0.1) to the fine-tuned initial learning rate for the first 5 training epochs.

During the training procedure, the tuned initial learning rate is decayed by the factor of 10 when accessing 50% and 75% of the total training epochs.

The learning rate is tuned by finding the optimal learning rate per sample??, where the learning rate (used locally) is determined by a linear scaling rule (i.e., degree of node ???? ?? per node mini-batch size).

The optimal?? is searched in a pre-defined grid and we ensure that the best performance was contained in the middle of the grids.

For example, if the best performance was ever at one of the extremes of the grid, we would try new grid points.

Same searching logic applies to the consensus stepsize.

Table 4 demonstrates the fine-tuned hpyerparameters of CHOCO-SGD for training ResNet-20 on Cifar10, while Table 6 reports our fine-tuned hpyerparameters of our baselines.

Table 5 demonstrates the fine-tuned hpyerparameters of CHOCO-SGD for training ResNet-20/LSTM on a social network topology.

Table 5 : Tuned hyper-parameters of CHOCO-SGD, corresponding to the social network topology with 32 nodes in Table 3 .

We randomly split the training data between the nodes and keep this partition fixed during the entire training (no shuffling).

The per node mini-batch size is 32 and the maximum degree of the node is 14.

Base learning rate (before scaling by node degree) Consensus stepsize Figure 1 .

n = 4 n = 16 n = 36 n = 64 Figure 1 .

n = 4 n = 16 n = 36 n = 64 We additionally plot the learning curve for the social network topology in Figure.

6 and Figure. topology.

The topology has 32 nodes and we assume each node can only access a disjoint subset of the whole dataset.

The local mini-batch size is 32.

We additionally provide plots for training top-1, top-5 accuracy and test top-5 accuracy for the datacenter experiment in Figure 8 .

On Figure 9 we additionally depict the test accuracy of the averaged model

i (left) and averaged distance of the local models from the averaged model (right).

Towards the end of the optimization the local models reach consensus (Figure 9, right) , and their individual test performances are the same as performance of averaged model.

Interestingly, before decreasing the stepsize at the epoch 225, the local models are in general diverging from the averaged model, while decreasing only when the stepsize decreases.

The same behavior was also reported in Assran et al. (2019) .

<|TLDR|>

@highlight

We propose Choco-SGD---decentralized SGD with compressed communication---for non-convex objectives and show its strong performance in various deep learning applications (on-device learning, datacenter case).