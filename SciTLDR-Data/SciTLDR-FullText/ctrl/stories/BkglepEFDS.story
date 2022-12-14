Owing to the ubiquity of computer software, software vulnerability detection (SVD) has become an important problem in the software industry and in the field of computer security.

One of the most crucial issues in SVD is coping with the scarcity of labeled vulnerabilities in projects that require the laborious manual labeling of code by software security experts.

One possible way to address is to employ deep domain adaptation which has recently witnessed enormous success in transferring learning from structural labeled to unlabeled data sources.

The general idea is to map both source and target data into a joint feature space and close the discrepancy gap of those data in this joint feature space.

Generative adversarial network (GAN) is a technique that attempts to bridge the discrepancy gap and also emerges as a building block to develop deep domain adaptation approaches with state-of-the-art performance.

However, deep domain adaptation approaches using the GAN principle to close the discrepancy gap are subject to the mode collapsing problem that negatively impacts the predictive performance.

Our aim in this paper is to propose Dual Generator-Discriminator Deep Code Domain Adaptation Network (Dual-GD-DDAN) for tackling the problem of transfer learning from labeled to unlabeled software projects in the context of SVD in order to resolve the mode collapsing problem faced in previous approaches.

The experimental results on real-world software projects show that our proposed method outperforms state-of-the-art baselines by a wide margin.

In the software industry, software vulnerabilities relate to specific flaws or oversights in software programs which allow attackers to expose or alter sensitive information, disrupt or destroy a system, or take control of a program or computer system (Dowd et al., 2006) .

The software vulnerability detection problem has become an important issue in the software industry and in the field of computer security.

Computer software development employs of a vast variety of technologies and different software development methodologies, and much computer software contains vulnerabilities.

This has necessitated the development of automated advanced techniques and tools that can efficiently and effectively detect software vulnerabilities with a minimal level of human intervention.

To respond to this demand, many vulnerability detection systems and methods, ranging from open source to commercial tools, and from manual to automatic methods have been proposed and implemented.

Most of the previous works in software vulnerability detection (SVD) (Neuhaus et al., 2007; Shin et al., 2011; Yamaguchi et al., 2011; Almorsy et al., 2012; Li et al., 2016; Grieco et al., 2016; Kim et al., 2017) have been developed based on handcrafted features which are manually chosen by knowledgeable domain experts who may have outdated experience and underlying biases.

In many situations, handcrafted features normally do not generalize well.

For example, features that work well in a certain software project may not perform well in other projects (Zimmermann et al., 2009) .

To alleviate the dependency on handcrafted features, the use of automatic features in SVD has been studied recently (Li et al., 2018; Lin et al., 2018; Dam et al., 2018) .

These works have shown the advantages of automatic features over handcrafted features in the context of software vulnerability detection.

However, most of these approaches lead to another crucial issue in SVD research, namely the scarcity of labeled projects.

Labelled vulnerable code is needed to train these models, and the process of labeling vulnerable source code is very tedious, time-consuming, error-prone, and challenging even for domain experts.

This has led to few labeled projects compared with the vast volume of unlabeled ones.

A viable solution is to apply transfer learning or domain adaptation which aims to devise automated methods that make it possible to transfer a learned model from the source domain with labels to the target domains without labels.

Studies in domain adaptation can be broadly categorized into two themes: shallow (Borgwardt et al., 2006; Gopalan et al., 2011) and deep domain adaptations (Ganin & Lempitsky, 2015; Shu et al., 2018; .

These recent studies have shown the advantages of deep over shallow domain adaptation (i.e., higher predictive performance and capacity to tackle structural data).

Deep domain adaptation encourages the learning of new representations for both source and target data in order to minimize the divergence between them (Ganin & Lempitsky, 2015; Shu et al., 2018; .

The general idea is to map source and target data to a joint feature space via a generator, where the discrepancy between the source and target distributions is reduced.

Notably, the work of (Ganin & Lempitsky, 2015; Shu et al., 2018) employed generative adversarial networks (GANs) to close the discrepancy gap between source and target data in the joint space.

However, most of aforementioned works mainly focus on transfer learning in the computer vision domain.

The work of is the first work which applies deep domain adaptation to SVD with promising predictive performance on real-world source code projects.

The underlying idea is to employ the GAN to close the gap between source and target domain in the joint space and enforce the clustering assumption (Chapelle & Zien, 2005) to utilize the information carried in the unlabeled target samples in a semi-supervised context.

GANs are known to be affected by the mode collapsing problem (Goodfellow, 2016; Santurkar et al., 2018) .

In particular, (Santurkar et al., 2018) recently studied the mode collapsing problem and further classified this into the missing mode problem i.e., the generated samples miss some modes in the true data, and the boundary distortion problem i.e., the generated samples can only partly recover some modes in the true data.

It is certain that deep domain adaptation approaches that use the GAN principle will inherently encounter both the missing mode and boundary distortion problems.

Last but not least, deep domain adaptation approaches using the GAN principle also face the data distortion problem.

The representations of source and target examples in the joint feature space degenerate to very small regions that cannot preserve the manifold/clustering structure in the original space.

Our aim in this paper is to address not only deep domain adaptation mode collapsing problems but also boundary distortion problems when employing the GAN as a principle in order to close the discrepancy gap between source and target data in the joint feature space.

Our two approaches are: i) apply manifold regularization for enabling the preservation of manifold/clustering structures in the joint feature space, hence avoiding the degeneration of source and target data in this space; and ii) invoke dual discriminators in an elegant way to reduce the negative impacts of the missing mode and boundary distortion problems in deep domain adaptation using the GAN principle as mentioned before.

We name our mechanism when applied to SVD as Dual Generator-Discriminator Deep Code Domain Adaptation Network (Dual-GD-DDAN).

We empirically demonstrate that our Dual-GD-DDAN can overcome the missing mode and boundary distortion problems which is likely to happen as in Deep Code Domain Adaptation (DDAN) in which the GAN was solely applied to close the gap between the source and target domain in the joint space (see the discussion in Sections 2.4 and 3.3, and the visualization in Figure 3 ).

In addition, we incorporate the relevant approaches -minimizing the conditional entropy and manifold regularization with spectral graph -proposed in to enforce the clustering assumption (Chapelle & Zien, 2005) and arrive at a new model named Dual Generator-Discriminator Semi-supervised Deep Code Domain Adaptation Network (Dual-GD-SDDAN).

We further demonstrate that our Dual-GD-SDDAN can overcome the mode collapsing problem better than SCDAN in , hence obtaining better predictive performance.

We conducted experiments using the data sets collected by (Lin et al., 2018) , that consist of five real-world software projects: FFmpeg, LibTIFF, LibPNG, VLC and Pidgin to compare our proposed Dual-GD-DDAN and Dual-GD-SDDAN with the baselines.

The baselines consider to include VULD (i.e., the model proposed in (Li et al., 2018) without domain adaptation), MMD, DIRT-T, DDAN and SCDAN as mentioned and D2GAN (Nguyen et al., 2017 ) (a variant of the GAN using dual-discriminator to reduce the mode collapse for which we apply this mechanism in the joint feature space).

Our experimental results show that our proposed methods are able to overcome the negative impact of the missing mode and boundary distortion problems inherent in deep domain adaptation approaches when solely using the GAN principle as in DDAN and SCDAN .

In addition, our method outperforms the rival baselines in terms of predictive performance by a wide margin.

is also a sequence of L embedding vectors.

We wish to bridge the gap between the source and target domains in the joint feature space.

This allows us to transfer a classifier trained on the source domain to predict well on the target domain.

We preprocess data sets before inputting into the deep neural networks.

Firstly, we standardize the source code by removing comments, blank lines and non-ASCII characters.

Secondly, we map user-defined variables to symbolic names (e.g., "var1", "var2") and user-defined functions to symbolic names (e.g., "func1", "func2").

We also replace integers, real and hexadecimal numbers with a generic <num> token and strings with a generic <str> token.

Thirdly, we embed statements in source code into vectors.

In particular, each statement x consists of two parts: the opcode and the statement information.

We embed both opcode and statement information to vectors, then concatenate the vector representations of opcode and statement information to obtain the final vector representation i of statement x. For example, in the following statement (C programming language) "if(func3(func4(num,num),&var2)!=var11)", the opcode is if and the statement information is (func3(func4(num,num),&var2)!=var11).

To embed the opcode, we multiply the one-hot vector of the opcode by the opcode embedding matrix.

To embed the statement information, we tokenize it to a sequence of tokens (e.g., (,func3,(,func4,(,num,num,) ,&,var2,),!=,var11,)), construct the frequency vector of the statement information, and multiply this frequency vector by the statement information embedding matrix.

In addition, the opcode embedding and statement embedding matrices are learnable variables.

To handle sequential data in the context of domain adaptation of software vulnerability detection, the work of proposed an architecture referred to as the Code Domain Adaptation Network (CDAN).

This network architecture recruits a Bidirectional RNN to process the sequential input from both source and target domains (i.e., x

.

A fully connected layer is then employed to connect the output layer of the Bidirectional RNN with the joint feature layer while bridging the gap between the source and target domains.

Furthermore, inspired by the Deep Domain Adaptation approach (Ganin & Lempitsky, 2015) , the authors employ the source classifier C to classify the source samples, the domain discriminator D to distinguish the source and target samples and propose Deep Code Domain Adaptation (DDAN) whose objective function is as follows:

where seeking the optimal generator G * , the domain discriminator D * , and the source classifier C * is found by solving:

Figure 1: An illustration of the missing mode and boundary distortion problems of DDAN.

In the joint space, the target distribution misses source mode 2, while the source distribution can only partly cover the target mode 2 in the target distribution and the target distribution can only partly cover the source mode 1 in the source distribution.

We observe that DDAN suffers from several shortcomings.

First, the data distortion problem (i.e., the source and target data in the joint space might collapse into small regions) may occur since there is no mechanism in DDAN to circumvent this.

Second, since DDAN is based on the GAN approach, DDAN might suffer from the mode collapsing problem (Goodfellow, 2016; Santurkar et al., 2018) .

In particular, (Santurkar et al., 2018) has recently studied the mode collapsing problem of GANs and discovered that they are also subject to i) the missing mode problem (i.e., in the joint space, either the target data misses some modes in the source data or vice versa) and ii) the boundary distortion problem (i.e., in the joint space either the target data partly covers the source data or vice versa), which makes the target distribution significantly diverge from the source distribution.

As shown in Figure 1 , both the missing mode and boundary distortion problems simultaneously happen since the target distribution misses source mode 2, while the source distribution can only partly cover the target mode 2 in the target distribution and the target distribution can only partly cover the source mode 1 in the source distribution.

We employ two discriminators (namely, D S and D T ) to classify the source and target examples and vice versa and two separate generators (namely, G S and G T ) to map the source and target examples to the joint space respectively.

In particular, D S produces high values on the source examples in the joint space (i.e., G S x S ) and low values on the target examples in the joint space (i.e., G T x T ), while D T produces high values on the target examples in the joint space (i.e., G T x T ) and low values on the source examples (i.e., G S x S ).

The generator G S is trained to push G S x S to the high value region of D T and the generator G T is trained to push G T x T to the high value region of D S .

Eventually, both D S G S x S and D S G T x T are possibly high and both

are possibly high.

This helps to mitigate the issues of missing mode and boundary distortion since as in Figure 1 , if the target mode 1 can only partly cover the source mode 1, then D T cannot receive large values from source mode 1.

Another important aspect of our approach is to maintain the cluster/manifold structure of source and target data in the joint space via the manifold regularization to avoid the data distortion problem.

To address the two inherent problems in the DDAN mentioned in Section 2.4, we employ two different generators G S and G T to map source and target domain examples to the joint space and two discriminators D S and D T to distinguish source examples against target examples and vice versa together with the source classifier C which is used to classify the source examples with labels as shown in Figure 2 .

We name our proposed model as Dual Generator-Discriminator Deep Code Domain Adaptation Network (Dual-GD-DDAN).

Updating the discriminators The two discriminators D S and D T are trained to distinguish the source examples against the target examples and vice versa as follows:

where ?? > 0.

Note that a high value of ?? encourages D s and D T place higher values on G S x S and G T x T respectively.

Updating the source classifier The source classifier is employed to classify the source examples with labels as follows: min

, y i where specifies the cross-entropy loss function for the binary classification (e.g., using crossentropy).

Updating the generators The two generators G S and G T are trained to i) maintain the manifold/cluster structures of source and target data in their original spaces to avoid the data distortion problem and ii) move the target samples toward the source samples in the joint space and resolve the missing mode and boundary distortion problems in the joint space.

To maintain the manifold/cluster structures of source and target data in their original spaces, we propose minimizing the manifold regularization term as: min

where M (G S , G T ) is formulated as:

where the weights are defined as

are the last hidden states of the bidirectional RNN with input x.

To move the target samples toward the source samples and resolve the missing mode and boundary distortion problems in the joint space, we propose minimizing the following objective function:

where K (G S , G T ) is defined as:

Moreover, the source generator G S has to work out the representation that is suitable for the source classifier, hence we need to minimize the following objective function: GD-DDAN) .

The generators G S and G T take the sequential code tokens of the source domain and target domain in vectorial form respectively and map this sequence to the joint layer (i.e., the joint space).

The discriminators D S and D T are invoked to discriminate the source and target data.

The source classifier C is trained on the source domain with labels.

We note that the source and target networks do not share parameters and are not identical.

Finally, to update G S and G T , we need to minimize the following objective function:

where ??, ?? > 0 are two non-negative parameters.

Below we explain why our proposed Dual-GD-DDAN is able to resolve the two critical problems that occur with the DDAN approach.

First, if x 4).

This increases the chance of the two representations residing in the same cluster in the joint space.

Therefore, Dual-GD-DDAN is able to preserve the clustering structure of the source data in the joint space.

By using the same argument, we reach the same conclusion for the target domain.

Second, following Eqs. (2, 3), the discriminator D S is trained to encourage large values for the source modes (i.e., G S x S ), while the discriminator D T is trained to produce large values for the target modes (i.e., G T x T ).

Moreover, as in Eq. (6), G s is trained to move the source domain examples x S to the high-valued region of D T (i.e., the target modes or G T x T ) and G T is trained to move the target examples x T to the high-valued region of D S (i.e., the source modes or G S x S ).

As a consequence, eventually, the source modes (i.e., G S x S ) and target modes (i.e., G T x T ) overlap, while D S and D T place large values on both source (i.e., G S x S ) and target (i.e., G T x T ) modes.

The mode missing problem is less likely to happen since, as shown in Figure  1 , if the target data misses source mode 2, then D T cannot receive large values from source mode 2.

Similarly, the boundary distortion problem is also less likely to happen since as in Figure 1 , if the target mode 1 can only partly cover the source mode 1, then D T cannot receive large values from source mode 1.

Therefore, Dual-GD-DDAN allows us to reduce the impact of the missing mode and boundary distortion problems, hence making the target distribution more identical to the source distribution in the joint space.

When successfully bridging the gap between the source and target domains in the joint layer (i.e., the joint space), the target samples can be regarded as the unlabeled portion of a semi-supervised learning problem.

Based on this observation, Nguyen et al. proposed to enforce the clustering assumption (Chapelle & Zien, 2005) by minimizing the conditional entropy and using the spectral graph to inspire the smoothness of the source classifier C. Using our proposed Dual-GD-DDAN, the conditional entropy H (C, G S , G T ) is defined as:

Let SG = (V, E) where the set of vertices V = S ??? T be the spectral graph defined as in .

The smoothness-inspired term is defined as:

where B u specifies the Bernoulli distribution with P (y = 1 | u) = C (u) and

, and KL (B u , B v ) specifies the Kullback-Leibler divergence between two distributions.

Here we note that u = G S x S and v = G T x T are two representations of the source sample x S and the target sample x T in the joint space.

We incorporate these two terms into our Dual Generator-Discriminator mechanism to propose Dual Generator-Discriminator Semi-supervised Deep Code Domain Adaptation Network (Dual-GD-SDDAN) with the following objective function:

where ??, ?? are two non-negative parameters.

We present experimental results of applying our Dual-GD-DDAN approach to five real-world software projects (Lin et al., 2018) .

We compare our proposed Dual-GD-DDAN with VulDeePecker without domain adaptation, MMD, D2GAN, DIRT-T and DDAN using the architecture CDAN proposed in .

We further compare our proposed Dual Generator-Discriminator Semi-supervised Deep Code Domain Adaptation (Dual-GD-SDDAN) and Semi-supervised Deep Code Domain Adaptation (SCDAN) introduced in .

We use the real-world data sets collected by (Lin et al., 2018) , which contain the source code of vulnerable and non-vulnerable functions obtained from five real-world software projects, namely FFmpeg (#vul-funcs: 187, #non-vul-funcs: 5,427), LibTIFF (#vul-funcs: 81, #non-vul-funcs: 695), LibPNG (#vul-funcs: 43, #non-vul-funcs: 551), VLC (#vul-funcs: 25, #non-vul-funcs: 5,548) and Pidgin (#vul-funcs: 42, #non-vul-funcs: 8,268) where #vul-funcs and #non-vul-funcs is the number of vulnerable and non-vulnerable functions respectively.

The data sets contain both multimedia (FFmpeg, VLC, Pidgin) and image (LibPNG, LibTIFF) application categories.

In our experiment, some of the data sets from the multimedia category were used as the source domain whilst other data sets from the image category were used as the target domain (see Table 1 ).

For training the eight methods -VulDeePecker, MMD, D2GAN, DIRT-T, DDAN, Dual-GD-DDAN, SCDAN and Dual-GD-SDDAN -we use one-layer bidirectional recurrent neural networks with LSTM cells where the size of hidden states is in {128, 256} for the generators.

For the source classifier and discriminators, we use deep feed-forward neural networks with two hidden layers in which the size of each hidden layer is in {200, 300}. We embed the opcode and statement information in the {150, 150} dimensional embedding spaces respectively.

We employ the Adam optimizer (Kingma & Ba, 2014) with an initial learning rate in {0.001, 0.0001}. The mini-batch size is 64.

The trade-off parameters ??, ??, ??, ?? are in {10 ???1 , 10 ???2 , 10 ???3 }, ?? is in {0, 1} and 1/(2?? 2 ) is in {2 ???10 , 2 ???9 }.

We split the data of the source domain into two random partitions containing 80% for training and 20% for validation.

We also split the data of the target domain into two random partitions.

The first partition contains 80% for training the models of VulDeePecker, MMD, D2GAN, DIRT-T, DDAN, Dual-GD-DDAN, SCDAN and Dual-GD-SDDAN without using any label information while the second partition contains 20% for testing the models.

We additionally apply gradient clipping regularization to prevent over-fitting in the training process of each model.

We implement eight mentioned methods in Python using Tensorflow which is an open-source software library for Machine Intelligence developed by the Google Brain Team.

We run our experiments on a computer with an Intel Xeon Processor E5-1660 which had 8 cores at 3.0 GHz and 128 GB of RAM.

For each method, we run the experiments 5 times and then record the average predictive performance.

Quantitative Results We first investigate the performance of our proposed Dual-GD-DDAN compared with other methods including VulDeePecker (VULD) without domain adaptation (Li et al., 2016) , DDAN , MMD , D2GAN (Nguyen et al., 2017) and DIRT-T with VAP applied in the joint feature layer using the architecture CDAN introduced in .

The VulDeePecker method is only trained on the source data and then tested on the target data, while the MMD, D2GAN, DIRT-T, DDAN and Dual-GD-DDAN methods employ the target data without using any label information for domain adaptation.

Quantitative Results To quantitatively demonstrate the efficiency of our proposed Dual-GD-DDAN in alleviating the boundary distortion problem caused by using the GAN principle, we reuse the experimental setting in Section 5.2 (Santurkar et al., 2018) .

The basic idea is, given two data sets S 1 and S 2 , to quantify the degree of cover of these two data sets.

We train a classifier C 1 on S 1 , then test on S 2 and another classifier C 2 on S 2 , then test on S 1 .

If these two data sets cover each other well with reduced boundary distortion, we expect that if C 1 predicts well on S 1 , then it should predict well on S 2 and vice versa if C 2 predicts well on S 2 , then it should predict well on S 1 .

This would seem reasonable since if boundary distortion occurs (i.e., assume that S 2 partly covers S 1 ), then C 2 trained on S 2 would struggle to predict S 1 well which is much larger and possibly more complex.

Therefore, we can utilize the magnitude of the accuracies and the accuracy gap of C 1 and C 2 when predicting their training and testing sets to assess the severity of the boundary distortion problem.

Figure 3: A 2D t-SNE projection for the case of the FFmpeg ??? LibPNG domain adaptation.

The blue and red points represent the source and target domains in the joint space respectively.

In both cases of the source and target domains, data points labeled 0 stand for non-vulnerable samples and data points labeled 1 stand for vulnerable samples.

Inspired by this observation, we compare our Dual-GD-DDAN with DDAN using the representations of the source and target samples in the joint feature space corresponding to their best models.

In particular, for a given pair of source and target data sets and for comparing each method, we train a neural network classifier on the best representations of the source data set in the joint space, then predict on the source and target data set and do the same but swap the role of the source and target data sets.

We then measure the difference of the corresponding accuracies as a means of measuring the severity of the boundary distortion.

We choose to conduct such a boundary distortion analysis for two pairs of the source (FFmpeg and Pidgin) and target (LibPNG) domains.

As shown in Table 2 , all gaps obtained by our Dual-GD-DDAN are always smaller than those obtained by DDAN, while the accuracies obtained by our proposed method are always larger.

We can therefore conclude that our Dual-GD-DDAN method produces a better representation for source and target samples in the joint space and is less susceptible to boundary distortion compared with the DDAN method.

Visualization We further demonstrate the efficiency of our proposed Dual-GD-DDAN in alleviating the boundary distortion problem caused by using the GAN principle.

Using a t-SNE (Laurens & Geoffrey, 2008) projection, with perplexity equal to 30, we visualize the feature distributions of the source and target domains in the joint space.

Specifically, we project the source and target data in the joint space (i.e., G (x)) into a 2D space with domain adaptation (DDAN) and with dual-domain adaptation (Dual-GD-DDAN).

In Figure 3 , we observe these cases when performing domain adaptation from a software project (FFmpeg) to another (LibPNG).

As shown in Figure 3 , with undertaking domain adaptation (DDAN, the left figure) and dual-domain adaptation (Dual-GD-DDAN, the right figure) , the source and target data sampled are intermingled especially for Dual-GD-DDAN.

However, it can be observed that DDAN when solely applying the GAN is seriously vulnerable to the boundary distortion issue.

In particular, in the clusters/data modes 2, 3 and 4 (the left figure) , the boundary distortion issue occurs since the blue data only partly cover the corresponding red ones (i.e., the source and target data do not totally mix up).

Meanwhile, for our Dual-GD-DDAN, the boundary distortion issue is much less vulnerable, and the mixing-up level of source and target data is significantly higher in each cluster/data mode.

In this section, we compare the performance of our Dual Generator-Discriminator Semi-supervised Deep Code Domain Adaptation (Dual-GD-SDDAN) with Semi-supervised Deep Code Domain Adaptation (SCDAN) on four pairs of source and target domain including FFmpeg ??? LibTIFF, FFmpeg ??? LibPNG, VLC??? LibPNG and Pidgin ??? LibTIFF.

In Table 3 , the experimental results show that our Dual-GD-SDDAN achieves a higher performance than SCDAN for detecting vulnerable and non-vulnerable functions in terms of FPR, Precision and F1-measure in almost cases of the source and target domains, especially for F1-measure.

For example, to the case of the source domain (VLC) and target domain (LibPNG), our Dual-GD-SDDAN achieves an F1-measure of 76.19% compared with an F1-measure of 72.73% obtained with SCDAN.

These results further demonstrate the ability of our Dual-GD-SDDAN for dealing with the mode collapsing problem better than SCDAN , hence obtaining better predictive performance in the context of software domain adaptation.

Software vulnerability detection (SVD) is an important problem in the software industry and in the field of computer security.

One of the most crucial issues in SVD is to cope with the scarcity of labeled vulnerabilities in projects that require the laborious labeling of code by software security experts.

In this paper, we propose the Dual Generator-Discriminator Deep Code Domain Adaptation Network (Dual-GD-DDAN) method to deal with the missing mode and boundary distortion problems which arise from the use of the GAN principle when reducing the discrepancy between source and target data in the joint space.

We conducted experiments to compare our Dual-GD-DDAN method with the state-of-the-art baselines.

The experimental results show that our proposed method outperforms these rival baselines by a wide margin in term of predictive performances.

We give an example of source code functions obtained from the VLC and LibPNG projects, to demonstrate that transfer learning for software vulnerability detection between different projects is plausible and promising.

Both C language functions obtained from the VLC and LibPNG projects depicted in Figure 4 invoke the memcpy function which is used to copy one memory buffer to another.

The misuse of this function can cause a buffer overflow error if insufficient memory is allocated in the target buffer for all of the data to be copied from the source buffer.

Furthermore, these functions also share rather similar semantic and syntactic relationships (i.e. the C language programming syntax, loop structure etc).

Therefore, a model that can capture the characteristics of the first function in the first project should be able to confidently predict the second function in the second project.

It therefore makes sense to undertake transfer learning from the first project to the second project.

Figure 4: An example of two source code functions (with some parts omitted for brevity) in the C programming language obtained from the VLC (Left) and LibPNG project (Right).

These two source code examples highlight the same vulnerability due to the misuse of the memcpy function.

In this section, we introduce work related to ours.

First, we present the recent work in automatic feature learning for software vulnerability detection.

Finally, we present the recent work in deep domain adaptation.

Automatic feature learning in software vulnerability detection minimizes intervention from security experts (Li et al., 2018; Lin et al., 2018; Dam et al., 2018) .

Particularly, (Dam et al., 2018; Lin et al., 2018) shared the same approach employing a Recurrent Neutral Network (RNN) to transform sequences of code tokens to vectorial features for automatic feature learning, which are then fed to a separate classifier (e.g., Support Vector Machine (Cortes & Vapnik, 1995) or Random Forest (Breiman, 2001) ) for classification purposes.

However, owing to the independence of learning the vector representations and training the classifier, it is likely that the resulting vector representations of (Lin et al., 2018; Dam et al., 2018) may not fit well with classifiers to enhance the predictive performance.

To deal with this problem, the study introduced in (Li et al., 2018) combined the learning of the vector representations and the training of a classifier in a deep neural network.

This work

<|TLDR|>

@highlight

Our aim in this paper is to propose a new approach for tackling the problem of transfer learning from labeled to unlabeled software projects in the context of SVD in order to resolve the mode collapsing problem faced in previous approaches.