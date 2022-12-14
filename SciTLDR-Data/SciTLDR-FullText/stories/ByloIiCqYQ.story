Due to the sharp increase in the severity of the threat imposed by software vulnerabilities, the detection of vulnerabilities in binary code has become an important concern in the software industry, such as the embedded systems industry, and in the field of computer security.

However, most of the work in binary code vulnerability detection has relied on handcrafted features which are manually chosen by a select few, knowledgeable domain experts.

In this paper, we attempt to alleviate this severe binary vulnerability detection bottleneck by leveraging recent advances in deep learning representations and propose the Maximal Divergence Sequential Auto-Encoder.

In particular, latent codes representing vulnerable and non-vulnerable binaries are encouraged to be maximally divergent, while still being able to maintain crucial information from the original binaries.

We conducted extensive experiments to compare and contrast our proposed methods with the baselines, and the results show that our proposed methods outperform the baselines in all performance measures of interest.

Software vulnerabilities are specific flaws or oversights in a piece of software that allow attackers to perform malicious acts including exposing or altering sensitive information, disrupting or destroying a system, or taking control of a computer system or program BID5 .

Due to the ubiquity of computer software, the growth and the diversity in its development process, a great amount of computer software potentially includes software vulnerabilities.

This fact makes the problem of software security vulnerability identification an important concern in the software industry and in the field of computer security.

Although a great effort has been made by the security community, the severity of the threat from software vulnerabilities has gradually increased over the years.

Numerous exist of examples and incidents in the past two decades in which software vulnerabilities have imposed significant damages to companies and individuals BID6 .

For example, vulnerabilities in popular browser plugins have threatened the security and privacy of millions of Internet users (e.g., Adobe Flash Player (US-CERT 2015; Adobe Security Bulletin 2015) and Oracle Java (US-CERT 2013)), vulnerabilities in popular and fundamental open-source software have also threatened the security of thousands of companies and their customers around the globe (e.g., Heartbleed (Codenomicon 2014) and ShellShock (Symantec Security Response 2014).Software vulnerability detection (SVD) can be categorized into source code and binary code vulnerability detection.

Source code vulnerability detection has been widely studied in a variety of works BID20 BID17 BID22 BID13 BID9 BID14 .

Most of the previous work in source code vulnerability detection BID17 BID20 BID22 BID13 BID9 has been based on handcrafted features which are manually chosen by a limited number of domain experts.

To mitigate the dependency on handcrafted features, the use of automatic features in SVD has been studied recently in BID4 BID14 BID15 .

In particular, BID4 ; BID15 employed a Recurrent Neural Network (RNN) to transform sequences of code tokens to vectorial features, which are further fed to a separate classifier, while BID14 combined learning the vector representation and the training of the classifier in a deep network.

Compared with source code vulnerability detection, binary code vulnerability detection is significantly more difficult because much of the syntactic and semantic information provided by high-level programming languages is lost during the compilation process.

The existence of such syntactic and semantic information makes it easier to reason how data and inputs drive the paths of execution.

Unfortunately, a software binary, such as proprietary binary code (with no access to source code) or embedded systems code, is generally all that is made available for code analysis (together perhaps with the processor architecture such as x86 etc.).

The ability to detect the presence or absence of vulnerabilities in binary code, without getting access to source code, is therefore a major importance in the context of computer security.

Some work has been proposed to detect vulnerabilities at the binary code level when source code is not available, notably work based on fuzzing, symbolic execution BID2 BID1 BID16 , or techniques using handcrafted features extracted from dynamic analysis BID8 BID3 BID21 .

To the best of our knowledge, there has been no work studying the use of automatically extracted features for binary code vulnerability detection, though there has been some work using automatic features in conjunction with deep learning methods for malware detection, notably BID19 BID18 .

It is worth noting that binary code vulnerability detection and malware detection are two different tasks.

In particular, binary code vulnerability detection aims to detect specific flaws or oversights in binary code, while malware detection aims to detect if a given binary is malicious or not.

The former is arguably harder in the sense that vulnerable and non-vulnerable binaries might be only slightly different, while there might be a clearer difference in general between malware and benign binaries.

In addition, a significant constraint in research on binary code vulnerability detection is the lack of suitable binaries labeled as either vulnerable or non-vulnerable.

Although we have some source code datasets for software vulnerability detection, to the best of our knowledge, there exists no large public binary dataset for the purpose of binary code vulnerability detection.

The reason is that most source code in source code vulnerability detection datasets is not compilable due to incompleteness, and they have important pieces missing (e.g., variables, data types) and relevant libraries -making the code compilable take a large effort in fixing a vast volume of source code.

This arises from the nature of the process that involves collecting and labeling source code wherein we start from security reports in CVE 1 and navigate through relevant websites to obtain code snippets of vulnerable and non-vulnerable source code.

In this work, we leverage recent advances in deep learning to derive the automatic features of binary code for vulnerability detection.

In particular, we view a given binary as a sequence of machine instructions and then use the theory of Variational Auto-Encoders (VAE) BID11 to develop the Maximal Divergence Sequential Auto-Encoder (MDSAE) that can work out representations of binary code in such a way that representations of vulnerable and nonvulnerable binaries are encouraged to be maximally different for vulnerability detection purposes, while still preserving crucial information inherent in the original binaries.

In contrast to the original VAE wherein the data prior is kept fixed, we propose using two learnable Gaussian priors, one for each class.

Based on the VAE principle, latent codes (i.e., data representations) are absorbed (or compressed) into the data prior distribution, we further propose maximizing a divergence (e.g., Wasserstein (WS) distance or Kullback-Leibler (KL) divergence) between these two priors to separate representations of vulnerable and non-vulnerable binaries.

Our MDSAE can be used to produce data representations for another independent classifier (e.g., Support Vector Machine or Random Forest) or incorporated with a shallow feedforward neural network built on top of the latent codes for simultaneously training both the mechanism to generate data representations and the classifier.

The former is named MDSAE-R and the latter is named MDSAE-C. We summarize our contributions in this paper as follows:??? We propose a novel method named Maximal Divergence Sequential Auto-Encoder (MDSAE) that leverages recent advances in deep learning representation (namely, VAE) for binary code vulnerability detection.??? One of our most significant contributions is to create a labeled dataset for use in binary code vulnerability detection.

In particular, we used the source code in the published NDSS18 dataset used in BID14 and then extracted vulnerable and non-vulnerable functions.

We developed a tool that can automatically detect the syntactical errors in a given piece of source code, fix them, and finally compile the fixed source code into binaries for various platforms (both Windows OS and Linux OS) and architectures (both x86 and x86-64 processors).

Specifically, after preprocessing and filtering out identical functions from the NDSS18 source code dataset, we obtain 13, 000 functions of which 9, 000 are able to be fixed and compiled to binaries.

By compiling the source code of these functions under the various platform and architecture options, we obtain 32, 281 binary functions including 17, 977 binaries for Windows and 14, 304 binaries for Linux.??? We conducted extensive experiments on the NDSS18 binary dataset.

The experimental results show that the two variants MDSAE-R and MDSAE-C outperform the baselines in all performance measures of interest.

It is not surprising that MDSAE-C achieves higher predictive performances compared with MDSAE-R, but the fact that MDSAE-R achieves good predictive performances confirms our hypothesis of encouraging the separation in representations of data in different classes so that a simple linear classifier subsequently trained on these data representations can obtain good predictive results.

The Variational Auto-Encoder (VAE) BID11 ) is a probabilistic auto-encoder that takes into account both the reconstruction of true samples and generalization of samples generated from a latent space.

The underlying idea is to learn a probabilistic decoder p ?? (x | z) , z ??? N (0, I) that can mimic the true data sample x 1 , . . .

, x N drawn from an existing but unknown data distribution p d (x).

VAE is developed based on the following lower bound: DISPLAYFORM0 where q ?? (z | x) is the approximate posterior distribution.

We need to maximize the log likelihood at each training example x.

Therefore the objective function is of the following form: DISPLAYFORM1 where x is drawn from the empirical data distribution.

To reduce the variance when using Monte Carlo (MC) estimation for tackling the above optimization problem, the reparameterization trick is employed.

More specifically, assuming that DISPLAYFORM2 where the source of randomness ??? N (0, I) and ?? ?? (z) , ?? ?? (z) are two neural networks representing the mean and covariance matrix of the approximate Gaussian posterior.

The optimization problem in Eq. (1) can be equivalently rewritten as: DISPLAYFORM3 The first term in Eq. FORMULA3 is regarded as the reconstruction term and the second term in this equation is regarded as the regularization term.

In this term, we minimize DISPLAYFORM4 , hence trying to compress and squash the latent codes z for each true example x into those sampled from the prior distribution p (z).

This observation is the key ingredient for us to develop our proposed model.

Given two distributions with the probability density functions p (z) and q (z) where z ??? R d , the Kullback-Leibler (KL) divergence between these two distributions is defined as: DISPLAYFORM0 Another divergence of interest is L2 Wasserstein (WS) distance with the cost function c (z 1 , z 2 ) = z 1 ??? z 2 2 2 .

The L2 Wasserstein divergence between two distributions is defined as: DISPLAYFORM1 where ?? (q, p) specifies the set of all joint distributions over p, q which admits p, q as marginals.

If p, q are two Gaussians, i.e., p (z) = N (z | ?? 1 , ?? 1 ) and q (z) = N (z | ?? 2 , ?? 2 ) then both KL divergence and L2 WS distance can be computed in close forms as: DISPLAYFORM2 where ?? F is the Frobenius norm and ?? 1 ?? 2 = ?? 2 ?? 1 .

For each machine instruction, we employ the Capstone 2 binary disassembly framework to detect entire machine instructions.

We then eliminate redundant prefixes to obtain the core parts that contain the opcode and other significant information.

Each core part in a machine instruction consists of two parts: the opcode and instruction information (i.e., memory location, registers, etc.).

We embed both the opcode and instruction information into vectors and then concatenate them.

To embed the opcode, we build a vocabulary of opcodes and then multiply the one-hot vector of the opcode with the corresponding embedding matrix.

To embed the instruction information, we build the vocabulary over 256 hex-bytes from 00 to F F , then view the instruction information as a sequence of hex-bytes to construct the frequency vector of a size 256, and finally multiply this frequency vector with the corresponding embedding matrix.

More specifically, the output embedding is e = e op e ii where e op = one-hot(op) ?? W op and e ii = freq (ii) ?? W ii with the opcode op, the instruction information ii and its frequency vector freq (ii), and the embedding matrices W op and W ii .

The process of embedding machine instructions is presented in FIG0 .

FOR

In this work, we view binary code x as a sequence of machine instructions, i.e., x = [ DISPLAYFORM0 where each x i is a machine instruction.

Our idea is to encode x to the latent code z in such a way that the latent codes of data in different classes are encouraged to be maximally divergent.

Let us denote the distributions of vulnerable and non-vulnerable sequences by p 1 (x) and p 0 (x) respectively.

Inspired by the Variational Auto-Encoder BID11 , we propose to use a probabilistic decoder p ?? (x | z) such that for z ??? p 0 (z), x drawn from p ?? (x | z) can mimic those drawn from p 0 (x) and for z ??? p 1 (z), x drawn from p ?? (x | z) can mimic those drawn from p 1 (x).

In other words, we aim to learn the probabilistic decoder p ?? (x | z) satisfying: DISPLAYFORM1 For any approximate posterior q ?? (z | x), we have the following lower bounds: DISPLAYFORM2 Figure 2: Maximal divergence sequential auto-encoder.

The latent codes of vulnerable and nonvulnerable are encouraged to be maximally divergent, while still maintaining crucial information from the original binaries.

Note that we use the same network for q ?? (z | x, y = 0) and q ?? (z | x, y = 1) and they are discriminated by the source of data used to fit.

Using the architecture shown in Figure 2 , we consider the probabilistic decoder p ?? (x | z) of the following parametric form DISPLAYFORM3 and hence we can further derive the lower bounds as: DISPLAYFORM4 We arrive at the following optimization problem: max DISPLAYFORM5 It is worth noting that since we are minimizing: DISPLAYFORM6 the encoding z ??? q ?? (z | h L ) with y = 0 are absorbed (compressed) into the prior p 0 (z).

Similarly, the encoding z ??? q ?? (z | h L ) with y = 1 are compressed into the prior p 1 (z).

Therefore, to maximize the difference between the encodings of data in the two classes, we propose to maximize the divergence between p 0 (z) and p 1 (z) and arrive the following optimization problem: DISPLAYFORM7 where ?? > 0 is a non-negative trade-off parameter and D p 0 (z) p 1 (z) is the divergence between the two priors.

To facilitate the evaluation, we endow these two priors with Gaussian distributions as follows: DISPLAYFORM8 We also propose using the Gaussian approximate posterior as: DISPLAYFORM9 1/2 , where the source of randomness ??? N (0, I) and ?? ?? (z) , ?? ?? (z) are two neural networks representing the mean and covariance matrix of the approximate Gaussian posterior.

Hence we come to the following optimization problem: DISPLAYFORM10 where we note that D p 0 (z) p 1 (z) is tractable for both the KL-divergence and L2 Wasserstein distance (See Section 2.2) and L 0 (x; ??, ??), L 1 (x; ??, ??) can be rewritten using the reparameterization trick as: DISPLAYFORM11 To classify data, we can train a classifier C over the latent space either independently or simultaneously with the maximal divergence auto-encoder.

If we train the classifier simultaneously, the final optimization problem is as follows: DISPLAYFORM12 where C ?? (x) stands for the probability to classify x as a vulnerable binary code (y = 1), and ??, ?? > 0 are two non-negative trade-off parameters.

It is worth noting that to model the conditional distributions p ?? (x i | h i???1 , z), we only take into account the opcode of the machine instruction x i .

Since this opcode lies in a fixed vocabulary of the opcodes, we can use the softmax distribution to define the corresponding distribution p ?? (x i | h i???1 , z).

By this means, the reconstruction phase aims to reconstruct the opcodes of the machine instructions in a given binary rather than the whole machine instructions.

One of the most significant contributions of our work is to create a labeled binary dataset for binary code vulnerability detection.

We first extracted the functions from the NDSS18 source code dataset.

We then preprocessed and filtered out any identical functions to obtain 13, 000 functions, of which 9, 000 could be fixed to compile to binaries using our automatic tool.

In addition, we developed a tool based on Joern 3 to parse the semantic and syntactical relationships in a given piece of source code.

In particular, our tool first used the compiler gcc/g++ (MinGW) to compile a given piece of source code, then captured the error messages, parsed these error messages, relied on Joern to be aware of the semantic and syntactical relationships of the error messages with respect to the source code, and finally fixed the corresponding error message.

This process was repeated until the given source is error-free and ready to compile to a binary.

By compiling the compilable function source code under various platforms and architectures, we obtained 32, 281 binary functions including 17, 977 binaries for Windows and 14, 304 binaries for Linux.

The statistics of our binary dataset is given in TAB1 .

Additionally, in order to obtain this binary dataset our tool fixed tens of thousands of errors of which many are strongly associated with specific source code.

We compared our proposed methods MDSAE-R (for learning maximally divergent representations in conjunction with an independent linear classifier to classify vulnerable and non-vulnerable functions) and MDSAE-C (for learning maximally divergent representations incorporating a linear classifier) with the following baselines:

??? RNN-R: A Recurrent Neural Network (RNN) for learning representations and linear classifier independently trained on the resulting representations for classifying vulnerable and non-vulnerable functions.

In addition, to learn representations in an unsupervised manner, we applied the method of language modeling whereby we trained the model to predict the opcode of the next machine instruction given the previous machine instructions.??? RNN-C: A RNN with a linear classifier built on the top of the last hidden unit.??? Para2Vec: The paragraph-to-vector distributional similarity model proposed in BID12 .

This work proposed to embed paragraphs including many words in a fixed vocabulary into a vector space.

To apply this work in our context, we view a binary as a sequence of opcodes residing in the fixed vocabulary of the opcodes.??? SeqVAE-C: Sequential VAE as in Section 3.2, but we set two priors to N (0, I) and kept fixed during training as in the original VAE.

A linear classifier was built up on the top of the latent codes and trained simultaneously.

With this setting, we aim to show that learning the priors produces more separable representations, hence boosts the performance.??? VulDeePecker: proposed in BID14 for source code vulnerability detection.

This model employed a Bidirectional RNN (BRNN) to take sequential inputs and then concatenated hidden units to input to a feedforward neural net classifier.

This method can inherently be applied to binaries wherein sequences of machine instructions are inputted to the BRNN.In addition, we also inspected two variants of divergence (i.e., KL divergence and L2 WS distance (See Section 2.2)) for formulating the divergence D p 0 (z) p 1 (z) in the optimization problem in Eq. (3).

Consequently, we have four variants of our proposed method, namely MDSAE-RKL, MDSAE-RWS, MDSAE-CKL, and MDSAE-CWS.

We split the data into 80% for training, 10% for validation, and the remaining 10% for testing.

We employed a dynamic RNN to tackle the variation in the number of machine instructions of the functions.

For the RNN baselines and our models, the size of hidden unit was set to 256.

For our model, the size of the latent space was set to 4,096, the trade-off parameters ??, ?? were set to 2??10 ???2 and 10 ???4 respectively.

We used the Adam optimizer BID10 with an initial learning rate equal to 0.0001.

The minibatch size was set to 64 and the number of epochs was set to 100.

We implemented our proposed method in Python using Tensorflow BID0 , an open-source software library for Machine Intelligence developed by the Google Brain Team.

The source code, as well as the dataset, is available in our GitHub repository 4 .

We ran our experiments on a computer with an Intel Xeon Processor E5-1660 which had 8 cores at 3.0 GHz and 128 GB of RAM.

We conducted the experiments on the subset of Windows binaries, the subset of Linux binaries, and the whole set of binaries to compare our methods with the baselines.

The experimental results are shown in Table 2 .

It can be seen that our proposed method outperforms the baselines in all performance measures of interest.

Specifically, in the field of computer security, the recall is a very important measure of completeness since a higher recall value leads to fewer vulnerable functions being incorrectly classified as non-vulnerable, which can otherwise present an issue for code auditors when there can be a large imbalance in the number of non-vulnerable and vulnerable functions in real-world use.

In addition, the fact that the resulting data representations of MDSAE-RKL and MDSAE-RWS work well with a linear classifier confirms our intuition and motivation of that the encouragement of data separation effectively supports the classifiers.

Table 2 : The experimental results in percent (%) of the proposed methods compared with the baselines on the NDSS18 binary dataset.

Acc, Rec, and Pre are shorthand for the performance measures accuracy, recall, and precision, respectively.

Distances between Two Priors, Distributions of Vulnerable, Non-vulnerable Classes During Training In this experiment, we study i) the L2 WS distance between the two priors, ii) the Euclidean distance of two means of priors (i.e., ?? 0 ??? ?? 1 ), iii) the KL divergence of DISPLAYFORM0 , and vi) the reconstruction loss across epochs of MDSAE-RWS-the variant of our proposed method for learning separable representations.

As shown in Figure 3 , during the training process, two distributions p (z | y = 0) and p (z | y = 1) become consistently and gradually more distant with the increase in their MMD distance (Figure 3 , second row, middle), hence implying the gradually increasing separation of the corresponding latent codes.

In addition, as we expect, the two priors become consistently and gradually more distant (Figure 3 , first row, left-hand side and Figure 3 , first row, middle) and the latent codes of vulnerable (y = 1) and non-vulnerable (y = 0) classes become more compressed into its priors respectively (Figure 3 , first row, right-hand side and Figure 3 , second row, left-hand side).

Furthermore, the reconstruction error consistently decreases which implies that the latent codes maintain crucial information of the original binaries (Figure 3 , second row, right-hand side).Figure 3: The L2 WS distance between two priors (first row, left-hand side), ii) the Euclidean distance of two means of priors (i.e., ?? 0 ??? ?? 1 ) (first row, middle), the KL divergence between DISPLAYFORM1 ) (second row, left-hand side), the MMD distance of q ?? (z | h L , y = 0) and q ?? (z | h L , y = 1) (second row, middle), and the reconstruction loss (second row, right-hand side) across epochs.

In this experiment, we set the dimension of the latent space to 2 to visualize the latent codes of the two classes before and after training.

As shown in FIG1 , the latent codes of the two classes are intermingled before training, whereas, they become more separable and distinct after training.

This shows that our proposed methods discover data representations that support the classification task.

The detection of vulnerabilities in binary code is an important problem in the software industry and in the field of computer security.

In this paper, we leverage recent advances in deep learning representation to propose the Maximal Divergence Sequential Auto-Encoder for binary vulnerability detection.

Specifically, latent codes representing vulnerable and non-vulnerable binaries are encouraged to be maximally different, while still being able to maintain crucial information from the original binaries.

To address the issue of limited labelled public binary datasets for this problem and to facilitate research in the application of machine learning and deep learning to the domain of binary vulnerability detection, we have created a labelled binary software dataset.

Furthermore, our developed tool and approach can be reused to create other high-quality binary datasets.

We conducted extensive experiments to compare our proposed methods with the baselines.

The experimental results show that our proposed methods outperform the baselines in all performance measures of interest.

The process of compiling the VulDeePecker dataset into binaries is divided into three main stages: collecting functions' source code, detecting and fixing source code, and compiling source code to binary functions.

The source code is collected from VulDeePecker GitHub 5 .

This source code involve two types of vulnerability in C/C++ programs: buffer error vulnerability CWE-119 (11,427 files) and resource management error vulnerability CWE-399 (2,088 files).

FIG5 provides an example of a source code file together with its highlighted buffer error vulnerability.

We then use Joern's parser 6 to identify the start and end points of each function in order to recognize the function scope.

After this step, 19,009 non-vulnerable and 12,946 vulnerable functions are detected and obtained.

However, there are a considerable number of functions which are identical to each other.

They are either some common functions that are widely used or some unchanged functions in different versions of a particular source code file.

To address this issue, these identical functions are removed.

Eventually, the numbers of distinctive non-vulnerable and vulnerable functions are 6,412 and 6,592 functions respectively (See FIG2 ).

At the second stage, as these functions are incomplete C/C++ code snippets and cannot be compiled successfully, the errors in source code are required being detected and fixed to generate binaries.

Therefore, we develop an automatic tool based on Joern to detect and fix these functions.

The activity diagram of our tool is described in FIG3 .

In the targeted directory, our automatic tool reads every file (each contains the source code of a function) sequentially.

The process of detecting and fixing each function commences with the preprocessing of source code, which adds some necessary C/C++ libraries and the main function.

It is worth noting that at this step, our tool is able to reformat the source code using the clang-format 7 .

Subsequently, the tool invokes the gcc/g++ (MinGW) compiler to compile the C/C++ source code respectively.

The compiler then captures the error messages and calls the corresponding solver for each specific error.

The semantic and syntactic relationships of the error messages with respect to the source code are mainly analyzed by Joern's parser.

A function's source code is fixed successfully and ready to compile when the compiler cannot issue any error messages in the process of detecting it.

The process of detecting errors and fixing source code also has its own challenges.

As mentioned before, we collect the code snippets of the functions which are always incomplete.

This leads to the missing of the declarations for some objects due to the lack of certain libraries to which those objects belong.

FIG4 refers to a typical example of an uncompilable function.

For that code, when our automatic tool starts detecting the error, the gcc/g++ (MinGW) compiler informs the following error message which needs to be fixed at line 23: 'CWE761 Free Pointer Not at Start of Buffer wchar t environment 34 unionType' has no member named 'unionFirst'.

The error information is then sent to Joern's parser in order to analyze and find the appropriate solution to fix this error.

Unfortunately, the parser is not confident enough to declare the 'unionFirst' variable as a member of the 'CWE761 Free Pointer Not at Start of Buffer wchar t environment 34 unionType' class.

The reason arising from the fact that the 'myUnion.unionFirst' was assigned to the 'data' variable, but the parser cannot give any information about the data type of this variable.

In this situation, the error description is logged into a log file, the execution of the function including this error is skipped, and our tool proceeds to the next function.

After detecting and fixing errors process, we synthesize and do some statistics from the log file to know which errors account for the most popular quantities to upgrade promptly our tool.

We also consider the complexity of functions and the priority order of errors to ensure errors fixed from easiest to hardest.

After each upgrade, our automatic tool is very likely to detect and fix more complex errors to become more completed and stable.

The result we obtain from this stage is 8,991 fixed source code.

It is noticeable that while an original function's source code has 40 errors on average, our automatic tool is able to detect and fix up to 22 and 28 general errors for each C and C++ source code respectively.

At the last stage, we compile 8,991 fixed functions which contain 4,501 non-vulnerable and 4,490 vulnerable functions into binaries under two platforms (Windows and Ubuntu) and architectures (x86/x64).

The process of compilation raises a certain number of small errors due to the behavior inconsistency of gcc/g++ between two platforms which leads to the fact that some fixed functions cannot be compiled.

The total number of binary functions are 32,281 wherein 15,954 functions are non-vulnerable binaries and 16,327 functions are vulnerable binaries.

As the consequence, we utilize the Capstone software to disassemble these binaries into assembly code.

FIG6 shows the assembly code together with its highlighted buffer error vulnerability of the source code file in FIG5 .

Overall, FIG2 shows the details of the stages in VulDeePecker dataset processing, together with the number of vulnerable and non-vulnerable functions obtained at the end of each stage.

Published as a conference paper at ICLR 2019

@highlight

We propose a novel method named Maximal Divergence Sequential Auto-Encoder that leverages Variational AutoEncoder representation for binary code vulnerability detection.

@highlight

This paper proposes a variational autoencoder-based architecture for code embeddings for binary software vulnerability detection, with learned embeddings more effective at distinguishing between vulnerable and non-vulnerable binary code than baselines.

@highlight

This paper proposes a model to automatically extract features for vulnerability detection using deep learning technique. 
