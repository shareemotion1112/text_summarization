We present a framework for automatically ordering image patches that enables in-depth analysis of dataset relationship to learnability of a classification task using convolutional neural network.

An image patch is a group of pixels residing in a continuous area contained in the sample.

Our preliminary experimental results show that an informed smart shuffling of patches at a sample level can expedite training by exposing important features at early stages of training.

In addition, we conduct systematic experiments and provide evidence that CNN’s generalization capabilities do not correlate with human recognizable features present in training samples.

We utilized the framework not only to show that spatial locality of features within samples do not correlate with generalization, but also to expedite convergence while achieving similar generalization performance.

Using multiple network architectures and datasets, we show that ordering image regions using mutual information measure between adjacent patches, enables CNNs to converge in a third of the total steps required to train the same network without patch ordering.

Adva nc e s in Deep Lear nin g (DL) and Conv olu tio na l Neura l Netw or ks (CNN ) have dram atic a l l y impro ve d the state-of-the -ar t in compu te r vision tasks.

Many of these brea kth ro ug h s are attribute d to the succe ssiv e featu re extrac tion and an increa sin g abstr a ct repre se nta tion of the underly ing training dat a using multi-stag e simple oper ation s such as conv olutio n. These opera tion s posse ss seve ra l mod e l para m ete r s such as conv olution filter whic h are traine d to amplif y and refine infor m a tio n that are relev a n t to the classific a tio n, and to suppr e ss irrele v an t infor m atio n (Ian BID8 .

The traini n g proce d u re uses backp ro p a ga tio n algorithm with super vision .

This algorith m comb ine d with Stocha s t i c Gradie nt Desc e nt (SGD ), attem pts to minim iz e the over all erro r or devia tio n from true label by compu ti n g the error grad ien t of each para m e te r and by perfo rm in g small upda te s in the opposite directio n. Desp i t e their succ e ss, theore tic a l char ac te riz ation of deep learnin g and CNN s is still at its infanc y and valua b l e corre latio ns such as numbe r of layer s need ed to achie ve a certain perfo rm a n c e are not well under sto o d .

However, the success of deep learning has spawned many research avenues in order to explain deep network's exceptional generalization performance BID19 BID14 BID16 Tishby and Zaslavsky, 2015) .

One promising theoretical characterization of deep learning that supports an intuition that motivated this work is the characterization that uses an information theoretic view of feature extraction.

In particular it is based on the information bottleneck (IB) method which is concerned with the problem of how one extracts an efficient representation of relevant information contained in a large set of features BID21 .

BID19 proposes to study deep learning through the lens of information theory using the IB principle.

In this characterization, deep learning is modeled as a representation learning.

Each layer of a deep neural network can be seen as a set of summary statistics which contain some of the information present in the training set, while retaining as much information about the target output as possible BID19 .

In this context a relevant information, of a cat vs dog classification task for instance, is the information pattern present in all the cat samples useful for predicting any picture of a cat.

With this view, the amount of information relating the training set and the labels encoded in the hidden layers can be measured over the course of training (Tishby and Zaslavsky, 2015) .

Inspired by this view, we use information theoretic measures of entropy extended to measure image characteristics, to develop preprocessing techniques that enable rob ust features extraction during training.

One relev a nt insigh t prese nte d in these pape r s is that the goal of DL is to captu re and efficie n tly repr e se nt the relev a nt inform a tion in the input varia b le that desc rib e the outp u t variab le.

This is equiv ale nt to the IB meth od whose goal is to find maxim a lly comp re sse d mappin g of the input while prese rvin g as much relev a nt inform ation of the output as possible .

This chara cte riz a ti o n leads us to ask the questio n:

In superv ise d learnin g, we are intere sted in good featu re repre se nta tio n s of the input patte rn that ena b l e good predictio n of the label BID9 .

As a result, a training set for ima g e classific a tio n tasks that employ superv ise d learnin g, is constr uc te d with the help of huma n labele r .

For instan ce , for a cat vs dog classifica tion proble m , the huma n labele r must cate go riz e each sample into eithe r one of the classe s. Durin g this proce ss, the labele r must recog niz e and classify each input usin g their own expe rie nc e and distin guis hing capa b ilitie s. Considering this, a natural question we first must answer before addressing the question above is:

In other word s, can the netw or ks learn from 'scra m ble d ' sample s that cann ot be classifie d by the nak e d eye?

This questio n was investig a ted in BID6 with intrig uin g outco m e s. The aut h o r s prese nte d results that indicate that CNN s are capa ble of easily fitting trainin g set conta inin g sample s that have no corr ela tio n with labels (see Fig. 3 for illustra tio n ).

These results have us recon side r the traditi o n a l view that netw or ks build hiera rc hy of featu re s in incre asin g abstr a ction , i. (2016) and in this pape r (see sectio n V for detail) .

We use the infor m atio n theor etic chara cte riz a tio n of deep learnin g to shed light on the question s by deve lo p i n g prepr oc e ssin g and learnin g techniq u es that reduc e conv er ge n c e time by improv ing featu re s extra c t i o n from imag e s using multila ye r e d CNN s.

We first rule out that human reco gn iza b le featur e s matc hin g lab e l s are not nece ssa r y for CNN s and that they are able to fit trainin g set contain ing scram b led samp les wit h minim a l impa ct on gene ra liza tion .

Equip p ed with this result we then utilize simila rity and inform a t i o n theore tic measur es of imag e char a cte ristic s to prep ro c e ss and ease featu re extra c tio n from image s dur i n g training .

Our methods aim to expose important features of each training sample earlier in training by reorganizing image regions.

The contrib ution of our appro ac h are:1.

We provid e a framework and algorithm s for prepro c e ssing datase t to reorder image patches usin g techniq ue s that minim iz e mutual entro p y of adjacent image patches of each training sample.

As the results demon stra te , orga niz ing patch es , of each training sample using measures such as entropy of a patch and mutual inform a tion index betwe e n patch es enable faste r conv e rg e nc e .2.

We prese nt several techniqu e s for rankin g samp les that use inform a tio n theore tic measur e s of the relatio nsh ip betwe e n adjac e nt patche s and prese nt results that show faste r conv e rg en c e comp ar e d to stand a rd training .Inception (Szegedy et al., 2015) architecture, known for achieving exceptional results on image classification tasks, is used for evaluation.

The network is first evaluated on the corresponding datasets to create baseline reference performance metrics for comparison.

For each network we used Adams optimization technique with cross -entropy loss to gather emperical training, validation and test data.

The remaining content is presented as follows.

In section 2, we present the patch ordering approach and highlight the design and implementation of algorithms used to preprocess data and feature maps based on patch ordering.

In Section 3, we discuss the experimental setup.

Then, section 4 presents analysis of our results obtained by training Inception using multiple unmodified and patch-ordered datasets.

Finally, we conclude by offering our insight as to why the outcomes are important for deep learning and future generation networks.

The succe ss of CNNs stem from their ability to autom a tic a lly learn featur e extra cto rs .

Durin g traini n g , CNN s constr uc t hierar c hy of featu re repr e se ntatio ns and use super po sitio n of the hiera rc hic al featur es when gene r alizin g to unse en input BID8 .

How e v er , we belie ve learnability of a classific a tio n task is close ly relate d to the amoun t of infor m a tio n conta ine d in the datas et that ena b l e disting uish a bility of one class from the others.

To furth e r explo re this claim , we developed techniques and condu cte d seve ral expe rim e nts by prep ro ce ssin g training set using vario u s techniq u es .

Th e techniq ue s and the gene ral proce d ur e used are describ e d below .

The results are summ a riz e d in sectio n 4.

Our intuitio n is that some order ing at a sample level can exped ite trainin g by expo sing featu re s that are importa nt for sepa ra tin g the classe s in the early stage s of training .For illustra tion , consid e r the toy image s in Fig. 1 .

If a person with know le dg e of the numb e r syste m , wa s asked to classify or label the two image s, they can give sever al answ e rs depe nd ing on their expe rie n c e s .

At first glanc e , they can label a) as 'larg e numb e r 1' and b) as 'larg e numb e r 2'.

If they were asked to give more details, upon elabor ation of the conte xt, the labeler can quick ly scan a) and realiz e that it is a pictur e of digits 0 throu gh 9.

Simila rly , b) would be classified as such, but analyz in g and classifyin g b) can cost more time beca us e the labele r must ensure ever y digit is prese nt (we enco ur a ge the reade rs to do the expe rim en t).

It's the time cost that is of intere st to us in the conte xt of learnin g system s. The mer e order ing of the numb e rs enable s the labeler to classif y a) faste r than b).Given this intuitio n , we aske d if orde ring patche s of trainin g imag es such that the adjac e nt patch e s are 'close r' to each other by simila rity measur e , could expe dite training and improv e gene ra liz a tion .

Based on the menta l exer cise , the proce d ur e can intuitive ly be justifie d by the fact that toy sample a) is easie r to classif y beca use , as our eyes scan from left to right the featu re s (0,1,2 . . .) are captur ed in orde r. Whe r e a s it might take sever al scan s of b) to deter min e the same outcom e .

Convo lution based featur e extra cto rs use a similar conc e pt to captu re featu re s used to disting uish one class from the others.

The featu re s are extrac te d by scan nin g the input imag e using conv olution filter s.

The output of conv olution at each spati a l locatio n are then stack e d to constru ct the featu re map.

Imple m e nta tion of this oper ation in most dee p learnin g frame w or ks maintain spat ia l locatio ns of featur e s whic h then can be obtain e d by deco n vo luti o n .

In other word s, there is a one-to-one mappin g betw e e n the locatio n of a featu re in a feature map and its locatio n on the origina l input (Fig.2 .) .

Note that the featur e map not only encod e s the featu re (ear or hea d ) but it also implic itly encod e s the locatio n of the featu re on the input imag e (gree n arro w in Fig. 2.) .

Th e enco din g of locatio n is requir e d for detectio n and localiz atio n tasks but not for classific a tio n tas k s .

Another questio n that arise s from these observ a tio ns is:

To answ e r this questio n, we searched for DL characterization that aligns with this intuition and found the work of Tishby and Zaslavsky (2015) captures this intuition by relating DL training from images to the Information Bottleneck principle (discussed below).

While the authors discuss IB in the context of the entire training set and end-to-end training of deep networks, our exploration is limited to individual training samples and aim to expose information that can be captured and presented to the network.

We deve lop e d techn iqu e s to reco nstr uc t training image s by brea king up the inputs into equal sized pat c h e s and reco n str uc t them using the conce pt of orde ring (Fig.3 ) .

Infor m atio n-th e or y-ba se d and tradition a l 1 2 measu re s of imag es were used for ranking and orde ring .

These measu re s can gene r ally be divide d into two:1.

Standalone measures -mea su re some char a cte ristic of a patch.

For exam ple , the peak signal-tonoise ratio meas ur e retur ns a ratio betw e en maxim u m useful signal to the amoun t of noise prese nt in a patch .2.

Similarity measures -these measures on the other hand, compare a pair of patch e s.

The comp a r i s o n measu re s can be measur e s of simila rity or dissim ila rity like L1-no rm and structu ral simila rity or infor m atio n-th e or etic -m ea su re s that comp a re distrib ution of pixel value s such as joint entrop y. Th e measu re s discu sse d in subsec tion s below are L1-n o rm , L2-n o rm , Struc tur al Similarity , Joint Entr o p y , KL -D iv e rg en c e , and Mutua l Infor m atio n.

Below we summarize the measures and present the sorting and recon struction algorithm.

The results are summarized in Section 4.

Information theory provides a theoretical foundation to quantify information content, or the uncertainty, of a random variable represented as a distribution BID2 BID3 .

Information theoretic measures of content can be extended to image processing and computer vision (Leff and Rex, 1990) .

One such measure is entropy.

Intuitively, entropy measures how much relevant information is contained within an image when representing an image as a discrete information source that is random BID3 .

Formally, let X be a discrete random variable with alphabet and a probability mass function ( ) , ∈ .

The Shannon entropy or information content of is defined as DISPLAYFORM0 where 0log ∞ = 0 and the base of the logarithm determines the unit, e.g. if base 2 the measure is in bits etc.

BID2 .

The term 1 ( ) can be viewed as the amount of information gained by observing the outcome ( ).

This measure can be extended to analyze images as realizations of random variables BID3 .

A simple model would assume that each pixel is an independent and identically distributed random variable (i.i.d) realization BID3 .

When dealing with discrete images, we express all entropies with sums.

One can obtain the probability distribution associated with each image by binning the pixel values into histograms.

The normalized histogram ca n be used as an estimate of the underlying probability of pixel intensities, i.e., ( ) = ( )/ , where ( ) denotes the histogram entry of intensity value in sample and is the total number of pixels of .

With this model the entropy of an image can be computed using: DISPLAYFORM1 Figure 3.

An illustration of patch ordering.

a) Input image, b) reconstruction of the input using structural similarity of patches and c) feature map generated by convolving b).

Note that the encoding of spatial location of a feature is not present in the feature map.

is reconstructed .

where = {( , ) : 1 ≤ ≤ } is the training set comprising both the input values and corresponding desired output values .

N is the total number of examples in the training set. ( ) represents the image as a vector of pixel values.

While individual entropy is the basic index used for ordering, we also consider strategies that relate two image patches.

These measures include joint entropy BID3 ), kl-divergence (Szeliski, 2010 , and mutual information BID18 .

By considering two random variables ( , ) as a single vector-valued random variable, we can define the joint entropy ( , ) of pair of variables with joint distribution ( , ) as follows: DISPLAYFORM0 When we model images as random variables, the joint entropy is computed by gathering joint histogram between the two images.

For two patches, 1 , 2 ∈ ∈ the joint entropy is given by: DISPLAYFORM1 where ( ) is the ℎ value of joint histogram between the two patches.

DISPLAYFORM2 Mutual information (MI) is the measure of the statistical dependency between two or more random variables BID3 .

The mutual information of two random variables and can be defined in terms of the individual entropies of both and and the joint entropy of the two variables ( , ) .

Assuming pixel values of the patches 1 , 2 the mutual information between the two patches is DISPLAYFORM3 As noted in BID18 , maximizing the mutual information between patches seems to try and find the most complex overlapping regions by maximizing the individual entropies such that they explain each other well by minimizing the joint entropy.

As image similarity measure, MI has been found to be successful in many application domains .

DISPLAYFORM4 K-L Divergence is another measure we use to assess similarity of patches with in a sample.

It's a natural distance measure from a pixel distribution 1 to another distribution 2 and is defined as: DISPLAYFORM5 where the index of a pixel value taken from the distributions.

Given two equal sized vectors and representing two patches of an image, the 1 distance BID15 is defined as DISPLAYFORM0 This is sum of lengths between corresponding pixel value at index i over the size of the patch.

L2 norm is a common measure used to assess similarity between images.

DISPLAYFORM0 This can be interpreted as the Euclidean distance between the two vectors 1 and 2 representing the patches BID15 .

SSIM is usually used for predicting image quality using a reference image.

Given two vectors and the SSIM index BID7 ) is given by:( 2 ) = (2µ 1 2 + 1 )(2 1 2 + 2 ) (µ 1 2 + µ 2 2 + 1 )( 1 2 + 2 2 + 2 )where the terms µ and are the mean and variances of the two vectors and 1 2is the covariance of 1 and 2 .

See BID7 for detail on this measure.

PSNR BID7 ) is another objective metric widely used in CODECs to assess picture quality.

PSNR can be defined in terms of the mean squared error (MSE).

The MSE of two metrices having the same size N is defined as: DISPLAYFORM0 The PSNR measure of two patches 1 and 2 can then be expressed as: DISPLAYFORM1 where is the maximu m possible pixel value of the reference patch 1 .

For evaluation we used CATSvsDOGS BID17 and CIFAR100 BID10 ).

The techniques described above along with the several network architectures, were employed to learn and classify these datasets.

To gather enough data that enable characterization of each preprocessing technique, we set up a consistent training environment with fixed network architecture s, training procedure, as well as hyper parameters configuration.

The results are summarized in section 4.

We performed two sets of experiments to determine the impacts of algorithm POR TAB2 on training.

The first experiment was designed to determine correlation between the preprocessing techniques and network training performance while the second was conducted to characterize the impact of granularity of patches on training.

Below we present the analysis of results obtained using each approach.

The results are summarized in Figs. 4 and 5.

FIG1 shows results obtained when training Inception network to classify CIFAR100 (Top) and Cats vs Dogs (Bottom) datasets using slow learning rate and Adams optimization BID9 .

Plots on the right side depict test performance of the network at different iterations.

In both setups, the mutual information technique speeds up learning rate more than all others while some techniques degrade the learning rate compared to regular training.

However, all techniques converge to the same performance as the regular training when trained for 10000 iterations.

Given these results we answer the questions posed in the earlier sections.

The question of whether ordering patches of the input based on some measure to help training can partially be answered by the empirical evidence that indicate reconstructing the input using the MI measure enables faster convergence.

Dataset reordered using the MI measure achieves similar accuracy as the unmodified dataset in ¼ of the total iterations.

In support of this we hypothesize that informed ordering techniques enable robust feature extraction and make learning efficient.

To conclusively prove this hypothesis, one must consider variety of experimental setup.

For instance, to rule out other factors for the observed results, we must perform similar experiments using different datasets, learning techniques, hyper parameter configuration and network architectures.

Given that most of these techniques remove human recognizable features by reordering (Figure 3 ) and the experimental results that not all ordering techniques compromise training or testing accuracy, we make the following claim:Training and generalization performance of classification networks based on the deep convolutional neural network architecture is uncorrelated with human ability to separate the training set into the various classes.

In this section we provide analysis of the impact of the patch -ordering preprocessing technique on training convolutional neural networks.

Let us consider the mutual information (MI) metric, which outperforms all other metric s. As mentioned in previous sections the MI index is used as a measure of statistical dependency between patches for patch ordering.

Given two patches (also applies to images) 1 , 2 the mutual information formula (Eqn.

5) computes an index that describes how well you can predict 2 given the pixel values in 1 .

This measures the amount of information that image 1 contains about 2 .

When this index is used to order patches of an input, the result consists of patches ordered in descending order according to their MI index.

For instance, consider a 32 by 32 wide image with sixteen 8 by 8 patches (see representation, I, below).

If we take patch (0,0) to be the reference patch, Algorithm 1 in the first iteration computes MI index of every other patch with the reference patch and moves the one with the highest index to position (0,1) and updates the reference patch.

At the end, the algorithm generates an image such that the patch at (0,0) has more similarity to patch at (0,1) which has more similarity to patch at (0,2) etc.

In other words, adjacent patches explain each other well more than patches that are further away from each other.

To answer this question let us consider the convolution operator BID4 and the gradient decent optimization BID0 approach.

This algorithm employs Adam optimizatio n technique and the SoftMax cross -entropy loss, to update network parameters.

We trained the networks using online training BID13 mechanism, where error calculations and weight updates occur after every sample.

Our hypothesis is that samples preprocessed using the MI measure enable rapid progress lowering the cost in the initial stages of the training.

In other words, when the input is rearranged such that adjacent samples have similar pixel value distribution, the convolution filters extract features that produce smaller error.

To illustrate this let us assume the following values for the first few patches of an image (color coded in the matrix below).

For simplicity let us assume the image is binary and all the pixel values are either 0 or 1.

To maintain resolution of the original image we use 0-padding before applying convolution.

Applying a 3x3 max pooling operation with stride 3 to the convolved sample generates a down-sampled feature-map of the ℎ training sample which is used as an input to compute probability score of each class in a classifier.

In this illustration we consider a binary classifier with two possible outcomes.

SoftMax cross-entropy loss for the correct class can be computed using the normalized probabilities assigned to the correct label given the image parameterized by (Eqn.

12).

DISPLAYFORM0 The probabilities of each class using ( , ) = ( + ) objective function after normalization are DISPLAYFORM1 Assuming the probability of the correct class is 0.01 the cross-entropy loss becomes 4.60.Note here patches are ordered left to right and adjacent patches have MI indices that are larger than those that are not adjacent.

After ranking each 3x3 patch using the MI measure and preprocessing the sample using Algorithm 1, the resulting sample ′ has ordering grey, green, pink and blue.

In this example MI of the green with the grey patch is 0.557 while the blue has MI index equal to 0.224 against the same reference patch.

in a prediction loss equal to 2.01.

This is the underlying effect we would like all measure to have when reordering the training dataset.

However, it is not guaranteed.

For instance, if we use l2-norm measure (Eqn. 8) to sort the patches, the resulting loss becomes 4.71, which is higher compared to the unmodified original sample.

As a result, the training is slowed down since larger loss means more iterations are required for the iterative optimizatio n to converge.

To characterize the effect of patch size, we performed controlled experiments where only the patch size is the varying parameter.

The results and unmodified and preprocessed samples are depicted in FIG5 .

As can clearly be seen in the plot, the network makes rapid pro gress lowering the cost when trained on a 4x4 patch ordered datasets.

Based on the empirical evidence and observations, we believe patch-ordering impact is more effective when mutual information index is combined with small patch size.

To clarify consider dividing the above sample into nine 2x2 patches.

This is one explanation for the observed results, however, we cannot draw a conclusion regarding proportionality of patch size to training performance.

If the pink and red patches of the above sample, which have same MI index, were to swap places, the resulting loss would have been 4.71 which is greater than the loss generated using 3x3 patch size.

In this scenario training is slowed down.

We proposed several automated patch ordering techniques to assess their impact on training and assess the relationship between dataset characteristics and training and generalization performances .

Our methods rank, and reorder patches of every sample based on a standalone meas ure and based on similarit y between patches.

We used traditional image similarity measures as well as information theory -based content measures of images to reconstruct training samples.

We started off with theoretical foundations for measures used and highlighted the intuition regarding ordering and classification performance.

We tested the proposed methods using several architectures, each effectively designed to achieve high accuracy on image classification tasks.

The empirical evidence and our analysis using multiple datasets and Inception network architecture, suggest that training a convolutional neural network by supplying inputs that have some ordering, at patch level, according to some measure, are effective in allowing a gradient step to be taken in a direction that minimizes cost at every iteration.

Specifically, our experiment s and CIFAR100 (right) datasets.

Total training loss (top) and regularization loss (bottom) for Unmodified dataset, and datasets modified by applying Algorithm 1 using the MI metric and patch sizes 4x4, 8x8 and 16x16).

The overall size of each sample is 32 by 32.show that supplying training sample such that the mutual information between adjacent patches is minimum, reduces the loss faster than all other techniques when optimizing a non-convex loss function.

In addition, using these systematic approaches, we have shown that image characteristics and human recognizable features contained within training samples are uncorrelated with network performance.

In other words, the view that CNNs learn combination of features in increasing abstraction does not explain their ability to fit images that have no recognizable features for the human eyes.

Such a view also discounts the ability of the networks to fit random noise during training .

Instead further investig a t i o n using theore tic a l chara cte riz a tio n s such as the IB metho d are nece ssa ry to form ally char a cte riz e learn ab il i t y of a given trainin g set using CNN .

A typical CNN arch ite c tur e is struc tur ed as a serie s of data proce ssin g and classific ation stage s.

It consi s t s of severa l layers with tens of thousa nd s of neur on s in each layer , as well as millio ns of conne c tio n s bet w e e n neuro n s. In the data proce ssin g stages , there are two kinds of layer s: convo lution al and poolin g layer s (Ian BID8 .

In a convo lutio n al layer, each neur on repre se ntin g a filter is conne c te d to a sma ll patch of a featur e map from the previo us layer throu g h a set of weig hts.

The result of the weigh te d sum is then passed throu g h an activ ation functio n that perfo rm s non-lin e a r transfo rm ation and prev e nt lear n i n g trivia l linea r combin ation s of the inputs.

The poolin g layers are used to reduc e comp uta tion time by subsamp lin g from conv olutio n outputs and to gradu ally build up furth er spatial and configu ra l invarian c e (Ian BID8 .Discr ete image convolu tio n [10] is used to extra ct inform ation from trainin g sample s.

For 2D function s I and K, the conv olu tio n opera tio n is define d as: DISPLAYFORM0 In CNN s, conv olutio n at a given layer is applie d to the output of the previou s layer and the limits of the summ a tio n are determ in e d by the size the input I and of the filter K. For a given layer , the input comprises ( ) 1 −1 feature maps from the previous layer BID5 .

When = 1, the input is a single image consisting of one or more channels.

The output of layer consists of ( ) 1 feature maps.

where ( −1 ) is the total number of feature maps generated by the previous layer, is a bias matrix and , is a filter connecting the ℎ feature map in layer BID8 .

The trainable weights are found in the filters , and the bias matrices .

During training, CNNs attempt to determine the filter weights to approximate target mapping BID5 .

In practice, is a function fitted by the training data using supervised training procedures.

The training set = {( , ) : 1 ≤ ≤ }comprises both the input values and corresponding desired output values ≈ ( ) .

N is the total number of examples in the training set.

Supervised training is accomplished by adjusting the weights of the network to minimize a chosen objective function that measures the deviation of the network output, , from the desired target output BID5 .

Some of the common measures are cross -entropy error measure BID9 given by DISPLAYFORM0 and the squared-error measure BID1 given by ( ) = ∑ ( ) =1 = ∑ ∑ , log( ( , )) ,=1 FORMULA1 where , is the ℎ entry of the target value and c is the number of distinct classes in .Deep learning with stochastic training seeks to minimize ( w ) with respect to the network weights .

The necessary criterion can be written as DISPLAYFORM1 where is the gradient of the error BID5 .

Since is a high dimensional function, a closed-form exact solution is too expensive.

Iterative optimization approach, commonly referred to as gradient descent (Algorithm 1), is used to find optimal values of the parameters that best approximate a mapping between each sample in the training set to the desired output.

At each iteration, for a given weight vector [ ] , gradient descent takes a step in the direction of the steepest descent to reach a global minimum BID5 There are few different training protocols used for parameter optimization.

These protocols are summarized in (Larochelle et al., n.d.) .

The most common ones are: Stochastic training: when this protocol is employed, an input sample is chosen at random and the network weights are updated based on the error function ( ) .

Figure 1 .

Illustration of convolution in a single convolution layer.

If layer l is a convolutional layer, the input image (if l = 1) or a feature map of the previous layer is convolved by different filters to yield the output feature maps, ( ) , of layer l.

@highlight

Develop new techniques that rely on patch reordering to enable detailed analysis of data-set relationship to training and generalization performances.