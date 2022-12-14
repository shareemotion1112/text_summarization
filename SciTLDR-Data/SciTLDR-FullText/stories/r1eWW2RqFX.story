A point cloud is an agile 3D representation, efficiently modeling an object's surface geometry.

However, these surface-centric properties also pose challenges on designing tools to recognize and synthesize point clouds.

This work presents a novel autoregressive model, PointGrow, which generates realistic point cloud samples from scratch or conditioned from given semantic contexts.

Our model operates recurrently, with each point sampled according to a conditional distribution given its previously-generated points.

Since point cloud object shapes are typically encoded by long-range interpoint dependencies, we augment our model with dedicated self-attention modules to capture these relations.

Extensive evaluation demonstrates that PointGrow achieves satisfying performance on both unconditional and conditional point cloud generation tasks, with respect to fidelity, diversity and semantic preservation.

Further, conditional PointGrow learns a smooth manifold of given images where 3D shape interpolation and arithmetic calculation can be performed inside.

3D visual understanding BID2 ; BID35 ) is at the core of next-generation vision systems.

Specifically, point clouds, agile 3D representations, have emerged as indispensable sensory data in applications including indoor navigation BID6 ), immersive technology BID20 ; BID21 ) and autonomous driving ).

There is growing interest in integrating deep learning into point cloud processing BID10 ; BID13 ; BID1 ; BID11 ; BID32 ; BID29 ).

With the expressive power brought by modern deep models, unprecedented accuracy has been achieved on high-level point cloud related tasks including classification, detection and segmentation BID16 BID5 ; BID25 ; ; BID31 ).

Yet, existing point cloud research focuses primarily on developing effective discriminative models BID29 ; BID19 ), rather than generative models.

This paper investigates the synthesis and processing of point clouds, presenting a novel generative model called PointGrow.

We propose an autoregressive architecture ; ) to accommodate the surface-centric nature of point clouds, generating every single point recurrently.

Within each step, PointGrow estimates a conditional distribution of the point under consideration given all its preceding points, as illustrated in Figure 1 .

This approach easily handles the irregularity of point clouds, and encodes diverse local structures relative to point distance-based methods BID7 ; BID1 ).However, to generate realistic point cloud samples, we also need long-range part configurations to be plausible.

We therefore introduce two self-attention modules BID14 ; BID24 ; BID34 ) in the context of point cloud to capture these long-range relations.

Each dedicated self-attention module learns to dynamically aggregate long-range information during the point generation process.

In addition, our conditional PointGrow learns a smooth manifold of given images where interpolation and arithmetic calculation can be performed on image embeddings.

Compared to prior art, PointGrow has appealing properties:??? Unlike traditional 3D generative models that rely on local regularities on grids BID26 BID5 BID30 ; BID22 ), PointGrow builds upon DISPLAYFORM0 The point cloud generation process in PointGrow (best viewed in color).

Given i ??? 1 generated points, our model first estimates a conditional distribution of z i , indicated as p(z i |s ???i???1 ), and then samples a value (indicated as a red bar) according to it.

The process is repeated to sample y i and x i with previously sampled coordinates as additional conditions.

The i th point (red point in the last column) is obtained as DISPLAYFORM1 autoregressive architecture that is inherently suitable for modeling point clouds, which are irregular and surface-centric.??? Our proposed self-attention module successfully captures the long-range dependencies between points, helping to generate plausible part configurations within 3D objects.??? PointGrow, as a generative model, enables effective unsupervised feature learning, which is useful for recognition tasks, especially in the low-data regime.

Extensive evaluations demonstrate that PointGrow can generate realistic and diverse point cloud samples with high resolution, on both unconditional and conditional point cloud generation tasks.

In this section, we introduce the formulation and implementation of PointGrow, a new generative model for point cloud, which generates 3D shapes in a point-by-point manner.

Unconditional PointGrow.

A point cloud, S, that consists of n points is defined as S = {s 1 , s 2 , ..., s n }, and the i th point is expressed as s i = {x i , y i , z i } in 3D space.

Our goal is to assign a probability p(S) to each point cloud.

We do so by factorizing the joint probability of S as a product of conditional probabilities over all its points: DISPLAYFORM0 The value p(s i |s ???i???1 ) is the probability of the i th point s i given all its previous points, and can be computed as the joint probability over its coordinates: DISPLAYFORM1 , where each coordinate is conditioned on all the previously generated coordinates.

To facilitate the point cloud generation process, we sort points in the order of z, y and x, which forces a shape to be generated in a "plane-sweep" manner along its primary axis (z axis).

Following and , we model the conditional probability distribution of each coordinate using a deep neural network.

Prior art shows that a softmax discrete distribution works better than mixture models, even though the data are implicitly continuous.

To obtain discrete point coordinates, we scale all point clouds to fall within the range [0, 1], and quantize their coordinates to uniformly distributed values.

We use 200 values as a trade-off between generative performance and minimizing quantization artifacts.

Other advantages of adopting discrete coordinates include (1) simplified implementation, (2) improved flexibility to approximate any arbitrary distribution, and (3) it prevent generating distribution mass outside of the range, which is common for continuous cases.

Context Awareness Operation.

Context awareness improves model inference.

For example, in BID16 and BID25 , a global feature is obtained by applying max pooling along each feature dimension, and then used to provide context information for solving semantic segmentation tasks.

Similarly, we obtain "semi-global" features for all sets of available points in the point cloud generation process, as illustrated in FIG0 (left).

Each row of the resultant features aggregates the context information of all the previously generated points dynamically by fetching and averaging.

This Context Awareness (CA) operation is implemented as a plug-in module in our model, and mean pooling is used in our experiments.

Self-Attention Context Awareness Operation.

The CA operation summarizes point features in a fixed way via pooling.

For the same purpose, we propose two alternative learning-based operations to determine the weights for aggregating point features.

We define them as Self-Attention Context Awareness (SACA) operations, and the weights as self-attention weights.

The first SACA operation, SACA-A, is shown in the middle of FIG0 .

To generate self-attention weights, the SACA-A first associates local and "semi-global" information by concatenating input and "semi-global" features after CA operation, and then passes them to a Multi-Layer Perception (MLP).

Formally, given a n ?? f point feature matrix, F, with its i th row, f i , representing the feature vector of the i th point for 1 ??? i ??? n, we compute the i th self-attention weight vector, w i , as below: DISPLAYFORM2 Under review as a conference paper at ICLR 2019 , where M ean{??} is mean pooling, ??? is concatenation, and M LP (??) is a sequence of fully connected layers.

The self-attention weight encodes information about the context change due to each newly generated point, and is unique to that point.

We then conduct element-wise multiplication between input point features and self-attention weights to obtain weighted features, which are accumulated sequentially to generate corresponding context features.

The process to calculate the i th context feature, c i , can be expressed as: DISPLAYFORM3 , where ??? is element-wise multiplication.

Finally, we shift context features downward by one row, because when estimating the coordinate distribution for point, s i , only its previous points, s ???i???1 , are available.

A zero vector of the same size is attached to the beginning as the initial context feature, since no previous point exists when computing features for the first point.

Figure 2 (right) shows the other SACA operation, SACA-B. SACA-B is similar to SACA-A, except the way to compute and apply self-attention weights.

In SACA-B, the i th "semi-global" feature after CA operation is shared by the first i point features to obtain self-attention weights, which are then used to compute c i .

This process can be described mathematically as: DISPLAYFORM4 Compared to SACA-A, SACA-B self-attention weights encode the importance of each point feature under a common context, as highlighted in Eq. (4) and (5).Learning happens only in MLP for both operations.

In FIG1 , we plot the attention maps, which visualize Euclidean distances between the context feature of a selected point and the point features of its accessible points before SACA operation.

Model Architecture.

FIG2 shows the proposed network model to output conditional coordinate distributions.

The top, middle and bottom branches model p(z i |s ???i???1 ), p(y i |s ???i???1 , z i ) and p(x i |s ???i???1 , z i , y i ), respectively, for i = 1, ..., n. The point coordinates are sampled according to the estimated softmax probability distributions.

Note that the input points in the latter two cases are masked accordingly so that the network cannot see information that has not been generated.

During the training phase, points are available to compute all the context features, thus coordinate distributions can be estimated in parallel.

However, the point cloud generation is a sequential procedure, since each sampled coordinate needs to be fed as input back into the network, as demonstrated in Figure 1 .Conditional PointGrow.

Given a condition or embedding vector, h, we hope to generate a shape satisfying the latent meaning of h. To achieve this, Eq. (1) and (2) are adapted to Eq. (6) and FORMULA8 , respectively, as below: DISPLAYFORM5 Figure 5: Generated point clouds for different categories from scratch by unconditional PointGrow.

From left to right for each set: table, car, airplane, chair and lamp.

DISPLAYFORM6 The additional condition, h, affects the coordinate distributions by adding biases in the generative process.

We implement this by changing the operation between adjacent fully-connected layers from DISPLAYFORM7 , where x i+1 and x i are feature vectors in the i + 1 th and i th layer, respectively, W is a weight matrix, H is a matrix that transforms h into a vector with the same dimension as Wx i , and f (??) is a nonlinear activation function.

In this paper, we experimented with h as an one-hot categorical vector which adds class dependent bias, and an high-dimensional embedding vector of a 2D image which adds geometric constraint.

Datasets.

We evaluated the proposed framework on ShapeNet dataset BID3 ), which is a collection of CAD models.

We used a subset consisting of 17,687 models across 7 categories.

To generate corresponding point clouds, we first sample 10,000 points uniformly from each mesh, and then use farthest point sampling to select 1,024 points among them representing the shape.

Each category follows a split ratio of 0.9/0.1 to separate training and testing sets.

ModelNet40 BID27 ) and PASCAL3D+ BID28 ), are also used for further analysis.

ModelNet40 contains CAD models from 40 categories, and we obtain their point clouds from BID16 .

PASCAL3D+ is composed of PASCAL 2012 detection images augmented with 3D CAD model alignment, and used to demonstrate the generalization ability of conditional PointGrow.

Figure 5 shows point clouds generated by unconditional PointGrow.

Since an unconditional model lacks knowledge about the shape category to generate, we train separate models for each category.

Figure 1 demonstrates point cloud generation for an airplane category.

Note that no semantic information of discrete coordinates (i.e.scattered points in point cloud) is provided during training, but the predicted distribution turns out to be categorically representative.

(e.g.in the second row, the network model outputs a roughly symmetric distribution along X axis, which describes the wings' shape of an airplane.)

The autoregressive architecture in PointGrow is capable of abstracting highlevel semantics even from unaligned point cloud samples.

Evaluation on Fidelity and Diversity.

The negative log-likelihood is commonly used to evaluate autoregressive models for image and audio generation ; ).

However, we observed inconsistency between its value and the visual quality of point cloud generation.

It is validated by the comparison of two baseline models: CA-Mean and CAMax, where the SACA operation is replaced with the CA operation implemented by mean and max pooling, respectively.

In FIG3 (left), we report negative log-likelihoods in bits per coordinate on ShapeNet testing sets of airplane and car categories, and visualize their representative results.

Despite CA-Max shows lower negative log-likelihoods values, it gives less visually plausible results (i.e. airplanes lose wings and cars lose rear ends). (SACA-B) 89.4 91.9Table 2: The comparison on classification accuracy between our models and other unsupervised methods on ModelNet40 dataset.

All the methods train a linear SVM on the high-dimensional representations obtained from unsupervised training.

Methods Accuracy SPH BID9 68.2 LFD BID4 75.2 T-L Network BID8 74.4 VConv-DAE BID18 75.5 3D-GAN BID26 83.3 Latent-GAN-EMD BID0 84.0 Latent-GAN-CD BID0 84.5 Ours (SACA-A) 85.8 Ours (SACA-B) 84.4To faithfully evaluate the generation quality, we conduct user study w.r.t.two aspects, fidelity and diversity, among CA-Max, CA-Mean and PointGrow (implemented with SACA-A).

We randomly select 10 generated airplane and car point clouds from each method.

To calculate the fidelity score, we ask the user to score 0, 0.5 or 1.0 for each shape, and take the average of them.

The diversity score is obtained by asking the user to scale from 0.1 to 1.0 with an interval of 0.1 about the generated shape diversity within each method.

8 subjects without computer vision background participated in this test.

We observe that (1) CA-Mean is more favored than CA-Max, and FORMULA3 our PointGrow receives the highest preference on both fidelity and diversity.

Evaluation on Semantics Preserving.

After generating point clouds, we perform classification as a measure of semantics preserving.

Original meshes are used for training while the generated point clouds are used for testing.

More specifically, after training on ShapeNet training sets, we generated 300 point clouds per category (2,100 in total for 7 categories), and conducted two classification tasks: one is training on original ShapeNet training sets, and testing on generated shapes; the other is training on generated shapes, and testing on original ShapeNet testing sets.

PointNet BID16 ), a widely-uesd model, was chosen as the point cloud classifier.

We implement two GAN-based competing methods, 3D-GAN BID26 ) and latent-GAN BID0 ), to sample different shapes for each category, and also include CA-Max and CA-Mean for comparison.

The results are reported in TAB0 .

Note that the CA-Mean baseline achieves comparable performance against both GAN-based competing methods.

In the first classification task, our SACA-A model outperforms existing models by a relatively large margin, while in the second task, SACA-A and SACA-B models show similar performance.

Unsupervised Feature Learning.

We next evaluate the learned point feature representations of the proposed framework, using them as features for classification.

We obtain the feature representation of a shape by applying different types of "symmetric" functions as illustrated in Qi et al. (2017a) (i.e. min, max and mean pooling) on features of each layer before the SACA operation, and concatenate them all.

Following BID26 , we first pre-train our model on 7 categories from the ShapeNet dataset, and then use the model to extract feature vectors for both training and testing shapes of ModelNet40 dataset.

A linear SVM is used for feature classification.

We report our best results in Table 2 .

SACA-A model achieves the best performance, and SACA-B model performs slightly worse than Latent-GAN-CD BID0 uses 57,000 models from 55 categories of ShapeNet dataset for pre-training, while we use 17,687 models across 7 categories).Shape Completion.

Our model can also perform shape completion.

Given an initial set of points, our model is capable of completing shapes multiple ways.

Figure 7 visualizes example predictions.

The input points are sampled from ShapeNet testing sets, which are not seen during the training Figure 7 : Shape completion results generated by PointGrow.process.

The shapes generated by our model are different from the original ground truth point clouds, but look plausible.

A current limitation of our model is that it works only when the input point set is given as the beginning part of a shape along its primary axis, since our model is designed and trained to generate point clouds along that direction.

More investigation is required to complete shapes when partial point clouds are given from other directions.

SACA-A is used to demonstrate conditional PointGrow here owing to its high performance.

Conditioned on Category Label.

We first experiment with class-conditional modelling of point clouds, given an one-hot vector h with its nonzero element h i indicating the i th shape category.

The one-hot condition provides categorical knowledge to guide the shape generation process.

We train the proposed model across multiple categories, and plausible point clouds for desired shape categories can be sampled (as shown in FIG4 ).

Failure cases are also observed: generated shapes present interwoven geometric properties from other shape types.

For example, the airplane misses wings and generates a car-like body; the lamp and the car develop chair leg structures.

Conditioned on 2D Image.

Next, we experiment with image conditions for point cloud generation.

Image conditions apply constraints constrains to the point cloud generation process because the geometric structures of sampled shapes should match their 2D projections.

In our experiments, we obtain an image condition vector by adopting the image encoder in BID22 to generate a feature vector of 512 elements, and optimize it along with the rest of the model components.

The model is trained on synthetic ShapeNet dataset, and one out of 24 views of a shape (provided by BID5 ) is randomly selected at each training step as the image condition input.

The trained model is also tested on real images from the PASCAL3D+ dataset to prove its generalizability.

For each input, we removed the background, and cropped the image so that the target object is centered.

The PASCAL3D+ dataset is challenging because the images are captured in real environments, and contain noisier visual signals which are not seen during the training process.

We show ShapeNet testing images and PASCAL3D+ real images together with their sampled point cloud results on FIG5 upper left.

We further quantitatively evaluate the conditional generation results by calculating mean Intersection-over-Union (mIoU) with ground truth volumes.

Here we only consider 3D volumes containing more than 500 occupied voxels, and using furthest point sampling method to select 500 out of them to describe the shape.

To compensate for the sampling randomness of PointGrow, we slightly align generated points to their nearest voxels within a neighborhood of 2-voxel radius.

As shown in TAB1 , PointGrow achieves above-par performance on conditional 3D shape generation.

BID7 ).

Image Condition Arithmetic.

Another interesting way to impose the image conditions is to perform arithmetic on them.

We demonstrate this by combining embedded image condition vectors with different weights.

Examples of this kind are shown on FIG5 upper right.

Note that the generated final shapes contain geometrical features shown in generated shapes of their operands.

In this work, we propose PointGrow, a new generative model that can synthesize realistic and diverse point cloud with high resolution.

Unlike previous works that rely on local regularities to synthesize 3D shapes, our PointGrow builds upon autoregressive architecture to encode the diverse surface information of point cloud.

To further capture the long-range dependencies between points, two dedicated self-attention modules are designed and carefully integrated into our framework.

PointGrow as a generative model also enables effective unsupervised feature learning, which is extremely useful for low-data recognition tasks.

Finally, we show that PointGrow learns a smooth image condition manifold where 3D shape interpolation and arithmetic calculation can be performed inside.

@highlight

An autoregressive deep learning model for generating diverse point clouds.

@highlight

An approach for generating 3D shapes as point clouds which considers the lexicographic ordering of points according to coordinates and trains a model to predict points in order.

@highlight

The paper introduces a generative model for point clouds using a pixel RNN-like auto-regressive model and an attention model to handle longer-range interactions.