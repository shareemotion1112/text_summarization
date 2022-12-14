Performance of neural networks can be significantly improved by encoding known invariance for particular tasks.

Many image classification tasks, such as those related to cellular imaging, exhibit invariance to rotation.

In particular, to aid convolutional neural networks in learning rotation invariance, we consider a simple, efficient conic convolutional scheme that encodes rotational equivariance, along with a method for integrating the magnitude response of the 2D-discrete-Fourier transform (2D-DFT) to encode global rotational invariance.

We call our new method the Conic Convolution and DFT Network (CFNet).

We evaluated the efficacy of CFNet as compared to a standard CNN and group-equivariant CNN (G-CNN) for several different image classification tasks and demonstrated improved performance, including classification accuracy, computational efficiency, and its robustness to hyperparameter selection.

Taken together, we believe CFNet represents a new scheme that has the potential to improve many imaging analysis applications.

Though the appeal of neural networks is their versatility for arbitrary classification tasks, there is still much benefit in designing them for particular problem settings.

In particular, their effectiveness can be greatly increased by encoding invariance to uniformative augmentations of the data BID17 .

If such invariance is not explicitly encoded, the network must learn it from the data, perhaps with the help of data augmentation, requiring more parameters and thereby increasing its susceptibility to overfitting.

A key invariance inherent to several computer vision settings, including satellite imagery and all forms of microscopy imagery, is rotation BID3 BID1 .

Recently, there have been a variety of proposed approaches for encoding rotation equivariance and invariance, the most promising of which have formulated convolution over groups BID6 BID24 .

Notably, G-CNNs have been applied to several biological imaging tasks, producing state-of-the-art results BID24 BID0 BID18 .Here we propose a new rotation-equivariant convolutional scheme, called conic convolution, which, in contrast to group convolution, encodes equivariance while still operating over only the spatial domain.

Rather than convolving each filter across the entire image, as in standard convolution, rotated filters are convolved over corresponding conic regions of the input feature map that emanate from the origin, thereby transforming rotations in the input directly to rotations in the output.

This scheme is intuitive, simple to implement, and computationally efficient.

We also show that the method yields improved performance over group convolution on several relevant applications.

Additionally, we propose the integration of the magnitude response of the 2D-discrete-Fourier transform (2D-DFT) into a transition layer between convolutional and fully-connected layers to encode rotational invariance.

Though the insight of using the DFT to encode rotational invariance has been employed for texture classification using wavelets BID9 BID13 BID21 BID2 and for general image classification BID23 , as of yet, its application to CNNs has been overlooked.

As in these prior works, rotations of the input are transformed to circular shifts, to which the magnitude response of the 2D-DFT is invariant, in the transformed space.

Most other recently proposed rotationinvariance CNNs impose this invariance by applying a permutation-invariant operation, such as the average or maximum, over the rotation group, but since this operation is applied for each filter individually, possibly valuable pose information between filters is lost.

In contrast, the 2D-DFT is able to integrate mutual pose information between different filter responses, yielding richer features for subsequent layers.

We demonstrate the effectiveness of these two novel contributions for various applications: classifying rotated MNIST images, classifying synthetic images that model biomarker expression in microscopy images of cells, and localizing proteins in budding yeast cells BID15 .

We show that CFNet improves classification accuracy generally over the standard raster convolution formulation and over the equivariant method of G-CNN across these settings.

We also show that the 2D-DFT clearly improves performance across these diverse data sets, and that not only for conic convolution, but also for group convolution.

Source code for the implementation of CFNet will be made available on GitHub.2 RELATED WORK BID6 introduced G-CNNs by formulating convolution over groups, including rotation, translation, and flips, for neural networks, which has inspired many subsequent improvements.

By convolving over groups, equivariance to these groups is maintained throughout the convolutional layers, and invariance is enforced at the end of the network by pooling over groups.

This work was improved upon by the design of steerable filters BID24 for convolution, similar to those proposed by BID25 , which allow for finer sampling of rotations of filters without inducing artifacts.

Steerable filters were first proposed by BID10 and had been explored previously for image classification BID19 , but as shallow features in the context of HOG descriptors.

An alternative means of encoding rotational equivariance is to transform the domain of the image to an alternative domain, such as the log-polar domain BID23 BID11 in which rotation becomes some other transformation that is easier to manage, in this case, translations.

The suitability of this transformation depends upon the signal of interest, since this warping will introduce distortion, as pixels near the center of the image are sampled more densely than pixels near the perimeter.

In addition, its stability to translations in the original domain is of concern.

Our proposed CFNet, by convolving over conic regions, also encodes global rotation equivariance about the origin, but without introducing such distortion, which greatly helps mitigate its susceptibility to translation.

The recently developed spatial transform layer BID12 and deformable convolutional layer BID7 allow the network to learn non-regular sampling patterns and can potentially help learning rotation invariance, though invariance is not explicitly enforced, which would most likely be a challenge for tasks with small training data.

A simple means for achieving rotation equivariance and invariance was proposed by BID8 , in which feature maps of standard CNNs are made equivariant or invariant to rotation by combinations of cyclic slicing, stacking, rolling, and pooling.

RotEqNet BID20 improved upon this idea by storing, for each feature map for a corresponding filter, only the maximal response across rotations and the value of the corresponding rotation, to preserve pose information.

This approach yielded improved results and considerable storage savings over BID8 and G-CNN.

These methods are most similar to our proposed conic convolution.

However, in contrast, our method applies each filter only at the appropriate rotation within each conic region, which further saves on storage.

To enforce rotation invariance, as noted, most of the previous methods apply some permutationinvariant, or pooling, operation over rotations.

BID3 recently proposed a strategy of encouraging a network to learn a rotation invariant transform, and follow-up work improved this learning process by incorporating a Fisher discriminant penalty BID4 .

However, the convolutional layers of the network do not maintain the property of rotation equivariance with the input image, which requires that the network learn this equivariance and could therefore hinder performance.

Also, learning such a transform that generalizes to unseen data could prove difficult for settings with limited training data.

BID23 previously proposed the 2D-DFT for rotational invariance.

However, no method has yet been proposed to integrate the 2D-DFT into a rotation-equivariant CNN.

We begin our formulation with a simpler, special case of conic convolution, which we call quadrant convolution.

Its only difference from standard convolution is that the filter being convolved is rotated by r??/2, r ??? {0, 1, 2, 3}, depending upon the corresponding quadrant of the domain.

We show that for quadrant convolution, rotations of ??/2 in the input are straightforwardly associated with rotations in the output feature map, which is a special form of equivariance called same-equivariance (as coined by BID8 ).For convenience, we represent feature maps, of dimension K, f : Z 2 ??? R K , and filters, ?? : Z 2 ??? R K , of a network as functions over 2D space, as in BID6 .

Relevant to our formulation is the set, or group, G of two-dimensional rotation matrices of ??/2, which can be easily parameterized by g(r), and which acts on points in Z 2 by matrix multiplication, i.e, for a given point DISPLAYFORM0 Let T g denote the transformation of a function by a rotation in G, where DISPLAYFORM1 ) applies the inverse of g to an element of the domain of f .

For an operation ?? : F ??? F, F being the set of K-dimensional functions f , to exhibit same-equivariance, applying rotation either before or after the operation yields the same result, i.e. DISPLAYFORM2 We now define quadrant convolution.

The expression for convolution in a standard CNN is given by DISPLAYFORM3 As noted in BID6 , standard convolution does not exhibit rotational equivariance unless certain constraints on the filters are met.

Quadrant convolution can be interpreted as weighting the convolution for each rotation with a function ?? : Z 2 ??? [0, 1] that simply "selects" the appropriate quadrant of the domain, which we define as DISPLAYFORM4 Since the origin does not strictly belong to a particular quadrant, it is handled simply by averaging the response of the filter at all four rotations.

Boundary values are assigned arbitrarily, but consistently, by the placement of the equality for either u or v. The output of the layer is then given by DISPLAYFORM5 Example convolutional regions with appropriate filter rotations are shown in FIG0 .The equivariance property is established (see Appendix) independent of the definition of ??, yet its definition will greatly influence the performance of the network.

For example, if ?? is simply the constant 1/4, we have the simple example of equivariance mentioned above, equivalent to averaging the filter responses.

The above formulation can be generalized to conic convolution in which the rotation angle is decreased by an arbitrary factor of ??/2R, for some positive integer R, instead of being fixed to ??/2.

Rather than considering quadrants of the domain, we can consider conic regions emanating from the origin, defined by DISPLAYFORM0 where I(??) is the indicator function.

The weighting function is changed to have value one only over this conic region: DISPLAYFORM1 of which ?? 1 = ?? q is a special case.

If we consider feature maps to be functions over the continuous domain R 2 , instead of Z 2 , and define the group G R , with parameterization DISPLAYFORM2 for r ??? {0, 1, . . .

, 4R ??? 1} and x = (u, v) ??? R 2 , it is easy to show similarly as above that DISPLAYFORM3 is equivariant to G R .However, due to subsampling artifacts when discretizing R 2 to Z 2 , as in an image, rotation equivariance for arbitrary values of R cannot be guaranteed and can only be approximated.

In particular, the filters will have to be interpolated for rotations that are not a multiple of ??/2.

In our experiments, we chose nearest neighbor interpolation, which at least preserves the energy of the filter under rotations.

This defect notwithstanding, it can be shown that conic convolution maintains equivariance to rotations of ??/2, and as our experiments show in the following section, the approximation of finer angles of rotation can still improve performance.

Additionally, we note that R need not be the same for each layer, and it may be advantageous to use a finer discretization of rotations for early layers, when the feature maps are larger, and gradually decrease R.

A note must be made about subsequent nonlinear operations for a convolutional layer.

It is typical in convolutional networks to perform subsampling, either by striding the convolution or by spatial pooling, to reduce the dimensionality of subsequent layers.

Again, due to downsampling artifacts, rotational equivariance to rotations smaller than ??/2 is not guaranteed.

However, given that the indices of the plane of the feature map are in Z 2 and are therefore centered about the origin, a downsampling of D ??? Z >0 can be applied while maintaining rotational equivariance for rotations of ??/2, regardless of the choice of R. After subsampling, the result is passed through a non-linear activation function ?? : R ??? R, such as ReLU, with an added offset c k ??? R.

In theory, the response for each rotation in conic convolution is only needed over its corresponding conic region.

However, since GPUs are more efficient operating on rectangular inputs, it is faster to compute the convolution over each quadrant in which the conic region resides.

In current neural network libraries, the output of conic convolution can be achieved by convolving over the corresponding quadrant, multiplying by the weighting function, summing the responses is in each quadrant together, and then concatenating the responses of quadrants.

For the special case of quadrant convolution, this process incurs negligible additional computation beyond standard convolution.

Additionally, conic convolution produces only one feature map per filter as in standard convolution and therefore incurs no additional storage costs, in contrast to G-CNN and cyclic slicing, which both produce one map per rotation BID6 BID8 , and two for RotEqNet, one for the filter response and one for the orientation BID20 .

After the final convolutional layer of a CNN, some number of fully-connected layers will be applied to combine information from the various filter responses.

In general, fully-connected layers will not maintain rotation equivariance or invariance properties.

In a fully-convolutional network, convolution and downsampling are applied until the spatial dimensions are eliminated and the resulting feature map of the final convolutional layer is merely a vector, with dimension equal to the number of filters.

Rather than encoding invariance for each filter separately, as in most other recent works BID6 BID24 , we consider instead to transform the collective filter responses to a space in which rotation becomes circular shift so that the 2D-DFT can be applied to encode invariance.

The primary merit of the 2D-DFT as an invariant transform is that each output node is a function of every input node, and not just the nodes of a particular filter response, thereby capturing mutual information across responses.

Since the formulation of this transition involves the DFT, which is defined only for finite-length signals, we switch to represent feature maps as tensors, rather than functions.

We denote the feature map generated by the penultimate convolutional layer by f ??? R M ??M ??K , where M ??? Z >1 .In a fully-convolutional network, the final convolutional layer is in reality just a fully-connected layer, in which the input f is passed through N fully-connected filters, ?? (n) ??? R M ??M ??K , n ??? {0, 1, . . .

, N ??? 1}. The operation of this layer can be interpreted as the inner product of the function and filter, ?? (n) , f .

If we again consider rotations of the filter from the group G R , DISPLAYFORM0 this is equivalent to the first layer of a G-CNN, mapping from the spatial domain to G R (though this group does not include the translation group since the convolution is only applied at the origin), and rotations of the final convolutional layer f will correspond to permutations of G R , which are just circular shifts in of the second dimension of the matrix ??.The magnitude response of the 2D-DFT can be applied to ?? to transform these circular shifts to an invariant space, DISPLAYFORM1 This process of encoding rotation invariance corresponds to the 'Convolutional-to-Full Transition' in FIG1 .

The result is then vectorized and passed into fully-connected layers that precede the final output layer, as in a standard CNN.

We note that it helped in practice to apply batch normalization after vectorizing, since the output of the magnitude of the 2D-DFT will not be normalized as such.

The 2D-DFT, as a rotation invariant transform, can also be integrated into other rotation-equivariant networks, such as G-CNN.

At the final layer of a fully-convolutional G-CNN, since the spatial dimension has been eliminated through successive convolutions and spatial downsampling, rotation is encoded along contiguous stacks of feature maps f ??? R L??4 of each filter at four rotations.

In this way, rotations similarly correspond to circular shifts in the final dimension.

This representation ?? is then passed through the 2D-DFT, as in Eqn.

11.

The rotated MNIST data set BID16 has been used as a benchmark for several previous works on rotation invariance.

As in previous works, to tune the parameters of each method, we first trained various models on a set of 10,000 images, using training augmentation of rotations of arbitrary angles as in BID6 1 , and then selected the best model based on the accuracy on a separate validation set of 5,000 images.

Our best CFNet architecture consisted of six convolution layers; the first were conic convolutions of R = 8 for the first three layers and R = 4 for the next four, with spatial max-pooling after the second layer.

We used a filter size of three pixels, with 15 filters per layer.

The final convolutional layer was the DFT transition layer as described in the previous section, which was followed by an output layer of ten nodes.

This architecture was similar in terms of number of layers and filters per layer as that of the G-CNN of BID6 .

To evaluate the G-CNN with the DFT, the only changes we made from the reported architecture for G-CNN was to reduce the number of filters for each layer to 7, to offset the addition of the 2D-DFT, which was applied to the output of the final convolutional layer.

The results on a held-out set of 50,000 test images are shown in TAB0 .

Adding the DFT transition to the output of G-CNN reduces the test error by 0.28%, demonstrating the value of incorporating mutual rotational information between filters when encoding invariance.

The replacement of group convolution with conic convolution in CFNet leads to an even further reduction in error of 0.25%.

Even with its simple conic convolutional scheme, CFNet is able to perform comparably to H-Net 2 , which constructs filters from the circular harmonic basis and operates on complex feature maps.

In order to explicitly control the manifestation of rotational invariance, we created a set of synthetic images, based upon Gaussian-mixture models (GMMs), which can also be used to emulate realworld microscopy images of biological signals BID26 .

Example synthetic images from across and within classes are shown in FIG2 and FIG2 , respectively.

We defined 50 distribution patterns and generated 50 and 100 examples per class for training and 200 examples per class for testing.

Each class was defined by a mixture of ten Gaussians.

The image size was 50 pixels.

A batch size of 50 examples, a learning rate of 5 ?? 10 ???3 , and a weight decay 2 penalty of 5 ?? 10 DISPLAYFORM0 were used during training.

To help all methods, we augmented the training data by rotations and random jitter of up to three pixels, as was done during image generation.

Classification accuracies on the test set over training steps for various numbers of training samples, denoted by N , for several methods are shown in FIG2 .

A variety of configurations were trained for each network, and each configuration was trained three times.

The darkest line shows the accuracy of the configuration that achieved the highest moving average, with a window size of 100 steps, for each method.

The spread of each method, which is the area between the point-wise maximum and minimum of the error, is shaded with a light color, and three standard-deviations around the mean is shaded darker.

We observe a consistent trend of CFNet outperforming G-CNN, which in turn marginally outperforms the CNN, both in overall accuracy and in terms of the number of steps required to attain that accuracy.

Additionally, the spread of CFNet is mostly above even the best performing models of G-CNN and the CNN, demonstrating that an instance of CFNet will outperform other methods even if the best set of hyperparameters has not been chosen.

We also included a network consisting of conic convolutional layers, but without the DFT, noted as 'CNet', to show the relative merit of the DFT.

CNet performs comparably to the standard CNN while requireing significantly less parameters to attain the same performance, though the true advantage of conic convolution is shown when integrated with the DFT to achieve global rotation invariance.

In comparison, including the 2D-DFT increases the performance of G-CNN, to a comparable level with CFNet in fact, though it does not train as quickly.

We extended our analysis to real biomarker images of budding yeast cells BID15 , shown in FIG2 .

Each image consists of four stains, where blue shows the cytoplasmic region, pink the nuclear region, red the bud neck, and green the protein of interest.

The classification for each image is the cellular subcompartmental region in which the protein is expressed, such as the cell periphery, mitochondria, or eisosomes, some of which exhibit very subtle differences.

Fig.

3f-g shows the results of using CFNet, G-CNN, and a standard CNN to classify the protein localization for each image.

We used the same architecture as reported in BID15 for all methods, except that we removed the last convolutional layer and reduced the number of filters per layer by roughly half for CFNet and G-CNN, to offset for encoding of equivariance and invariance.

The same training parameters and data augmentation were used as for the synthetic data, except that a dropout probability of 0.8 was applied at the final layer and the maximum jitter was increased to five pixels, since many examples were not well-centered.

For each method, several iterations were run and the spread and the best performing model are shown.

Again, CFNet outperforms G-CNN and a standard CNN, when the number of training examples per class is either 50 or 100 (see FIG2 , demonstrating that the gains of the 2D-DFT and proposed convolutional layers translate to real-world microscopy data.

We note that the best reported algorithm that did not use deep learning, called ensLOC , was only able to achieve an average precision of 0.49 for a less challenging set of yeast phenotypes and with ???20,000 samples, whereas all runs of CFNet achieved an average precision of between 0.60 -0.67 with ???10% of the data used for training.

We note that the transform T g distributes over both multiplication and convolution, i.e. DISPLAYFORM0 .

From these properties, it is easy to show that such an operation ?? is same-equivariant for rotation.

Theorem.

DISPLAYFORM1 Proof.

DISPLAYFORM2 where for (a) we make a change of variables, combining g and h into p, where p = g ???1 h and T

g T h = T g ???1 h = T p .

Mathematically, each class k ??? {1, . . .

, K} is described by the parameters ?? of the GMM: DISPLAYFORM0 , where ?? g ??? [???1, 1] 2 and the number of Gaussians per class is a parameter G of the data set.

For simplicity, we consider the image I : R 2 ??? R to be nonzero only over a region slightly larger than the [???1, 1] 2 box, so that it captures the majority of points generated by the Gaussians.

To generate a sample image from the generating distribution, first, a constant background intensity is set for the image according to b ??? Exp(0, ?? B ), so I(p) = b, ???p ??? R 2 .

Then a random angle ?? ??? Uniform [0, 2??] is drawn to determine the rotation of the image.

The mean ?? g ??? N ( ?? g , ??) for each Gaussian of the class is drawn from an underlying Gaussian with mean ?? g , which introduces some small jitter of the relative locations of the Gaussians.

A number n g ??? N (?? n,g , ?? n ) of points {p} in [???1, 1] 2 , which vary for each Gaussian, are drawn from this Gaussian according to p ??? N R ?? g , R?? g R ???1 , where the realized mean and covariance have been rotated by ?? by the rotation matrix: R = cos(??) ??? sin(??) sin(??) cos(??) .For each point p, its corresponding intensity value is drawn according to I(p) ??? Uniform [

?? I ??? m I , ?? I + m I ], replacing the background value.

Having drawn all of the points, the image is smoothed with a Gaussian kernel with variance ?? s to emulate the point-spread function of the imager and pixel noise is added: I(p) = I(p) + Exp(0, ?? I ).

To simulate camera jitter, the image is translated by a random offset of up to three pixels.

@highlight

We propose conic convolution and the 2D-DFT to encode rotation equivariance into an neural network.

@highlight

In the context of image classification, the paper proposes a convolutional neural network architecture with rotation-equivariant feature maps that are eventually made rotation-invariant by using the magnitude of the 2D discrete Fourier transform (DFT).

@highlight

Authors provide a rotation invariant neural network via combining conic convolution and 2D-DFT