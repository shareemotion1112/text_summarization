We propose a novel generative model architecture designed to learn representations for images that factor out a single attribute from the rest of the representation.

A single object may have many attributes which when altered do not change the identity of the object itself.

Consider the human face; the identity of a particular person is independent of whether or not they happen to be wearing glasses.

The attribute of wearing glasses can be changed without changing the identity of the person.

However, the ability to manipulate and alter image attributes without altering the object identity is not a trivial task.

Here, we are interested in learning a representation of the image that separates the identity of an object (such as a human face) from an attribute (such as 'wearing glasses').

We demonstrate the success of our factorization approach by using the learned representation to synthesize the same face with and without a chosen attribute.

We refer to this specific synthesis process as image attribute manipulation.

We further demonstrate that our model achieves competitive scores, with state of the art, on a facial attribute classification task.

Latent space generative models, such as generative adversarial networks (GANs) BID11 BID27 and variational autoencoders (VAEs) BID28 BID14 , learn a mapping from a latent encoding space to a data space, for example, the space of natural images.

It has been shown that the latent space learned by these models is often organized in a near-linear fashion BID27 BID14 , whereby neighbouring points in latent space map to similar images in data space.

Certain "directions" in latent space correspond to changes in the intensity of certain attributes.

In the context of faces, for example, directions in latent space would correspond to the extent to which someone is smiling.

This may be useful for image synthesis where one can use the latent space to develop new design concepts BID9 Zhu et al., 2016) , edit an existing image (Zhu et al., 2016) or synthesize avatars BID35 BID32 .

This is because semantically meaningful changes may be made to images by manipulating the latent space BID27 Zhu et al., 2016; BID17 .One avenue of research for latent space generative models has been class conditional image synthesis BID25 BID24 , where an image of a particular object category is synthesized.

Often, object categories may be sub-divided into fine-grain subcategories.

For example, the category "dog" may be split into further sub-categories of different dog breeds.

Work by BID3 propose latent space generative models for synthesizing images from fine-grained categories, in particular for synthesizing different celebrities' faces conditional on the identity of the celebrity.

Rather than considering fine-grain categories, we propose to take steps towards solving the different, but related problem of image attribute manipulation.

To solve this problem we want to be able to synthesize images and only change one element or attribute of its content.

For example, if we are synthesizing faces we would like to edit whether or not a person is smiling.

This is a different problem to fine-grain synthesis; we want to be able to synthesize two faces that are similar, with only a single chosen attribute changed, rather than synthesizing two different faces.

The need to synthesis two faces that are similar makes the problem of image attribute manipulation more difficult than the fine-grain image synthesis problem; we need to learn a latent space representation that separates an object category from its attributes.

In this paper, we propose a new model that learns a factored representation for faces, separating attribute information from the rest of the facial representation.

We apply our model to the CelebA BID21 dataset of faces and control several facial attributes.

Our contributions are as follows:1.

Our core contribution is the novel cost function for training a VAE encoder to learn a latent representation which factorizes binary facial attribute information from a continuous identity representation (Section 3.2).

2.

We provide an extensive quantitative analysis of the contributions of each of the many loss components in our model (Section 4.2).

3.

We obtain classification scores that are competitive with state of the art (Zhuang et al., 2018) using the classifier that is already incorporated into the encoder of the VAE (Section 4.3).

4.

We provide qualitative results demonstrating that our latent variable, generative model may be used to successfully edit the 'Smiling' attribute in more than 90% of the test cases (Section 4.4).

5.

We discuss and clarify the distinction between conditional image synthesis and image attribute editing (Section 5).

6.

We present code to reproduce experiments shown in this paper: (provided after review).

Latent space generative models come in various forms.

Two state-of-art generative models are Variational Autoencoders (VAE) BID28 BID14 and Generative Adversarial Networks (GAN).

Both models allow synthesis of novel data samples from latent encodings, and are explained below in more detail.

Variational autoencoders BID14 BID28 consist of an encoder q ?? (z|x) and decoder p ?? (x|z); oftentimes these can be instantiated as neural networks, E ?? (??) and D ?? (??) respectively, with learnable parameters, ?? and ??.

A VAE is trained to maximize the evidence lower bound (ELBO) on log p(x), where p(x) is the data-generating distribution.

The ELBO is given by: DISPLAYFORM0 where p(z) is a chosen prior distribution such as p(z) = N (0, I).

The encoder predicts, ?? ?? (x) and ?? ?? (x) for a given input x and a latent sample,???, is drawn from q ?? (z|x) as follows: DISPLAYFORM1 .

By choosing a multivariate Gaussian prior, the KL-divergence may be calculated analytically BID14 .

The first term in the loss function is typically approximated by calculating the reconstruction error between many samples of DISPLAYFORM2 New data samples, which are not present in the training data, are synthesised by first drawing latent samples from the prior, z ??? p(z), and then drawing data samples from p ?? (x|z).

This is equivalent to passing the z samples through the decoder, D ?? (z).VAEs offer both a generative model, p ?? (x|z), and an encoding model, q ?? (z|x), which are useful as starting points for image editing in the latent space.

However, samples drawn from a VAE are often blurred BID27 .

An alternative generative model, which may be used to synthesize much sharper images, is the Generative Adversarial Network (GAN) BID11 BID27 .

GANs consist of two models, a generator, G ?? (??), and a discriminator, C ?? (??), both of which may be implemented using convolutional neural networks BID27 BID7 .

GAN training involves these two networks engaging in a mini-max game.

The discriminator, C ?? , is trained to classify samples from the generator, G ?? , as being 'fake' and to classify samples from the data-generating distribution, p(x), as being 'real'.

The generator is trained to synthesize samples that confuse the discriminator; that is, to synthesize samples that the discriminator cannot distinguish from the 'real' samples.

The objective function is given by: DISPLAYFORM0 where p g (x) is the distribution of synthesized samples, sampled by: DISPLAYFORM1 where p(z) is a chosen prior distribution such as a multivariate Gaussian.

The vanilla GAN model does not provide a simple way to map data samples to latent space.

Although there are several variants on the GAN that do involve learning an encoder type model BID10 BID8 , only the approach presented by allows data samples to be faithfully reconstructed.

The approach presented by requires adversarial training to be applied to several high dimensional distributions.

Training adversarial networks on high dimensional data samples remains challenging BID1 despite several proposed improvements BID29 .

For this reason, rather than adding a decoder to a GAN, we consider an alternative latent generative model that combines a VAE with a GAN.

In this arrangement, the VAE may be used to learn an encoding and decoding process, and a discriminator may be placed after the decoder to ensure higher quality of the data samples outputted from the decoder.

Indeed, there have been several suggestions on how to combine VAEs and GANs BID3 BID17 BID23 each with a different structure and set of loss functions, however, none are designed specifically for attribute editing.

The content of image samples synthesized from a vanilla VAE or GAN depends on the latent variable z, which is drawn from a specified random distribution, p(z).

For a well-trained model, synthesised samples will resemble samples in the training data.

If the training data consists of images from multiple categories, synthesized samples may come from any, or possibly a combination, of those categories.

For a vanilla VAE, it is not possible to choose to synthesize samples from a particular category.

However, conditional VAEs (and GANs) BID25 BID24 provide a solution to this problem as they allow synthesis of class-specific data samples.

Autoencoders may be augmented in many different ways to achieve category-conditional image synthesis BID3 .

It is common to append a one-hot label vector, y, to inputs of the encoder and decoder BID31 .

However, for small label vectors, relative to the size of the inputs to the encoder and the decoder model, it is possible for the label information, y, to be ignored.

1 .

A more interesting approach, for conditional (non-variational and semi-supervised) autoencoders is presented by BID22 , where the encoder outputs both a latent vector,???, and an attribute vector,??.

The encoder is updated to minimize a classification loss between the true label, y, and??.

We incorporate a similar architecture into our model with additional modifications to the training of the encoder for the reasons explained below.

There is a drawback to incorporating attribute information in the way described above BID22 BID26 when the purpose of the model is to edit specific attributes, rather than to synthesize samples from a particular category.

We observe that in this naive implementation of conditional VAEs, varying the attribute (or label) vector,??, for a fixed??? can result in unpredictable changes in synthesized data samples,x.

Consider for example the case where, for a fixed???, modifying?? does not result in any change in the intended corresponding attribute.

This suggests that information about the attribute one wishes to edit, y, is partially contained in??? rather than solely in y. Similar problems have been discussed and addressed to some extent in the GAN literature BID24 BID25 , where it has been observed that label information in?? is often ignored during sample synthesis.

In general, one may think that??? and?? should be independent.

However, if attributes, y, that should be described by?? remain unchanged for a reconstruction where only?? is changed, this suggests that z contains most of the information that should have been encoded within??.

We propose a process to separate the information about y from??? using a mini-max optimization involving y,???, the encoder E ?? , and an auxiliary network A ?? .

We refer to our proposed process as 'Adversarial Information Factorization'.

For a given image of a face, x, we would like to describe the face using a latent vector,???, that captures the identity of the person, along with a single unit vector,?? ??? [0, 1], that captures the presence, or absence, of a single desired attribute, y. If a latent encoding,???, contains information about the desired attribute, y, that should instead be encoded within the attribute vector,??, then a classifier should be able to accurately predict y from???.

Ideally,??? contains no information about y and so, ideally, a classifier should not be able to predict y from???.

We propose to train an auxiliary network to predict y from??? accurately while updating the encoder of the VAE to output??? values that cause the auxiliary network to fail.

If??? contains no information about the desired attribute, y, that we wish to edit, then the information can instead be conveyed in?? sincex must still contain that information in order to minimize reconstruction loss.

We now formalize these ideas.

In what follows, we explain our novel approach to training the encoder of a VAE, to factor (separate) out information about y from???, such that H(y|???) ??? H(y).

We integrate this novel factorisation method into a VAE-GAN.

The GAN component of the model is incorporated only to improve image quality.

Our main contribution is our proposed adversarial method for factorising the label information, y, out of the latent encoding,???.

A schematic of our architecture is presented in Figure 1 .

In addition to the encoder, E ?? , decoder, D ?? , and discriminator, C ?? , we introduce an auxiliary network, A ?? :??? ??????, whose purpose is described in detail in Section 3.2.

Additionally, the encoder also acts as a classifier, outputting an attribute vector,??, along with a latent vector,???.

The parameters of the decoder, ??, are updated with gradients from the following loss function: et al., 2017) , where y real and y f ake are vectors of ones and zeros respectively and z ??? p(z).

Note that L bce is the binary cross-entropy loss given by DISPLAYFORM0 DISPLAYFORM1 The parameters of the encoder, ??, intended for use in synthesizing images from a desired category, may be updated by minimizing the following function: DISPLAYFORM2 where ?? and ?? are additional regularization coefficients; DISPLAYFORM3 is the classification loss on the input image.

Unfortunately, the loss function in Equation FORMULA7 is not sufficient for training an encoder used for attribute manipulation.

For this, we propose an additional network and cost function, as described below.(a) Current work (b) Previous work BID3 Figure 1: (a) Current work (adversarial information factorization) This figure shows our model where the core, shown in blue, is a VAE with information factorization.

Note that E ?? is split in two, E z,?? and E y,?? , to obtain both a latent encoding,???, and the label,??, respectively.

D ?? is the decoder and A ?? the auxiliary network.

The pink blocks show how a GAN architecture may be incorporated by placing a discriminator, C ?? , after the encoder, E ?? , and training C ?? to classify decoded samples as "fake" and samples from the dataset as "real".

For simplicity, the KL regularization is not shown in this figure.

(b) Previous work: cVAE-GAN BID3 Architecture most similar to our own.

Note that there is no auxiliary network performing information factorization and a label,??, is predicted only for the reconstructed image, rather than for the input image (??).

To factor label information, y, out of??? we introduce an additional auxiliary network, A ?? , that is trained to correctly predict y from???.

The encoder, E ?? , is simultaneously updated to promote A ?? to make incorrect classifications.

In this way, the encoder is encouraged not to place attribute information, y, in???.

This may be described by the following mini-max objective: DISPLAYFORM0 where E z,??(x) is the latent output of the encoder.

Training is complete when the auxiliary network, A ?? , is maximally confused and cannot predict y from??? = E z,?? (x), where y is the true label of x. The encoder loss is therefore given by: DISPLAYFORM1 We call the conditional VAE-GAN trained in this way an Information Factorization cVAE-GAN (IFcVAE-GAN).

The training procedure is presented in Algorithm 1.

To edit an image such that it has a desired attribute, we encode the image to obtain a???, the identity representation, append it to our desired attribute label,?? ??? y, and pass this through the decoder.

We use?? = 0 and?? = 1 to synthesize samples in each mode of the desired attribute e.g. 'Smiling' and 'Not Smiling'.

Thus, attribute manipulation becomes a simple 'switch flipping' operation in the representation space.

In this section, we show both quantitative and qualitative results to evaluate our proposed model.

We begin by quantitatively assessing the contribution of adversarial information factorization in an x ??? D D is the training data 5: DISPLAYFORM0 output of the auxiliary network 9:# Calculate updates, u 10:# do updates 11: DISPLAYFORM1 end for 16: end procedure ablation study.

Following this we perform facial attribute classification using our model.

We use a standard deep convolutional GAN, DCGAN, architecture for the ablation study BID27 , and subsequently incorporate residual layers BID12 into our model in order to achieve competitive classification results compared with a state of the art model (Zhuang et al., 2018) .

We finish with a qualitative evaluation of our model, demonstrating how our model may be used for image attribute editing.

For our qualitative results we continue to use the same residual networks as those used for classification, since these also improved visual quality.

We refer to any cVAE-GAN that is trained without an L aux term in the cost function as a naive cVAE-GAN and a cVAE-GAN trained with the L aux term as an Information Factorization cVAE-GAN (IFcVAE-GAN).

When performing image attribute manipulation, there are two important things that we would like to quantify.

The first, is reconstruction quality, approximated by the mean squared error, MSE, between DISPLAYFORM0 The second, is the proportion of edited images that have a desired attribute.

To approximate this, we train an independent classifier on real images to classify the presence (y = 1) or absence (y = 0) of a desired attribute.

We apply the trained classifier to edited images, synthesized using?? = 1 and y = 0 to obtain classification scores,?? Smiling and?? N ot???Smiling respectively.

TAB0 shows the contributions of each component of our novel function (Equation 6).

We consider reconstruction error and classification scores on edited image samples.

Smaller reconstruction error indicates better reconstruction, and larger classification scores (?? Smiling and?? N ot???Smiling ) suggest better control over attribute changes.

Note that all input test images for this experiment were from the 'Smiling' category.

From TAB0 , we make the following observations:

(1) Our model: Our model is able to successfully edit an image to have the 'Not Smiling' attribute in 81.3% of cases and the 'Smiling' attribute in all cases.(2) Effect of Removing Information Factorization: Without our proposed L aux term in the encoder loss function, the model fails completely to perform attribute editing.

Since?? Smiling + C N ot???Smiling ??? 100%, this strongly suggests that samples are synthesized independently of?? and that the synthesized images are the same for?? = 0 and?? = 1.(3) Effect of classifying reconstructed samples: We explored the effect of including a classification loss on reconstructed samples, L bce (y,??), where?? = E y,?? (D ?? (x)).

A similar loss had been proposed by both BID3 and in the GAN literature BID25 for conditional image synthesis (rather than attribute editing).

To the best of our knowledge, this approach has not been used in the VAE literature.

This term is intended to maximise I(x; y) by providing a gradient containing label information to the decoder, however, it does not contribute to the factorization of attribute information, y, from??? and does not provide any clear benefit in our model.(4) IcGAN Perarnau et al. (2016) : We choose to include the IcGAN in our ablation study, since it is similar to our model without L KL and L aux .

While the IcGAN achieves a similar reconstruction error to our model it performs less well at attribute editing tasks.

We have proposed a model that learns a representation, {???,??}, for faces such that the identity of the person, encoded in???, is factored from a particular facial attribute.

We achieve this by minimizing the mutual information between the identity encoding and the facial attribute encoding to ensure that H(y|???) ??? H(y), while also training E y,?? as an attribute classifier.

Our training procedure encourages the model to put all label information into??, rather than???.

This suggests that our model may be useful for facial attribute classification.

To further illustrate that our model is able to separate the representation of particular attributes from the representation of the person's identity, we can measure the model's ability, specifically the encoder, to classify facial attributes.

We proceed to use E y,?? directly for facial attribute classification and compare the performance of our model to that of a state of the art classifier proposed by Zhuang et al. (2018) .

Results in Figure 2 show that our model is highly competitive with a state of the art facial attribute classifier Zhuang et al. (2018) .

We outperformed by more than 1% on 2 out of 10 categories, underperformed by more than 1% on only 1 category and remained competitive with all other attributes.

These results demonstrate that the model is effectively factorizing out information about the attribute from the identity representation.

In this section, we focus on attribute manipulation (described previously in Section 3.3).

Briefly, this involves reconstructing an input image, x, for different attribute values,?? ??? {0, 1}.We begin by demonstrating how a cVAE-GAN BID3 ) may fail to edit desired attributes, particularly when it is trained to achieve low reconstruction error.

The work of Bao et al. BID3 focused solely on the ability to synthesise images with a desired attribute, rather than to reconstruct a particular image and edit a specific attribute.

It is challenging to learn a representation that both preserves identity and allows factorisation BID46 .

Figure 3 (c,e) shows edited images, setting?? = 0 for 'Not Smiling' and?? = 1 for 'Smiling'.

We found that the cVAE-GAN BID3 ) failed to edit samples for the y = 0 ('Not Smiling') case.

This failure demonstrates the need for models that learn a factored latent representation, while maintaining good reconstruction quality.

Note that we achieve good reconstruction by reducing Figure 2 : Facial Attribute Classification.

We compare the performance of our classifier, E y,?? , to a state of art classifier (Zhuang et al., 2018) .weightings on the KL and GAN loss terms, using ?? = 0.005 and ?? = 0.005 respectively.

We trained the model using RMSProp Tieleman & Hinton (2012) with momentum = 0.5 in the discriminator.

We train our proposed IFcVAE-GAN model using the same optimiser and hyper-parameters that were used for the BID3 model above.

We also used the same number of layers (and residual layers) in our encoder, decoder and discriminator networks as those used by BID3 .

Under this set-up, we used the following additional hyper-parameter: {?? = 1.0} in our model.

Figure 3 shows reconstructions when setting?? = 0 for 'Not Smiling' and?? = 1 for 'Smiling'.

In contrast to the naive cVAE-GAN BID3 , our model is able to achieve good reconstruction, capturing the identity of the person, while also being able to change the desired attribute.

Table 2 shows that the model was able to synthesize images with the 'Not Smiling' attribute with a 98% success rate, compared with a 22% success rate using the naive cVAE- GAN Bao et al. (2017) .

Figure 3: Reconstructions, 'Smiling' and 'Not Smiling'.

The goal here was to reconstruct the face, changing only the desired 'Smiling' attribute.

This demonstrates how other conditional models BID3 ) may fail at the image attribute editing task, when high quality reconstructions are required.

Both models are trained with the same optimizers and hyper-parameters.

Table 2 : Comparing our model, the IFcVAE-GAN, to the naive cVAE- GAN Bao et al. (2017) .

C Smiling and?? N ot???Smiling denote the proportion of edited samples that have the desired attribute.

We see that both models achieve comparable (MSE) reconstruction errors, however, only our model is able to synthesize images of faces without smiles.

A complete ablation study for this model (with residual layers) is given in the appendix TAB1 ..

Ours (with residual layers) 0.011 98% 100% cVAE- GAN Bao et al. (2017) (with residual layers) 0.011 22% 85%

In this section we apply our proposed method to manipulate other facial attributes where the initial samples, from which the???'s are obtained, are test samples whose labels are y = 1 indicating the presence of the desired attribute (e.g. 'Blonde Hair').

In Figure 4 , we observe that our model is able to both achieve high quality reconstruction and edit the desired attributes.

Figure 4 : Editing other attributes.

We obtain a???, the identity representation, by passing an image, x through the encoder.

We append??? with a desired attribute label,?? ??? y, and pass this through the decoder.

We use?? = 0 and?? = 1 to synthesize samples in each mode of the desired attributeWe have presented the novel IFcVAE-GAN model, and (1) demonstrated that our model learns to factor attributes from identity, (2) performed an ablation study to highlight the benefits of using an auxiliary classifier to factorize the representation and (3) shown that our model may be used to achieve competitive scores on a facial attribute classification task.

We now discuss this work in the context of other related approaches.

We have used adversarial training (involving an auxiliary classifier) to factor attribute label information, y, out of the encoded latent representation,???.

Schmidhuber FORMULA3 and BID16 perform similar factorization of the latent space.

Similarly to us, BID16 incorporate this factorisation technique into the encoder of a generative model, however, unlike in our model, their encoder does not predict attribute information and so may not be used as a classifier.

BID4 proposed a general approach for predicting the mutual information, which may then be minimized via an additional model.

Rather than predicting mutual information BID4 between latent representations and labels, we implicitly minimize it via adversarial information factorization.

Our work has the closest resemblance to the cVAE-GAN architecture (see Figure 1) proposed by BID3 .

cVAE-GAN is designed for synthesizing samples of a particular class, rather than manipulating a single attribute of an image from a class.

In short, their objective is to synthesize a "Hathway" face, whereas our objective would be to make "Hathway smiling" or "Hathway not smiling", which has different demands on the type of factorization in the latent representation.

Separating categories is a simpler problem since it is possible to have distinct categories and changing categories may result in more noticeable changes in the image.

Changing an attribute requires a specific and targeted change with minimal changes to the rest of the image.

Additionally, our model simultaneously learns a classifier for input images unlike the work by BID3 .In a similar vein to our work, BID0 acknowledge the need for "identity preservation" in the latent space.

They achieve this by introducing an identity classification loss between an input data sample and a reconstructed data sample, rather than trying to separate information in the encoding itself.

Similar to our work, BID17 use a VAE-GAN architecture.

However, they do not condition on label information and their image "editing" process is not done in an endto-end fashion (likewise with BID34 ).Our work highlights an important difference between category conditional image synthesis BID3 and attribute editing in images BID16 BID26 : what works for category conditional image synthesis may not work for attribute editing.

Furthermore, we have shown (Section 4.2) that for attribute editing to be successful, it is necessary to factor label information out of the latent encoding.

In this paper, we have focused on latent space generative models, where a small change in latent space results in a semantically meaningful change in image space.

Our approach is orthogonal to a class of image editing models, called "image-to-image" models, which aim to learn a single latent representation for images in different domains.

Recently, there has been progress in image-to-image domain adaptation, whereby an image is translated from one domain (e.g. a photograph of a scene) to another domain (e.g. a painting of a similar scene) (Zhu et al., 2017; BID19 .

Image-to-image methods may be used to translate smiling faces to non-smiling faces, however, these models BID19 require significantly more resources than ours 2 .

By performing factorization in the latent space, we are able to use a single generative model, to edit an attribute by simply changing a single unit of the encoding, y, from 0 to 1 or vice versa.

Finally, while we use labelled data to learn representations, we acknowledge that there are many other models that learn factored, or disentangled, representations from unlabelled data including several VAE variants BID5 .

The ??-VAE objective is similar to the information bottleneck BID5 , minimizing mutual information, I(x; z), which forces the model to exploit regularities in the data and learn a disentangled representation.

In our approach we perform a more direct, supervised, factorisation of the latent space, using a mini-max objective, which has the effect of approximately minimizing I(z; y).

We have proposed a novel perspective and approach to learning representations of images which subsequently allows elements, or attributes, of the image to be modified.

We have demonstrated our approach on images of the human face, however, the method is generalisable to other objects.

We modelled a human face in two parts, with a continuous latent vector that captures the identity of a person and a binary unit vector that captures a facial attribute, such as whether or not a person is smiling.

By modelling an image with two separate representations, one for the object and the other for the object's attribute, we are able to change attributes without affecting the identity of the object.

To learn this factored representation we have proposed a novel model aptly named Information Factorization conditional VAE-GAN.

The model encourages the attribute information to be factored out of the identity representation via an adversarial learning process.

Crucially, the representation learned by our model both captures identity faithfully and facilitates accurate and easy attribute editing without affecting identity.

We have demonstrated that our model performs better than pre-existing models intended for category conditional image synthesis (Section 4.4), and have performed a detailed ablation study TAB0 which confirms the importance and relevance of our proposed method.

Indeed, our model is highly effective as a classifier, achieving state of the art accuracy on facial attribute classification for several attributes (Figure 2 ).

Our approach to learning factored representations for images is both a novel and important contribution to the general field of representation learning.

For completeness we include a table TAB1 demonstrating an ablation study for our model with the residual network architecture discussed in Section 4.4, note that this is the same architecture that was used by BID3 .

TAB1 and additionally, Figure 5 , demonstrate the need for the L aux loss and shows that increased regularisation reduces reconstruction quality.

The table also shows that there is no significant benefit to using theL class loss.

These findings are consistent with those of the ablation study in the main body of the text for the IFcVAE-GAN with a the GAN architecture of BID27 .We additionally show results without L KL and L gan .

Results show that small amounts of KL regularisation are required to achieve good reconstruction.

Models trained without L gan achieve slightly lower reconstruction error than other models, however, the reconstructed images are blurred (see Figure 6 ).

Interestingly, when our model is trained without the L gan or L KL loss, it is still able to edit attributes with high accuracy, however, the visual quality of samples is poor.

This shows that the attribute information is still factored from the rest of the latent representation, which is the main contribution of our work.

In our model we use labelled data to learn factored representations, however, there are many other models that learn factored, or disentangled, representations from unlabelled data including several variational autoencoder BID28 BID14 variants BID5 .

Once trained, the representation learned by each of these models may be evaluated by training a linear classifier on latent encodings Figure 7 : Facial Attribute Classification.

We compare the performance of our classifier, E y,?? , to a linear classifier trained on latent representations extracted from a trained DIP-VAE .

DIP-VAE is one of the best models for learning disentangled (, or factored,) representations from unlabelled data.

<|TLDR|>

@highlight

Learn representations for images that factor out a single attribute.

@highlight

This paper builds on Conditional VAE GANs to allow attribute manipulation in the synthesis process.

@highlight

This paper proposes a generative model to learn the representation which can separate the identity of an object from an attribute, and extends the autoencoder adversarial by adding an auxiliary network.