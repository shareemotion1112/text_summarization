Learning representations of data is an important issue in machine learning.

Though GAN has led to significant improvements in the data representations, it still has several problems such as unstable training, hidden manifold of data, and huge computational overhead.

GAN tends to produce the data simply without any information about the manifold of the data, which hinders from controlling desired features to generate.

Moreover, most of GAN’s have a large size of manifold, resulting in poor scalability.

In this paper, we propose a novel GAN to control the latent semantic representation, called LSC-GAN, which allows us to produce desired data to generate and learns a representation of the data efficiently.

Unlike the conventional GAN models with hidden distribution of latent space, we define the distributions explicitly in advance that are trained to generate the data based on the corresponding features by inputting the latent variables that follow the distribution.

As the larger scale of latent space caused by deploying various distributions in one latent space makes training unstable while maintaining the dimension of latent space, we need to separate the process of defining the distributions explicitly and operation of generation.

We prove that a VAE is proper for the former and modify a loss function of VAE to map the data into the pre-defined latent space so as to locate the reconstructed data as close to the input data according to its characteristics.

Moreover, we add the KL divergence to the loss function of LSC-GAN to include this process.

The decoder of VAE, which generates the data with the corresponding features from the pre-defined latent space, is used as the generator of the LSC-GAN.

Several experiments on the CelebA dataset are conducted to verify the usefulness of the proposed method to generate desired data stably and efficiently, achieving a high compression ratio that can hold about 24 pixels of information in each dimension of latent space.

Besides, our model learns the reverse of features such as not laughing (rather frowning) only with data of ordinary and smiling facial expression.

Developing generative model is a crucial issue in artificial intelligence.

Creativity was a human proprietary, but many recent studies have attempted to make machines to mimic it.

There has been an extensive research on generating data and one of them, generative adversarial network (GAN), has led to significant achievements, which might be helpful to deep learning model because, in general, lots of data result in good performance BID12 .

Many approaches to creating data as better quality as possible have been studied: for example, variational auto-encoder (VAE) BID9 and GAN BID4 .

The former constructs an explicit density, resulting in an explicit likelihood which can be maximized, and the latter constructs an implicit density BID3 .

Both can generate data from manifold which is hidden to us so that we cannot control the kind of data that we generate.

Because it is costly to structure data manually, we need not only data generation but also automatically structuring data.

Generative models produce only data from latent variable without any other information so that we cannot control what we want to generate.

To cope with this problem, the previous research generated data first and found distributions of features on latent space by investigating the model with data, since the manifold of data is hidden in generative models.

This latent space is deceptive for finding an area which represents a specific feature of our interest; it would Figure 1 : Examples of the manifold.

Left: a complex manifold which can be seen in general models, Right: a relatively simple manifold in the proposed model.

The midpoint M of A and B can be easily calculated in the right manifold, but not in the left one.

The midpoint of A and B is computed as N in the left manifold, which is incorrect. take a long time even if we can find that area.

Besides, in the most of research, generative models had a large latent space, resulting in a low compression rate which leads to poor scalability.

To work out these problems, we propose a model which can generate the data whose type is what we want and learn a representation of data with a higher compression rate, as well.

Our model is based on VAE and GAN.

We pre-define distributions corresponding to each feature and modify the loss function of VAE so as to generate the data from the latent variable which follows the specific distribution according to its features.

However, this method makes the latent space to become a more complex multimodal distribution which contains many distributions, resulting in an instability in training the LSC-GAN.

We prove that this problem can be solved and even made more efficiently by using an auto-encoder model with the theorem in Section 3.

Although the proposed model compresses the data into small manifold, it is well-defined with Euclidean distance as shown in Fig. 1 , which compares the manifolds in general models and in our model.

The distance can be calculated with Euclidean distance in adjacent points but not in far points at the left manifold in Fig. 1 .

However, in the right manifold, we can calculate the distance between points regardless of the distance of them, where we can recognize the manifold more easily as shown in the left side.

Thanks to a relatively simple manifold, it can produce neutral features regardless of their location in latent space, so that all features can be said as independent to each other.

Our main contribution is summarized as follows.• We propose a method to improve the stability of a LSC-GAN with LSC-VAE by performing the weight initialization, and prove it theoretically.• We achieve conditional generation without additional parameters by controlling the latent space itself, rather than adding additional inputs like the existing model for condition generation.• We propose a novel model that automatically learns the ability to process data continuously through latent space control.• Finally, we achieve an efficient compression rate with LSC-GAN based on weight initialization of LSC-VAE.The rest of the paper is organized as follows.

Section 2 reviews the related works and the proposed LSC-GAN model is illustrated in Section 3.

In Section 4, we evaluate the performance of the proposed method with some generated data.

The conclusion and discussion are presented in Section 5.

Many research works have been conducted to generate data such as text, grammar, and images BID24 BID10 BID2 .

We divide the approaches for data generation into three categories: only generation, conditioned generation, and transforming data to have different features.

Several researchers proposed generative models of VAE and GAN BID9 BID4 .

These are basis of the generative models.

Both use maximum likelihood approach, but they have different policies to construct density: explicitly and implicitly.

There are lots of variations of these models.

BID17 constructed deep convolutional GAN (DCGAN) with convolutional neural networks (CNN) for improving the performance with the fact that CNN had been huge adoption in computer vision applications.

BID26 introduced energy-based GAN (EBGAN) using autoencoder in discriminator.

BID5 BID6 b) proposed transferred encoder-decoder GAN (TED-GAN) for stabilizing process of training GAN and used it to classify the data.

These studies focused on high productivity in generation so that they could not control the type of generated data.

Recently, some researchers began to set conditions on the data they generate.

BID20 and BID23 inputted data and conditions together into VAE and generated data whose type is what they want, called conditional VAE (CVAE).

van den Oord et al. FORMULA0 set discrete embedding space for generating a specific data using vector quantized variational auto-encoder (VQ-VAE), but because of discrete space, they could not control latent space continuously.

Larsen et al. FORMULA0 used both VAE and GAN in one generative model.

As they just mixed two models and did not analyzed a latent space, so that the manifold of data was hidden to us.

To generate image with a specific feature, they extracted a visual attribute vector which is a mean of vector in latent space.

BID15 inputted not only data but also conditions into GAN to create data that we want, called conditional GAN (CGAN).

BID0 used mutual information for inducing latent codes (InfoGAN) and BID16 added a condition network that tells the generator what to generate (PPGN) .

These two models needed an additional input to generate the type of data we want.

These studies make us to generate data with condition, but we still do not know about latent space and it is hard to find the location of a specific feature in the latent space.

Therefore, we propose a model that learns to generate concrete features that we want from the latent space determined when LSC-VAE is trained.

Some studies attempted to transfer the given data to others which have different features or even in different domain.

BID21 proposed disentangled representation learning GAN (DRGAN) for pose-invariant face recognition.

BID18 b) tried matching latent space of text and images and finally they translated text to image.

BID25 also translated text to image and generated photo-realistic images conditioned on text by stacking models (StackGAN).

BID27 and BID8 discovered cross-domain relations with CycleGAN and DiscoGAN.

They can translate art style, face features, and bags to shoes.

While other models could only do one conversion task, BID1 proposed StarGAN that could do multiple translation tasks with one model.

These studies have been conducted to transform the data into those in other domains.

However, they could not generate new data without input data.

In addition, the size of latent space of most of them was too large.

We aim to generate conditioned data even with a small size of latent space.

In this section, we present a method to generate the data with the corresponding characteristics by inputting the latent variable which follows the specific distribution in latent space.

As the instability caused by the larger scale of latent space in this process, we use the modified VAE, called LSC-VAE 1 .

As shown in Fig. 2(a) , we train the LSC-VAE with L prior for the data to be projected by the encoder into the desired position in the latent space according to the characteristics of the data.

The trained decoder of the LSC-VAE is used as a generator of LSC-GAN so that the LSC-GAN generates the data with the corresponding features by using latent variables sampled from a specific distribution.

The proposed model is divided into two phases: initializing latent space ( Fig. 2(a) ) and generating data ( Fig. 2(b) ).

In the first phase, latent semantic controlling VAE (LSC-VAE) is trained to project data into a specific location of latent space according to its features, and it learns to reconstruct data which is compressed.

The decoder of LSC-VAE is used in the generator (G) of LSC-GAN in the second phase.

G and discriminator (D) are trained simultaneously so that G can produce data similar to real data as much as possible and that D can distinguish the real from the fake.

The architecture of the generation process is shown in Fig. 2(b) .

Auto-encoder has been traditionally used to represent manifold without supervision.

In particular, VAE, one type of auto-encoders, is one of the most popular approaches to unsupervised learning of complicated distributions.

Since any supervision is not in training process, the manifold constructed is hidden to us.

As we mentioned in Section 1, this is usually too complex to generate the conditioned Figure 2 : (a) The process of pre-defining a latent space.

The LSC-VAE is trained to project the data into the appropriate position on latent space.

(b) Generating process of the proposed method.

The latent space is pre-defined in the process of (a).data.

Therefore, we allow LSC-VAE to learn a representation of data with supervision.

It compresses data into a particular place on latent space according to its features.

The proposed model consists of two modules that encode a data x i to a latent representation z i and decode it back to the data space, respectively.

DISPLAYFORM0 Index i means a feature which is included in data x and latent space z. The encoder is regularized by imposing a prior over the latent distribution P (z).

In general, z ∼ N (0, I) is chosen, but we choose z i ∼ N (µ i , I)for controlling latent space.

In addition, if we want to produce data which has multiple features i, j, we generate data from z ij ∼ N (µ i + µ j , I) 2 .

The loss function of LSC-VAE is as follows.

DISPLAYFORM1 where D KL is the Kullback-Leibler divergence.

The first term of equation 3 is related to reconstruction error and the second term is related to appropriate projection of data to the latent space.

For example, when LSC-VAE projects the data with i− and j−features into the latent space, it is trained to map the data into the pre-defined latent space (N (µ i + µ j , I)) with L prior in equation 3 so as to locate the reconstructed data as similar to the input data according to its characteristics using L V AE .

Therefore, LSC-VAE can be used in initializing GAN and it is demonstrated that LSC-VAE is valid and efficient for LSC-GAN in the next section.

GAN has led to significant improvements in data generation BID4 .

The basic training process of GAN is to adversely interact and simultaneously train G and D. However, because the original GAN has a critical problem, unstable process of training BID17 , the least squares GAN (LS-GAN) is proposed to reduce the gap between the distributions of real data and fake data by BID14 .

FIG1 shows the objective function of the LS-GAN.

p data is the probability distribution of the real data.

G(z) is generated data from a probability distribution p z , and it is distinguished from the real by D. DISPLAYFORM0 The main differences of the proposed model with VAE-GAN and LS-GAN is that LSC-GAN is based on LSC-VAE for initializing a latent space to control it.

To produce the type of data we want, we just input latent variable z i ∼ N (µ i , I) to G, if the data has i−feature.

Besides, we add the encoder of LSC-VAE into LSC-GAN to make sure that the generated data actually have the desired features.

The encoder projects back to latent space so as to be trained to minimize the difference between latent space where data is generated and the space where the compressed data is projected.

Equation 5 is about loss of D and loss of encoder and G. DISPLAYFORM1

Since the original GAN has disadvantage that the generated data are insensible because of the unstable learning process of the G, we pre-train G with decoder of LSC-VAE.

The goal of the learning process of generating data of G is the same as equation 6 from equation 5, and it is equivalent to that of equation 7.

However, it is not efficient to pre-train the G, because it depends on the parameters of the D. Therefore, we change this equation to equation 8 again, and it is represented only by the parameters of G. In this paper, to train the G with equation 8, we use the decoder of LSC-VAE, which is trained by using Dec(Enc(x))

≈

x.

The result of LSC-VAE is that DISPLAYFORM0 can reach a goal of GAN (p data ≈ p G ) stably, which is proved by Theorem 1 and 2.

DISPLAYFORM1 where X i is real dataset with i−feature.

From the game theory point of view, the GAN converges to the optimal point when G and D reach the Nash equilibrium.

In this section, let p G be the probability distribution of data created from G. We show that if G(z) ≈ x, i.e., p data ≈ p G , the GAN reaches the Nash equilibrium.

We define DISPLAYFORM0 We train G and D to minimize J(D, G) and K(D, G) for each.

Then, we can define the Nash equilibrium of the LSC-GAN as a state that satisfies equation equation 9 and equation equation 10.

Fully trained G and D are denoted by G * and D * , respectively.

DISPLAYFORM1 Theorem 1.

If p data ≈ p G almost everywhere, then the Nash equilibrium of the LSC-GAN is reached.

Before proving this theorem, we need to prove the following two lemmas.

DISPLAYFORM2 The proof of Lemma 1 and 2 were discussed by Kim et al. BID7 .

We assume that p data ≈ p G .

From Lemma 1 and Lemma 2, if p data ≈ p G , then J(D, G) and K(D, G) both reach minima.

Therefore, the proposed GAN reaches the Nash equilibrium and converges to optimal points.

By theorem 1, GAN converges when p d ≈ p g , and it is done to some extent by the modified VAE, i.e. | p DISPLAYFORM3 Therefore, the proposed method is useful to initialize the weight of the generative model.

However, it shows only validity of using VAE when learning GAN.

We prove that it is also efficient by proving theorem 2.

Assume that a model f is well-trained if and only if ∇L f ≈ 0, where L f is the loss function of f .Theorem 2.

Let en k , de k be k epoch-trained encoder and decoder whose goal is DISPLAYFORM4 Proof.

Notice that the derivative is unique, and a derivative of linear function is itself.

Since en and de are trained with L V AE and L prior , the following statement is satisfied.

DISPLAYFORM5 By differentiating the formula, DISPLAYFORM6 Since the derivative of linear function is itself, it derives to DISPLAYFORM7 With the fact that D(x) = 1, ∀x ∈ X and equation 11, it finally derives to DISPLAYFORM8 By theorem 1 and 2, the proposed learning process is valid and efficient.

To verify the performance of the proposed model, we use the celebA dataset BID13 .

It is a large-scale face attributes dataset.

We crop the initial 178×218 size to 138×138, and resize them as 64×64.

We use 162,769 images in celebA and 14 attributes: black hair, blond hair, gray hair, male, female, smile, mouth slightly open, young, narrow eyes, bags under eyes, mustache, eyeglasses, pale skin, and chubby.

We assign 20 dimensions to each feature and set mean of the i th -20 dimensions as 1.

For example, if an image has i-feature, the elements of i * 20 th to (i + 1) * 20 th of the image's latent variable are 1 in average and 0 in the remainder and we denote that latent variable as n i .

As shown in FIG0 , we generate images from a specific latent space by using LSC-GAN.

The images in the first column are generated to have 'female' and 'blond hair' features.

We confirm that the condition works well.

The images in the remaining columns are transformed using equation 15 for the features listed below.

For example, if we generate an image x i which has i−feature from the latent variable z i , we add n j to add j−feature into the image.

DISPLAYFORM0 where x ij is an image which has i− and j−features, and z i is the latent variable for i−feature.

To show that the proposed model does not simply memorize data but understand features of data and generate them, we generate images from a series between two random images as in DCGAN.

As shown in FIG1 , the change between images is natural so that we can say that the latent space of LSC-GAN is a manifold.

Besides, the images in the middle column have both features of images in leftmost and rightmost, resulting in more simple manifold as shown in Fig. 1 .Unlike other GAN models, the LSC-GAN fully understands features of data so as to generate data including inverse-feature.

We only train the model about the presence of the 'pale skin' and 'smile' features, but the model also learned about the reverse of 'pale skin' and 'smile' automatically as shown in the fourth and the ninth column of FIG2 .

Besides, if we assign a value of 2 rather than 1 to the average of latent variable which is related to 'mustache', we can see that more mustaches are created in the last column in FIG2 .

Therefore, our model can automatically infer and generate the data with inverse-feature that do not exist in the dataset.

This shows that the proposed model has the ability to deduce a negative feature by itself although only positive features are used in trainingTo verify the proposed model, we conduct subjective test about the quality of the generated data.

We generate data by using DCGAN, EBGAN, and the proposed GAN.

We randomly choose 25 generated data for each model.

We perform the subjective test on 30 subjects and ask them to evaluate the quality of the generated data in 5 ways: very low, low, medium, high, and very high.

We collect the results of 750 questionnaires, which are the evaluated result of 25 generated images by 30 subjects, and summarize them in TAB0 .

We score 1,2,3,4, and 5 points for each evaluation result which is shown in the last column in TAB0 .

Our model not only generates images according to input conditions, but also compress efficiently.

We calculate the compression rate with rate= size inputdata /size bottleneck /#classes.

As shown in TAB1 , our proposed model has the best compression rate compared to others.

This proves experimentally that LSC-VAE, theoretically proven with theorems 1 and 2, has been helpful in initializing the weights of the LSC-GAN, and it can achieve good performance even in small latent spaces.

In this paper, we address some of significant issues in generative models: unstable training, hidden manifold of data, and extensive hardware resource.

To generate a data whose type is what we want, we propose a novel model LSC-GAN which can control a latent space to generate the data that we want.

To deal with a larger scale of latent space cause by deploying various distributions in one latent space, we use the LSC-VAE and theoretically prove that it is a proper method.

Also, we confirm that the proposed model can generate data which we want by controlling the latent space.

Unlike the existing generative model, the proposed model deals with features continuously, not discretely and compresses the data efficiently.

Based on the present findings, we hope to extend LSC-GAN to more various datasets such as ImageNet or voice dataset.

In future work, we plan to conduct more experiments with various parameters to confirm the stability of model.

We will also experiment by reducing the dimension of the latent space to verify that the proposed model is efficient.

Besides, since the encoder can project the data to the latent space according to the features inherent in data, it could be used as a classifier.

<|TLDR|>

@highlight

We propose a generative model that not only produces data with desired features from the pre-defined latent space but also fully understands the features of the data to create characteristics that are not in the dataset.