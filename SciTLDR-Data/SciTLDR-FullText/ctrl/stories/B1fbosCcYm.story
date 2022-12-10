The ability to look multiple times through a series of pose-adjusted glimpses is fundamental to human vision.

This critical faculty allows us to understand highly complex visual scenes.

Short term memory plays an integral role in aggregating the information obtained from these glimpses and informing our interpretation of the scene.

Computational models have attempted to address glimpsing and visual attention but have failed to incorporate the notion of memory.

We introduce a novel, biologically inspired visual working memory architecture that we term the Hebb-Rosenblatt memory.

We subsequently introduce a fully differentiable Short Term Attentive Working Memory model (STAWM) which uses transformational attention to learn a memory over each image it sees.

The state of our Hebb-Rosenblatt memory is embedded in STAWM as the weights space of a layer.

By projecting different queries through this layer we can obtain goal-oriented latent representations for tasks including classification and visual reconstruction.

Our model obtains highly competitive classification performance on MNIST and CIFAR-10.

As demonstrated through the CelebA dataset, to perform reconstruction the model learns to make a sequence of updates to a canvas which constitute a parts-based representation.

Classification with the self supervised representation obtained from MNIST is shown to be in line with the state of the art models (none of which use a visual attention mechanism).

Finally, we show that STAWM can be trained under the dual constraints of classification and reconstruction to provide an interpretable visual sketchpad which helps open the `black-box' of deep learning.

Much of the current effort and literature in deep learning focuses on performance from a statistical pattern recognition perspective.

In contrast, we go back to a biological motivation and look to build a model that includes aspects of the human visual system.

The eminent computational neuroscientist David Marr posited that vision is composed of stages which lead from a two dimensional input to a three dimensional contextual model with an established notion of object BID27 .

This higher order model is built up in the visual working memory as a visual sketchpad which integrates notions of pattern and texture with a notion of pose BID3 .

Visual attention models often draw inspiration from some of these concepts and perform well at various tasks BID0 BID1 BID12 BID18 BID38 .

Inspired by vision in nature, visual attention corresponds to adaptive filtering of the model input, typically, through the use of a glimpsing mechanism which allows the model to select a portion of the image to be processed at each step.

Broadly speaking, visual attention models exist at the crux of two key challenges.

The first is to separate notions of pose and object from visual features.

The second is to effectively model long range dependencies over a sequence of observations.

Various models have been proposed and studied which hope to enable deep networks to construct a notion of pose.

For example, transformational attention models learn an implicit representation of object pose by applying a series of transforms to an image BID18 BID0 .

Other models such as Transformational Autoencoders and Capsule Networks harness an explicit understanding of positional relationships between objects BID16 BID36 .

Short term memories have previously been studied as a way of improving the ability of Recurrent Neural Networks (RNNs) to learn long range dependencies.

The ubiquitous Long Short-Term Memory (LSTM) network is perhaps the most commonly used example of such a model BID17 .

More recently, the fast weights model, proposed by BID2 provides a way of imbuing recurrent networks with an ability to attend to the recent past.

From these approaches, it is evident that memory is a central requirement for any method which attempts to augment deep networks with the ability to attend to visual scenes.

The core concept which underpins memory in neuroscience is synaptic plasticity, the notion that synaptic efficacy, the strength of a connection, changes as a result of experience BID33 .

These changes occur at multiple time scales and, consequently, much of high level cognition can be explained in terms of the interplay between immediate, short and long term memories.

An example of this can be found in vision, where each movement of our eyes requires an immediate contextual awareness and triggers a short term change.

We then aggregate these changes to make meaningful observations over a long series of glimpses.

Fast weights BID2 draw inspiration from the Hebbian theory of learning BID14 which gives a framework for how this plasticity may occur.

Furthermore, differentiable plasticity BID28 combines neural network weights with weights updated by a Hebbian rule to demonstrate that backpropagation can be used to learn a substrate over which the plastic network acts as a content-addressable memory.

In this paper, we propose augmenting transformational attention models with a visual working memory in order to move towards two key goals.

Firstly, we wish to understand if visual attention and working memory provide more than just increased efficiency and enable functions that cannot otherwise be achieved.

Secondly, we wish to understand and seek answers to some of the challenges faced when attempting to model such psychophysical concepts in deep networks.

We demonstrate classification performance on MNIST (LeCun, 1998) and CIFAR-10 ( BID21 ) that is competitive with the state of the art and vastly superior to previous models of attention, demonstrating the value of a working memory.

We then demonstrate that it is possible to learn this memory representation in an unsupervised manner by painting images, similar to the Deep Recurrent Attentive Writer (DRAW) network BID12 .

Using this representation, we demonstrate competitive classification performance on MNIST with self supervised features.

Furthermore, we demonstrate that the model can learn a disentangled space over the images in CelebA BID25 , shedding light on some of the higher order functions that are enabled by visual attention.

Finally, we show that the model can perform multiple tasks in parallel and how a visual sketchpad can be used to produce interpretable classifiers.

In this section we discuss related work on visual attention and relevant concepts from psychology and neuroscience which have motivated our approach.

We will use the terms motivation and inspiration interchangeably throughout the paper to capture the notion that our decisions have been influenced by atypical factors.

This is necessary as there are several facets to our approach which may seem nonsensical outside of a biological context.

Attention in deep models can be broadly split into two types: hard attention and soft attention.

In hard attention a non-differentiable step is performed to extract the desired pixels from the image to be used in later operations.

Conversely, in soft attention, differentiable interpolation is used.

The training mechanism differs between the two approaches.

Early hard attention models such as the Recurrent Attention Model (RAM) and the Deep Recurrent Attention Model (DRAM) used the REINFORCE algorithm to learn an attention policy over non-differentiable glimpses BID1 .

More recent architectures such as Spatial Transformer Networks (STNs) BID18 , Recurrent STNs BID38 and the Enriched DRAM (EDRAM) BID0 use soft attention and are trained end to end with backpropagation.

The DRAM model and its derivatives use a two layer LSTM to learn the attention policy.

The first layer is intended to aggregate information from the glimpses and the second layer to observe this information in the context of the whole image and decide where to look next.

The attention models described thus far predominantly focus on single and multi-object classification on the MNIST and Street View House Numbers BID10 datasets respectively.

Conversely, the DRAW network from BID12 is a fully differentiable spatial attention model that can make a sequence of updates to a canvas in order to draw the input image.

Here, the canvas acts as a type of working memory which is constrained to also be the output of the network.

Figure 1 : The two layer Hebb-Rosenblatt memory architecture.

This is a novel, differentiable module inspired by Rosenblatts perceptron that can be used to 'learn' a Hebbian style memory representation over a sequence and can subsequently be projected through to obtain a latent space.

DISPLAYFORM0 A common theme to these models is the explicit separation of pose and feature information.

The Two Streams Hypothesis, suggests that in human vision there exists a 'what' pathway and a 'where' pathway BID9 .

The 'what' pathway is concerned with recognition and classification tasks and the 'where' pathway is concerned with spatial attention and awareness.

In the DRAM model, a multiplicative interaction between the pose and feature information, first proposed in BID23 , is used to emulate this.

The capacity for humans to understand pose has been studied extensively.

Specifically, there is the notion that we are endowed with two methods of spatial understanding.

The first is the ability to infer pose from visual cues and the second is the knowledge we have of our own pose BID8 .Memory is generally split into three temporal categories, immediate memory, short term (working) memory and long term memory BID33 .

Weights learned through error correction procedures in a neural network are analogous to a long term memory in that they change gradually over the course of the models existence.

We can go further to suggest that the activation captured in the hidden state of a recurrent network corresponds to an immediate memory.

There is, however, a missing component in current deep architectures, the working memory, which we study here.

We will later draw inspiration from the Baddeley model of working memory BID3 which includes a visual sketchpad that allows individuals to momentarily create and revisit a mental image that is pertinent to the task at hand.

This sketchpad is the integration of 'what' and 'where' information that is inferred from context and/or egomotion.

Here we will outline our approach to augmenting models of visual attention with a working memory in order to seek a deeper understanding of the value of visual attention.

In Section 3.1 we will detail the 'plastic' Hebb-Rosenblatt network and update policy that comprise our visual memory.

Then in Section 3.2 we will describe the STAWM model of visual attention which is used to build up the memory representation for each image.

The key advantage of this approach that will form the basis of our experimentation lies in the ability to project multiple query vectors through the memory and obtain different, goal-oriented, latent representations for one set of weights.

In Section 3.3 we will discuss the different model 'heads' which use these spaces to perform different tasks.

We will now draw inspiration from various models of plasticity to derive a differentiable learning rule that can be used to iteratively update the weights of a memory space, during the forward pass of a deep network.

Early neural network models used learning rules derived from the Hebbian notion of synaptic plasticity BID14 BID4 BID35 .

The phrase 'neurons that fire together wire together' captures this well.

If two neurons activate for the same input, we increase the synaptic weight between them; otherwise we decrease.

In a spiking neural network this activation is binary, on or off.

However, we can obtain a continuous activation by integrating over the spike train in a given window ∆t.

Although the information in biological networks is typically seen as being encoded by the timing of activations and not their number, this spike train integral can approximate it to a reasonable degree BID6 Figure 2 : The attentional working memory model which produces an update to the memory.

See Section 3.2 for a description of the structure and function of each module.

a novel short-term memory unit that can be integrated with other networks that are trained with backpropagation.

Consider the two layer network shown in Figure 1 with an input layer L I and output layer L II , each of size M with weights W ∈ R M ×M .

For a signal vector e ∈ R M , propagating through L I , L II will activate according to some activation function φ of the projection We.

The increase in the synaptic weight will be proportional to the product of the activations in the input and output neurons.

We can model this with the outer product (denoted ⊗) to obtain the increment expression DISPLAYFORM0 where at each step, we reduce the weights matrix by some decay rate δ and apply the increment with some learning rate η.

This rule allows for the memory to make associative observations over the sequence of states such that salient information will become increasingly prevalent in the latent space.

However, if we initialize the weights to zero, the increment term will always be zero and nothing can be learned.

We could perhaps consider a different initialisation such as a Gaussian but this would be analogous to implanting a false memory, which may be undesirable.

Instead, a solution is offered in the early multi layer perceptron models of BID35 .

Here, neurons in the first layer transmit a fixed amount, θ, of their input to a second layer counterpart as well as the weighted signal, as seen in Figure 1 .

Combining both terms we obtain the final, biologically inspired learning rule that we will herein refer to as the Hebb-Rosenblatt rule, DISPLAYFORM1 Note that if we remove the projection term (We) and set θ = 0 and φ(x) = x we obtain the term used in the fast weights model of BID2 which also shares some similarities with the rules considered in BID28 .

Analytically, this learning rule can be seen to develop high values for features which are consistently active and low values elsewhere.

In this way the memory can make observations about the associations between salient features from a sequence of images.

Here we describe in detail the STAWM model shown in Figure 2 .

This is an attention model that allows for a sequence of sub-images to be extracted from the input so that we can iteratively learn a memory representation from a single image.

STAWM is based on the Deep Recurrent Attention Model (DRAM) and uses components from Spatial Transformer Networks (STNs) and the Enriched DRAM (EDRAM) BID1 BID18 BID0 .

The design is intented to be comparable with previous models of visual attention whilst preserving proximity to the discussed psychological concepts, allowing us to achieve the goals set out in the introduction.

At the core of the model, a two layer RNN defines an attention policy over the input image.

As with EDRAM, each glimpse is parameterised by an affine matrix, A ∈ R 3×2 , which is sampled from the output of the RNN.

At each step, A is used to construct a flow field that is interpolated over the image to obtain a fixed size glimpse in a process denoted as the glimpse transform, t A :

R Hi×Wi → R Hg×Wg , where H g × W g and H i × W i are the sizes of the glimpse and image respectively.

Typically the glimpse is a square of size S g such that H g = W g = S g .

Features obtained from the glimpse are then combined with the location features and used to update the memory with Equation 2.

The context and glimpse CNNs are used to obtain features from the image and the glimpses respectively.

The context CNN is given the full input image and expected to establish the contextual information required when making decisions about where to glimpse.

The precise CNN architecture depends on the dataset used and can be found in Appendix D. We avoid pooling as we wish for the final feature representation to preserve as much spatial information as possible.

Output from the context CNN is used as the initial input and initial hidden state for the emission RNN.

From each glimpse we extract features with the glimpse CNN which typically has a similar structure to the context CNN.Aggregator and Emission RNNs: The aggregator and emission RNNs, shown in Figure 2 , formulate the glimpse policy over the input image.

The aggregator RNN is intended to collect information from the series of glimpses in its hidden state which is initialised to zero.

The emission RNN takes this aggregate of knowledge about the image and an initial hidden state from the context network to inform subsequent glimpses.

By initialising the hidden states in this way, we expect the model to learn an attention policy which is motivated by the difference between what has been seen so far and the total available information.

We use LSTM units for both networks because of their stable learning dynamics BID17 , both with the same hidden size.

As these networks have the same size they can be conveniently implemented as a two layer LSTM.

We use two fully connected layers to transpose the output down to the six dimensions of A for each glimpse.

The last of these layers has the weights initialised to zero and the biases initialised to the affine identity matrix as with STNs.

Memory Network: The memory network takes the output from a multiplicative 'what, where' pathway and passes it through a square Hebb-Rosenblatt memory with weights W ∈ R M ×M , where M is the memory size.

The 'what' and 'where' pathways are fully connected layers which project the glimpse features ('what') and the RNN output ('where') to the memory size.

We then take the elementwise product of these two features to obtain the input to the memory, e ∈ R M .

We can think of the memory network as being in one of three states at any point in time, these are: update, intermediate and terminal.

The update state of the memory is a dynamic state where any signal which propagates through it will trigger an update to the weights using the rule in Equation 2.

In STAWM, this update will happen N times per image, once for each glimpse.

Each of the three learning rates (δ, η and θ) are hyper-parameters of the model.

These can be made learnable to allow for the model to trade-off between information recall, associative observation and the individual glimpse.

However, the stability of the memory model is closely bound to the choices of learning rate and so we derive necessary initial conditions in Appendix A.For the intermediate and terminal states, no update is made to the Hebb-Rosenblatt memory for signals that are projected through.

In the intermediate state we observe the memory at some point during the course of the attention policy.

Conversely, in the terminal state we observe the fixed, final value for the memory after all N glimpses have been made.

We can use the intermediate or terminal states to observe the latent space of our model conditioned on some query vector.

That is, at different points during the glimpse sequence, different query vectors can be projected through the memory to obtain different latent representations.

For a self-supervised setting we can fix the weights of STAWM so that the memory inputs cannot be changed by the optimizer.

We now have an architecture that can be used to build up or 'learn' a Hebb-Rosenblatt memory over an image.

Note that when projecting through the memory, we do not include θe in the output.

This is so that it is not possible for the memory to simply learn the identity function.

We use the output from the context CNN as a base that is projected into different queries for the different aims of the network.

We do this using linear layers, the biases of which can learn a static representation which is then modulated by the weighted image context.

We detach the gradient of the image context so that the context network is not polluted by divergent updates.

In this section we will characterise the two network 'heads' which make use of latent spaces derived from the memory to perform the tasks of classification and drawing.

We will also go on to discuss ways of constraining the glimpse sub-spaces in a variational setting.

Classification:

For the task of classification, the query vector is projected through the memory in exactly the same fashion as a linear layer to derive a latent vector.

This latent representation is then passed through a single classification layer which projects from the memory space down to the number of classes as shown in Figure 3 .

A softmax is applied to the network output and the entire model (including the attenion mechanism) is trained by backpropagation to minimise the categorical cross entropy between the network predictions and the ground truth targets.

Learning to Draw: To construct a visual sketchpad we will use a novel approach to perform the same task as the DRAW network with the auxiliary model in Figure 4 .

Our approach differs from DRAW in that we have a memory that is independent from the canvas and can represent more than just the contents of the reconstruction.

However, it will be seen that an important consequence of this is that it is not clear how to use STAWM as a generative model in the event that some of the skecthes are co-dependent.

The drawing model uses each intermediate state of the memory to query a latent space and compute an update, U ∈ R Hg×Wg , to the canvas, C ∈ R Hi×Wi , that is made after each glimpse.

Computing the update or sketch is straightforward, we simply use a transpose convolutional network ('Transpose CNN') with the same structure as the glimpse CNN in reverse.

However, as the features were observed under the glimpse transform, t A , we allow the emission network to further output the parameters, A −1 ∈ R 3×2 , of an inverse transform, t A −1 :

R Hg×

Wg → R Hi×Wi , at the same time.

The sketch will be warped according to t A −1 before it is added to the canvas.

To add the sketch to the canvas there are a few options.

We will consider two possibilities here: the addition method and the Bernoulli method.

The addition method is to simply add each update to the canvas matrix and finally apply a sigmoid after the glimpse sequence has completed to obtain pixel values.

This gives an expression for the final canvas DISPLAYFORM0 where C 0 ∈ {

−6} Hi×Wi , S is the sigmoid function and N is the total number of glimpses.

We set C 0 to −6 so that when no additions are made the canvas is black and not grey.

The virtue of this method is its simplicity, however, for complex reconstructions, overlapping sketches will be required to counteract each other such that they may not be viewable independently in an interprettable manner.

An alternative approach, the Bernoulli method, could help to prevent these issues.

Ideally, we would like the additions to the canvas to be as close as possible to painting in real life.

In such a case, each brush stroke replaces what previously existed underneath it, which is dramatically different to the effect of the addition method.

In order to achieve the desired replacement effect we can allow the model to mask out areas of the canvas before the addition is made.

We therefore add an extra channel to the output of the transpose CNN, the alpha channel.

This mask, P ∈ R Hg×Wg , is warped by t A −1 as with the rest of the sketch.

A sigmoid is applied to the output from the transpose CNN so that the mask contains values close to one where replacement should occur and close to zero elsewhere.

Ideally, the mask values would be precisely zero or one.

To achieve this, we could take P as the probabilities of a Bernoulli distribution and then draw B ∼ Bern(t A −1 (S(P)).

However, the Bernoulli distribution cannot be sampled in a differentiable manner.

We will therefore use an approximation, the Gumbel-Softmax BID19 or Concrete BID26 distribution, which can be differentiably sampled using the reparameterization trick.

The Concrete distribution is modulated with the temperature parameter, τ , such that lim τ →0 Concrete(p, τ ) = Bern(p).

We can then construct the canvas iteratively with the expression DISPLAYFORM1 where C 0 ∈ {0} Hi×Wi and is the elementwise multiplication operator.

Note that this is a simplified over operator from alpha compositing where we assume that objects already drawn have an alpha value of one BID32 .Constraining Glimpse Sub-spaces: A common approach in reconstructive models is to model the latent space as a distribution whose shape over a mini-batch is constrained using a Kullback-Liebler divergence against a (typically Gaussian) prior distribution.

We employ this variational approach as shown in Figure 4 where, as with Variational Auto-Encoders (VAEs), the latent space is modelled as the variance (σ 2 ) and mean (µ) of a multivariate Gaussian with K components and diagonal covariance BID20 .

For input x ∈ R Hi×Wi and some sample z ∈ R K from the latent space, the β-VAE BID15 uses the objective DISPLAYFORM2 For our model, we do not have a single latent space but a sequence of glimpse sub-spaces DISPLAYFORM3 For the addition method, we can simply concatenate the sub-spaces to obtain z ∈ R N ×K and use the divergence term above.

In the Bernoulli method, however, we can derive a more appropriate objective by acknowledging that outputs from the decoder are conditioned on the joint distribution of the glimpse sub-spaces.

In this case, assuming that elements of G are conditionally independent, following from the derivation given in Appendix B, we have DISPLAYFORM4

In this section we discuss the results that have been obtained using the STAWM model.

The training scheme and specific architecture details differ for each setting, for full details see Appendix D. For the memory we use a ReLU6 activation, y = min(max(x, 0), 6) (Krizhevsky and Hinton, 2010) on both the input and output layers.

Following the analysis in Appendix A, we initialise the memory learning rates δ, η and θ to 0.2, 0.4 and 0.5 respectively.

For some of the experiments these are learnable and are updated by the optimiser during training.

Our code is implemented in PyTorch BID31 with torchbearer BID13 and can be found at https://github.

com/iclr2019-anon/STAWM.

Examples in all figures have not been cherry picked.

We first perform classification experiments on handwritten digits from the MNIST dataset (LeCun, 1998) using the model in Figure 3 .

We perform some experiments with S g = 8 in order to be comparable to previous results in the literature.

We also perform experiments with S g = 28 to see whether attention can be used to learn a positional manifold over an image by subjecting it to different transforms.

The MNIST results are reported in TAB1 and show that STAWM obtains superior classification performance on MNIST compared to the RAM model.

It can also be seen that the over-complete strategy obtains performance that is competitive with the state of the art of around 0.25% for a single model BID36 , with the best STAWM model from the 5 trials obtaining a test error of 0.31%.

This suggests an alternative view of visual attention as enabling the model to learn a more powerful representation of the image.

We experimented with classification on CIFAR-10 ( BID21 ) but found that the choice of glimpse CNN was the dominating factor in performance.

For example, using MobileNetV2 BID37 as the glimpse CNN we obtained a single run accuracy of 93.05%.

(b) MNIST Self-supervised Model Error DBM, Dropout BID39 0.79% Adversarial BID11 0.78% Virtual Adversarial BID29 0.64% Ladder BID34 0.57% STAWM, S g = 6, N = 12 0.77%(c) CIFAR-10 Self-supervised Model ErrorBaseline β-VAE 63.44% ±0.31 STAWM, S g = 16, N = 8 55.40% ±0.63

Following experimentation with MNIST we observed that there are three valid ways to draw using the addition method.

First, the model could simply compress the image into a square equal to the glimpse size and decompress later.

Second, the model could learn to trace lines such that all of the notion of object is contained in the pose information instead of the features.

Finally, the model could learn a pose invariant, parts based representation as we originally intended.

The most significant control of this behaviour is the glimpse size.

At S g = 8 or above, enough information can be stored to make a simple compression the most successful option.

Conversely, at S g = 4 or below, the model is forced to simply draw lines.

In light of these observations, we use N = 12 with S g = 6 to obtain an appropriate balance.

Sample update sequences for these models are shown in FIG6 in Appendix C. For S g = 6 it can be seen that the model has learned a repeatable parts-based representation which it uses to draw images.

In this way the model has learned an implicit notion of class.

We also experimented with painting images in CIFAR-10.

To establish a baseline we also show reconstructions from a reimplementation of β-VAE BID15 .

To be as fair as possible, our baseline uses the same CNN architecture and latent space size as STAWM.

This is still only an indicative comparison as the two models operate in fundamentally different ways.

Autoencoding CIFAR-10 is a much harder task due to the large diversity in the training set for relatively few images.

However, our model significantly outperforms the baseline with a terminal mean squared error of 0.0083 ±0.0006 vs 0.0113 ±0.0001 for the VAE.

On inspection of the glimpse sequence given in Appendix C we can see that although STAWM has not learned the kind of parts-based representation we had hoped for, it has learned to scan the image vertically and produce a series of slices.

We again experimented with different glimpse sizes and found that we were unable to induce the desired behaviour.

The reason for this seems clear; any 'edges' in the output where one sketch overlaps another would have values that are scaled away from the target, resulting in a constant pressure for the sketch space to expand.

This is not the case in MNIST, where overlapping values will only saturate the sigmoid.

8 2 9 0 5 2 7 6 3 7 5 9 6 2 5 5 7 2 3 7 7 6 7 5 2 3 7 4 6 3 8 2 0 7 4 3 8 4 7 9 3 5 8 5 9 8 0 3 8 7 Figure 7 : Top: sketchpad results and associated predictions for a sample of misclassifications.

Bottom: associated input images and target classes.

We can fix the weights of the STAWM model learned from the drawing setting and add the classification head to learn to classify using the self-supervised features.

In this case, the only learnable parameters are the weights of the two linear layers and the learning rates for the memory construction.

This allows the model to place greater emphasis on previous glimpses to aid classification.

As can be seen from TAB1 , we obtain performance on the selfsupervised MNIST features that is competitive with the state of the art models, none of which use an attention mechanism.

The self-supervised performance on CIFAR-10 is less competitive with the state of the art, but does show a clear improvement over the results obtained using the baseline VAE.

As discussed previously, the canvas for the drawing model can be constructed using the Bernoulli method given in Section 3.3.

In this setting we trained on the CelebA dataset BID25 which contains a large number of pictures of celebrity faces and add the KL divergence for the joint distribution of the glimpse spaces with a Gaussian prior given in Equation 25 in Appendix B. An advantage of the Bernoulli method is that we sample an explicit mask for the regions drawn at each step.

If interesting regions are learned, we can use this as an unsupervised segmentation mask.

FIG3 shows the result of the drawing model on CelebA along with the target image and the learned mask from the final glimpse elementwise multiplied with the ground truth.

Here, STAWM has learned to separate the salient face region from the background in an unsupervised setting.

Analytically, the KL term we have derived will ask for the subsequent glimpse spaces to transition smoothly as the image is observed.

Coupled with the fact that the memory can only build up a representation over the sequence, it follows that the model will learn to sketch increasingly complex areas with each new glimpse.

This is precisely the case, as seen from the sketch sequence, FIG8 in Appendix C.

One of the interesting properties of the STAWM model is the ability to project different things through the memory to obtain different views on the latent space.

We can therefore use both the drawing network and the classification network in tandem by simply summing the losses.

We scale each loss so that one does not take precedence over the other.

We also must be careful to avoid overfitting so that the drawing is a good reflection of the classification space.

For this experiment, with S g = 8, the terminal classification error for the model is 1.0%.

We show the terminal state of the canvas for a sample of misclassifications in Figure 7 .

Here, the drawing gives an interesting reflection of what the model 'sees' and in many cases the drawn result looks closer to the predicted class than to the target.

For example, in the rightmost image, the model has drawn and predicted a '2' despite attending to a '7'.

This advantage of memories constructed as the weights of a layer, coupled with their ability to perform different tasks well with shared information, is evident and a clear point for further research.

In this paper we have described a novel, biologically motivated short term attentive working memory model (STAWM) which demonstrates impressive results on a series of tasks and makes a strong case for further study of short term memories in deep networks.

As well as demonstrating competitive classification results on MNIST and CIFAR-10, we have shown that the core model can be used for image reconstruction and for disentangling foreground from background in an unsupervised setting with CelebA. Finally, we have given a concrete example of how a model augmented with a visual sketchpad can 'describe what it sees' in a way that is naturally interpretable for humans.

It is easy to see how similar systems could be used in future technologies to help open the 'black-box' and understand why a decision was made, through the eyes of the model that made it.

Furthermore, we have explored the notion that building up a memory representation over an attention policy, coupled with a smooth changing latent space can result in a movement from simple to complex regions of a scene.

This perhaps gives us some insight into how humans learn to attend to their environment when endowed with a highly capable visual memory.

Future work will look to see if variants of this model can be used to good effect on higher resolution images.

Experimentation and analysis should also be done to further understand the dynamics of the Hebb-Rosenblatt memory and the representation it learns.

We further intend to investigate if the memory model can be used for other applications such as fusion of features from multi-modal inputs.

Finally, we will look further into the relationship between visual memories and saliency.

A STABILISING THE MEMORY The working memory model described in the paper exhibits a potential issue with stability.

It is possible for the gradient to explode, causing damaging updates which halt learning and from which the model cannot recover.

In this section we will briefly demonstrate some properties of the learning rule in Equation 2 and derive some conditions under which the dynamics are stable.

We will broadly follow the method of with minor alterations for our approach.

We use the terms stimuli and response to represent input to and output from a neuron respectively.

We will consider a sequence of DISPLAYFORM0 Hg×Wg , presented to L I when attending to a single image.

ν as the identity input to some ν ∈ L II .

We then define γ DISPLAYFORM1 ν , the projection of e (i) through the weights matrix W ∈ R M ×M .

The total input to ν if stimulus i is presented to L I at time t is the sum of these two terms given in Equation 9.

DISPLAYFORM2 Suppose that a stimulus i is presented at time t 0 for some period ∆t.

The subsequent change in the weight, w µν , of some connection is defined in Equation 10, where φ II is a nonlinear activation function of neurons in L II .

Note that the change in the matrix W for this step is the learning rule in Equation 2.

DISPLAYFORM3 From FORMULA1 and (8) we derive Equation 11 which gives the change in the weighted component γ (q) ν for some query stimulus q over a single time step.

We omit the subscript ν for brevity.

The response of L II to q is the latent representation derived from the memory during the glimpse sequence.

DISPLAYFORM4 The next step in is to define some sequence over the set of possible stimuli to be presented in order.

We deviate slightly here as our sequence is not drawn from a bounded set.

Instead, as we have seen, each glimpse is a sample from the manifold space of affine transforms over the image.

As such, in Equation 12 we generalise Equation 11 to represent the change in the query response for some arbitrary point, n, in the sequence, with stimulus g n at time t + n∆t.

Summing over the whole sequence we obtain Equation 13.

We can now obtain an expression for the gradient of Equation 13 by dividing by N ∆t in the limit of ∆t → 0 (Equation 14).

DISPLAYFORM5 DISPLAYFORM6 Although Equation 14 is only an approximation of the dynamics of the system, we have demonstrated stability under certain conditions: firstly, the input to the memory network must not vary with time.

This is satisfied in the case where the feature vector is not dependent on the output of a recurrent network.

It is possible to extend this proof to incorporate such networks BID35 , however, that is outside the scope of this paper.

Secondly, the activation functions must be nonnegative and have an upper bound (as with ReLU6 or Sigmoid).

This introduces an interesting similarity as there is an upper bound to the number of times a real neuron can fire in a given window, governed by its refractory period.

Finally, we can observe that Equation 15 is nondecreasing iff η is greater than δ, δ is positive and η and θ are nonnegative.

The section details the derivation of the joint KL divergence term for our glimpse sequence, adapted from a derivation given in BID7 .

For a sequence of glimpse sub-spaces with K components, G = (g n ) N n=0 , g n ∈ R K , we now derive a term for the KL divergence between the posterior and the prior for the joint distribution of the glimpses, D KL (q φ (g 0 , . . .

, g N | x) p(g 0 , . . .

, g N )).

Assuming that elements of G are conditionally independent we have q φ (G | x) = N n=0 q φ (g n | x) , and (17) DISPLAYFORM0 We can therefore re-write the joint KL term and simplify DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 Since E q φ (gi | x) , g i ∈ G \ {g n } does not depend on n, we have DISPLAYFORM4 DISPLAYFORM5 For our experiments we set the prior to be an isotropic unit Gaussian (p(z) = N (0, I)) yielding the KL term DISPLAYFORM6 where (µ n , σ n ) = g n .

Note that in the case where each glimpse has the same prior, when taken in expectation over the latent dimensions, this summation is identical to concatenating the spaces.

This Appendix gives the drawing stages diagrams for each of the drawing experiments.

As discussed in the paper, there are multiple ways the model can learn to reconstruct line drawings such as those from the MNIST dataset.

FIG6 show the compression, parts based and the line drawing modes that are obtained for different glimpse sizes.

FIG7 shows the canvas sequence for CIFAR-10.

Here, as discussed, the model is only able to compress the image with each glimpse and subsequently decompress with each sketch.

FIG8 , however, shows the sketch sequence for the CelebA dataset, using the Bernoulli method to update the canvas.

Here, the model has learned to transition smoothly from reconstructing low-information regions of the image to high-information regions resulting in a separation of background and foreground.

<|TLDR|>

@highlight

A biologically inspired working memory that can be integrated in recurrent visual attention models for state of the art performance

@highlight

Introduces a new network architecture inspired by visual attentive working memory and applies it to classification tasks and using it as a generative model

@highlight

The paper augments the recurrent attention model with a novel Hebb-Rosenblatt working memory model and achieves competitive results on MNIST