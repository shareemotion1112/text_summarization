We've seen tremendous success of image generating models these years.

Generating images through a neural network is usually pixel-based, which is fundamentally different from how humans create artwork using brushes.

To imitate human drawing, interactions between the environment and the agent is required to allow trials.

However, the environment is usually non-differentiable, leading to slow convergence and massive computation.

In this paper we try to address the discrete nature of software environment with an intermediate, differentiable simulation.

We present  StrokeNet, a novel model where the agent is trained upon a well-crafted neural approximation of the painting environment.

With this approach, our agent was able to learn to write characters such as MNIST digits faster than reinforcement learning approaches in an unsupervised manner.

Our primary contribution is the neural simulation of a real-world environment.

Furthermore, the agent trained with the emulated environment is able to directly transfer its skills to real-world software.

To learn drawing or writing, a person first observes (encodes) the target image visually and uses a pen or a brush to scribble (decode), to reconstruct the original image.

For an experienced painter, he or she foresees the consequences before taking any move, and could choose the optimal action.

Stroke-based image generation is fairly different from traditional image generation problems due to the intermediate rendering program.

Raster-based deep learning approaches for image generation allow effective optimization using back-propagation.

While for stroke-based approaches, rather than learning to generate the image, it is more of learning to manipulate the painting program.

An intuitive yet potentially effective way to tackle the problem is to first learn this mapping from "stroke data" to the resulting image with a neural network, which is analogous to learning painting experience.

An advantage of such a mapping over software is that it provides a continuous transformation.

For any painting program, the pixel values of an image are calcuated based on the coordinate points along the trajectory of an action.

Specific pixels are indexed by the discrete pixel coordinates, which cuts the gradient flow with respect to the action.

In our implementation, the indexing is done by an MLP described in Section 3.We further define "drawing" by giving a formal definition of "stroke".

In our context, a "stroke" consists of color, brush radius, and a sequence of tuples containing the coordinate and pressure of each point along the trajectory.

We will later describe this in detail in Section 3.Based on these ideas, we train a differentiable approximator of our painting software, which we call a "generator".

We then tested the generator by training a vanilla CNN as an agent that encodes the image into "stroke" data as an input for the environment.

Our proposed architecture, StrokeNet, basically comprises the two components, a generator and an agent.

Finally, an agent is trained to write and draw pictures of several popular datasets upon the generator.

For the MNIST (LeCun & Cortes, 2010 ) digits, we evaluated the quality of the agent with a classifier trained solely on the original MNIST dataset, and tested the classifier on generated images.

We also compared our method with others to show the efficiency.

We explored the latent space of the agent as well.

Generative models such as VAEs BID13 BID26 and GANs BID5 BID17 BID21 BID0 have achieved huge success in image generation in recent years.

These models generate images directly to pixel-level and thus could be trained through back-propagation effectively.

To mimic human drawing, attempts have been made by both graphics and machine learning communities.

Traditionally, trial-and-error algorithms BID9 are designed to optimize stroke placement by minimizing an energy function, incorporating heuristics, e.g., constraining the number of strokes.

Concept learning is another example tackling this problem using Bayesian program learning BID14 .

Recent deep learning based approaches generally falls into two categories: RNN-based approaches and reinforcement learning.

For RNN-based approaches such as SketchRNN BID7 and handwriting generation with RNN by Graves BID6 , they both rely on sequential datasets.

Thus for unpaired data, those models cannot be applied.

Another popular solution is to adopt reinforcement learning such as "artist agent" BID27 and SPIRAL BID4 .

These methods train an agent that interact with the painting environment.

For reinforcement learning tasks with large, continuous action space like this, the training process can be computationally costly and it could take the agent tens of epochs to converge.

To mitigate this situation, we simulate the environment in a differentiable manner much alike the idea in World Models BID8 BID22 BID23 , where an agent learns from a neural network simulated environment.

Similar approach is also used in character reconstruction for background denoising BID11 .

In our scenario, we train our generator (auto-encoder) by parts for flexible stroke sequence length and image resolution, discussed in Section 3 and 4.Differentiable rendering is an extensively researched topic in computer graphics.

It is used to solve inverse rendering problems.

Some differentiable renderers explicitly model the relationship between the parameters and observations BID16 , others use neural network to approximate the result BID18 since neural nets are powerful function approximators.

While little has been done on simulating 2D rendering process adopted in digital painting software, we used a generator neural network to meet our needs.

We define a single stroke as follows, DISPLAYFORM0 where c ∈ R 3 stands for RGB color, scalar r for brush radius, and tuple (x i , y i , p i ) for an anchor point on the stroke, consisting of x, y coordinate and pressure p, and n is the maximum number of points in a single stroke, in this case, 16.

These values are normalized such that the coordinates correspond to the default OpenGL coordinate system.

DISPLAYFORM1 for k = 1, 2, 3 and i = 1, 2, · · · , n. We used absolute coordinates for each point.

It is notable that compared to the QuickDraw BID7 dataset which contains longer lines, our strokes consist of much fewer points.

We consider many trajectory points redundant since the stroke lines can be fitted by spline curves with fewer anchor points.

For example, to fit a straight line, only two end-points are needed regardless of the length, in other words, stroke curves are usually scaleinvariant.

However, if we are to sample the data from a user input, we could have dozens of points along the trajectory.

Hence we made the assumption of being able to represent curves with a few anchors.

We later showed that a single stroke with only 16 anchors is able to fit most MNIST digits and generate twisted lines in Section 5.

We further assumed that longer and more complicated lines can be decomposed into simple segments and extended our experiments to include recurrent drawing of multiple strokes to generate more complex drawings.

The outline of the StrokeNet architecture is shown in FIG0 .

The generator takes s as input, and projects the stroke data with two MLPs.

One is the position encoder which encodes (x i , y i , p i ) into 64 × 64 feature maps, the other, brush encoder encodes the color and radius of the brush to a single 64 × 64 feature map.

The color c is a single gray scale scalar whose value equals to 1 3 3 k=1 c k , while color strokes are approximated by channel mixing described in Section 3.4.

The features are then concatenated and passed to the (de)convolution layers.

To preserve the sequential and pressure information of each point (x i , y i , p i ), the position encoder first maps (x i , y i ) to the corresponding position onto a 64 × 64 matrix by putting a bright dot on that point.

This is modeled by a 2D Gaussian function with its peak scaled to 1, which simplifies to: DISPLAYFORM0 for i = 1, 2, · · · , n where the value is calculated for each point (x, y) on the 64 × 64 map.

Denote this mapping from (x i , y i ) to R 64×64 as pos: DISPLAYFORM1 (4) By multiplying the corresponding pressure p i , we now have n position features, in our setup, sixteen.

This part of the generator is trained separately with random coordinates until it generates accurate and reliable signals.

However, if we directly feed these features into the (de)convolutional layers of the network, the generator fails partly due to the sparsity of the single brightness feature.

Instead, we take every two neighbouring feature maps and add them together (denoted by "reduce" in FIG0 .), DISPLAYFORM2 Now, each feature map f i represents a segment of the stroke.

By learning to connect and curve the n − 1 "segments", we are able to reconstruct the stroke.

By appending the encoded color and radius data we now have the feature with shape 64 × 64 × n. We then feed the features into three (de)convolutional layers with batch-normalization BID12 activated by LeakyReLU BID28 .

The last layer is activated by tanh.

The agent is a VGG BID25 )-like CNN that encodes the target image into its underlying stroke representation s. Three parallel FC-decoders with different activations are used to decode position (tanh), pressure (sigmoid) and brush data (sigmoid) from the feature.

We used average-pooling instead of max-pooling to improve gradient flow.

For the recurrent version of StrokeNet, two separate CNNs are trained for the target image and the drawing frame, as shown in FIG1 .

In practice the target image feature is computed once for all steps.

We first built a painting software using JavaScript and WebGL.

We later tailored this web application for our experiment.2 The spline used to fit the anchor points is centripetal CatmullRom BID2 BID1 .

A desirable feature about Catmull-Rom spline is that the curve goes through all control points, unlike the more commonly used Bezier curve BID24 .We then interpolate through the sampled points and draw circles around each center point as shown in Figure 3 .

For each pixel inside a circle, its color depends on various factors including attributes of the brush, blending algorithm, etc.

Our generator is trained on the naive brush.

When it comes to the color blending of two frames, the generator is fed with the mean value of input RGB color as a gray scale scalar, and its output is treated as an alpha map.

Normalization and alpha-blending is then performed to yield the next color frame, to simulate real blending algorithm underlying the software.

Denote the generator output at time-step t by q (t) ∈ R 256×256 , the frame image by r (t) ∈ R 3×256×256 , RGB color of the brush by c ∈ R 3 , the blending process is approximated as follows, DISPLAYFORM0 DISPLAYFORM1 for k = 1, 2, 3 corresponding to the RGB channels, where J denotes a 256 × 256 all-one matrix.

For the generator, we synthesize a large amount of samples, each of length n. We would like to capture both the randomness and the smoothness of human writing, thus it is natural to incorporate chaos, most notably, the motion of three-body BID19 ).

Figure 3 : Illustration of how a stroke is rendered.

Figure 4 : Images from our three-body dataset.

There is no closed-form solution to three-body problem, and error accumulates in simulation using numerical methods, leading to unpredictable and chaotic results.

We simulate three-body motion in space (z-component for pressure) with random initial conditions and sample the trajectories as strokes for our dataset.

The simulation is done with a set of equations using Newton's universal law of gravitation: DISPLAYFORM0 where P i (i = 1, 2, 3, P i ∈ R 3 ) denotes the position of the three objects respectively, F i denotes the gravitational force exerted on the ith object.

In our simulation we set mass m 1 = m 2 = m 3 = 1 and gravitational constant G = 5 × 10 −5 .

We also always keep our camera (origin point) at the center of the triangle formed by the three objects to maintain relatively stable "footage".Using this method we collected about 600K images since there is virtually no cost to generate samples.

Samples from the dataset are shown in Figure 4 .

To prove the effectivess of our neural environment, we trained an agent to perform drawing task on several popular datasets, from characters to drawings, with the generator part frozen.

For MNIST and Omniglot, we trained an agent to draw the characters within one stroke.

We later trained the recurrent StrokeNet on more complex datasets like QuickDraw and KanjiVG BID20 .

We resized all the input images to 256 × 256 with anti-alias and paddings.

At first we train the position encoder guided by function pos that maps a coordinate to a 64×64 matrix with l 2 distance to measure the loss.

Next we freeze the position encoder and train the other parts of the generator, again with l 2 loss to measure the performance on the three-body dataset.

It can be found that smaller batch size results in more accurate images.

We trained the generator with a batch size of 64 until the loss no longer improves.

We then set the batch size to 32 to sharpen the neural network.

To train the agent, we freeze the generator.

Denote the agent loss as l agent , the generated image and ground-truth image as i gen and i gt respectively, the loss is defined as: DISPLAYFORM0 where DISPLAYFORM1 T is the data describing the kth anchor point on the stroke.

Here the summation term constrains the average distance between neighbouring points, where λ denotes the penalty strength.

If we drop this term, the agent fails to learn the correct order of the points in a stroke because the generator itself is, after all, not robust to all cases of input, and is very likely to produce wrong results for sequences with large gaps between neighbouring points.

All experiments are conducted on a single NVIDIA Tesla P40 GPU.

We first experimented with single step StrokeNet on MNIST and Omniglot, then we experimented recurrent StrokeNet with QuickDraw and Kanji.

For the MNIST dataset, we later enforced a Gaussian prior to the latent variable and explored the latent space of the agent by linear interpolation.

Finally, for a quantitative evaluation of the model, we trained a classifier on MNIST, and tested the classifier with images generated by the agent.

The close accuracies indicate the quality of the generated images.

It can be seen that a single stroke provides rich expressive power for different shapes.

The generator generalizes well to unseen stroke patterns other than the synthesized three-body dataset.

On the Omniglot dataset, since many characters consist of multiple strokes while the agent can only draw one, the agent tries to capture the contour of the character.

For more complex datasets, multiple steps of strokes are needed.

Again the agent does pretty well to capture the contour of the given image.

However, it seems that the agent has trouble to recover the details of the pictures, and tends to smear inside the boundaries with thick strokes.

To convert the agent into a latent space generative model, we experimented with the VAE version of the agent, where the feature obtained from the last layer of CNN is projected into two vectors representing the means µ and standard deviations (activated by softplus) σ, both of 1024 dimensions.

A vector noise of i.i.d.

Gaussian U ∼ N (0, I) is sampled, latent variable z is given by DISPLAYFORM0 We did latent space interpolation with the agent trained on MNIST.

The simple data led to easily interpretable results.

Since the images are generated by strokes, the digits transform smoothly to one another.

That is to say, the results looked as if we were directly interpolating the stroke data.

Results are shown in Figure 9 and 10.

In order to evaluate the agent, we trained a 5-layer CNN classifier solely on pre-processed MNIST dataset, which is also the input to the MNIST agent.

The size of the image is 256 × 256, so there is some performance drop to the classification task compared to standard 28 × 28 images.

The classifier is then used to evaluate the paired test-set image generated by the agent.

The accuracies reflect the quality of the generated images.

We also compared the l 2 loss with SPIRAL on MNIST to illustrate that our method has the advantage of faster convergence over reinforcement learning approaches, shown in FIG0 .

Pre-processed images 90.82% Agent Output (3 steps) 88.43% Agent Output TAB0 79.33% Agent Output (1 step, VAE) 67.21% (a) (b) FIG0 : Comparison of stroke orders between human and agent.

We can see the stroke order is completely chaotic compared to natural order.

For future work, there are several major improvements we want to make both to the network structure and to the algorithm.

The recurrent structure adopted here is of the simplest form.

We use this setup because we consider drawing as a Markov process, where the current action only depends on what the agent sees, the target image and the previous frame.

More advanced structures like LSTM BID10 or GRU BID3 may boost the performance.

A stop sign can be also introduced to determine when to stop drawing, which can be useful in character reconstruction.

For the agent, various attention mechanism could be incorporated to help the agent focus on undrawn regions, so that smear and blurry scribbles might be prevented.

Secondly, The generator and the agent were trained as two separate parts throughout the experiment.

We can somehow train them as a whole: during the training of the agent, store all the intermediate stroke data.

After a period of training, sample images from the real environment with the stroke data just collected, and train the generator with the data.

By doing so in an iterative manner, the generator could fit better to the current agent and provide more reliable reconstructions, while a changing generator may potentially provide more valuable overall gradients.

It is also found useful to add a bit of randomness to the learning rate.

Since different decoders of the agent learn at different rates, stochasticity results in more appealing results.

For example, the agent usually fails to generalize to color images because it always sticks with one global average color (as shown in FIG0 ).

However, it sometimes generates appealing results with some randomness added during the training.

As a result of this immobility, the way agent writes is dull compared to humans and reinforcement learning agents like SPIRAL.

For instance, when writing the digit "8", the agent is simply writing "3" with endpoints closed.

Also, the agent avoids to make intersecting strokes over all datasets, although such actions are harmless and should be totally encouraged and explored!

Thus, random sampling techniques could be added to the decision making process to encourage bolder moves.

Finally, for the evaluation metrics, the naive l 2 loss can be combined with adversarial learning.

If paired sequential data is available, we believe adding it to training will also improve the results.

In this paper we bring a proof-of-concept that an agent is able to learn from its neural simulation of an environment.

Especially when the environment is deterministic given the action, or contains a huge action space, the proposed approach could be useful.

Our primary contribution is that we devised a model-based method to approximate non-differentiable environment with neural network, and the agent trained with our method converges quickly on several datasets.

It is able to adapt its skills to real world.

Hopefully such approaches can be useful when dealing with more difficult reinforcement learning problems.

T denote the coordinate of a sampled point.

For a curve defined by points P 0 , P 1 , P 2 , P 3 , the spline can be produced by: DISPLAYFORM0 where DISPLAYFORM1 with α = 0.5, t 0 = 0 and i = 0, 1, 2, 3By interpolating t from t 1 to t 2 linearly, we generate the curve between P 1 and P 2 .

The pressure values between neighbouring points are interpolated linearly.

The agent loss equals to the l 2 distance between the generator output and agent input plus the penalty term constraining the average point distance within a stroke.

For (c) and (d) the learning rate is set to 10 −4 , batch size equals to 64.

Figure 15: A trained StrokeNet generates images that resemble the output of painting software.

The first row depicts results generated by our model (left) and by the software (right) given the same input.

The second row shows the model could produce strokes with color and texture using simple arithmetic operations.

The third and fourth row shows the model's ability to draw MNIST digits (left) on both its own generative model (middle) and real-world painting software (right).

@highlight

StrokeNet is a novel architecture where the agent is trained to draw by strokes on a differentiable simulation of the environment, which could effectively exploit the power of back-propagation.