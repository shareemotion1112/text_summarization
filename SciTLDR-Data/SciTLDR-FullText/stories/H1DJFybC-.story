We introduce a model that learns to convert simple hand drawings   into graphics programs written in a subset of \LaTeX.~

The model   combines techniques from deep learning and program synthesis.

We   learn a convolutional neural network that proposes plausible drawing   primitives that explain an image.

These drawing primitives are like   a trace of the set of primitive commands issued by a graphics   program.

We learn a model that uses program synthesis techniques to   recover a graphics program from that trace.

These programs have   constructs like variable bindings, iterative loops, or simple kinds   of conditionals.

With a graphics program in hand, we can correct   errors made by the deep network and extrapolate drawings.

Taken   together these results are a step towards agents that induce useful,   human-readable programs from perceptual input.

How can an agent convert noisy, high-dimensional perceptual input to a symbolic, abstract object, such as a computer program?

Here we consider this problem within a graphics program synthesis domain.

We develop an approach for converting hand drawings into executable source code for drawing the original image.

The graphics programs in our domain draw simple figures like those found in machine learning papers (see FIG0 ).

The key observation behind our work is that generating a programmatic representation from an image of a diagram involves two distinct steps that require different technical approaches.

The first step involves identifying the components such as rectangles, lines and arrows that make up the image.

The second step involves identifying the high-level structure in how the components were drawn.

In FIG0 , it means identifying a pattern in how the circles and rectangles are being drawn that is best described with two nested loops, and which can easily be extrapolated to a bigger diagram.

We present a hybrid architecture for inferring graphics programs that is structured around these two steps.

For the first step, a deep network to infers a set of primitive shape-drawing commands.

We refer FIG8 : Both the paper and the system pipeline are structured around the trace hypothesisThe new contributions of this work are: (1) The trace hypothesis: a framework for going from perception to programs, which connects this work to other trace-based models, like the Neural Program Interpreter BID17 ; BID26 A model based on the trace hypothesis that converts sketches to high-level programs: in contrast to converting images to vectors or low-level parses BID11 BID14 BID24 BID1 BID2 .

FORMULA8 A generic algorithm for learning a policy for efficiently searching for programs, building on Levin search BID13 and recent work like DeepCoder BID0 .

Even with the high-level idea of a trace set, going from hand drawings to programs remains difficult.

We address these challenges: (1) Inferring trace sets from images requires domain-specific design choices from the deep learning and computer vision toolkits (Sec. 2) .

FORMULA4 Generalizing to noisy hand drawings, we will show, requires learning a domain-specific noise model that is invariant to the variations across hand drawings (Sec. 2.1).

(3) Discovering good programs requires solving a difficult combinatorial search problem, because the programs are often long and complicated (e.g., 9 lines of code, with nested loops and conditionals).

We give a domain-general framework for learning a search policy that quickly guides program synthesizers toward the target programs (Sec. 3.1).

We developed a deep network architecture for efficiently inferring a trace set, T , from an image, I. Our model combines ideas from Neurally-Guided Procedural Models BID18 and Attend-Infer-Repeat (Eslami et al., 2016) .

The network constructs the trace set one drawing command at a time, conditioned on what it has drawn so far.

FIG1 illustrates this architecture.

We first pass a 256 × 256 target image and a rendering of the trace set so far (encoded as a two-channel image) to a convolutional network.

Given the features extracted by the convnet, a multilayer perceptron then predicts a distribution over the next drawing command to add to the trace set (see Tbl.

1).

We also use a differentiable attention mechanism (Spatial Transformer Networks: ) to let Blue: network inputs.

Black: network operations.

Red: samples from a multinomial.

Typewriter font: network outputs.

Renders snapped to a 16 × 16 grid, illustrated in gray.

STN (spatial transformer network) is a differentiable attention mechanism .

Table 1 : Primitive drawing commands currently supported by our model.

Circle at (x, y) rectangle(x 1 , y 1 , x 2 , y 2 )Rectangle with corners at (x 1 , y 1 ) & (x 2 , y 2 ) line(x 1 , y 1 , x 2 , y 2 , arrow ∈ {0, 1}, dashed ∈ {0, 1})Line from (x 1 , y 1 ) to (x 2 , y 2 ), optionally with an arrow and/or dashed STOP Finishes trace set inference the model attend to different regions of the image while predicting drawing commands.

We currently constrain coordinates to lie on a discrete 16 × 16 grid, but the grid could be made arbitrarily fine.

We train the network by sampling trace sets T and target images I for randomly generated scenes and maximizing the likelihood of T given I with respect to the model parameters, θ, by gradient ascent.

We trained the network on 10 5 scenes, which takes a day on an Nvidia TitanX GPU.

FIG10 : Parsing L A T E X output after training on diagrams with ≤ 12 objects.

Model generalizes to scenes with many more objects.

Neither SMC nor the neural network are sufficient on their own.

# particles varies by model: we compare the models with equal runtime (≈ 1 sec/object) Our network can "derender" random synthetic images by doing a beam search to recover trace sets maximizing DISPLAYFORM0 But, if the network predicts an incorrect drawing command, it has no way of recovering from that error.

For added robustness we treat the network outputs as proposals for a Sequential Monte Carlo (SMC) sampling scheme (Doucet et al., 2001) .

The SMC sampler is designed to sample from the distribution DISPLAYFORM1 , where L(·|·) uses the pixel-wise distance between two images as a proxy for a likelihood.

Here, the network is learning a proposal distribution in an amortized way BID15 and using it to invert a generative model (the renderer).Experiment 1: FIG10 .

To evaluate which components of the model are nec-essary to parse complicated scenes, we compared the neural network with SMC against the neural network by itself or SMC by itself.

Only the combination of the two passes a critical test of generalization: when trained on images with ≤ 12 objects, it successfully parses scenes with many more objects than the training data.

We compare with a baseline that produces the trace set in one shot by using the CNN to extract features of the input which are passed to an LSTM which finally predicts the trace set token-by-token (LSTM in FIG10 ).

This architecture is used in several successful neural models of image captioning (e.g., ), but, for this domain, cannot parse cluttered scenes with many objects.

We trained the model to generalize to hand drawings by introducing noise into the renderings of the training target images.

We designed this noise process to introduce the kinds of variations found in hand drawings (see supplement for details).Our neurally-guided SMC procedure used pixel-wise distance as a surrogate for a likelihood function (L(·|·) in section 2).

But pixel-wise distance fares poorly on hand drawings, which never exactly match the model's renders.

So, for hand drawings, we learn a surrogate likelihood function, L learned (·|·).The density L learned (·|·) is predicted by a convolutional network that we train to predict the distance between two trace sets conditioned upon their renderings.

We train our likelihood surrogate to approximate the symmetric difference, which is the number of drawing commands by which two trace sets differ: DISPLAYFORM0 Experiment 2: Figures 5-7.

We evaluated, but did not train, our system on 100 real hand-drawn figures; see Fig. 5 -6.

These were drawn carefully but not perfectly with the aid of graph paper.

For each drawing we annotated a ground truth trace set and had the neurally guided SMC sampler produce 10 3 samples.

For 63% of the drawings, the Top-1 most likely sample exactly matches the ground truth; with more samples, the model finds trace sets that are closer to the ground truth annotation FIG3 .

We will show that the program synthesizer corrects some of these small errors (Sec. 4.1).

Although the trace set of a graphics program describes the contents of a scene, it does not encode higher-level features of the image, such as repeated motifs or symmetries.

A graphics program better describes such structures.

We seek to synthesize graphics programs from their trace sets.

We constrain the space of programs by writing down a context free grammar over programs -what in the program languages community is called a Domain Specific Language (DSL) BID16 .

Our DSL (Tbl.

2) encodes prior knowledge of what graphics programs tend to look like.

DISPLAYFORM0 Given the DSL and a trace set T , we want a program that both evaluates to T and, at the same time, is the "best" explanation of T .

For example, we might prefer more general programs or, in the spirit of Occam's razor, prefer shorter programs.

We wrap these intuitions up into a cost function over programs, and seek the minimum cost program consistent with T : DISPLAYFORM1 We define the cost of a program to be the number of Statement's it contains (Tbl.

2).

We also penalize using many different numerical constants; see supplement.

The constrained optimization problem in Eq. 2 is intractable in general, but there exist efficient-inpractice tools for finding exact solutions to such program synthesis problems.

We use the state-ofthe-art Sketch tool BID20 .

Sketch takes as input a space of programs, along with a specification of the program's behavior and optionally a cost function.

It translates the synthesis problem into a constraint satisfaction problem and then uses a SAT solver to find a minimum-cost program satisfying the specification.

Sketch requires a finite program space, which here means that the depth of the program syntax tree is bounded (we set the bound to 3), but has the guarantee that it always eventually finds a globally optimal solution.

In exchange for this optimality guarantee it comes with no guarantees on runtime.

For our domain synthesis times vary from minutes to hours, with 27% of the drawings timing out the synthesizer after 1 hour.

Tbl.

3 shows programs recovered by our system.

A main impediment to our use of these general techniques is the prohibitively high cost of searching for programs.

We next describe how to learn to synthesize programs much faster (Sec. 3.1), timing out on 2% of the drawings and solving 58% of problems within a minute.

figure, and the complicated program in the second figure to bottom.

Line BID26 15, BID28 15) Line BID28 9, BID28 13) Line BID27 11, BID27 14) Line BID26 13, BID26 15) Line BID27 14, 6, 14) Line BID28 13, 8, 13) for(i<3) DISPLAYFORM0 Circle BID29 8) Circle BID26 8) Circle(8,11) Line(2,9, 2,10) Circle(8, 8) Line BID27 8, BID28 8) Line BID27 11, BID28 11)

... etc. ...

; 21 lines for(i<3) for FORMULA8 if (j>0) line(-3 * j+8,-3 * i+7, -3 * j+9,-3 * i+7) line(-3 * i+7,-3 * j+8, -3 * i+7,-3 * j+9) circle(-3 * j+7,-3 * i+7) 21 6 = 3.5xRectangle BID25 10, BID27 11) Rectangle BID25 12, BID27 13) Rectangle BID28 8, 6, 9) Rectangle BID28 10, 6, 11) ... etc. ...; 16 lines for(i<4) for FORMULA9 rectangle(-3 * i+9,-2 * j+6, -3 * i+11,-2 * j+7) DISPLAYFORM1 Line (3,10,3,14,arrow) Rectangle (11, 8, 15, 10) Rectangle (11, 14, 15, 15) Line (13,10,13,14,arrow) ... etc. ...; 16 lines for(i<3) line (7,1,5 * i+2,3,arrow) for(j<i+1) if(j>0) line(5 * j-1,9,5 * i,5,arrow) line(5 * j+2,5,5 * j+2,9,arrow) rectangle (5 * i,3,5 * i+4,5) rectangle(5 * i,9,5 * i+4,10) rectangle BID26 0, 12, BID25 16 9 = 1.8xCircle BID26 8) Rectangle (6, 9, 7, 10) Circle(8, 8) Rectangle(6, 12, 7, 13) Rectangle BID27 9, BID28 10) ... etc. ...; 9 lines reflect(y=8) for FORMULA8 if (i>0) rectangle (3 * i-1,2,3 * i,3) circle(3 * i+1,3 * i+1) 9 5 = 1.8x

We want to leverage powerful, domain-general techniques from the program synthesis community, but make them much faster by learning a domain-specific search policy.

A search policy poses search problems like those in Eq. 2, but also offers additional constraints on the structure of the program (Tbl.

4).

For example, a policy might decide to first try searching over small programs before searching over large programs, or decide to prioritize searching over programs that have loops.

A search policy π θ (σ|T ) takes as input a trace set T and predicts a distribution over synthesis problems, each of which is written σ and corresponds to a set of possible programs to search over (so σ ⊆ DSL).

Good policies will prefer tractable program spaces, so that the search procedure will terminate early, but should also prefer program spaces likely to contain programs that concisely explain the data.

These two desiderata are in tension: tractable synthesis problems involve searching over smaller spaces, but smaller spaces are less likely to contain good programs.

Our goal now is to find the parameters of the policy, written θ, which best navigate this trade-off.

Given a search policy, what is the best way of using it to quickly find minimum cost programs?

We use a bias-optimal search algorithm BID19 :Definition: Bias-optimality.

A search algorithm is n-bias optimal with respect to a distribution P bias [·] if it is guaranteed to find a solution in σ after searching for at least time n × t(σ) DISPLAYFORM0 , where t(σ) is the time it takes to verify that σ contains a solution to the search problem.

An example of a 1-bias optimal search algorithm is a time-sharing system that allocates P bias [σ] of its time to trying σ.

We construct a 1-bias optimal search algorithm by identifying P bias [σ] = π θ (σ|T ) and t(σ) = t(σ|T ), where t(σ|T ) is how long the synthesizer takes to search σ for a program for T .

This means that the search algorithm explores the entire program space, but spends most of its time in the regions of the space that the policy judges to be most promising.

Now in theory any π θ (·|·) is a bias-optimal searcher.

But the actual runtime of the algorithm depends strongly upon the bias P bias [·] .

Our new approach is to learn P bias [·] by picking the policy minimizing the expected bias-optimal time to solve a training corpus, D, of graphics program synthesis problems: DISPLAYFORM1 where σ ∈ BEST(T ) if a minimum cost program for T is in σ.

Practically, bias optimality has now bought us the following: FORMULA10 a guarantee that the policy will always find the minimum cost program; and FORMULA4 a differentiable loss function for the policy parameters that takes into account the cost of searching, in contrast to e.g. DeepCoder BID0 .To generate a training corpus for learning a policy which minimizes this loss, we synthesized minimum cost programs for each trace set of our hand drawings and for each σ.

We locally minimize this loss using gradient descent.

Because we want to learn a policy from only 100 hand-drawn diagrams, we chose a simple low-capacity, bilinear model for a policy: DISPLAYFORM2 where φ params (σ) is a one-hot encoding of the parameter settings of σ (see Tbl.

4) and φ trace (T ) extracts a few simple features of the trace set T ; see supplement for details.

Experiment 3: Figure 8 .

We compare synthesis times for our learned search policy with two alternatives: Sketch, which poses the entire problem wholesale to the Sketch program synthesizer; and an Oracle, a policy which always picks the quickest to search σ also containing a minimum cost program.

Our approach improves upon Sketch by itself, and comes close to the Oracle's performance.

One could never construct this Oracle, because the agent does not know ahead of time which σ's contain minimum cost programs nor does it know how long each σ will take to search.

With this learned policy in hand we can synthesize 58% of programs within a minute.

Solve the problem piece-by-piece or all at once?

{True, False} Maximum depth Bound on the depth of the program syntax tree {1, 2, 3}

Why synthesize a graphics program, if the trace set already suffices to recover the objects in an image?

Within our domain of hand-drawn figures, graphics program synthesis has several uses: The program synthesizer corrects errors made by the neural network by favoring trace sets which lead to more concise or general programs.

For example, figures with perfectly aligned objects are preferable, and precise alignment lends itself to short programs.

Concretely, we run the program synthesizer on the Top-k most likely trace sets output by the neurally guided sampler.

Then, the system reranks the Top-k by the prior probability of their programs.

The prior probability of a program is learned by picking the prior maximizing the likelihood of the ground truth trace sets; see supplement for details.

But, this procedure can only correct errors when a correct trace set is in the Top-k.

Our sampler could only do better on 7/100 drawings by looking at the Top-100 samples (see FIG3 ), precluding a statistically significant analysis of how much learning a prior over programs could help correct errors.

But, learning this prior does sometimes help correct mistakes made by the neural network; see Fig. 9 for a representative example of the kinds of corrections that it makes.

See supplement for details.

Having access to the source code of a graphics program facilitates coherent, high-level image editing.

For example we can extrapolate figures by increasing the number of times that loops are executed.

Extrapolating repetitive visuals patterns comes naturally to humans, and is a practical application: imagine hand drawing a repetitive graphical model structure and having our system automatically induce and extend the pattern.

FIG0 shows extrapolations produced by our system.

Program Induction: Our approach to learning to search for programs draws theoretical underpinnings from Levin search BID13 BID21 ) and Schmidhuber's OOPS model (Schmidhuber, 2004) .

DeepCoder BID0 ) is a recent model which, like ours, learns to predict likely program components.

Our work differs because we treat the problem as metareasoning, identifying and modeling the trade-off between tractability and probability of success.

TerpreT (Gaunt et al., 2016) systematically compares constraint-based program synthesis techniques against gradient-based search techniques, like those used to train Differentiable Neural Computers BID9 .

The TerpreT experiments motivate our use of constraint-based techniques.

Deep Learning: Our neural network bears resemblance to the Attend-Infer-Repeat (AIR) system, which learns to decompose an image into its constituent objects BID5 .

AIR learns an iterative inference scheme which infers objects one by one and also decides when to stop inference.

Our network differs in its architecture and training regime: AIR learns a recurrent auto-encoding model via variational inference, whereas our parsing stage learns an autoregressive-style model from randomly-generated (trace, image) pairs.

IM2LATEX BID2 ) is a recent work that also converts images to L A T E X. Their goal is to derender L A T E X equations, which recovers a markup language representation.

Our goal is to go from noisy input to a high-level program, which goes beyond markup languages by supporting programming constructs like loops and conditionals.

Recovering a high-level program is more challenging than recovering markup because it is a highly under constrained symbolic reasoning problem.

Our image-to-trace parsing architecture builds on prior work on controlling procedural graphics programs BID18 .

We adapt this method to a different visual domain (figures composed of multiple objects), using a broad prior over possible scenes as the initial program and viewing the trace through the guide program as a symbolic parse of the target image.

We then show how to efficiently synthesize higher-level programs from these traces.

In the computer graphics literature, there have been other systems which convert sketches into procedural representations.

One uses a convolutional network to match a sketch to the output of a parametric 3D modeling system BID11 .

Another uses convolutional networks to support sketch-based instantiation of procedural primitives within an interactive architectural modeling system BID14 .

Both systems focus on inferring fixed-dimensional parameter vectors.

In contrast, we seek to automatically infer a structured, programmatic representation of a sketch which captures higher-level visual patterns.

Hand-drawn sketches: Prior work has also applied sketch-based program synthesis to authoring graphics programs.

Sketch-n-Sketch is a bi-directional editing system in which direct manipulations to a program's output automatically propagate to the program source code BID10 .

We see this work as complementary to our own: programs produced by our method could be provided to a Sketch-n-Sketch-like system as a starting point for further editing.

The CogSketch system BID6 ) also aims to have a high-level understanding of handdrawn figures.

Their primary goal is cognitive modeling (they apply their system to solving IQ-test style visual reasoning problems), whereas we are interested in building an automated AI application (e.g. in our system the user need not annotate which strokes correspond to which shapes; our neural network produces something equivalent to the annotations).The Trace Hypothesis:

The idea that an execution trace could assist in program learning goes back to the 1970's BID22 and has been applied in neural models of program induction, like Neural Program Interpreters BID17 , or DeepCoder, which predicts what functions occur in the execution trace BID0 .

Our contribution to this idea is the trace hypothesis: that trace sets can be inferred from perceptual data, and that the trace set is a useful bridge between perception and symbolic representation.

Our work is the first to articulate and explore this hypothesis by demonstrating how a trace could be inferred and how it can be used to synthesize a high-level program.

We have presented a system for inferring graphics programs which generate L A T E X-style figures from hand-drawn images.

The system uses a combination of deep neural networks and stochastic search to parse drawings into symbolic trace sets; it then feeds these traces to a general-purpose program synthesis engine to infer a structured graphics program.

We evaluated our model's performance at parsing novel images, and we demonstrated its ability to extrapolate from provided drawings.

In the near future, we believe it will be possible to produce professional-looking figures just by drawing them and then letting an artificially-intelligent agent write the code.

More generally, we believe the trace hypothesis, as realized in our two-phase system-parsing into trace sets, then searching for a low-cost symbolic program which generates those traces-may be a useful paradigm for other domains in which agents must programmatically reason about noisy perceptual input.

Concretely, we implemented the following scheme: for an image I, the neurally guided sampling scheme of section 3 of the main paper samples a set of candidate traces, written F(I).

Instead of predicting the most likely trace in F(I) according to the neural network, we can take into account the programs that best explain the traces.

WritingT (I) for the trace the model predicts for image I, DISPLAYFORM0 where P β [·] is a prior probability distribution over programs parameterized by β.

This is equivalent to doing MAP inference in a generative model where the program is first drawn from P β [·], then the program is executed deterministically, and then we observe a noisy version of the program's output, where L learned (I|render(·)) × P θ [·|I] is our observation model.

Given a corpus of graphics program synthesis problems with annotated ground truth traces (i.e. (I, T ) pairs), we find a maximum likelihood estimate of β: DISPLAYFORM1 where the expectation is taken both over the model predictions and the (I, T ) pairs in the training corpus.

We define P β [·] to be a log linear distribution ∝ exp(β · φ(program)), where φ(·) is a feature extractor for programs.

We extract a few basic features of a program, such as its size and how many loops it has, and use these features to help predict whether a trace is the correct explanation for an image.

We synthesized programs for the top 10 traces output by the deep network.

Learning this prior over programs can help correct mistakes made by the neural network, and also occasionally introduces mistakes of its own; see FIG0 for a representative example of the kinds of corrections that it makes.

On the whole it modestly improves our Top-1 accuracy from 63% to 67%.

Recall that from Fig. 6 of the main paper that the best improvement in accuracy we could possibly get is 70% by looking at the top 10 traces.

We measure the similarity between two drawings by extracting features of the best programs that describe them.

Our features are counts of the number of times that different components in the DSL were used.

We project these features down to a 2-dimensional subspace using primary component analysis (PCA); see FIG8 .

One could use many alternative similarity metrics between drawings which would capture pixel-level similarities while missing high-level geometric similarities.

We used our learned distance metric between traces, L learned (·|·), and projected to a 2-dimensional subspace using multidimensional scaling (MDS: FORMULA10 ).

This reveals similarities between the objects in the drawings, while missing similarities at the level of the program.

Recall from the main paper that our goal is to estimate the policy minimizing the following loss: DISPLAYFORM0 where σ ∈ BEST(T ) if a minimum cost program for T is in σ.

We make this optimization problem tractable by annealing our loss function during gradient descent: DISPLAYFORM1 where DISPLAYFORM2 Notice that SOFTMINIMUM β=∞ (·) is just min(·).

We set the regularization coefficient λ = 0.1 and minimize equation 4 using Adam for 2000 steps, linearly increasing β from 1 to 2.We parameterize the space of policies as a simple log bilinear model: DISPLAYFORM3 where: DISPLAYFORM4

For the model in FIG10 , the distribution over the next drawing command factorizes as: DISPLAYFORM0 where t 1 t 2 · · · t K are the tokens in the drawing command, I is the target image, T is a trace set, θ are the parameters of the neural network, f θ (·, ·) is the image feature extractor (convolutional network), and a θ (·|·) is an attention mechanism.

The distribution over traces factorizes as: DISPLAYFORM1 where |T | is the length of trace T , the subscripts on T index drawing commands within the trace (so T n is a sequence of tokens: t 1 t 2 · · · t K ), and the STOP token is emitted by the network to signal that the trace explains the image.

The convolutional network takes as input 2 256 × 256 images represented as a 2 × 256 × 256 volume.

These are passed through two layers of convolutions separated by ReLU nonlinearities and max pooling:• Layer 1: 20 8 × 8 convolutions, 2 16 × 4 convolutions, 2 4 × 16 convolutions.

Followed by 8 × 8 pooling with a stride size of 4.• Layer 2: 10 8 × 8 convolutions.

Followed by 4 × 4 pooling with a stride size of 4.

Given the image features f , we predict the first token (i.e., the name of the drawing command: circle, rectangle, line, or STOP) using logistic regression: DISPLAYFORM0 where W t1 is a learned weight matrix and b t1 is a learned bias vector.

Given an attention mechanism a(·|·), subsequent tokens are predicted as: DISPLAYFORM1 Thus each token of each drawing primitive has its own learned MLP.

For predicting the coordinates of lines we found that using 32 hidden nodes with sigmoid activations worked well; for other tokens the MLP's are just logistic regression (no hidden nodes).We use Spatial Transformer Networks (2) as our attention mechanism.

The parameters of the spatial transform are predicted on the basis of previously predicted tokens.

For example, in order to decide where to focus our attention when predicting the y coordinate of a circle, we condition upon both the identity of the drawing command (circle) and upon the value of the previously predicted x coordinate: DISPLAYFORM2 So, we learn a different network for predicting special transforms for each drawing command (value of t 1 ) and also for each token of the drawing command.

These networks (MLP t1,n in equation 11) have no hidden layers and output the 6 entries of an affine transformation matrix; see FORMULA4 for more details.

Training takes a little bit less than a day on a Nvidia TitanX GPU.

The network was trained on 10 5 synthetic examples.

We compared our deep network with a baseline that models the problem as a kind of image captioning.

Given the target image, this baseline produces the program trace in one shot by using a CNN to extract features of the input which are passed to an LSTM which finally predicts the trace token-by-token.

This general architecture is used in several successful neural models of image captioning (e.g., FORMULA9 ).Concretely, we kept the image feature extractor architecture (a CNN) as in our model, but only passed it one image as input (the target image to explain).

Then, instead of using an autoregressive decoder to predict a single drawing command, we used an LSTM to predict a sequence of drawing commands token-by-token.

This LSTM had 128 memory cells, and at each time step produced as output the next token in the sequence of drawing commands.

It took as input both the image representation and its previously predicted token.

Our architecture for L learned (render(T 1 )|render(T 2 )) has the same series of convolutions as the network that predicts the next drawing command.

We train it to predict two scalars: |T 1 − T 2 | and |T 2 − T 1 |.

These predictions are made using linear regression from the image features followed by a ReLU nonlinearity; this nonlinearity makes sense because the predictions can never be negative but could be arbitrarily large positive numbers.

We train this network by sampling random synthetic scenes for T 1 , and then perturbing them in small ways to produce T 2 .

We minimize the squared loss between the network's prediction and the ground truth symmetric differences.

T 1 is rendered in a "simulated hand drawing" style which we describe next.

We introduce noise into the L A T E X rendering process by:• Rescaling the image intensity by a factor chosen uniformly at random from [0.

5, 1.5]

• Translating the image by ±3 pixels chosen uniformly random• Rendering the L A T E X using the pencildraw style, which adds random perturbations to the paths drawn by L A T E Xin a way designed to resemble a pencil.• Randomly perturbing the positions and sizes of primitive L A T E Xdrawing commands 6 LIKELIHOOD SURROGATE FOR SYNTHETIC DATA For synthetic data (e.g., L A T E X output) it is relatively straightforward to engineer an adequate distance measure between images, because it is possible for the system to discover drawing commands that Figure 5 : Example synthetic training data exactly match the pixels in the target image.

We use: DISPLAYFORM0 where α, β are constants that control the trade-off between preferring to explain the pixels in the image (at the expense of having extraneous pixels) and not predicting pixels where they don't exist (at the expense of leaving some pixels unexplained).

Because our sampling procedure incrementally constructs the scene part-by-part, we want α > β.

That is, it is preferable to leave some pixels unexplained; for once a particle in SMC adds a drawing primitive to its trace that is not actually in the latent scene, it can never recover from this error.

In our experiments on synthetic data we used α = 0.8 and β = 0.04.

We generated synthetic training data for the neural network by sampling L A T E X code according to the following generative process: First, the number of objects in the scene are sampled uniformly from 1 to 12.

For each object we uniformly sample its identity (circle, rectangle, or line).

Then we sample the parameters of the circles, than the parameters of the rectangles, and finally the parameters of the lines; this has the effect of teaching the network to first draw the circles in the scene, then the rectangles, and finally the lines.

We furthermore put the circle (respectively, rectangle and line) drawing commands in order by left-to-right, bottom-to-top; thus the training data enforces a canonical order in which to draw any scene.

To make the training data look more like naturally occurring figures, we put a Chinese restaurant process prior FORMULA14 over the values of the X and Y coordinates that occur in the execution trace.

This encourages reuse of coordinate values, and so produces training data that tends to have parts that are nicely aligned.

In the synthetic training data we excluded any sampled scenes that had overlapping drawing commands.

As shown in the main paper, the network is then able to generalize to scenes with, for example, intersecting lines or lines that penetrate a rectangle.

When sampling the endpoints of a line, we biased the sampling process so that it would be more likely to start an endpoint along one of the sides of a rectangle or at the boundary of a circle.

If n is the number of points either along the side of a rectangle or at the boundary of a circle, we would sample an arbitrary endpoint with probability 2 2+n and sample one of the "attaching" endpoints with probability 1 2+n .

See FIG3 for examples of the kinds of scenes that the network is trained on.

For readers wishing to generate their own synthetic training sets, we refer them to our source code at: redactedForAnonymity.com.

We seek the minimum cost program which evaluates to (produces the drawing primitives in) an execution trace T : DISPLAYFORM0 Programs incur a cost of 1 for each command (primitive drawing action, loop, or reflection).

They incur a cost of 1 3 for each unique coefficient they use in a linear transformation beyond the first coefficient.

This encourages reuse of coefficients, which leads to code that has translational symmetry; rather than provide a translational symmetry operator as we did with reflection, we modify what is effectively a prior over the space of program so that it tends to produce programs that have this symmetry.

Programs also incur a cost of 1 for having loops of constant length 2; otherwise there is often no pressure from the cost function to explain a repetition of length 2 as being a reflection rather a loop.

Below we show our full data set of drawings.

The leftmost column is a hand drawing.

The middle column is a rendering of the most likely trace discovered by the neurally guided SMC sampling scheme.

The rightmost column is the program we synthesized from a ground truth execution trace of the drawing.

Note that because the inference procedure is stochastic, the top one most likely sample can vary from run to run.

Below we report a representative sample from a run with 2000 particles.

line(6,2,6,3, arrow = False,solid = True); line(6,2,3,2, arrow = True,solid = True); reflect(y = 9){ line BID27 7, BID29 BID29 , arrow = True,solid = True); rectangle BID25 BID25 BID27 BID27 ; rectangle (5,3,7,6); rectangle(0,0,8,9 ) } for (i < 2){ line (8, 8, BID27 8 , arrow = True,solid = False); line(-2 * i + 12,5,-2 * i + 13,5 arrow = True,solid = True); line (6, BID29 7, BID29 , arrow = True,solid = True); line(3,-6 * i + 8,5,-2 * i + 6, arrow = True,solid = True); rectangle(-2 * i + 13,4,-2 * i + rectangle (1,-6 * i + 7,3,-6 * i }; circle(8,5) ; rectangle BID29 BID27 6, 7) ; rectangle(0,0,10,10); line (8, 6, 8, 8, arrow = False, solid = False) reflect(y = 7){ line BID26 6, BID28 BID28 , arrow = True,solid = True); rectangle(0,0,2,2) }; rectangle BID28 BID26 6, BID29 line (7, BID29 9, BID29 , arrow = True,solid = True); rectangle BID29 BID27 7, 7) ; rectangle(0,0,12,10); reflect(y = 10){ circle(10,5); line BID27 BID26 BID29 BID28 , arrow = True,solid = True); rectangle BID25 BID25 BID27 BID27 } line(10,1,2,1, arrow = True,solid = False); line(10,1,10,3, arrow = False,solid = False); line (7, BID28 9, BID28 , arrow = True,solid = True); reflect(y = 8){ circle(10,4); line BID26 BID25 BID28 BID27 , arrow = True,solid = True); rectangle (4,2,7,6); rectangle(0,6,2,8 ) } line(12,9,12,0, arrow = True,solid = True); rectangle (9, BID27 11, 9) ; rectangle (6, BID29 8, 9) ; rectangle(0,7,2,9); rectangle BID27 8, BID29 9) for (i < 3){ for (j < (1 * i + 1)){ if (j > 0){ line(3 * j + -3,3 * i + -2,3 * j arrow = False,solid = True); line(0,3 * j + -2,3 * j + -3,4, arrow = False,solid = True) } rectangle (2,0,5, rectangle (0, 0, BID27 BID28 circle BID25 BID25 reflect(x = 7){ circle(6,1); line(6,2,6,5, arrow = False,solid = True); rectangle BID29 BID29 7, 7) }; line BID26 6, BID29 6 , arrow = False,solid = True); line BID26 BID25 BID29 BID25 arrow = False, solid = True) line ( BID28 BID28 6, 6 ); rectangle(4 * i,0,4 * i + 2,2) }; rectangle (8, BID28 10, 6) for ( BID29 7, BID29 6 , arrow = True,solid = True); line BID27 BID27 BID27 BID26 arrow = True, solid = True) ; line BID25 7, BID25 6 , arrow = True,solid = True); rectangle (0, BID27 6, 6) ; rectangle BID26 0, BID28 BID26 ; rectangle (0, 7, 6, 9) line ( BID28 6, 6, 6 , arrow = True,solid = True); line BID26 10, BID26 8 , arrow = True,solid = True); rectangle BID25 0, BID27 BID26 }; rectangle (0, BID28 BID28 8) reflect(y = 9){ reflect(x = 9){ circle(8,8); line BID27 8, 6, 8 , arrow = False,solid = True); line BID25 BID27 BID25 6 , arrow = False,solid = True) } } reflect(x = 11){ rectangle(9,4,10,7); reflect(y = 11){ rectangle (8, 0, 11, BID27 ; rectangle(4,9,7,10) } } for (i < 3){ line(2 * i,-2 * i + 5,2 * i + 2, arrow = False,solid = True); line(2 * i + 1,-2 * i + 4,2 * i arrow = False,solid = True) } rectangle BID28 BID25 6, BID26 ; rectangle(7,0,9,2); reflect(y = 10){ rectangle (0, 0, BID27 BID27 ; rectangle BID25 BID28 BID26 6 ) } line(7,4,9,4, arrow = True,solid = True); line (8, BID27 7, BID27 , arrow = True,solid = True); reflect(y = 7){ line BID26 BID25 BID28 BID27 , arrow = True,solid = True); rectangle(0,0,2,2) }; line(8,3,9,3, arrow = False,solid = True); rectangle(9,2,12,5); rectangle BID28 BID26 7, BID29 for ( BID25 BID25 BID27 BID27 ; rectangle BID25 BID29 BID27 7) circle BID25 BID29 (0, BID28 BID29 6) circle BID27 BID25 ; reflect(x = 6){ circle BID29 BID29 ; circle BID25 9) ; line BID29 BID28 BID27 BID26 , arrow = True,solid = True); line BID29 8, BID26 BID29 , arrow = True,solid = True); line BID25 8, BID25 6 , arrow = True,solid = True) } for (i < 3){ line(7,1,5 * i + 2,3, arrow = True,solid = True); for (j < (1 * i + 1)){ if (j > 0){ line(5 * j + -1,9,5 * i,5, arrow = True,solid = True) } line(5 * j + 2,5,5 * j + 2,9, arrow = True,solid = True) }; rectangle (5 * i,3,5 * i + 4,5) ; rectangle(5 * i,9,5 * i + 4,10) }; rectangle BID26 0, 12, BID25 reflect(y = 8){ for (i < 3){ circle(-3 * i + 7,-3 * i + 7) }; rectangle BID26 BID26 BID27 BID27 ; rectangle BID29 BID29 6, 6 BID25 8) ; reflect(x = 10){ line(6,1,9,3, arrow = False,solid = True); line BID26 8, BID28 8 , arrow = False,solid = True); line(9,5,9,7, arrow = False,solid = True); rectangle(0,3,2,5) }; rectangle BID28 7, 6, 9) line BID27 BID26 BID29 BID28 , arrow = True,solid = True); line(6,6,6,5, arrow = True,solid = True); line (8, BID27 7, BID28 , arrow = True,solid = True); line BID28 0, 12, 8 BID25 8, BID28 BID29 arrow = True, solid = True) reflect(x = 14){ circle(11,10); circle BID27 BID28 ; circle(7,1); reflect(y = 20){ circle(13,7); circle(9,7) }; line BID27 BID27 7, BID26 , arrow = True,solid = True); line(10,10,5,8, arrow = True,solid = True); reflect(x = 6){ line BID29 12, BID27 11 , arrow = True,solid = True); line BID25 6, BID27 BID29 , arrow = True,solid = True); line BID27 9, BID29 8 , arrow = True,solid = True) } } circle FIG0 ; circle(4,1); rectangle (6, 0, 8, BID26 ; rectangle (9, 0, 11, BID26 Solver timeout reflect(x = 10){ circle(5,1); circle BID26 BID28 ; line (2,3,5,2, BID26 BID28 BID28 6) ; rectangle BID25 0, 13, 9) ; line(9,6,9,8, arrow = False,solid = True); line(6,4,7,5, arrow = False,solid = True); line(10,5,12,5, arrow = False,solid = True); line BID27 BID25 BID27 BID28 , arrow = False,solid = True) circle(6,2); for (i < 3){ circle(5 * i + 1,7) }; line BID29 7, BID26 7 , arrow = True,solid = True); line(6,6,6,3, arrow = True,solid = True); line(10,7,7,7, arrow = True,solid = True); rectangle BID28 0, 8, 9) reflect ( BID25 9) ; for (i < 4){ circle(-2 * i + 9,-2 * i + 9) } } for (i < 3){ circle(1,-4 * i + 9); circle(5,-4 * i + 9); for (j < 3){ if (j > 0){ line(4 * i + -3,-4 * j + 10,4 * arrow = False,solid = True) } line(2,-4 * j + 9,4,-4 * j + 9, arrow = False,solid = True) } } circle(2,2); circle(2,6); circle BID26 11) ; line BID26 BID29 BID26 BID27 , arrow = True,solid = True); line(2,10,2,7, arrow = True,solid = True); rectangle (0, 0, BID28 9) for (i < 2){ circle(4,6 * i + 1); circle(1,6 * i + 4); rectangle(0,6 * i,2,6 * i + 2); rectangle (3,6 * i + 3,5,6 * i + }

@highlight

Learn to convert a hand drawn sketch into a high-level program