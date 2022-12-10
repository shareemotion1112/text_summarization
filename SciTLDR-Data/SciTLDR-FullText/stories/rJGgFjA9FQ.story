This paper presents two methods to disentangle and interpret contextual effects that are encoded in a pre-trained deep neural network.

Unlike convolutional studies that visualize image appearances corresponding to the network output or a neural activation from a global perspective, our research aims to clarify how a certain input unit (dimension) collaborates with other units (dimensions) to constitute inference patterns of the neural network and thus contribute to the network output.

The analysis of local contextual effects w.r.t.

certain input units is of special values in real applications.

In particular, we used our methods to explain the gaming strategy of the alphaGo Zero model in experiments, and our method successfully disentangled the rationale of each move during the game.

Interpreting the decision-making logic hidden inside neural networks is an emerging research direction in recent years.

The visualization of neural networks and the extraction of pixel-level inputoutput correlations are two typical methodologies.

However, previous studies usually interpret the knowledge inside a pre-trained neural network from a global perspective.

For example, BID17 BID14 BID10 mined input units (dimensions or pixels) that the network output is sensitive to; BID2 visualized receptive fields of filters in intermediate layers; BID33 BID15 BID24 BID5 BID6 BID20 illustrated image appearances that maximized the score of the network output, a filter's response, or a certain activation unit in a feature map.

However, instead of visualizing the entire appearance that is responsible for a network output or an activation unit, we are more interested in the following questions.• How does a local input unit contribute to the network output?

Here, we can vectorize the input of the network into a high-dimensional vector, and we treat each dimension as a specific "unit" without ambiguity.

As we know, a single input unit is usually not informative enough to make independent contributions to the network output.

Thus, we need to clarify which other input units the target input unit collaborates with to constitute inference patterns of the neural network, so as to pass information to high layers.• Can we quantitatively measure the significance of above contextual collaborations between the target input unit and its neighboring units?Method: Therefore, given a pre-trained convolutional neural network (CNN), we propose to disentangle contextual effects w.r.t.

certain input units.

As shown in Fig. 1 , we design two methods to interpret contextual collaborations at different scales, which are agnostic to the structure of CNNs.

The first method estimates a rough region of contextual collaborations, i.e. clarifying whether the target input unit mainly collaborates with a few neighboring units or most units of the input.

This method distills knowledge from the pre-trained network into a mixture of local models (see Fig. 2 ), where each model encodes contextual collaborations within a specific input region to make predictions.

We hope that the knowledge-distillation strategy can help people determine quantitative contributions from different regions.

Then, given a model for Extracting fine-grained contextual effects from a student net A lattice within the Go board Figure 1 : Explaining the alphaGo model.

Given the state of the Go board and the next move, we use the alphaGo model to explain the rationale of the move.

We first estimate a rough region of contextual collaborations w.r.t.

the current move by distilling knowledge from the value net to student nets that receive different regions of the Go board as inputs.

Then, given a student net, we analyze fine-grained contextual collaborations within its region of the Go board.

In this figure, we use a board state from a real Go game between humans for clarity.local collaborations, the second method further analyzes the significance of detailed collaborations between each pair of input units, when we use the local model to make predictions on an image.

The quantitative analysis of contextual collaborations w.r.t.

a local input unit is of special values in some tasks.

For example, explaining the alphaGo model BID22 BID7 October 2017) is a typical application.

The alphaGo model contains a value network to evaluate the current state of the game-a high output score indicates a high probability of winning.

As we know, the contribution of a single move (i.e. placing a new stone on the Go board) to the output score during the game depends on contextual shapes on the Go board.

Thus, disentangling explicit contextual collaborations that contribute to the output of the value network is important to understand the logic of each new move hidden in the alphaGo model.

More crucially, in this study, we explain the alphaGo Zero model BID7 , which extends the scope of interests of this study from diagnosing feature representations of a neural network to a more appealing issue letting self-improving AI teach people new knowledge.

The alphaGo Zero model is pre-trained via self-play without receiving any prior knowledge from human experience as supervision.

In this way, all extracted contextual collaborations represent the automatically learned intelligence, rather than human knowledge.

As demonstrated in well-known Go competitions between the alphaGo and human players (alp, Retrieved 17 March 2016; 2017-05-27) , the automatically learned model sometimes made decisions that could not be explained by existing gaming principles.

The visualization of contextual collaborations may provide new knowledge beyond people's current understanding of the Go game.

Contributions of this paper can be summarized as follows.(i) In this paper, we focus on a new problem, i.e. visualizing local contextual effects in the decisionmaking of a pre-trained neural network w.r.t.

a certain input unit.(ii) We propose two new methods to extract contextual effects via diagnosing feature representations and knowledge distillation.(iii) We have combined two proposed methods to explain the alphaGo Zero model, and experimental results have demonstrated the effectiveness of our methods.

Understanding feature representations inside neural networks is an emerging research direction in recent years.

Related studies include 1) the visualization and diagnosis of network features, 2) disentangling or distilling network feature representations into interpretable models, and 3) learning neural networks with disentangled and interpretable features in intermediate layers.

Network visualization: Instead of analyzing network features from a global view BID30 BID19 BID16 , BID2 BID33 BID15 BID24 BID5 BID32 BID34 showed the appearance that maximized the score of a given unit.

BID5 used up-convolutional nets to invert CNN feature maps to their corresponding images.

Pattern retrieval: Some studies retrieved certain units from intermediate layers of CNNs that were related to certain semantics, although the relationship between a certain semantics and each neural unit was usually convincing enough.

People usually parallel the retrieved units similar to conventional mid-level features BID25 of images.

BID37 selected units from feature maps to describe "scenes".

BID23 discovered objects from feature maps.

Model diagnosis and distillation: Model-diagnosis methods, such as the LIME BID17 , the SHAP (Lundberg & Lee, 2017), influence functions BID11 , gradientbased visualization methods BID6 BID20 , and BID12 extracted image regions that were responsible for network outputs.

BID29 BID36 ) distilled knowledge from a pre-trained neural network into explainable models to interpret the logic of the target network.

Such distillation-based network explanation is related to the first method proposed in this paper.

However, unlike previous studies distilling knowledge into explicit visual concepts, our using distillation to disentangle local contextual effects has not been explored in previous studies.

A new trend is to learn networks with meaningful feature representations in intermediate layers BID9 BID26 BID13 in a weakly-supervised or unsupervised manner.

For example, capsule nets BID18 and interpretable RCNN learned interpretable middle-layer features.

InfoGAN BID3 and β-VAE BID8 learned meaningful input codes of generative networks.

BID35 ) developed a loss to push each middle-layer filter towards the representation of a specific object part during the learning process without given part annotations.

All above related studies mainly focused on semantic meanings of a filter, an activation unit, a network output.

In contrast, our work first analyzes quantitative contextual effects w.r.t.

a specific input unit during the inference process.

Clarifying explicit mechanisms of how an input unit contributes to the network output has special values in applications.

In the following two subsections, we will introduce two methods that extract contextual collaborations w.r.t.

a certain input unit from a CNN at different scales.

Then, we will introduce the application that uses the proposed methods to explain the alphaGo Zero model.

Since the input feature usually has a huge number of dimensions (units), it is difficult to accurately discover a few input units that collaborate with a target input unit.

Therefore, it is important to first approximate the rough region of contextual collaborations before the unit-level analysis of contextual collaborations, i.e. clarifying in which regions contextual collaborations are contained.

Given a pre-trained neural network, an input sample, and a target unit of the sample, we propose a method that uses knowledge distillation to determine the region of contextual collaborations w.r.t.

the target input unit.

Let I ∈

I denote the input feature (e.g. an image or the state in a Go board).Note that input features of most CNNs can be represented as a tensor I ∈ R H×W ×D , where H and W indicate the height of the width of the input, respectively; D is the channel number.

We clip different lattices (regions) Λ 1 , Λ 2 , . . .

, Λ N ∈ Λ from the input tensor, and input units within the i-th lattice are given as I Λi ∈ R h×w×D , h ≤ H, w ≤ W .

Different lattices overlap with each other.

The core idea is that we use a mixture of models to approximate the function of the given pretrained neural network (namely the teacher net), where each model is a student net and uses input information within a specific lattice I Λi to make predictions.

DISPLAYFORM0 Generate weights 2x2 lattices for the first type of student nets 3x3 lattices for the second type of student nets.

We only illustrate three of the nine lattices for clarity.

Figure 2: Division of lattices for two types of student nets.

We distill knowledge from the value net into a mixture of four/nine student nets to approximate decision-making logic of the value net.

whereŷ = f (I) and y i = f i (I Λi ) denote the output of the pre-trained teacher net f and the output of the i-th student net f i , respectively.

α i is a scalar weight, which depends on the input I. Because different lattices within the input are not equally informative w.r.t.

the target task, input units within different lattices make different contributions to final network output.

More crucially, given different inputs, the importance for the same lattice may also change.

For example, as shown in BID20 , the head appearance is the dominating feature in the classification of animal categories.

Thus, if a lattice corresponds to the head, then this lattice will contribute more than other lattices, thereby having a large weight α i .

Therefore, our method estimates a specific weight α i for each input I, i.e. α i is formulated as a function of I (which will be introduced later).Significance of contextual collaborations: Based on the above equation, the significance of contextual collaborations within each lattice Λ i w.r.t.

an input unit can be measured as DISPLAYFORM0 Impacts from the first lattice Λ1 DISPLAYFORM1 where we revise the value of the target unit in the input and check the change of network outputs, DISPLAYFORM2 If contextual collaborations w.r.t.

the target unit mainly localize within the i-th lattice Λ i , then α i · ∆y i can be expected to contribute the most to the change ofŷ.

We conduct two knowledge-distillation processes to learn student nets and a model of determining {α i }, respectively.

The first process distills knowledge from the teacher net to each student net f i with parameters θ i based on the distillation loss min θi I∈I y I,i −ŷ I 2 , where the subscript I indicates the output for the input I. Considering that Λ i only contains partial information of I, we do not expect y I,i to reconstructŷ I without any errors.

Distilling knowledge to weights: Then, the second distillation process estimates a set of weights α = [α I,1 , α I,2 , . . .

, α I,n ] for each specific input I. We use the following loss to learn another neural network g with parameters θ g to infer the weight.

DISPLAYFORM0 3.2 FINE-GRAINED CONTEXTUAL COLLABORATIONS w.r.t.

AN INPUT UNITIn the above subsection, we introduce a method to distill knowledge of contextual collaborations into student nets of different regions.

Given a student net, in this subsection, we develop an approach to disentangling from the student net explicit contextual collaborations w.r.t.

a specific input unit u, i.e. identifying which input unit v collaborates with u to compute the network output.

We can consider a student net as a cascade of functions of N layers, i.e. DISPLAYFORM1 , where x (l) denotes the output feature of the l-th layer.

In particular, x (0) and x (n) indicate the input and output of the network, respectively.

We only focus on a single scalar output of the network (we may handle different output dimensions separately if the network has a high-dimensional output).

If the sigmoid/softmax layer is the last layer, we use the score before the softmax/sigmoid operation as x (n) to simplify the analysis.

As preliminaries of our algorithm, we extend the technique of BID21 to estimate the quantitative contribution of each neural activation in a feature map to the final prediction.

We use C x ∈ R H l ×W l ×D l to denote the contribution distribution of neural activations on the l-th layer x ∈ R H l ×W l ×D l .

The score of the i-th element C xi denotes the ratio of the unit x i 's score contribution w.r.t.

the entire network output score.

Because x (n) is the scalar network output, it has a unit contribution C x (n) = 1.

Then, we introduce how to back-propagate contributions to feature maps in low layers.

The method of contribution propagation is similar to network visualization based on gradient backpropagation BID15 BID32 .

However, contribution propagation reflects more objective distribution of numerical contributions over {x i }, instead of biasedly boosting compacts of the most important activations.

Without loss of generality, in this paragraph, we use o = φ(x) to simplify the notation of the function of a certain layer.

If the layer is a conv-layer or a fully-connected layer, then we can represent the convolution operation for computing each elementary activation score o i of o in a vectorized form DISPLAYFORM0 We consider x j w j as the numerical contribution of x j to o i .

Thus, we can decompose the entire contribution of o i , C oi , into elementary contributions of x j , i.e. C oi→xj = C oi · xj wj oi+max{−b,0} , which satisfies C oi→xj ∝ x j w j (see the appendix for details).

Then, the entire contribution of x j is computed as the sum of elementary contributions from all o i in the above layer, i.e. C xj = i C oi→xj .A cascade of a conv-layer and a batch-normalization layer can be rewritten in the form of a single conv-layer, where normalization parameters are absorbed into the conv-layer 1 .

For skip connections, a neural unit may receive contributions from different layers, C x DISPLAYFORM1 .

If the layer is a ReLU layer or a Pooling layer, the contribution propagation has the same formulation as gradient back-propagations of those layers 1 .

As discussed in BID2 , each neural activation o i of a middle-layer feature o can be considered as the detection of a mid-level inference pattern.

All input units must collaborate with neighboring units to activate some middle-layer feature units, in order to pass their information to the network output.

Therefore, in this research, we develop a method to 1. determine which mid-level patterns (or which neural activations o i ) the target unit u constitutes; 2. clarify which input units v help the target u to constitute the mid-level patterns; 3. measure the strength of the collaboration between u and v.

Let o bfr and o denote the feature map of a certain conv-layer o = f (x) when the network receives input features with the target unit u being activated and the feature map generated without u being activated, respectively.

In this way, we can use |o − o bfr | to represent the absolute effect of u on the feature map o. The overall contribution of the i-th neural unit C oi depends on the activation score o i , C oi ∝ max{o i , 0}, where max{o i , 0} measures the activation strength used for inference.

The proportion of the contribution is affected by the target unit u can be roughly formulated asC o .

DISPLAYFORM0 where C oi = 0 and thusC oi = 0 if o i ≤ 0, because negative activation scores of a conv-layer cannot pass information through the following ReLU layer (o is not the feature map of the last conv-layer before the network output).In this way,C oi highlights a few mid-level patterns (neural activations) related to the target unit u.

C o measures the contribution proportion that is affected by the target unit u.

We can useC o to replace C o and use techniques in Section 3.2.1 to propagateC o back to input units DISPLAYFORM1 represents a map of fine-grained contextual collaborations w.r.t.

u. Each element in the mapC DISPLAYFORM2 j 's collaboration with u.

We can understand the proposed method as follows.

The relative activation change DISPLAYFORM3 can be used as a weight to evaluate the correlation between u and the i-th activation unit (inference pattern).

In this way, we can extract input units that make great influences on u's inference patterns, rather than affect all inference patterns.

Note that both u and v may either increase or decrease the value of o i .

It means that the contextual unit v may either boost u's effects on the inference pattern, or weaken u's effects.

We use the ELF OpenGo BID28 BID30 as the implementation of the alphaGo Zero model.

We combine the above two methods to jointly explain each move's logic hidden in the value net of the alphaGo Zero model during the game.

As we know, the alphaGo Zero model contains a value net, policy nets, and the module of the Monte-Carlo Tree Search (MCTS).

Generally speaking, the superior performance of the alphaGo model greatly relies on the enumeration power of the policy net and the MCTS, but the value net provides the most direct information about how the model evaluates the current state of the game.

Therefore, we explain the value net, rather than the policy net or the MCTS.

In the ELF OpenGo implementation, the value net is a residual network with 20 residual blocks, each containing two conv-layers.

We take the scalar output 2 before the final (sigmoid) layer as the target value to evaluate the current state on the Go board.

Given the current move of the game, our goal is to estimate unit-level contextual collaborations w.r.t.

the current move.

I.e. we aim to analyze which neighboring stones and/or what global shapes help the current move make influences to the game.

We distill knowledge from the value net to student networks to approximate contextual collaborations within different regions.

Then, we estimate unitlevel contextual collaborations based on the student net.

Determining local contextual collaborations: We design two types of student networks, which receive lattices at the scales of 13 × 13 and 10 × 10, respectively.

In this way, we can conduct two distillation processes to learn neural networks that encode contextual collaborations at different scales.

As shown in Fig. 2 , we have four student nets {f i |i = 1, . . . , 4} oriented to 13 × 13 lattices.

Except for the output, the four student nets have the same network structure as the value net.

The four student nets share parameters in all layers.

The input of a student net only has two channels corresponding to maps of white stones and black stones, respectively, on the Go board.

We crop four overlapping lattices at the four corners of the Go board for both training and testing.

Note that we rotate the board state within each lattice I Λi to make the top-left position corresponds to the corner of the board, before we input I Λi to the student net.

The neural network g has the same settings as the value net.

g receives a concatenation of [I Λ1 , . . .

, I Λ4 ] as the input.

g outputs four scalar weights {α i } for the four local student networks {y i }.

We learn g via knowledge distillation.

Student nets for 10×10 lattices have similar settings as those for 13×13 lattices.

We divide the entire Go board into 3 × 3 overlapping 10 × 10 lattices.

Nine student nets encode local knowledge from nine local lattices.

We learn another neural network g, which uses a concatenation of [I Λ1 , . . .

, I Λ9 ] to weight for the nine local lattices.

Finally, we select the most relevant 10 × 10 lattice and the most relevant 13 × 13 lattice, via max i s i , for explanation.

Estimating unit-level contextual collaborations: In order to obtain fine-grained collaborations, we apply the method in Section 3.2.2 to explain two student nets corresponding to the two selected relevant lattices.

We also use our method to explain the value net.

We compute a map of contextual collaborations for each neural network and normalize values in the map.

We sum up maps of the three networks together to obtain the final map of contextual collaborationsĈ.More specifically, given a neural network, we use the feature of each conv-layer to compute the initialC o in Equation (4) and propagatedC o to obtain a map of collaborationsC x (0) .

We sum up maps based on the 1st, 3rd, 5th, and 7th conv-layers to obtain the collaboration map of the network.

In experiments, we distilled knowledge of the value network to student nets, and disentangled finegrained contextual collaborations w.r.t.

each new move.

We compared the extracted contextual collaborations and human explanations for the new move to evaluate the proposed method.

In this section, we propose two metrics to evaluate the accuracy of the extracted contextual collaborations w.r.t.

the new move.

Note that considering the high complexity of the Go game, there is no exact ground-truth explanation for contextual collaborations.

Different Go players usually have different analysis of the same board state.

More crucially, as shown in competitions between the alphaGo and human players (alp, Retrieved 17 March 2016; 2017-05-27) , the knowledge encoded in the alphaGo was sometimes beyond humans' current understanding of the Go game and could not be explained by existing gaming principles.

In this study, we compared the similarity between the extracted contextual collaborations and humans' analysis of the new move.

The extracted contextual collaborations were just rough explanations from the perspective of the alphaGo.

We expected these collaborations to be close to, but not exactly the same as human understanding.

More specifically, we invited Go players who had obtained four-dan grading rank to label contextual collaborations.

To simplify the metric, Go players were asked to label a relative strength value of the collaboration between each stone and the target move (stone), no matter whether the relationship between the two stones was collaborative or adversarial.

Considering the double-blind policy, the paper will introduce the Go players if the paper is accepted.

Let Ω be a set of existing stones except for the target stone u on the Go board.

p v ≥ 0 denotes the labeled collaboration strength between each stone v ∈ Ω and the target stone u. q v = |Ĉ v | is referred to as the collaboration strength estimated by our method, whereĈ v denotes the final estimated collaboration value on the stone v. We normalized the collaboration strength,p v = p v / v p v , q v = q v / v q v and computed the Jaccard similarity between the distribution of p and the distribution of q as the similarity metric.

In addition, considering the great complexity of the Go game, different Go players may annotate different contextual collaborations.

Therefore, we also required Go players to provide a subjective rating for the extracted contextual collaborations of each board state, i.e. selecting one of the five ratings: 1-Unacceptable, 2-Problematic, 3-Acceptable, 4-Good, and 5-Perfect.

FIG0 shows the significance of the extracted contextual collaborations, as well as possible explanations for contextual collaborations, where the significance of the stone v's contextual collaboration was reported as the absolute collaboration strength q v instead of the original scoreĈ v in experiments.

Without loss of generality, let us focus on the winning probability of the black.

Considering the complexity of the Go game, there may be two cases of a positive (or negative) value of the collaboration scoreĈ v .

The simplest case is that when a white stone had a negative value ofĈ v , it means that the white stone decreased the winning probability of the black.

However, sometimes a white stone had a positiveĈ v .

It may be because that this white stone did not sufficiently exhibit its power due to its contexts.

Since the white and the white usually had a very similar number of stones in the Go board, putting a relatively ineffective white stone in a local region also wasted the opportunity of winning advantages in other regions in the zero-sum game.

Similarly, the black stone may also have either a positive or a negative value ofĈ v .

The Jaccard similarity between the extracted collaborations and the manually-annotated collaborations was 0.3633.

Nevertheless, considering the great diversity of explaining the same game state,

The black stone at (7,2) has a high value, because it collaborates with the new stone to escape from the surrounding of the white.

The white stone at (7,4) has a high value, because it is about to be eaten by the new black stone.

The black stone at (8,4) has a high value, because it collaborates with the new stone to eat the white stone at (7,4).(0,0) (18, 18) Explanations for the estimated collaborations (0,0) (18,18) The black stone at (8,6) has a high value, because it collaborates with the new stone, because it collaborates with the new black stone to get a head out of the white's regime.

The two black stone also communicate with black stones on the top.

The white stone at (7,7) has a high value, because future white stones can only be placed to the right to escape from the regime of the new black stone.(0,0) (18, 18) The black stone at (2,7) has a high value, because it collaborates with the new black stone to separate white stones into the left and right groups, which increases the probability of attacking white stones on the left in the future.

The white stone at (1,6) has a high value, because the new black stone reduces the white stone's space of "making eyes" in the future.(0,0) (18, 18) The black stone at (4,3) has a high value, because the new black stone helps this black stone to get a head out of the white's regime.

The white stone at (5,2) has a high value, because the new black stone limits the potential of the white's future development in its neighboring area.

the average rating score that was made by Go players for the extracted collaborations was 3.7 (between 3-Acceptable and 4-Good).

Please see the appendix for more results.

In this paper, we have proposed two typical methods for quantitative analysis of contextual collaborations w.r.t.

a certain input unit in the decision-making of a neural network.

Extracting fine-grained contextual collaborations to clarify the reason why and how an input unit passes its information to the network output is of significant values in specific applications, but it has not been well explored before, to the best of our knowledge.

In particular, we have applied our methods to the alphaGo Zero model, in order to explain the potential logic hidden inside the model that is automatically learned via self-play without human annotations.

Experiments have demonstrated the effectiveness of the proposed methods.

Note that there is no exact ground-truth for contextual collaborations of the Go game, and how to evaluate the quality of the extracted contextual collaborations is still an open problem.

As a pioneering study, we do not require the explanation to be exactly fit human logics, because human logic is usually not the only correct explanations.

Instead, we just aim to visualize contextual collaborations without manually pushing visualization results towards human-interpretable concepts.

This is different from some previous studies of network visualization BID15 BID32 that added losses as the natural image prior, in order to obtain beautiful but biased visualization results.

In the future, we will continue to cooperate with professional Go players to further refine the algorithm to visualize more accurate knowledge inside the alphaGo Zero model.

Let o = ω ⊗ x + β denote the convolutional operation of a conv-layer.

We can rewrite the this equation in a vectorized form as DISPLAYFORM0 If the conv-layer is a fully-connected layer, then each element W ij corresponds to an element in ω.

Otherwise, W is a sparse matrix, i.e. W ij = 0 if o i and x j are too far way to be covered by the convolutional filter.

Thus, we can write o i = j x j w j + b to simplify the notation.

Intuitively, we can propagate the contribution of o i to its compositional elements x j based on their numerical scores.

Note that we only consider the case of o i > 0, because if o i ≤ 0, o i cannot pass information through the ReLU layer, and we obtain C oi = 0 and thus C oi→xj = 0.

In particular, when b ≥ 0, all compositional scores just contribute an activation score o i − b, thereby receiving a total contribution of C oi oi−b oi .

When b < 0, we believe the contribution of C oi all comes from elements of {x j }, and each element's contribution is given a C oi · xj wj oi−b .

Thus, we get DISPLAYFORM1 When a batch-normalization layer follows a conv-layer, then the function of the two cascaded layers can be written as DISPLAYFORM2 Thus, we can absorb parameters for the batch normalization into the conv-layer, i.e. w j ← For ReLU layers and Pooling layers, the formulation of the contribution propagation is identical to the formulation for the gradient back-propagation, because the gradient back-propagation and the contribution propagation both pass information to neural activations that are used during the forward propagation.

Considering the great complexity of the Go game, there do not exist ground-truth annotations for the significance of contextual collaborations.

Different Go players may have different understanding of the same Go board state, thereby annotating different heat maps for the significance of contextual collaborations.

More crucially, our results reflect the logic of the automatically-learned alphaGo Zero model, rather than the logic of humans.

Therefore, in addition to manual annotations of collaboration significance, we also require Go players to provide a subjective evaluation for the extracted contextual collaborations.

94.63% Figure 6 : We show the significance of contextual collaborations within a local lattice.

The score for the i-th lattice is reported as si j sj .

@highlight

This paper presents methods to disentangle and interpret contextual effects that are encoded in a deep neural network.