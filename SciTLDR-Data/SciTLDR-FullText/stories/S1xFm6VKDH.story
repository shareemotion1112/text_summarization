There are two main lines of research on visual reasoning: neural module network (NMN) with explicit multi-hop reasoning through handcrafted neural modules, and monolithic network with implicit reasoning in the latent feature space.

The former excels in interpretability and compositionality, while the latter usually achieves better performance due to model flexibility and parameter efficiency.

In order to bridge the gap of the two, we present Meta Module Network (MMN), a novel hybrid approach that can efficiently utilize a Meta Module to perform versatile functionalities, while preserving compositionality and interpretability through modularized design.

The proposed model first parses an input question into a functional program through a Program Generator.

Instead of handcrafting a task-specific network to represent each function like traditional NMN, we use Recipe Encoder to translate the functions into their corresponding recipes (specifications), which are used to dynamically instantiate the Meta Module into Instance Modules.

To endow different instance modules with designated functionality, a Teacher-Student framework is proposed, where a symbolic teacher pre-executes against the scene graphs to provide guidelines for the instantiated modules (student) to follow.

In a nutshell, MMN adopts the meta module to increase its parameterization efficiency, and uses recipe encoding to improve its generalization ability over NMN.

Experiments conducted on the GQA benchmark demonstrates that: (1) MMN achieves significant improvement over both NMN and monolithic network baselines; (2) MMN is able to generalize to unseen but related functions.

Visual reasoning requires a model to learn strong compositionality and generalization abilities, i.e., understanding and answering compositional questions without having seen similar semantic compositions before.

Such compositional visual reasoning is a hallmark for human intelligence that endows people with strong problem-solving skills given limited prior knowledge.

Recently, neural module networks (NMNs) (Andreas et al., 2016a; Hu et al., 2017; Johnson et al., 2017b; Hu et al., 2018; Mao et al., 2019) have been proposed to perform such complex reasoning tasks.

First, NMN needs to pre-define a set of functions and explicitly encode each function into unique shallow neural networks called modules, which are composed dynamically to build an instance-specific network for each input question.

This approach has high compositionality and interpretability, as each module is specifically designed to accomplish a specific sub-task and multiple modules can be combined to perform unseen combinations during inference.

However, with increased complexity of the task, the set of functional semantics and modules also scales up.

As observed in Hudson & Manning (2018) , this leads to higher model complexity and poorer scalability on more challenging scenarios.

Another line of research on visual reasoning is focused on designing monolithic network architecture, such as MFB (Yu et al., 2017) , BAN (Kim et al., 2018) , DCN (Nguyen & Okatani, 2018) , and MCAN .

These black-box methods have achieved state-of-the-art performance on more challenging realistic image datasets like VQA (Hudson & Manning, 2019a) , surpassing the aforementioned NMN approach.

They use a unified neural network to learn general-purpose reasoning skills (Hudson & Manning, 2018) , which is known to be more flexible and scalable without making strict assumption about the inputs or designing operation-specific networks for the predefined functional semantics.

As the reasoning procedure is conducted in the latent feature space, the reasoning process is difficult to interpret.

Such a model also lacks the ability to capture the compositionality of questions, thus suffering from poorer generalizability than module networks.

Final steps

Figure 1: The model architecture of Meta Module Network: the lower part describes how the question is translated into programs and instantiated into operation-specific modules; the upper part describes how execution graph is built based on the instantiated modules.

Motivated by this, we propose a Meta Module Network (MMN) to bridge the gap, which preserves the merit of interpretability and compositionality of traditional module networks, but without requiring strictly defined modules for different semantic functionality.

As illustrated in Figure 1 , instead of handcrafting a shallow neural network for each specific function like NMNs, we propose a flexible meta (parent) module g( * , * ) that can take a function recipe f as input and instantiates a (child) module g f ( * ) = g( * , f ) to accomplish the functionality specified in the recipe.

These instantiated modules with tied parameters are used to build an execution graph for answer prediction.

The introduced meta module empowers the MMN to scale up to accommodate a larger set of functional semantics without adding complexity to the model itself.

To endow each instance module with the designated functionality, we introduce module supervision to enforce each module g f ( * ) to imitate the behavior of its symbolic teacher learned from ground-truth scene graphs provided in the training data.

The module supervision can dynamically disentangle different instances to accomplish small sub-tasks to maintain high compositionality.

Our main contributions are summarized as follows.

(i) We propose Meta Module Network for visual reasoning, in which different instance modules can be instantiated from a meta module. (ii) Module supervision is introduced to endow different functionalities to different instance modules.

(iii) Experiments on GQA benchmark validate the outperformance of our model over NMN and monolithic network baselines.

We also qualitatively provide visualization on the inferential chain of MMN to demonstrate its interpretability, and conduct experiments to quantitatively showcase the generalization ability to unseen functional semantics.

The visual reasoning task (Hudson & Manning, 2019a ) is formulated as follows: given a question Q grounded in an image I, where Q = {q 1 , ?? ?? ?? , q M } with q i representing the i-th word, the goal is to select an answer a ??? A from a set A of possible answers.

During training, we are provided with an additional scene graph G for each image I, and a functional program P for each question Q. During inference, scene graphs and programs are not provided.

Figure 2 : Architecture of the Coarse-to-fine Program Generator: the left part depicts the coarse-tofine two-stage generation; the right part depicts the resulting execution graph.

The Visual Encoder is based on a pre-trained object detection model (Ren et al., 2015; Anderson et al., 2018 ) that extracts from image I a set of regional features

, where r i ??? R Dv , N denotes the number of region of interest, and D v denotes the feature dimension.

Similar to a Transformer block (Vaswani et al., 2017) , we first use two self-attention networks, SA q and SA r , to encode the question and the regional features asQ = SA q (Q, Q; ??) andR = SA r (R, R; ??), respectively, whereQ ??? R M ??D ,R ??? R N ??D , and D is the network's hidden dimension.

Based on this, a cross-attention network CA is applied to use the question as guidance to refine the visual features into V = CA(R,Q; ??) ??? R N ??D , whereQ is used as the query vector, and ?? denotes all the parameters in the Visual Encoder.

The attended visual features V will then be fed into the meta module, detailed in Sec. 2.3.

We visualize the encoder in the Appendix for better illustration.

Similar to other programming languages, we define a set of syntax rules for building valid programs and a set of semantics to determine the functionality of each program.

Specifically, we define a set of functions F with their fixed arity n f ??? {1, 2, 3, 4} based on the semantic string provided in Hudson & Manning (2019a) .

The definitions for all the functions are provided in the Appendix.

The defined functions can be divided into 10 different categories based on their abstract semantics (e.g., "relate, verify, filter"), and each abstract function type is further implemented with different realizations depending on their arguments (e.g., "verify attribute, verify geometric, verify relation").

In total, there are 48 different functions defined, whose returned values could be List of Objects, Boolean or String.

A program P is viewed as a sequence of function calls f 1 , ?? ?? ?? , f L .

For example, in Figure 2 , f 2 is Relate([1], beside, boy), the functionality of which is to find a boy who is beside the objects returned by f 1 : Select(ball).

Formally, we call Relate the "function name", [1] the "dependency", and beside, boy the "arguments".

By exploiting the dependency relationship between functions, we build an execution graph for answer prediction.

In order to generate syntactically plausible programs, we follow Dong & Lapata (2018) and adopt a coarse-to-fine two-stage generation paradigm, as illustrated in Figure 2 .

Specifically, the Transformer-based program generator (Vaswani et al., 2017) first decodes a sketch containing only function names, and then fills the dependencies and arguments into the sketch to generate the program P .

Such a two-stage generation process helps guarantee the plausibility and grammaticality of synthesized programs.

We apply the known constraints to enforce the syntax in the fine-grained generation stage.

For example, if function Filter is sketched, we know there are two tokens required to complete the function.

The first token should be selected from the dependency set ([1], [2], ...), while the second token should be selected from the attribute set (e.g., color, size).

With these syntactic constraints, our program synthesizer can achieve a 98.8% execution accuracy.

Instead of learning a full inventory of task-specific modules for different functions as in NMN (Andreas et al., 2016b), we design an abstract Meta Module that can instantiate a generic meta mod- ule into instance modules based on an input function recipe, which is a set of pre-defined keyvalue pairs specifying the properties of the function.

As exemplified in Figure 3 , when taking Function:relate; Geometric:to the left as the input, the Recipe Embedder produces a recipe vector to transform the meta module into a "geometric relation" module, which can search for target objects that the current object is to the left of.

The left part of Figure 3 demonstrates the computation flow in Meta Module based on multi-head attention network (Vaswani et al., 2017) .

Specifically, a Recipe Embedder encodes a function recipe into a real-valued vector r f ??? R D .

In the first attention layer, r f is fed into an attention network g d as the query vector to incorporate the output (?? 1:K ) of neighbor modules on which the current module is dependent.

The intermediate output (o d ) from this attention layer is further fed into a second attention network g v to incorporate the visual representation V of the image.

The final output from the is denoted as g(r f ,?? 1:

Here is how the instantiation process of Meta Module works.

First, we feed a function f to instantiate the meta module g into an instance module g f (?? 1:K , V; ??), where ?? denotes the parameters of the meta module.

The instantiated module is then used to build the execution graph on the fly as depicted in Figure 1 .

Each module g f outputs o(f ) ??? R D , which acts as the message passed to its neighbor modules.

For brevity, we use o(f i ) to denote the MMN's output at the i-th function f i .

The final output o(f L ) of function f L will be fed into a softmax-based classifier for answer prediction.

During training, we optimize the parameters ?? (in Meta Module) and the parameters ?? (in Visual Encoder) to maximize the likelihood p ??,?? (a|P, Q, R) on the training data, where a is the answer, and P, Q, R are programs, questions and visual features, respectively.

As demonstrated, Meta Module Network excels over standard module network in the following aspects.

(i) The parameter space of different functions is shared, which means similar functions can be jointly optimized, benefiting from more efficient parameterization.

For example, query color and verify color share the same partial parameters related to the input color. (ii) Our Meta Module can accommodate larger function semantics by using function recipes and scale up to more complex reasoning scenes. (iii) Since all the functions are embedded into the recipe space, functionality of an unseen recipe can be inferred from its neighboring recipes (see Sec. 3.4 for details), which equips our Meta Module with better generalization ability to unseen functions.

In this sub-section, we explain how to extract supervision signals from scene graphs and programs provided in the training data, and how to adapt these learning signals during inference when no scene graphs or programs are available.

We call this "Module Supervision", which is realized by a Teacher-Student framework as depicted in Figure 4 .

First, we define a Symbolic Executor as the 'Teacher', which can traverse the ground-truth scene graph provided in training data and obtain intermediate results by executing the programs.

The 'Teacher' exhibits these results as guideline ?? for the 'Student' instance module g f to adhere to during training.

i=1 .

Knowledge Transfer: As no scene graphs are provided during inference, we need to train a Student to mimic the Symbolic Teacher in associating objects between input images and generated programs for end-to-end model training.

To this end, we compare the execution results from the Symbolic Teacher with object detection results from the Visual Encoder to provide learning guideline for the Student.

Specifically, for the i-th step function f i , we compute the overlap between its execution result b i and all the model-detected regions R as a i,j = Intersect(bi,rj ) U nion(bi,rj ) .

If j a i,j > 0, which means that there exists detected bounding boxes overlapping with the ground-truth object, we normalize a i,j over R to obtain a guideline distribution ?? i,j = ai,j j ai,j and append an extra 0 in the end to obtain ?? i ??? R N +1 .

If j a i,j = 0, which means no detected bounding box has overlap with the ground-truth object (or

as the learning guideline.

The last bit represents "No Match".

Student Training: To explicitly teach the student module g f to follow the learning guideline from the Symbolic Teacher, we add an additional head to each module output o(f i ) to predict the execution result distribution, denoted as?? i = sof tmax(M LP (o(f i ))).

During training, we propel the instance module to align its prediction?? i with the guideline distribution ?? i by minimizing their KL divergence KL(?? i ||?? i ).

Formally, given the quadruple of (P, Q, R, a) and the pre-computed guideline distribution ??, we propose to add KL divergence to the standard loss function with a balancing factor ??:

In this section, we conduct the following experiments.

(i) We evaluate the proposed Meta Module Network on the GQA v1.1 dataset (Hudson & Manning, 2019a) , and compare with the state-ofthe-art methods.

(ii) We provide visualization of the inferential chains and perform fine-grained error analysis based on that. (iii) We design synthesized experiments to quantitatively measure our model's generalization ability towards unseen functional semantics.

Dataset The GQA dataset contains 22M questions over 140K images.

This full "all-split" dataset has unbalanced answer distributions, thus, is further re-sampled into a "balanced-split" with a more balanced answer distribution.

The new split consists of 1M questions.

Compared with the VQA v2.0 dataset (Goyal et al., 2017) , the questions in GQA are designed to require multi-hop reasoning to test the reasoning skills of developed models.

Compared with the CLEVR dataset (Johnson et al., 2017a) , GQA greatly increases the complexity of the semantic structure of questions, leading to a more diverse function set.

The real-world images in GQA also bring in a bigger challenge in visual understanding.

In GQA, around 94% of questions need multi-hop reasoning, and 51% questions are about the relationships between objects.

Following Hudson & Manning (2019a), the main evaluation metrics used in our experiments are accuracy, consistency, plausibility, and validity. (Pennington et al., 2014) are used to encode both questions and function keywords with 300 dimensions.

The total vocabulary size is 3761, including all the functions, objects, and attributes.

For training, we first use the 22M unbalanced "all-split" to bootstrap our model with a mini-batch size 2048 for 3-5 epochs, then fine-tune on the "balanced-split" with a mini-batch size 256.

The testdev-balanced split is used for selecting the best model.

We report our experimental results on the test2019 split (from the public GQA leaderboard) in Table 1.

First, we observe significant performance gain from MMN over NMN (Andreas et al., 2016b) , which demonstrates the effectiveness of the proposed meta module mechanism.

Further, we observe that our model outperforms the VQA state-of-the-art monolithic model MCAN by a large margin, which demonstrates the strong compositionality of our module-based approach.

Overall, our single model achieves competitive performance (tied top 2) among published approaches.

Notably, we achieve the same performance as LXMERT (Tan & Bansal, 2019) , which is pre-trained on large-scale out-of-domain datasets.

The performance gap with NSM (Hudson & Manning, 2019b) is debatable since our model is self-contained without relying on well-tuned external scene graph generation model (Xu et al., 2017; Yang et al., 2016; Chen et al., 2019) .

To verify the contribution of each component in MMN, we perform several ablation studies: (1) w/o Module Supervision vs. w/ Module Supervision.

We investigate the influence of module supervision by changing the hyper-parameter ?? from 0 to 2.0.

(2) Attention Supervision vs. Guideline: We investigate different module supervision strategies, by directly supervising multi-head attention in multi-modal fusion stage (Figure 1 ).

Specifically, we supervise different number of heads or the mean/max over different heads.

(3) w/o Bootstrap vs w/ Bootstrap: We investigate the effectiveness of bootstrapping in training to validate the influence of pre-training on the final model performance.

Results are summarized in Table 2 .

From Ablation (1), we observe that without module supervision, our MMN achieves decent performance improvement over MCAN , but with much fewer parameters.

By increasing ?? from 0.1 to 0.5, accuracy steadily improves, which reflects the importance of module supervision.

Further increasing the value of ?? did not improve the performance empirically.

From Ablation (2), we observe that directly supervising the attention weights in different Transformer heads only yields marginal improvement, which justifies the effectiveness of the implicit regularization in MMN.

From Ablation (3), we observe that bootstrapping is an important step for MMN, as it explores more data to better regularize functionalities of reasoning modules.

It is also observed that the epoch number of bootstrap also influences the final model performance.

Choosing the optimal epoch size can lead to a better initialization for the following fine-tuning stage.

Figure 5: Visualization of the inferential chains learned by our model.

To demonstrate the interpretability of MMN, Figure 5 provides some visualization results to show the inferential chain during reasoning.

As shown, the model correctly executes the intermediate results and yields the correct final answer.

To better interpret the model's behavior, we also perform quantitative analysis to diagnose the errors in the inferential chain.

Here, we held out a small validation set to analyze the execution accuracy of different functions.

Our model obtains Recall@1 of 59% and Recall@2 of 73%, which indicates that the object selected by the symbolic teacher has 59% chance of being top-1, and 73% chance as the top-2 by the student model, significantly higher than random-guess Recall@1 of 2%, demonstrating the effectiveness of module supervision.

Furthermore, we conduct detailed analysis on function-wise execution accuracy to understand the limitation of MMN.

Results are shown in Table 3 .

Below are the observed main bottlenecks: (i) relation-type functions such as relate, relate inv; and (ii) object/attribute recognition functions such as query name, query color.

We hypothesize that this might be attributed to the quality of visual features from standard object detection models (Anderson et al., 2018) , which does not capture the relations between objects well.

Besides, the object and attribute classification network is not fine-tuned on GQA.

This suggests that scene graph modeling for visual scene understanding is critical to surpassing NSM (Hudson & Manning, 2019b) on performance.

To demonstrate the generalization ability of the meta module, we perform additional experiments to validate whether the recipe representation can generalize to unseen functions.

Specifically, we held out all the training instances containing verify shape, relate name, choose name to quantitatively measure model's on these unseen functions.

Standard NMN (Andreas et al., 2016b) fails to handle these unseen functions, as it requires training instances for the randomly initialized shallow network for these unseen functions.

In contrast, MMN can transform the unseen functions Table 4 shows that the zero-shot accuracy of the proposed meta module is significantly higher than NMN (equivalent to random guess), which demonstrates the generalization ability of MMN and validate the extensibility of the proposed recipe encoding.

Instead of handcrafting new modules every time when new functional semantics comes in like NMN (Andreas et al., 2016b) , our MMN is more flexible and extensible for handling growing function sets under incremental learning.

Monolithic Networks: Most monolithic networks for visual reasoning resort to attention mechanism for multimodal fusion Zhou et al., 2017; Kim et al., 2016; Kafle et al., 2018; Li et al., 2019; Hu et al., 2019) .

To realize multi-hop reasoning on complex questions, SAN (Yang et al., 2016) , MAC (Hudson & Manning, 2018) and MuRel (Cadene et al., 2019) models have been proposed.

However, their reasoning procedure is built on a general-purpose reasoning block, which can not be disentangled to perform specific tasks, resulting in limited model interpretability and compositionality.

Neural Module Networks: By parsing a question into a program and executing the program through dynamically composed neural modules, NMN excels in interpretability and compositionality by design (Andreas et al., 2016a; Hu et al., 2017; Johnson et al., 2017b; Hu et al., 2018; Yi et al., 2018; Mao et al., 2019; Vedantam et al., 2019) .

However, its success is mostly restricted to the synthetic CLEVR dataset, whose performance can be surpassed by simpler methods such as relational network (Santoro et al., 2017) and FiLM (Perez et al., 2018) .

Our MMN is a module network in concept, thus possessing high interpretability and compositionality.

However, different from traditional NMN, MMN uses only one Meta Module for program execution recurrently, similar to an LSTM cell (Hochreiter & Schmidhuber, 1997) in Recurrent Neural Network.

This makes MMN a monolithic network in practice, which ensures strong empirical performance without sacrificing model interpretability.

State of the Art on GQA: GQA was introduced in Hudson & Manning (2019a) for real-world visual reasoning.

Simple monolithic networks ), MAC netowrk (Hudson & Manning, 2018 , and language-conditioned graph neural networks (Hu et al., 2019; Guo et al., 2019) have been developed for this task.

LXMERT (Tan & Bansal, 2019) , a large-scale pre-trained encoder, has also been tested on this dataset.

Recently, Neural State Machine (NSM) (Hudson & Manning, 2019b) proposed to first predict a probabilistic scene graph, then perform multi-hop reasoning over the graph for answer prediction.

The scene graph serves as a strong prior to the model.

Our model is designed to leverage dense visual features extracted from object detection models, thus orthogonal to NSM and can be enhanced with their scene graph generator once it is publicly available.

Different from the aforementioned approaches, MMN also performs explicit multi-hop reasoning based on predicted programs, so the inferred reasoning chain can be directly used for model interpretability.

In this paper, we propose Meta Module Network that bridges the gap between monolithic networks and traditional module networks.

Our model is built upon a Meta Module, which can be instantiated into an instance module performing specific functionalities.

Our approach significantly outperforms baseline methods and achieves comparable performance to state of the art.

Detailed error analysis shows that relation modeling over scene graph could further boost MMN for higher performance.

For future work, we plan to incorporate scene graph prediction into the proposed framework.

A APPENDIX

The visual encoder and multi-head attention network is illustrated in Figure 6 and Figure 7 , respectively.??#??$??%??&

Figure 6: Illustration of the Visual Encoder described in Section.

2.1.

The recipe embedder is illustrated in Figure 8 .

Figure 8 : Illustration of the recipe embedder.

The function statistics is listed in Table 5 .

The detailed function descriptions are provided in Figure 9 .

More inferential chains are visualized in Figure 10 and Figure 11 .

@highlight

We propose a new Meta Module Network to resolve some of the restrictions of previous Neural Module Network to achieve strong performance on realistic visual reasoning dataset.