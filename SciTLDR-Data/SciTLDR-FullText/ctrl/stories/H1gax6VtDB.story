A structured understanding of our world in terms of objects, relations, and hierarchies is an important component of human cognition.

Learning such a structured world model from raw sensory data remains a challenge.

As a step towards this goal, we introduce Contrastively-trained Structured World Models (C-SWMs).

C-SWMs utilize a contrastive approach for representation learning in environments with compositional structure.

We structure each state embedding as a set of object representations and their relations, modeled by a graph neural network.

This allows objects to be discovered from raw pixel observations without direct supervision as part of the learning process.

We evaluate C-SWMs on compositional environments involving multiple interacting objects that can be manipulated independently by an agent, simple Atari games, and a multi-object physics simulation.

Our experiments demonstrate that C-SWMs can overcome limitations of models based on pixel reconstruction and outperform typical representatives of this model class in highly structured environments, while learning interpretable object-based representations.

Compositional reasoning in terms of objects, relations, and actions is a central ability in human cognition (Spelke & Kinzler, 2007) .

This ability serves as a core motivation behind a range of recent works that aim at enriching machine learning models with the ability to disentangle scenes into objects, their properties, and relations between them (Chang et al., 2016; Battaglia et al., 2016; Watters et al., 2017; van Steenkiste et al., 2018; Kipf et al., 2018; Sun et al., 2018; 2019b; Xu et al., 2019) .

These structured neural models greatly facilitate predicting physical dynamics and the consequences of actions, and provide a strong inductive bias for generalization to novel environment situations, allowing models to answer counterfactual questions such as "What would happen if I pushed this block instead of pulling it?".

Arriving at a structured description of the world in terms of objects and relations in the first place, however, is a challenging problem.

While most methods in this area require some form of human annotation for the extraction of objects or relations, several recent works study the problem of object discovery from visual data in a completely unsupervised or self-supervised manner (Eslami et al., 2016; Greff et al., 2017; Nash et al., 2017; van Steenkiste et al., 2018; Kosiorek et al., 2018; Janner et al., 2019; Xu et al., 2019; Burgess et al., 2019; Greff et al., 2019; Engelcke et al., 2019) .

These methods follow a generative approach, i.e., they learn to discover object-based representations by performing visual predictions or reconstruction and by optimizing an objective in pixel space.

Placing a loss in pixel space requires carefully trading off structural constraints on latent variables vs. accuracy of pixel-based reconstruction.

Typical failure modes include ignoring visually small, but relevant features for predicting the future, such as a bullet in an Atari game (Kaiser et al., 2019) , or wasting model capacity on visually rich, but otherwise potentially irrelevant features, such as static backgrounds.

To avoid such failure modes, we propose to adopt a discriminative approach using contrastive learning, which scores real against fake experiences in the form of state-action-state triples from an experience buffer (Lin, 1992) , in a similar fashion as typical graph embedding approaches score true facts in the form of entity-relation-entity triples against corrupted triples or fake facts.

We introduce Contrastively-trained Structured World Models (C-SWMs), a class of models for learning abstract state representations from observations in an environment.

C-SWMs learn a set of abstract state variables, one for each object in a particular observation.

Environment transitions are modeled using a graph neural network (Scarselli et al., 2009; Li et al., 2015; Kipf & Welling, 2016; Gilmer et al., 2017; Battaglia et al., 2018 ) that operates on latent abstract representations.

This paper further introduces a novel object-level contrastive loss for unsupervised learning of object-based representations.

We arrive at this formulation by adapting methods for learning translational graph embeddings (Bordes et al., 2013; Wang et al., 2014) to our use case.

By establishing a connection between contrastive learning of state abstractions (François-Lavet et al., 2018; and relational graph embeddings (Nickel et al., 2016a) , we hope to provide inspiration and guidance for future model improvements in both fields.

In a set of experiments, where we use a novel ranking-based evaluation strategy, we demonstrate that C-SWMs learn interpretable object-level state abstractions, accurately learn to predict state transitions many steps into the future, demonstrate combinatorial generalization to novel environment configurations and learn to identify objects from scenes without supervision.

Our goal is to learn an object-oriented abstraction of a particular observation or environment state.

In addition, we would like to learn an action-conditioned transition model of the environment that takes object representations and their relations and interactions into account.

We start by introducing the general framework for contrastive learning of state abstractions and transition models without object factorization in Sections 2.1-2.2, and in the following describe a variant that utilizes object-factorized state representations, which we term a Structured World Model.

We consider an off-policy setting, where we operate solely on a buffer of offline experience, e.g., obtained from an exploration policy.

Formally, this experience buffer B = {(s t , a t , s t+1 )} T t=1 contains T tuples of states s t ∈ S, actions a t ∈ A, and follow-up states s t+1 ∈ S, which are reached after taking action a t .

We do not consider rewards as part of our framework for simplicity.

Our goal is to learn abstract or latent representations z t ∈ Z of environment states s t ∈ S that discard any information which is not necessary to predict the abstract representation of the followup state z t+1 ∈ Z after taking action a t .

Formally, we have an encoder E : S → Z which maps observed states to abstract state representations and a transition model T : Z × A → Z operating solely on abstract state representations.

Our starting point is the graph embedding method TransE (Bordes et al., 2013) : TransE embeds facts from a knowledge base K = {(e t , r t , o t )} T t=1 , which consists of entity-relation-entity triples (e t , r t , o t ), where e t is the subject entity (analogous to the source state s t in our case), r t is the relation (analogous to the action a t in our experience buffer), and o t is the object entity (analogous to the target state s t+1 ).

function, conditioned on both the action and the (embedded) source state via z t = E(s t ).

The overall energy of a state-action-state triple then can be defined as follows:

This additive form of the transition model provides a strong inductive bias for modeling effects of actions in the environment as translations in the abstract state space.

Alternatively, one could model effects as linear transformations or rotations in the abstract state space, which motivates the use of a graph embedding method such as RESCAL (Nickel et al., 2011) , CompleX (Trouillon et al., 2016) , or HolE (Nickel et al., 2016b) .

With the aforementioned modifications, we arrive at the following energy-based hinge loss:

defined for a single (s t , a t , s t+1 ) with a corrupted abstract statez t = E(s t )

.s t is sampled at random from the experience buffer.

The margin γ is a hyperparameter for which we found γ = 1 to be a good choice.

Unlike Bordes et al. (2013), we place the hinge only on the negative term instead of on the full loss and we do not constrain the norm of the abstract states z t , which we found to work better in our context (see Appendix A.3).

The overall loss is to be understood as an expectation of the above over samples from the experience buffer B.

Our goal is to take into account the compositional nature of visual scenes, and hence we would like to learn a relational and object-oriented model of the environment that operates on a factored abstract state space Z = Z 1 × . . .

× Z K , where K is the number of available object slots.

We further assume an object-factorized action space A = A 1 × . . .

× A K .

This factorization ensures that each object is independently represented and it allows for efficient sharing of model parameters across objects in the transition model.

This serves as a strong inductive bias for better generalization to novel scenes and facilitates learning and object discovery.

The overall C-SWM model architecture using object-factorized representations is shown in Figure 1 .

Figure 1: The C-SWM model is composed of the following components: 1) a CNN-based object extractor, 2) an MLP-based object encoder, 3) a GNN-based relational transition model, and 4) an object-factorized contrastive loss.

Colored blocks denote abstract states for a particular object.

Encoder and Object Extractor We split the encoder into two separate modules: 1) a CNN-based object extractor E ext , and 2) an MLP-based object encoder E enc .

The object extractor module is a CNN operating directly on image-based observations from the environment with K feature maps in its last layer.

Each feature map m k t = [E ext (s t )] k can be interpreted as an object mask corresponding to one particular object slot, where [. . .]

k denotes selection of the k-th feature map.

For simplicity, we only assign a single feature map per object slot which sufficed for the experiments considered in this work (see Appendix A.4).

To allow for encoding of more complex object features (other than, e.g., position/velocity), the object extractor can be adapted to produce multiple feature maps per object slot.

After the object extractor module, we flatten each feature map m k t (object mask) and feed it into the object encoder E enc .

The object encoder shares weights across objects and returns an abstract state representation:

D in the following, where D is a hyperparameter.

We implement the transition model as a graph neural network (Scarselli et al., 2009; Li et al., 2015; Kipf & Welling, 2016; Battaglia et al., 2016; Gilmer et al., 2017; Battaglia et al., 2018) , which allows us to model pairwise interactions between object states while being invariant to the order in which objects are represented.

After the encoder stage, we have an abstract state description z k t ∈ Z k and an action a k t ∈ A k for every object in the scene.

We represent actions as one-hot vectors (or a vector of zeros if no action is applied to a particular object), but note that other choices are possible, e.g., for continuous action spaces.

The transition function then takes as input the tuple of object representations z t = (z 1 t , . . .

, z K t ) and actions a t = (a 1 t , . . .

, a K t ) at a particular time step:

T (z t , a t ) is implemented as a graph neural network (GNN) that takes z

where e

is an intermediate representation of the edge or interaction between nodes i and j.

This corresponds to a single round of node-to-edge and edge-to-node message passing.

Alternatively, one could apply multiple rounds of message passing, but we did not find this to be necessary for the experiments considered in this work.

Note that this update rule corresponds to message passing on a fully-connected scene graph, which is O(K 2 ).

This can be reduced to linear complexity by reducing connectivity to nearest neighbors in the abstract state space, which we leave for future work.

We denote the output of the transition function for the k-th object as ∆z

We only need to change the energy function to take the factorization of the abstract state space into account, which yields the following energy H for positive triples andH for negative samples:

wherez k t is the k-th object representation of the negative state samplez t = E(s t ).

The overall contrastive loss for a single state-action-state sample from the experience buffer then takes the form:

3 RELATED WORK For coverage of related work in the area of object discovery with autoencoder-based models, we refer the reader to the Introduction section.

We further discuss related work on relational graph embeddings in Section 2.2.

Structured Models of Environments Recent work on modeling structured environments such as interacting multi-object or multi-agent systems has made great strides in improving predictive accuracy by explicitly taking into account the structured nature of such systems (Sukhbaatar et al., 2016; Chang et al., 2016; Battaglia et al., 2016; Watters et al., 2017; Hoshen, 2017; Wang et al., 2018; van Steenkiste et al., 2018; Kipf et al., 2018; Sanchez-Gonzalez et al., 2018; Xu et al., 2019) .

These methods generally make use of some form of graph neural network, where node update functions model the dynamics of individual objects, parts or agents and edge update functions model their interactions and relations.

Several recent works succeed in learning such structured models directly from pixels (Watters et al., 2017; van Steenkiste et al., 2018; Xu et al., 2019; Watters et al., 2019) , but in contrast to our work rely on pixel-based loss functions.

The latest example in this line of research is the COBRA model (Watters et al., 2019) , which learns an action-conditioned transition policy on object representations obtained from an unsupervised object discovery model (Burgess et al., 2019) .

Unlike C-SWM, COBRA does not model interactions between object slots and relies on a pixel-based loss for training.

Our object encoder, however, is more limited and utilizing an iterative object encoding process such as in MONet (Burgess et al., 2019) would be interesting for future work.

Contrastive Learning Contrastive learning methods are widely used in the field of graph representation learning (Bordes et al., 2013; Perozzi et al., 2014; Grover & Leskovec, 2016; Bordes et al., 2013; Schlichtkrull et al., 2018; Veličković et al., 2018) , and for learning word representations (Mnih & Teh, 2012; Mikolov et al., 2013) .

The main idea is to construct pairs of related data examples (positive examples, e.g., connected by an edge in a graph or co-occuring words in a sentence) and pairs of unrelated or corrupted data examples (negative examples), and use a loss function that scores positive and negative pairs in a different way.

Most energy-based losses (LeCun et al., 2006) are suitable for this task.

Recent works (Oord et al., 2018; Hjelm et al., 2018; Hénaff et al., 2019; Sun et al., 2019a; Anand et al., 2019) connect objectives of this kind to the principle of learning representations by maximizing mutual information between data and learned representations, and successfully apply these methods to image, speech, and video data.

State Representation Learning State representation learning in environments similar to ours is often approached by models based on autoencoders (Corneil et al., 2018; Watter et al., 2015; Ha & Schmidhuber, 2018; Hafner et al., 2019; Laversanne-Finot et al., 2018) or via adversarial learning (Kurutach et al., 2018; Wang et al., 2019) .

Some recent methods learn state representations without requiring a decoder back into pixel space.

Examples include the selectivity objective in , the contrastive objective in François-Lavet et al. (2018), the mutual information objective in Anand et al. (2019) , the distribution matching objective in Gelada et al. (2019) or using causalitybased losses and physical priors in latent space (Jonschkowski & Brock, 2015; Ehrhardt et al., 2018) .

Most notably, Ehrhardt et al. (2018) propose a method to learn an object detection module and a physics module jointly from raw video data without pixel-based losses.

This approach, however, can only track a single object at a time and requires careful balancing of multiple loss functions.

Our goal of this experimental section is to verify whether C-SWMs can 1) learn to discover object representations from environment interactions without supervision, 2) learn an accurate transition model in latent space, and 3) generalize to novel, unseen scenes.

Our implementation is available under https://github.com/tkipf/c-swm.

We evaluate C-SWMs on two novel grid world environments (2D shapes and 3D blocks) involving multiple interacting objects that can be manipulated independently by an agent, two Atari 2600 games (Atari Pong and Space Invaders), and a multi-object physics simulation (3-body physics).

See Figure 2 for example observations.

For all environments, we use a random policy to collect experience for both training and evaluation.

Observations are provided as 50 × 50 × 3 color images for the grid world environments and as 50 × 50×6 tensors (two concatenated consecutive frames) for the Atari and 3-body physics environments.

Additional details on environments and dataset creation can be found in Appendix B.

In order to evaluate model performance directly in latent space, we make use of ranking metrics, which are commonly used for the evaluation of link prediction models, as in, e.g., Bordes et al. (2013) .

This allows us to assess the quality of learned representations directly without relying on auxiliary metrics such as pixel-based reconstruction losses, or performance in downstream tasks such as planning.

Given an observation encoded by the model and an action, we use the model to predict the representation of the next state, reached after taking the action in the environment.

This predicted state representation is then compared to the encoded true observation after taking the action in the environment and a set of reference states (observations encoded by the model) obtained from the experience buffer.

We measure and report both Hits at Rank 1 (H@1) and Mean Reciprocal Rank (MRR).

Additional details on these evaluation metrics can be found in Appendix C.

Autoencoder-based World Models The predominant method for state representation learning is based on autoencoders, and often on the VAE (Kingma & Welling, 2013; Rezende et al., 2014) model in particular.

This World Model baseline is inspired by Ha & Schmidhuber (2018) and uses either a deterministic autoencoder (AE) or a VAE to learn state representations.

Finally, an MLP is used to predict the next state after taking an action.

This model by Jaques et al. (2019) is based on an encoderdecoder architecture and trained with pixel-based reconstruction losses, but uses a differentiable physics engine in the latent space that operates on explicit position and velocity representations for each object.

Thus, this model is only applicable to the 3-body physics environment.

We train C-SWMs on an experience buffer obtained by running a random policy on the respective environment.

We choose 1000 episodes with 100 environment steps each for the grid world environments, 1000 episodes with 10 steps each for the Atari environments and 5000 episodes with 10 steps each for the 3-body physics environment.

For evaluation, we populate a separate experience buffer with 10 environment steps per episode and a total of 10.000 episodes for the grid world environments, 100 episodes for the Atari environments and 1000 episodes for the physics environment.

For the Atari environments, we minimize train/test overlap by 'warm-starting' experience collection in these environments with random actions before we start populating the experience buffer (see Appendix B), and we ensure that not a single full test set episode coincides exactly with an episode from the training set.

The state spaces of the grid world environments are large (approx.

6.4M unique states) and hence train/test coincidence of a full 10-step episode is unlikely.

Overlap is similarly unlikely for the physics environment which has a continuous state space.

Hence, performing well on these tasks will require some form of generalization to new environment configurations or an unseen sequence of states and actions.

All models are trained for 100 epochs (200 for Atari games) using the Adam (Kingma & Ba, 2014) optimizer with a learning rate of 5 · 10 −4 and a batch size of 1024 (512 for baselines with decoders due to higher memory demands, and 100 for PAIG as suggested by the authors).

Model architecture details are provided in Appendix D.

We present qualitative results for the grid world environments in Figure 3 and for the 3-body physics environment in Figure 4 .

All results are obtained on hold-out test data.

Further qualitative results (incl. on Atari games) can be found in Appendix A.

In the grid world environments, we can observe that C-SWM reliably discovers object-specific filters for a particular scene, without direct supervision.

Further, each object is represented by two coordinates which correspond (up to a random linear transformation) to the true object position in (a) Discovered object masks in a scene from the 3D cubes (top) and 2D shapes (bottom) environments.

(b) Learned abstract state transition graph of the yellow cube (left) and the green square (right), while keeping all other object positions fixed at test time.

Figure 3: Discovered object masks (left) and direct visualization of the 2D abstract state spaces and transition graphs for a single object (right) in the block pushing environments.

Nodes denote state embeddings obtained from a test set experience buffer with random actions and edges are predicted transitions.

The learned abstract state graph clearly captures the underlying grid structure of the environment both in terms of object-specific latent states and in terms of predicted transitions, but is randomly rotated and/or mirrored.

The model further correctly captures that certain actions do not have an effect if a neighboring position is blocked by another object (shown as colored spheres), even though the transition model does not have access to visual inputs.

The model learns to smoothly embed object trajectories, with the circular motion represented in the latent space (projected from four to two dimensions via PCA).

In the abstract state transition graph, orange nodes denote starting states for a particular episode, green links correspond to ground truth transitions and violet links correspond to transitions predicted by the model.

One trajectory (in the center) strongly deviates from typical trajectories seen during training, and the model struggles to predict the correct transition.

the scene.

Although we choose a two-dimensional latent representation per object for easier visualization, we find that results remain unchanged if we increase the dimensionality of the latent representation.

The edges in this learned abstract transition graph correspond to the effect of a particular action applied to the object.

The structure of the learned latent representation accurately captures the underlying grid structure of the environment.

We further find that the transition model, which only has access to latent representations, correctly captures whether an action has an effect or not, e.g., if a neighboring position is blocked by another object.

Similarly, we find that the model can learn object-specific encoders in the 3-body physics environment and can learn object-specific latent representations that track location and velocity of a particular object, while learning an accurate latent transition model that generalizes well to unseen environment instances.

We set up quantitative experiments for evaluating the quality of both object discovery and the quality of the learned transition model.

We compare against autoencoder baselines and model variants that do not represent the environment in an object-factorized manner, do not use a GNN, or do not make use of contrastive learning.

Performing well under this evaluation setting requires some degree of (combinatorial) generalization to unseen environment instances.

We report ranking scores (in %) in latent space, after encoding source and target observations, and taking steps in the latent space using the learned model.

Reported results are mean and standard error of scores over 4 runs on hold-out environment instances.

Results are summarized in Table 1 .

We find that baselines that make use of reconstruction losses in pixel space (incl.

the C-SWM model variant without contrastive loss) typically generalize less well to unseen scenes and learn a latent space configuration that makes it difficult for the transition model to learn the correct transition function.

See Appendix A.1 for a visual analysis of such latent state transition graphs.

This effect appears to be even stronger when using a VAE-based World Model, where the prior puts further constraints on the latent representations.

C-SWM recovers this structure well, see Figure 3 .

On the grid-world environments (2D shapes and 3D blocks), C-SWM models latent transitions almost perfectly, which requires taking interactions between latent representations of objects into account.

Removing the interaction component, i.e., replacing the latent GNN with an object-wise MLP, makes the model insensitive to pairwise interactions and hence the ability to predict future states deteriorates.

Similarly, if we remove the state factorization, the model has difficulties generalizing to unseen environment configurations.

We further explore variants of the grid-world environments in Appendix A.5, where 1) certain actions have no effect, and 2) one object moves randomly and independent of agent actions, to test the robustness of our approach.

For the Atari 2600 experiments, we find that results can have a high variance, and that the task is more difficult, as both the World Model baseline and C-SWM struggle to make perfect longterm predictions.

While for Space Invaders, a large number of object slots (K = 5) appears to be beneficial, C-SWM achieves best results with only a single object slot in Atari Pong.

This suggests that one should determine the optimal value of K based on performance on a validation set if it is not known a-priori.

Using an iterative object encoding mechanism, such as in MONet (Burgess et al., 2019) , would enable the model to assign 'empty' slots which could improve robustness w.r.t.

the choice of K, which we leave for future work.

We find that both C-SWMs and the autoencoder-based World Model baseline excel at short-term predictions in the 3-body physics environment, with C-SWM having a slight edge in the 10 step prediction setting.

Under our evaluation setting, the PAIG baseline (Jaques et al., 2019) underperforms using the hyperparameter setting recommended by the authors.

Note that we do not tune hyperparameters of C-SWM separately for this task and use the same settings as in other environments.

Instance Disambiguation In our experiments, we chose a simple feed-forward CNN architecture for the object extractor module.

This type of architecture cannot disambiguate multiple instances of the same object present in one scene and relies on distinct visual features or labels (e.g., the green square) for object extraction.

To better handle scenes which contain potentially multiple copies of the same object (e.g., in the Atari Space Invaders game), one would require some form of iterative disambiguation procedure to break symmetries and dynamically bind individual objects to slots or object files (Kahneman & Treisman, 1984; Kahneman et al., 1992) , such as in the style of dynamic routing (Sabour et al., 2017)

Our formulation of C-SWMs does not take into account stochasticity in environment transitions or observations, and hence is limited to fully deterministic worlds.

A probabilistic extension of C-SWMs is an interesting avenue for future work.

For simplicity, we make the Markov assumption: state and action contain all the information necessary to predict the next state.

This allows us to look at single state-action-state triples in isolation.

To go beyond this limitation, one would require some form of memory mechanism, such as an RNN as part of the model architecture, which we leave for future work.

Structured world models offer compelling advantages over pure connectionist methods, by enabling stronger inductive biases for generalization, without necessarily constraining the generality of the model: for example, the contrastively trained model on the 3-body physics environment is free to store identical representations in each object slot and ignore pairwise interactions, i.e., an unstructured world model still exists as a special case.

Experimentally, we find that C-SWMs make effective use of this additional structure, likely because it allows for a transition model of significantly lower complexity, and learn object-oriented models that generalize better to unseen situations.

We are excited about the prospect of using C-SWMs for model-based planning and reinforcement learning in future work, where object-oriented representations will likely allow for more accurate counterfactual reasoning about effects of actions and novel interactions in the environment.

We further hope to inspire future work to think beyond autoencoder-based approaches for object-based, structured representation learning, and to address some of the limitations outlined in this paper.

We visualize abstract state transition graphs separated by object slot for the 3D cubes environment in Figure 5 .

Discovered object representations in the 2D shapes dataset (not shown) are qualitatively very similar.

We apply the same visualization technique to the model variant without contrastive loss, which is instead trained with a decoder model and a loss in pixel space.

See Figure 6 for this baseline and note that the regular structure in the latent space is lost, which makes it difficult for the transition model to learn transitions which generalize to unseen environment instances.

Qualitative results for the 3-body physics dataset are summarized in Figures 7 and 8 for two different random seeds.

Figure 5: Abstract state transition graphs per object slot for a trained C-SWM model on the 3D cubes environment (with all objects allowed to be moved, i.e., none are fixed in place).

Edge color denotes action type.

The abstract state graph is nearly identical for each object, which illustrates that the model successfully represents objects in the same manner despite their visual differences.

Figure 6: Abstract state transition graphs per object slot for a trained SWM model without contrastive loss, using instead a loss in pixel space, on the 3D cubes environment.

Edge color denotes action type.

(a) Discovered object-specific filters.

For the Atari 2600 environments, we generally found latent object representations to be less interpretable.

We attribute this to the fact that a) objects have different roles and are in general not exchangeable (in contrast to the block pushing grid world environments and the 3-body physics environment), b) actions affect only one object directly, but many other objects indirectly in two consecutive frames, and c) due to multiple objects in one scene sharing the same visual features.

See Figure 9 for an example of learned representations in Atari Pong and Figure 10 for an example in Space Invaders.

Figure 11: Qualitative model comparison in pixel space on a hold-out test instance of the 2D shapes environment.

We train a separate decoder model for 100 epochs on both the C-SWM and the World Model baseline using all training environment instances to obtain pixel-based reconstructions for multiple prediction steps into the future.

To supplement our model comparison in latent space using ranking metrics, we here show a direct comparison in pixel space at the example of the 2D shapes environment.

This requires training a separate decoder model for C-SWM.

For fairness of this comparison, we use the same protocol to train a separate decoder for the World Model (AE) baseline (discarding the one obtained from the original end-to-end auto-encoding training procedure).

This decoder has the same architecture as in the other baseline models and is trained for 100 epochs.

1 2 3 4 5 6 7 8 9 10 Prediction steps For a qualitative comparison, see Figure 11 .

The C-SWM model, as expected from our ranking analysis in latent space, performs almost perfectly at this task.

Although the World Model (AE) baseline makes clear mistakes which compound over time, it nonetheless often gets several object positions correct after many time steps.

The ranking loss in latent space captures this behaviour well, and, for example, assigns an almost perfect score for 1-step prediction to the World Model (AE) baseline.

The typically used mean-squared error (MSE) in pixel space (see Figure 12 ), however, differs by several orders of magnitude between the two models and does not capture any of the nuanced differences in qualitative predictive behavior.

We hence strongly encourage researchers in this area to consider ranking-based evaluations directly in latent space in addition to, or as a replacement for comparisons in pixel space.

In Table 2 , we summarize results of a comparison between our multi-object contrastive loss from Eq. 6, denoted by C-SWM (original), and a full-hinge (or triplet) loss that places the hinge over both the positive and the negative energy term, i.e., L = max(0, γ + H −H).

The latter loss is similar to the loss employed in Bordes et al. (2013) , but without imposing a norm constraint on the embeddings z t .

Reported results are mean and standard error over 4 runs on hold-out environment instances.

Table 2 : Comparison of hinge loss variants on 2D shapes environment.

Numbers denote ranking scores (in %) for multi-step prediction in latent space.

Highest (mean) scores in bold.

Step 5 We find that placing the hinge only on the negative energy term significantly improves results, likely because the full hinge loss can be minimized by simply increasing the norms of the embeddings z t , as outlined in Bordes et al. (2013) .

This could be addressed by introducing additional constraints, such as by forcing the embeddings z t to lie on the unit-hypersphere, as done in TransE (Bordes et al., 2013), which we do not consider in this work as most of the environment dynamics in our experiments obey a flat Euclidean geometry (e.g., translation on a grid).

We experimented with assigning multiple feature maps per object slot to investigate whether this can improve performance on the Space Invaders environment (as an example of one of the more complex environments considered in this work).

All other model settings are left unchanged.

Results are summarized in Table 3 .

The original model variant with one feature map per object slot (using a total of K = 5 slots) is denoted by C-SWM (original).

Reported results are mean and standard error over 4 runs on hold-out environment instances.

We find that there is no clear advantage in using multiple feature maps per object slot for the Space Invaders environment, but this might nonetheless prove useful for environments of more complex nature.

To further evaluate the robustness of our approach, we experimented with two variants of the 2D shapes grid-world environment with five objects.

In the first variant, termed no-op action, we include an additional 'no-op' action for each object that has no effect.

This mirrors the setting present in many Atari 2600 benchmark environments that have an explicit no-op action.

In the second variant, termed random object, we assign a special property to one out of the five objects: a random action is applied to this object at every turn, i.e., the agent only has control over the other four objects and in every environment step, a total of two objects receive an action (one by the agent, which only affects one out of the first four objects, and one random action on the last object).

This is to test how the model behaves in the presence of other agents that act randomly.

As our model is fully deterministic, it is to be expected that adding a source of randomness in the environment will pose a challenge.

We report results in Table 4 .

Reported numbers are mean and standard error over 4 runs on hold-out environment instances.

We find that adding a no-op action has little effect on our model, but adding a randomly moving object reduces predictive performance.

Performance in the latter case could potentially be improved by explicitly modeling stochasticity in the environment.

Table 4 : Comparison of C-SWM on variants of the 2D shapes environment.

Numbers denote ranking scores (in %) for multi-step prediction in latent space.

Step 5 The discovered object representations and identifications can vary between runs.

While we found that this process is very stable for the simple grid world environments where actions only affect a single object, we found that results can be initialization-dependent on the Atari environments, where actions can have effects across all objects, and the 3-body physics simulation (see Figures 7 and 8) , which does not have any actions.

In some cases, the discovered representation can be less suitable for forward prediction in time or for generalization to novel scenes, which explains the variance in some of our results on these datasets.

A.7 TRAINING TIME

We found that the overall training time of the C-SWM model was comparable to that of the World Model baseline.

Both C-SWM and the World Model baseline trained for approx.

1 hour wall-clock time on the 2D shapes dataset, approx.

2 hours on the 3D cubes dataset, and approx.

30min on the 3-body physics environment using a single NVIDIA GTX1080Ti GPU.

The Atari Pong and Space Invaders models trained for typically less than 20 minutes.

A notable exception is the PAIG baseline model (Jaques et al., 2019) which trained for approx.

6 hours on a NVIDIA TitanX Pascal GPU using the recommended settings by the authors of the paper.

To generate an experience buffer for training, we initialize the environment with random object placements and uniformly sample an object and an object-specific action at every time step.

We provide state observations as 50 × 50 × 3 tensors with RGB color channels, normalized to [0, 1].

Actions are provided as a 4-dim one-hot vector (if an action is applied) or a vector of zeros per object slot in the environment.

The action one-hot vector encodes the directional movement action applied to a particular object, or is represented as a vector of zeros if no action is applied to a particular object.

Note that only a single object receives an action per time step.

For the Atari environments, we provide a copy of the one-hot encoded action vector to every object slot, and for the 3-body physics environment, which has no actions, we do not provide an action vector.

2D Shapes This environment is a 5 × 5 grid world with 5 different objects placed at random positions.

Each location can only be occupied by at maximum one object.

Each object is represented by a unique shape/color combination, occupying 10 × 10 pixels on the overall 50 × 50 pixel grid.

At each time step, one object can be selected and moved by one position along the four cardinal directions.

See Figure 2a for an example.

The action has no effect if the target location in a particular direction is occupied by another object or outside of the 5 × 5 grid.

Thus, a learned transition model needs to take pairwise interactions between object properties (i.e., their locations) into account, as it would otherwise be unable to predict the effect of an action correctly.

To investigate to what degree our model is robust to partial occlusion and perspective changes, we implement a simple block pushing environment using Matplotlib (Hunter, 2007) as a rendering tool.

The underlying environment dynamics are the same as in the 2D Shapes dataset, and we only change the rendering component to make for a visually more challenging task that involves a different perspective and partial occlusions.

See Figure 2b for an example.

Atari Pong We make use of the Arcade Learning Environment (Bellemare et al., 2013) to create a small environment based on the Atari 2600 game Pong which is restricted to the first interaction between the ball and the player-controlled paddle, i.e., we discard the first couple of frames from the experience buffer where the opponent behaves completely independent of any player action.

Specifically, we discard the first 58 (random) environment interactions.

We use the PONGDETERMINISTIC-V4 variant of the environment in OpenAI Gym (Brockman et al., 2016) .

We use a random policy and populate an experience buffer with 10 environment interactions per episode, i.e., T = 10.

An observation consists of two consecutive frames, cropped (to exclude the score) and resized to 50×50 pixels each.

See Figure 2c for an example.

Space Invaders This environment is based on the Atari 2600 game Space Invaders, using the SPACEINVADERSDETERMINISTIC-V4 variant of the environment in OpenAI Gym (Brockman et al., 2016) , and processed / restricted in a similar manner as the Pong environment.

We discard the first 50 (random) environment interactions for each episode and only begin populating the experience buffer thereafter.

See Figure 2d for an example observation.

The 3-body physics simulation environment is an interacting system that evolves according to classical gravitational dynamics.

Different from the other environments considered here, there are no actions.

This environment is adapted from Jaques et al. (2019) using their publicly available implementation 1 , where we set the step size (dt) to 2.0 and the initial maximum x and y velocities to 0.5.

We concatenate two consecutive frames of 50 × 50 pixels each to provide the model with (implicit) velocity information.

See Figure 4 for an example observation sequence.

This score is 1 for a particular example if the predicted state representation is in the k-nearest neighbor set around the encoded true observation, where we define the neighborhood of a node to include the node itself.

Otherwise this score is 0.

In other words, this score measures whether the rank of the predicted state representation is smaller than or equal to k, when ranking all reference state representations by distance to the true state representation.

We report the average of this score over a particular evaluation dataset.

This score is defined as the average inverse rank, i.e., MRR = 1 N N n=1 1 rankn , where rank n is the rank of the n-th sample.

For the 3D cubes environment, the object extractor is a 4-layer CNN with 3 × 3 filters, zero-padding, and 16 feature maps per layer, with the exception of the last layer, which has K = 5 feature maps, i.e., one per object slot.

After each layer, we apply BatchNorm (Ioffe & Szegedy, 2015) and a ReLU(x) = max(0, x) activation function.

For the 2D shapes environment, we choose a simpler CNN architecture with only a single convolutional layer with 10 × 10 filters and a stride of 10, followed by BatchNorm and a ReLU activation or LeakyReLU (Xu et al., 2015) for the Atari 2600 and physics environments.

This layer has 16 feature maps and is followed by a channel-wise linear transformation (i.e., a 1 × 1 convolution), with 5 feature maps as output.

For both models, we choose a sigmoid activation function after the last layer to obtain object masks with values in (0, 1).

We use the same two-layer architecture for the Atari 2600 environments and the 3-body physics environment, but with 9 × 9 filters (and zero-padding) in the first layer, and 5 × 5 filters with a stride of 5 in the second layer.

After reshaping/flattening the output of the object extractor, we obtain a vector representation per object (2500-dim for the 3D cubes environment, 25-dim for the 2D shapes environment, and 1000-dim for Atari 2600 and physics environments).

The object encoder is an MLP with two hidden layers of 512 units and each, followed by ReLU activation functions.

We further use LayerNorm (Ba et al., 2016) before the activation function of the second hidden layer.

The output of the final output layer is 2-dimensional (4-dimensional for Atari 2600 and physics environments), reflecting the ground truth object state, i.e., the object coordinates in 2D (although this is not provided to the model), and velocity (if applicable).

Both the node and the edge model in the GNN-based transition model are MLPs with the same architecture / number of hidden units as the object encoder model, i.e., two hidden layers of 512 units each, LayerNorm and ReLU activations.

The margin in the hinge loss is chosen as γ = 1.

We further multiply the squared Euclidean distance d(x, y) in the loss function with a factor of 0.5/σ 2 with σ = 0.5 to control the spread of the embeddings.

We use the same setting in all experiments.

World Model Baseline The World Model baseline is trained in two stages: First, we train an autoencoder or a VAE with a 32-dim latent space, where the encoder is a CNN with the same architecture as the object extractor used in the C-SWM model, followed by an MLP with the same architecture as the object encoder module in C-SWM on the flattened representation of the output of the encoder CNN.

The decoder exactly mirrors this architecture where we replace convolutional layers with deconvolutional layers.

We verified that this architecture can successfully build representations of single frames.

Example reconstructions from the latent code are shown in Figure 13 .

We experimented both with mean squared error and binary cross entropy (using the continuous channel values in [0, 1] as targets) as reconstruction loss in the (V)AE models, both of which are typical choices in most practical implementations.

We generally found binary cross entropy to be more stable to optimize and to produce better results, which is why we opted for this loss in all the baselines using decoders considered in the experimental section.

In the second stage, we freeze the model parameters of the auto-encoder and train a transition model with mean-squared error on the latent representations.

For the VAE model, we use the predicted mean values of the latent representations.

This transition model takes the form of an MLP with the same architecture and number of hidden units as the node model in C-SWM.

We experimented both with a translational transition model (i.e., the transition model only predicts the latent state difference, instead of the full next state) and direct prediction of the next state.

We generally found that the translational transition model performed better and used it throughout all the reported results.

The World Model baselines are trained with a smaller batch size of 512, which slightly improved performance and simplified memory management.

Ablations We perform the following ablations: 1) we replace the latent GNN with an MLP (per object, i.e., we remove the edge update function) to investigate whether a structured transition model is necessary, 2) we remove the state factorization and embed the full scene into a single latent variable of higher dimensionality (original dimensionality × number of original object slots) with an MLP as transition model, and 3) we replace the contrastive loss with a pixel-based reconstruction loss on both the current state and the predicted next state (we add a decoder that mirrors the architecture of the encoder).

Physics-as-Inverse-Graphics (PAIG) For this baseline, we train the PAIG model from Jaques et al. (2019) with the code provided by the authors 2 on our dataset with the standard settings recommended by the authors for this particular task, namely: model=PhysicsNet, epochs=500, batch size=100, base lr=1e-3, autoencoder loss=5.0, anneal lr=true, color=true, and cell type=gravity ode cell.

We use the same input size as in C-SWM, i.e., frames are of shape 50 × 50 × 3.

We further set input steps=2, pred steps=10 and extrap steps=0 to match our setting of predicting for a total of 10 frames while conditioning on a pair of 2 initial frames to obtain initial position and velocity information.

We train this model with four different random seeds.

For evaluation, we extract learned position representations for all frames of the test set and further run the latent physics prediction module (conditioned on the first two initial frames) to obtain model predictions.

We further augment position representations with velocity information by taking the difference between two consecutive position representations and concatenating this representation with the 2-dim position representation, which we found to slightly improve results.

In total, we obtain a 12-dim (4-dim × 3 objects) representation for each time step, i.e., the same as C-SWM.

From this, we obtain ranking metrics.

We found that one of the training runs (seed=2) collapsed all latent representations to a single point.

We exclude this run in the results reported in Table 1 , i.e., we report average and standard error over the three other runs only.

<|TLDR|>

@highlight

Contrastively-trained Structured World Models (C-SWMs) learn object-oriented state representations and a relational model of an environment from raw pixel input.

@highlight

The authors overcome the problem of using pixel-based losses in the construction and learning of structured world models by using a contrastive latent space.