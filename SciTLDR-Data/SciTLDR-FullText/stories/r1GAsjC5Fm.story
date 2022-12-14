The Vision-and-Language Navigation (VLN) task entails an agent following navigational instruction in photo-realistic unknown environments.

This challenging task demands that the agent be aware of which instruction was completed, which instruction is needed next, which way to go, and its navigation progress towards the goal.

In this paper, we introduce a self-monitoring agent with two complementary components: (1) visual-textual co-grounding module to locate the instruction completed in the past, the instruction required for the next action, and the next moving direction from surrounding images and (2) progress monitor to ensure the grounded instruction correctly reflects the navigation progress.

We test our self-monitoring agent on a standard benchmark and analyze our proposed approach through a series of ablation studies that elucidate the contributions of the primary components.

Using our proposed method, we set the new state of the art by a significant margin (8% absolute increase in success rate on the unseen test set).

Code is available at https://github.com/chihyaoma/selfmonitoring-agent.

Recently, the Vision-and-Language (VLN) navigation task BID3 , which requires the agent to follow natural language instructions to navigate through a photo-realistic unknown environment, has received significant attention BID46 BID19 ).

In the VLN task, an agent is placed in an unknown realistic environment and is required to follow natural language instructions to navigate from its starting location to a target location.

In contrast to some existing navigation tasks BID24 BID53 BID33 , we address the class of tasks where the agent does not have an explicit representation of the target (e.g., location in a map or image representation of the goal) to know if the goal has been reached or not BID31 BID22 BID18 BID6 .

Instead, the agent needs to be aware of its navigation status through the association between the sequence of observed visual inputs to instructions.

Consider an example as shown in FIG0 , given the instruction "Exit the bedroom and go towards the table.

Go to the stairs on the left of the couch.

Wait on the third step.", the agent first needs to locate which instruction is needed for the next movement, which in turn requires the agent to be aware of (i.e., to explicitly represent or have an attentional focus on) which instructions were completed or ongoing in the previous steps.

For instance, the action "Go to the stairs" should be carried out once the agent has exited the room and moved towards the table.

However, there exists inherent ambiguity for "go towards the table".

Intuitively, the agent is expected to "Go to the stairs" after completing "go towards the table".

But, it is not clear what defines the completeness of "Go towards the table".

The completeness of an ongoing action often depends on the availability of the next action.

Since the transition between past and next part of the instructions is a soft boundary, in order to determine when to transit and to follow the instruction correctly the agent is required to keep track of both grounded instructions.

On the other hand, assessing the progress made towards the goal has indeed been shown to be important for goal-directed tasks in humans decision-making BID8 BID12 BID9 .

While a number of approaches have been proposed for VLN BID3 BID46 BID19 , previous approaches generally are not aware of which instruction is next nor progress towards the goal; indeed, we qualitatively show that even the attentional mechanism of the baseline does not successfully track this information through time.

In this paper, we propose an agent endowed with the following abilities: (1) identify which direction to go by finding the part of the instruction that corresponds to the observed images-visual grounding, (2) identify which part of the instruction has been completed or ongoing and which part is potentially needed for the next action selection-textual grounding, and (3) ensure that the grounded instruction can correctly be used to estimate the progress made towards the goal, and apply regularization to ensure this -progress monitoring.

Therefore, we introduce the self-monitoring agent consisting of two complementary modules: visual-textual co-grounding and progress monitor.

More specifically, we achieve both visual and textual grounding simultaneously by incorporating the full history of grounded instruction, observed images, and selected actions into the agent.

We leverage the structural bias between the words in instructions used for action selection and progress made towards the goal and propose a new objective function for the agent to measure how well it can estimate the completeness of instruction-following.

We then demonstrate that by conditioning on the positions and weights of grounded instruction as input, the agent can be self-monitoring of its progress and further ensure that the textual grounding accurately reflects the progress made.

Overall, we propose a novel self-monitoring agent for VLN and make the following contributions: (1) We introduce the visual-textual co-grounding module, which performs grounding interdependently across both visual and textual modalities.

We show that it can outperform the baseline method by a large margin.

(2) We propose to equip the self-monitoring agent with a progress monitor, and for navigation tasks involving instructions instantiate this by introducing a new objective function for training.

We demonstrate that, unlike the baseline method, the position of grounded instruction can follow both past and future instructions, thereby tracking progress to the goal.

(3) With the proposed self-monitoring agent, we set the new state-of-the-art performance on both seen and unseen environments on the standard benchmark.

With 8% absolute improvement in success rate on the unseen test set, we are ranked #1 on the challenge leaderboard.

Given a natural language instruction with L words, its representation is denoted by X = x 1 , x 2 , . . .

, x L , where x l is the feature vector for the l-th word encoded by an LSTM language encoder.

Following BID19 , we enable the agent with panoramic view.

At each time step, the agent perceives a set of images at each viewpoint v t = v t,1 , v t,2 , ..., v t,K , where K Figure 2: Proposed self-monitoring agent consisting of visual-textual co-grounding, progress monitoring, and action selection modules.

Textual grounding: identify which part of the instruction has been completed or ongoing and which part is potentially needed for next action.

Visual grounding: summarize the observed surrounding images.

Progress monitor: regularize and ensure grounded instruction reflects progress towards the goal.

Action selection: identify which direction to go.is the maximum number of navigable directions 1 , and v t,k represents the image feature of direction k. The co-grounding feature of instruction and image are denoted asx t andv t respectively.

The selected action is denoted as a t .

The learnable weights are denoted with W , with appropriate sub/super-scripts as necessary.

We omit the bias term b to avoid notational clutter in the exposition.

First, we propose a visual and textual co-grounding model for the vision and language navigation task, as illustrated in FIG1 .

We model the agent with a sequence-to-sequence architecture with attention by using a recurrent neural network.

More specifically, we use Long Short Term Memory (LSTM) to carry the flow of information effectively.

At each step t, the decoder observes representations of the current attended panoramic image featurev t , previous selected action a t???1 and current grounded instruction featurex t as input, and outputs an encoder context h t : DISPLAYFORM0 where [, ] denotes concatenation.

The previous encoder context h t???1 is used to obtain the textual grounding featurex t and visual grounding featurev t , whereas we use current encoder context h t to obtain next action a t , all of which will be illustrated in the rest of the section.

Textual grounding.

When the agent moves from one viewpoint to another, it is required to identify which direction to go by relying on a grounded instruction, i.e. which parts of the instruction should be used.

This can either be the instruction matched with the past (ongoing action) or predicted for the future (next action).

To capture the relative position between words within an instruction, we incorporate the positional encoding P E(??) BID43 into the instruction features.

We then perform soft-attention on the instruction features X, as shown on the left side of FIG1 .

The attention distribution over L words of the instructions is computed as: DISPLAYFORM1 where W x are parameters to be learnt.

z textual t,l is a scalar value computed as the correlation between word l of the instruction and previous hidden state h t???1 , and ?? t is the attention weight over features in X at time t. Based on the textual attention distribution, the grounded textual featurex t can be obtained by the weighted sum over the textual featuresx t = ?? T t X.Visual grounding.

In order to locate the completed or ongoing instruction, the agent needs to keep track of the sequence of images observed along the navigation trajectory.

We thus perform visual attention over the surrounding views based on its previous hidden vector h t???1 .

The visual attention weight ?? t can be obtained as: DISPLAYFORM2 where g is a one-layer Multi-Layer Perceptron (MLP), W v are parameters to be learnt.

Similar to Eq. 2, the grounded visual featurev t can be obtained by the weighted sum over the visual feature?? v t = ?? T t V .

Action selection.

To make a decision on which direction to go, the agent finds the image features on navigable directions with the highest correlation with the grounded navigation instructionx t and the current hidden state h t .

We use the inner-product to compute the correlation, and the probability of each navigable direction is then computed as: DISPLAYFORM3 where W a are the learnt parameters, g(??) is the same MLP as in Eq. 3, and p t is the probability of each navigable direction at time t.

We use categorical sampling during training to select the next action a t .

Unlike the previous method with the panoramic view BID19 , which attends to instructions only based on the history of observed images, we achieve both textual and visual grounding using the shared hidden state output containing grounded information from both textual and visual modalities.

During action selection, we rely on both hidden state output and grounded instruction, instead of only relying on grounded instruction.

It is imperative that the textual-grounding correctly reflects the progress towards the goal, since the agent can then implicitly know where it is now and what the next instruction to be completed will be.

In the visual-textual co-grounding module, we ensure that the grounded instruction reasonably informs decision making when selecting a navigable direction.

This is necessary but not sufficient for ensuring that the notion of progress to the goal is encoded.

Thus, we propose to equip the agent with a progress monitor that serves as regularizer during training and prunes unfinished trajectories during inference.

Since the positions of localized instruction can be a strong indication of the navigation progress due to the structural alignment bias between navigation steps and instruction, the progress monitor can estimate how close the current viewpoint is to the final goal by conditioning on the positions and weights of grounded instruction.

This can further enforce the result of textual-grounding to align with the progress made towards the goal and to ensure the correctness of the textual-grounding.

The progress monitor aims to estimate the navigation progress by conditioning on three inputs: the history of grounded images and instructions, the current observation of the surrounding images, and the positions of grounded instructions.

We therefore represent these inputs by using (1) the previous hidden state h t???1 and the current cell state c t of the LSTM, (2) the grounded surrounding image?? v t , and (3) the distribution of attention weights of textual-grounding ?? t , as shown at the bottom of Our proposed progress monitor first computes an additional hidden state output h pm t by using grounded image representationsv t as input, similar to how a regular LSTM computes hidden states except we use concatenation over element-wise addition for empirical reasons 2 .

The hidden state output is then concatenated with the attention weights ?? t on textual-grounding to estimate how close the agent is to the goal 3 .

The output of the progress monitor p pm t , which represents the completeness of instruction-following, is computed as: DISPLAYFORM0 where W h and W pm are the learnt parameters, c t is the cell state of the LSTM, ??? denotes the element-wise product, and ?? is the sigmoid function.

Training.

We introduce a new objective function to train the proposed progress monitor.

The training target y pm t is defined as the normalized distance in units of length from the current viewpoint to the goal, i.e., the target will be 0 at the beginning and closer to 1 as the agent approaches the goal 4 .

Note that the target can also be lower than 0, if the agent's current distance from the goal is farther than the starting point.

Finally, our self-monitoring agent is optimized with a cross-entropy loss and a mean squared error loss, computed with respect to the outputs from both action selection and progress monitor.

DISPLAYFORM0 where p k,t is the action probability of each navigable direction, ?? = 0.5 is the weight balancing the two losses, and y nv t is the ground-truth navigable direction at step t. Inference.

During inference, we follow BID19 by using beam search.

we propose that, while the agent decides which trajectories in the beams to keep, it is equally important to evaluate the state of the beams on actions as well as on the agent's confidence in completing the given instruction at each traversed viewpoint.

We accomplish this idea by integrating the output of our progress monitor into the accumulated probability of beam search.

At each step, when candidate trajectories compete based on accumulated probability, we integrate the estimated completeness of instruction-following p pm t (normalized between 0 to 1) with action probability p k,t to directly evaluate the partial and unfinished candidate routes: DISPLAYFORM1 Without beam search, we use greedy decoding for action selection with one condition.

If the progress monitor output decreases (p pm t+1 < p pm t ), the agent is required to move back to the previous viewpoint and select the action with next highest probability.

We repeat this process until the selected action leads to increasing progress monitor output.

We denote this procedure as progress inference.

R2R Dataset.

We use the Room-to-Room (R2R) dataset BID3 for evaluating our proposed approach.

The R2R dataset is built upon the Matterport3D dataset BID11 and has 7,189 paths sampled from its navigation graphs.

Each path has three ground-truth navigation instructions written by humans.

The whole dataset is divided into 4 sets: training, validation seen, validation unseen, and test sets unseen.

Evaluation metrics.

We follow the same evaluation metrics used by previous work on the R2R task: (1) Navigation Error (NE), mean of the shortest path distance in meters between the agent's final 4 We set the target to 1 if the agent's distance to the goal is less than 3.

BID3 , RPA BID46 , and Speaker-Follower BID19 .

*: with data augmentation.

leaderboard: when using beam search, we modify our search procedure to comply with the leaderboard guidelines, i.e., all traversed viewpoints are recorded.

Figure 3 : The positions and weights of grounded instructions as agents navigate by following instructions.

Our self-monitoring agent with progress monitor demonstrates the grounded instruction used for action selection shifts gradually from the beginning of instructions towards the end.

This is not true of the baseline method.

DISPLAYFORM0 position and the goal location.

(2) Success Rate (SR), the percentage of final positions less than 3m away from the goal location.

(3) Oracle Success Rate (OSR), the success rate if the agent can stop at the closest point to the goal along its trajectory.

In addition, we also include the recently introduced Success rate weighted by (normalized inverse) Path Length (SPL) BID2 , which trades-off Success Rate against trajectory length.

We first compare the proposed self-monitoring agent with existing approaches.

As shown in TAB1 , our method achieves significant performance improvement compared to the state of the arts without data augmentation.

We achieve 70% SR on the seen environment and 57% on the unseen environment while the existing best performing method achieved 63% and 50% SR respectively.

When trained with synthetic data 5 , our approach achieves slightly better performance on the seen environments and significantly better performance on both the validation unseen environments and the test unseen environments when submitted to the test server.

We achieve 3% and 8% improvement on SR on both validation and test unseen environments.

Both results with or without data augmentation indicate that our proposed approach is more generalizable to unseen environments.

At the time of writing, our self-monitoring agent is ranked #1 on the challenge leader-board among the state of the arts.

Note that both Speaker-Follower and our approach in TAB1 use beam search.

For comparison without using beam search, please refer to the Appendix.

Textually grounded agent.

Intuitively, an instruction-following agent is required to strongly demonstrate the ability to correctly focus and follow the corresponding part of the instruction as it navigates through an environment.

We thus record the distribution of attention weights on instruction at each step as indications of which parts of the instruction being used for action selection.

We average all runs across both validation seen and unseen dataset splits.

Ideally, we expect to see the distribution of attention weights lies close to a diagonal, where at the beginning, the agent focuses on the beginning of the instruction and shifts its attention towards the end of instruction as it moves closer to the goal.

To demonstrate, we use the method with panoramic action space proposed in BID19 as a baseline for comparison.

As shown in Figure 3 , our self-monitoring agent with progress monitor demonstrates that the positions of grounded instruction over time form a line similar to a diagonal.

This result may further indicate that the agent successfully utilizes the attention on instruction to complete the task sequentially.

We can also see that both agents were able to focus on the first part of the instruction at the beginning of navigation consistently.

However, as the agent moves further in unknown environments, our self-monitoring agent can still successfully identify the parts of instruction that are potentially useful for action selection, whereas the baseline approach becomes uncertain about which part of the instruction should be used for selecting an action.

We now discuss the importance of each component proposed in this work.

We begin with the same baseline as before (agent with panoramic action space in Fried et al. FORMULA0 ) 6 .Co-grounding.

When comparing the baseline with row #1 in our proposed method, we can see that our co-grounding agent outperformed the baseline with a large margin.

This is due to the fact that we use the LSTM to carry both the textually and visually grounded content, and the decision on each navigable direction is predicted with both textually grounded instruction and the hidden state output of the LSTM.

On the other hand, the baseline agent relies on the LSTM to carry visually grounded content, and uses the hidden state output for predicting the textually grounded instruction.

As a result, we observed that instead of predicting the instruction needed for selecting a navigable direction, the textually grounded instruction may match with the past sequence of observed images implicitly saved within the LSTM.Progress monitor.

Given the effective co-grounding, the proposed progress monitor further ensure that the grounded instruction correctly reflects the progress made toward the goal.

This further improves the performance especially on the unseen environments as we can see from row #1 and #2.When using the progress inference, the progress monitor serve as a progress indicator for the agent to decide when to move back to the last viewpoint.

We can see from row #2 and #4 that the SR performance can be further improved around 2% on both seen and unseen environments.

Finally, we integrate the output of the progress monitor with the state-factored beam search BID19 , so that the candidate paths compete not only based on the probability of selecting a certain navigable direction but also on the estimated correspondence between the past trajectory and the instruction.

As we can see by comparing row #2, #6, and #7, the progress monitor significantly improved the success rate on both seen and unseen environments and is the key for surpassing the state of the arts even without data augmentation.

We can also see that when using beam search without progress monitor, the SR on unseen improved 7% (row #1 vs #6), while using beam search integrated with progress estimation improved 13% (row #2 vs #7).Data augmentation.

In the above, we have shown each row in our approach contributes to the performance.

Each of them increases the success rate and reduces the navigation error incrementally.

By further combining them with the data augmentation pre-trained from the speaker BID19 , the SR and OSR are further increased, and the NE is also drastically reduced.

Interestingly, the performance improvement introduced by data augmentation is smaller than from Speaker-Follower on the validation sets (see TAB1 for comparison).

This demonstrates that our proposed method is more data-efficient.

Figure 4: Successful self-monitoring agent navigates in two unseen environments.

The agent is able to correctly follow the grounded instruction and achieve the goal successfully.

The percentage of instruction completeness estimated by the proposed progress monitor gradually increases as the agent navigates and approaches the goal.

Finally, the agent grounded the word "Stop" to stop (see the supplementary material for full figures).

To further validate the proposed method, we qualitatively show how the agent navigates through unseen environments by following instructions as shown in Fig. 4 .

In each figure, the agent follows the grounded instruction (at the top of the figure) and decides to move towards a certain direction (green arrow).

For the full figures and more examples of successful and failed agents in both unseen and seen environments, please see the supplementary material.

Consider the trajectory on the left side in Fig. 4 , at step 3, the grounded instruction illustrated that the agent just completed "turn right" and focuses mainly on "walk straight to bedroom".

As the agent entered the bedroom, it then shifts the textual grounding to the next action "Turn left and walk to bed lamp".

Finally, at step 6, the agent completed another "turn left" and successfully stop at the rug (see the supplementary material for the importance of dealing with duplicate actions).

Consider the example on the right side, the agent has already entered the hallway and now turns right to walk across to another room.

However, it is ambiguous that which room the instructor is referring to.

At step 5, our agent checked out the room on the left first and realized that it does not match with "Stop in doorway in front of rug".

It then moves to the next room and successfully stops at the goal.

In both cases, we can see that the completeness estimated by progress monitor gradually increases as the agent steadily navigates toward the goal.

We have also observed that the estimated completeness ends up much lower for failure cases (see the supplementary material for further details).

Vision, Language, and Navigation..

There is a plethora work investigating the combination of vision and language for a multitude of applications BID51 b; BID5 BID41 BID14 , etc.

While success has been achieved in these tasks to handle massive corpora of static visual input and text data, a resurgence of interest focuses on equipping an agent with the ability to interact with its surrounding environment for a particular goal such as object manipulation with instructions BID36 BID6 , grounded language acquisition BID1 BID25 BID40 BID17 , embodied question answering BID15 BID21 , and navigation BID31 BID22 BID18 BID53 BID16 BID53 BID37 BID47 BID45 BID33 BID50 .

In this work, we concentrate on the recently proposed the Visionand-Language Navigation task BID3 -asking an agent to carry out sophisticated natural-language instructions in a 3D environment.

This task has application to fields such as robotics; in contrast to traditional map-based navigation systems, navigation with instructions provides a flexible way to generalize across different environments.

A few approaches have been proposed for the VLN task.

For example, BID3 address the task in the form of a sequence-to-sequence translation model.

BID48 introduce a guided feature transformation for textual grounding.

BID46 present a planned-head module by combing model-free and model-based reinforcement learning approaches.

Recently, BID19 propose to train a speaker to synthesize new instructions for data augmentation and further use it for pragmatic inference to rank the candidate routes.

These approaches leverage attentional mechanisms to select related words from a given instruction when choosing an action, but those agents are deployed to explore the environment without knowing about what progress has been made and how far away the goal is.

In this paper, we propose a self-monitoring agent that performs co-grounding on both visual and textual inputs and constantly monitors its own progress toward the goal as a way of regularizing the textual grounding.

Visual and textual grounding.

Visual grounding learns to localize the most relevant object or region in an image given linguistic descriptions, and has been demonstrated as an essential component for a variety of vision tasks like image captioning BID28 , visual question answering BID27 BID0 , relationship detection BID26 BID29 and referral expression BID38 BID20 .

In contrast to identifying regions or objects, we perform visual grounding to locate relevant images (views) in a panoramic photo constructed by stitching multiple images with the aim of choosing which direction to go.

Extensive efforts have been made to ground language instructions into a sequence of actions BID30 BID10 BID44 BID42 BID7 BID4 BID32 BID13 BID35 .

These early approaches mainly emphasize the incorporation of structural alignment biases between the linguistic structure and sequence of actions BID32 BID4 , and assume the agents are in relatively easy environment where limited visual perception is required to fulfill the instructions.

We introduce a self-monitoring agent which consists of two complementary modules: visual-textual co-grounding module and progress monitor.

The visual-textual co-grounding module locates the instruction completed in the past, the instruction needed in the next action, and the moving direction from surrounding images.

The progress monitor regularizes and ensures the grounded instruction correctly reflects the progress towards the goal by explicitly estimating the completeness of instruction-following.

This estimation is conditioned on the positions and weights of grounded instruction.

Our approach sets a new state-of-the-art performance on the standard Room-to-Room dataset on both seen and unseen environments.

While we present one instantiation of self-monitoring for a decision-making agent, we believe that this concept can be applied to other domains as well.

BID46 , and Speaker-Follower BID19 .

*: with data augmentation.

TAB4 .

We can see that our proposed method outperformed existing approaches with a large margin on both validation unseen and test sets.

Our method with greedy decoding for action selection improved the SR by 9% and 8% on validation unseen and test set.

When using progress inference for action selection, the performance on the test set significantly improved by 5% compared to using greedy decoding, yielding 13% improvement over the best existing approach.

Network architecture.

The embedding dimension for encoding the navigation instruction is 256.

We use a dropout layer with ratio 0.5 after the embedding layer.

We then encode the instruction using a regular LSTM, and the hidden state is 512 dimensional.

The MLP g used for projecting the raw image feature is BN ??? ??? F C ??? ??? BN ??? ??? Dropout ??? ??? ReLU .

The FC layer projects the 2176-d input vector to a 1024-d vector, and the dropout ratio is set to be 0.5.

The hidden state of the LSTM used for carrying the textual and visual information through time in Eq. 1 is 512.

We set the maximum length of instruction to be 80, thus the dimension of the attention weights of textual grounding ?? t is also 80.

The dimension of the learnable matrices from Eq. 2 to 5 are: DISPLAYFORM0 DISPLAYFORM1 closest previous trajectory, so that when a single agent traverses through all recorded trajectories, the overhead for switching from one trajectory to another can be reduced significantly.

The final selected trajectory from beam search is then lastly logged to the trajectory.

This therefore yields exactly the same success rate and navigation error, as the metrics are computed according to the last viewpoint from a trajectory.

We provide and discuss additional qualitative results on the self-monitoring agent navigating on seen and unseen environments.

We first discuss four successful examples in FIG3 and 6, and followed by two failure examples in FIG5 .

In FIG3 , at the beginning, the agent mostly focuses on "walk up" for making the first movement.

While the agent keeps its attention on "walk up" as completed instruction or ongoing action, it shifts the attention on instruction to "turn right" as it walks up the stairs.

Once it reached the top of the stairs, it decides to turn right according to the grounded instruction.

Once turned right, we can again see that the agent pays attention on both the past action "turn right" and next action "walk straight to bedroom".

The agent continues to do so until it decides to stop by grounding on the word "stop".In FIG3 , the agent starts by focusing on both "enter bedroom from balcony" and "turn left" to navigate.

It correctly shifts the attention on textual grounding on the following instruction.

Interestingly, the given instruction "walk straight across rug to room" at step 3 is ambiguous since there are two rooms across the rug.

Our agent decided to sneak out of the first room on the left and noticed that it does not match with the description from instruction.

It then moved to another room across the rug and decided to stop because there is a rug inside the room as described.

In FIG4 , the given instruction is ambiguous as it only asks the agent to take actions around the stairs.

Since there are multiple duplicated actions described in the instruction, e.g. "walk up" and "turn left", only an agent that is able to precisely follow the instruction step-by-step can successfully complete the task.

Otherwise, the agent is likely to stop early before it reaches the goal.

The agent also needs to demonstrate its ability to assess the completeness of instruction-following task in order to correctly stop at the right amount of repeated actions as described in the instruction.

In FIG4 , at the beginning (step 0), the agent only focuses on 'left' for making the first movement (the agent is originally facing the painting).

We can see that at each step, the agent correctly focuses on parts of the instruction for making every movements, and it finally believes that the instruction is completed (attention on the last sentence period) and stopped.

In FIG5 (a) step 1, although the attention on instruction correctly focused on "take a left" and "go down", the agent failed to follow the instruction and was not able to complete the task.

We can however see that the progress monitor correctly reflected that the agent did not follow the given instruction successfully.

The agent ended up stopping with progress monitor reporting that only 16% of the instruction was completed.

In FIG5 (b) step 2, the attention on instruction only focuses on "go down" and thus failed to associate the "go down steps" with the stairs previously mentioned in "turn right to stairs".

The agent was however able to follow the rest of the instruction correctly by turning right and stopping near a mirror.

Note that, different from FIG5 , the final estimated completeness of instruction-following from progress monitor is much higher (16%), which indicates that the agent failed to be aware that it was not correctly following the instruction.

The given instruction is ambiguous as it only asks the agent to take actions around the stairs.

Since there are multiple duplicated actions described in the instruction, e.g. "walk up" and "turn left", only an agent that is able to precisely follow the instruction step-by-step can successfully complete the task.

Otherwise, the agent is likely to stop early before it reaches the goal.

(b) The agent correctly pays attention to parts of the instruction for making decisions on selecting navigable directions.

Both the agents decide to stop when shifting the textual grounding on the last sentence period.

The agent missed the "take a left" at step 1, and consequently unable to follow the following instruction correctly.

However, note that the progress monitor correctly reflected that the instruction was not completed.

When the agent decides to end the navigation, it reports that only 16% of the instruction was completed.(b) At step 2, the attention on instruction only focuses on "go down" and thus failed to associate the "go down steps" with the stairs previously mentioned in "turn right to stairs".

The agent was however able to follow the rest of the instruction correctly by turning right and stopping near a mirror.

Note that, different from (a), the final estimated completeness of instruction-following is much higher, which suggests that the agent failed to correctly be aware of its progress towards the goal.

@highlight

We propose a self-monitoring agent for the Vision-and-Language Navigation task.

@highlight

A method for vision+language navigation which tracks progress on the instruction using a progress monitor and a visual-textual co-grounding module, and performs well on standard benchmarks.

@highlight

This paper describes a model for vision-and-language navigation with a panoramic visual attention and an auxillary progress monitoring loss, giving state-of-the-art results.