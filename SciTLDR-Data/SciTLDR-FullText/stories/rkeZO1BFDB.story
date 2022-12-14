Vision-Language Navigation (VLN) is the task where an agent is commanded to navigate in photo-realistic unknown environments with natural language instructions.

Previous research on VLN is primarily conducted on the Room-to-Room (R2R) dataset with only English instructions.

The ultimate goal of VLN, however, is to serve people speaking arbitrary languages.

Towards multilingual VLN with numerous languages, we collect a cross-lingual R2R dataset, which extends the original benchmark with corresponding Chinese instructions.

But it is time-consuming and expensive to collect large-scale human instructions for every existing language.

Based on the newly introduced dataset, we propose a general cross-lingual VLN framework to enable instruction-following navigation for different languages.

We first explore the possibility of building a cross-lingual agent when no training data of the target language is available.

The cross-lingual agent is equipped with a meta-learner to aggregate cross-lingual representations and a visually grounded cross-lingual alignment module to align textual representations of different languages.

Under the zero-shot learning scenario, our model shows competitive results even compared to a model trained with all target language instructions.

In addition, we introduce an adversarial domain adaption loss to improve the transferring ability of our model when given a certain amount of target language data.

Our methods and dataset demonstrate the potentials of building a cross-lingual agent to serve speakers with different languages.

Recently, the Vision-Language Navigation (VLN) task (Anderson et al., 2018) , which requires the agent to follow natural language instructions and navigate in houses, has drawn significant attention.

In contrast to some existing navigation tasks (Mirowski et al., 2016; Zhu et al., 2017) , where the agent has an explicit representation of the target to know if the goal has been reached or not, an agent in the VLN task can only infer the target from natural language instructions.

Therefore, in addition to normal visual challenges in navigation tasks, language understanding and cross-modal alignment are essential to complete the VLN task.

However, existing benchmarks (Anderson et al., 2018; Chen et al., 2019) for the VLN task are all monolingual in that they only contain English instructions.

Specifically, the navigation agents are trained and tested with only English corpus and thus unable to serve non-English speakers.

To fill this gap, one can collect the corresponding instructions in the language that the agent is expected to execute.

But it is not scalable and practical as there are thousands of languages on this planet and collecting large-scale data for each language would be very expensive and time-consuming.

Therefore, in this paper, we study the task of cross-lingual VLN to endow an agent the ability to understand multiple languages.

First, can we learn a model that has been trained on existing English instructions but is still able to perform reasonably well on a different language (e.g. Chinese)?

This is indeed a zero-shot learning scenario where no training data of target language is available.

An intuitive approach is to train the agent with English data, and at test time, use a machine translation system to translate the target language instructions to English, which are then fed into the agent for testing (see the above part of Figure 1 ).

The inverse solution is also rational: we can translate all English instructions into the target language and train the agent on the translated data, so it can be directly tested with target language instructions (see the below part of Figure 1) .

The former agent is tested on translated instructions while the latter is trained on translated instructions.

Both solutions suffer from translation errors and deviation from the corresponding human-annotated instructions.

But meanwhile, the former is trained on human-annotated English instructions (which we view as "golden" data) and the latter is tested on "golden" target language instructions.

Motivated by this fact, we design a cross-lingual VLN framework that learns to benefit from both solutions.

As shown in Figure 1 , we combine these two principles and introduce a meta-learner, which learns to produce beliefs for human-annotated instruction and its translation pair and dynamically fuse the cross-lingual representations for better navigation.

In this case, however, the training and inference are mismatched.

During training, the agent takes source human language and target machine translation (MT) data as input, while during inference, it needs to navigate with target human instructions and source MT data.

To better align the source and target languages, we propose a visually grounded cross-lingual alignment module to align the paired instructions via the same visual feature because they are describing the same demonstration path.

The cross-lingual alignment loss can also implicitly alleviate the translation errors by aligning the human language and its MT pair in the latent visual space.

After obtaining an efficient zero-shot agent, we investigate the question that, given a certain amount of data for the target language, can we learn a better adaptation model to improve source-to-target knowledge transfer?

The meta-learner and visually grounded cross-lingual alignment module provide a foundation for solving the circumstances that the agent has access to the source language and (partial) target language instructions for training.

To further leverage the fact that the agent has access to the target language training data, we introduce an adversarial domain adaption loss to alleviate the domain shifting issue between human-annotated and MT data, thus enhancing the model's transferring ability.

To validate our methods, we collect a cross-lingual VLN dataset (XL-R2R) by extending complimentary Chinese instructions for the English instructions in the R2R dataset.

Overall, our contributions are four-fold: (1) We collect the first cross-lingual VLN dataset to facilitate navigation models towards accomplishing instructions of various languages such as English and Chinese, and conduct analysis between English and Chinese corpus.

(2) We introduce the task of cross-lingual visionlanguage navigation and propose a principled meta-learning method that dynamically utilizes the augmented MT data for zero-shot cross-lingual VLN.

(3) We propose a visually grounded crosslingual alignment module for better cross-lingual knowledge transfer.

(4) We investigate how to transfer knowledge between human-annotated and MT data and introduce an adversarial domain adaption loss to improve the navigation performance given a certain amount of human-annotated target language data.

The cross-lingual vision-language navigation task is defined as follows: we consider an embodied agent that learns to follow natural language instructions and navigate from a starting pose to a goal location in photo-realistic 3D indoor environments.

Formally, given an environment E, an initial pose p 1 = (v 1 , ?? 1 , ?? 1 ) (spatial position, heading, elevation angles) and natural language instructions x 1:N , the agent takes a sequence of actions a 1:T to finally reach the goal G. Thus the VLN dataset D is defined as {(E, p 1 , x 1:N , G)} |D | .

Note that we eliminate the footscript here for simplicity.

At each time step t, the agent at pose p t receives a new observation I t = E(p t ), which is a raw RGB image pictured by the mounted camera.

Then it takes an action a t and leads to a new pose p t+1 = (v t+1 , ?? t+1 , ?? t+1 ).

Taking actions sequentially, the agent stops when a stop action is taken.

A cross-lingual VLN agent learns to understand multiple languages and navigate to the goal.

Without loss of generality, we consider a bilingual situation coined as cross-lingual VLN.

For this specific task, we built the XL-VLN dataset D, which extends the VLN dataset and includes a bilingual version of instructions.

Specifically,

where S and T indicate source and target language domains separately.

The source language domain S contains instructions in the source language covering the full VLN dataset D (including training and testing splits), while the target language domain T consists of a fully annotated testing set and a training set in the target language that covers a varying percentage of trajectories of the training set in D ( may range from 0% to 100%).

The agent is allowed to leverage both source and target language training sets and expected to perform navigation given an instruction from either the source or target language testing sets.

In this study, we first focus on a more challenging setting where no human-annotated target language data are available for training ( = 0%), i.e., with no training data for the target language but the only access to the source language training set, the agent is required to follow a target language instruction x T 1:N to navigate in houses.

Then we investigate the agent's transferring ability by gradually increasing the percentage of human-annotated target language instructions for training ( = 0%, 10%, ..., 100%).

We build a Cross-Lingual Room-to-Room (XL-R2R) dataset 1 , the first cross-lingual dataset for the vision-language navigation task.

The XL-R2R dataset includes 4,675/340/783 trajectories for train/validation seen/validation unseen sets, preserving the same split as in the R2R dataset.

The official testing set of R2R is unavailable because the testing trajectories are held for challenge use.

Each trajectory is described with 3 English and 3 Chinese instructions independently annotated by different workers.

Data Collection.

We keep the English instructions of the R2R dataset and collect Chinese instructions via a public Chinese crowdsourcing platform.

The Chinese instructions are annotated by native speakers through an interactive 3D WebGL environment, following the R2R dataset guidance (Anderson et al., 2018) .

More details can be found in the Appendix.

Data Analysis.

XL-R2R dataset includes 5,798 trajectories in total and 17,394 instructions for both languages.

The bilingual instruction part here is compared with each other from four perspectives for a broad understanding, including vocabulary, instruction length, sub-instruction number per instruction and part-of-speech tags.

Removing words with less than 5 frequency, we obtain an English vocabulary with 1,583 words and a Chinese one with 1,134 words.

The Chinese instructions are relatively shorter than English ones and less likely to be long sentences ( Figure 2a ).

The instructions usually consist of several sub-instructions separated by punctuation tokens, and the number of sub-instructions per instruction distributes similar across language ( Figure 2b ).

Figure 2c and Figure 2d show that nouns and verbs, which often refer to landmarks and actions respectively, are more frequent in Chinese dataset (32.9% and 29.0%) than in English one (24.3% and 13.7%) 2 .

We present a general cross-lingual VLN framework in Figure 3 .

It is based on an encoder-decoder architecture and composed of three novel modules: a cross-lingual meta-learner (Meta), a visually grounded cross-lingual alignment module (txt2img), and an adversarial domain adaptation module.

Particularly, as shown in Figure 3 , both English and Chinese instructions are encoded by a shared encoder.

Then the shared decoder takes the encoded contextual embeddings c In addition, the txt2img module is introduced to align h en t and h zh t in the visual space with the visual feature I t as an anchor point, improving the cross-lingual knowledge transfer via a visually grounded cross-lingual alignment loss.

The adversarial domain adaptation module is particularly designed for the transfer setting where human-annotated target language instructions are also provided.

We employ a domain discriminator that is trying to distinguish human-annotated instructions from machine translation (MT) instructions, and a gradient reversal layer to reverse the gradients back-propagated to the encoder so that the encoder is indeed trying to generate indistinguishable representations for both human-annotated and MT data and align the distributions of the two domains.

We employ the sequence-to-sequence architecture in Anderson et al. (2018) for both languages.

Receiving a pair of natural language instruction x L 1:N , L ??? {S, T }, the agent encodes it with an embedding matrix followed by an LSTM encoder to obtain contextual word representations c L 1:N .

The decoder LSTM takes the concatenation of current image feature I t and previous action embedding a t???1 as input, and updates the hidden state s t???1 to s t aware of the historical trajectory:

An attention mechanism is used to compute a weighted context representation, grounded on the instruction c

To bridge the gap between source and target languages, we leverage a machine translation (MT) system to translate the source language in the training data into the target language.

During testing, the MT system will translate the target language instruction into the source language.

The MT data serves as augmented data for zero-shot or low-resource settings as well as associates two different human languages in general.

We take two instructions (the human language instruction and its MT pair) as input for both training and testing.

We observed that, even if one instruction is a direct translation from the other, when the paired instructions are fed into the same encoder and decoder, the two instructions will often generate different predictions when executing.

At each time step, when the agent observed the local visual environment, with two instructions at hand but lead to different next positions.

It remains a challenging question which language representation the agent shall trust more.

Therefore, we propose a cross-lingual meta-learner that tries to help the agent make the judgment.

At each time step, we let the cross-lingual meta-learner decide which language representation we should have more faith in, i.e., "learning to trust".

The meta-learner is a SoftMax layer which takes the concatenation of two hidden states h S t and h T t as input, and produces a probability ?? t representing the belief of the source language representation.

The final hidden vector used for predicting actions is defined as a mixture of the representations in two languages:

Finally, the predicted action distribution for the next time step is computed as:

To better ground and align two languages to the images they describe, we map h S t and h T t into the latent space of image representations such that their similarity is maximized.

In other words, we use the image space as an anchor point to align cross-lingual representations.

Let I t be the latent representation of the local visual environment on the target trajectory at time step t (e.g. the final layer of a ResNet), the loss function is formulated as:

where ?? denotes a non-linearity activation such as ReLU or tanh, W I , W S and W T are the projection matrices.

L2 distance is used to measure the similarity between contextual word and image features in the same vector space.

The intuition behind such an aligning mechanism is that, since the human instruction and the MT instruction are both describing the same trajectory, their representation should be close to the visual environment in some way (a projected latent space in our case).

This module would ensure the consistency among cross-lingual representations and the visual inputs.

During training, we have human-annotated data for the source language and the machine-translated data for the target language, while an opposite situation for testing.

To bridge the gap between training and inference, we leverage an adversarial domain adaption loss to make the context representations indistinguishable across domains for the transfer setting, where a certain amount of human-annotated instructions for the target language is available.

A sentence vector c is computed as the mean of the context vector c 1:N to a single vector, then forwarded to a domain discriminator through the gradient reversal layer.

With the gradient reversal layer, the gradients minimizing domain classification errors are passed back with opposed sign to the language encoder, which adversarially encourages the encoder to be domain-agnostic.

We then minimize the following domain adaptation loss:

where y H is the domain label of an instruction indicating whether it is from human annotation or machine translation, and?? H is the approximation by the domain discriminator.

Figure 1 .

meta+txt2img equips the meta-learner with txt2img module.

The first three models are all for zero-shot learning.

The last one, train w/ AN is to train the agent with 100% Chinese human annotated data.

All models are tested with Chinese human instructions. (Jain et al., 2019) , measures the fidelity to the described path, unlike previous metrics which are mostly based on goal completion.

Model PL NE ??? SR ??? SPL ??? CLS ??? PL NE ??? SR ??? SPL ??? CLS ??? train

We first report results for the zero-shot setting, to show the effectiveness of our two components, meta-learner, and txt2img.

We compare with two models, a seq2seq model trained with humanannotated Chinese instructions (collected in XL-R2R dataset), and that trained with MT Chinese instructions translated from English.

Results are shown in Table 1 .

First, there is a clear gap between training with human-annotated and MT data, indicating the insufficiency of using only an MT system for zero-shot learning.

Second, our meta-learner can successfully aggregate the information of the annotated data and MT data, which enables efficient zero-shot learning.

Third, the txt2img module further improves vision-language alignment, which especially helps the agent generalize better on the unseen data.

Besides, even though the agent does not have access to the target annotation data, it achieves competitive results compared to training with 100% annotated data.

To investigate the potential of transferring knowledge from English to Chinese, we draw learning curves by utilizing varying percentages of Chinese annotations for training (see Figure 4) .

The starting point is our zero-shot setting, where one has no access to the human-annotated data for the target language, and the endpoint is when one has 100% training data of the target language.

The figures demonstrate that the proposed adversarial domain adaption module provides consistent improvement over other methods, for both seen and unseen environments.

The approach works for both low-resource and high-resource settings and is capable of transferring knowledge steadily as the size of target data grows.

Besides, our transfer method trained with 40% Chinese human data can achieve similar performance as trained with 100% Chinese human data.

This demonstrates the potential of building a functioning cross-lingual VLN agent by collecting a large-scale dataset for a certain language (i.e., English), and a small amount of data for other languages.

One can also observe that pretraining with English data and MT Chinese data help the model learn useful encoding that is especially valuable when only limited Chinese training data are available.

To enable cross-lingual VLN, we examine four models (see Figure 5 ) equipped with our metalearner and txt2img modules: (1) Base-Bi, which has two separate encoder-decoder.

(2) Shared Enc, which has a shared language encoder.

(3) Shared Dec, which has a shared policy decoder.

(4) Share Enc-Dec, which shares both the encoder and the decoder, with different word embeddings for different languages.

These models take English and Chinese natural language instructions as input, for both training and testing.

They are also compared with Base-mono, which is a single encoder-decoder model trained and tested with Chinese human instructions only.

Table 2 : Performance comparison for cross-lingual VLN models.

All models are trained and tested with English and Chinese annotation data.

Results are averaged over 3 runs.

Table 2 shows the results of four architectures on the validation seen and unseen part.

First, the performance of multi-lingual models are consistently improved over the monolingual model (Basemono), indicating the potential of cross-lingual learning for improving navigation results.

Second, sharing parameters can further boost navigation performance.

Finally, Shared Enc and Shared EncDec produce similar results, which motivates us to use a Shared Enc-Dec design since it yields a competitive good result with fewer parameters required.

For a more intuitive understanding of the meta-learner, we visualize the confidences assigned to each language in Figure 6 .

In this case, the meta-learner trusts more on the human-annotated Chinese instruction which is of better quality.

More specifically, at time step 10, when the meta-learner has the highest faith in the Chinese instruction, we visualize the textual attention on the whole instruction at this time step.

Evidently, the corresponding textual attention on the Chinese command makes more sense than the machine-translated English command.

The agent is supposed to keep turning left and then move forward to the green plant.

The attentions on Chinese instruction assigns 0.25 to "turn left", and nearly zero attention to "head towards the door" which is already completed by previous actions.

While the attention on English is more uniform and less accurate than on Chinese.

Textual attention at time step 10: Turn left Go to Figure 6 : Case Study.

We choose a succeeded instruction from the validation set for illustration.

Vision and Language Grounding.

Over the past years, deep learning approaches have boosted the performance of computer vision and natural language processing tasks (Krizhevsky et al., 2012; Sutskever et al., 2014; He et al., 2016; Vaswani et al., 2017) .

A large body of benchmarks are proposed to facilitate the research, including Image and Video caption (Lin et al., 2014; Krishna et al., 2017) , VQA (Antol et al., 2015; Das et al., 2018) , and visual dialog (Das et al., 2017) .

These tasks require grounding on both visual and textual modalities, but mostly limited to a fixed visual input.

Thus, we focus on the task of vision-language navigation (VLN) (Anderson et al., 2018) , where an agent needs to actively interact with the visual environment following language instructions.

Vision-Language Navigation.

Several approaches have been proposed for the VLN task on the R2R dataset.

For example, Wang et al. (2018) presented a planned-ahead module combining modelfree and model-based reinforcement learning methods, Fried et al. (2018) introduced a speaker which can synthesize new instructions and implement pragmatic reasoning.

Subsequent methods extend the speaker-follower model with Reinforced Cross-modal Matching (Wang et al., 2019a) , self-monitoring (Ma et al., 2019 ), back-translation (Tan et al., 2019 etc.

Previous works mainly improve navigation performance by data augmentation or leveraging efficient searching methods.

In this paper, we address the task from a cross-lingual perspective, aiming at building an agent to execute instructions for different languages.

Cross-lingual Language Understanding.

Learning cross-lingual representations is a crucial step to make natural language tasks scalable to all the world's languages.

Recently, cross-lingual studies on typical NLP tasks has achieved success, such as Part-of-Speech tagging Kim et al., 2017) , sentiment classification (Zhou et al., 2016) and Named Entity Recognition (Pan et al., 2017; Ni et al., 2017) These studies successfully disentangle the linguistic knowledge into languagecommon and language-specific parts and learn both knowledges with individual modules.

Moreover, cross-lingual image and video captioning (Miyazaki & Shimizu, 2016; Wang et al., 2019b) aim to bridge vision and language towards a deeper understanding, by learning a cross-lingual model grounded on visual inputs.

Our dataset and method address the cross-lingual representation learning for the vision-language navigation task.

To our knowledge, we are the first to study the cross-lingual learning in a dynamic visual environment, where the agent needs to interact with its surroundings and take a sequence of actions.

In this paper, we introduce a new task, namely cross-lingual vision-language navigation, to study cross-lingual representation learning situated in the navigation task where cross-modal interaction with the real world is involved.

We collect a cross-lingual R2R dataset and conduct pivot studies towards solving this challenging but practical task.

The proposed cross-lingual VLN framework is proven effective in cross-lingual knowledge transfer.

There are still lots of promising future directions for this task and dataset, e.g. to incorporate recent advances in VLN and greatly improve the model capacity.

It would also be valuable to extend the cross-lingual setting to support numerous different languages in addition to English and Chinese.

We follow the same preprocessing procedure as in previous work.

A ResNet-152 pretrained on ImageNet is used to extract image features, which are 2, 048-d vectors.

Instructions are clipped with a maximum length of 80.

Words are embedded into a 256-d vector space, and the action embedding is 32.

The hidden size for the encoder and decoder LSTM is 512.

The dropout ratio is 0.5.

The meta-learner is a single fully connected layer.

The dimension of vision-language alignment vector space is set to 1, 024.

Each episode consists of no more that 40 actions.

The network is optimized via the ADAM optimizer with an initial learning rate of 0.001, a weight decay of 0.0005, and a batch size of 100.

The learning rate of domain adaptation loss is scheduled with an adaption factor (Ganin & Lempitsky, 2014) :

where ?? is set to 10 and p is learning steps.

We use 0.2?? p to train the domain discriminator.

We run each model 30, 000 iterations and report the iteration with the highest SPL, and evaluate the models every 500 iterations.

For evaluating our proposed approach on the unseen test set, we participate in the Vision and Language Navigation challenge and submitted our results to the test server.

Here we treat Chinese as the source language and English as the target language.

Hence for zeroshot learning, the agent has 100% Chinese annotated data but no English annotated data.

The agent is commanded to follow English human instructions during testing.

Results are shown in Table 3 .

For zero-shot learning, our method (meta+txt2img) improves over the model trained with MT data only.

For transfer learning, our method can efficiently transfer knowledge between Chinese and English data.

The results are coherent with the reported results on the validation set. (See Table 1 and Figure 4) .

Table 3 : Performance comparison on the English test set.

The first two rows are for zero-shot learning, the last two rows are trained with access to 100% target training data (i.e. English annotated instructions).

Meta-learner To validate the effectiveness of the meta-learner, we compare it with a simple ensemble, which assigns equal confidence to two languages at all time steps, without any learnable parameters.

The results are summarized in Table 4 .

Our meta-leaner has higher performance on the validation unseen set, suggests that "learning to trust" is important for cross-lingual vision-language navigation.

Adversarial Domain Adaptation Loss To demonstrate the domain adaptation loss indeed enhance knowledge transfer between two languages, we compare it with our vanilla zero-shot model, a metalearner equipped with a txt2img module.

Table 5 shows that, as the size of target training data grows, although the vanilla model can also benefit from the augmented data, the performance stops growing as the data size reaches 40% or 60%.

Meanwhile, domain adaptation loss provides a more consistent Table 4 : Ablation study for the meta-learner.

Reported results are averages of 5 individual runs.

ensemble is to assign equal weight to each language.

meta-learner is our basic framework in Figure 1 .

Both results are reported on Chinese human instructions.

and steady improvement.

At the endpoint (100%), the SPL is 22.14 vs 21.50, proves its efficiency and the potential of transferring knowledge between different languages.

We compare the statistics of the Chinese annotated dataset with a machine-translated one.

The annotated instructions are more likely to contain fewer words as well as fewer instructions.

Besides, nouns and verbs, which usually represent landmarks and actions in VLN task, are more frequent in annotated instructions than machine-translated ones.

@highlight

We introduce a new task and dataset on cross-lingual vision-language navigation, and propose a general cross-lingual VLN framework for the task.