The task of visually grounded dialog involves learning goal-oriented cooperative dialog between autonomous agents who exchange information about a scene through several rounds of questions and answers.

We posit that requiring agents to adhere to rules of human language while also maximizing information exchange is an ill-posed problem, and observe that humans do not stray from a common language, because they are social creatures and have to communicate with many people everyday, and it is far easier to stick to a common language even at the cost of some efficiency loss.

Using this as inspiration, we propose and evaluate a multi-agent dialog framework where each agent interacts with, and learns from, multiple agents, and show that this results in more relevant and coherent dialog (as judged by human evaluators) without sacrificing task performance (as judged by quantitative metrics).

Intelligent assistants like Siri and Alexa are increasingly becoming an important part of our daily lives, be it in the household, the workplace or in public places.

As these systems become more advanced, we will have them interacting with each other to achieve a particular goal BID9 .

We want these conversations to be interpretable to humans for the sake of transparency and ease of debugging.

Having the agents communicate in natural language is one of the most universal ways of ensuring interpretability.

This motivates our work on goal-driven agents which interact in coherent language understandable to humans.

To that end, this paper builds on work by BID2 on goal-driven visual dialog agents.

The task is formulated as a conversation between two agents, a Question (Q-) and an Answer (A-) bot.

The A-Bot is given an image, while the QBot is given only a caption to the image.

Both agents share a common objective, which is for the Q-Bot to form an accurate mental representation of the unseen image using which it can retrieve, rank or generate that image.

This is facilitated by the exchange of 10 pairs of questions and answers between the two agents, using a shared vocabulary.

BID2 trained the agents first in isolation via supervision from the VisDial dataset BID1 , followed by making them interact and adapt to each other via reinforcement learning by optimizing for better task performance.

While trying to maximize performance, the agents learn to communicate in non-grammatical and semantically meaningless sentences in order to maximize the exchange of information.

This reduces transparency of the AI system to human observers and is undesirable.

We address this problem by proposing a multi-agent dialog framework where each agent interacts with multiple agents.

This is motivated by our observation that humans adhere to syntactically and semantically coherent language, which we hypothesize is because they have to interact with an entire community, and having a private language for each person would be extremely inefficient.

We show that our multi-agent (with multiple Q-Bots and multiple A-Bots) dialog system results in more coherent and human-interpretable dialog between agents, without compromising on task performance, which also validates our hypothesis.

This makes them seem more helpful, transparent and trustworthy.

We will make our code available as open-source.

1

The game involves two collaborative agents a question bot (Q-bot) and an answer bot (A-bot).

The A-Bot is provided an image, I (represented as a feature embedding y gt extracted by, say, a pretrained CNN model BID13 ), while the Q-Bot is provided with only a caption of the image.

The Q-Bot is tasked with estimating a vector representationŷ of I, which is used to retrieve that image from a dataset.

Both agents receive a common penalty from the environment which is equal to the error inŷ with respect to y gt .

Thus, an unlimited number of games may be simulated without human supervision, motivating the use of reinforcement learning in this framework.

Our primary focus for this work is to ensure that the agents' dialog remains coherent and understandable while also being informative and improving task performance.

For concreteness, an example of dialog that is informative yet incoherent: question: "do you recognize the guy and age is the adult?", answered with: "you couldn't be late teens, his".

The example shows that the bots try to extract and convey as much information as possible in a single question/answer (sometimes by incorporating multiple questions or answers into a single statement).

But in doing so they lose basic semantic and syntactic structure.

Most of the major works which combine vision and language have traditionally been focusing on the problem of image captioning (( BID7 , BID17 , BID14 , BID5 , BID12 , BID19 ) and visual question answering ( BID0 , BID20 , BID4 , BID18 ).

The problem of visual dialog is relatively new and was first introduced by BID1 who also created the VisDial dataset to advance the research on visually grounded dialog.

The dataset was collected by pairing two annotators on Amazon Mechanical Turk to chat about an image.

They formulated the task as a 'multi-round' VQA task and evaluated individual responses at each round in an image guessing setup.

In a subsequent work by BID2 they proposed a Reinforcement Learning based setup where they allowed the Question bot and the Answer bot to have a dialog with each other with the goal of correctly predicting the image unseen to the Question bot.

However, in their work they noticed that the reinforcement learning based training quickly lead the bots to diverge from natural language.

In fact recently showed that language emerging from two agents interacting with each other might not even be interpretable or compositional.

Our multi-agent framework aims to alleviate this problem and prevent the bots from developing a specialized language between them.

Interleaving supervised training with reinforcement learning also helps prevent the bots from becoming incoherent and generating non-sensical dialog.

Recent work has also proposed using such goal driven dialog agents for other related tasks including negotiation BID10 and collaborative drawing (Kim et al., 2017) .

We believe that our work can easily extend to those settings as well.

proposed a generative-discriminative framework for visual dialog where they train only an answer bot to generate informative answers for ground truth questions.

These answers were then fed to a discriminator, which was trained to rank the generated answer among a set of candidate answers.

This is a major restriction of their model as it can only be trained when this additional information of candidate answers is available, which restricts it to a supervised setting.

Furthermore, since they train only the answer bot and have no question bot, they cannot simulate an entire dialog which also prevents them from learning by self-play via reinforcement.

BID16 further improved upon this generative-discriminative framework by formulating the discriminator as a more traditional GAN BID3 , where the adversarial discriminator is tasked to distinguish between human generated and machine generated dialogs.

Furthermore, unlike they modeled the discriminator using an attention network which also utilized the dialog history in predicting the next answer allowing it to maintain coherence and consistency across dialog turns.

We briefly describe the agent architectures and leave the details for the appendix.

The question bot architecture we use is inspired by the answer bot architecture in BID2 and but the individual units have been modified to provide more useful representations.

Similar to the original architecture, our Q-Bot, shown in FIG2 , also consists of 5 parts, (a) fact encoder, (b) state-history encoder, (c) caption encoder, (d) image regression network and (e) question decoder.

The fact encoder is modeled using a Long-Short Term Memory (LSTM) network, which encodes the previous question-answer pair into an embedding F Q t .

We modify the statehistory encoder to incorporate a two-level hierarchical encoding of the dialog.

It uses the fact embedding F Q t at each time step to compute attention over the history of dialog, (F DISPLAYFORM0 and produce a history encoding S Q t .

The key modification (compared to ) in our architecture is the addition of a separate LSTM to compute a caption embedding C. This is key to ensuring that the hierarchical encoding does not exclusively attend on the caption while generating the history embedding, and prevents the occurrence of repetitive questions in a dialog since the encoding now has an adequate representation of the previous facts.

The caption embedding is then concatenated with F Q t and S Q t , and passed through multiple fully connected layers to compute the state-history encoder embedding e t and the predicted image feature embeddingŷ t = f (S Q t ).

The encoder embedding, e Q t is fed to the question decoder, another LSTM, which generates the question, q t .

For all LSTMs and fully connected layers in the model we use a hidden layer size of 512.

The image feature vector is 4096 dimensional.

The word embeddings and the encoder embeddings are 300 dimensional.

The architecture for A-Bot, also inspired from , shown in FIG2 , is similar to that of the Q-Bot.

It has 3 components: (a) question encoder, (b) state-history encoder and (c) answer decoder.

The question encoder computes an embedding, Q t for the question to be answered, q t .

The history encoding (F A 1 , F A 2 , F A 3 ...F A t ) →

S A t uses a similar two-level hierarchical encoder, where the attention is computed using the question embedding Q t .

The caption is passed on to the A-Bot as the first element of the history, which is why we do not use a separate caption encoder.

Instead, we use the fc7 feature embedding of a pretrained VGG-16 BID13 model to compute the image embedding I. The three embeddings S A t , Q t , I are concatenated and passed through another fully connected layer to extract the encoder embedding e A t .

The answer decoder, which is another LSTM, uses this embedding e A t to generate the answer a t .

Similar to the Q-Bot, we use a hidden layer size of 512 for all LSTMs and fully connected layers.

The image feature vector coming from the CNN is 4096 dimensional (FC7 features from VGG16).

The word embeddings and the encoder embeddings are 300 dimensional.

We follow the training process proposed in BID2 .

Two agents, a Q-Bot and an A-Bot are first trained in isolation, by supervision from the VisDial dataset.

After this supervised pretraining for 15 epochs over the data, we smoothly transition the agents to learn by reinforcement via a curriculum.

Specifically, for the first K rounds of dialog for each image, the agents are trained using supervision from the VisDial dataset.

For the remaining 10-K rounds, however, they are trained via reinforcement learning.

K starts at 9 and is linearly annealed to 0 over 10 epochs.

The individual phases of training will be described below, with some details in the appendix.

In the supervised part of training, both the Q-Bot and A-Bot are trained separately.

Both the Q-Bot and A-Bot are trained with a Maximum Likelihood Estimation (MLE) loss computed using the ground truth questions and answers, respectively, for every round of dialog.

The Q-Bot simultaneously optimizes another objective: minimizing the Mean Squared Error (MSE) loss between the true and predicted image embeddings.

The ground truth dialogs and image embeddings come from the VisDial dataset.

Given the true dialog history, image features and ground truth question, the A-Bot generates an answer to that question.

Given the true dialog history and the previous question-answer pair, the Q-Bot is made to generate the next question to ask the A-Bot.

However, there are multiple problems with this training scheme.

First, MLE is known to result in models that generate repetitive dialogs and often produce generic responses.

Second, the agents are never allowed to interact during training.

Thus, when they interact during testing, they end up facing out of distribution questions and answers, and produce unexpected responses.

Third, the sequentiality of dialog is lost when the agents are trained in an isolated, supervised manner.

To alleviate the issues pointed out with supervised training, we let the two bots interact with each other via self-play (no ground-truth except images and captions).

This interaction is also in the form of questions asked by the Q-Bot, and answered in turn by the A-Bot, using a shared vocabulary.

The state space is partially observed and asymmetric, with the Q-Bot observing {c, q 1 , a 1 . . .

q 10 , a 10 } and the A-Bot observing the same, plus the image I. Here, c is the caption, and q i , a i is the i th dialog pair exchanged where i = 1 . . .

10.

The action space for both bots consists of all possible output sequences of a specified maximum length (Q-Bot: 16, ABot: 9 as specified by the dataset) under a fixed vocabulary (size 8645).

Note that these parameter values are chosen to comply with the VisDial dataset.

Each action involves predicting words sequentially until a stop token is predicted, or the generated statement reaches the maximum length.

Additionally, Q-Bot also produces a guess of the visual representation of the input image (VGG fc-7 embedding of size 4096).

Both Q-Bot and A-Bot share the same objective and get the same reward to encourage cooperation.

Information gain in each round of dialog is incentivized by setting the reward as the change in distance of the predicted image embedding to the ground-truth image representation.

This means that a QA pair is of high quality only if it helps the Q-Bot make a better prediction of the image representation.

Both policies are modeled by neural networks, as discussed in Section 4.However, as noted above, this RL optimization problem is ill-posed, since rewarding the agents for information exchange does not motivate them to stick to the rules and conventions of the English language.

Thus, we follow the elaborate curriculum scheme described above, despite which the bots are still observed to diverge from natural language and produce non-grammatical and incoherent dialog.

Thus, we propose a multi bot architecture to help the agents interact in diverse and coherent, yet informative, dialog.

Learning Algorithm:

A dialog round at time t consists of the following steps: 1) The Q-Bot, conditioned on the state encoding, generates a question q t , 2) A-Bot updates its state encoding with q t and then generates an answer a t , 3) Both Q-Bot and A-Bot encode the completed exchange as a fact embedding, 4) Q-Bot updates its state encoding to incorporate this fact and finally 5) Q-Bot predicts the image representation for the unseen image conditioned on its updated state encoding.

Similar to BID1 , we use the REIN-FORCE BID15 algorithm that updates network parameters in response to experienced rewards.

The per-round rewards maximized are: DISPLAYFORM0 This reward is positive if the distance between image representation generated at time t is closer to the ground truth than the representation at time t − 1, hence incentivizing information gain at each round of dialog.

The REINFORCE update rule ensures that a (q t , a t ) exchange that is informative has its probabilities pushed up.

Do note that the image feature regression network f is trained directly via supervised gradient updates on the L-2 loss.

In this section we describe our proposed MultiAgent Dialog architecture in detail.

The motivation end while 25: end procedure behind this is the observation that allowing a pair of agents to interact with each other and learn via reinforcement for too long leads to them developing an idiosyncratic private language which does not adhere to the rules of human language, and is hence not understandable by human observers.

We claim that if instead of allowing a single pair of agents to interact, we were to make the agents more social, and make them interact and learn from multiple other agents, they would be disincentivized to develop a private language, and would have to conform to the common language.

In particular, we create either multiple Q-bots to interact with a single A-bot, or multiple A-bots to interact with a single Q-bot.

All these agents are initialized with the learned parameters from the supervised pretraining as described in Section 5.1.

Then, for each batch of images from the VisDial dataset, we randomly choose a Q-bot to interact with the A-bot, or randomly choose an A-bot to interact with the Q-bot, as the case may be.

The two chosen agents then have a complete dialog consisting of 10 question-answer pairs about each of those images, and update their respective weights based on the rewards received (as per Equation 1) during the conversation, using the REINFORCE algorithm.

This process is repeated for each batch of images, over the entire VisDial dataset.

It is important to note that histories are not shared across batches.

MADF can be understood in detail using the pseudocode in Algorithm 1.

We use the VisDial 0.9 dataset for our task introduced by BID1 .

The data is collected using Amazon Mechanical Turk by pairing 2 annotators and asking them to chat about the image as a multi round VQA setup.

One of the annotators acts as the questioner and has access to only the caption of the image and has to ask questions from the other annotator who acts as the answerer and must answer the questions based on the visual information from the actual image.

This dialog repeats for 10 rounds at the end of which the questioner has to guess what the image was.

We perform our Model MRR Mean Rank R@1 R@5 R@10 Answer Prior BID1 0 Figure 3: The percentile scores of the ground truth image compared to the entire test set of 40k images.

The X-axis denotes the dialog round number (from 1 to 10), while the Y-axis denotes the image retrieval percentile score.experiments on VisDial v0.9 (the latest available release) containing 83k dialogs on COCO-train and 40k on COCO-val images, for a total of 1.2M dialog question-answer pairs.

We split the 83k into 82k for train, 1k for validation, and use the 40k as test, in a manner consistent with BID1 .

The caption is considered to be the first round in the dialog history.

We evaluate the performance of our model using 6 metrics, proposed by BID2 : 1) Mean Reciprocal Rank (MRR), 2) Mean Rank, 3) Recall@1, 4) Recall@5, 5) Recall@10 and 6) Image Retrieval Percentile.

Mean Rank and MRR compute the average rank (and its reciprocal, respectively) assigned to the answer generated by the A-bot, over a set of 100 candidate answers for each question (also averaged over all the 10 rounds).

Recall@k computes the percentage of answers with rank less than k. Image Retrieval percentile is a measure of how close the image prediction generated by the Q-bot is to the ground truth.

All the images in the test set are ranked according to their distance from the predicted image embedding, and the rank of the ground truth embedding is used to calculate the image retrieval percentile.

Table 1 compares the Mean Rank, MRR, Recall@1, Recall@5 and Recall@10 of our agent architecture and dialog framework (below the horizontal line) with previously proposed architectures (above the line).

SL refers to the agents after only the isolated, supervised training of Section 5.1.

RL-1Q,1A refers to a single, idiosyncratic pair of agents trained via reinforcement as in Section 5.2.

RL-1Q,3A and RL-3Q,1A refer to social agents trained via our Multi-Agent Dialog framework in Section 5.3, with 1Q,3A referring to 1 Q-Bot and 3 A-Bots, and 3Q,1A referring to 3 Q-Bots and 1 A-Bot.

It can be seen that our agent architectures clearly outperform all previously proposed generative architectures in MRR, Mean Rank and R@10, but not in R@1 and R@5.

This indicates that our approach produces consistently good answers (as measured by MRR, Mean Rank and R@10), even though they might not be the best possible answers (as measured by R@1 and R@5).

SL has the best MRR and Mean Rank, which drops drastically in RL-1Q,1A.

The agents trained by MADF recover and are able to outperform all previously proposed models.

Fig. 3 shows image retrieval percentile scores over dialog rounds.

The percentile score decreases monotonically for SL, but is stable for all versions using RL.

There are no quantitative metrics to comprehensively evaluate dialog quality, hence we do a hu- .

A total of 20 evaluators (randomly chosen students) were shown the caption and the 10 QA-pairs generated by each system for one of 4 randomly chosen images, and asked to give an ordinal ranking (from 1 to 4) for each metric.

If the evaluator was also given access to the image, she was asked only to evaluate metrics 3, 4 and 5, while if the evaluator was not shown the image, she was asked only to evaluate metrics 1, 2 and 5.

Table 2 contains the average ranks obtained on each metric (lower is better).The results convincingly prove our hypothesis that having multiple A-Bots to interact and learn from will improve the Q-Bot, and vice versa.

This is because having multiple A-Bots to interact with gives the Q-Bot access to a variety of diverse dialog, leading to more stable updates with lower bias.

The results confirm this, with Q-Bot Relevance rank being best in RL-1Q,3A, and A-Bot Relevance rank being best in RL-3Q,1A.

These two dialog systems, which were trained via MADF, also have the best overall dialog coherence by a significant margin over RL-1Q,1A and SL.

We show two of the examples shown to the human evaluators in Figure 4 .

The trends observed in the scores given by human evaluators is also clearly visible in these examples.

MADF agents are able to model the human responses much better than the other agents.

It can also be seen that although the RL-1Q,1A system has greater diversity in its responses, the quality of those responses is greatly degraded, with the A-Bot's answers especially being both non-grammatical and irrelevant.

In Section 5.1, we discussed how the MSE loss used in SL results in models which generate repetitive dialog, which can be seen in Fig. 4 .

Consider the first image, where in the SL QA-generations, the Q-Bot repeats the same questions multiple times, and gets inconsistent answers from the A-Bot for the same question.

By contrast, all 10 QA-generations for RL-3Q,1A are grammatically correct.

The Q-Bot's questions are very relevant to the image being considered, and the A-Bot's answers appropriate and correct.

In this paper we propose a novel Multi-Agent Dialog Framework (MADF), inspired from human communities, to improve the dialog quality of AI agents.

We show that training 2 agents with supervised learning can lead to uninformative and repetitive dialog.

Furthermore, we observe that the task performance (measured by the image retrieval percentile scores) for the system trained via supervision only deteriorates as dialog round number increases.

We hypothesize that this is because the agents were trained in isolation and never allowed to interact during supervised learning, which leads to failure during testing when they encounter out of distribution samples (generated by the other agent, instead of ground truth) for the first time.

We show how allowing a single pair of agents to interact and learn from each other via reinforcement learning dramatically improve their percentile scores, which additionally does not deteriorate over multiple rounds of dialog, since the agents have interacted with one another and been exposed to the other's generated questions or answers.

However, the agents, in an attempt to improve task performance end up developing their own private language which does not adhere to the rules and conventions of human languages, and generates nongrammatical and non-sensical statements.

As a result, the dialog system loses interpretability and sociability.

Figure 4: Two randomly selected images from the VisDial dataset followed by the ground truth (human) and generated dialog about that image for each of our 4 systems (SL, RL-1Q,1A, RL-1Q,3A, RL-3Q,1A).

These images were also used in the human evaluation results shown in Table 2 .multi-agent dialog framework based on self-play reinforcement learning, where a single A-Bot is allowed to interact with multiple Q-Bots and vice versa.

Through a human evaluation study, we show that this leads to significant improvements in dialog quality measured by relevance, grammar and coherence.

This is because interacting with multiple agents prevents any particular pair from maximizing performance by developing a private language, since it would harm performance with all the other agents.

We plan to explore several other multi bot architectural settings and perform a more thorough human evaluation for qualitative analysis of our dialog.

We also plan on incorporating an explicit perplexity based reward term in our reinforcement learning setup to further improve the dialog quality.

We will also experiment with using a discriminative answer decoder which uses information of the possible answer candidates to rank the generated answer with respect to all the candidate answers and use the ranking performance to train the answer decoder.

Another avenue for future exploration is to use a richer image feature embedding to regress on.

Currently, we use a regression network to compute the estimated image embedding which represents the Q-Bot's understanding of the image.

We plan to implement an image generation GAN which can use this embedding as a latent code to generate an image which can be visualized.

While the MADF in its current form only works if we have multiple Q-Bots or multiple A-Bots but not both, future work could possibly look at incorporating that into the framework, while ensuring that the updates do not become too unstable.

then generate their respective embeddings which are fed into the feature regression network and the question decoder to produceŷ t and update the hidden state of the question decoder respectively.

The Q-Bot is then trained by maximizing the likelihood p(q gt |E Q t ) of the training data q gt , computed using the softmax probabilities given by the question decoder.

Simultaneously, the Mean Squared Error (MSE) loss between the predicted image embedding and ground truth is also minimized.

Effectively, the loss DISPLAYFORM0 is minimized.

At time step t, the A-Bot's question encoder is fed with the ground-truth question for t, the state/history/image encoder is fed with all the ground-truth QA pairs up to t − 1 and the image I. These encoders then generate their respective embeddings which are fed into the answer decoder to produce a t .

The A-Bot is trained by maximizing the likelihood p(a gt |E A t ) of the training data a gt , computed using the softmax probabilities given by the answer decoder.

Effectively, the loss DISPLAYFORM0

The Q-Bot is given only the caption c gt and the A-Bot is given only the image I and caption c gt as inputs.

At time step t, the Q-Bot's fact encoder is fed with the generated QA pair for t − 1, the state/history encoder is fed with all the generated QA pairs up to t − 1 and the caption encoder is given the true image caption c gt as input.

These encoders then generate their respective embeddings which are fed into the feature regression network and the question decoder to produceŷ t and q t respectively.

The change in distance betweenŷ t and y gt due to the current QA-pair is given as a reward to Q-Bot (Eqn.

1), which it uses to train itself via REINFORCE.

Simultaneously, the Mean Squared Error (MSE) loss between the predicted image embedding and ground truth is also minimized via supervision.

Effectively, the loss DISPLAYFORM0 is minimized, where G t = 10−t k=0 γ k r t+k+1 indicates the Monte-Carlo return at step t, and γ is a discount factor equal to 0.99.

At time step t, the A-Bot's question encoder is fed with the generated question q t , the state/history/image encoder is fed with all the generated QA pairs up to t − 1 and the image I. These encoders then generate their respective embeddings which are fed into the answer decoder to produce a t .

The A-Bot also receives the same reward as the Q-Bot, and trains itself via REINFORCE.

Effectively, the loss DISPLAYFORM0 is minimized, where G t = 10−t k=0 γ k r t+k+1 indicates the Monte-Carlo return at step t, and γ is a discount factor equal to 0.99.

@highlight

Social agents learn to talk to each other in natural language towards a goal