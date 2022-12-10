In open-domain dialogue intelligent agents should exhibit the use of knowledge, however there are few convincing demonstrations of this to date.

The most popular sequence to sequence models typically “generate and hope” generic utterances that can be memorized in the weights of the model when mapping from input utterance(s) to output, rather than employing recalled knowledge as context.

Use of knowledge has so far proved difficult, in part because of the lack of a supervised learning benchmark task which exhibits knowledgeable open dialogue with clear  grounding.

To that end we collect and release a large dataset with conversations directly grounded with knowledge retrieved from Wikipedia.

We then design architectures capable of retrieving knowledge, reading and conditioning on it, and finally generating natural responses.

Our best performing dialogue models are able to conduct knowledgeable discussions on open-domain topics as evaluated by automatic metrics and human evaluations, while our new benchmark allows for measuring further improvements in this important research direction.

Arguably, one of the key goals of AI, and the ultimate the goal of natural language research, is for humans to be able to talk to machines.

In order to get close to this goal, machines must master a number of skills: to be able to comprehend language, employ memory to retain and recall knowledge, to reason about these concepts together, and finally output a response that both fulfills functional goals in the conversation while simultaneously being captivating to their human speaking partner.

The current state-of-the-art approaches, sequence to sequence models of various kinds BID20 BID23 BID17 BID21 attempt to address some of these skills, but generally suffer from an inability to bring memory and knowledge to bear; as indicated by their name, they involve encoding an input sequence, providing limited reasoning by transforming their hidden state given the input, and then decoding to an output.

To converse intelligently on a given topic, a speaker clearly needs knowledge of that subject, and it is our contention here that more direct knowledge memory mechanisms need to be employed.

In this work we consider setups where this can be naturally measured and built.

We consider the task of open-domain dialogue, where two speakers conduct open-ended chit-chat given an initial starting topic, and during the course of the conversation the topic can broaden or focus on related themes.

During such conversations, an interlocutor can glean new information and personal points of view from their speaking partner, while providing similarly themselves.

This is a challenging task as it requires several components not found in many standard models.

We design a set of architectures specifically for this goal that combine elements of Memory Network architectures BID19 to retrieve knowledge and read and condition on it, and Transformer architectures BID21 to provide state-of-the-art text representations and sequence models for generating outputs, which we term Transformer Memory Networks.

As, to our knowledge, no public domain dataset of requisite scale exists, we build a supervised dataset of human-human conversations using crowd-sourced workers, first crowd-sourcing 1365 diverse discussion topics and then conversations involving 201, 999 utterances about them.

Each topic is connected to Wikipedia, and one of the humans (the wizard) is asked to link the knowledge they use to sentences from existing articles.

In this way, we have both a natural way to train a knowledgeable conversation agent, by employing a memory component that can recall and ground on this existing text, and a natural way to evaluate models that we build, by assessing their ability at locating and using such knowledge.

Our Transformer Memory Network architectures, both in retrieval and generative versions, are tested in this setup using both automatic metrics and human evaluations.

We show their ability to execute engaging knowledgeable conversations with humans, compared to a number of baselines such as standard Memory Networks or Transformers.

Our new benchmark, publicly in ParlAI (http:// parl.ai/projects/wizard of wikipedia/), aims to encourage and measure further improvements in this important research direction.

Many existing dialogue tasks do not study the use of knowledge explicitly.

For example, popular chit-chat datasets such as Open-Subtitles BID23 , Persona-Chat BID26 and Twitter BID18 have tested the ability of sequence-to-sequence models that attend over the recent dialogue history, but do not attempt to recall long-term knowledge beyond encoding it directly into the weights of the feed-forward network.

In the area of goal-directed dialogue, separate from open domain chit-chat, such as airline BID6 or restaurant booking BID9 BID24 , knowledge conditioning is typically employed by allowing access to a database through API calls or otherwise.

In contrast, our work investigates unstructured knowledge across a large, diverse set of topics potentially spanning all of Wikipedia.

In question answering one does not produce a dialogue response based on a conversation history, but a factual answer based on a question.

In that case, it is clear that retrieving and conditioning knowledge is vital.

For example, in SQuAD neural models have been developed that attend to a given paragraph from Wikipedia to answer questions BID15 , or Open-SQuAD which extends this to answering without being given the paragraph, instead performing retrieval over the entirety of Wikipedia BID3 .

Recently, the QuAC dataset investigates similar themes, but as a sequence of questions and answers in dialogue form instead BID4 .

In this work we do not address question answering, but focus on natural human dialogues which contain a diverse set of utterances, not just questions and answers.

The closest work to ours lies in the area of non-goal directed dialogue incorporating knowledge.

The work of BID5 employed Memory Networks to perform dialogue discussing movies in terms of recommendation and open-ended discussion from Reddit, conditioning on a structured knowledge base.

Zhou et al. (2018) also links Reddit to structured knowledge.

Both BID14 and BID8 use unstructured text instead, as we do: the former to discuss news articles using Wikipedia summaries as knowledge, and the latter to discuss local businesses in two-turn dialogues using Foursquare tips as knowledge.

BID8 uses an extended Encoder-Decoder where the decoder is provided with an encoding of the context along with the external knowledge encoding.

Neither involves dialogue authored with the given knowledge, so it is unclear when knowledge is useful or not.

In contrast, in our task, we know the Wikipedia articles and sentences that ground crowdworkers dialogues.

Model-wise, BID14 uses a Bag-of-Words Memory Network type fact encoder and an RNN decoder.

Our work compares Memory Networks BID19 and Transformers which have been shown to be on-par or superior to RNN encoder-decoders BID21 , and develops an architecture that combines these approaches.

Concurrently with our work BID13 proposed a dataset based on the closed domain of movie chats.

Our paper shows models working on full multi-turn dialogue in an open-domain setting, which to our knowledge was not shown before.

We consider the following general open-domain dialogue setting: two participants engage in chitchat, with one of the participants selecting a beginning topic, and during the conversation the topic is allowed to naturally change.

The two participants, however, are not quite symmetric: one will play the role of a knowledgeable expert (which we refer to as the wizard) while the other is a curious learner (the apprentice).Apprentice At each stage of the conversation the apprentice talks to the wizard freely, playing the role of a curious learner, eager to chat.

Their goal is to go into depth about a chosen topic that interests themselves or their partner, while keeping the conversation engaging and fun.

Note that the instruction to delve deeply into a topic makes this different to more "shallow" chit-chat tasks.

In this task the use of knowledge is emphasized more.

Wizard The wizard is given the following instructions: "You have just met the other person, who seems quite curious, and you are eager to discuss a topic with them!"

Their goal is to inform their conversation partner about a topic that one of them will choose.

Crucially, the wizard has access to an information retrieval system that shows them paragraphs from Wikipedia possibly relevant to the conversation, which are unobserved by the apprentice.

Before each conversation turn the wizard can read these paragraphs and then potentially base their next reply on that observed knowledge.

Note, the wizard is particularly instructed not to simply parrot this knowledge, but to use it to craft a relevant reply, and to present any relevant knowledge in a fun and engaging way, if possible.

The flow of the conversation thus takes place as follows.1.

Either the wizard or apprentice is picked to choose the topic and speak first.

The other player receives the topic information, and the conversation begins.2.

When the apprentice sends the wizard a message, the wizard is shown relevant knowledge (described below), and chooses a relevant sentence in order to construct a response, or else chooses the no sentence used option.3.

The Wizard responds to the apprentice basing their response on their chosen sentence.4.

The conversation repeats until one of the conversation partners ends the chat (after a minimum of 4 or 5 turns each, randomly chosen beforehand).After collecting data of such wizard-apprentice conversations between humans, the goal is to then replace the human wizard with a learned agent that will speak to a human apprentice instead, similar to the procedure in Wizard of Oz experiments BID0 .Topics We crowd-sourced a set of 1365 natural, open-domain dialogue topics, each linked to a Wikipedia article.

These include diverse topics such as commuting, Gouda cheese, music festivals, podcasts, bowling, and Arnold Schwarzenegger.

Knowledge Retrieval At each step of the dialogue the wizard has access to a set of passages of knowledge which may be relevant to the given dialogue context.

While this is a potentially learnable part of the model, we required for this to be fixed so that we could present the results to the annotator when collecting the dataset.

We thus used exactly the same retriever that is commonly used for the Open-SQuAD dataset in BID3 .

It uses a simple inverted index lookup followed by term vector model scoring.

Articles and queries are compared as TF-IDF weighted bag-of-word and n-gram vectors, using the hashing trick.

We retrieve the top 7 articles (first paragraph only) for the last two turns of dialogue (by wizard and apprentice) and the article (first 10 sentences only) for the original topic, and present these articles to the wizard as knowledge context, along with their titles.

Note that while this system is used to build the dataset, a superior method can in principle be learned and used by a model at test time.

Knowledge Selection and Response Generation During data collection, the wizard can click on any of the retrieved article titles in the dialogue UI to expand that article, at which point they can click on a sentence that is most relevant to the response they want to make (only one article, and one sentence can be selected on any turn, for simplicity).

If they see no relevant article or sentence they can choose no sentence used instead.

The wizard then enters their response to the apprentice.

An image of the Wizard's UI is shown in Appendix A.1.

In this work we consider learning dialogue models to replace the wizard in our learning tasks, i.e. the knowledgeable speaker.

The dialogue model thus can have access to a knowledge source, in this case Wikipedia, to ground the conversation with.

We thus develop extensions of the Memory Network BID19 and Transformer BID21 models that can (i) retrieve from a large memory relevant information conditioned on the dialogue history, (ii) carefully read and attend over the retrieved set of knowledge, and then (iii) generate the next dialogue utterance.

This model is then used consecutively on each turn to form an entire dialogue with a user.

We develop two classes of models capable of leveraging knowledge: (i) retrieval models that produce an output among a set of candidate responses (the set of utterances from the training set); and (ii) generative models that generate word-by-word (using a beam).The input to either model is the same: at each dialogue turn where the model is intended to make a response, it is given the current dialogue context x 1 , . . .

, x t of t dialogue turns, where x 1 is always the initial starting topic (e.g. "Kurt Cobain"), and the remaining turns swap between the two speakers.

The goal at each stage is to output the next utterance x t+1 .Knowledge Retrieval We assume a large knowledge base (memory) m 1 , . . .

, m N which is hierarchically organized into documents consisting of paragraphs and sentences.

As it is infeasible for current neural attention techniques to operate on this scale, we use standard information retrieval (IR) techniques (c = IR(x, m)) as a first step to return a smaller set of candidates m c1 , . . .

, m c K for fine-grained selection.

In our experiments, we use the IR system provided to the human annotators during dataset creation, detailed in Section 3.

The retriever operates on the topic (x 1 ) and the last two turns (x t and x t−1 ) if they exist, effectively calling the IR system three times with three different queries.

Empirically, this provided better performance compared to merging into one query, likely because it can address quite different topics.

We retrieve the top 7 articles (first paragraph only) for each lookup and then flatten all the results into separate sentences (i.e. remove the organization of sentences belonging to articles), but prepend every sentence with its article title.

In this way the candidates m c1 , . . .

, m c K given to the neural model in the next stage can be attended to independently without having to deal with hierarchical issues.

Knowledge Attention We use an attention mechanism to perform fine-grained selection of which knowledge sentences will be used to produce the next turn of dialogue.

Each sentence in the memory is independently encoded with a Transformer encoder BID21 , and the same Trans- former is used to encode the dialogue context x. We then perform standard dot-product attention between the memory candidates and the dialogue context.

Utterance Prediction Given the hidden state derived from the memory attention process described above, the final stage is to predict the output utterance that will form the next dialogue turn.

We consider different variants of the two stages above, knowledge attention and utterance prediction, when considering retrieval and generative variants of our models.

We will now detail these in turn.

This model encodes each knowledge sentence m c1 , . . .

, m c K and the dialogue context x with a Transformer, as described above.

The final input encoding is calculated by performing dot-product attention over enc(m c1 ), . . .

, enc(m c K ) and adding the resulting weighted sum of these vectors to enc(x) to get the representation rep LHS (m c1 , . . .

, m c K , x).

The candidate responses r 1 , . . .

, r L are encoded with a separate Transformer to get rep RHS (r i ) for each i. We choose as a response r where DISPLAYFORM0 The model is trained to minimize the cross-entropy loss, where the negative candidates for each example are the responses to the other examples in the batch BID10 .

We consider two versions: a Two-stage and an End-to-end version.

Both models find the most relevant piece of knowledge m best , and then perform an encoding step by concatenating it with the dialogue context, allowing the decoder to attend over both the knowledge and dialogue when formulating its response.

We employ a beam search of 5 to select our best response.

All generative models employ BPE encoding BID16 , which we found effective at enabling generators to copy rare words from Wikipedia sentences BID7 .In the End-to-end version, a shared Transformer encoder is used to encode all candidates m ci and the dialogue history.

The encoded candidates are flattened into vectors using the normalization from BID2 (summing, and normalizing by the square root of the sentence length in order to balance short and long sentences) to produce an attention prediction over the memory.

The full sequence encoding of the single highest selected knowledge m best is concatenated with the encoding of the dialogue, and passed into a Transformer decoder.

An illustration of our End-to-end model is shown in FIG0 .

We train the model to minimize the negative log-likelihood of the response utterance.

We can add additional supervision by forcing the knowledge selection to correctly choose the same knowledge candidate as the human wizard in the training set by adding an additional crossentropy loss over the knowledge attention, modulated by a weight λ: DISPLAYFORM0 In the Two-stage version, we employ two separately trained models for each of these two tasks, knowledge selection and utterance prediction.

As the knowledge selection step creates a hard deci- sion influencing the output of the generator, we find maximizing the performance of this component to be vital.

We can also improve performance of the decoder by employing knowledge dropout (K.D.), wherein we artificially prevent the model from attending to knowledge a fraction of the time during training.

We find this helps the generator be more resilient to errors at the knowledge selection stage, and makes training faster.

K. D. is a novel technique we propose here, however it is similar to many other dropout techniques, e.g. feature dropout used in BID25 .

We describe each of our experimental setups and results.

We first investigate the ability of our models to select knowledge appropriately, and then consider the full task of dialogue with knowledge.

Before looking at the full Wizard dialogue task, we assess the ability of models to predict the knowledge selected by human wizards in the dataset given the dialogue history.

This will inform us of the feasibility of this task and the best models to use in a two-stage architecture.

We compare Transformers against various baselines including a random baseline; an Information Retrieval (IR) baseline, which uses simple word overlap; and a Bag-of-Words Memory Network BID19 .

Where noted, the Transformer is pretrained on Reddit data BID12 , and fine-tuned for our task.

The results are shown in TAB1 .

Transformers work best, as long as they are pretrained on a large dataset (Reddit), while multi-tasking on SQuAD provides marginal impact.

Further analysis of this task using other models is provided in Appendix B.1.

We use the best performing Transformer model reported here for our two-stage generative Memory Network in the full dialogue task.

We evaluate our models on the full task of dialogue generation given knowledge in two settings: given the gold knowledge sentence chosen by a human, or where the model needs to predict which knowledge to use.

We separately describe experiments for retrieval and generative models.

We use similar baselines as in the knowledge selection experiments, but now also apply Transformer Memory Networks, which attend over knowledge.

Models are evaluated measuring Recall@1 when ranking the gold response among 99 randomly chosen candidates, and unigram F1 of the model's prediction with the gold response.

The results are shown in TAB2 .

We find that the addition of knowledge improves all models (improving Bow MemNet from 56 to 71 R@1 and the Transformer MemNet from 79 to 87 R@1) for predicted knowledge.

Performance improves dramatically when models are provided gold knowledge, but otherwise retain similar trends.

Generative Experiments We compare our generative End-to-end and Two-stage Transformer Memory Network models to two more baselines: repeating the last utterance, and a generative Transformer model trained to respond to dialogue but without access to knowledge.

Models are evaluated using perplexity (PPL) of the gold response and unigram F1.

The results are given in TAB3 .

Our experiments show that both the End-to-end and Two-stage models employ the knowledge in their response predictions, as they outperform their counterpart Transformer without knowledge, and demonstrate substantial improvements when provided the gold knowledge.

While the Two-stage model produces significantly stronger perplexity and F1 scores using the predicted knowledge, the End-to-end model outperforms the Two-stage model in the gold knowledge experiments.

This suggests that the Two-stage model benefits from the strong knowledge selection module (Section 5.1), but that the End-to-end model is better at employing the selected knowledge.

Furthermore, we find that the additional knowledge selection supervision (auxiliary loss) in the End-to-end model improves it on every metric, suggesting that tightly integrating these tasks is beneficial.

Knowledge dropout (K. D.) also helps (compare last two rows).

More evidenceTopic: E-book (seen) Human:

i do not know why, but I have never gotten into E-books Model:

Same here.

I have a kindle, but I don't use it for books.

It's nicer to hold a real book and turn pages.

Human: yes, i like the physical feel and smell of a real book Model: I actually had a Nook e-book where you had to download the e-book after selecting it online.

I grew out of it, I prefer paper books also.

Human: i tried e-books but I do not like reading on a screen, its disorientated for me Model: Yes, reading a book on a tablet just isn't the same!

Human: plus I like owning the physical copy of a real book, I feel like I own it Model: I prefer print and newsreels.

for this is shown in Appendix B.1.

Lastly, we note that both Two-stage models give higher F1 scores than any of the retrieval models shown in TAB2 .

We perform human evaluation of our models using crowd-sourced workers.

Humans are paired with our models and asked to chat about a specific topic (given a choice of 2-3 topics) for 3-5 dialogue turns.

Following their conversation, the humans are asked to rate their dialogue partner on a scale of 1-5, with the rating indicating how much they "liked" the conversation (5 is best), which we refer to as the engagingness rating.

Using the collected conversations, we also calculate a metric we call the Wiki F1 sore: the F1 overlap of the model's utterances with the first 10 sentences of the Wikipedia page for the chosen topic as a proxy for how much knowledge the model exhibits.

We seek a model that can be simultaneously engaging and knowledgeable, hence we would like to maximize both these metrics 1 .

For comparison, we also collect 100 human-human conversations, with only one human choosing the topic and performing evaluation.

In total, we collect a total of 546 conversations with ratings from 464 distinct workers.

These results are shown in TAB4 .We find that the retrieval models significantly outperform the generative models on the human engagingness evaluation(Student's t-test, p < .05).

The human engagingness differences between retriever models with and without knowledge are not significant, but note they both trend toward use of knowledge due to the candidate sentences retrieved, with the knowledgeable version obtaining significantly higher Wiki F1 scores in both seen and unseen test sets.

For the generative models, we find human engagingness ratings are significantly improved by the use of knowledge (p < .01).

The significantly higher Wiki F1 scores indicate that (i) these models convey more knowledge than their counterparts without knowledge conditioning; and (ii) on both seen and unseen sets they convey more knowledge than the retrieval models.

In particular, on unseen data the gap between retrieval and generative models is larger.

This is understandable, as retrieval models are limited to producing a response from the training set where the unseen topic did not appear.

There is still a considerable gap to human ratings of other humans compared to all our models (first row of TAB4 ).

FIG2 shows example conversations with the retrieval and generative models.

Additional analysis and examples can be found in Appendix B.3 and C.

In this work we build dialogue agents which are able to employ large memory systems containing encyclopedic knowledge about the world in order to conduct engaging open-domain conversations.

We develop a set of architectures, Transformer Memory Network models, that are capable of retrieving and attending to such knowledge and outputting a response, either in retrieval or generative modes.

To train and evaluate such models, we collect the Wizard of Wikipedia dataset, a large collection of open-domain dialogues grounded by Wikipedia knowledge, and demonstrated the effectiveness of our models in automatic and human experiments.

Our new publicly available benchmark aims to encourage further model exploration, and we expect such efforts will result in significant advances in this important research direction.

There is much future work to be explored using our task and dataset.

Some of these include: (i) bridging the gap between the engagingness of retrieval responses versus the ability of generative models to work on new knowledge and topics, (ii) learning to retrieve and reason simultaneously rather than using a separate IR component; and (iii) investigating the relationship between knowledge-grounded dialogue and existing QA tasks which also employ such IR systems.

The aim is for those strands to come together to obtain an engaging and knowledgeable conversational agent.

Examples of collected conversations from the dataset, where both wizard and apprentice are humans.

The wizard has access to an information retrieval system over Wikipedia, so that they can ask and answer questions, and make statements relevant to the discussion.

For each utterance, knowledge retrieval is performed based on dialogue history, giving ∼61 knowledge candidates per turn, with wizards clicking no sentence used 6.2% of the time.

Assuming that a question contains a question mark or begins with 'how', 'why', 'who', 'where', 'what' or 'when' , in the dataset Apprentices ask questions in 13.9% of training set utterances, and answer questions (i.e., the Wizard has asked a question)

39.5% of the time, while saying new or follow-on statements (neither asking nor answering a question) 49.3% of the time.

Hence, the wizard and apprentice conduct conversations with a variety of dialogue acts.

To choose between topics that are natural we employed the existing Persona-Chat dataset BID26 where crowdworkers where asked to create personas of typical speakers.

There are ∼1000 personas, each of which consists of 4-5 sentences describing that person's interests, e.g. "I love watching Game of Thrones", "I like to eat cheetos" and "I recently got a cat".

These can thus naturally be seen as topics of interest, and using another set of annotators we mapped each sentence to 1 or more relevant Wikipedia pages, if possible, e.g. "Ariel is my favorite princess" was labeled with the Wikipedia page for The Little Mermaid.

As some sentences are harder to connect with Wikipedia, e.g. "I am witty", they are left unlabeled.

We thus obtain 1,431 topics in total to use for our task.

We retain the persona topic sets and thus present 2-3 related topic choices as conversation starters per dialogue during data collection.

B ADDITIONAL EXPERIMENTS B.1 KNOWLEDGE SELECTION Table 6 : Test performance of the Knowledge Selection Tasks.

We also tested the performance of our models trained to do the full dialogue task (see Section 5.2) on the knowledge selection task.

For our retrieval system, this refers to the performance of the knowledge attention.

The results show that our retrieval system could be improved, and the auxiliary loss clearly helps the generative models.

We perform an analysis of the dialogues produced from the human evaluation experiments detailed in Section 5.3.

We sample 20 conversations from each experimental setting, split between seen and unseen.

Conversations are re-tokenized and lowercased to reduce superficial differences between models, and then analyzed in a single-blind setup.

We note of common errors and behaviors exhibited in each of the different conversations.

In general, the human-human conversations are starkly different than any of the bot conversations -humans tend to have more small talk, or use the topic of discussion as a mere icebreaker, with neither human behaving as a wizard.

This is in contrast to human-human conversations from the Wizard dataset itself, where one human has access to Wikipedia, and the conversation becomes more grounded in factual sentences.

Similarly, all models attempt to play the role of wizard and produce more factual sentences too.

In some rounds, humans treat the bot as a sort of question-answer machine, suggesting that the models could be improved by additionally employing SQuAD-like training data.

The retriever without knowledge is particularly prone to non sequiturs, or rapidly changing the subject.

During unseen conversations, it is especially likely to discuss something other than the chosen topic.

In contrast, the retriever with knowledge tends to stick to the chosen topic strongly, but has difficulty if the human changes the subject.

Frequently in unseen topics, the retriever with Table 7 : Retrieval methods on the full Wizard task.

In addition to the models we tested in the paper, we also tested a two-stage retrieval system in which we used our best-performing model on the knowledge selection task to choose a single knowledge sentence to condition on for the dialogue retrieval task.

This outperformed our best retrieval method in terms of F1 but not not in terms of Recall@1.

Furthermore, we compared these results to a two-stage retrieval system in which the dialogue retrieval module is optimized for seeing the gold chosen knowledge sentence.

The performance of this system on the gold knowledge task suggests that the retrieval system could be improved by increasing performance on the knowledge selection subtask.

Table 8 : Human Experiments.

We calculate the Wiki F1 score for the wizard and apprentice as they appear in the dataset for the sake of comparison to our human evaluations.

Note that this differed from the human-human evaluation set-up in the sense that the wizard had direct access to Wikipedia passages in the UI, which explains the higher values of Wiki F1 both for the wizard (who uses Wikipedia) and for the apprentice (who would likely reference that use).

knowledge produces similar, but factually inaccurate answers to user queries.

For example, when one user asks about parts of Ireland to visit, the model enumerates a list of locations in Greece.

Nonetheless, its repertoire of available responses often include inviting responses, allowing the bot to have a more natural conversational flow.

Selected conversations with the retriever with knowledge may be found in FIG5 , for both seen and unseen topics.

The generator without knowledge is particularly prone to many of the typical behaviors of seq2seq systems BID11 BID22 , including local repetition ("cookies are made of flour, oil, oil, and oil"), global repetition (producing the near same utterance for multiple turns), or inconsistencies in its personality (saying it both likes and dislikes movies).

The generator with knowledge has significantly fewer issues with repetition, as it errs on the side of copying large fragments from the Wikipedia knowledge.

The generator with knowledge can also act as a selfish conversationalist, choosing to respond or detail information without inviting a response.

Although it generally produces accurate statements, it sometimes produces statements using an incorrect date, name or word.

It also frequently produces formulaic responses, like "I don't know, but I do know that [Wikipedia excerpt]".

Nonetheless, we find the generator with knowledge is able to successfully generalize to unseen topics using the knowledge from Wikipedia.

Selected conversations with the generator with knowledge may be found in Figure 5 .

@highlight

We build knowledgeable conversational agents by conditioning on Wikipedia + a new supervised task.