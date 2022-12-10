We present a new unsupervised method for learning general-purpose sentence embeddings.

Unlike existing methods which rely on local contexts, such as words inside the sentence or immediately neighboring sentences, our method selects, for each target sentence, influential sentences in the entire document based on a document structure.

We identify a dependency structure of sentences using metadata or text styles.

Furthermore, we propose a novel out-of-vocabulary word handling technique to model many domain-specific terms, which were mostly discarded by existing sentence embedding methods.

We validate our model on several tasks showing 30% precision improvement in coreference resolution in a technical domain, and 7.5% accuracy increase in paraphrase detection compared to baselines.

Distributed representations are ever more leveraged to understand text BID20 b; BID16 BID23 .

Recently, BID12 proposed a neural network model, SKIP-THOUGHT, that embeds a sentence without supervision by training the network to predict the next sentence for a given sentence.

However, unlike human reading with broader context and structure in mind, the existing approaches focus on a small continuous context of neighboring sentences.

These approaches work well on less structured text like movie transcripts, but do not work well on structured documents like encylopedic articles and technical reports.

To better support semantic understanding of such technical documents, we propose a new unsupervised sentence embedding framework to learn general-purpose sentence representations by leveraging long-distance dependencies between sentences in a document.

We observe that understanding a sentence often requires understanding of not only the immediate context but more comprehensive context, including the document title, previous paragraphs or even related articles as shown in Figure 1.

For instance, all the sentences in the document can be related to the title of the document (1(a)).

The first sentence of each item in a list structure can be influenced by the sentence introducing the list (1(b)).

Moreover, html documents can contain hyperlinks to provide more information about a certain term (1(c)).

With the contexts obtained from document structure, we can connect ransomware with payment (1(a)) and the four hashes with Locky (1(b)).

Millions of spam emails spread new ransomware variant on the day it first appeared.

A new variant of ransomware known as Locky (detected by Symantec as Trojan.

Cryptolocker.

AF) has been spreading quickly since it first appeared on Tuesday (February 16).

The attackers behind Locky have pushed the malware aggressively, using massive spam campaigns and compromised websites.

…

… Ransomware is computer malware that installs covertly on a victim's computer, executes a cryptovirology attack that adversely affects it, and demands a ransom payment to decrypt it or not publish it.

Locky is a new ransomware that has been released (most probably) by the Dridex gang.

Not surprisingly, it is well prepared, which means that the threat actor behind it has invested sufficient resources for it, including its mature infrastructure.

Let's take a look.

o payload: 74dde1905eff75cf3328832988a785de <-main focus of this analysis • d9df60c24ceca5c4d623ff48ccd4e9b9 • e7aad826559c8448cd8ba9f53f401182

These spam campaigns have many similarities to campaigns used to spread the Dridex financial Trojan.

The sheer size of the campaigns, their disguise as financial documents such as invoices, and the use of malicious macros in attached Word documents are all hallmarks of the Dridex group.

Built to harvest the banking credentials of victims, the virulent Dridex is now one of the most dangerous pieces of financial malware in circulation.

Our approach leveraging such structural elements has several advantages.

First, it can learn from technical documents containing several subtopics that may cause sudden context changes.

Some sentences have dependences to distant ones if a different perspective of the topic is introduced.

Using We validate our model on several NLP tasks using a Wikipedia corpus.

When trained with the Wikipedia corpus, our model produces much lower loss than SKIP-THOUGHT in the target sentence prediction task, confirming that training with only local context does not work well for such documents.

We also compare the performance of the learned embedding on several NLP tasks including coreference resolution and paraphrase identification.

For coreference resolution, our model shows roughly 30% improvement in precision over a state-of-the-art deep learning-based approach on cybersecurity domain, and produces 7.5% increase in accuracy compared with SKIP-THOUGHT for paraphrase identification.

The main contributions of the paper include:• We propose a general-purpose sentence embedding method which leverages long distance sentence dependencies extracted from the document structure.• We developed a rule-baed dependency annotator to automatically determine the document structure and extract all governing sentences for each sentence.• We also present a new OOV handling technique based on the document structure.• We have applied our methods to several NLP applications using cybersecurity datasets.

The experiments show that our model consistently outperform existing methods.

Distributed representation of sentences, which is often called sentence embedding, has gained much attention recently, as word-level representations BID20 b; BID16 BID23 are not sufficient for many sentence-level or document-level tasks, such as machine translation, sentiment analysis and coreference resolution.

Recent approaches using neural networks consider some form of dependencies to train the network.

Dependencies can be continuous (relating two adjacent words or sentences) or discontinuous (relating two distant words or sentences), and intra-sentence (dependency of words within a sentence) or inter-sentence (dependency between sentences).

Many sentence embedding approaches leverage these dependencies of words to combine word embeddings, and can be categorized as shown in 1.One direct extension of word embedding to sentences is combining words vectors in a continuous context window.

BID13 use a weighted average of the constituent word vectors.

BID27 , BID3 , and BID22 use supervised approaches to train a long short-term memory (LSTM) network that merges word vectors.

BID10 and BID11 use convolutional neural networks (CNN) over continuous context window to generate sentence representations.

BID14 include a paragraph vector in the bag of word vectors, and apply a word embedding approaches BID20 b) .Recently, several researchers have proposed dependency-based embedding methods using a dependency parser to consider discontinuous intra-sentence relationships BID25 BID18 BID26 .

BID25 uses recursive neural network to consider discontinuous dependencies.

BID18 proposes a dependency-based convolutional neural network which concatenate a word with its ancestors and siblings based on the dependency tree structure.

BID26 proposes tree structured long short-term memory networks.

These studies show that dependency-based (discontinuous) networks outperform their sequential (continuous) counterparts.

Unlike these approaches, considering only intra-sentence dependencies, SKIP-THOUGHT BID12 joins two recurrent neural networks, encoder and decoder.

The encoder combines the words in a sentence into a sentence vector, and the decoder generates the next sentence.

Our approach is similar to SKIP-THOUGHT since both approaches are unsupervised and use inter-sentential dependencies.

However, SKIP-THOUGHT considers only continuous dependency.

Furthermore, we propose a new method to handle OOV words in sentence embedding based on the position of an OOVword in a sentence and the dependency type of the sentence.

To our knowledge, there has been no sentence embedding work incorporating OOV words in formulating the training goal.

Most existing systems map all OOV words to a generic unknown word token (i.e., < unk >).Santos & Zadrozny FORMULA2 and BID9 build an embedding of an OOV word on the fly that can be used as input to our system, but not to set the training goal.

BID17 propose a word position-based approach to address the OOV problem for neural machine translation (NMT) systems.

Their methods allow a neural machine translation (NMT) system to emit, for each unknown word in the target sentence, the position of the corresponding word in the source sentence.

However, their methods are not applicable to sentence embedding, as they rely on an aligned corpus.

Also, our approach considers not only word positions but also the dependency types to represent OOV words in a finer-grained OOV level.

Previous methods use intra-sentence dependencies such as dependency tree, or immediately neighboring sentences for sentence embedding.

However, we identify more semantically related content to a target sentence based on the document structure as shown in FIG1 .

In this section, we describe a range of such inter-sentence dependencies that can be utilized for sentence embedding and the techniques to automatically identify them.

We use the following notations to describe the extraction of document structure-based context for a given sentence.

Suppose we have a document D = {S 1 , . . .

, S |D| }, which is a sequence of sentences.

Each sentence S i is a sequence of words: s i,1 , . . .

, s i,|Si| .

For each target sentence S t ∈ D, there can be a subset G ⊂ D that S t depends on (For simplicity, we use G to denote a S t specific set).

We call such a sentence in G a governing sentence of S t , and say G i governs S t , or S t depends on G i .

Each G i is associated with S t through one of the dependency types in D described below.

The title of a document, especially a technical document, contains the gist of the document, and all other sentences support the title in a certain way.

For instance, the title of the document can clarify the meaning of a definite noun in the sentence.

Section titles play a similar role, but, mostly to the sentences within the section.

We detect different levels of titles, starting from the document title to chapter, section and subsection titles.

Then, we identify the region in the document which each title governs and incorporate the title in the embedding of all the sentences in the region.

To identify titles in a document, we use various information from the metadata and the document content.

DISPLAYFORM0 We extract a document title from the <title> tag in a HTML document or from the title field in Word or PDF document metadata.

Since the document title influences all sentences in a document, we consider a title obtained from D T M governs every sentence in D.Heading Tag (D T Hn ): The heading tags <h1> to <h6> in HTML documents are often used to show document or section titles.

We consider all the sentences between a heading tag and the next occurrence of the same level tag are considered under the influence of the title.

Header and Footer (D T R ): Technical documents often contain the document or section titles in the headers or footers.

Thus, if the same text is repeated in the header or in the footer in many pages, we take the text as a title and consider all the sentences appearing in these pages belong to the title.

Text Styles (D T S ): Titles often have a distinctive text style.

They tend to have no period at the end and contain a larger font size, a higher number of italic or bold text, and a higher ratio of capitalized words compared to non-title sentences.

We first build a text style model for sentences appearing in the document body, capturing the three style attributes.

If a sentence ends without a period and any dimension of its style model has higher value than that of the text style model, we consider the sentence as a title.

Then, we split the document based on the detected titles and treat each slice as a section.

Authors often employ a list structure to describe several elements of a subject.

These list structures typically state the main concept first, and, then, the supporting points are described in a bulleted, numbered or in-text list as illustrated in FIG2 .

In these lists, an item is conceptually more related to the introductory sentence than the other items in the list, but the distance can be long because of other items.

Once list items are identified, we consider the sentence appearing prior to the list items as the introductory sentence and assume that it governs all the items in the list.

The categories of the products State Farm offers are as follows:• We have property and casualty insurance.• We offer comprehensive types of life and health insurances.•

We have bank products.

To extract numbered or bulleted lists, we use the list tags (e.g., <ul>, <ol>, <li>) for HTML documents.

For non-HTML documents, we detect a number sequence (i.e., 1, 2, ...) or bullet symbols (e.g., -, ·) repeating in multiple lines.

In-text List (D LT ): We also identify in-text lists such as "First(ly), . .

..

Second(ly), . .

..

Last(ly), . . ." by identifying these cue words.

We consider the sentence appearing prior to the list items as the introductory sentence and assume that it governs the list items.

Hyperlinks (D H ): Some sentences contain hyperlinks or references to provide additional information or clarify the meaning of the sentence.

We can enrich the representation of the sentence using the linked document.

In this work, we use the title of the linked document in the embedding of the sentence.

Alternatively, we can use the embedding of the linked document.

Footnotes and In-document Links (D F ): Footnotes also provide additional information for the target sentence.

In an HTML document, such information is usually expressed with in-document hyperlinks, which ends with "#dest".

In this case, we identify a sentence marked with "#dest" and add a dependency between the two sentences.

We also consider the traditional sequential dependency used in previous methods BID12 BID7 .

Given a document D = {S 1 , . . .

, S |D| }, the target sentence S t is considered to be governed by n sentences prior to (n < 0) or following (n > 0) S t .

In our implementation, we use only one left sentence.

Similarly to SKIP-THOUGHT BID12 , we train our model to generate a target sentence S t using a set of governing sentences G. However, SKIP-THOUGHT takes into account only the window-based context (D W n ), while our model considers diverse long distance context.

Furthermore, we handle out-of-vocabulary (OOV) words based on their occurrences in the context.

Our model has several encoders (one encoder for each G i ∈ G), a decoder and an OOV handler as shown in FIG3 .

The input to each cell is a word, represented as a dense vector.

In this work, we use the pre-trained vectors from the CBOW model BID21 , and the word vectors can be optionally updated during the training step.

Unlike existing sentence embedding methods, which include only a small fraction of words (typically high frequency words) in the vocabulary and map all other words to one OOV word by averaging all word vectors, we introduce a new OOV handler in our model.

The OOV handler maps all OOV words appearing in governing sentences to variables and extend the vocabulary with the OOV variables.

More details about OOV handler is described in Section 5.We now formally describe the model given a target sentence S t and a set G of its governing sentences.

We first describe the encoders that digest each G i ∈ G. Given the i-th governing sentence G i = FIG1 , . . . , g i,|Gi| ) let w(g i,t ) be the word representation (pre-trained or randomly initialized) of word g i,t .

Then, the following equations define the encoder for S i .

DISPLAYFORM0 where RC is a recurrent neural network cell (e.g., LSTM or GRU) that updates the memory h i,t ; θ E is the parameters for the encoder RC; λ i is an OOV weight vector that decides how much we rely on out-of-vocabulary words; d i denotes the OOV features for G i ; U and g are linear regression parameters; σ(·) is the sigmoid function; u dep and a dep are dependency-specific weight parameters; W and b are a matrix and a bias for a fully connected layer; andh 0 is the aggregated information of G and is passed to the decoder for target sentence generation.

Now, we define the decoder as follows: DISPLAYFORM1 where RC is a recurrent neural network cell that updates the memoryh t and generates the output o t ; θ D is a set of parameters for the decoder RC; softmax(·) is the softmax function; and V o t + c transforms the output into the vocabulary space.

That is, V o t + c generates logits for words in the vocabulary set and is used to predict the words in the target sentence.

To strike a balance between the model accuracy and the training time, we use K randomly chosen governing sentences from G for all target sentence.

We use the cross entropy between y t and o t as the optimization function and update θ E , W dep(i) , b, V, c, θ D and optionally w(·).

DISPLAYFORM2

Incorporating all the words from a large text collection in deep learning models is infeasible, since the amounts of memory use and training time will be too costly.

Existing sentence embedding techniques reduce the vocabulary size mainly by using only high frequency words and by collapsing all other words to one unknown word.

The unknown word is typically represented by the average vector of all the word vectors in the vocabulary or as a single dimension in a bag-of-word representation.

However, this frequency-based filtering can lose many important words including domain-specific words and proper nouns resulting in unsatisfactory results for technical documents.

Specifically, OOV word handling is desired in the following three places: (1) input embeddings to encode the governing sentences (G); (2) input embeddings to decode the target sentence (S t ); and (3) output logits to compute the loss with respect to S t .

In this work, we apply the most commonly used approach, i.e., using the average vector of all the words in the vocabulary to represent all OOV words, to generate the input embeddings of G or S t for the encoder and the decoder.

To handle the OOV words in the output logits, we propose a new method using two vocabulary sets.

We first select N most frequent words in the training corpus as an initial vocabulary V 0 .

Note that N (typically, tens of thousands) is much smaller than the vocabulary size in the training corpus (typically, millions or billions).

The OOV mapper reduces the OOV words into a smaller vocabulary V OOV of OOV variables that can represent certain OOV words given a context (e.g., an OOV variable may indicate the actor in the previous sentence).We note that only the OOV words appearing in governing sentences influence in model training, and many semantically important words tend to appear in the beginning or at the end of the governing sentences.

Thus, we use OOV variables to represent the first and the last η OOV words in a governing sentences.

Specifically, we denote a j-th OOV word in the i dependency governing sentence by an OOV variable O i (j) ∈ V OOV .

This idea of encoding OOV words based on their positions in a sentence is similar to BID17 .

However, we encode OOV words using the dependency type of the sentence as well as their position in the sentence.

Our OOV handler performs the following steps.

First, we build an OOV map to convert OOV words to OOV variables and vice versa.

Algorithm 1 summarizes the steps to build a map which converts the first η OOV words into OOV variables.

To model the last η OOV words, we reverse the words in each G i , and index them as w −1 , w −2 , . . ., then pass them to BuildOOVMap to construct DISPLAYFORM0 Note that the mapping between OOV words and OOV variables is many-to-many.

For example, suppose "We discuss Keras first' is a target sentence S t , and, "Slim and Keras are two tools you must know" is extracted as the document title by the dependency type D T S , "PLA's weekly review: Slim and Keras are two tools you must know" is extracted as the document title by D T M for S t , and, words 'Slim', 'Keras' and 'PLA' are OOV words.

Then, we map the 'Slim' and 'Keras' from the first title to OOV variable O T S (1) and O T S (2) and 'PLA', 'Slim' and 'Keras' from the second title to O T M (1), O T M (2), and O T M (3) respectively.

As a result, 'Keras' in S t is mapped to O T S (1) and O T M (3).Once we have the OOV mapping and the augmented vocabulary, we can formulate an optimization goal taking into account the OOV words with a vocabulary with a manageable size.

The optimization goal of each RNN cell without OOV words is to predict the next word with one correct answer.

In contrast, our model allows multiple correct answers, since an OOV word can be mapped to multiple OOV variables.

We use the cross entropy with soft labels as the optimization loss function.

The weight of each label is determined by the inverse-square law, i.e., the weight is inversely proportional to the square of the number of words associated with the label.

This weighting scheme gives a higher weight to less ambiguous dependency.

One additional component we add related to OOV words is a weight function for the governing sentences based on occurrences of proper nouns (λ i in Equation 1).

Instead of equally weighing all governing sentences, we can give a higher weight to sentences with proper nouns, which are more likely to be OOV words.

Thus, we introduce a feature vector representing the number of OOV proper nouns in the i-th governing sentence (d i in FIG1 ).

Currently, the features include # of OOV words whose initials are uppercased, # of OOV words that are uppercased, and # of OOV words with any of the letters are uppercased.

Together with the linear regression parameters, U and g, the model learns the weights for different dependency types.

In this section, we empirically evaluate our approach on various NLP tasks and compare the results with other existing methods.

We trained the proposed model (OURS) and the baseline systems on 807,647 randomly selected documents from the 2009 Wikipedia dump, which is the latest Wikipedia dump in HTML format, after removing the discussion and resource (e.g., images) articles among.

Since our approach leverages HTML tags to identify document structures, our model use the raw HTML files.

For the baseline systems, we provide plain text version of the same articles.

All models were train for 300K steps with 64-sized batches and the Adagrad optimizer BID5 .

For the evaluation, we use up-to 8 governing sentences as the context for a target sentence.

When a sentence has more than 8 governing sentences, we randomly choose 8 sentences.

We set the maximum number of words in a sentence to be 30 and pad each sentence with special start and end of sentence symbols.

We set η to 4, resulting in |V OOV | = 80.

Unlike most other approaches, our model and SKIP-THOUGHT BID12 can learn application-independent sentence representations without task-specific labels.

Both models are trained to predict a target sentence given context.

The prediction is a sequence of vectors representing probabilities of words in the target sentence.

For a quantitative evaluation between the two models, we compare the prediction losses by using the same loss function, namely cross entropy loss.

We randomly chose 640,000 target sentences for evaluation and computed the average loss over the 640K sentences.

We compare SKIP-THOUGHT with two versions of our model.

OURS denotes our model using the document structure-based dependencies and the OOV handler.

OURS−DEP denotes our model with the OOV handler but using only local context like SKIP-THOUGHT to show the impact of the OOV handler.

TAB3 shows the comparison of the three models.

The values in the table are the average loss per sentence.

We measure the average loss value excluding OOV words for SKIP-THOUGHT, as it cannot handle OOV words.

However, for our models, we measure the loss values with and without OOV words.

As we can see, both OURS−DEP and OURS significantly outperform SKIP-THOUGHT resulting in 25.8% and 26.9% reduction in the loss values respectively.

Further, we compare our model with SKIP-THOUGHT on a paraphrase detection task using the Microsoft Research Paraphrase corpus BID19 .

The data consists of 5,801 sentence pairs extracted from news data and their boolean assessments (if the pair of sentences are paraphrases of each other or not), which were determined by three assessors using majority voting.

The goal is correctly classifying the boolean assessments and accuracy (# correct pairs / # all pairs) is measured.

We used 4,076 pairs for training and 1,725 pairs for testing.

Since the data sets contain sentence pairs only and no structural context, we evaluate only the effectiveness of the trained encoder.

To compare the qualities of sentence embeddings by the two models, we use the same logistic regression classifier with features based on embedded sentences as in BID12 .

Given a pair of sentences S 1 and S 2 , the features are the two embeddings of S 1 and S 2 , their entry-wise absolute difference, and their entry-wise products.

Our model shows a 5% points higher accuracy than SKIP-THOUGHT in paraphrase detection (Table 3 ), demonstrating the effectiveness of our encoder trained with the structural dependencies.

Note that SKIP-THOUGHT trained on the Wikipedia corpus performs worse than a model trained on books or movie scripts due to more sophisticated and less sequential structure in Wikipedia documents.

Traditionally, the coreference resolution problem is considered as a supervised pairwise classification (i.e., mention linking) or clustering problem (coreference cluster identification) relying on an annotated corpus BID8 BID6 BID1 b; BID15 .

While, recently, there have been an impressive improvement in coreference resolution, existing coreference models are usually trained for general domain entity types (i.e., 'Person', 'Location', 'Organization') and leverage metadata that are not available in technical documents (e.g., Speaker).

D'Souza & Ng FORMULA2 and BID0 have shown that general domain coreference resolution models do not work well for domain specific entity types.

While our system is not intended to be a coreference resolution tool, the rich sentence embedding can be used for unsupervised coreference resolution allowing it applicable to any domain.

Although building a dedicated coreference resolution method to a given domain can produce better results, we claim that our approach can build a good starting set of features without supervision for a new domain.

Specifically, we treat the coreference resolution problem as an inference problem given the context.

To apply our model, we assume that entity mentions are detected in advance (any mention detection tool can be used), and, for a pronoun or a generic entity reference (e.g., a definite noun phrase), we select a list of candidate referents that conform to the mention types allowed by the pronoun or the definite noun.

We apply the mention type-based filtering to reduce the search space, but, a span-based approach as in BID15 can be used as well.

Then, we replace the entity reference with each of the candidate referents and compute the loss of the new sentence.

Finally, we choose the referent with the lowest loss value as the result, if the ratio of its loss to the original sentence loss value is less than a threshold value θ.

To show the effectiveness of the unsupervised coreference resolution method, we compare our approach with the Stanford Deep Coreference Resolution tool BID2 ) using a set of cybersecurity-related documents.

The evaluation data consists of 563 coreferences extracted from 38 Wikipedia articles about malware programs which were not included in the training document set.

We conducted experiments for several cybersecurity related entity types such as 'Malware' and 'Operating System' in addition to general entity types including 'Person' and 'Organization'.

For the evaluation, we set θ to 0.99 and 1.00.

TAB4 summarizes the results of the two systems.

Our model achieves higher precision and recall than DEEPCOREF.

Since DEEPCOREF was trained for a general domain, its overall performance on domain specific documents is very low.

FIG4 shows the two systems' performance on different entity types.

As we can see, OURS works well for domain specific entities such as 'Malware' and 'Vulnerability', while DEEPCOREF shows higher precision for 'Person' and 'Organization'.

The reason OURS performs worse for 'Person' and 'Organization' is because the security documents have only a few mentions about people or organizations, and we did not use carefully crafted features as in DEEPCOREF.

In this paper, we presented a novel sentence embedding technique exploiting diverse types of structural contexts and domain-specific OOV words.

Our method is unsupervised and applicationindependent, and it can be applied to various NLP applications.

We evaluated the method on several NLP tasks including coreference resolution, paraphrase detection and sentence prediction.

The results show that our model consistently outperforms the existing approaches confirming that considering the structural context generates better quality sentence representations.

<|TLDR|>

@highlight

To train a sentence embedding using technical documents, our approach considers document structure to find broader context and handle out-of-vocabulary words.

@highlight

Presents ideas for improving sentence embedding by drawing from more context.

@highlight

Learning sentence representations with sentences dependencies information

@highlight

Extends the idea of forming an unsupervised representation of sentences used in the SkipThough approach by using a broader set of evidence for forming the representation of a sentence