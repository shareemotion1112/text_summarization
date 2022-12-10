We study the problem of model extraction in natural language processing, in which an adversary with only query access to a victim model attempts to reconstruct a local copy of that model.

Assuming that both the adversary and victim model fine-tune a large pretrained language model such as BERT (Devlin et al., 2019), we show that the adversary does not need any real training data to successfully mount the attack.

In fact, the attacker need not even use grammatical or semantically meaningful queries: we show that random sequences of words coupled with task-specific heuristics form effective queries for model extraction on a diverse set of NLP tasks including natural language inference and question answering.

Our work thus highlights an exploit only made feasible by the shift towards transfer learning methods within the NLP community: for a query budget of a few hundred dollars, an attacker can extract a model that performs only slightly worse than the victim model.

Finally, we study two defense strategies against model extraction—membership classification and API watermarking—which while successful against some adversaries can also be circumvented by more clever ones.

Machine learning models represent valuable intellectual property: the process of gathering training data, iterating over model design, and tuning hyperparameters costs considerable money and effort.

As such, these models are often only indirectly accessible through web APIs that allow users to query a model but not inspect its parameters.

Malicious users might try to sidestep the expensive model development cycle by instead locally reproducing an existing model served by such an API.

In these attacks, known as "model stealing" or "model extraction" (Lowd & Meek, 2005; Tramèr et al., 2016) , the adversary issues a large number of queries and uses the collected (input, output) pairs to train a local copy of the model.

Besides theft of intellectual property, extracted models may leak sensitive information about the training data (Tramèr et al., 2016) or be used to generate adversarial examples that evade the model served by the API (Papernot et al., 2017) .

With the recent success of contextualized pretrained representations for transfer learning, NLP APIs based on ELMo (Peters et al., 2018) and BERT (Devlin et al., 2019) have become increasingly popular (Gardner et al., 2018) .

Contextualized pretrained representations boost performance and reduce sample complexity (Yogatama et al., 2019) , and they typically require only a shallow task-specific network-sometimes just a single layer as in BERT.

While these properties are advantageous for representation learning, we hypothesize that they also make model extraction easier.

In this paper, we demonstrate that NLP models obtained by fine-tuning a pretrained BERT model can be extracted even if the adversary does not have access to any training data used by the API provider.

In fact, the adversary does not even need to issue well-formed queries: our experiments show that extraction attacks are possible even with queries consisting of randomly sampled sequences of words coupled with simple task-specific heuristics (Section 3).

This result contrasts with prior work, which for large-scale attacks requires at minimum that the adversary can access a small amount of semantically-coherent data relevant to the task (Papernot et al., 2017; Correia-Silva et al., 2018; Orekondy et al., 2019a; Pal et al., 2019; Jagielski et al., 2019) .

Extraction performance improves further by using randomly-sampled sentences and paragraphs from Wikipedia (instead of random word sequences) as queries (Section 4).

These attacks are cheap; our most expensive attack cost around $500, estimated using rates of current API providers.

Step 1: Attacker randomly samples words to form queries and sends them to victim BERT model

Step 2: Attacker fine-tunes their own BERT on these queries using the victim outputs as labels Figure 1 : Overview of our model extraction setup for question answering.

1 An attacker first queries a victim BERT model, and then uses its predicted answers to fine-tune their own BERT model.

This process works even when passages and questions are random sequences of words as shown here.

We perform a fine-grained analysis of the randomly-generated queries to shed light on why they work so well for model extraction.

Human studies on the random queries show that despite their effectiveness in extracting good models, they are mostly nonsensical and uninterpretable, although queries closer to the original data distribution seem to work better for extraction (Section 5.1).

Furthermore, we discover that pretraining on the attacker's side makes model extraction easier (Section 5.2).

Finally, we study the efficacy of two simple defenses against extraction -membership classification (Section 6.1) and API watermarking (Section 6.2) -and find that while they work well against naïve adversaries, they fail against more clever ones.

We hope that our work spurs future research into stronger defenses against model extraction and, more generally, on developing a better understanding of why these models and datasets are particularly vulnerable to such attacks.

We relate our work to prior efforts on model extraction, most of which have focused on computer vision applications.

Because of the way in which we synthesize queries for extracting models, our work also directly relates to zero-shot distillation and studies of rubbish inputs to NLP systems.

Model extraction attacks have been studied both empirically (Tramèr et al., 2016; Orekondy et al., 2019a; Juuti et al., 2019) and theoretically (Chandrasekaran et al., 2018; Milli et al., 2019) , mostly against image classification APIs.

These works generally synthesize queries in an active learning setup by searching for inputs that lie close to the victim classifier's decision boundaries.

This method does not transfer to text-based systems due to the discrete nature of the input space.

2 The only prior work attempting extraction on NLP systems is Pal et al. (2019) , who adopt pool-based active learning to select natural sentences from WikiText-2 and extract 1-layer CNNs for tasks expecting single inputs.

In contrast, we study a more realistic extraction setting with nonsensical inputs on modern BERT-large models for tasks expecting pairwise inputs like question answering.

Our work is related to prior work on data-efficient distillation, which attempts to distill knowledge from a larger model to a small model with access to limited input data (Li et al., 2018) or in a zeroshot setting (Micaelli & Storkey, 2019; Nayak et al., 2019) .

However, unlike the model extraction setting, these methods assume white-box access to the teacher model to generate data impressions.

Rubbish inputs, which are randomly-generated examples that yield high-confidence predictions, have received some attention in the model extraction literature.

Prior work (Tramèr et al., 2016) reports successful extraction on SVMs and 1-layer networks using i.i.d noise, but no prior work has scaled this idea to deeper neural networks for which a single class tends to dominate model predictions on most noise inputs (Micaelli & Storkey, 2019; Pal et al., 2019) .

Unnatural text inputs have previously been shown to produce overly confident model predictions (Feng et al., 2018) , break translation systems (Belinkov & Bisk, 2018) , and trigger disturbing outputs from text generators (Wallace et al., 2019) .

In contrast, here we show their effectiveness at training models that work well on real NLP tasks despite not seeing any real examples during training.

What is BERT?

We study model extraction on BERT, Bidirectional Encoder Representations from Transformers (Devlin et al., 2019) .

BERT-large is a 24-layer transformer (Vaswani et al., 2017) , f bert,θ , which converts a word sequence x = (x 1 , ..., x n ) of length n into a high-quality sequence of vector representations v = (v 1 , ..., v n ).

These representations are contextualized -every vector v i is conditioned on the whole sequence x. BERT's parameters θ * are learnt using masked language modelling on a large unlabelled corpus of natural text.

The public release of f bert,θ * revolutionized NLP, as it achieved state-of-the-art performance on a wide variety of NLP tasks with minimal task-specific supervision.

A modern NLP system for task T typically leverages the fine-tuning methodology in the public BERT repository: 3 a task-specific network f T,φ (generally, a 1-layer feedforward network) with parameters φ expecting v as input is used to construct a composite function g T = f T,φ • f bert,θ .

The final parameters φ T , θ T are learned end-to-end using training data for T with a small learning rate ("fine-tuning"), with φ initialized randomly and θ initialized with θ * .

Description of extraction attacks: Assume g T (the "victim model") is a commercially available black-box API for task T .

A malicious user with black-box query access to g T attempts to reconstruct a local copy g T (the "extracted model").

Since the attacker does not have training data for T , they use a task-specific query generator to construct several possibly nonsensical word sequences {x i } m 1 as queries to the victim model.

The resulting dataset {x i , g T (x i )} m 1 is used to train g T .

Specifically, we assume that the attacker fine-tunes the public release of f bert,θ * on this dataset to obtain g T .

4 A schematic of our extraction attacks is shown in Figure 1 .

We extract models on four diverse NLP tasks that have different kinds of input and output spaces: (1) binary sentiment classification using SST2 (Socher et al., 2013) , where the input is a single sentence and the output is a probability distribution between positive and negative; (2) ternary natural language inference (NLI) classification using MNLI (Williams et al., 2018) , where the input is a pair of sentences and the output is a distribution between entailment, contradiction and neutral; (3) extractive question answering (QA) using SQuAD 1.1 (Rajpurkar et al., 2016) , where the input is a paragraph and question and the output is an answer span from the paragraph; and (4) boolean question answering using BoolQ (Clark et al., 2019) , where the input is a paragraph and question and the output is a distribution between yes and no.

Query generators: We study two kinds of query generators, RANDOM and WIKI.

In the RANDOM generator, an input query is a nonsensical sequence of words constructed by sampling 5 a Wikipedia vocabulary built from WikiText-103 (Merity et al., 2017) .

In the WIKI setting, input queries are formed from actual sentences or paragraphs from the WikiText-103 corpus.

We found these generators insufficient by themselves to extract models for tasks featuring complex interactions between different parts of the input space (e.g., between premise and hypothesis in MNLI or question and paragraph in SQuAD).

Hence, we additionally apply the following task-specific heuristics:

• MNLI: since the premise and hypothesis often share many words, we randomly replace three words in the premise with three random words to construct the hypothesis.

• SQuAD / BoolQ: since questions often contain words in the associated passage, we uniformly sample words from the passage to form a question.

We additionally prepend a question starter word (like "what") to the question and append a ? symbol to the end.

For more details on the query generation, see Appendix A.3.

Representative example queries and their outputs are shown in Table 1 .

More examples are provided in Appendix A.5.

First, we evaluate our extraction procedure in a controlled setting where an attacker uses an identical number of queries as the original training dataset (Table 2 ); afterwards, we investigate different query budgets for each task (Table 3) .

We provide commercial cost estimates for these query budgets using the Google Cloud Platform's Natural Language API calculator.

6 We use two metrics for eval- Table 3 : Development set accuracy of various extracted models on the original development set at different query budgets expressed as fractions of the original dataset size.

Note the high accuracies for some tasks even at low query budgets, and diminishing accuracy gains at higher budgets.

uation: Accuracy of the extracted models on the original development set, and Agreement between the outputs of the extracted model and the victim model on the original development set inputs.

In our controlled setting (Table 2) , our extracted models are surprisingly accurate on the original development sets of all tasks, even when trained with nonsensical inputs (RANDOM) that do not match the original data distribution.

7 Accuracy improves further on WIKI: extracted SQuAD models recover 95% of original accuracy despite seeing only nonsensical questions during training.

While extracted models have high accuracy, their agreement is only slightly better than accuracy in most cases.

Agreement is even lower on held-out sets constructed using the WIKI and RANDOM sampling scheme.

On SQuAD, extracted WIKI and RANDOM have low agreements of 59.2 F1 and 50.5 F1 despite being trained on identically distributed data.

This indicates poor functional equivalence between the victim and extracted model as also found by Jagielski et al. (2019) .

An ablation study with alternative query generation heuristics for SQuAD and MNLI is conducted in Appendix A.4.

Classification with argmax labels only: For classification datasets, we assumed the API returns a probability distribution over output classes.

This information may not be available to the adversary in practice.

To measure what happens when the API only provides argmax outputs, we re-run our WIKI experiments for SST2, MNLI and BoolQ with argmax labels and present our results in Table 2 (WIKI-ARGMAX).

We notice a minimal drop in accuracy from the corresponding WIKI experiments, indicating that access to the output probability distribution is not crucial for model extraction.

Hence, hiding the full probability distribution is not a viable defense strategy.

Query efficiency: We measure the effectiveness of our extraction algorithms with varying query budgets, each a different fraction of the original dataset size, in Table 3 .

Even with small query budgets, extraction is often successful; while more queries is usually better, accuracy gains quickly diminish.

Approximate costs for these attacks can be extrapolated from Table 2 .

These results bring many natural questions to mind.

What properties of nonsensical input queries make them so amenable to the model extraction process?

How well does extraction work for these tasks without using large pretrained language models?

In this section, we perform an analysis to answer these questions.

Previously, we observed that nonsensical input queries are surprisingly effective for extracting NLP models based on BERT.

Here, we dig into the properties of these queries in an attempt to understand why models trained on them perform so well.

Do different victim models produce the same answer when given a nonsensical query?

Are some of these queries more representative of the original data distribution than others?

Did our task-specific heuristics perhaps make these nonsensical queries "interpretable" to humans in some way?

We specifically examine the RANDOM and WIKI extraction configurations for SQuAD in this section to answer these questions.

Do different victim models agree on the answers to nonsensical queries?

We train five victim SQuAD models on the original training data with identical hyperparameters, varying only the random seed; each achieves an F1 of between 90 and 90.5.

Then, we measure the average pairwise F1 ("agreement") between the answers produced by these models for different types of queries.

As expected, the models agree very frequently when queries come from the SQuAD training set (96.9 F1) or development set (90.4 F1).

However, their agreement drops significantly on WIKI queries (53.0 F1) and even further on RANDOM queries (41.2 F1).

Are high-agreement queries closer to the original data distribution?

While these results indicate that on average, victim models tend to be brittle on nonsensical inputs, it is possible that highagreement queries are more useful than others for model extraction.

To measure this, we sort queries from our 10x RANDOM and WIKI datasets according to their agreement and choose the highest and lowest agreement subsets, where subset size is a varying fraction of the original training data size ( Figure 2 ).

We observe large F1 improvements when extracting models using high-agreement subsets, consistently beating random and low-agreement subsets of identical sizes.

This result shows that agreement between victim models is a good proxy for the quality of an input-output pair for extraction.

Measuring this agreement in extracted models and integrating this observation into an active learning objective for better extraction is an interesting direction that we leave to future work.

Are high-agreement nonsensical queries interpretable to humans?

Prior work (Xu et al., 2016; Ilyas et al., 2019) has shown deep neural networks can leverage non-robust, uninterpretable features to learn classifiers.

Our nonsensical queries are not completely random, as we do apply task-specific heuristics.

Perhaps as a result of these heuristics, do high-agreement nonsensical textual inputs have a human interpretation?

To investigate, we asked three human annotators 9 to answer twenty SQuAD questions from each of the WIKI and RANDOM subsets that had unanimous agreement among victim models, and twenty original SQuAD questions as a control.

On the WIKI subset, annotators matched the victim models' answer exactly 23% of the time (33 F1).

Similarly, a 22% exact match (32 F1) was observed on RANDOM.

In contrast, annotators scored significantly higher on original SQuAD questions (77% exact match, 85 F1 against original answers).

Interviews with the annotators revealed a common trend: annotators used a word overlap heuristic (between the question and paragraph) to select entities as answer spans.

While this heuristic partially interprets the extraction data's signal, most of the nonsensical question-answer pairs remain mysterious to humans.

More details on inter-annotator agreement are provided in Appendix A.6.

So far we assumed that the victim and the attacker both fine-tune a pretrained BERT-large model.

However, in practical scenarios, the attacker might not have information about the victim architecture.

What happens when the attacker fine-tunes a different base model than the victim?

What if the attacker extracts a QA model from scratch instead of fine-tuning a large pretrained language model?

Here, we examine how much the extraction accuracy depends on the pretraining setup.

BERT comes in two different sizes: the 24 layer BERT-large and the 12 layer BERT-base.

In Table 4 , we measure the development set accuracy on MNLI and SQuAD when the victim and attacker use different configurations of these two models.

Accuracy is always higher when the attacker starts from BERT-large, even when the victim was initialized with BERT-base.

Additionally, given a fixed attacker architecture, accuracy is better when the victim uses the same model (e.g., if the attacker starts from BERT-base, they will have better results if the victim also used BERT-base).

This is reminiscent of similar discussion in Tramèr et al. (2016) What if we train from scratch?

Finetuning BERT seems to give attackers a significant headstart, as only the final layer of the model is randomly initialized and the BERT parameters start from a good representation of language.

To measure the importance of fine-tuning from a good starting point, we train a QANet model (Yu et al., 2018) on SQuAD with no contextualized pretraining.

This model has 1.3 million randomly initialized parameters at the start of training.

Table 5 shows that QANet achieves high accuracy when original SQuAD inputs are used (TRUE X) with BERT-large labels, indicating sufficient model capacity.

However, the F1 significantly degrades when training on nonsensical RANDOM and WIKI queries.

The F1 drop is particularly striking when compared to the corresponding rows in Table 2 (only 4.5 F1 drop for WIKI).

This reinforces our finding that better pretraining allows models to start from a good representation of language, thus simplifying extraction.

Having established that BERT-based models are vulnerable to model extraction, we now shift our focus to investigating defense strategies.

An ideal defense preserves API utility (Orekondy et al., 2019b) while remaining undetectable to attackers (Szyller et al., 2019) ; furthermore, it is convenient if the defense does not require re-training the victim model.

Here we explore two defenses that satisfy these properties but are also only effective against a class of weak adversaries.

Our first defense uses membership inference, which determines whether a classifier was trained on a particular input point (Shokri et al., 2017; Nasr et al., 2018) , to identify nonsensical out-ofdistribution inputs or adversarial examples (Szegedy et al., 2014; Papernot & McDaniel, 2018) that are unlikely to be issued by a legitimate user.

When such inputs are detected, the API issues a random output instead of the model's predicted output, which eliminates the extraction signal.

We treat membership inference as a binary classification problem, constructing datasets for MNLI and SQuAD by labeling their original training and validation examples as real and WIKI extraction examples as fake.

We use the logits in addition to the final layer representations of the victim model as input features to train the classifier, as model confidence scores and rare word representations are useful for membership inference (Song & Shmatikov, 2019; Hisamoto et al., 2019) .

Table 6 shows that these classifiers transfer well to a balanced development set with the same distribution as their training data (WIKI).

They are also robust to the query generation process: accuracy remains high on auxiliary test sets where fake examples are either RANDOM (described in Section 3) or SHUFFLE, in which the word order of real examples is shuffled.

An ablation study on the input features of the classifier is provided in Appendix A.7.

Another defense against extraction is watermarking (Szyller et al., 2019), in which a tiny fraction of queries are chosen at random and modified to return a wrong output.

These "watermarked queries" and their outputs are stored on the API side.

Since deep neural networks have the ability to memorize arbitrary information (Zhang et al., 2017; , this defense anticipates that extracted models will memorize some of the watermarked queries, leaving them vulnerable to post-hoc detection if they are deployed publicly.

We evaluate watermarking on MNLI (by randomly permuting the predicted probability vector to ensure a different argmax output) and SQuAD (by returning a Table 8 : Results on watermarked models.

Dev Acc represents the overall development set accuracy, WM Label Acc denotes the accuracy of predicting the watermarked output on the watermarked queries and Victim Label Acc denotes the accuracy of predicting the original labels on the watermarked queries.

A watermarked WIKI has high WM Label Acc and low Victim Label Acc.

single word answer which has less than 0.2 F1 overlap with the actual output).

For both tasks, we watermark just 0.1% of all queries to minimize the overall drop in API performance.

Table 8 shows that extracted models perform nearly identically on the development set (Dev Acc) with or without watermarking.

When looking at the watermarked subset of the training data, however, non-watermarked models get nearly everything wrong (low WM Label Acc%) as they generally predict the victim model's outputs (high Victim Label Acc%), while watermarked models behave oppositely.

Training with more epochs only makes these differences more drastic.

Limitations: Watermarking works, but it can only be used after an attack has been carried out.

Importantly, it assumes that an attacker will deploy an extracted model publicly with black-box query access and is thus irrelevant if the attacker instead keeps the model private.

Furthermore, an attacker who anticipates watermarking might take steps to prevent detection, including (1) differentially private training on extraction data (Dwork et al., 2014; Abadi et al., 2016) ; (2) fine-tuning or re-extracting an extracted model with different queries (Chen et al., 2019; Szyller et al., 2019) ; or (3) issuing random outputs on queries exactly matching inputs in the extraction data.

We study model extraction attacks against NLP APIs that serve BERT-based models.

These attacks are surprisingly effective at extracting good models with low query budgets, even when an attacker uses nonsensical input queries.

Our results show that fine-tuning large pretrained language models simplifies the process of extraction for an attacker.

Unfortunately, existing defenses against extraction, while effective in some scenarios, are generally inadequate, and further research is necessary to develop defenses robust in the face of adaptive adversaries who develop counter-attacks anticipating simple defenses.

Other interesting future directions that follow from the results in this paper include 1) leveraging nonsensical inputs to improve model distillation on tasks for which it is difficult to procure input data; 2) diagnosing dataset complexity by using query efficiency as a proxy; 3) further investigation of the agreement between victim models as a method to identify proximity in input distribution and its incorporation into an active learning setup for model extraction.

We provide a distribution of agreement between victim SQuAD models on RANDOM and WIKI queries in Figure 3 .

In this paper, we have used the cost estimate from Google Cloud Platform's Calculator.

10 The Natural Language APIs typically allows inputs of length up to 1000 characters per query (https: //cloud.google.com/natural-language/pricing).

To calculate costs for different datasets, we counted input instances with more than 1000 characters multiple times.

Since Google Cloud did not have APIs for all tasks we study in this paper, we extrapolated the costs of the entity analysis and sentiment analysis APIs for natural language inference (MNLI) and reading comprehension (SQuAD, BoolQ).

We believe this is a reasonable estimate since every model studied in this paper is a single layer in addition to BERT-large (thereby needing a similar number of FLOPs for similar input lengths).

It is hard to provide a widely applicable estimate for the price of issuing a certain number of queries.

Several API providers allow a small budget of free queries.

An attacker could conceivably set up multiple accounts and collect extraction data in a distributed fashion.

In addition, most APIs are implicitly used on webpages -they are freely available to web users (such as Google Search or Maps).

If sufficient precautions are not taken, an attacker could easily emulate the HTTP requests used to call these APIs and extract information at a large scale, free of cost ("web scraping").

Besides these factors, API costs could also vary significantly depending on the computing infrastructure involved or the revenue model of the company deploying them.

Given these caveats, it is important to focus on the relatively low costs needed to extract datasets rather than the actual cost estimates.

Even complex text generation tasks like machine translation and speech recognition (for which Google Cloud has actual API estimates) are relatively inexpensive.

It costs -$430.56 to extract Switchboard LDC97S62 (Godfrey et al., 1992) , a large conversational speech recognition dataset with 300 hours of speech; $2000.00 to issue 1 million translation queries, each having a length of 100 characters.

In this section we provide more details on the input generation algorithms adopted for each dataset.

(SST2, RANDOM) -A vocabulary is built using wikitext103.

The top 10000 tokens (in terms of unigram frequency in wikitext103) are preserved while the others are discarded.

A length is chosen from the pool of wikitext-103 sentence lengths.

Tokens are uniformly randomly sampled from the top-10000 wikitext103 vocabulary up to the chosen length.

(SST2, WIKI) -A vocabulary is built using wikitext103.

The top 10000 tokens (in terms of unigram frequency in wikitext103) are preserved while the others are discarded.

A sentence is chosen at random from wikitext103.

Words in the sentence which do not belong to the top-10000 wikitext103 vocabulary are replaced with words uniformly randomly chosen from this vocabulary.

(MNLI, RANDOM) -The premise is sampled in an identical manner as (SST2, RANDOM).

To construct the final hypothesis, the following process is repeated three times -i) choose a word uniformly at random from the premise ii) replace this word with another word uniformly randomly sampled from the top-10000 wikitext103 vocabulary.

(MNLI, WIKI) -The premise is sampled in a manner identical to (SST2, WIKI).

The hypothesis is sampled in a manner identical (MNLI, RANDOM).

(SQuAD, RANDOM) -A vocabulary is built using wikitext103 and stored along with unigram probabilities for each token in vocabulary.

A length is chosen from the pool of paragraph lengths in wikitext103.

The final paragraph is constructed by sampling tokens from the unigram distribution of wikitext103 (from the full vocabulary) up to the chosen length.

Next, a random integer length is chosen from the range [5, 15] .

Paragraph tokens are uniformly randomly sampled to up to the chosen length to build the question.

Once sampled, the question is appended with a ? symbol and prepended with a question starter word chosen uniformly randomly from the list [A, According, After, Along, At, By, During, For, From, How, In, On, The, To, What, What's, When, Where, Which, Who, Whose, Why].

(SQuAD, WIKI) -A paragraph is chosen at random from wikitext103.

Questions are sampled in a manner identical to (SQuAD, RANDOM).

(BoolQ, RANDOM) -identical to (SQuAD, RANDOM).

We avoid appending questions with ?

since they were absent in BoolQ. Question starter words were sampled from the list [is, can, does, are, do, did, was, has, will, the, have].

(BoolQ, WIKI) -identical to (SQuAD, WIKI).

We avoid appending questions with ? since they were absent in BoolQ. The question starter word list is identical to (BoolQ, RANDOM).

In this section we study some additional query generation heuristics.

In Table 11 , we compare numerous extraction datasets we tried for SQuAD 1.1.

Our general findings are -i) RANDOM works much better when the paragraphs are sampled from a distribution reflecting the unigram frequency in wikitext103 compared to uniform random sampling ii) starting questions with common question starter words like "what" helps, especially with RANDOM schemes.

We present a similar ablation study on MNLI in Table 12 .

Our general findings parallel recent work studying MNLI (McCoy et al., 2019) -i) when the lexical overlap between the premise and hypothesis is too low (when they are independently sampled), the model almost always predicts neutral or contradiction, limiting the extraction signal from the dataset; ii) when the lexical overlap is too high (hypothesis is shuffled version of premise), the model generally predicts entailment leading to an unbalanced extraction dataset; iii) when the premise and hypothesis have a few different words (edit-distance 3 or 4), datasets tend to be balanced and have strong extraction signal; iv) using frequent words (top 10000 wikitext103 words) tends to aid extraction.

More examples have been provided in Table 13 .

For our human studies, we asked fifteen human annotators to annotate five sets of twenty questions.

Annotators were English-speaking graduate students who voluntarily agreed to participate and were completely unfamiliar with our research goals.

Three annotators were used per question set.

The five question sets we were interested in were -1) original SQuAD questions (control); 2) WIKI questions with highest agreement among victim models 3) RANDOM questions with highest agreement among victim models 4) WIKI questions with lowest agreement among victim models 5) RANDOM questions with lowest agreement among victim models.

In Table 10 we show the inter-annotator agreement.

Notice that average pairwise F1 (a measure of inter-annotator agreement) follows the order original SQuAD >> WIKI, highest agreement > RANDOM, highest agreement ∼ WIKI, lowest agreement > RANDOM, lowest agreement.

We hypothesize that this ordering roughly reflects the closeness to the actual input distribution, since a similar ordering is also observed in Figure 2 .

Individual annotation scores have been shown below.

In this section we run an ablation study on the input features for the membership classifier.

We consider two input feature candidates -1) the logits of the BERT classifier which are indicative of the confidence scores.

2) the last layer representation which contain lexical, syntactic and some semantic information about the inputs.

We present our results in Table 9 .

Our ablation study indicates that the last layer representations are more effective than the logits in distinguishing between real and fake inputs.

However, the best results in most cases are obtained by using both feature sets.

Table 9 : Ablation study of the membership classifiers.

We measure accuracy on an identically distributed development set (WIKI) and differently distributed test sets (RANDOM, SHUFFLE).

Note the last layer representations tend to be more effective in classifying points as real or fake.

<|TLDR|>

@highlight

Outputs of modern NLP APIs on nonsensical text provide strong signals about model internals, allowing adversaries to steal the APIs.