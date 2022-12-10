In contrast to the older writing system of the 19th century, modern Hawaiian orthography employs characters for long vowels and glottal stops.

These extra characters account for about one-third of the phonemes in Hawaiian, so including them makes a big difference to reading comprehension and pronunciation.

However, transliterating between older and newer texts is a laborious task when performed manually.

We introduce two related methods to help solve this transliteration problem automatically, given that there were not enough data to train an end-to-end deep learning model.

One approach is implemented, end-to-end, using finite state transducers (FSTs).

The other is a hybrid deep learning approach which approximately composes an FST with a recurrent neural network (RNN).

We find that the hybrid approach outperforms the end-to-end FST by partitioning the original problem into one part that can be modelled by hand, using an FST, and into another part, which is easily solved by an RNN trained on the available data.

From 1834 to 1948, more than 125,000 newspaper pages were published in the Hawaiian language (Nogelmeier, 2010 ).

Yet by 1981, many expected this once flourishing language to die BID0 .

Hawaiian has since defied expectations and experienced the beginnings of a remarkable recovery (Warner, 2001; Wilson and Kamanā, 2001) .

However much of the literary inheritance that is contained in the newspapers has become difficult for modern Hawaiians to read, since the newspapers were written in an orthography that failed to represent about one-third of the language's phonemes.

This orthography, which we will refer to as the missionary orthography, excluded Hawaiian phonemes that did not have equivalents in American English (see Schütz, 1994) , namely long vowels /i: e: a: o: u:/ and glottal stop /P/. By contrast, the modern Hawaiian orthography, an innovation of Pukui and Elbert's Hawaiian dictionary (Pukui and Elbert, 1957) , presents a nearly perfect, one-to-one mapping between graphemes and phonemes.

The process of manual transliteration from missionary to modern Hawaiian orthography is extremely labor intensive.

Yet the cultural benefits are so great that hundreds of pages of newspaper-serials have already been transliterated by hand, such as Nogelmeier's new edition of the epic tale of Hi'iakaikapoliopele, the volcano goddess's sister BID6 .

Critically important as such efforts are to the continued revitalization of this endangered language, they are still only a small sample of the material that could be made available to a modern Hawaiian audience.

In this paper, we propose to automate, or semi-automate, the transliteration of old Hawaiian texts into the modern orthography.

Following a brief review of related work (Section 2), we begin by describing a dataset of modern Hawaiian (Section 3).

In Section 4, we present two methods for recovering missing graphemes (and hence phonemes) from the missionary orthography.

The first composes a series of weighted FSTs; the second approximately composes an FST with a recurrent neural network language model (RNNLM) using a beam search procedure.

Both approaches require only modern Hawaiian texts for training, which are much more plentiful than parallel corpora.

Section 5 reports the results of our transliteration experiments using a simulated parallel corpus, as well as two 19th century newspaper articles for which we also have modern Hawaiian transcriptions.

Being based on FSTs, both approaches are modular and extensible.

We observe useful and promising results for both of our methods, with the best results obtained by the hybrid FST-RNNLM.

These results showcase the strength of combining established hand-engineering methods with deep learning in a smaller data regime, with practical applications for an endangered language.

Many of the themes that we address relate to existing literature.

For example, BID4 and Scannell (2014) have written on machine translation (MT) for closely related languages and on multilingual text normalization, respectively.

Though language-relatedness makes MT easier BID10 , state-of-the-art techniques such as neural machine translation (NMT) have not performed well for languages with little data (Östling and Tiedemann, 2017) .

So while the Hawaiian transliteration problem could be cast as an instance of MT or of NMT, we chose to sidestep the scarcity of parallel data by not considering such approaches.

Hybrid approaches, which combine expert knowledge for well-understood structures with deep learning for data-plentiful subproblems, offer rich opportunities for data-efficient modelling.

In particular, prior work has explored ways to combine FSTs and RNNs.

For example, Sproat and Jaitly (2016) used an FST to restrict the search space when decoding from an RNN; Rastogi et al. (2016) incorporated RNN information into an FST.

However, these approaches differ from the approximate FST-to-RNN composition algorithm that we introduce here (in Section 4.2).

Ignoring case, there is a neat mapping between the modern Hawaiian orthography and the Hawaiian phonemic inventory.

The phonemic inventory contains eight consonants /h k l m n p v P/ and ten vowels, of which five are short /a e i o u/ and five are long /a: e: i: o: u:/ (Parker Jones, 2018).The consonants map onto the orthographic symbols H h K k L l M m N n P p W w ' , where we give the upper-and lower-case variants in adjacent pairs: H h for /h/, K k for /k/, . . .

, W w for /v/. An exception, the symbol ' has only one variant which maps to /P/. The vowels map onto the symbols: A a E e I i O o U uĀāĒēĪīŌōŪū .

Note that vowel length is denoted by the absence or presence of a macron (e.g. A and a map onto short /a/ and Ā and ā map onto long /a:/).The Hawaiian conventions for capitalization, numbering, and punctuation are analogous to those in English, except again there is no upper-case variant of ' , so the following vowel is capitalized instead (e.g. 'Okakopa 'October').

In foreign words, such as kolorofolorokalapona 'chlorofluorocarbon', one can find the additional consonants: DISPLAYFORM0

The primary difference between the missionary and modern Hawaiian orthographies is that the missionary orthography does not encode long vowels or the glottal stop.

For example, the following Hawaiian phrases were recorded by a 19th-century German traveller in the missionary orthography: Ua oia au, E ue ae oe ia Ii, E ao ae oe ia ia (Chamisso, 1837, p. 7).

In the modern orthography these become: Ua 'ō 'ia au 'I am speared', E uē a'e 'oe iā 'Ī'ī 'You must weep for 'Ī'ī (a person)', and E a'o a'e 'oe iā ia 'You teach him ' (Elbert and Pukui, 1979, p. 3).We can convert text in the modern Hawaiian orthography backward chronologically to an approximate missionary orthography by mapping each glottal stop ' to the empty string , and each long vowel, e.g. āēīōū , to its corresponding short vowel, a e i o u .

As a first approximation, we may treat mappings from the modern-to-missionary orthographies as unambiguously many-to-one; thus there is information loss.

We will return to secondary differences between the orthographies in Section 6.

To illustrate, the following four words in the modern orthography all map to the same missionary string aa: a'a (root), 'a'a (brave), 'a'ā (crumbly lava rock), and 'ā'ā (stutter).The forward mapping from missionary-to-modern orthographies is one-to-many.

Thus the missionary string aa could map to a'a, 'a'a, 'a'ā, or 'ā'ā.

The transliteration problem we address here seeks to discover how we can use context to recover the information not present in the missionary orthography that modern Hawaiian orthography retains.

We draw on three sources for modern Hawaiian text: the main text of Hi'iakaikapoliopele BID6 , 160 short texts from Ulukau:

The Hawaiian Electronic Library, and the full Hawaiian Wikipedia (see TAB1 ).

For evaluation, we simulate a missionary-era version of the modern texts using the backward mapping described above.

In addition, we evaluated our models on a couple of 19th century newspaper samples for which we have parallel missionary-era and modern text.

Both simulated and real parallel corpora will be described in Section 5.

We can frame the task of transliterating from missionary-to-modern Hawaiian orthographies as a sequence transduction problem.

Many deep learning approaches (e.g. Sutskever et al., 2014; BID3 are not easily applicable to this task since we do not have a sufficiently large dataset of parallel texts.

Instead, we focus on approaches that mix hand-designed FSTs with trained language models, including deep learning approaches like RNNLMs (Mikolov et al., 2010) .

Our initial approach represents the mapping from missionary to modern orthography using a composition of (weighted) FSTs; for a thorough review, see Mohri (1997) .First, we construct a finite state acceptor, I, from the input text.

Here we construct a trivial chainshaped acceptor that accepts only the input text.

Each symbol in the input text is represented by a state which emits this symbol on a single transition that moves to the next state.

The transition emitting the final symbol in the string leads to the sole accepting state.

Second, we construct an FST called C, which models potential orthography changes that can occur when transliterating from the missionary to modern Hawaiian orthography (see Figure 1 ).

For example, two non-deterministic transitions introduce an optional long-vowel map for a : (a : a) and (a :ā).

Another transition inserts glottal stops: ( : ').

By capturing the orthographic changes we know to occur, the composition I • C produces a large set of candidates to be narrowed using the language model.

Third, we use the modern Hawaiian text from Section 3.3 to construct and evaluate a number of character-level n-gram language models, of various combinations of order and Katz backoff and Kneser-Ney (KN) smoothing BID8 BID9 (see Section 5.1 for a list of models that we trained).

N-gram language models can be expressed as weighted FSTs.

We denote the n-gram or weighted FST language model as G. Character-level models are used as we wanted to generalize to out-of-vocabulary words, which we expect to occur frequently in a relatively small corpus like the one we have for Hawaiian.

Finally, we use this model to infer modern orthography given a piece of text in missionary orthography as input, then compose the FSTs to form the search graph FST: S = I • C • G. The minimum cost path through S gives the predicted modern orthography.

Of these n-gram-based approaches, we found the Kneser-Ney-based models to perform best.

These approaches will be referred to as FST-C-NGRAM-KN and FST-C wb -NGRAM-KN.We circumvent the lack of a large, non-simulated parallel corpus by training the language model exclusively on text in the modern Hawaiian orthography.

In turn, the orthographic transliteration FST C produces candidates which are disambiguated by the language model.

The result is finally evaluated against the ground-truth modern text.

Although the orthographic transliteration model is an approximation, and thus not exhaustive, it embodies an explicit and interpretable representation that can be extended independently of the rest of the model.

To illustrate this, we constructed a variant C wb (where wb stands for word boundary).

C wb optionally inserts a space after each vowel using an additional arc that maps ( : space) (again see Figure 1 ).

This variant is able to model some changes in Hawaiian's word-boundary conventions (Wilson, 1976) , such as alaila becoming a laila which demarcates the preposition a 'until' and noun laila 'then'.

We report on the use of C wb to predict modern equivalents of 19th century newspaper samples in Section 5.

An example prediction can be found in Section 6, with more in Appendix A.Figure 1: Two FSTs.

The first, C, transduces between missionary and modern orthographies.

The second, C wb , introduces optional spaces (or word boundaries) after a vowel.

In each FST, s 1 serves as both the initial and end state, while labelled arrows denote arcs.

In the labelled transitions, V and V are variables for short and long vowels, respectively, and C can be any consonant other than ' .

Some arcs, for upper-case letters, numbers, and punctuation, have been omitted for brevity.

As an alternative approach, we tried combining the FST C in the previous section with an RNNLM, since RNNLMs often generalize better than n-gram language models (Mikolov et al., 2010 ).An RNN is a neural network that models temporal or sequential data, by iterating a function mapping a state and input to a new state and output.

These can be stacked to form a deep RNN.

For language modelling, each step of the final RNN layer models a word or character sequence via p(w 1 , . . . ,w n ) = n i=1 p(w i |w 1:i−1 ) and can be trained by maximum likelihood.

Recent language modeling work has typically used the long short-term memory (LSTM) unit, because of its favorable gradient propagation properties BID5 .

All RNNs in this paper are LSTMs.

Our goal in the hybrid approach is to replace the n-gram language model in the end-to-end FST with an RNNLM.

While the minimum cost path through an FST can be computed exactly as done in the previous section, it is not straightforward to compose the relation defined by an FST with an arbitrary one like that defined by an RNNLM.

Nonetheless, a minimum cost path through the composition of the FST and the RNNLM can be defined as a path (i.e. label sequence) that minimizes the sum of the FST path cost and the RNNLM cost.

We can approximately find a minimum cost path for the composition of the two models by a breadthfirst search over the FST graph, using a beam search, as follows.

At any particular iteration, consider a single beam element.

The beam element holds the current FST and RNN states, and the path taken through the FST so far.

We follow each possible arc from the current FST state, each producing a new child beam element, and feed the output symbol into the RNN (unless it is ).

We note that there may be duplicate beam elements due the nondeterminicity of the FST, in which case the lower cost edge wins.

We sort by the sum of the FST and RNN costs, keep the lowest-cost K, and then proceed to the next iteration.

If a beam element is on an accepting state of the FST, it is kept as-is between iterations.

Detailed pseudocode is provided in FIG0 .

Conceptually, this algorithm performs the same operation as the n-gram language model, except that we replace the n-gram language model with an RNN language model, and then we search over the FST graph, producing scores from FST weights and RNN's outputs.

Incidentally, we note that our implementation of the algorithm is slightly different than how we present it in the pseudocode, as we grouped RNN operations into batches for computational efficiency.

In this paper, we will refer to the hybrid models as FST-RNNLM in general, and as FST-RNNLM-C and FST-RNNLM-C wb if we want to distinguish which FST was used.

The end-to-end FST models will similarly be referred to as FST-C and FST-C wb , with suffixes denoting what kind of n-gram and smoothing were used.

For example, FST-C-7GRAM-KN denotes an end-to-end FST with a 7-gram language model and Kneser-Ney smoothing.

Here is a list of n-gram language models that we considered (with perplexity scores in parentheses): We note that 13-and higher n-gram language models performed far worse.

The numbers in parentheses show character-level perplexities produced using a validation set that was drawn from a synthetic parallel corpus, which will be explained in Section 5.3.

DISPLAYFORM0 We also trained two character-level RNNLMs with the following configurations:• 2 layers × 200 unit LSTM (2.70) • 3 layers × 200 unit LSTM (2.65) Both RNNLMs were trained with plain SGD, using a batch size of 30, a learning rate of 10, truncated backpropagation through time with 45 unrolling steps, and the gradient renormalized to norm 1 when it exceeded 1.

Both RNNLMs used a dropout rate of 0.2 at the input, between RNN layers, and after the last RNN layer.

The best n-gram and RNNLMs are highlighted in bold.

Overall, the best results were produced with the hybrid approach, using RNNLMs.

Because we were unable to find a sufficiently large corpus of parallel texts in the missionary and modern Hawaiian orthographies, we instead trained the n-gram and RNN language models on a corpus of modern Hawaiian texts (ground-truth) (see Section 3.3).

Parallel corpora were only required to test predictions from missionary-era to modern texts, which were produced by composing one of the FSTs, C or C wb , with either an n-gram or RNN language model.

To evaluate the accuracy of our approaches, we first derived a synthetic parallel corpus from our collection of modern Hawaiian texts.

We also used a small but real parallel corpus, based on two 19th century newspaper texts and their hand-edited modern equivalents.

Results based on these parallel corpora are reported in the following subsections.

To produce a simulated parallel corpus (input-missionary), we systematically reduced the orthography in the modern texts (Section 3.3) using the backward mapping described above (Section 3.2).

We then applied the end-to-end FST and hybrid FST-RNNLM models (Section 4), with the aim of learning a forward mapping between orthographies that recovers the lost information.

We evaluated the predicted modern text (predictions) by computing DISPLAYFORM0 , where d denotes character-level edit distance.

This is a modification of character error rate, normalized by the distance of the input and target rather than by the length of the target.

We note that CERR may be high even when the predictions are very accurate as d(input-missionary, ground-truth) is small when the text is similar in both orthographies.

Table 2 gives more details about the strongest models from both approaches.

Out of the Kneser-Ney n-gram models, we found that the FST-C-9GRAM-KN and the version modelling word boundaries (FST-C wb -9GRAM-KN) performed best on the synthetic parallel corpus and newspapers, respectively.

C wb was not applied to the synthetic parallel corpus as the synthetic parallel corpus did not model word splitting.

However, the hybrid models (FST-RNNLM) outperformed all end-to-end FSTs.

Table 2 : Performance (%CERR).

Slash-separated pairs denote FSTs incapable/capable of inserting word boundaries, respectively; see Section 4.

The -KN suffix denotes Kneser-Ney smoothing.

The data from Section 3.3 is used for evaluating the modern-orthography language model perplexity, and "Corpus" evaluates test-set transliteration performance from the synthetic missionary text back to the original modern text.

Not content to evaluate the model on simulated missionary orthography, we also evaluated it on two newspaper texts, using selections originally published in 1867 and 1894 for which we had 19th century and manually-edited modern equivalents.

The newspaper selections discuss Kahahana, one of the last kings of O'ahu BID7 , and Uluhaimalama, a garden party and secret political gathering, held after the deposition of Hawai'i's last queen (Pukui et al., 2006) .

Unlike the synthetic missionary corpus evaluation where we did not model word splitting, we found that replacing C with C wb on the newspaper texts significantly improved the output, especially on the FST-RNNLM model.

Again, we found a hybrid model (FST-RNNLM-C wb ) to be the best performing model overall (Table 2) .

With this paper we introduced a new transliteration problem to the field, that of mapping between old and new Hawaiian orthographies-where the modern Hawaiian orthography represents linguistic information that is missing from older missionary-era texts.

One difficulty of this problem is that there is a limited amount of Hawaiian data, making data-hungry solutions like end-to-end deep learning unlikely to work.

To solve the transliteration problem, we therefore proposed two models: the first was implemented end-to-end using weighted FSTs; the second was a hybrid deep learning approach that combined an FST and an RNNLM.

Both models gave promising results, but the hybrid approach performed best.

It allowed us to use a more powerful recurrent neural network-based language model, despite our dataset's small size.

Factoring a problem like ours into one part that can be modelled exactly using expert domain knowledge and into another part that can be learned directly from data using deep learning is not novel; however it is a promising research direction for data-efficient modelling.

To our knowledge, this paper is the first to describe a procedure to compose an FST with an RNN by approximately performing beam search over the FST.While the role of the RNNLM part of the hybrid approach may be obvious, the FST component plays an important role too.

For example, the hand-designed FST can be replaced without needing to re-train the RNNLM.

We tried to showcase this modularity by constructing two FSTs which we referred to as C and C wb , where only the latter allowed the insertion of spaces.

Future work could extend the FST to model orthographic changes suggested by an error analysis of the current model's predictions.

Input:

Weheia ka Malapua Alii a Kanuia na Uluwehi no ia Wao.

Prediction:

Wehe 'ia ka Māla pua Ali'i a Kanu 'ia na Uluwehi nō ia Wao.

Ground-truth:

Wehe 'ia ka Māla Pua Ali'i a Kanu 'ia nā Uluwehi no ia Wao. (Pukui et al., 2006) .

Correct predictions are green and bold.

Characters omitted by the model as compared to the ground-truth are denoted by blue italics; erroneous insertions or substitutions are denoted by red underline.

To make white spaces explicit, we represent them with the symbol ' '.

More sample predictions can be found in Appendix A.An example of the current model's predictions (i.e. missionary input, predicted modern text, modern ground-truth) is given in TAB4 .

In this example, we see the model correctly predicting some word boundaries, glottal stops and long vowels; however, we note that the model could not predict uppercase Pua (correct), because the input text contained lowercase pua (incorrect), and no (p : P) transitions were included in C or C wb .

Similar observations (see Appendix A) motivate new mappings for consonant substitutions like (r : l) and (s : k) that occur in loanword adaptations (e.g. rose ⇒ loke).

The error analysis also motivates mappings to delete spaces ( : ) and to handle contractions, like na'lii ⇒ nā ali'i.

We could further incorporate linguistic knowledge of Hawaiian into the FST, which tells us, for example, about expected sequences of vowels (Parker Jones, 2010) .

Additional improvements to the hybrid model might be obtained by increasing the amount of modern Hawaiian text used to train the RNNLM.

One way to do this would be to accelerate the rate at which missionary-era Hawaiian texts are modernized.

To this end, we hope that the present models will be used within the Hawaiian community to semi-automate, and thereby accelerate, the modernization of old Hawaiian texts.

Tomas Mikolov, Martin Karafiát, Lukás Burget, JanČernocký, and Sanjeev Khudanpur.

2010.

Each block of three lines shows the input, followed by the prediction, followed by the ground-truth.

Characters omitted by the model as compared to the ground-truth are denoted by blue and italic, whereas characters that are erroneously inserted or substituted (or should have been substituted) for another character are denoted by red underline.

When the incorrect character is a space, the space is replaced with ' '.First 10 sentences in Newspaper 1:Input 1: Weheia ka Malapua Alii a Kanuia na Uluwehi no ia Wao.

Prediction 1: Wehe 'ia ka Māla pua Ali'i a Kanu 'ia na Uluwehi nō ia Wao.

Ground-truth 1: Wehe 'ia ka Māla Pua Ali'i a Kanu 'ia nā Uluwehi no ia Wao.

Input 2: E like no hoi me ka mea i hoike akea ia ae no ka manawa a me ka la e weheia ai a e kanuia ai hoi o na pua a me na mea ulu e ae ma kahi nona ka inoa kilakila maluna ae, pela no i hoea io mai ai i ka Poaha iho la, hora 9 A. M. a mahope mai.

Mamua ae o ia manawa, ua lehulehu na poe i pii aku me na mea kanu, maluna o na kaa a malalo no hoi.

Prediction 2: E like nō ho'i me ka mea i hō'ikeākea 'ia a'e no ka manawa a me ka lā e wehe '

ia ai a e kanu 'ia ai ho'i 'o nā pua a me nā mea ulu 'ē a'e ma kahi nona ka inoa kilakila ma luna a'e, pēlā nō i hō'ea 'i'o mai ai i ka Po'ahā iho la, hora 9 A. M. a ma hope mai.

Ma mua a'e o ia manawa, ua lehulehu nā po'e i pi'i aku me nā mea kanu, ma luna o nā ka'a a ma lalo nō ho'i.

Ground-truth 2: E like nō ho'i me ka mea i hō'ikeākea 'ia a'e no ka manawa a me ka lā e wehe '

ia ai a e kanu 'ia ai ho'i o nā pua a me nā mea ulu 'ē a'e ma kahi nona ka inoa kilakila ma luna a'e, pēlā nō i hō'ea 'i'o mai ai i ka Pō'ahā ihola, hola 9 A. M.

a ma hope mai.

Ma mua a'e o ia manawa, ua lehulehu nā po'e i pi'i aku me nā mea kanu, ma luna o nā ka'a a ma lalo nō ho'i.

Input 3: Hoomakaia ke Kanu Ana.

Prediction 3: Ho'omaka 'ia ke Kanu 'Ana.

Ground-truth 3: Ho'omaka 'ia ke Kanu 'Ana.

Input 4:

Ua hoea ae no ilaila ka Puali Puhiohe Lahui, a i ka aneane ana ae i ka manawa, a i ole ia, ua hala no paha he hapalua hora mahope iho o ka hora 9, ua uhene mai la lakou i ke mele Liliuokalani, a o ka wa no ia o Kamalii Kawananakoa, ma ka aoao o ke Aliiaimoku, i kanu iho ai i kekahi kumu lehua o Mokaulele iwaenakonu, i hoopuniia ae me na ohawai a me kekahi mau mea kanu Hawaii e ae iloko o kekahi ponaha poepoe, a makai iho hoi o ia wahi i kanu ai o Kamalii Kalanianaole i kekahi kumu lehua ahihi ma ka aoao o ke Alii ka Moiwahine Kanemake.

Pau keia mau hana ae la, ua noa i na mea a pau, a ua hele no hoi ia wahi a eeu i na oiwi palupalu o kakou, e kanu ana i kela a me keia mea, a he mau oiwi oolea no hoi kekahi malaila e kokua ana.

He mea makehewa paha ke helupapa aku i na pua i kanuia.

Hookahi a makou mea i mahalo, oia no kekahi wahi i kanu mua e ia no makai iki mai o ka puka komo, me ka inoa o ia kihapai e kau ae la maluna, a he ku maoli no i ka nani.

Prediction 4: Ua hō'ea a'e nō i laila ka Pū'ali Puhi 'ohe Lāhui, a i ka 'ane'ane 'ana a'e i ka manawa, a i 'ole ia, ua hala nō paha he hapalua hora ma hope iho o ka hora 9, ua 'uhene mai la lākou i ke mele Lili'uokalani, a 'o ka wā nō ia o Kāmali'i Kawānanakoa, ma ka 'ao'ao o ke Ali'i 'ai moku, i kanu iho ai i kekahi kumu lehua o Mōkaulele i waenakonu, i ho'opuni 'ia a'e me nā 'ohā wai a me kekahi mau mea kanu Hawai'i 'ē a'e i loko o kekahi pōnaha poepoe, a ma kai iho ho'i o ia wahi i kanu ai 'o Kamāli'i Kalaniana'ole i kekahi kumu lehua 'āhihi ma ka '

ao'ao o ke Ali'i, ka Mō'ī wahine Kāne make.

Pau kēia mau hana a'e lā, ua noa i nā mea a pau, a ua hele nō ho'i ia wahi a 'e '

eu i nā 'ōiwi palupalu o kākou, e kanu ana i kēlā a me kēia mea, a he mau 'ōiwi 'o'ole'a nō ho'i kekahi ma laila e kōkua ana.

He mea makehewa paha ke helu papa aku i nā pua i

@highlight

A novel, hybrid deep learning approach provides the best solution to a limited-data problem (which is important to the conservation of the Hawaiian language)