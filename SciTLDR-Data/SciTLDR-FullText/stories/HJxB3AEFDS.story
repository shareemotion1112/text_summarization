We introduce a novel method for converting text data into abstract image representations, which allows image-based processing techniques (e.g. image classification networks) to be applied to text-based comparison problems.

We apply the technique to entity disambiguation of inventor names in US patents.

The method involves converting text from each pairwise comparison between two inventor name records into a 2D RGB (stacked) image representation.

We then train an image classification neural network to discriminate between such pairwise comparison images, and use the trained network to label each pair of records as either matched (same inventor) or non-matched (different inventors), obtaining highly accurate results (F1: 99.09%, precision: 99.41%, recall: 98.76%).

Our new text-to-image representation method could potentially be used more broadly for other NLP comparison problems, such as disambiguation of academic publications, or for problems that require simultaneous classification of both text and images.

Databases of patent applications and academic publications can be used to investigate the process of research and innovation.

For example, patent data can be used to identify prolific inventors (Gay et al., 2008) or to investigate whether mobility increases inventor productivity (Hoisl, 2009 ).

However, the names of individuals in large databases are rarely distinct, hence individuals in such databases are not uniquely identifiable.

For example, an individual named "Chris Jean Smith" may have patents under slightly different names such as "Chris Jean Smith", "Chris J. Smith", "C J Smith", etc. . .

There may also be different inventors with patents under the same or similar names, such as "Chris Jean Smith", "Chris J. Smith", "Chris Smith", etc. . .

Thus it is ambiguous which names (and hence patents) should be assigned to which individuals.

Resolving this ambiguity and assigning unique identifiers to individuals -a process often referred to as named entity disambiguation -is important for research that relies on such databases.

Machine learning algorithms have been used increasingly in recent years to perform automated disambiguation of inventor names in large databases (e.g. Li et al. (2014) ; Ventura et al. (2015) ; Kim et al. (2016) ).

See Ventura et al. (2015) for a review of supervised, semi-supervised, and unsupervised machine learning approaches to disambiguation.

These more recent machine learning approaches have often out-performed more traditional rule-and threshold-based methods, but they have generally used feature vectors containing several pre-selected measures of string similarity as input for their machine learning algorithms.

That is, the researcher generally pre-selects a number of string similarity measures which they believe may be useful as input for the machine learning algorithm to make discrimination decisions.

Here we introduce a novel approach of representing text-based data, which enables image classifiers to perform text classification.

This new representation enables a supervised machine learning algorithm to learn its own features from the data, rather than selecting from a number of pre-defined string similarity measures chosen by the researcher.

To do this, we treat the name disambiguation problem primarily as a classification problem -i.e.

we assess pairwise comparisons between records as either matched (same inventor) or non-matched (different inventors) (Trajtenberg et al., 2006; Miguélez & Gómez-Miguélez, 2011; Li et al., 2014; Ventura et al., 2015; Kim et al., 2016) .

Then, for a given pairwise comparison between two inventor records, our text-to-image representa-tion method converts the associated text strings into a stacked 2D colour image (or, equivalently, a 3D tensor) which represents the underlying text data.

We describe our text-to-image representation method in detail in Section 4.1 (see Figure 1 in that section for an example of text-to-image conversion).

We also test a number of alternative representations in Section 5.4.

Our novel method of representing text-based records as abstract images enables image processing algorithms (e.g. image classification networks), to be applied to textbased natural language processing (NLP) problems involving pairwise comparisons (e.g. named entity disambiguation).

We demonstrate this by combining our text-to-image conversion method with a commonly used convolutional neural network (CNN) (Krizhevsky et al., 2012) , obtaining highly accurate results (F1: 99.09%, precision: 99.41%, recall: 98.76%).

Inventor name disambiguation studies have often used measures of string similarity in order to make automated discrimination decisions.

For example, counts of n-grams (sequences of n words or characters) can be used to vectorise text, with the cosine distance between vectors providing a measure of string similarity (Raffo & Lhuillery, 2009; Pezzoni et al., 2014) .

Measures of edit distance consider the number of changes required to transform one string to another, e.g. the number of additions, subtractions, or substitutions used in the calculation of Levenshtein distance (Levenshtein, 1966) , or of other operations such as transpositions (the switching of 2 letters) used to calculate Jaro-Winkler distance (Jaro, 1989; Winkler, 1990) .

Phonetic algorithms, such as Soundex, recode strings according to pronunciation, providing a phonetic measure of string similarity (Raffo & Lhuillery, 2009 ).

Measures of string similarity such as these have been used to guide rule-and threshold-based name disambiguation algorithms (e.g. Miguélez & Gómez-Miguélez (2011) and Morrison et al. (2017) ).

They can also be used within feature vectors inputted into machine learning algorithms.

For example, Kim et al. (2016) use such string similarity feature vectors to train a random forest to perform pairwise classification.

Ventura et al. (2015) reviewed several supervised, semi-supervised, and unsupervised machine learning approaches to inventor name disambiguation, as well as implementing their own supervised approach utilising selected string similarity features as input to a random forest model.

Two-dimensional CNNs have been used extensively in recent image processing applications (e.g. Krizhevsky et al. (2012) ), and one-dimensional (temporal) CNNs have been used recently as character-level CNNs for text classification (e.g. Zhang et al. (2015) ).

Also, neural networks (usually CNNs) have been used previously to assess pairwise comparison decisions -e.g.

in the case of pairs of: images (Koch et al., 2015) , image patches (Zbontar & LeCun, 2016; Zagoruyko & Komodakis, 2015) , sentences (Yin et al., 2016) , images of signatures (Bromley et al., 1993) , and images of faces (Hu et al., 2014) .

These networks are generally constructed for multiple images to be provided simultaneously as input, such as in the case of siamese neural networks where two identical sub-networks are connected at their output (Bromley et al., 1993; Koch et al., 2015) .

In this work we generate a single 2-dimensional RGB (stacked) image for a given pairwise record comparison.

Thus any image classification network that processes single images can be used (with minimal modification) to process our pairwise comparison images, therefore enabling such neural networks to classify associated text records.

We demonstrate this using the seminal "AlexNet" image classification network (Krizhevsky et al., 2012) .

We use a combination of two labelled datasets in this work to train the neural network and assess its performance.

Each dataset was derived by separate authors, from the US National Bureau of Economics Research (NBER) Patent Citation Data File (Hall et al., 2001 ); i.e. a labelled dataset of Israeli inventors (Trajtenberg et al., 2006) (the "IS" dataset), and a dataset of patents filed by engineers and scientists (Ge et al., 2016) (the "E&S" dataset).

These datasets were combined with US Patent and Trademark Office (USPTO) patent data as part of the PatentsView Inventor Disambigua-

Each labelled dataset contains unique IDs that identify all inventor-name records from different patents belonging to each unique inventor.

We also extracted several other variables from inventorname records in the bulk USPTO patent data to use in our disambiguation algorithm: first name, middle name, last name, city listed in address, international patent classification (IPC) codes (i.e. subjects/fields covered by the patent), assignees (i.e. associated companies/institutes), and co-inventor names on the same patent.

Our novel inventor disambiguation algorithm involves the following main steps:

1.

Duplicate removal: remove duplicate inventor records.

2.

Blocking: block (or "bin") all names by last name, and also by first name in some cases.

3.

Generate pairwise comparison-map images: convert text from each within-block pairwise record comparison into a 2D RGB image representation.

4.

Train neural network: use 2D comparison-map images generated from manually labelled data to train a neural network to classify whether a given pairwise record comparison is a match (same inventor) or non-match (different inventors).

5.

Classify pairwise comparison-map images: deploy the trained neural network to classify pairwise comparison images generated from the bulk patent data, producing a match probability for each record pair.

6.

Convert pairwise match probabilities into clusters: convert the pairwise match/nonmatch probabilities generated by the neural net into inventor clusters -i.e.

groups of inventor-name records that each belong to a distinct individual inventor.

Assigning a unique ID (UID) to each of these groups then leads to a single set of disambiguated inventor names.

Note that the main purpose of the first two steps is to improve computational efficiency: i.e. rather than process all possible pairs of patent-inventor records (which has time complexity O(n 2 ) for n records), the records are first grouped into similar clusters, or "blocks", and pairwise comparisons are only made within those blocks.

For further detail regarding steps 1 and 2, see Appendices A and B. Steps 3-6 are described in detail below.

Our intent is to assess all possible within-block pairwise comparisons between patent-inventor records, classifying each comparison as either a match or non-match.

To do this, we introduce a new method of converting any string of text into an abstract image representation of that text, which we refer to as a "comparison-map" image.

Any image classification neural network can then be used to process these images and hence effectively perform text classification.

To generate a comparison-map image, we firstly define a specific 2D character layout -i.e.

a grid of pixels specifying the positions of each letter.

The layout of this "string-map" is shown in Figure  1 (identical in each of the five images).

For a given word (e.g. "JEN"), we then add a particular colour (e.g. red) to the pixels of each letter in the word, as well as to any pixels in straight lines connecting those letters.

In particular, we add colour to the pixels of the first and last letters ( the beginning of each string-map, we also repeat the process for the first bi-gram only ("JE") in blue, rather than red ( Figure 1 , fourth image).

The final string-map for the word "JEN" is shown in Figure  1 (right-most image).

If we then add the string-map of any other word to the green channel of the same RGB image (with the first bi-gram again highlighted in blue), the resulting image represents the pairwise comparison of the two words (e.g. Figure 2 , right-most image).

For a given inventor name record, we generate string-maps for each variable in the record -i.e.

first name, middle name, last name, city, IPC codes, co-inventors, and assignees.

These string-maps are combined into a single image, arranged as shown in Figure 3 , which we refer to as a "recordmap".

Note that we use slightly different string-maps for IPCs, co-inventors, and assignees due to differences in those variables, as described in Appendix C.

We compare any two inventor name records by stacking the two associated 2D record-maps into the same RGB image, one as the red channel and the other as green (with the beginning two-letter bigram of each record sharing the blue channel).

We refer to the resulting RGB image (or 3D tensor) representation as a "comparison-map" (Figure 4 ).

Since red and green combined produce yellow in the RGB colour model, a comparison-map image generated from two similar records should contain more yellow (e.g. Figure 4 , left image), whereas a comparison-map image from two dissimilar records should contain more red and green (e.g. Figure 4, right image) due to less overlap between the two record-maps.

When training on labelled comparison-maps, we expect that the neural network will learn to identify features such as these, which are useful for discriminating between matched/non-matched records.

That is, the neural network's learned pattern recognition on comparison-map images will essentially recognise underlying text patterns which are present in the associated patent-inventor name records.

Note that we chose the particular layout of the letters in the string-map shown in Figure 1 heuristically, such that vowels (which are less important than consonants when assessing string similarity) are positioned towards the centre of the grid, where pixels are more likely to saturate.

We also grouped letters with similar phonetic interpretations, such as "S" and "Z", close to each other.

We anticipated that this heuristic layout might make it more straightforward for the network to learn which features are associated with matches/non-matches.

However, we test how the heuristic layouts shown in Figures 1, 2 , and A1 (see Appendix C) perform compared with alternative random layouts later in Section 5.4, and find similar performance regardless of the chosen layout.

Our method of converting text into a stacked 2D RGB bitmap for neural net-based image classification has several benefits.

Firstly, the powerful classification capabilities of previous image classification networks can be utilised for text-based record matching, with minimal modification.

The neural network also learns its own features from the data, rather than learning from a feature vec- The left comparison-map image was generated using two matched records (Table 1 , rows 1 and 2), and the right image from two non-matched records (Table 1 , rows 1 and 3).

tor of pre-defined string similarity measures chosen by the researcher.

Additionally, minor spelling variations and errors do not alter the resulting string-map very much, and the neural network has the potential to learn that such minor features are unimportant for discriminating between matches and non-matches.

When matched records have different word ordering (e.g. re-ordered co-inventor names on different patents), those records are likely to be matched due to overlapping pixels.

The neural net can potentially learn to ignore certain shapes of common words (e.g. "Ltd", "LLC", "Incorporated", etc. . . )

which are not useful for discrimination decisions.

Also, we show later in Section 5.4 that our novel disambiguation algorithm performs well under multiple different choices of alternative string-maps other than those shown in Figures 1, 2 , & A1 (Appendix C), suggesting that the neural network has quite robust pattern recognition of features within our comparison-map representations.

Note that these benefits of our text-to-image conversion method could also potentially apply to other text-based comparison problems (e.g. data linkage, or disambiguation of academic papers), or to problems that require simultaneous classification of both text and images.

To demonstrate that our text-to-image conversion method can be combined with an image classifier to perform text-based classification, we apply the method to a commonly used image classification neural network; i.e. the seminal "AlexNet" CNN (Krizhevsky et al., 2012) .

AlexNet was originally designed to classify colour images (224×224×3-pixel bitmaps) amongst 1,000 classes.

We slightly modify the network architecture to enable classification of pairwise comparison-map images (31×31×3-pixel bitmaps) into two classes (match/non-match), by using appropriate input and output layers for our problem and smaller kernels in the first convolutional layer.

See Appendix D for details on our implementation of AlexNet.

After running the trained neural network on bulk patent data, each within-block pairwise comparison has an associated match probability.

To assign unique IDs (UIDs) to the bulk data, we convert these pairwise probabilities into linked (matched) "inventor groups" using a clustering algorithm.

Each inventor group is a linked cluster of inventor name records which all refer to the same individual.

Briefly, the clustering algorithm involves converting each pairwise probability value to a binary value (match/non-match) using a pre-selected probability threshold (p) as a cut-off.

Each matched record is then clustered into a larger inventor group if the number of links (l) it has to the that group is the number of nodes in the group (n) times some threshold proportion value (l); i.e. if l nl.

This removes weakly-linked records from each group.

For further detail on the clustering algorithm, see Appendix E. Note that choosing differentp andl values generates different trade-offs between precision and recall.

Once the clustering algorithm has been applied to each block, every patent-inventor name instance has an associated unique inventor ID, and the disambiguation process is complete.

Here we firstly describe our procedure for dividing our labelled datasets into training and test data.

We then evaluate our inventor disambiguation algorithm, compare those results to previous studies, and test alternative string-map layouts.

We use the IS and E&S labelled datasets to train the neural network to discriminate between matched and non-matched pairwise comparisons.

Each of the labelled datasets are randomly separated into 80% training data (used to train the neural network) and 20% test data (used to assess algorithm performance).

We use 75% of the training data to train the network, and the remaining 25% to perform validation assessments during training in order to monitor potential overfitting.

Duplicate removal and blocking is then performed on the labelled data, and comparison-map images are generated for all possible pairwise record comparisons within each block (723,178 comparisonmaps for training and 144,552 comparison-maps for testing).

We also perform duplicate removal and blocking on the bulk data, generating comparison-maps for all possible pairwise within-block comparisons (stored as 3D numerical arrays).

The trained neural network is then deployed on the bulk patent data, generating match/non-match probabilities for all pairwise within-block comparisons (112,068,838 comparison-maps).

Prior to processing the bulk data, we experimented with multiple different values for the pairwise comparison probability threshold (p) and linking proportion threshold (l), based on evaluating the trained neural network on the labelled test data.

Differentp andl values produce different trade-offs between precision and recall, and we use values that produce an optimal trade-off (highest F1 score).

We state eachp andl value whenever quoting results from a given run of our disambiguation algorithm.

To evaluate the performance of the disambiguation algorithm, we use the labelled IS and E&S test data to estimate pairwise precision, recall, splitting, and lumping based on numbers of true positive (tp), false positive (fp), true negative (tn), and false negative (fn) pairwise links within the labelled test data, as follows (e.g. Ventura et al. * Note that this result was obtained using a randomlygenerated string-map character order (see Section 5.4).

Higher values are better for precision and recall, while lower values are better for lumping and splitting errors.

We also use the pairwise F1 score:

Since the F1 score accounts for the trade-off between precision and recall, it is the primary measure we use to compare the performance of different disambiguation algorithms.

The precision, recall, and F1 estimates for two example runs of our disambiguation algorithm are shown in the bottom two rows of Table 2 -first is the highest F1 result obtained using the heuristic string-map character order (Figures 1, 2 , and A1), and second is the highest F1 result obtained using a randomly-generated string-map character order (see Section 5.4 for details).

Table 2 also shows the best results (highest F1) obtained by previous studies which (1) disambiguate bulk USPTO patent data, and (2) evaluate their results using the same labelled datasets we use in this work (i.e. the IS and E&S datasets).

Our inventor disambiguation algorithm performs well compared with these other disambiguation studies (Table 2 , bottom row), marginally out-performing the previous stateof-the-art study of Kim et al. (2016) and obtaining a much higher F1 score than Yang et al. (2017) .

For completeness, we also compare our results to those of other studies which use alternative labelled datasets to the IS and E&S datasets used in this work -i.e.

Table 3 shows the best results obtained by each study, regardless of the evaluation dataset.

Note that Table 3 provides a slightly less equitable comparison than Table 2 , as there is generally a small amount of variation in an algorithm's F1 score when evaluated on different labelled datasets.

Nonetheless, we include Table 3 here for completeness and consistency with previous inventor name disambiguation studies, which often include comparison to other studies with different evaluation datasets.

Our disambiguation algorithm is again competitive with the other state-of-the-art inventor name disambiguation algorithms in Table 3 , obtaining the highest F1 score compared with the other three studies which quote F1 results, and lower splitting and lumping errors compared with the Li et al. (2014) and Ventura et al. (2015) studies.

Here we compare the performance of our heuristic string-map layouts (Figures 1, 2 , and A1) to several alternative string-maps.

The first alternative string-map we test has random character order; i.e. we keep the pixel co-ordinates identical to the co-ordinates of the associated heuristic layout, but randomise the order of each character (these randomised string-maps are shown in Appendix F, Figure A2 ).

We also test two alternative string-maps in which we randomise both the pixel co-ordinate layout and character order (Appendix F, Figure A3 ).

One alternative uses the large string-map for co-inventors and assignees ( Figure A3 , right image).

The other alternative uses the smaller 5 × 5 pixel string-map for co-inventors and assignees ( Figure A3 , left image), leading to a smaller comparison-map (Appendix F, Figure A4 ).

We also investigate a string-map with random character order in which we exclude the blue channel for leading bi-grams (Figure 1 , fourth image).

Estimates of precision, recall, and F1 for each of these alternative string-maps are shown in Table  4 .

For each alternative string-map, we ran the algorithm multiple times using different settings of the comparison probability threshold (p) and linking proportion threshold (l), and only show results from the run which produced the highest F1 score.

Results obtained from each of the alternative string-maps are quite similar to those obtained using the heuristically-determined layout (F1 scores range from 98.99% to 99.09%).

This suggests that our method of converting text into abstract image representations facilitates robust feature learning for several alternative choices of string-map structure, such as randomised string-map character order and/or layout, heuristic order and/or layout, different string-map sizes, and the inclusion/exclusion of a blue channel for leading bi-grams.

Our name disambiguation algorithm provides a novel way of combining image processing with NLP, allowing image classifiers to perform text classification.

We demonstrated this with the AlexNet CNN, producing highly accurate results (F1 score: 99.09%).

We also analysed several variants of alternative string-maps, and found that the accuracy of the disambiguation algorithm was quite robust to such variation.

Our disambiguation algorithm could easily be adapted to other NLP problems requiring text matching of multiple strings (e.g. academic author name disambiguation or record linkage problems).

The algorithm could also potentially be modified to process records that contain both text and image data, by combining each record's associated image with the abstract image representation of the record's text, in a single comparison-map.

It is sometimes obvious that two inventor name records likely belong to the same individual, because the two records contain several fields that are identical.

For example, if the last name, first name, city, and IPCs of two different records are all exactly identical, it is highly likely that the two records belong to the same individual.

We remove such duplicate records based on the following duplication keys: duplicnkey_ipc = lastname + firstname + city + '_'.join(ipcs) duplicnkey_assignee = lastname + firstname + city + '|'.join(assignees)

For a given group of duplicate records sharing the same duplication key, all records except for the first record to be processed are removed from the bulk data.

The first record then remains within the bulk data to be processed by the disambiguation algorithm, receiving a unique inventor ID once the algorithm has completed its run.

That same ID is then assigned to each removed record in the corresponding group of duplicate records.

The blocking procedure broadly involves grouping together inventor name records into "blocks" (or "bins") using each inventor's last name, and sometimes also their first name.

Latter parts of the algorithm will only assess pairwise comparisons within these blocks, never across different blocks.

We firstly group patent-inventor name records together by the first three letters of the last name (this first step is identical to the initial stage of the blocking procedure used by Ventura et al. (Ventura et al., 2015) ).

However, some of the resulting blocks contain very large numbers of records, and hence large numbers of pairwise comparisons.

To improve efficiency, we further divide such large blocks into smaller blocks by progressively increasing the number of letters used for blocking.

That is, if the number of records within a given block (n b ) is above some threshold number (n b ), then the records within that block are separated into smaller blocks according to the first four letters of the last name.

We then continue sub-dividing any blocks that still have n b >n b , according to the first five letters of the last name, then six letters, and so on.

If all letters of the last name have been used and any blocks still have n b >n b , then we append a comma to the string and begin progressively appending letters from the first name as well.

We usen b = 100 throughout this work, as initial testing indicated that it produced a good balance between:

• computational efficiency: i.e. smallern b leads to more numerous, smaller bins, and hence fewer comparisons (which are O(n Note that since latter parts of the algorithm only assess within-block pairwise comparisons and some inventors' sets of records may have been separated across two or more different blocks, there is a maximum limit to the possible recall attainable by the disambiguation algorithm.

After running the blocking procedure on the labelled dataset, we use known pairwise matches in the labelled data to estimate this maximum limit to recall, obtaining the following values: 99.47% (E&S training data), 99.98% (E&S test data), 99.83 (IS training data), and 99.86 (IS test data).

Since a given patent-inventor record can have multiple assignees and/or co-inventors, we use a larger string-map for those variables (see Figure A1 , left image).

This reduces the possibility that pixels will become saturated in cases where many assignees (or co-inventors) are overlayed onto the same string-map.

We also add less colour to each pixel in these larger string-maps, again to reduce the possibility of saturation.

For international patent classification (IPC) codes, which contain numbers as well as letters, we use a different string-map shown in Figure A1 (right image).

We slightly modify the AlexNet network architecture to enable classification of pairwise comparison-map images (31×31×3-pixel bitmaps) into two classes (match/non-match), by altering four hyperparameters as shown in Table 5 .

We use the NVIDIA Deep Learning GPU Training Figure A1 : Larger string-map for assignees and co-inventors, and IPC-map.

The larger stringmap used to convert a given list of assignees or co-inventors into an abstract image representation (left), and the IPC-map used to convert a given list of IPC classes into an abstract image representation (right). (Jia et al., 2014) .

We use the default settings for the DIGITS solver (stochastic gradient descent), batch size (100), and number of training epochs (30).

Rather than use the default learning rate (0.01), we use a sigmoid decay function to progressively decrease the learning rate from 0.01 to 0.001 over the course of the 30 training epochs, as testing indicated that this produced slightly higher accuracies.

Instead of the 1,000-neuron softmax output layer in AlexNet, we use a 2-neuron softmax output layer, which outputs a probability distribution across our two possible classes (match/non-match).

Note that the default settings of the DIGITS v2.0.0 implementation of AlexNet transform the input data by (1) altering input images to show the deviation from the mean of all input images (by subtracting the mean image from each input image), (2) randomly mirroring input images, and (3) taking a random square crop from the input image.

The main purpose of performing such transformations is to introduce variability into the training images that are expected to be present in the unlabelled data, however we do not use any of those transformations in this work because our images are much more self-consistent than those in the ImageNet database.

Here we describe the clustering algorithm we use to convert pairwise match probabilities into groups of records each belonging to a single unique inventor.

We firstly convert each pairwise probability between the ith and jth record (p ij ) into one of the binary classes (c ij ; either "match" or "nonmatch") based on a threshold probability value (p) as follows:

The inventor group linking algorithm then primarily involves combining different sub-groups together into the one group if they share enough links (pairwise matches).

Within a given block, the algorithm involves the following steps:

1.

Order all patent-inventor name records by the number of links they have to other records (i.e. the number of asserted matches to other records), highest first.

Figure A2 : Random character order (string-maps).

Here we show the smaller string-map (left), IPC-map (middle), and larger string-map (right) we use for runs in which the character order has been randomised.

2.

Assign a UID to each isolated (non-matched) patent-inventor name.

3.

Assign records to inventor groups.

That is, for a given record, the corresponding inventor group initially comprises just the record itself and all records it is linked (matched) to.

Each of these linked records (nodes) are kept in the current inventor group only if the number of links (l) it has to the current group is the number of nodes in the group (n) times some threshold proportion (l); i.e. if l nl.

This removes the most weakly-linked records from each group (i.e. the nodes with fewest links to their group), which are more likely to be false positive matches.

Any outside-group links -i.e. links to nodes that are not within the current group -are also recorded during this step.

Step 2, because some records may have become isolated (non-matched) following

Step 3.

5.

Combine inventor groups together if the number of links they share is greater than a specified threshold.

In particular, for an inventor group with n self records (nodes), we combine it with any other group with n other nodes if the number of links to that other group (l) satisfies both: l l n self , and: l l n other .

6.

For each resulting inventor group, assign an identical UID to all patent-inventor name records within the group.

Here we show the random string layouts analysed in Section 5.4.

Figure A2 shows the string-maps we use for runs where characters are positioned using an identical pixel co-ordinate layout to the heuristic layouts shown in Figures 1 (main text) and A1, but where the order of each character has been randomised.

Figure A3 shows the string-maps we use for runs where both the pixel co-ordinate layout and character order have been randomised.

Figure A4 shows the comparison-map with random layout and character order in which we use the smaller 5 × 5 pixel string-map ( Figure A3 , left image) for co-inventors and assignees, rather than the larger string-map ( Figure A3 , right image).

Figure A5 shows the record-map layout associated with the comparison-map in Figure A4 .

Figure A3 : Random character order and layout (string-maps).

Here we show the smaller stringmap (left; identical to the left string-map in Figure A2 ), IPC-map (middle), and larger string-map (right) with both random character order and random pixel co-ordinate layout.

Figure A4 : Random character order and layout, small string-maps (comparison-map) .

This shows the comparison-map used for runs with smaller string-maps for co-inventors and assignees, as well as random character order and random pixel co-ordinate layout.

Figure A5 : Record-map layout with smaller string-map for co-inventors and assignees.

The record-map layout associated with the comparison-map in Figure A4 .

@highlight

We introduce a novel text representation method which enables image classifiers to be applied to text classification problems, and apply the method to inventor name disambiguation.

@highlight

A method to map a pair of textual information into a 2D RGB image that can be fed to 2D convoutional neural networks (image classifiers).

@highlight

The authors consider the problem of names disambiguisation for patent names inventors and propose to build an image page representation of the two name strings to compare and to apply an image classifier.