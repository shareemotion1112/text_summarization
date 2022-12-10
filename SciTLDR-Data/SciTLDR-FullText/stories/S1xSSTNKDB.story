Existing public face image datasets are strongly biased toward Caucasian faces, and other races (e.g., Latino) are significantly underrepresented.

The models trained from such datasets suffer from inconsistent classification accuracy, which limits the applicability of face analytic systems to non-White race groups.

To mitigate the race bias problem in these datasets, we constructed a novel face image dataset containing 108,501 images which is balanced on race.

We define 7 race groups: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino.

Images were collected from the YFCC-100M Flickr dataset and labeled with race, gender, and age groups.

Evaluations were performed on existing face attribute datasets as well as novel image datasets to measure the generalization performance.

We find that the model trained from our dataset is substantially more accurate on novel datasets and the accuracy is consistent across race and gender groups.

We also compare several commercial computer vision APIs and report their balanced accuracy across gender, race, and age groups.

To date, numerous large scale face image datasets (Huang et al., 2007; Kumar et al., 2011; Escalera et al., 2016; Yi et al., 2014; Liu et al., 2015; Joo et al., 2015; Parkhi et al., 2015; Guo et al., 2016; Kemelmacher-Shlizerman et al., 2016; Rothe et al., 2016; Cao et al., 2018; Merler et al., 2019) have been proposed and fostered research and development for automated face detection (Li et al., 2015b; Hu & Ramanan, 2017) , alignment (Xiong & De la Torre, 2013; Ren et al., 2014) , recognition (Taigman et al., 2014; Schroff et al., 2015) , generation (Yan et al., 2016; Bao et al., 2017; Karras et al., 2018; Thomas & Kovashka, 2018) , modification (Antipov et al., 2017; Lample et al., 2017; He et al., 2017) , and attribute classification (Kumar et al., 2011; Liu et al., 2015) .

These systems have been successfully translated into many areas including security, medicine, education, and social sciences.

Despite the sheer amount of available data, existing public face datasets are strongly biased toward Caucasian faces, and other races (e.g., Latino) are significantly underrepresented.

A recent study shows that most existing large scale face databases are biased towards "lighter skin" faces (around 80%), e.g. White, compared to "darker" faces, e.g. Black (Merler et al., 2019) .

This means the model may not apply to some subpopulations and its results may not be compared across different groups without calibration.

Biased data will produce biased models trained from it.

This will raise ethical concerns about fairness of automated systems, which has emerged as a critical topic of study in the recent machine learning and AI literature (Hardt et al., 2016; Corbett-Davies et al., 2017) .

For example, several commercial computer vision systems (Microsoft, IBM, Face++) have been criticized due to their asymmetric accuracy across sub-demographics in recent studies (Buolamwini & Gebru, 2018; Raji & Buolamwini, 2019) .

These studies found that the commercial face gender classification systems all perform better on male and on light faces.

This can be caused by the biases in their training data.

Various unwanted biases in image datasets can easily occur due to biased selection, capture, and negative sets (Torralba & Efros, 2011) .

Most public large scale face datasets have been collected from popular online media -newspapers, Wikipedia, or web search-and these platforms are more frequently used by or showing White people.

To mitigate the race bias in the existing face datasets, we propose a novel face dataset with an emphasis on balanced race composition.

Our dataset contains 108,501 facial images collected primarily from the YFCC-100M Flickr dataset (Thomee et al.) , which can be freely shared for a research purpose, and also includes examples from other sources such as Twitter and online newspaper outlets.

We define 7 race groups: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino.

Our dataset is well-balanced on these 7 groups (See Figures 1 and 2) Our paper makes three main contributions.

First, we emprically show that existing face attribute datasets and models learned from them do not generalize well to unseen data in which more nonWhite faces are present.

Second, we show that our new dataset performs better on novel data, not only on average, but also across racial groups, i.e. more consistently.

Third, to the best of our knowledge, our dataset is the first large scale face attribute dataset in the wild which includes Latino and Middle Eastern and differentiates East Asian and Southeast Asian.

Computer vision has been rapidly transferred into other fields such as economics or social sciences, where researchers want to analyze different demographics using image data.

The inclusion of major racial groups, which have been missing in existing datasets, therefore significantly enlarges the applicability of computer vision methods to these fields.

The goal of face attribute recognition is to classify various human attributes such as gender, race, age, emotions, expressions or other facial traits from facial appearance (Kumar et al., 2011; Joo et al., 2013; Liu et al., 2015) .

Table 1 summarizes the statistics of existing large scale public and in-the-wild face attribute datasets including our new dataset.

As stated earlier, most of these datasets were constructed from online sources and are typically dominated by the White race.

Face attribute recognition has been applied as a sub-component to other computer vision tasks such as face verification (Kumar et al., 2011) and person re-idenfication (Layne et al., 2012; Li et al., 2015a; Su et al., 2018) .

It is imperative to ensure that these systems perform evenly well on different gender and race groups.

Failing to do so can be detrimental to the reputations of individual service providers and the public trust about the machine learning and computer vision research community.

Most notable incidents regarding the racial bias include Google Photos recognizing African American faces as Gorilla and Nikon's digital cameras prompting a message asking "did someone blink?" to Asian users (Zhang, 2015) .

These incidents, regardless of whether the models were trained improperly or how much they actually affected the users, often result in the termination of the service or features (e.g. dropping sensitive output categories).

For this reason, most commercial service providers have stopped providing a race classifier.

Face attribute recognition is also used for demographic surveys performed in marketing or social science research, aimed at understanding human social behaviors and their relations to demographic backgrounds of individuals.

Using off-the-shelf tools (Amos et al., 2016; Baltrusaitis et al., 2018) and commercial services, social scientists have begun to use images of people to infer their demographic attributes and analyze their behaviors.

Notable examples are demographic analyses of social media users using their photographs (Chakraborty et al., 2017; Reis et al., 2017; Won et al., 2017; Xi et al., 2019; Wang et al., 2017) .

The cost of unfair classification is huge as it can over-or under-estimate specific sub-populations in their analysis, which may have policy implications.

AI and machine learning communities have increasingly paid attention to algorithmic fairness and dataset and model biases (Zemel et al., 2013; Corbett-Davies et al., 2017; Zou & Schiebinger, 2018; .

There exist many different definitions of fairness used in the literature (Verma & Rubin, 2018) .

In this paper, we focus on balanced accuracy-whether the attribute classification accuracy is independent of race and gender.

More generally, research in fairness is concerned with a model's ability to produce fair outcomes (e.g. loan approval) independent of protected or sensitive attributes such as race or gender.

Studies in algorithmic fairness have focused on either 1) discovering (auditing) existing bias in datasets or systems (Shankar et al., 2017; Buolamwini & Gebru, 2018; Kiritchenko & Mohammad, 2018; McDuff et al., 2019) , 2) making a better dataset (Merler et al., 2019; Alvi et al., 2018) , or 3) designing a better algorithm or model (Das et al., 2018; Alvi et al., 2018; Ryu et al., 2017; Zemel et al., 2013; Zafar et al., 2017) .

Our paper falls into the first two categories.

The main task of interest in our paper is (balanced) gender classification from facial images.

Buolamwini & Gebru (2018) demonstrated many commercial gender classification systems are biased and least accurate on dark-skinned females.

The biased results may be caused by biased datasets, such as skewed image origins (45% of images are from the U.S. in Imagenet) (Suresh et al., 2018) or biased underlying associations between scene and race in images (Stock & Cisse, 2018) .

It is, however, "infeasible to balance across all possible co-occurrences" of attributes (Hendricks et al., 2018) , except in a lab-controlled setting.

Therefore, the contribution of our paper is to mitigate, not entirely solve, the current limitations and biases of existing databases by collecting more diverse face images from non-White race groups.

We empirically show this significantly improves the generalization performance to novel image datasets whose racial compositions are not dominated by the White race.

Furthermore, as shown in Table 1 , our dataset is the first large scale in-the-wild face image dataset which includes Southeast Asian and Middle Eastern races.

While their faces share similarity with East Asian and White groups, we argue that not having these major race groups in datasets is a strong form of discrimination.

3 DATASET CONSTRUCTION 3.1 RACE TAXONOMY Our dataset defines 7 race groups: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino.

Race and ethnicity are different categorizations of humans.

Race is defined based on physical traits and ethnicity is based on cultural similarities (Schaefer, 2008) .

For example, Asian immigrants in Latin America can be of Latino ethnicity.

In practice, these two terms are often used interchangeably.

We first adopted a commonly accepted race classification from the U.S. Census Bureau (White, Black, Asian, Hawaiian and Pacific Islanders, Native Americans, and Latino).

Latino is often treated as an ethnicity, but we consider Latino a race, which can be judged from the facial appearance.

We then further divided subgroups such as Middle Eastern, East Asian, Southeast Asian, and Indian, as they look clearly distinct.

During the data collection, we found very few examples for Hawaiian and Pacific Islanders and Native Americans and discarded these categories.

All the experiments conducted in this paper were therefore based on 7 race classification.

An important criterion to measure dataset bias is on which basis the bias should be measured: skin color or race?

A few recent studies (Buolamwini & Gebru, 2018; Merler et al., 2019) use skin color as a proxy to racial or ethnicity grouping.

While skin color can be easily computed without subjective annotations, it has limitations.

First, skin color is heavily affected by illumination and light conditions.

The Pilot Parliaments Benchmark (PPB) dataset (Buolamwini & Gebru, 2018) only used profile photographs of government officials taken in well controlled lighting, which makes it non-in-the-wild.

Second, within-group variations of skin color are huge.

Even same individuals can show different skin colors over time.

Third, most importantly, race is a multidimensional concept whereas skin color (i.e. brightness) is one dimensional.

Figure 5 in Appendix shows the distributions of the skin color of multiple race groups, measured by Individual Typology Angle (ITA) (Wilkes et al., 2015) .

As shown here, the skin color provides no information to differentiate many groups such as East Asian and White.

Therefore, we explicitly use race and annotate the physical race by human annotators' judgments.

To complement the limits of race categorization, however, we also use skin color, measured by ITA, following the same procedure used by Merler et al. (2019) .

Many existing face datasets have been sourced from photographs of public figures such as politicians or celebrities (Kumar et al., 2011; Huang et al., 2007; Joo et al., 2015; Rothe et al., 2016; Liu et al., 2015) .

Despite the easiness of collecting images and ground truth attributes, the selection of these populations may be biased.

For example, politicians may be older and actors may be more attractive than typical faces.

Their images are usually taken by professional photographers in limited situations, leading to the quality bias.

Some datasets were collected via web search using keywords such as "Asian boy" (Zhang et al., 2017) .

These queries may return only stereotypical faces or prioritize celebrities in those categories rather than diverse individuals among general public.

Our goal is to minimize the selection bias introduced by such filtering and maximize the diversity and coverage of the dataset.

We started from a huge public image dataset, Yahoo YFCC100M dataset (Thomee et al.) , and detected faces from the images without any preselection.

A recent work also used the same dataset to construct a huge unfiltered face dataset (Diversity in Faces, DiF) (Merler et al., 2019) .

Our dataset is smaller but more balanced on race (See Figure 1) .

For an efficient collection, we incrementally increased the dataset size.

We first detected and annotated 7,125 faces randomly sampled from the entire YFCC100M dataset ignoring the locations of images.

After obtaining annotations on this initial set, we estimated demographic compositions of each country.

Based on this statistic, we adaptively adjusted the number of images for each country sampled from the dataset such that the dataset is not dominated by the White race.

Consequently, we excluded the U.S. and European countries in the later stage of data collection after we sampled enough White faces from those countries.

The minimum size of a detected face was set to 50 by 50 pixels.

This is a relatively smaller size compared to other datasets, but we find the attributes are still recognizable and these examples can actually make the classifiers more robust against noisy data.

We only used images with "Attribution" and "Share Alike" Creative Commons licenses, which allow derivative work and commercial usages.

We used Amazon Mechanical Turk to annotate the race, gender and age group for each face.

We assigned three workers for each image.

If two or three workers agreed on their judgements, we took the values as ground-truth.

If all three workers produced different responses, we republished the image to another 3 workers and subsequently discarded the image if the new annotators did not agree.

These annotations at this stage were still noisy.

We further refined the annotations by training a model from the initial ground truth annotations and applying back to the dataset.

We then manually re-verified the annotations for images whose annotations differed from model predictions.

We first measure how skewed each dataset is in terms of its race composition.

For the datasets with race annotations, we use the reported statistics.

For the other datasets, we annotated the race labels for 3,000 random samples drawn from each dataset.

See Figure 1 for the result.

As expected, most existing face attribute datasets, especially the ones focusing on celebrities or politicians, are biased toward the White race.

Unlike race, we find that most datasets are relatively more balanced on gender ranging from 40%-60% male ratio.

To compare model performance of different datasets, we used an identical model architecture, ResNet-34 , to be trained from each dataset.

We used ADAM optimization (Kingma & Ba, 2014 ) with a learning rate of 0.0001.

Given an image, we detected faces using the dlib's (dlib.net) CNN-based face detector (King, 2015) and ran the attribute classifier on each face.

The experiment was done in PyTorch.

Throughout the evaluations, we compare our dataset with three other datasets: UTKFace (Zhang et al., 2017) , LFWA+, and CelebA (Liu et al., 2015) .

Both UTKFace and LFWA+ have race annotations, and thus, are suitable for comparison with our dataset.

CelebA does not have race annotations, so we only use it for gender classification.

See Table 1 for more detailed dataset characteristics.

.971 --* CelebA doesn't provide race annotations.

The result was obtained from the whole set (white and non-white).

† FairFace defines 7 race categories but only 4 races (White, Black, Asian, and Indian) were used in this result to make it comparable to UTKFace.

Using models trained from these datasets, we first performed cross-dataset classifications, by alternating training sets and test sets.

Note that FairFace is the only dataset with 7 races.

To make it compatible with other datasets, we merged our fine racial groups when tested on other datasets.

CelebA does not have race annotations but was included for gender classification.

Tables 2 and 3 show the classification results for race, gender, and age on the datasets across subpopulations.

As expected, each model tends to perform better on the same dataset on which it was trained.

However, the accuracy of our model was highest on some variables on the LFWA+ dataset and also very close to the leader in other cases.

This is partly because LFWA+ is the most biased dataset and ours is the most diverse, and thus more generalizable dataset.

To test the generalization performance of the models, we consider three novel datasets.

Note that these datasets were collected from completely different sources than our data from Flickr and not used in training.

Since we want to measure the effectiveness of the model on diverse races, we chose the test datasets that contain people in different locations as follows.

Geo-tagged Tweets.

First we consider images uploaded by Twitter users whose locations are identified by geo-tags (longitude and latitude), provided by (Steinert-Threlkeld, 2018) .

From this set, we chose four countries (France, Iraq, Philippines, and Venezuela) and randomly sampled 5,000 faces.

Media Photographs.

Next, we also use photographs posted by 500 online professional media outlets.

Specifically, we use a public dataset of tweet IDs (Littman et al., 2017) posted by 4,000 known media accounts, e.g. @nytimes.

Note that although we use Twitter to access the photographs, these tweets are simply external links to pages in the main newspaper sites.

Therefore this data is considered as media photographs and different from general tweet images mostly uploaded by ordinary users.

We randomly sampled 8,000 faces from the set.

Protest Dataset.

Lastly, we also use a public image dataset collected for a recent protest activity study (Won et al., 2017) .

The authors collected the majority of data from Google Image search by using keywords such as "Venezuela protest" or "football game" (for hard negatives).

The dataset exhibits a wide range of diverse race and gender groups engaging in different activities in various countries.

We randomly sampled 8,000 faces from the set.

These faces were annotated for gender, race, and age by Amazon Mechanical Turk workers.

Table 4 : Gender classification accuracy measured on external validation datasets across gender-race groups.

Table 7 shows the classification accuracy of different models.

Because our dataset is larger than LFWA+ and UTKFace, we report the three variants of the FairFace model by limiting the size of a training set (9k, 18k, and Full) for fair comparisons.

Improved Accuracy.

As clearly shown in the result, the model trained by FairFace outperforms all the other models for race, gender, and age, on the novel datasets, which have never been used in training and also come from different data sources.

The models trained with fewer training images (9k and 18k) still outperform other datasets including CelebA which is larger than FairFace.

This suggests that the dataset size is not the only reason for the performance improvement.

Balanced Accuracy.

Our model also produces more consistent results -for race, gender, age classification -across different race groups compared to other datasets.

We measure the model consistency by standard deviations of classification accuracy measured on different sub-populations, as shown in Table 5 .

More formally, one can consider conditional use accuracy equality (Berk et al.) or equalized odds (Hardt et al., 2016) as the measure of fair classification.

For gender classification:

where Y is the predicted gender, Y is the true gender, A refers to the demographic group, and D is the set of different demographic groups being considered (race).

When we consider different gender groups for A, this needs to be modified to measure accuracy equality Berk et al.:

We therefore define the maximum accuracy disparity of a classifier as follows: Table 4 shows the gender classification accuracy of different models measured on the external validation datasets for each race and gender group.

The FairFace model achieves the lowest maximum accuracy disparity.

The LFWA+ model yields the highest disparity, strongly biased toward the male category.

The CelebA model tends to exhibit a bias toward the female category as the dataset contains more female images than male.

The FairFace model achieves less than 1% accuracy discrepancy between male ↔ female and White ↔ non-White for gender classification (Table 7) .

All the other models show a strong bias toward the male class, yielding much lower accuracy on the female group, and perform more inaccurately on the non-White group.

The gender performance gap was the biggest in LFWA+ (32%), which is the smallest among the datasets used in the experiment.

Recent work has also reported asymmetric gender biases in commercial computer vision services (Buolamwini & Gebru, 2018) , and our result further suggests the cause is likely due to the unbalanced representation in training data.

Data Coverage and Diversity.

We further investigate dataset characteristics to measure the data diversity in our dataset.

We first visualize randomly sampled faces in 2D space using t-SNE (Maaten & Hinton, 2008) as shown in Figure 3 .

We used the facial embedding based on ResNet-34 from dlib, which was trained from the FaceScrub dataset (Ng & Winkler, 2014) , the VGG-Face dataset (Parkhi et al., 2015) and other online sources, which are likely dominated by the White faces.

The faces in FairFace are well spread in the space, and the race groups are loosely separated from each other.

This is in part because the embedding was trained from biased datasets, but it also suggests that the dataset contains many non-typical examples.

LFWA+ was derived from LFW, which was developed for face recognition, and therefore contains multiple images of the same individuals, i.e. clusters.

UTKFace also tends to focus more on local clusters compared to FairFace.

To explicitly measure the diversity of faces in these datasets, we examine the distributions of pairwise distance between faces (Figure 4 ).

On the random subsets, we first obtained the same 128-dimensional facial embedding from dlib and measured pair-wise distance.

Figure 4 shows the CDF functions for 3 datasets.

As conjectured, UTKFace had more faces that are tightly clustered together and very similar to each other, compared to our dataset.

Surprisingly, the faces in LFWA+ were shown very diverse and far from each other, even though the majority of the examples contained a white face.

We believe this is mostly due to the fact that the face embedding was also trained on a very similar white-oriented dataset which will be effective in separating white faces, not because the appearance of their faces is actually diverse. (See Figure 2) Figure 4: Distribution of pairwise distances of faces in 3 datasets measured by L1 distance on face embedding.

Previous studies have reported that popular commercial face analytic models show inconsistent classification accuracies across different demographic groups (Buolamwini & Gebru, 2018; Raji & Buolamwini, 2019) .

We used the FairFace images to test several online APIs for gender classification: Microsoft Face API, Amazon Rekognition, IBM Watson Visual Recognition, and Face++.

Compared to prior work using politicians' faces, our dataset is much more diverse in terms of race, age, expressions, head orientation, and photographic conditions, and thus serves as a much better benchmark for bias measurement.

We used 7,476 random samples from FairFace such that it contains an equal number of faces from each race, gender, and age group.

We left out children under the age of 20, as these pictures were often ambiguous and the gender could not be determined for certain.

The experiments were conducted on August 13th -16th, 2019.

.923 .966 .901 .955 .925 .949 .918 .914 .921 .987 .951 .979 .906 .983 .941 .030 Microsoft .822 .777 .766 .717 .824 .775 .852 .794 .843 .848 .863 .790 .839 .772 .806 .042 Face++ .888 .959 .805 .944 .876 .904 .884 .897 .865 .981 .770 .968 .822 .978 .896 .066 IBM .910 .966 .758 .927 .899 .910 .852 .919 .884 .972 .811 .957 .871 .959 .900 .061 FairFace .987 .991 .964 .974 .966 .979 .978 .961 .991 .989 .991 .987 .972 .991 .980 .011 *Microsoft .973 .998 .962 .967 .963 .976 .960 .957 .983 .993 .975 .991 .966 .993 .975 .014 *Face++ .893 .968 .810 .956 .878 .911 .886 .899 .870 .983 .773 .975 .827 .983 .901 .067 *IBM .914 .981 .761 .956 .909 .920 .852 .926 .892 .977 .819 .975 .881 .979 .910 .066 Table 6 shows the gender classification accuracies of the tested APIs.

These APIs first detect a face from an input image and classify its gender.

Not all 7,476 faces were detected by these APIs with the exception of Amazon Rekognition which detected all of them.

Table 8 in Appendix reports the detection rate.

1 We report two sets of accuracies: 1) treating mis-detections as mis-classifications and 2) excluding mis-detections.

For comparison, we included a model trained with our dataset to provide an upper bound for classification accuracy.

Following prior work (Merler et al., 2019) , we also show the classification accuracy as a function of skin color in Figure 6 .

The results suggest several findings.

First, all tested gender classifiers still favor the male category, which is consistent with the previous report (Buolamwini & Gebru, 2018) .

Second, dark-skinned females tend to yield higher classification error rates, but there exist many exceptions.

For example, Indians have darker skin tones ( Figure 5 ), but some APIs (Amazon and MS) classified them more accurately than Whites.

This suggests skin color alone, or any other individual phenotypic feature, is not a sufficient guideline to study model bias.

Third, face detection can also introduce significant gender bias.

Microsoft's model failed to detect many male faces, an opposite direction from the gender classification bias.

This was not reported in previous studies which only used clean profile images of frontal faces.

This paper proposes a novel face image dataset balanced on race, gender and age.

Compared to existing large-scale in-the-wild datasets, our dataset achieves much better generalization classification performance for gender, race, and age on novel image datasets collected from Twitter, international online newspapers, and web search, which contain more non-White faces than typical face datasets.

We show that the model trained from our dataset produces balanced accuracy across race, whereas other datasets often lead to asymmetric accuracy on different race groups.

This dataset was derived from the Yahoo YFCC100m dataset (Thomee et al.) for the images with Creative Common Licenses by Attribution and Share Alike, which permit both academic and commercial usage.

Our dataset can be used for training a new model and verifying balanced accuracy of existing classifiers.

Algorithmic fairness is an important aspect to consider in designing and developing AI systems, especially because these systems are being translated into many areas in our society and affecting our decision making.

Large scale image datasets have contributed to the recent success in computer vision by improving model accuracy; yet the public and media have doubts about its transparency.

The novel dataset proposed in this paper will help us discover and mitigate race and gender bias present in computer vision systems such that such systems can be more easily accepted in society.

A APPENDIX Figure 5: Individual Typology Angle (ITA), i.e. skin color, distribution of different races measured in our dataset.

@highlight

A new face image dataset for balanced race, gender, and age which can be used for bias measurement and mitigation