Representing entities and relations in an embedding space is a well-studied approach for machine learning on relational data.

Existing approaches however primarily focus on simple link structure between a finite set of entities, ignoring the variety of data types that are often used in relational databases, such as text, images, and numerical values.

In our approach, we propose a multimodal embedding using different neural encoders for this variety of data, and combine with existing models to learn embeddings of the entities.

We extend existing datasets to create two novel benchmarks, YAGO-10-plus and MovieLens-100k-plus, that contain additional relations such as textual descriptions and images of the original entities.

We demonstrate that our model utilizes the additional information effectively to provide further gains in accuracy.

Moreover, we test our learned multimodal embeddings by using them to predict missing multimodal attributes.

Knowledge bases (KB) are an essential part of many computational systems with applications in variety of domains, such as search, structured data management, recommendations, question answering, and information retrieval.

However, KBs often suffer from incompleteness, noise in their entries, and inefficient inference.

Due to these deficiencies, learning the relational knowledge representation has been a focus of active research BID1 BID10 BID21 BID29 BID4 .

These approaches represent relational triples, consisting of a subject entity, relation, and an object entity, by estimating fixed, low-dimensional representations for each entity and relation from observations, thus encode the uncertainty and infer missing facts accurately and efficiently.

The subject and the object entities come from a fixed, enumerable set of entities that appear in the knowledge base.

Knowledge bases in the real world, however, are rich with a variety of different data types.

Apart from a fixed set of entities, the relations often not only include numerical attributes (such as ages, dates, financial, and geoinformation), but also textual attributes (such as names, descriptions, and titles/designations) and images (profile photos, flags, posters, etc.) .

Although these different types of relations cannot directly be represented as links in a graph over a fixed set of nodes, they can be crucial pieces of evidences for knowledge base completion.

For example the textual descriptions and images might provide evidence for a person's age, profession, and designation.

Further, this additional information still contains similar limitations as the conventional link data; they are often missing, may be noisy when observed, and for some applications, may need to be predicted in order to address a query.

There is thus a crucial need for relational modeling that goes beyond just the link-based, graph view of knowledge-base completion, is able to utilize all the observed information, and represent the uncertainty of multimodal relational evidence.

In this paper, we introduce a multimodal embedding approach for modeling knowledge bases that contains a variety of data types, such as textual, images, numerical, and categorical values.

Although we propose a general framework that can be used to extend many of the existing relational modeling approaches, here we primary apply our method to the DistMult approach .

We extend this approach that learns a vector for each entity and relation by augmenting it with additional neural encoders for different evidence data types.

For example, when the object of a triple is an image, we encode it into a fixed-length vector using a CNN, while the textual attributes are encoded using sequential embedding approaches like LSTMs.

The scoring module remains identical; given the vector representations of the subject, relation, and object of a triple, this module produces a score indicating the probability that the triple is correct.

This unified model allows for flow of the information across the different relation types, enabling more accurate modeling of relational data.

We provide an evaluation of our proposed approach on two relational databases.

Since we are introducing a novel formulation in the relational setting, we introduce two benchmarks, created by extending the existing YAGO-10 and MovieLens-100k datasets to include additional relations such as textual descriptions, numerical attributes, and images of the original entities.

In our evaluation, we demonstrate that our model utilizes the additional information effectively to provide gains in link-prediction accuracy, and present a breakdown of how much each relation benefits from each type of the additional information.

We also present results that indicate the learned multimodal embeddings are capable of predicting the object entities for different types of data which is based on the similarity between those entities.

Knowledge bases (KB) often contain different types of information about entities including links, textual descriptions, categorical attributes, numerical values, and images.

In this section, we briefly introduce the existing approaches to the embedded relational modeling that focus on modeling of the linked data using dense vectors.

We then describe our model that extends these approaches to the multimodal setting, i.e., modeling the KB using all the different information.

The goal of the relational modeling is to train a machine learning model that can score the truth value of any factual statement, represented here as a triplet of subject, relation and object, (s, r, o), where s, o ∈ ξ, a set of entities, and r ∈ R, a set of relations.

Accordingly, the link prediction problem can be defined as learning a scoring function ψ : ξ × R × ξ → R (or sometimes, [0, 1]).

In order to learn the parameters of such a model, training data consists of the observed facts for the KB, i.e., a set of triples, which may be incomplete and noisy.

In the last few years, the methods that have achieved impressive success on this task consist of models that learn fixed-length vectors, matrices, or tensors for each entity in ξ and relation in R, with the scoring function consisting of varying operators applied to these learned representations (described later in Section 3).

Although our proposed framework can be used with many of the existing relational models, here we focus on the DistMult approach because of its simplicity, popularity, and high accuracy.

In DistMult, each entity i is mapped to a d-dimensional dense vector (e i ∈ R d×1 ) and each relation r to a diagonal matrix R r ∈ R d×d , and consequently, the score for any triple (s, r, o) is computed as: ψ(s, r, o) = e T s R r e o .

Since we cannot guarantee that the unobserved triples are true negatives, we use a pairwise ranking loss that tries to score existing (positive) triples higher than non-existing triples (negatively sampled), as: DISPLAYFORM0 where D + , D − denote the set of existing and non-existing (sampled) triples, γ is the width of margin, φ i is the score of the i th triple and Θ is the set of all embeddings.

Following BID2 , we generate negative samples of training triplets by replacing either subject or object entity with a random entity.

DistMult thus learns entity and relation representations that encode the knowledge base, and can be used for completion, queries, or cleaning.

Existing approaches to this problem assume that the subjects and the objects are from a fixed set of entities ξ, and thus are treated as indices into that set.

However, in the most of the real-world KBs, the objects of triples (s, r, o) are not restricted to be in some indexed set, and instead, can be of any data type such as numerical, categorical, images, and text.

In order to incorporate such objects into the existing relational models like DistMult, we propose to learn embeddings for any of these types of Architecture of the proposed work that, given any movie and any of its attributes, like the title, poster, genre, or release year, uses domain-specific encoders to embed each attribute value.

The embeddings of the subject entity, the relation, and the object value are then used to score the truth value of the triple by the Scorer, using the DistMult operation.data.

We utilize recent advances in deep learning to construct encoders for these objects to represent them, essentially providing an embedding e o for any object value.

The overall goal remains the same: the model needs to utilize all the observed subjects, objects, and relations, across different data types, in order to estimate whether any fact (s, r, o) holds.

We present an example of an instantiation of our model for a knowledge base containing movie details in FIG0 .

For any triple (s, r, o), we embed the subject (movie) and the relation (such as title, release year, or poster) using a direct lookup.

For the object, depending on the domain (indexed, string, numerical, or image, respectively), we use an appropriate encoder to compute its embedding e o .

We use appropriate encoders for each data type, such as CNNs for images and LSTMs for text.

As in DistMult, these embeddings are used to compute the score of the triple.

Training such a model remains identical to DistMult, except that for negative sampling, here we replace the object entity with a random entity from the same domain as the object (either image, text, numerical or etc.).

Here we describe the encoders we use for multimodal objects.

Structured knowledge Consider a triplet of information in the form of (s, r, o).

To represent the subject entity s and the relation r as independent embedding vectors (as in previous work), we pass their one-hot encoding through a dense layer.

Furthermore, for the case that the object entity is categorical, we embed it through a dense layer with a recently introduced selu activation BID14 , with the same number of nodes as the embedding space dimension.

Numerical Objects in the form of real numbers can provide a useful source of information and are often quite readily available.

We use a feed forward layer, after applying basic normalization, in order to embed the numbers into the embedding space.

Note that we are projecting them to a higher-dimensional space, from R → R d .

It is worth contrasting this approach to the existing methods that often treat numbers as distinct entities, i.e., learning independent vectors for numbers 39 and 40, relying only on data to learn that these values are similar to each other.

Text Since text can be used to store a wide variety of different types of information, for example names versus paragraph-long descriptions, we create different encoders depending on the lengths of the strings involved.

For attributes that are fairly short, such as names and titles, we use character-based stacked, bidirectional LSTM to encode these strings, similar to BID31 , using the final output of the top layer as the representation of the string.

For strings that are much longer, such as detailed descriptions of entities consisting of multiple sentences, we treat them as a sequence of words, and use a CNN over the word embeddings, similar to BID6 , in order to embed such values.

These two encoders provide a fixed length encoding that has been shown for multiple tasks to be an accurate semantic representation of the strings (Dos Santos and Gatti, 2014).Images Images can also provide useful evidence for modeling entities.

For example, we can extract person's details such as gender, age, job, etc., from image of the person BID17 , or location information such as its approximate coordinates, neighboring locations, and size from map images BID33 .

A variety of models have been used to compactly represent the semantic information in the images, and have been successfully applied to tasks such as image classification, captioning BID12 , and question-answering .

To embed images such that the encoding represents such semantic information, we use the last hidden layer of VGG pretrained network on Imagenet BID24 , followed by compact bilinear pooling , to obtain the embedding of the images.

Other Data Types Although in this paper we only consider the above data types, there are many others that can be utilized for learning KB representations.

Our framework is amenable to such data types as long as an appropriate encoder can be designed.

For example, speech/audio data can be accurately encoded using CNNs BID0 , time series data using LSTM and other recurrent neural networks BID3 , and geospatial coordinates using feedforward networks BID15 .

We leave the modeling of these types of objects for the future work.

There is a rich literature on modeling knowledge bases using low-dimensional representations, differing in the operator used to score the triples.

In particular, they use matrix and tensor multiplication BID19 BID25 , euclidean distance BID2 BID32 BID18 , circular correlation BID21 , or the Hermitian dot product BID29 as scoring function.

However, the objects for all of these approaches are a fixed set of entities, i.e., they only embed the structured links between the entities.

Here, we use different types of information such as text, numerical values and images in the encoding component, by treating them as relational triples of information.

A number of methods utilize a single extra type of information as the observed features for entities, by either merging, concatenating, or averaging the entity and its features to compute its embeddings, such as numerical values (Garcia-Duran and Niepert, 2017), images BID34 , and text BID27 BID30 .

Along the same line, BID31 address multilingual relation extraction task to attain a universal schema by considering raw text with no annotation as extra feature and using matrix factorization to jointly embed KB and textual relations BID22 .

In addition to treating the extra information as features, graph embedding approaches BID4 BID23 BID13 consider fixed number of attributes as a part of encoding component to achieve more accurate embedding.

The difference between our model and these mentioned approaches is three-fold: (1) we are the first to use different types of information in a unified model, (2) we treat these different type of information (numerical, text, image) as relational triples of structured knowledge instead of predetermined features, i.e., first-class citizens of the data, and not auxiliary features, and (3) our model represents uncertainty in them, supporting the missing values and facilitating the recovery of the lost information, which is not possible with previous approaches.

To evaluate the performance of our mutimodal relational embeddings approach, we provide two new benchmarks by extending existing datasets.

The first one is built by adding posters to movie recommendation dataset, MovieLens 100k, and the second one by adding image and textual information for YAGO-10 dataset from DBpedia and numerical information from YAGO-3 database.

We will release TAB0 provide the statistics of these datasets 1 .

We start with the MovieLens-100k dataset 2 BID11 , a popular benchmark for recommendation system for predicting user ratings with contextual features, containing 100, 000 ratings from around 1000 users on 1700 movies.

MovieLens already contains rich relational data about occupation, gender, zip code, and age for users and genre, release date, and the titles for movies.

We consider the genre attribute for each movie as a binary vector with length 19 (number of different genres provided by MovieLens).

We use this representation because each movie genre is a combination of multiple, related categories.

Moreover, we collect the movie posters for each movie from TMDB 3 .

We treat the 5-point ratings as five different relations in KB triple format, i.e., (user, r = 5, movie), and evaluate the rating predictions as data for other relations is introduced into to the model.

We use 10% of rating samples as the validation data.

MovieLens has a variety of data types, it is still quite small, and is over a specialized domain.

We also consider a second dataset that is much more appropriate for knowledge graph completion and is popular for link prediction, the YAGO3-10 knowledge graph BID26 BID20 .

This graph consists of around 120,000 entities, such as people, locations, and organizations, and 37 relations, such as kinship, employment, and residency, and thus much closer to the traditional information extraction goals.

We extend this dataset with the textual description (as an additional relation) and the images associated with each entity (we have collected images for half of the entities), provided by DBpedia 4 BID16 .

We also identify few more additional relations such as wasBornOnDate, happenedOnDate, etc, that have dates as values.

In this section, we first evaluate the ability of our model to utilize the multimodal information by comparing to the DistMult method through a variety of link prediction tasks.

Then, by considering the recovery of missing multimodal values (text, images, and categorical) as the motivation, we examine the capability of our model in genre prediction on MovieLens and date prediction on YAGO.

Further, we provide a qialitative analysis on title, poster and genre prediction for MovieLens data.

To facilitate a fair comparison we implement all methods using the identical loss and optimization for training, i.e., AdaGrad and the ranking loss.

We tune all the hyperparameters on the validation data and use grid search to find the best hyperparameters, such as regularization parameter λ = [10 −6 , 3 × 10 −6 ], dimensionality of embedding d = [128, 200, 250, 360] and number of training iterations T = 12k.

For evaluation we use three metrics: mean reciprocal rank (MRR), Hits@K, and RMSE, which are commonly used by existing approaches.

In this section, we evaluate the capability of our model in the link prediction task.

The goal is to calculate MRR and Hits@ metric (ranking evaluations) of recovering the missing entities from triples in the test dataset, performed by ranking all the entities and computing the rank of the correct entity.

Similar to previous works, we here focus on providing the results in a filtered setting, that is we only rank triples in the test data against the ones that never appear in either train or test datasets.

MovieLens-100k-plus We train the model for MovieLens using Rating as the relation between users and movies.

For encoding other relations, we use a character-level LSTM for the movie titles, a feed-forward network for age, zip code, and release date, and finally, we use a VGG network on the posters (for every other relation we use dense layer embeddings).

TAB2 shows the link (rating) prediction evaluation on MovieLens dataset when test data is consisting only of rating triples.

We calculate our metrics by ranking the five relations representing the ratings instead of object entities.

The reason behind presenting these metrics is the fact that they are compatible with classification accuracy evaluation on recommendation system algorithms.

We label models using rating information as R, movie-attribute as M, user-attribute as U, movies' title as T, and poster encoding as P.As it is shown, the model R+M+U+T outperforms other methods with a considering gap, which shows the importance of incorporating the extra information.

Furthermore, Hits@1 for our baseline model is 40%, which matches existing recommendation systems BID9 .

Based on results it seems that adding titles information has a higher impact compared to the poster information.

YAGO-10-plus The result of link prediction on our YAGO dataset is provided in TAB3 .

We label models using structured information as S, entity-description as D, numerical information as N, and entity-image as I. We see that the model that encodes all type of information consistently performs better than other models, indicating that the model is effective in utilizing the extra information.

On the other hand, the model that uses only text performs the second best, suggesting the entity descriptions contain more information than others.

It is notable that model S is outperformed by all other models, demonstrating the importance of using different data types for attaining higher accuracy.

We also include the performance of a recently introduced approach, ConvE BID4 that is the state-of-art on this dataset.

Although it achieves higher results than our models (which are based on DistMult), it primarily differs from DistMult in how it scores the triples, and thus we can also incorporate our approach into ConvE in future.

Relation Breakdown We perform additional analysis on the YAGO dataset to gain a deeper understanding of the performance of our model.

TAB4 compares our models on the top five most frequent relations.

As shown, the model that includes textual description significantly benefits isAffiliatedTo, isLocatedIn and wasBornIn relations, as this information often appears in text.

Moreover, images are useful to detect genders (hasGender), while for the relation playsFor, numerical (dates) are more effective than images.

Here we present an evaluation on multimodal attributes prediction (text, image and numerical) on our benchmarks.

Note that approaches that use this information as features cannot be used to recovering missing information, i.e., they cannot predict any relation that is not to existing entities.

TAB5 shows the link prediction evaluation on MovieLens when test data is consisting only of movies' genre.

The test dataset is obtained by keeping only 80% of movies' genre information in the training dataset and treat the rest as the test data.

The evaluation metrics is calculated by ranking the test triplets in comparison to all 216 different possible combination of genres (binary vectors with length 19) provided by MovieLens.

As shown, model utilizing all the information outperforms other methods by a considerable gap, indicating that our model is able to incorporate information from posters and titles to predict the genre of movies (with posters providing more information than titles).

Along the same line, TAB6 shows the link prediction evaluation on YAGO-10-plus when test data is consisting only of numerical triples.

The test dataset is obtained by holding out 10% of numerical information in the training dataset.

Furthermore, we only consider the the numerical values (dates) that are larger than 1000 to obtain a denser distribution.

To make a prediction on the year, we divide the numerical interval [1000, 2017] to 1000 bins, and for each triple in the test data find the mid-point of the bin that the model scored the highest; we use this value to compute the RMSE.

As we can see, S+N+D+I outperform other methods with a considering gap, demonstrating our model utilizes other multimodal values for more fruitful modeling of the numerical information.

Querying Multimodal Attributes Although we only encode multimodal data, and cannot decode in this setting directly, we provide examples in which we query for a multimodal attribute (like the "Action, Crime, Drama" "Drama, Romance, War, Western", "Drama, Romance, War", "Drama, War" "Die Hard" "The Band Wagon", "Underground", "Roseanna's Grave" "Action, Thriller" "Drama, War", "Action, Drama, War", "Comedy, Drama, War" poster), and rank all existing values (other posters) to observe which ones get ranked the highest.

In other words, we are asking the model, if the actual poster is not available, which of the existing posters would the model recommend as a replacement (and same for title and genre).

In TAB7 we show top-3 predicted values.

We can see that the selected posters have visual similarity to the original poster in regarding the background, and appearance of a face and the movie title in the poster.

Along the same line, genres, though not exact, are quite similar as well (at least one of original genres appear in the predicted ones).

And finally, the selected titles are also somewhat similar in meaning, and in structure.

For example, two of the predicted titles for "Die Hard" have something to do with dying and being buried.

Furthermore, both "The Godfather" and its first predicted title "101 dalmatians" consist of a three-character word followed by a longer word.

We leave extensions that directly perform such decoding to future work.

Motivated by the need to utilize multiple source of information to achieve more accurate link prediction we presented a novel neural approach to multimodal relational learning.

In this paper we introduced a universal link prediction model that uses different types of information to model knowledge bases.

We proposed a compositional encoding component to learn unified entity embedding that encode the variety of information available for each entity.

In our analysis we show that our model in comparison to a common link predictor, DistMult, can achieve higher accuracy, showing the importance of employing the available variety of information for each entity.

Since all the existing datasets are designed for previous methods, they lack mentioned kind of extra information.

In result, we introduced two new benchmarks YAGO-10-plus and MovieLens-100k-plus, that are extend version of existing datasets.

Further, in our evaluation, we showed that our model effectively utilizes the extra information in order to benefit existing relations.

We will release the datasets and the open-source implementation of our models publicly.

There are number of avenues for future work.

We will investigate the performance of our model in completing link prediction task using different scoring function and more elaborate encoding component and objective function.

We are also interested in modeling decoding of multimodal values in the model itself, to be able to query these values directly.

Further, we plan to explore efficient query algorithms for embedded knowledge bases, to compete with practical database systems.

<|TLDR|>

@highlight

Extending relational modeling to support multimodal data using neural encoders.

@highlight

This paper proposes to perform link prediction in Knowledge Bases by supplementing the original entities with multimodal information, and presents a model able to encode all sorts of information when scoring triples.

@highlight

The paper is about incorporating information from different modalities into link prediction approaches