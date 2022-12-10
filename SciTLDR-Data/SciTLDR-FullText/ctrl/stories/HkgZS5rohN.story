Structured tabular data is the most commonly used form of data in industry according to a Kaggle ML and DS Survey.

Gradient Boosting Trees, Support Vector Machine, Random Forest, and Logistic Regression are typically used for classification tasks on tabular data.

The recent work of Super Characters method using two-dimensional word embeddings achieved state-of-the-art results in text classification tasks, showcasing the promise of this new approach.

In this paper, we propose the SuperTML method, which borrows the idea of Super Characters method and two-dimensional embeddings to address the problem of classification on tabular data.

For each input of tabular data, the features are first projected into two-dimensional embeddings like an image, and then this image is fed into fine-tuned ImageNet CNN models for classification.

Experimental results have shown that the proposed SuperTML method have achieved state-of-the-art results on both large and small datasets.

In data science, data is categorized into structured data and unstructured data.

Structured data is also known as tabular data, and the terms will be used interchangeably.

Anthony Goldbloom, the founder and CEO of Kaggle observed that winning techniques have been divided by whether the data was structured or unstructured BID12 .

Currently, DNN models are widely applied for usage on unstructured data such as image, speech, and text.

According to Anthony, "When the data is unstructured, its definitely CNNs and RNNs that are carrying the day" BID12 .

The successful CNN model in the ImageNet competition BID8 has outperformed human for image classification task by ResNet BID6 since 2015.On the other side of the spectrum, machine learning models such as Support Vector Machine (SVM), Gradient Boosting Trees (GBT), Random Forest, and Logistic Regression, have been used to process structured data.

According to a recent survey of 14,000 data scientists by Kaggle (2017) , a subdivision of structured data known as relational data is reported as the most popular type of data in industry, with at least 65% working daily with relational data.

Regarding structured data competitions, Anthony says that currently XGBoost is winning practically every competition in the structured data category BID4 .

XGBoost BID2 is one popular package implementing the Gradient Boosting method.

Recent research has tried using one-dimensional embedding and implementing RNNs or one-dimensional CNNs to address the TML (Tabular Machine Learning) tasks, or tasks that deal with structured data processing BID7 BID11 , and also categorical embedding for tabular data with categorical features BID5 .

However, this reliance upon onedimensional embeddings may soon come to change.

Recent NLP research has shown that the two-dimensional embedding of the Super Characters method BID9 is capable of achieving state-of-the-art results on large dataset benchmarks.

The Super Characters method is a two-step method that was initially designed for text classification problems.

In the first step, the characters of the input text are drawn onto a blank image.

In the second step, the image is fed into two-dimensional CNN models for classification.

The two-dimensional CNN models are trained by fine-tuning from pretrained models on large image dataset, e.g. ImageNet.

In this paper, we propose the SuperTML method, which borrows the concept of the Super Characters method to address TML problems.

For each input, tabular features are first projected onto a two-dimensional embedding and fed into fine-tuned two-dimensional CNN models for classification.

The proposed SuperTML method handles the categorical type and missing values in tabular data automatically, without need for explicit conversion into numerical type values.

The SuperTML method is motivated by the analogy between TML problems and text classification tasks.

For any sample given in tabular form, if its features are treated like stringified tokens of data, then each sample can be represented as a concatenation of tokenized features.

By applying this paradigm of a tabular sample, the existing CNN models used in Super Characters method could be extended to be applicable to TML problems.

As mentioned in the introduction, the combination of twodimensional embedding (a core competency of the Super Characters methodology) and pre-trained CNN models has achieved state-of-the-art results on text classification tasks.

However, unlike the text classification problems studied in BID9 , tabular data has features in separate dimensions.

Hence, generated images of tabular data should reserve some gap between features in different dimensions in order to guarantee that features will not overlap in the generated image.

SuperTML is composed of two steps, the first of which is two-dimensional embedding.

This step projects features in the tabular data onto the generated images, which will be called the SuperTML images in this paper.

The conversion of tabular training data to SuperTML image is illustrated in Figure 1 , where a collection of samples containing four tabular features is being sorted.

The second step is using pretrained CNN models to finetune on the generated SuperTML images.

Figure 1 only shows the generation of SuperTML images for the training data.

It should be noted that for inference, each instance of testing data goes through the same preprocessing to generate a SuperTML image (all of which use the same configuration of two-dimensional embedding) before getting fed into the CNN classification model.

the generated SuperTML images.

9: return the trained CNN model on the tabular data known as SuperTML VF, is described in Algorithm 1.To make the SuperTML more autonomous and remove the dependency on feature importance calculation done in Algorithm 1, the SuperTML EF method is introduced in Algorithm 2.

It allocates the same size to every feature, and thus tabular data can be directly embedded into SuperTML images without the need for calculating feature importance.

This algorithm shows even better results than 1, which will be described more in depth later in the experimental section.

The data statistics from UCI Machine Learning Repository is shown in TAB2 .

"

This is perhaps the best known database to be found in the pattern recognition literature"1 .

The Iris dataset is widely used in machine learning courses and tutorials.

FIG2 shows an example of a generated SuperTML image, created using Iris data.

The experimental results of using SEnet-154 shown in Table 2 for each feature of the sample do 3:Draw the feature in the same font size without overlapping, such that the total features of the sample will occupy the image size as much as possible.

For this dataset 2 , we use SuperTML VF, which gives features different sizes on the SupterTML image according to their importance score.

The feature importance score is obtained using the XGBoost package BID2 .

One example of a SuperTML image created using data from this dataset is shown in FIG2 .

The results in Table 2 shows that the SuperTML method obtained a slightly better accuracy than XGBoost on this dataset.

The task of this Adult dataset 3 is to predict whether a persons income is larger or smaller than 50,000 dollars per year based on a collection of surveyed data.

For categorical features that are represented by strings, the Squared English Word (SEW) method BID10 Figure 3 .

SuperTML VF image example from Adult dataset.

This sample has age = 59, capital gain = 0, capital loss = 0, hours per week = 40, fnlweight = 372020, education number = 13, occupation = "?" (missing value), marital status = "Married-civ-spouse", relationship = "Husband", workclass = "?" (missing value), education = "Bachelors", sex = "Male", race = "White", native country = "United-States".is used.

One example of a generated SuperTML image is given in Figure 3 .

Table 2 shows the results on Adult dataset.

We can see that on this dataset, the SuperTML method still has a higher accuracy than the fine-tuned XGBoost model, outperforming it by 0.32% points of accuracy.

The Higgs Boson Machine Learning Challenge involved a binary classification task to classify quantum events as signal or background.

It was hosted by Kaggle, and though the contest is over, the challenge data is available on opendata BID1 .

It has 25,000 training samples, and 55,000 testing samples.

Each example has 30 features, each of which is stored as a real number value.

In this challenge, AMS score BID0 (a) SuperTML EF background event example. is used as the performance metric.

FIG4 shows two examples of generated SuperTML images.

TAB3 shows the comparison of different algorithms.

The DNN method and XGBoost used in the first two rows are using the numerical values of the features as input to the models, which is different from the SuperTML method of using two-dimensional embeddings.

It shows that SuperTML EF method gives the best AMS score of 3.979.

In addition, the SuperTML EF gives better results than SuperTME VF results, which indicates SuperTML method can work well without the calculation of the importance scores.

The proposed SuperTML method borrows the idea of twodimensional embedding from Super Characters and transfers the knowledge learned from computer vision to the structured tabular data.

Experimental results shows that the proposed SuperTML method has achieved state-of-the-art results on both large and small tabular dataset.

<|TLDR|>

@highlight

Deep learning on structured tabular data using two-dimensional word embedding with fine-tuned ImageNet pre-trained CNN model.