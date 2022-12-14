Fine-tuning language models, such as BERT, on domain specific corpora has proven to be valuable in domains like scientific papers and biomedical text.

In this paper, we show that fine-tuning BERT on legal documents similarly provides valuable improvements on NLP tasks in the legal domain.

Demonstrating this outcome is significant for analyzing commercial agreements, because obtaining large legal corpora is challenging due to their confidential nature.

As such, we show that having access to large legal corpora is a competitive advantage for commercial applications, and academic research on analyzing contracts.

Businesses rely on contracts to capture critical obligations with other parties, such as: scope of work, amounts owed, and cancellation policies.

Various efforts have gone into automatically extracting and classifying these terms.

These efforts have usually been modeled as: classification, entity and relation extraction tasks.

In this paper we focus on classification, but in our application we have found that our findings apply equally and sometimes, more profoundly, on other tasks.

Recently, numerous studies have shown the value of fine-tuning language models such as ELMo [3] and BERT [4] to achieve state-of-the-art results [5] on domain specific tasks [6, 7] .

In this paper we investigate and quantify the impact of utilizing a large domain-specific corpus of legal agreements to improve the accuracy of classification models by fine-tuning BERT.

Specifically, we assess: (i) the performance of a simple model that only uses the pre-trained BERT language model, (ii) the impact of further fine tuning BERT, and (iii) how this impact changes as we train on larger corpora.

Ultimately, our investigations show marginal, but valuable, improvements that increase as we grow the size of the legal corpus used to fine-tine BERT -and allow us to confidently claim that not only is this approach valuable for increasing accuracy, but commercial enterprises seeking to create these models will have an edge if they can amass a corpus of legal documents.

Lexion is commercial venture that is building an "intelligent repository" for legal agreements that automatically classifies documents and then, based on the document type, fills a schema of metadata values using entity extraction, classification, and relationship extraction.

Our application then uses this metadata to perform a variety of tasks that are valuable to end users: automatically organizing documents; linking related documents; calculating date milestones; identifying outlier terms; and a host of features to run reports, receive alerts, share with permissions, and integrate with other systems.

(See Fig 1, screenshot) .

To deliver this application, we have developed an extensive pipeline and user-interface to ingest raw documents, perform OCR with multiple error detection and cleanup steps, rapidly annotate thousands of documents in hours, and train and deploy several models.

Delivering the most accurate models possible, while managing our annotation costs, is an important challenge for us.

Furthermore, we wish to leverage the massive legal corpus that we have acquired, and turn it into a competitive advantage using unsupervised techniques.

For these reasons, applying pre-trained language models, and fine-tuning them further on our legal corpus, is an attractive approach to maximize accuracy and provide a more beneficial solution than our competitors.

To fine-tune BERT, we used a proprietary corpus that consists of hundreds of thousands of legal agreements.

We extracted text from the agreements, tokenized it into sentences, and removed sentences without alphanumeric text.

We selected the BERT-Base uncased pre-trained model for fine-tuning.

To avoid including repetitive content found at the beginning of each agreement we selected the 31st to 50th sentence of each agreement.

We ran unsupervised fine-tuning of BERT using sequence lengths of 128, 256 and 512.

The loss function over epochs is shown in Figure 2 .

We used a proprietary dataset consisting of a few thousand legal agreements.

These were hand annotated by our model-development team using our internal rapid-annotation tools.

We annotate a few dozen attributes per document, but for this paper we hand picked a single common and high value class: the "Term" of an agreement.

In practice, the term of agreement can be one of about half a dozen possible classes, but we chose to focus on the two most common classes for this research: the "fixed" term, i.e. the term of an agreement that expires after a fixed amount of time; and the "auto-renewing" term, i.e. the term of an agreement that automatically renews.

While this attribute might seem simple at a glance, there are many subtleties that make it challenging to extract this with a high enough accuracy for practical applications.

Our end-to-end system does a great deal of preand post-processing to achieve an impressive level of accuracy that makes our application viable for end users, the details of which are beyond the scope of this paper.

We split our classification dataset into train (80%) and validation (20%) sets.

For all architecture variations, we train for a variable number of epochs as long as the validation error is decreasing.

We stop training when validation error starts increasing again and then report the final result on a held-out test set.

In doing so we try to avoid over-fitting on the training set.

For a baseline, we trained a simple neural network with the architecture shown in figure 5 .

The input to the network was a Bag-of-Word representation of the text.

The BERT classifier we used consisted of the BERT layers, followed by the last three layers of our baseline network shown in figure 4.

When training our BERT-based models, we also fine tuned the BERT layers on the end task.

In order to assess the delta from using the Language Model (LM) that was fine-tuned on our legal corpus, we performed another experiment where we froze the BERT layers and only trained the last portion of the network.

While the final accuracy of this model was sub par, even compared to our baseline model, the gains from using a fine-tuned instead of a pre-trained LM is much more pronounced, providing further evidence for the value of domain-specific fine tuning.

These results are shown in Table 1 .

We use 4 metrics to compare performance across various experiments: Matthews Correlation Coefficient, as well as Precision, Recall and F1 score weighted by class size.

In table 2 we show the various results we got from different configurations.

It's clear that using pre-trained BERT, we're able to achieve a significant performance lift compared to the base line.

It is also clear that fine-tuning BERT on a domain-specific corpus noticeably improves this lift, even when the corpus size is small and we train for a short time.

In Figure 3 we also show the different rates of change in train loss across epochs between pre-trained BERT vs fine-tuned Bert.

As shown, the model trained on the fine-tuned version learns faster as evident in the faster drop in train loss on the training set (note the logarithmic y-axis).

It is worth mentioning that our BERT-based architecture is very simplistic for the sake of a fair comparison.

In practice, having a deeper neural network on top of BERT that is specialized in the end task yields much more impressive results, and that's the architecture we use in our system.

We show the result of using a slightly more advanced architecture with fine-tuned BERT in table 3 to demonstrate what's possible without any sophisticated feature engineering or hyper-parameter tuning.

@highlight

Fine-tuning BERT on legal corpora provides marginal, but valuable, improvements on NLP tasks in the legal domain.