Language modeling tasks, in which words are predicted on the basis of a local context, have been very effective for learning word embeddings and context dependent representations of phrases.

Motivated by the observation that efforts to code world knowledge into machine readable knowledge bases tend to be entity-centric, we investigate the use of a fill-in-the-blank task to learn context independent representations of entities from the contexts in which those entities were mentioned.

We show that large scale training of neural models allows us to learn extremely high fidelity entity typing information, which we demonstrate with few-shot reconstruction of Wikipedia categories.

Our learning approach is powerful enough to encode specialized topics such as Giro d’Italia cyclists.

A long term goal of artificial intelligence has been the development and population of an entitycentric representation of human knowledge.

Efforts have been made to create the knowledge representation with knowledge engineers BID10 or crowdsourcers BID1 .

However, these methods have relied heavily on human definitions of their ontologies, which are both limited in scope and brittle in nature.

Conversely, due to recent advances in deep learning, we can now learn robust general purpose representations of words BID13 and contextualized phrases BID16 BID6 directly from large textual corpora.

Consider the following context in which an entity mention is replaced with the [MASK] symbol:. . . [MASK] , a Russian factory worker, was the first woman in space . .

.As readers, we understand that first woman in space is a unique identifier, and we are able to fill in the blank unambiguously.

The central hypothesis of this paper is that, by matching entities to the contexts in which they are mentioned, we should be able to build a representation for Valentina Tereshkova that encodes the fact that she was the first woman in space.

To do this, we start with BERT BID6 , a powerful pretrained text encoder, to encode contexts-Wikipedia text in which a hyperlinked span has been blanked out-and we train an entity encoder to match the BERT representation of the entity's contexts.

We experiment with a lookup table that maps each entity to a fixed length vector, which we call RELIC (Representations of Entities Learned In Context).

We hypothesize that the dedicated entity representations in RELIC should be able to capture knowledge that is not present in BERT.

To test this, we compare RELIC to two BERT-based entity encoders: one that encodes the entity's canonical name, and one that encodes the first paragraph of the entity's Wikipedia page.

Ultimately, we would like our representations to encode all of the salient information about each entity.

However, for this initial work, we study our representations' ability to capture Wikipedia categorical information encoded by human experts.

We show that given just a few exemplar entities of a Wikipedia category such as Giro d'Italia cyclists, we can use RELIC to recover the remaining entities of that category with good precision.

Several works have tackled the problem of learning distributed representations of entities in a knowledge base (KB).

Typical approaches rely on the (subject, relation, object) ontology of KBs like Free-base BID1 .

These methods embed the entities and relations in vector space, then maximize the score of observed triples against negative triples to do KB completion BID3 BID21 BID28 BID24 .There is relatively less work in learning entity representations directly from text.

The contextual approach of word2vec BID13 has been applied to entities, but there has been little analysis of how effective such a method would be for answering the entity typing queries we study in this work.

Most methods for entity representations that do use raw text will combine it with structure from an existing KB BID18 BID23 BID9 in an effort to leverage as much information as possible.

While there may be gains to be had from using structure, our goal in this work is to isolate and understand the limits of inducing entity representations from raw text alone.

We also note the similarity of our RELIC task to entity linking BID17 BID4 BID22 BID7 and entity typing BID27 BID19 , where entity mentions are processed in context.

Unlike previous work in context-dependent entity typing BID11 BID5 BID15 , we consider types of RELIC from a global perspective.

We are interested in identifying contextindependent types of entities so that they can be used to identify structure in the entity latent space.

We study the ability of current models to learn entity encoders directly from the contexts in which those entities are seen.

Formally, we define an entity encoder to be a function h e = f (e) that maps each entity e to a vector h e ∈ R d .

We outline the training procedure used to learn the encoders.

RELIC training input Let E = {e 0 . . .

e N } be a predefined set of entities, and let DISPLAYFORM0 is a sequence of words x i ∈ V. Each context contains exactly one instance of the [MASK] symbol.

Our training data is a corpus of (context, entity) pairs DISPLAYFORM1 .

Each y i ∈ E is an entity, and the [MASK] symbol in x i substitutes for a single mention y i .

For clean training data, we extract our corpus from English Wikipedia, taking advantage of its rich hyperlink structure (Section 3.2).

We introduce a context encoder h x = g(x) that maps the context x into the same space R d as our entity encodings.

Then we define a compatibility score between the entity e and the context x as the scaled cosine similarity s(x, e) = a · DISPLAYFORM0 ||g(x)||||f (e)|| where the scaling factor a is a learned parameter, following BID26 .

Now, given a context x, the conditional probability that e was the entity seen with x is defined as p(e|x) = exp(s(x,e)) e ∈E exp(s(x,e )) and we train RELIC by maximizing the average log probability 1 |D| (x,y)∈D log p(y|x).

In practice, we use a noise contrastive loss BID8 BID14 ), where we sample K negative entities e 1 , e 2 , . . .

, e K from a noise distribution p noise (e).

Denoting e 0 := e, our per-example loss is l(s, x, e) = − log DISPLAYFORM1 .

We train our model with minibatch gradient descent and use all other entries in the batch as negatives.

This is roughly equivalent to p noise (e) being proportional to entity frequency.

BERT context encoder For g, we use the pretrained BERT model BID6 , a powerful Transformer-based BID25 text encoder, to encode contexts into a fixed-size representation 1 .

We project the BERT hidden state into R d using a weight matrix W ∈ R d×768 to obtain our context encoding.

Table 1 : Results for the Wikipedia category population task.

Mean Average Precision for ranking entities given a set of exemplars of a given Wikipedia class.

K represents the number of candidates to be ranked, andp is the average number of positive labels among the candidates.

Results are averaged over 100 categories sampled at random from those containing at least 1000 entities.

Wikipedia name, and the first paragraph of its Wikipedia page.

We consider three encoders that operate on different representations of the entities: (1) embedding lookup, (2) BERT name encoder, and (3) BERT description encoder.

In the standard RELIC setup, we map each entity, identified by its unique ID, directly onto its own dedicated vector in R d via a |E| × d dimensional embedding matrix.

We also consider two alternate BERT-based options for distributed encoding of entities, which are fine-tuned on the RELIC data.

The name encoder applies a BERT Transformer to the canonical name of the entity to obtain a fixedsize representation.

The description encoder applies a BERT Transformer to an entity's description to obtain a fixed size representation 3 .

Note that both name and description encoders can do zero-shot encoding of new entities, assuming that a name or description is provided.

To train RELIC, we obtain data from the 2018-10-22 dump of English Wikipedia.

We take E to be the set of all entities in Wikipedia (of which there are over 5 million).

For each hyperlink, we take the context as the surrounding sentence, replace all tokens in the anchor text with a single [MASK] symbol, and set the entity linked to as ground truth.

We limit each context to 64 tokens.

We set the entity embedding size to d = 300.

For the name and description encoders, we take the initial hidden state of the Transformer as the fixed-size representation.

We limit to 16 name tokens and 128 description tokens.

We train the model using TensorFlow BID0 ) with a batch size of 1024 for 5 epochs.

We hold out about 1/30,000 of the data for use as validation, on which the final model achieves roughly 85% accuracy in-batch negative prediction for all models.

We introduce a fine-grained entity typing task based on Wikipedia categories, where the task is to populate a category from a small number of exemplars.

We evaluate if RELIC benefits from dedicated embeddings over the BERT encoders that share parameters between entities.

We filter the Wikipedia categories in Yago 3.1 BID12 to get the 1,129 categories that cover at least 1,000 entities and consider an exemplar based "few-shot" scenario, based on the prototypical approach of BID20 .

For each category, we provide a small number of exemplars (3 or 10), one correct candidate entity drawn from the category, and K −1 other candidate entities.

The candidate entities are ranked according to the inner product between their RELIC embeddings and the centroid of the exemplar embeddings, and we report the mean average precision (MAP) for entities belonging to the query class.

Wikipedia categories are often incompletely labeled, and when K covers all entities, this confounds the MAP calculation.

Therefore, we also present results for K = 10 and K = 1000 for a cleaner experimental setup.

TAB2 show results.

RELIC outperforms both the BERT name and description encoders when we restrict the candidate set to the entities seen at least 10 times in RELIC's training data, and the gap in performance increases as we increase the entity frequency threshold.

However, both the name and description encoders outperform RELIC on very infrequent entities, since they can generalize from other entities with similar naming conventions or descriptions, while RELIC's embedding matrix treats every entity completely separately.

FIG0 shows examples of predictions for Wikipedia categories given 3 exemplars for 5 randomly sampled categories.

Most categories show high precision in the top 10 predictions.

The category Butterflies of Africa fails-this is likely due to the fact that the 3 exemplars appeared only a total of 4 times in our pretraining data.

The Giro d'Italia cyclists category is very well predicted-the single incorrect prediction Thibaut Pinot did cycle in the Giro d'Italia.

However, for Video games featuring female protagonists, most of RELIC's success is due to just retrieving variations of the Final Fantasy series.

We demonstrated that the RELIC fill-in-the-blank task allows us to learn highly interesting representations of entities with their own latent ontology, which we empirically verify through a few-shot Wikipedia category reconstruction task.

We encourage researchers to explore the properties of our entity representations and BERT context encoder, which we will release publicly.

@highlight

We learn entity representations that can reconstruct Wikipedia categories with just a few exemplars.