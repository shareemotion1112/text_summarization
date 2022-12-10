Understanding procedural language requires anticipating the causal effects of actions, even when they are not explicitly stated.

In this work, we introduce Neural Process Networks to understand procedural text through (neural) simulation of action dynamics.

Our model complements existing memory architectures with dynamic entity tracking by explicitly modeling actions as state transformers.

The model updates the states of the entities by executing learned action operators.

Empirical results demonstrate that our proposed model can reason about the unstated causal effects of actions, allowing it to provide more accurate contextual information for understanding and generating procedural text, all while offering more interpretable internal representations than existing alternatives.

Understanding procedural text such as instructions or stories requires anticipating the implicit causal effects of actions on entities.

For example, given instructions such as "add blueberries to the muffin mix, then bake for one half hour," an intelligent agent must be able to anticipate a number of entailed facts (e.g., the blueberries are now in the oven; their "temperature" will increase).

While this common sense reasoning is trivial for humans, most natural language understanding algorithms do not have the capacity to reason about causal effects not mentioned directly in the surface strings BID12 BID7 BID14 .

The process is a narrative of entity state changes induced by actions.

In each sentence, these state changes are induced by simulated actions and must be remembered.

In this paper, we introduce Neural Process Networks, a procedural language understanding system that tracks common sense attributes through neural simulation of action dynamics.

Our network models interpretation of natural language instructions as a process of actions and their cumulative effects on entities.

More concretely, reading one sentence at a time, our model attentively selects what actions to execute on which entities, and remembers the state changes induced with a recurrent memory structure.

In FIG0 , for example, our model indexes the "tomato" embedding, selects the "wash" and "cut" functions and performs a computation that changes the "tomato" embedding so that it can reason about attributes such as its "SHAPE" and "CLEANLINESS".Our model contributes to a recent line of research that aims to model aspects of world state changes, such as language models and machine readers with explicit entity representations BID4 BID6 , as well as other more general purpose memory network variants BID30 BID26 BID5 BID23 .

This worldcentric modeling of procedural language (i.e., understanding by simulation) abstracts away from the surface strings, complementing text-centric modeling of language, which focuses on syntactic and semantic labeling of surface words (i.e., understanding by labeling).Unlike previous approaches, however, our model also learns explicit action representations as functional operators (See FIG0 .

While representations of action semantics could be acquired through an embodied agent that can see and interact with the world BID22 , we propose to learn these representations from text.

In particular, we require the model to be able to explain the causal effects of actions by predicting natural language attributes about entities such as "LOCATION" and "TEMPERATURE".

The model adjusts its representations of actions based on errors it makes in predicting the resultant state changes to attributes.

This textual simulation allows us to model aspects of action causality that are not readily available in existing simulation environments.

Indeed, most virtual environments offer limited aspects of the world -with a primary focus on spatial relations BID22 BID1 BID29 .

They leave out various other dimensions of the world states that are implied by diverse everyday actions such as "dissolve" (change of "COMPOSITION") and "wash" (change of "CLEANLINESS").Empirical results demonstrate that parametrizing explicit action embeddings provides an inductive bias that allows the neural process network to learn more informative context representations for understanding and generating natural language procedural text.

In addition, our model offers more interpretable internal representations and can reason about the unstated causal effects of actions explained through natural language descriptors.

Finally, we include a new dataset with fine-grained annotations on state changes, to be shared publicly, to encourage future research in this direction.

The neural process network is an interpreter that reads in natural language sentences, one at a time, and simulates the process of actions being applied to relevant entities through learned representations of actions and entities.

The main component of the neural process network is the simulation module ( §2.5), a recurrent unit whose internals simulate the effects of actions being applied to entities.

A set of V actions is known a priori and an embedding is initialized for each one, F = {f 1 , ...f V }.

Similarly, a set of I entities is known and an embedding is initialized for each one: E = {e 1 , ...

e I }.

Each e i can be considered to encode information about state attributes of that entity, which can be extracted by a set of state predictors ( §2.6).

As the model reads text, it "applies" action embeddings to the entity vectors, thereby changing the state information encoded about the entities.

For any document d, an initial list of entities I d is known and E d = {e i |i ∈ I d } ⊂ E entity state embeddings are initialized.

As the neural process network reads a sentence from the document, it selects a subset of both F ( §2.3) and E d ( §2.4) based on the actions performed and entities affected in the sentence.

The entity state embeddings are changed by the action and the new embeddings are used to predict end states for a set of state changes ( §2.6).

The prediction error for end states is backpropagated to the action embeddings, learning action representations that model the simulation of desired causal effects on entities.

This process is broken down into five modules below.

Unless explicitly defined, all W and b variables are parametrized linear projections and biases.

We use the notation {e i } t when referring to the values of the entity embeddings before processing sentence s t .

Given a sentence s t , a Gated Recurrent Unit encodes each word and outputs its last hidden vector as a sentence encoding h t .

Given h t from the sentence encoder, the action selector (bottom left in Fig. 2 beets", both f wash and f cut must be selected.

To account for multiple actions, we make a soft selection over F, yielding a weighted sum of the selected action embeddingsf t : DISPLAYFORM0 where MLP is a parametrized feed-forward network with a sigmoid activation and w p ∈ R V is the attention distribution over V possible actions ( §3.1).

We compose the action embedding by taking the weighted average of the selected actions.

Sentence Attention Given h t from the sentence encoder, the entity selector chooses relevant entities using a soft attention mechanism: DISPLAYFORM0 where W 2 is a bilinear mapping, e i0 is a unique key for each entity ( §2.5), and d i is the attention weight for entity embedding e i .

For example, in "wash and cut beets and carrots", the model should select e beet and e carrot .

Recurrent Attention While sentence attention would suffice if entities were always explicitly mentioned, natural language often elides arguments or uses referent pronouns.

As such, the module must be able to consider entities mentioned in previous sentences.

Usingh t , the model computes a soft choice over whether to choose affected entities from this step's attention d i or the previous step's attention distribution.

DISPLAYFORM1 where c ∈ R 3 is the choice distribution, a it−1 is the previous sentence's attention weight for each entity, a it is the final attention for each entity, and 0 is a vector of zeroes (providing the option to not change any entity).

Prior entity attentions can propagate forward for multiple steps.

Entity Memory A unique state embedding e i is initialized for every entity i in the document.

A unique key to index each embedding e i0 is set as the initial value of the embedding BID4 BID17 .

After the model reads s t , it modifies {e i } t to reflect changes influenced by actions.

At every time step, the entity memory receives the attention weights from the entity selector, normalizes them and computes a weighted average of the relevant entity state embeddings: DISPLAYFORM0 Applicator Given the action summary embeddingf t and the entity summary embeddingē t , the applicator (middle right in Fig. 2 ) applies the selected actions to the selected entities, and outputs the new proposal entity embedding k t .

DISPLAYFORM1 where W 4 is a third order tensor projection.

The vector k t is the new representation of the entityē t after the applicator simulates the action being applied to it.

Entity Updater The entity updater interpolates the new proposal entity embedding k t and the set of current entity embeddings {e i } t : DISPLAYFORM2 yielding an updated set of entity embeddings {e i } t+1 .

Each embedding is updated proportional to its entity's unnormalized attention a i , allowing the model to completely overwrite the state embedding for any entity.

For example, in the sentence "mix the flour and water," the embeddings for e f lour and e water must both be overwritten by k t because they no longer exist outside of this new composition.

Given the new proposal entity embedding k t , the state predictor (bottom right in Fig. 2 ) predicts changes to the resulting entity embedding k t along the following six dimensions: location, cookedness, temperature, composition, shape, and cleanliness.

Discrete multi-class classifiers, one for each dimension, take in k t and predict a unique end state for their corresponding state change type: DISPLAYFORM0 For location changes, which require contextual information to predict the end state, k t is concatenated with the original sentence representation h t to predict the final state.

In this work we focus on physical action verbs in cooking recipes.

We manually collect a set of 384 actions such as cut, bake, boil, arrange, and place, organizing their causal effects along the following predefined dimensions: LOCATION, COOKEDNESS, TEMPERATURE, SHAPE, CLEANLI-NESS and COMPOSITION.

The textual simulation operated by the model induces state changes along these dimensions by applying actions functions from the above set of 384.

For example, cut entails a change in SHAPE, while bake entails a change in TEMPERATURE, COOKEDNESS, and even LO-CATION.

We annotate the state changes each action induces, as well as the end state of the action, using Amazon Mechanical Turk.

The set of possible end states for a state change can range from 2 for binary state changes to more than 200 (See Appendix C for details).

For learning and evaluation, we use a subset of the Now You're Cooking dataset BID9 .

We chose 65816 recipes for training, 175 recipes for development, and 700 recipes for testing.

For the development and test sets, crowdsourced workers densely annotate actions, entities and state changes that occur in each sentence so that we can tune hyperparameters and evaluate on gold evaluation sets.

Annotation details are provided in Appendix C.3.

The neural process network is trained by jointly optimizing multiple losses for the action selector, entity selector, and state change predictors.

Importantly, our training scheme uses weak supervision because dense annotations are prohibitively expensive to acquire at a very large scale.

Thus, we heuristically extract verb mentions from each recipe step and assign a state change label based on the state changes induced by that action ( §3.1).

Entities are extracted similarly based on string matching between the instructions and the ingredient list.

We use the following losses for training: Action Selection Loss Using noisy supervision, the action selector is trained to minimize the cross-entropy loss for each possible action, allowing multiple actions to be chosen at each step if multiple actions are mentioned in a sentence.

The MLP in the action selector (Eq. 1) is pretrained.

Entity Selection Loss Similarly, to train the attentive entity selector, we minimize the binary cross-entropy loss of predicting whether each entity is affected in the sentence.

State Change Loss For each state change predictor, we minimize the negative loglikelihood of predicting the correct end state for each state change.

Coverage Loss An underlying assumption in many narratives is that all entities that are mentioned should be important to the narrative.

We add a loss term that penalizes narratives whose combined attention weights for each entity does not sum to more than 1.

DISPLAYFORM0 where a it is the attention weight for a particular entity at sentence t and I d is the number of entities in a document.

S t=1 a it is upper bounded by 1.

This is similar to the coverage penalty used in neural machine translation BID28 .

We evaluate our model on a set of intrinsic tasks centered around tracking entities and state changes in recipes to show that the model can simulate preliminary dynamics of the recipe task.

Additionally, we provide a qualitative analysis of the internal components of our model.

Finally, we evaluate the quality of the states encoded by our model on the extrinsic task of generating future steps in a recipe.

In the tracking task, we evaluate the model's ability to identify which entities are selected and what changes have been made to them in every step.

We break the tracking task into two separate evalua- Metrics In the entity selection test, we report the F1 score of choosing the correct entities in any step.

A selected entity is defined as one whose attention weight a i is greater than 50% ( §2.4).

Because entities may be harder to predict when they have been combined with other entities (e.g., the mixture may have a new name), we also report the recall for selecting combined (CR) and uncombined (UR) entities.

In the end state prediction test, we report how often the model correctly predicts the state change performed in a recipe step and the resultant end state.

This score is then scaled by the accuracy of predicting which entities were changed in that same step.

We report the average F1 and accuracy across the six state change types.

Baselines We compare our models against two baselines.

First, we built a GRU model that is trained to predict entities and state changes independently.

This can be viewed as a bare minimum network with no action representations or recurrent entity memory.

The second baseline is a Recurrent Entity Network BID4 with changes to fit our task.

First, the model can tie memory cells to a subset of the full list of entities so that it only considers entities that are present in a particular recipe.

Second, the entity distribution for writing to the memory cells is re-used when we query the memory cells.

The normalized weighted average of the entity cells is used as the input to the state predictors.

The unnormalized attention when writing to each cell is used to predict selected entities.

Both baselines are trained with entity selection and state change losses ( §3.3).Ablations We report results on six ablations.

First, we remove the recurrent attention (Eq. 3).

The model only predicts entities using the current encoder hidden state.

In the second ablation, the model is trained with no coverage penalty (Eq. 9).

The third ablation prunes the connection from the action selector w p to the entity selector (Eq. 2).

We also explore not pretraining the action selector.

Finally, we look at two ablations where we intialize the action embeddings with vectors from a skipgram model.

In the first, the model operates normally, and in the second, we do not allow gradients to backpropagate to the action embeddings, updating only the mapping tensor W 4 instead (Eq. 6).

The generation task tests whether our system can produce the next step in a recipe based on the previous steps that have been performed.

The model is provided all of the previous steps as context.

We report the combined BLEU score and ROUGE score of the generated sequence relative to the reference sequence.

Each candidate sequence has one reference sentence.

Both metrics are computed at the corpus-level.

Also reported are "VF1", the F1 score for the overlap of the actions performed in the reference sequence and the verbs mentioned in the generated sequence, and "SF1", the F1 score for the overlap of end states annotated in the reference sequence and predicted by the generated sequences.

End states for the generated sequences are extracted using the lexicon from Section 3.1 based on the actions performed in the sentence.

Setup To apply our model to the task of recipe step generation, we input the context sentences through the neural process network and record the entity state vectors once the entire context has st Let cool.

selected oats, sugar, flour, corn syrup, milk, vanilla extract, salt correct oats, sugar, flour, corn syrup, milk, vanilla extract, salt Good st−1 In a large saucepan over low heat, melt marshmallows.st Add sprinkles, cereal, and raisins, stir until well coated.

selected marshmallows, cereal, raisins correct marshmallows, cereal, raisins, sprinkles Bad st−3 Ladle the barbecue sauce around the crust and spread.

st−2 Add mozzarella, yellow cheddar, and monterey jack cheese.

st−1 Next, add onion mixture and sliced chicken breast .st Top pizza with jalapeno peppers.

selected jalapenos correct crust, sauce, mozzarella, cheddar, monterey jack, white onion, chicken, jalapenos Bad st−2 Combine 1 cup flour, salt, and 1 tbsp sugar.

st−1 Cut in butter until mixture is crumbly, then sprinkle with vinegar .st Gather dough into a ball and press into bottom of 9 inch springform pan.

selected butter, vinegar correct flour, salt, sugar, butter, vinegar TAB5 : Examples of the model selecting entities for sentence s t .

The previous sentences are provided as context in cases where they are relevant. been read ( §2.5).

These vectors can be viewed as a snapshot of the current state of the entities once the preceding context has been simulated inside the neural process network.

We encode these vectors using a bidirectional GRU and take the final time step hidden state e I .

A different GRU encodes the context words in the same way (yielding h T ) and the first hidden state input to the decoder is computed using the projection function: DISPLAYFORM0 where • is the Hadamard product between the two encoder outputs.

All models are trained by minimizing the negative loglikelihood of predicting the next word for the full sequence.

Implementation details can be found in Appendix A. Baselines For the generation task, we use three baselines: a seq2seq model with no attention , an attentive seq2seq model BID0 , and a similar variant as our NPN generator, except where the entity states have been computed by the Recurrent Entity Network (EntNet) baseline ( §4.1).

Implementation details for baselines can be found in Appendix B.

Entity Selection As shown in Table 8 , our full model outperforms all baselines at selecting entities, with an F1 score of 55.39%.

The ablation study shows that the recurrent attention, coverage loss, action connections and action selector pretraining improve performance.

Our success at predicting entities extends to both uncomposed entities, which are still in their raw forms (e.g., melt the butter → butter), and composed entities, in which all of the entities that make up a composition must be selected.

For example, in a Cooking lasagna recipe, if the final step involves baking the prepared lasagna, the model must select all the entities that make up the lasagna (e.g., lasagna sheets, beef, tomato sauce).

In as compositional entities (Ex.

1, 3), and elided arguments over long time windows (Ex.

2).

We also provide examples where the model fails to select the correct entities because it does not identify the mapping between a reference construct such as "pizza" (Ex.

4) or "dough" (Ex. 5) and the set of entities that composes it, showcasing the difficulty of selecting the full set for a composed entity.

State Change Tracking In Table 8 , we show that our full model outperforms competitive baselines such as Recurrent Entity Networks BID4 and jointly trained GRUs.

While the ablation without the coverage loss shows higher accuracy, we attribute this to the fact that it predicts a smaller number of total state changes.

Interestingly, initializing action embeddings with skipgram vectors and locking their values shows relatively high performance, indicating the potential gains in using powerful pretrained representations to represent actions.

Action Embeddings In our model, each action is assigned its own embedding, but many actions induce similar changes in the physical world (e.g.,"cut" and "slice").

After training, we compute the pairwise cosine similarity between each pair of action embeddings.

In TAB6 , we see that actions that perform similar functions are neighbors in embedding space, indicating the model has captured certain semantic properties of these actions.

Learning action representations through the state changes they induce has allowed the model to cluster actions by their transformation functions.

Entity Compositions When individual entities are combined into new constructs, our model averages their state embeddings (Eq. 5), applies an action embedding to them (Eq. 6), and writes them to memory (Eq. 7).

The state embeddings of entities that are combined should be overwritten by the same new embedding.

In Figure 3 , we present the percentage increase in cosine similarity for state embeddings of entities that are combined in a sentence (blue) as opposed to the percentage increase for those that are not (red bars).

While the soft attention mechanism for entity selection allows similarities to leak between entity embeddings, our system is generally able to model the compositionality patterns that result from entities being combined into new constructs.

Step Generation Our results in TAB7 indicate that sequences generated using the neural process network entity states as additional input yield higher scores than competitive baselines.

The entity states allow the model to predict next steps conditioned on a representation of the world being simulated by the neural process network.

Additionally, the higher VF1 and SF1 scores indicate that the model is indeed using the extra information to better predict the actions that should follow the context provided.

Example generations for each baselines from the dev set are provided in Table 6 , Context Preheat oven to 425 degrees.

Reference Melt butter in saucepan and mix in bourbon, thyme, pepper, and salt.

NPN Melt butter in skillet.

Seq2seq Lightly grease 4 x 8 baking pan with sunflower oil.

Attentive Seq2seq Combine all ingredients and mix well.

EntNet In a large bowl, combine flour, baking powder, baking soda, salt, and pepper.

Context Pour egg mixture over caramelized sugar in cake pan.

Place cake pan in large shallow baking dish.

Bake for 55 minutes or until knife inserted into flan comes out clean.

Reference Cover and chill at least 8 hours.

NPN Refrigerate until ready to use.

Seq2seq Serve at room temperature.

Attentive Seq2seq Store in an airtight container.

EntNet Store in an airtight container.

Context Cut squash into large pieces and steam.

Remove cooked squash from shells; Reference Measure 4 cups pulp and reserve remainder for another dish.

NPN Drain.

Seq2seq Mash pulp with a fork.

Attentive Seq2seq Set aside.

EntNet Set aside.

Table 6 : Examples of the model generating sentences compared to baselines.

The context and reference are provided first, followed by our model's generation and then the baseline generationsshowing that the NPN generator can use information about ingredient states to reason about the most likely next step.

The first and second examples are interesting as it shows that the NPN-aware model has learned to condition on entity state -knowing that raw butter will likely be melted or that a cooked flan must be refrigerated.

The third example is also interesting because the model learns that cooked vegetables such as squash will sometimes be drained, even if it is not relevant to this recipe because the squash is steamed.

The seq2seq and EntNet baselines, meanwhile, output reasonable sentences given the immediate context, but do not exhibit understanding of global patterns.

Recent studies in machine comprehension have used a neural memory component to store a running representation of processed text BID30 BID26 BID5 BID23 .

While these approaches map text to memory vectors using standard neural encoder approaches, our model, in contrast, directly interprets text in terms of the effects actions induce in entities, providing an inductive bias for learning how to represent stored memories.

More recent work in machine comprehension also sought to couple the memory representation with tracking entity states BID4 .

Our work seeks to provide a relatively more structured representation of domain-specific action knowledge to provide an inductive bias to the reasoning process.

Neural Programmers BID20 have also used functions to simulate reasoning, by building a model to select rows in a database and applying operation on those selected rows.

While their work explicitly defined the effect of a number of operations for those rows, we provide a framework for learning representations for a more expansive set of actions, allowing the model to learn representations for how actions change the state space.

Works on instructional language studied the task of building discrete graph representations of recipes using probabilistic models BID8 BID19 BID18 .

We propose a complementary new model by integrating action and entity relations into the neural network architecture and also address the additional challenge of tracking the state changes of the entities.

Additional work in tracking states with visual or multimodal context has focused on 1) building graph representations for how entities change in goal-oriented domains BID3 BID24 or 2) tracking visual state changes based on decisions taken by agents in environment simulators such as videos or games BID1 BID29 BID22 .

Our work, in contrast, models state changes in embedding space using only text-based signals to map real-world actions to algebraic transformations.

We introduced the Neural Process Network for modeling a process of actions and their causal effects on entities by learning action transformations that change entity state representations.

The model maintains a recurrent memory structure to track entity states and is trained to predict the state changes that entities undergo.

Empirical results demonstrate that our model can learn the causal effects of action semantics in the cooking domain and track the dynamic state changes of entities, showing advantages over competitive baselines.

A TRAINING DETAILS OF OUR FULL MODEL AND ABLATIONS

The hidden size of the instruction encoder is 100, the embedding sizes of action functions and entities are 30.

We use dropout with a rate of 0.3 before any non-recurrent fully connected layers BID25 .

We use the Adam optimizer BID11 ) with a learning rate of .001 and decay by a factor of 0.1 if we see no improvement on validation loss over three epochs.

We stop training early if the development loss does not decrease for five epochs.

The batch size is 64.

We use two instruction encoders, one for the entity selector, and one for the action selector.

Word embeddings and entity embeddings are initialized with skipgram embeddings BID15 ;b) using a word2vec model trained on the training set.

We use a vocabulary size of 7358 for words, and 2996 for entities.

Gradients with respect to the coverage loss (Eq. 9) are only backpropagated in steps where no entity is annotated as being selected.

To account for the false negatives in the training data due to the heuristic generation of the labels, gradients with respect to the entity selection loss are zeroed when no entity label is present.

The hidden size of the context encoder is 200.

The hidden size of the state vector encoder is 200.

State vectors have dimensionality 30 (the same as in the neural process network).

Dropout of 0.3 is used during training in the decoder.

The context and state representations are projected jointly using an element-wise product followed by a linear projection BID10 .

Both encoders and the decoder are single layer.

The learning rate is 0.0003 initially and is halved every 5 epochs.

The model is trained with the Adam optimizer.

Joint Gated Recurrent Unit The hidden state of the GRU is 100.

We use a dropout with a rate of 0.3 before any non-recurrent fully connected layers.

We use the Adam optimizer with a learning rate of .001 and decay by a factor of 0.1 if we see no improvement on validation loss over a single epoch.

We stop training early if the development loss does not decrease for five epochs.

The batch size is 64.

We use encoders, one for the entity selector, and one for the state change predictors.

Word embeddings are initialized with skipgram embeddings using a word2vec model trained on the training set.

We use a vocabulary size of 7358 for words.

Recurrent Entity Networks Memory cells are tied to the entities in the document.

For a recipe with 12 ingredients, 12 entity cells are initialized.

All hyperparameters are the same as the in the bAbI task from BID4 .

The learning rate start at 0.01 and is halved every 25 epochs.

Entity cells and word embeddings are 100 dimensional.

The encoder is a multiplicative mask initialized the same as in BID4 .

Intermediate supervision from the weak labels is provided to help predict entities.

A separate encoder is used for computing the attention over memory cells and the content to write to the memory.

Dropout of 0.3 is used in the encoders.

The batch size is 64.

We use a vocabulary size of 7358 for words, and 2996 for entities.

Seq2seq The encoder and decoder are both single-layer GRUs with hidden size 200.

We use dropout with probability 0.3 in the decoder.

We train with the Adam optimizer starting with a learning rate 0.0003 that is halved every 5 epochs.

The encoder is bidirectional.

The model is trained to minimize the negative loglikelihood of predicting the next word.

The encoder is the same as in the seq2seq baseline.

A multiplicative attention between the decoder hidden state and the context vectors is used to compute the attention over the context at every decoder time step.

The model is trained with the same learning rate, learning schedule and loss function as the seq2seq baseline.

The model is trained in the same way as the NPN generator model in Appendix A.2 except that the state representations used as input are produced from by EntNet baseline described in Section 4.1 and Appendix B.1.

We provide workers with a verb, its definition, an illustrative image of the action, and a set of sentences where the verb is mentioned.

Workers are provided a checklist of the six state change types and instructed to identify which of them the verb causes.

They are free to identify multiple changes.

Seven workers annotate each verb and we assign a state change based on majority vote.

Of the set of 384 verbs extracted, only 342 have a state change type identified with them.

Of those, 74 entail multiple state change types.

We give workers a verb, a state change type, and an example with the verb and ask them to provide an end state for the ingredient the verb is applied to in the example.

We then use the answers to manually aggregate a set of end states for each state change type.

These end states are used as labels when the model predicting state changes.

For example, a LOCATION change might lead to an end state of "pan," "pot", or "oven." End states for each state change type are provided in Annotators are instructed to note any entities that undergo one of the six state changes in each step, as well as to identify new combinations of ingredients that are created.

For example, the sentence "Cut the tomatoes and add to the onions" would involve a SHAPE change for the tomatoes and a combination created from the "tomatoes" and "onions".

In a separate task, three workers are asked to identify the actions performed in every sentence of the development and test set recipes.

If an action receives a majority vote that it is performed, it is included in the annotations.

Table 8 : Results for entity selection and state change selection on the development set when randomly dropping a percentage of the training labels

@highlight

We propose a new recurrent memory architecture that can track common sense state changes of entities by simulating the causal effects of actions.