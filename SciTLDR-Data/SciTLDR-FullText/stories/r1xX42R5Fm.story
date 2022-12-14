The conventional approach to solving the recommendation problem greedily ranks individual document candidates by prediction scores.

However, this method fails to optimize the slate as a whole, and hence, often struggles to capture biases caused by the page layout and document interdepedencies.

The slate recommendation problem aims to directly find the optimally ordered subset of documents (i.e. slates) that best serve users’ interests.

Solving this problem is hard due to the combinatorial explosion of document candidates and their display positions on the page.

Therefore we propose a paradigm shift from the traditional viewpoint of solving a ranking problem to a direct slate generation framework.

In this paper, we introduce List Conditional Variational Auto-Encoders (ListCVAE), which learn the joint distribution of documents on the slate conditioned on user responses, and directly generate full slates.

Experiments on simulated and real-world data show that List-CVAE outperforms greedy ranking methods consistently on various scales of documents corpora.

Recommender systems modeling is an important machine learning area in the IT industry, powering online advertisement, social networks and various content recommendation services BID0 Lu et al., 2015) .

In the context of document recommendation, its aim is to generate and display an ordered list of "documents" to users (called a "slate" in BID2 ; BID3 ), based on both user preferences and documents content.

For large scale recommender systems, a common scalable approach at inference time is to first select a small subset of candidate documents S out of the entire document pool D. This step is called "candidate generation".

Then a function approximator such as a neural network (e.g., a Multi-Layer Perceptron (MLP)) called the "ranking model" is used to predict probabilities of user engagements for each document in the small subset S and greedily generates a slate by sorting the top documents from S based on estimated prediction scores BID4 .

This two-step process is widely popular to solve large scale recommendation problems due to its scalability and fast inference at serving time.

The candidate generation step can decrease the number of candidates from millions to hundreds or less, effectively dealing with scalability when faced with a large corpus of documents D. Since |S| is much smaller than |D|, the ranking model can be reasonably complicated without increasing latency.

However, there are two main problems with this approach.

First the candidate generation and the ranking models are not trained jointly, which can lead to having candidates in S that are not the highest scoring documents of the ranking model.

Second and most importantly, the greedy ranking method suffers from numerous biases that come with the visual presentation of the slate and context in which documents are presented, both at training and serving time.

For example, there exists positional biases caused by users paying more attention to prominent slate positions BID5 , and contextual biases, due to interactions between documents presented together in the same slate, such as competition and complementarity, relative attractiveness, etc. .In this paper, we propose a paradigm shift from the traditional viewpoint of solving a ranking problem to a direct slate generation framework.

We consider a slate "optimal" when it maximizes some type of user engagement feedback, a typical desired scenario in recommender systems.

For example, given a database of song tracks, the optimal slate can be an ordered list (in time or space) of k songs such that the user ideally likes every song in that list.

Another example considers news articles, the optimal slate has k ordered articles such that every article is read by the user.

In general, optimality can be defined as a desired user response vector on the slate and the proposed model should be agnostic to these problem-specific definitions.

Solving the slate recommendation problem by direct slate generation differs from ranking in that first, the entire slate is used as a training example instead of single documents, preserving numerous biases encoded into the slate that might influence user responses.

Secondly, it does not assume that more relevant documents should necessarily be put in earlier positions in the slate at serving time.

Our model directly generates slates, taking into account all the relevant biases learned through training.

In this paper, we apply Conditional Variational Auto-Encoders (CVAEs) BID7 BID8 to model the distributions of all documents in the same slate conditioned on the user response.

All documents in a slate along with their positional, contextual biases are jointly encoded into the latent space, which is then sampled and combined with desired conditioning for direct slate generation, i.e. sampling from the learned conditional joint distribution.

Therefore, the model first learns which slates give which type of responses and then directly generates similar slates given a desired response vector as the conditioning at inference time.

We call our proposed model List-CVAE.

The key contributions of our work are:1.

To the best of our knowledge, this is the first model that provides a conditional generative modeling framework for slate recommendation by direct generation.

It does not necessarily require a candidate generator at inference time and is flexible enough to work with any visual presentation of the slate as long as the ordering of display positions is fixed throughout training and inference times.2.

To deal with the problem at scale, we introduce an architecture that uses pretrained document embeddings combined with a negatively downsampled k-head softmax layer within the List-CVAE model, where k is the slate size.

The structure of this paper is the following.

First we introduce related work using various CVAE-type models as well as other approaches to solve the slate generation problem.

Next we introduce our List-CVAE modeling approach.

The last part of the paper is devoted to experiments on both simulated and the real-world datasets.2 RELATED WORK Traditional matrix factorization techniques have been applied to recommender systems with success in modeling competitions such as the Netflix Prize BID10 .

Later research emerged on using autoencoders to improve on the results of matrix factorization BID11 (CDAE, CDL).

More recently several works use Boltzmann Machines BID13 and variants of VAE models in the Collaborative Filtering (CF) paradigm to model recommender systems BID14 BID15 BID16 ) (Collaborative VAE, JMVAE, CVAE-CF, JVAE-CF).

See FIG0 for model structure comparisons.

In this paper, unless specified otherwise, the user features and any context are routinely considered part of the conditioning variables (in Appendix A Personalization Test, we test List-CVAE generating personalized slates for different users).

These models have primarily focused on modeling individual document or pairs of documents in the slate and applying greedy ordering at inference time.

Our model is also using a VAE type structure and in particular, is closely related to the Joint Multimodel Variational Auto-Encoder (JMVAE) architecture FIG0 ).

However, we use whole slates as input instead of single documents, and directly generate slates instead of using greedy ranking by prediction scores.

Other relevant work from the Information Retrieval (IR) literature are listwise ranking methods BID17 BID18 BID19 BID20 BID21 .

These methods use listwise loss functions that take the contexts and positions of training examples into account.

However, they eventually assign a prediction score for each document and greedily rank them at inference time.

In the Reinforcement Learning (RL) literature, BID3 view the whole slates as actions and use a deterministic policy gradient update to learn a policy that generates these actions, given concatenated document features as input.

Finally, the framework proposed by BID22 predicts user engagement for document and position pairs.

It optimizes whole page layouts at inference time but may suffer from poor scalability due to the combinatorial explosion of all possible document position pairs.

We formally define the slate recommendation problem as follows.

Let D denote a corpus of documents and let k be the slate size.

Then let r = (r 1 , . . .

, r k ) be the user response vector, where r i ∈ R is the user's response on document d i .

For example, if the problem is to maximize the number of clicks on a slate, then let r i ∈ {0, 1} denote whether the document d i is clicked, and thus an optimal slate DISPLAYFORM0

Variational Auto-Encoders (VAEs) are latent-variable models that define a joint density P θ (x, z) between observed variables x and latent variables z parametrized by a vector θ.

Training such models requires marginalizing the latent variables in order to maximize the data likelihood P θ (x) = P θ (x, z)dz.

Since we cannot solve this marginalization explicitly, we resort to a variational approximation.

For this, a variational posterior density Q φ (z|x) parametrized by a vector φ is introduced and we optimize the variational Evidence Lower-Bound (ELBO) on the data loglikelihood: DISPLAYFORM0 DISPLAYFORM1 where KL is the Kullback-Leibler divergence and where P θ (z) is a prior distribution over latent variables.

In a Conditional VAE (CVAE) we extend the distributions P θ (x, z) and Q φ (z|x) to also depend on an external condition c. The corresponding distributions are indicated by P θ (x, z|c) and Q φ (z|x, c).

Taking the conditioning c into account, we can write the variational loss to minimize as DISPLAYFORM2

We assume that the slates s = (d 1 , d 2 , . . .

d k ) and the user response vectors r are jointly drawn from a distribution P D k ×R k .

In this paper, we use a CVAE to model the joint distribution of all DISPLAYFORM0 DISPLAYFORM1 is the conditioning vector, where r = (r 1 , r 2 , . . .

, r k ) is the user responses on the slate s. The concatenation of s and c makes the input vector to the encoder.

The latent variable z ∈ R m has a learned prior distribution N (µ 0 , σ 0 ).

The raw output from the decoder are k vectors x 1 , x 2 . . .

, x k , each of which is mapped to a real document through taking the dot product with the matrix Φ containing all document embeddings.

Thus produced k vectors of logits are then passed to the negatively downsampled k-head softmax operation.

At inference time, c is the ideal condition whose concatenation with sampled z is the input to the decoder.

documents in the slate conditioned on the user responses r, i.e. P(d 1 , d 2 , . . .

d k |r).

At inference time, the List-CVAE model attempts to generate an optimal slate by conditioning on the ideal user response r .As we explained in Section 1, "optimality" of a slate depends on the task.

With that in mind, we define the mapping Φ : R k → C. It transforms a user response vector r into a vector in the conditioning space C that encodes the user engagement metric we wish to optimize for.

For instance, if we want to maximize clicks on the slate, we can use the binary click response vectors and set the conditioning to c = Φ(r) := k i=0 r i .

Then at inference time, the corresponding ideal user response r would be (1, 1, . . . , 1), and correspondingly the ideal conditioning would be c = Φ(r ) = k i=0 1 = k. As usual with CVAEs, the decoder models a distribution P θ (s|z, c) that, conditioned on z, is easy to represent.

In our case, P θ (s|z, c) models an independent probability for each document on the slate, represented by a softmax distribution.

Note that the documents are only independent to each other conditional on z. In fact, the marginalized posterior P θ (s|c) = z P θ (s|z, c)P θ (z|c)dz can be arbitrarily complex.

When the encoder encodes the input slate s into the latent space, it learns the joint distribution of the k documents in a fixed order, and thus also encodes any contextual, positional biases between documents and their respective positions into the latent variable z. The decoder learns these biases through reconstruction of the input slates from latent variables z with conditions.

At inference time, the decoder reproduces the input slate distribution from the latent variable z with the ideal conditioning, taking into account all the biases learned during training time.

Figure 3: Predictive prior distribution of the latent variable z in R 2 , conditioned on ideal user response c = (1, 1, . . . , 1).

The color map corresponds to the expected total responses of the corresponding slates.

Plots are generated from the simulation experiment with 1000 documents and slate size 10.To shed light onto what is encoded in the latent space, we simplify the prior distribution of z to be a fixed Gaussian distribution N (0, I) in R 2 .

We train List-CVAE and plot the predictive prior z. As training evolves, generated output slates with low total responses are pushed towards the edge of the latent space while high response slates cluster towards a growing center area (Figure 3 ).

Therefore after training, if we sample z from its prior distribution N (0, I) and generate the corresponding output slates, they are more likely to have high total responses.

Since the number of documents in D can be large, we first embed the documents into a low dimensional space.

Let Ψ : D → S q−1 be that normalized embedding where S q−1 denotes the unit sphere in R q .

Ψ can easily be pretrained using a standard supervised model that predicts user responses from documents or through a standard auto-encoder technique.

For the i-th document in the slate, our model produces a vector x i in R q that is then matched to each document from D via a dot-product.

This operation produces k vectors of logits for k softmaxes, i.e. the k-head softmax.

At training time, for large document corpora D, we uniformly randomly downsample negative documents and compute only a small subset of the logits for every training example, therefore efficiently scaling the nearest neighbor search to millions of documents with minimal model quality loss.

We train this model as a CVAE by minimizing the sum of the reconstruction loss and the KLdivergence term: DISPLAYFORM2 where β is a function of the training step BID23 .During inference, output slates are generated by first sampling z from the conditionally learned prior distribution N (µ , σ ), concatenating with the ideal condition c = Φ(r ), and passed into the decoder, generating (x 1 , . . .

, x k ) from the learned P θ (s|z, c ), and finally taking arg max over the dot-products with the full embedding matrix independently for each position i = 1, . . .

, k.

for i = 1, . . .

, k, where B represents the Bernoulli distribution.

During training, all models see uniformly randomly generated slates s ∼ U({1, n} k ) and their generated responses r. During inference time, we generate slates s by conditioning on c = (1, . . . , 1).

We do not require document de-duplication since repetition may be desired in certain applications (e.g. generating temporal slates in an online advertisement session).

Moreover List-CVAE should learn to produce the optimal slates whether those slates contain duplication or not from learning the training data distribution.

Evaluation: For evaluation, we cannot use offline ranking evaluation metrics such as Normalized Discounted Cumulative Gain (NDCG) BID24 , Mean Average Precision (MAP) (Baeza-Yates and Ribeiro-Neto, 1999) or Inverse Propensity Score (IPS) BID26 , etc.

These metrics either require prediction scores for individual documents or assume that more relevant documents should appear in earlier ranking positions, unfairly favoring greedy ranking methods.

Moreover, we find it limiting to use various diversity metrics since it is not always the case that a higher diversity-inclusive score gives better slates measured by user's total responses.

Even though these metrics may be more transparent, they do not measure our end goal, which is to maximize the expected number of total responses on the generated slates.

Instead, we evaluate the expected number of clicks over the distribution of generated slates and over the distribution of clicks on each document: DISPLAYFORM0 In practice, we distill the simulated environment of Eq. 5 using the cross-entropy loss onto a neural network model that officiates as our new simulation environment.

The model consists of an embedding layer, which encodes documents into 8-dimensional embeddings.

It then concatenates the embeddings of all the documents that form a slate and follows this concatenation with two hidden layers and a final softmax layer that predicts the slate response amongst the 2 k possible responses.

Thus we call it the "response model".

We use the response model to predict user responses on 100,000 sampled output slates for evaluation purposes.

This allows us to accurately evaluate our output slates by List-CVAE and all other baseline models.

Our experiments compare List-CVAE with several greedy ranking baselines that are often deployed in industry productions, namely Greedy MLP, Pairwise MLP, Position MLP and Greedy Long Short-Term Memory (LSTM) models.

In addition to the greedy baselines, we also compare against auto-regressive (AR) versions of Position MLP and LSTM, as well as randomly-selected slates from the training set as a sanity check.

List-CVAE generates slates s = arg max s∈{1,...,n} k P θ (s|z, c ).

The encoder and decoder of List-CVAE, as well as all the MLP-type models consist of two fully-connected neural network layers of the same size.

Greedy MLP trains on (d i , r i ) pairs and outputs the greedy slate consisting of the top k highestP (r|d) scoring documents.

Pairwise MLP is an MLP model with a pairwise ranking loss DISPLAYFORM0 where L x is the cross entropy loss and (x 1 , x 2 ) are pairs of documents randomly selected with different user responses from the same slate.

We sweep on hyperparameters α and η in addition to the shared MLP model structure sweep.

Position MLP uses position in the slate as a feature during training time and simply sets it to 0 for fast performance at inference time.

AR Position MLP is identical to Position MLP with the exception that the position feature is set to each corresponding slate position at inference time (as such it takes into account position biases).

Greedy LSTM is an LSTM model with fully-connect layers before and after the recurrent middle layers.

We tune the hyperparameters corresponding to the number of layers and their respective widths.

We use sequences of documents that form slates as the input at training time, and use single examples as inputs with sequence length 1 at inference time, which is similar to scoring documents as if they are in the first position of a slate of size 1.

Then we greedily rank the documents based on their prediction scores.

AR LSTM is identical to Greedy LSTM during training.

During inference, however, it selects documents sequentially by first selecting the best document for position 1, then unrolling the LSTM for 2 steps to select the best document for position 2, and so on.

This way it takes into account the context of all previous documents and their positions.

Random selects slates uniformly randomly from the training set.

Small-scale experiment (n = 100, 1000, k = 10): 0 1000 2000 3000 4000 5000Step FORMULA1 Step FORMULA2 We use the trained document embeddings from the response model for List-CVAE and all the baseline models.

For List-CVAE, we also use trained priors P θ (z |c) = N (µ , σ ) where µ , σ = f prior (c ) and f prior is modeled by a small MLP (16, 32).

Additionally, since we found little difference between different hyperparameters, we fixed the width of all hidden layers to 128, the learning rate to 10 DISPLAYFORM1 and the number of latent dimensions to 16.

For all other baseline models, we sweep on the learning rates, model structures and any model specific hyperparameters such as α, η for Position MLP and the forget bias for the LSTM model.

FIG5 show the performance comparison when the number of documents n = 100, 1000 and slate size to k = 10.

While List-CVAE is not quite capable of reaching a perfect performance of 10 clicks (which is probably even above the optimal upper bound), it easily outperforms all other ranking baselines after only a few training steps.

Appendix A includes an additional personalization test.

Due to a lack of publicly available large scale slate datasets, we use the data provided by the RecSys 2015 YOOCHOOSE Challenge BID27 .

This dataset consists of 9.2M user purchase sessions around 53K products.

Each user session contains an ordered list of products on which the user clicked, and whether they decided to buy them.

The List-CVAE model can be used on slates with temporal ordering.

Thus we form slates of size 5 by taking consecutive clicked products.

We then build user responses from whether the user bought them.

We remove a portion of slates with no positive responses such that after removal they only account for 50% of the total number of slates.

After filtering out products that are rarely bought, we get 375K slates of size 5 and a corpus of 10,000 candidate documents.

FIG9 shows the user response distribution of the training data.

Notice that in the response vector, 0 denotes a click without purchase and 1 denotes a purchase.

For example, (1,0,0,0,1) means the user clicked on all five products but bought only the first and the last products.

Medium-scale experiment (n = 10, 000, k = 5):Similarly to the previous section, we train a two-layer response model that officiates as a new semisynthetic simulation environment.

We use the same hyperparameters used previously.

FIG9 shows that List-CVAE outperforms all other baseline models within 500 training steps, which corresponds to having seen less than 10 −11 % of all possible slates.

Large-scale experiment (n = 1 million, 2 millions, k = 5):We synthesize 1,990k documents by adding independent Gaussian noise N (0, 10 −2 · I) to the original 10k documents and label the synthetic documents by predicted responses from the response model.

Thus the new pool of candidate documents consists of 10k original documents and 1,990k synthetic ones, totaling 2 million documents.

To match each of the k decoder outputs (x 1 , x 2 , . . .

, x k ) with Step 0 Step 0 Step 0 real documents, we uniformly randomly downsample the negative document examples keeping in total only 1000 logits (the dot product outputs in the decoder) during training.

At inference time, we pick the argmax for each of k dot products with the full embedding matrix without sampling.

This technique speeds up the total training and inference time for 2 million documents to merely 4 minutes on 1 GPU for both the response model (with 40k training steps) and List-CVAE (with 5k training steps).

We ran 2 experiments with 1 million and 2 millions document respectively.

From the results shown in FIG9 and 5d, List-CVAE steadily outperforms all other baselines again.

The greatly increased number of training examples helped List-CVAE really learn all the interactions between documents and their respective positional biases.

The resulting slates were able to receive close to 5 purchases on average due to the limited complexity provided by the response model.

In practice, we may not have any close-to-optimal slates in the training data.

Hence it is crucial that List-CVAE is able to generalize to unseen optimal conditions.

To test its generalization capacity, we use the medium-scale experiment setup on RecSys 2015 dataset and eliminate from the training data all slates where the total user response exceeds some ratio h of the slate size k, i.e. k i=1 r i > hk for h = 80%, 60%, 40%, 20%.

FIG10 shows test results on increasingly difficult training sets from which to infer on the optimal slates.

Without seeing any optimal slates FIG10 ) or slates with 4 or 5 total purchases FIG10 ), List-CVAE can still produce close to optimal slates.

Even training on slates with only 0, 1 or 2 total purchases (h = 40%), List-CVAE still surpasses the performance of all greedy baselines within 1000 steps FIG10 .

Thus demonstrating the strong generalization power of the model.

List-CVAE cannot learn much about the 0 1000 2000 3000 4000 5000Step 0 interactions between documents given only 0 or 1 total purchase per slate FIG10 ), whereas the MLP-type models learn purchase probabilities of single documents in the same way as in slates with higher responses.

Although evaluation of our model requires choosing the ideal conditioning c at or near the edge of the support of P (c), we can always compromise generalization versus performance by controlling c in practice.

Moreover, interactions between documents are governed by similar mechanisms whether they are from the optimal or sub-optimal slates.

As the experiment results indicate, List-CVAE can learn these mechanisms from sub-optimal slates and generalize to optimal slates.

The List-CVAE model moves away from the conventional greedy ranking paradigm, and provides the first conditional generative modeling framework that approaches slate recommendation problem using direct slate generation.

By modeling the conditional probability distribution of documents in a slate directly, this approach not only automatically picks up the positional and contextual biases between documents at both training and inference time, but also gracefully avoids the problem of combinatorial explosion of possible slates when the candidate set is large.

The framework is flexible and can incorporate different types of conditional generative models.

In this paper we showed its superior performance over popular greedy and auto-regressive baseline models with a conditional VAE model.

In addition, the List-CVAE model has good scalability.

We designed an architecture that uses pretrained document embeddings combined with a negatively downsampled k-head softmax layer that greatly speeds up the training, scaling easily to millions of documents.

This test complements the small-scale experiment.

To the 100 documents with slate size 10, we add user features into the conditioning c, by adding a set U of 50 different users to the simulation engine (|U| = 50, n = 100, k = 10), permuting the innate attractiveness of documents and their interactions matrix W by a user-specific function π u .

Let be the response of the user u on the document d i .

During training, the condition c is a concatenation of 16 dimensional user embeddings Θ(u) obtained from the response model, and responses r. At inference time, the model conditions on c = (r , Θ(u)) for each randomly generated test user u. We sweep over hidden layers of 512 or 1024 units in List-CVAE, and all baseline MLP structures.

FIG12 show that slates generated by List-CVAE have on average higher clicks than those produced by the greedy baseline models although its convergence took longer to reach than in the small-scale experiment.

@highlight

We used a CVAE type model structure to learn to directly generate slates/whole pages for recommendation systems.