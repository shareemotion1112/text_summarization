Up until very recently, inspired by a mass of researches on adversarial examples for computer vision, there has been a growing interest in designing adversarial attacks for Natural Language Processing (NLP) tasks, followed by very few works of adversarial defenses for NLP.

To our knowledge, there exists no defense method against the successful synonym substitution based attacks that aim to satisfy all the lexical, grammatical, semantic constraints and thus are hard to perceived by humans.

We contribute to fill this gap and propose a novel adversarial defense method called Synonym Encoding Method (SEM), which inserts an encoder before the input layer of the model and then trains the model to eliminate adversarial perturbations.

Extensive experiments demonstrate that SEM can efficiently defend current best synonym substitution based adversarial attacks with little decay on the accuracy for benign examples.

To better evaluate SEM, we also design a strong attack method called Improved Genetic Algorithm (IGA) that adopts the genetic metaheuristic for synonym substitution based attacks.

Compared with existing genetic based adversarial attack, IGA can achieve higher attack success rate while maintaining the transferability of the adversarial examples.

Deep Neural Networks (DNNs) have made great success in various machine learning tasks, such as computer vision (Krizhevsky et al., 2012; He et al., 2016) , and Natural Language Processing (NLP) (Kim, 2014; Lai et al., 2015; Devlin et al., 2018) .

However, recent studies have discovered that DNNs are vulnerable to adversarial examples not only for computer vision tasks (Szegedy et al., 2014) but also for NLP tasks (Papernot et al., 2016) , causing a serious threat to their safe applications.

For instance, spammers can evade spam filtering system with adversarial examples of spam emails while preserving the intended meaning.

In contrast to numerous methods proposed for adversarial attacks (Goodfellow et al., 2015; Nicholas & David, 2017; Anish et al., 2018) and defenses (Goodfellow et al., 2015; Guo et al., 2018; Song et al., 2019) in computer vision, there are only a few list of works in the area of NLP, inspired by the works for images and emerging very recently in the past two years (Zhang et al., 2019) .

This is mainly because existing perturbation-based methods for images cannot be directly applied to texts due to their discrete property in nature.

Furthermore, if we want the perturbation to be barely perceptible by humans, it should satisfy the lexical, grammatical and semantic constraints in texts, making it even harder to generate the text adversarial examples.

Current attacks in NLP can fall into four categories, namely modifying the characters of a word (Liang et al., 2017; Ebrahimi et al., 2017) , adding or removing words (Liang et al., 2017) , replacing words arbitrarily (Papernot et al., 2016) , and substituting words with synonyms (Alzantot et al., 2018; Ren et al., 2019) .

The first three categories are easy to be detected and defended by spell or syntax check (Rodriguez & Rojas-Galeano, 2018; Pruthi et al., 2019) .

As synonym substitution aims to satisfy all the lexical, grammatical and semantic constraints, it is hard to be detected by automatic spell or syntax check as well as human investigation.

To our knowledge, currently there is no defense method specifically designed against the synonym substitution based attacks.

In this work, we postulate that the model generalization leads to the existence of adversarial examples: a generalization that is not strong enough causes the problem that there usually exists some neighbors x of a benign example x in the manifold with a different classification.

Based on this hypothesis, we propose a novel defense mechanism called Synonym Encoding Method (SEM) that encodes all the synonyms to a unique code so as to force all the neighbors of x to have the same label of x. Specifically, we first cluster the synonyms according to the Euclidean Distance in the embedding space to construct the encoder.

Then we insert the encoder before the input layer of the deep model without modifying its architecture, and train the model again to defend adversarial attacks.

In this way, we can defend the synonym substitution based adversarial attacks efficiently in the context of text classification.

Extensive experiments on three popular datasets demonstrate that the proposed SEM can effectively defend adversarial attacks, while maintaining the efficiency and achieving roughly the same accuracy on benign data as the original model does.

To our knowledge, SEM is the first proposed method that can effectively defend the synonym substitution based adversarial attacks.

Besides, to demonstrate the efficacy of SEM, we also propose a genetic based attack method, called Improved Genetic Algorithm (IGA), which is well-designed and more efficient as compared with the first proposed genetic based attack algorithm, GA (Alzantot et al., 2018) .

Experiments show that IGA can degrade the classification accuracy more significantly with lower word substitution rate than GA.

At the same time IGA keeps the transferability of adversarial examples as GA does.

Let W denote the word set containing all the legal words.

Let x = {w 1 , . . .

, w i , . . . , w n } denote an input text, C the corpus that contains all the possible input texts, and Y ∈ N K the output space.

The classifier f : C → Y takes an input x and predicts its label f (x), and let S m (x, y) denote the confidence value for the y-th category at the softmax layer.

Let Syn(w, σ, k) represent the set of first k synonyms of w within distance σ, namely

where w − w p is the p-norm distance evaluated on the corresponding embedding vectors.

Suppose we have an ideal classifier c : C → Y that could always output the correct label for any input text x. For a subset of (train or test) texts T ⊆ C and a small constant , we could define the natural language adversarial examples as follows:

where d(x − x adv ) is a distance metric to evaluate the dissimilarity between the benign example x = {w 1 , . . .

, w i , . . . , w n } and the adversarial example x adv = {w 1 , . . . , w i , . . .

, w n }.

It is usually defined as the p-norm distance:

In this subsection, we provide a brief overview of three popular synonym substitution based adversarial attack methods.

Greedy Search Algorithm (GSA).

Kuleshov et al. (2018) propose a greedy search algorithm to substitute words with their synonyms so as to maintain the semantic and syntactic similarity.

GSA first constructs a synonym set W s for an input text x = {w 1 , . . .

, w i , . . . , w n }:

Initially, let x adv = x.

Then at each stage for x adv = {w 1 , . . .

, w i , . . . , w n }, GSA finds a wordŵ i ∈ W that satisfies the syntactic constraint and minimizes S m (x, y true ) wherex = {w 1 , . . .

, w i−1 ,ŵ i , w i+1 , . . . , w n }, and updates x adv =x. Such process iterates until x adv becomes an adversarial example or the word replacement rate reaches a threshold.

Genetic Algorithm (GA).

Alzantot et al. (2018) propose a population-based algorithm to replace words with their synonyms so as to generate semantically and syntactically similar adversarial examples.

There are three operators in GA:

• M utate(x): Randomly choose a word w i in text x that has not been updated and substitute w i with w i , one of its synonyms Syn(w i , σ, k) that does not violate the syntax constraint by the "Google 1 billion words language model" (Chelba et al., 2013) and minimize S m (x, y true ) wherex = {w 1 , . . .

, w i−1 , w i , w i+1 , . . . , w n } and S m (x, y true ) < S m (x, y true );

• Sample(P): Randomly sample a text from population P with a probability proportional to 1 − S m (x i , y true ) where x i ∈ P;

• Crossover ( For a text x, GA first generates an initial population P 0 of size m:

Then at each iteration, GA generates the next generation of population through crossover and mutation operators:

GA terminates when it finds an adversarial example or reaches the maximum number of iteration limit.

Probability Weighted Word Saliency (PWWS).

Ren et al. (2019) propose a new synonym substitution method called Probability Weighted Word Saliency (PWWS), which considers the word saliency as well as the classification confidence.

Given a text x = {w 1 , . . .

, w i , . . . , w n }, PWWS first calculates the saliency of each word S(x, w i ):

where "unk" means the word is removed.

Then PWWS calculates the maximum possible change in the classification confidence resulted from substituting word w i with one of its synonyms:

Then, PWWS sequentially checks the words in descending order of φ(S(x, w i )

, and substitutes the current word w i with its optimal synonym w * i :

PWWS terminates when it finds an adversarial example x adv or it has replaced all the words in x.

There exist very few works for text adversarial defenses.

• In the character-level, Pruthi et al. (2019) propose to place a word recognition model in front of the downstream classifier to defend character-level adversarial attacks by combating adversarial spelling mistakes.

• In the word level, for defenses on synonym substitution based attacks, only Alzantot et al. (2018) and Ren et al. (2019) incorporate the adversarial training strategy proposed in the image domain (Goodfellow et al., 2015) with their text attack methods respectively, and demonstrate that adversarial training can promote the model's robustness.

However, there is no defense method specifically designed to defend the synonym substitution based adversarial attacks.

We first introduce our motivation, then present the proposed text defense method, Synonym Encoding Method (SEM).

Let X denote the input space, V (x) denote the -neighborhood of data point x ∈ X , where V (x) = {x ∈ X | x − x < }.

As illustrated in Figure 1 (a), we postulate that the generalization of the model leads to the existence of adversarial examples.

More generally, given a data point x ∈ X , ∃x ∈ V (x), f (x ) = y true where x is an adversarial example of x. Ideally, to defend the adversarial attack, we need to train a classifier f which not only guarantees f (x) = y true , but also assures ∀x ∈ V (x), f (x ) = y true .

Thus, the most effective way is to add more labeled data to improve the adversarial robustness (Schmidt et al., 2018) .

Ideally, as illustrated in Figure 1 (b), if we have infinite labeled data, we can train a model f : ∀x ∈ V (x), f (x ) = y true with high probability so that the model f is robust enough to adversarial examples.

Practically, however, labeling data is very expensive and it is impossible to have infinite labeled data.

Because it is impossible to have enough labeled data to train a robust model, as illustrated in Figure  1 (c), Wong & Kolter (2018) propose to construct a convex outer bound and guarantee that all data points in this bound share the same label.

The goal is to train a model f : ∀x ∈ V , f (x ) = f (x) = y true .

Specifically, they propose a linear-programming (LP) based upper bound on the robust loss by adopting a linear relaxation of the ReLU activation and minimize this upper bound during the training.

Then they bound the LP optimal value and calculate the elementwise bounds on the activation functions based on a backward pass through the network.

Although their method does not need any extra data, it is hard to scale to realistically-sized networks due to its high complexity.

In this work, as illustrated in Figure 1 (d), we propose a novel way to find a mapping m : X → X where ∀x ∈ V (x), m(x ) = x. In this way, we force the classification to be more smooth and we do not need any extra data to train the model or modify the architecture of the model.

All we need to do is to insert the mapping before the input layer and train the model on the original training set.

Now the problem turns into how to locate the neighbors of data point x. For image tasks, it is hard to find all images in the neighborhood of x in the input space, and there could be infinite number of neighbors.

For NLP tasks, however, utilizing the property that words in sentences are discrete tokens, we can easily find almost all neighbors of an input text.

Based on this insight, we propose a new method called Synonym Encoding Method to locate the neighbors of an input x .

We assume that the closer the meaning of two sentences is, the closer their distance is in the input space.

Thus, we can suppose that the neighbors of x are its synonymous sentences.

To find the synonymous sentence, we can substitute words in the sentence with their synonyms.

To construct the mapping m, all we need to do is to cluster the synonyms and allocate a unique token for each cluster, which we call the Synonym Encoding Method (SEM).

The details are in Algorithm 1.

else 8:

end if 10:

for each word w j in Syn(w i , σ, k) do

The current synonym substitution based text adversarial attacks (Alzantot et al., 2018; Kuleshov et al., 2018; Ren et al., 2019) have a constraint that they only substitute words at the same position once or replace words with the first k synonyms of the word in the original input x. This constraint can lead to local minimum for adversarial examples, and it is hard to choose a suitable k as different words may have different number of synonyms.

To address this issue, we propose an Improved Genetic Algorithm (IGA), which allows to substitute words in the same position more than once based on the current text x .

In this way, IGA can traverse all synonyms of a word no matter what value k is.

Meanwhile, we can avoid local minimum to some extent as we allow the substitution of the word by the original word in the current position.

In order to guarantee that the substituted word is still a synonym of the original word, each word in the same position can be replaced at most λ times.

Differs to the first genetic based text attack algorithm of Alzantot et al. (2018) , we change the structure of the algorithm, including the operators for crossover and mutation.

For more details of IGA, see Appendix A.

We evaluate SEM with four attacks, GSA (Kuleshov et al., 2018) , GA (Alzantot et al., 2018) , PWWS (Ren et al., 2019) and our IGA, on three popular datasets involving three neural network classification models.

The results demonstrate that SEM can significantly improve the robustness of neural networks and IGA can achieve better attack performance as compared with existing attacks.

And we further provide discussion on the hyper-parameter of SEM in Appendix B.

We first provide an overview of the datasets and classification models used in the experiments.

Datasets.

In order to evaluate the efficacy of SEM, we choose three popular datasets: IMDB, AG's News, and Yahoo!

Answers.

IMDB (Potts, 2011 ) is a large dataset for binary sentiment classification, containing 25, 000 highly polarized movie reviews for training and 25, 000 for testing.

AG's News (Zhang et al., 2015) consists news article pertaining four classes: World, Sports, Business and Sci/Tech.

Each class contains 30, 000 training examples and 1, 900 testing examples.

Yahoo!

Answers (Zhang et al., 2015) is a topic classification dataset from the "Yahoo!

Answers Comprehensive Questions and Answers" version 1.0 dataset with 10 categories, such as Society & Culture, Science & Mathematics, etc.

Each class contains 140,000 training samples and 5,000 testing samples.

Models.

To better evaluate our method, we adopt several state-of-the-art models for text classification, including Convolution Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).

The embedding dimension for all models are 300 .

We replicate the CNN's architecture from Kim (2014) , which contains three convolutional layers with filter size of 3, 4, and 5 respectively, a max-pooling layer and a fully-connected layer.

LSTM consists of three LSTM layers where each layer has 300 LSTM units and a fully-connected layer (Liu et al., 2016) .

Bi-LSTM contains a bi-directional LSTM layer whose forward and reverse have 300 LSTM units respectively and a fully-connected layer.

Baselines.

We take the method of adversarial training (Goodfellow et al., 2015) as our baseline.

However, due to the low efficiency of text adversarial attacks, we cannot implement adversarial training as it is in the image domain.

In the experiments, we adopt PWWS, which is quicker than GA and IGA, to generate 10% adversarial examples of the training set, and re-train the model incorporating adversarial examples with the training data.

To evaluate the efficiency of the SEM method, we randomly sample 200 correctly classified examples on different models from each dataset and use the above attack methods to generate adversarial examples with or without defense.

The more effective the defense method is, the smaller the classification accuracy of the model drops.

Table 1 shows the efficacy of various attack and defense methods.

For each network model, we look at each row to find the best defense result under the setting of no attack, or GSA, PWWS, GA, and IGA attacks:

• Under the setting of no attack, adversarial training (AT) could improve the classification accuracy of the models on all datasets, as adversarial training (AT) is also the way to augment the training data.

Our defense method SEM reaches an accuracy closing to normal training (NT).

•

Under the four attacks, however, the classification accuracy with normal training (NT) and adversarial training (AT) drops significantly.

For normal training (NT), the accuracy degrades more than 75%, 42% and 40% on the three datasets respectively.

And adversarial training (AT) cannot defend these attacks effectively, especially for PWWS and IGA on IMDB and Yahoo!

Answers, where AT only improves the accuracy a little (smaller than 5%).

By contrast, SEM can remarkably improve the robustness of the deep models for all the four attacks.

In the image domain, the transferability of adversarial attack refers to its ability to decrease the accuracy of models using adversarial examples generated based on other models (Szegedy et al., 2014; Goodfellow et al., 2015) .

Papernot et al. (2016) find that the adversarial examples in NLP also exhibite a good transferability.

Therefore, a good defense method not only could defend the adversarial attacks but also resists the transferability of adversarial examples.

To evaluate the ability of preventing the transferability of adversarial examples, we generate adversarial examples on each model under normal training, and test them on other models with or without defense on Yahoo!

Answers.

The results are shown in Table 2 .

Almost on all models with adversarial examples generated by other models, SEM could yield the highest classification accuracy.

For text attacks, we compare the proposed IGA with GA from various aspects, including attack efficacy, transferability and human evaluation.

Attack Efficacy.

As shown in Table 1 , looking at each column, we see that under normal training (NT) and adversarial training (AT), IGA can always achieve the lowest classification accuracy, which corresponds to the highest attack success rate, on all models and datasets among the four attacks.

Under the third column of SEM defense, though IGA may not be the best among all attacks, IGA always outperforms GA.

Besides, as depicted in Table 3 , IGA can yield lower word substitution rate than GA on most models.

Note that for SEM, GA can yield lower word substitution rate, because GA may not replace the word as most words cannot bring any benefit for the first replacement.

This indicates that GA stops at local minimum while IGA continues to substitute words and gain a lower classification accuracy, as demonstrated in Table 1 .

Transferability.

As shown in Table 2 , the adversarial examples generated by IGA maintain roughly the same transferability as those generated by GA.

For instance, if we generate adversarial examples on Word-CNN (column 2, NT), GA can achieve better transferability on LSTM with NT (column 5) while IGA can achieve better transferability on LSTM with AT and SEM (column 6, 7).

Human Evaluation.

To further verify that the perturbations in the adversarial examples generayed by IGA are hard for humans to perceive, we also perform a human evaluation on IMDB with 35 volunteers.

We first randomly choose 100 benign examples that can be classified correctly and generate adversarial examples by GA and IGA on the three models so that we have a total of 700 examples.

Then we randomly split them into 7 groups where each group contains 100 examples.

We ask every five volunteers to classify one group independently.

The accuracy of human evaluation on benign examples is 93.7%.

As shown in Figure 2 , the classification accuracy of human on adversarial examples generated by IGA is slightly higher than those generated by GA, and is slightly closer to the accuracy of human on benign examples.

Summary.

IGA can achieve the highest attack success rate when compared with previous synonyms substitution based adversarial attacks and yield lower word replacement rate than GA.

Besides, the adversarial examples generated by IGA maintains the same transferability as GA does and are a little bit harder for humans to distinguish.

Several generated adversarial examples by GA and IGA are listed in Appendix C.

Synonym substitution based adversarial attacks are currently the best text attack methods, as they are hard to be checked by automatic spell or syntax check as well as human investigation.

In this work, we propose a novel defense method called Synonym Encoding Method (SEM), which encodes the synonyms of each word to defend adversarial attacks for text classification task.

Extensive experiments show that SEM can defend adversarial attacks efficiently and degrade the transferability of adversarial examples, at the same time SEM maintains the classification accuracy on benign data.

To our knowledge, this is the first and efficient text defense method in word level for state-of-the-art synonym substitution based attacks.

In addition, we propose a new text attack method called Improved Genetic Attack (IGA), which in most cases can achieve much higher attack success rate as compared with existing attacks, at the same time IGA could maintain the transferability of adversarial examples.

Here we introduce our Improved Genetic Algorithm (IGA) in details and show how IGA differs from the first proposed generic attack method, GA (Alzantot et al., 2018) .

Regard a text as a chromosome, there are two operators in IGA:

• • M utate(x, w i ): For a text x = {w 1 , . . .

, w i−1 , w i , w i+1 , . . . , w n } and a position i, replace w i withŵ i whereŵ i ∈ Syn(w , σ, k) to get a new textx = {w 1 , . . .

, w i−1 ,ŵ i , w i+1 , . . . , w n } that minimizes S m (x, y true ).

The details of IGA are described in Algorithm 2.

Algorithm 2 The Improved Genetic Algorithm Input: x: input text, y true : true label for x, M : maximum number of iterations Output: x adv : adversarial example 1: for each word w i ∈ x do 2:

if f (x adv ) = y true then Randomly sample parent 1 , parent 2 from P g−1

:

Randomly choose a word w in child 14: Compared with GA, IGA has the following differences:

• Initialization: GA initializes the first population randomly, while IGA initializes the first population by replacing each word by its optimal synonym, so our population is more diversified.

• M utation: Different from GA, IGA allows to replace the word that has been replaced before so that we can avoid local minimum.

• Crossover: To better simulate the reproduction and biological crossover, we randomly cut the text from two parents and concat two fragments into a new text rather than randomly choose a word of each position from the two parents.

The selection of the next generation is similar to GA, greedily choose one offspring, and then generate other offsprings by M utate(Crossover(·, ·)) on two randomly chosen parents.

But as M utate and Crossover are different, IGA has very different offsprings.

To explore how hyper-parameter of SEM influences the efficacy, we try different ranging from 0 to 1.2 for three models on IMDB with or without adversarial attacks.

The results are illustrated in Figure 3 .

On benign data, as shown in Figure 3(a) , the classification accuracy of the models decreases a little when increases.

Because a bigger indicates that we need less words to train the model, which could degrade the efficacy of the models.

Nevertheless, the classification accuracy does not decrease much as SEM could maintain the semantic invariance of the original text after encoding.

Then we show the defense efficacy of SEM on the three models when changing the value of , as shown in Figure 3 (b)-(d).

When = 0 where SEM could not take any impact, we see that the accuracy is the lowest under all attacks.

When increases, SEM starts to defend the attacks, the accuracy increases rapidly and reach the peak when = 0.5.

Then the accuracy decays slowly if we continue to increase .

Thus, we choose = 0.5 to be a good trade-off on the accuracy of benign examples and adversarial examples.

C ADVERSARIAL EXAMPLES GENERATED BY GA AND IGA

To show the generated adversarial examples, we randomly pick some benign examples from IMDB and generate adversarial examples by GA and IGA respectively on several models.

The examples are shown in Table 5 to Table 6 .

We see that IGA substitutes less words than GA on these models under normal training.

I am sorry but this is the worst film I have ever seen in my life.

I cannot believe that after making the first one in the series, they were able to get a budget to make another.

This is the least scary film I have ever watched and laughed all the way through to the end.

This is a unique masterpiece made by the best director ever lived in the ussr.

He knows the art of film making and can use it very well.

If you find this movie, buy or copy it!

50.6 0 I cared this film which I thought was well written and acted, there was plenty of humour and a igniting storyline, a tepid and enjoyable experience with an emotional ending.

GA 92.7 1 I am sorry but this is the harshest film I have ever seen in my life.

I cannot believe that after making the first one in the series, they were able to get a budget to make another.

This is the least scary film I have ever watched and laughed all the way through to the end.

This is a sole masterpiece made by the nicest director permanently lived in the ussr.

He knows the art of film making and can use it much well.

If you find this movie, buy or copy it!

88.3 0 I enjoyed this film which I think was well written and acted, there was plenty of humour and a causing storyline, a lukewarm and enjoyable experience with an emotional ending.

IGA 70.8 1 I am sorry but this is the hardest film I have ever seen in my life.

I cannot believe that after making the first one in the series, they were able to get a budget to make another.

This is the least scary film I have ever watched and laughed all the way through to the end.

This is a sole masterpiece made by the best director permanently lived in the ussr.

He knows the art of film making and can use it very well.

If you find this movie, buy or copy it!

I am sorry but this is the worst film I have ever seen in my life.

I cannot believe that after making the first one in the series, they were able to get a budget to make another.

This is the least scary film I have ever watched and laughed all the way through to the end.

This is a unique masterpiece made by the best director ever lived in the ussr.

He knows the art of film making and can use it very well.

If you find this movie, buy or copy it!

88.2 0 I enjoyed this film which I thought was well written and proceeded, there was plenty of humorous and a igniting storyline, a tepid and enjoyable experience with an emotional terminate.

GA 99.9 1 I am sorry but this is the hardest film I have ever seen in my life.

I cannot believe that after making the first one in the series they were able to get a budget to make another.

This is the least terrifying film I have ever watched and laughed all the way through to the end.

This is a unique masterpiece made by the best superintendent ever lived in the ussr.

He knows the art of film making and can use it supremely alright.

If you find this movie, buy or copy it!

72.1 0 I enjoyed this film which I thought was well written and acted, there was plenty of humour and a provoking storyline, a lukewarm and agreeable experience with an emotional ending.

IGA 99.8 1 I am sorry but this is the hardest film I have ever seen in my life.

I cannot believe that after making the first one in the series, they were able to get a budget to make another.

This is the least scary film I have ever watched and laughed all the way through to the end.

This is a sole masterpiece made by the best director ever lived in the ussr.

He knows the art of film making and can use it very well.

If you find this movie, buy or copy it!

Table 6 : The adversarial examples generated by GA and IGA on IMDB using Bi-LSTM model.

99.6 1 I enjoyed this film which I thought was well written and acted , there was plenty of humour and a provoking storyline, a warm and enjoyable experience with an emotional ending.

Original 97.0 0 I am sorry but this is the worst film I have ever seen in my life.

I cannot believe that after making the first one in the series, they were able to get a budget to make another.

This is the least scary film I have ever watched and laughed all the way through to the end.

This is a unique masterpiece made by the best director ever lived in the ussr.

He knows the art of film making and can use it very well.

If you find this movie, buy or copy it!

98.2 0 I enjoyed this film which I thought was well written and proceeded, there was plenty of humorous and a igniting storyline, a tepid and enjoyable experiment with an emotional terminate.

GA 78.8 1 I am sorry but this is the hardest film I have ever seen in my life.

I cannot believe that after making the first one in the series, they were able to get a budget to make another.

This is the least scary film I have ever watched and laughed all the way through to the end.

This is a unique masterpiece made by the best superintendent ever lived in the ussr.

He knows the art of film making and can use it supremely alright.

If you find this movie buy or copy it!

81.2 0 I enjoyed this film which I thought was alright written and acted, there was plenty of humour and a arousing storyline, a lukewarm and enjoyable experiment with an emotional ending.

IGA 78.8 1 I am sorry but this is the hardest film I have ever seen in my life.

I cannot believe that after making the first one in the series they were able to get a budget to make another.

This is the least scary film I have ever watched and laughed all the way through to the end.

This is a sole masterpiece made by the best director ever lived in the ussr.

He knows the art of film making and can use it very alright.

If you find this movie buy or copy it!

@highlight

The first text adversarial defense method in word level, and the improved generic based attack method against synonyms substitution based attacks.