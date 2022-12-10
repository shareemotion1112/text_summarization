As our experience shows, humans can learn and deploy a myriad of different skills to tackle the situations they encounter daily.

Neural networks, in contrast, have a fixed memory capacity that prevents them from learning more than a few sets of skills before starting to forget them.

In this work, we make a step to bridge neural networks with human-like learning capabilities.

For this, we propose a model with a growing and open-bounded memory capacity that can be accessed based on the model’s current demands.

To test this system, we introduce a continual learning task based on language modelling where the model is exposed to multiple languages and domains in sequence, without providing any explicit signal on the type of input it is currently dealing with.

The proposed system exhibits improved adaptation skills in that it can recover faster than comparable baselines after a switch in the input language or domain.

In a classic cartoon by Gary Larson, a student raises his hand to ask the teacher: "Mr. Osborne, may I be excused?

My brain is full." (Larson & Martin, 2003) .

We laugh at this situation because we know it is absurd.

Human brains don't just get full.

Instead, they seem to be able to keep in their long-term memory massive amounts of information encoding well-acquired knowledge and skills.

Furthermore, the information stored in memory is not necessarily relevant at all times.

For instance, a person may have a phone call in French in the morning, then go about her daily errands in German, and later write an email in English.

Different linguistic knowledge will be required for each of these situations, and context alone, rather than some explicit signal, will dictate what is needed at each given moment.

Vanilla neural network models have been successfully deployed in various applications in the past.

However, they rely on fixed sized memories and suffer from the problem known as "catastrophic forgetting" (McCloskey & Cohen, 1989; Ratcliff, 1990) , which refers to the fact that previously acquired information is quickly forgotten as novel skills need to be mastered.

Earlier work attempted to correct this problem by looking for available capacity on a fixed-sized network that would allow encoding a new solution without affecting previously learned tasks (Kirkpatrick et al., 2017; Zenke et al., 2017; Serrà et al., 2018; Lopez-Paz & Ranzato, 2017; Fernando et al., 2017; Lee et al., 2017) .

The problem with this approach is that eventually, the system will run out of available capacity.

Instead, here we argue for developing models that can grow their internal capacity.

While some work has also relied on growing the model to face catastrophic forgetting (Rusu et al., 2016; Li & Hoiem, 2018; Aljundi et al., 2017) , they all rely, to the best of our knowledge, on an explicit signal identifying the task that the system is currently solving.

Indeed, most work dealing with catastrophic forgetting has evaluated the models on settings often making unrealistic assumptions.

Not only they typically provided the model with an explicit identifier for the task at hand, but also tasks featured unnatural properties, such as scrambled pixels, or categories that were incrementally added, but presented sequentially on blocks once and for all, and never encountered again during training.

Only recently, some work has started tackling continual learning in a more realistic task-agnostic way (Aljundi et al., 2019 ).

Yet, there are no standard publicly available datasets that can help the evaluation of continual learning systems on more natural settings.

In this paper, we make a two-fold contribution towards task agnostic continual learning.

First, we introduce a recurrent neural network that can grow its memory by creating new modules as training progresses.

Rather than using all modules simultaneously, or indexing them based on a task identification signal, our model learns to weight their contributions to adapt to the current context.

Second, we introduce to the community a multilingual/multidomain language modelling task with switching domains that we hope can fit this bill.

We propose two variants of it.

The first is a character-based language modelling benchmark with text written in 5 different languages that randomly switch between one another.

The second one is a word-based language modelling task, where the text oscillates between 4 different domains.

No segmentation signal is given when there is a switch, making the models having to discover it autonomously while they are evaluated for their adaptation skills.

Our experimental results show that our system can switch between different domains faster than comparable neural networks.

Furthermore, our model is very general because it does not make any assumption about the type of underlying neural network architecture and thus, it can easily be adopted for tackling other tasks in conjunction with any other neural network system.

Growth in neural networks has been explored with different perspectives.

Here, we present a discussion of the possible avenues for developing neural networks with unbounded memory.

1.

Growth of layers: Early work on Neural Networks used this method to reduce the amount of computational complexity or for escaping local minima (Fahlman & Lebiere, 1990; Hirose et al., 1991 ).

The goal, back then, was using the smallest possible number of hidden units.

Here, instead, we are interested in allowing neural networks to grow for endowing them with larger memory capacity.

In this sense, this strategy seems limited because all units remain fully connected at all time, forcing the network to access all memories simultaneously.

2.

Growth of architecture: A different type of growth could be the one dictated by a different model that decides the characteristics of the learning system, including how many units to put in it.

Neural architecture search (Elsken et al., 2018) and, particularly, neuro-evolution (Stanley et al., 2019) provide good examples of this.

Note, however, that this type of growth is different from the main problem that we are dealing with here, in which a model needs to be able to extend itself.

3. Learned, structured growth: Models, like the Stack-RNNs permit the model to create new units, which are placed on a stack data structure, allowing it thus to have a flexible memory to process different problem instances of varying sizes.

The model itself learns how many computational resources to use, but so far this has been demonstrated only on toy problems like sequence memorization.

Moreover, Stack-RNNs are also unable to quickly recover "memories" from distant past because it would imply cycling through the whole stack.

4.

Sparse growth: This is the strategy that we focus on in this paper.

The network is grown by blocks or modules.

One potential drawback with this strategy is the linear increase of time complexity as the network grows.

To prevent this, here we simply limit the maximum number of modules that are kept alive at any given time.

Other, more sophisticated, options could employ a Hierarchical Softmax (Goodman, 2001 ) operation over the modules or Winner-Takes-All types of rules (Srivastava et al., 2013) , essentially searching for just the right memories to answer the current situation.

The proposed Growing Long-Term Memory Network (GLTMN) is composed of modules operating in concert to compute the network's output by means of a weighted combination of their predictions.

As such, it belongs to a family of architectures that, depending on whether the predictions are additively or multiplicatively combined, are referred as Mixture-of-Experts (Jacobs et al., 1991; Eigen et al., 2013) or Product-of-Experts (Hinton, 1999) .

Before combining them, all the module's predictions are weighted by a vector of coefficients that can be produced by another module that is jointly trained.

Or system differs in the following ways.

First, the modules in our system are subdivided into two main blocks: the short-term memory (STM) and long-term memory (LTM).

These two components differ in the following ways.

First, while the STM has a fixed number of modules, the LTM grows incrementally, only being limited in size by the hosting computer memory capacity.

Second, while predictions are computed as a standard MoE architecture, using both LTM and STM modules, only the latter ones gets trained on incoming experience.

Mixture weights that encompass both LTM and STEM are kept as a separate parameter of the system that is continually trained based on recent experience.

Modules in the STM are consolidated into LTM whenever a trigger point is reached (such as, after a given number of examples have been processed) choosing the module that has been contributing the most to the output according to its corresponding mixture weight recent history.

At this point, the module is removed from STM and frozen into LTM.

Similarly, LTM modules can be reinstated back into STM by picking the module with the highest contribution weight and copying back into STM.

When a maximum size is reached, the LTM module that was reinstated into STM is removed, thus keeping the overall size of the model constant (see Figure 1 for a general sketch).

Learning More formally, our model is composed of a set of modules M = {M 1 , . . .

, M n }, where M 1 , . . . , M l are the modules in the LTM, and {M l+1 , . . .

, M n } are the ones in STM.

At the beginning of training all modules belong to STM, thus l = 0.

The system computes its predictions as follows.

When an input x (with target y) is observed, it is fed to all modules M 1...n , obtaining log-linear output vectorsŷ

.

An additional vector of mixture weights w ∈ R n is used to linearly combine them.

The output of the full model y is computed as a linear combination of the individual modules outputs weighted by the parameters w i :

Note that since we are combining the model unnormalized predictions before the application of the softmax, we are effectively computing a geometric combination of each individual module's unnormalized probabilities:

.

Thus, this model can be seen as a Product of Experts.

Compared to a Mixture of Experts, this approach does not require to normalize the output of each individual model, thus being much more efficient to compute.

The predictions are then used to compute the cross-entropy loss L(ŷ, y), which is then used to backpropagate both into the STM modules and the mixture weights, and then optimized independently through gradient descent.

In particular, to swiftly adapt the weights, we repeat the training update for w multiple (k = 100) times, whereas we only do it once for the modules.

Note that in order to compute the gradients of the loss with respect to the weights after each update there is no need to recompute each module's output and thus, each step is not expensive to compute.

Memory management Every T processed examples, the system consolidates a STM module into LTM.

For this, it picks the module that has been most active in recent experience, as measured by the absolute mean weight value over the past examples (we use the last 20 batches).

To limit the amount of computational power needed, we restricted the maximum total number of modules to n = 30.

When this limit is reached, another module is removed from LTM and reinstated back into STM for further training.

We pick the module with the highest mixture weight in absolute value.

That is, while STM modules are selected for consolidation based on their past importance so they can be preserved for future use, LTM modules are reinstated based on their present relevance, so they can be further adapted.

Note that, despite that, in practice the model has a memory limit, its potential to allocate new memory is unbounded, not unlike modern computer systems where physical memory limits do not affect the way programs are written.

In the future, the memory efficiency could be improved by incorporating mechanisms such as distillation (Hinton et al., 2015) to compress the information stored across different modules into a single one.

Last, but not least, note that the above-described model does not make any assumption about the type of architecture used.

In our following experiments, we evaluate the model using an LSTM architecture, but there is nothing preventing it to be applied to feed-forward, convolutional or other types of networks; or even a mixture thereof.

In this work we instantiate the GLTMN for an online language modelling task.

For this we adopt double-layered LSTM networks as modules.

Each of these modules observe text as small sequences of tokens, as is standard when applying the backpropagation through time algorithm, and has to predict each of the upcoming tokens by acting in concert with all the other modules 1 .

When the system is not allowed to use the LTM, our model reduces to a Products of Experts (PoE) where the mixture coefficients are given by an independent weight vector parameter.

To make this architecture comparable to previously proposed neural mixture models (e.g. Eigen et al. (2013) ), we consider as a baseline a PoE model where the weights are computed by another network.

In particular, we used another LSTM network that looks at the current batch and produces the weights.

Our experimental setup aims to establish whether our model is able to adapt to a continuous stream of circumstances in which it needs to develop, improve and use a wide range of skills.

While most previous work interested in continual learning considered sequences of tasks that were unambiguously identified by a marker given as an extra input to the model, here we are interested in a more realistic setup where only context can dictate which skills are required at a given time.

For this, we introduce two lifelong language modelling tasks, where the model is continually exposed to a novel linguistic stream that switches between different languages or domains without any explicit marker signalling the change.

More specifically, we propose two language modelling benchmarks: One is wordlevel and multi-domain whereas the other is character-level and multilingual.

Both benchmarks feature conflicting learning signals when moving between domains or languages, making the network susceptible to catastrophic forgetting.

A good model should be very good at transitioning between languages or domains, while still maintaining good overall performance for (online) language modelling.

We are interested in modelling the continual adaptation to incoming non-i.i.d.

data, a situation that is closer to the learning experience of any human being.

Therefore, the traditional train-test split approach is not adequate here.

Instead, we adopt an online learning paradigm.

This means that at each time step the model receives an instance x t and makes a predictionŷ t .

Then, the true target y t will be observed, with the model incurring in a loss L(ŷ t , y t ).

After reporting this loss, the model is trained, possibly for more than a single iteration, on the just observed example.

The goal is minimizing the cumulative loss English, French, Spanish, German and Czech because they all have similar character sets, while also showing interesting linguistic variability thanks to belonging to three different Indo-European branches: Romance (French and Spanish), Germanic (English and German) and Slavic (Czech).

Compared to earlier multilingual corpora (Kawakami et al., 2017) , our dataset was carefully constructed to include only linguistically valid character sets, in order to prevent non-linguistic noise from interfering with our experiments.

For this, we removed all lines from the input that containing characters appearing less than 100 times on the full orpus.

The resulting character vocabulary is no bigger than 215 characters.

The second dataset is an English multi-domain dataset.

For this, we used four different source corpora: news (same as above), europarl (Koehn, 2005) , the Toronto Book Corpus (Zhu et al., 2015) and Wikipedia (Merity et al., 2016) .

We kept in the vocabulary the top 25K words for each corpus, which after merging yielded a vocabulary size of 58K words.

We then split the corpus in fragments coming from different languages or domains with lengths randomly sampled from a (truncated) exponential distribution.

Thanks to the memorylessness property of this distribution, it is virtually impossible to estimate when the next switch is going to happen.

For the multilingual dataset, we extracted 1M and 10M-characters-long randomly alternating combinations of 100 sequences, 10 for each language, with lengths sampled from a (truncated) exponential distribution with means λ = 10k and λ = 100k characters, respectively, and a 10M-characters-long one with 1000 sequences with mean length of 10k characters.

For the multi-domain dataset we followed the same procedure, extracting 100 alternating sequences with mean lengths of λ = 10k and λ = 20k, for a total of 1M and 2M words.

We used a smaller corpus in this last case because to allow for faster experimentation as the models have now to predict over a larger vocabulary, and thus they require more training time.

We set the maximum size of the GLTMN to 30 LSTM modules having two layers and 200 hidden units each.

We first compared it to the performance of a single equally-sized LSTM network, allowing us to measure the advantage of any mixture model with respect to any given single module.

Second, we included as a further baseline another double-layered LSTM with 1300 units which has the same amount of parameters as our fully grown model on the multilingual task.

As reference points, we also trained independent LSTMs, one for each domain or language (thus, using for diagnostic purposes a "forbidden" domain supervision signal), enabling us to compare the performance of our model to a situation where there is no forgetting from conflicting learning signals, but also where there is no possibility of transferring learned representations across possibly related domains.

Finally, we evaluated a PoE, as described in Section 3.1.

This is a model whose mixture coefficients are produced by a simple LSTM module with 10 hidden units.

We also experimented with passing the weights through a softmax layer, thus enforcing a convex rather than a linear combination of the modules, but this did not prove useful neither for the GLTMN nor the PoE. We tuned the hyperparameters of all the models on a development set for each corpus.

Among the ones that we considered for the GLTMN was the size of the STM choosing between 10, 15, 20 or all 30 modules.

In the latter case, the model reduces to a PoE with a weights parametrized by a trainable vector of coefficients.

Using 20 modules for the STM proved to be the best-performing strategy.

Details of the hyperparamter search for the models are included in Appendix A.1.

We are interested in measuring whether the growing model brings in any advantage at recovering information that the network had learned before, while remaining competitive in terms of overall performance.

To measure these aspects, we propose the following metrics:

Online perplexity This is the general perplexity over the data measured during model training.

Note that since the task is framed as an online learning one, the training loss serves as a test measure because no example is seen twice.

Post-switch confusion: When a distribution shift is experienced, a neural network that suffers from forgetting typically produces a spike.

With this measure we aim at capturing how large was this spike.

Let L avg be the average cross-entropy loss of the model between a switch occurring at t = 0 and the following one, and let t avg be the time step at which the model touches this level for the first time.

Then, we define confusion as:

That is, confusion computes the number of time steps weighted by the relative perplexity increase during which the model remains above the average loss for that sequence.

We also complement this measure with plots that illustrate this process.

In order to observe the asymptotic behaviour of the models, we restrict our analysis by reporting measures pertaining only to the second half of the data.

We report our experimental results for both the multilingual task and for the multi-domain data in Table 1 .

Results disaggregated by domain or language are reported in Appendix A.2.

There are several aspects that are worth pointing out in these results.

First, we can see that the results in overall online perplexity are mixed.

The PoE with LSTM weights scores the biggest number of wins (two), followed by ours and a plain LSTM, winning one each.

In general, we can conclude that there is a slight advantage for the PoE model, which may be due to its higher learning capacity during stable non-switching periods, but the GLTMN performs almost on-par with it.

Moreover, when looking at the surprisal measures, we can see that the GLTMN excels at recovering after changes in the distribution.

This observation can be graphically confirmed by looking at Figure 2 .

As it can be seen, the GLTMN recovers much faster than the stand-alone LSTM and the PoE models.

It does spike, however, on the first batch, although this is completely expected.

Recall that weights of the model are optimized according to each new batch.

Thus, before the fist batch of the new data was observed it was not possible for it to adapt.

We also note that while on the multilingual data, a LSTM trained independently on each different language (first row in the table) exhibits the lowest perplexity, this does not hold in the multi-domain corpus.

This shows that while there is limited room for transferring knowledge in the multilingual case, in consonance with previous results (Dhar & Bisazza, 2018) .

In contrast, the multi-domain setting provides plenty of opportunities for transferring knowledge across each domain, and thus task-agnostic systems can benefit them.

The main takeaways are that enhancing the neural network with a modular structure (e.g. PoE, GLTMN) brings an improvement to the general performance, but the LTM component helps the network to recover faster after the task switch.

To get some further insights into the workings of the model we analyzed its weight vectors in the multilingual and the multi-domain tasks (λ = 10k) for the last seven exposures to each language or domain.

We then averaged all the weight vectors on every batch for each of these seven linguistic sequences.

In Figure 3 we show all these vectors sorted by the corresponding type of sequence, while weight indices were sorted by means of a clustering scheme to highlight modules that are working in tandem.

Note that any of these modules could be in LTM or STM at any given time.

In the multilingual case, we can see that there are a large number of modules allocated both to Czech and German.

This may be explained by the fact that these are the two hardest domains as measured by the disaggregated perplexities (see Appendix for details).

While there is some transfer of modules between languages, this remains quite limited, consistently with our observation above that models trained independently on each linguistic stream reach the lowest online perplexity.

In contrast, for the multi-domain corpus, the clustering is much less clear.

While the weights for the book corpus and the europarl domains seem to be mostly anti correlated and quite idiosyncratic, the modules acting on the news and wiki domains seem to be more distributed.

This may also be explaining our findings that knowledge transfer helps in this task.

Our work is related to the efforts aimed at solving the problem of catastrophic forgetting in neural networks (McCloskey & Cohen, 1989; Ratcliff, 1990; French, 1999; Goodfellow et al., 2013) , which have received considerable attention in the machine learning community.

These approaches can be categorized into mainly two different branches: Those that keep the neural network size fixed and attempt to correct the learning procedure to avoid forgetting (Kirkpatrick et al., 2017; Zenke et al., 2017; Serrà et al., 2018; Lopez-Paz & Ranzato, 2017; Fernando et al., 2017; Lee et al., 2017) , and those that allow the system to grow new modules to account for novel tasks (Rusu et al., 2016; Li & Hoiem, 2018; Aljundi et al., 2017; d'Autume et al., 2019) .

Our work is closer to the second stem.

Also, close to our approach are mixture of experts systems (Jacobs et al., 1991; Eigen et al., 2013; Shazeer et al., 2017; Wang et al., 2018; Yang et al., 2017; Bakhtin et al., 2018) , in particular the product of experts approach (Hinton, 1999 ).

Other models with unbounded memory were proposed in the past (Fahlman & Lebiere, 1990; Rusu et al., 2016; Li & Hoiem, 2018; Aljundi et al., 2017) , although not all of them were studied in the context of continual learning, as we are doing here, and those who were assumed training tasks to be properly identified, as previously noted.

Similar to our work are the models enahnced with a memory component, such as: memory networks (Sukhbaatar et al., 2015) , stack RNNs and neural turing machines (Graves et al., 2014) which show that having a structured memory helps with learning longer dependencies and remembering.

While our approach has some similarities, the proposed model saves fully connected modules which can save into the memory not only data but also the algorithms learned by the modules.

The interaction between recent and remote memory has been extensively studied in the neuroscientific literature (McClelland et al., 1995) .

We do not claim any direct connection between our model and how the human brain works, but we borrow the terms associated with consolidation and reinstatement of memories, as they fit quite neatly into our context.

Finally, our problem formulation is an instance of neural-network-assisted language modelling (Bengio et al., 2003; Mikolov et al., 2010) and character level language modeling (Sutskever et al., 2011; Mikolov et al., 2012; Graves, 2013; Bojanowski et al., 2015) .

Some models conceived for language modeling can extend their memories to support fast-changing statistics from the recent past, as in cache models (Grave et al., 2017; Merity et al., 2016) .

Also, some other work has extended these models towards the multilingual setting (Östling & Tiedemann, 2016) .

Here, we adapt these problems to a life-long learning setup where different languages can be conceived as different tasks.

Differently form cache models, context switching implies retrieving a vast set of skills from a relatively distant past.

We believe that developing more flexible forms of artificial intelligence will probably require flexible memory capabilities that can only be delivered by models capable of growth.

Here we have proposed a method based on growing full-fledged modules over time.

We explored a particular instantiation of this architecture in which modules are grown at a constant rate and consolidated into a long-term memory (LTM).

Once the model has reached a maximum size, memories can be still be consolidated into LTM by reinstating LTM modules back into STM (see Figure 1 ).

Furthermore, we introduced to the community two lifelong language modelling tasks.

One, characterbased and multilingual, and other, word-based on multiple domains.

Our experiments confirm the efficacy of our Growing LTM model, showing that it can learn to adapt much faster than comparable baselines without suffering in terms of its overall performance.

The proposed system is very flexible, allowing it to be used with any neural network architecture.

While here we have studied it in the lifelong language modeling setting, we believe that the system will also show promising results in other domains with similar requirements, such as robotics -where the model can learn to deal with different kinds of terrains-or image recognition -where it can learn different kinds of visual information depending on the contextual requirements (Rebuffi et al., 2017) .

In the future, mechanisms that exploit the structure of the input data for associating it with the relevant sets of models (Aljundi et al., 2017; Milan et al., 2016) can be explored.

Furthermore, we plan to study mechanisms that would allow the model to decide when to grow, rather than keeping a constant schedule.

In the long term, the model should be capable of deciding how to structure its long-term memory and whether or not to grow it, as Stack-RNNs do to grow the working memory.

Moreover, we are interested in exploring how communication between memories can be enabled through a central routing mechanism, in a similar fashion to the model proposed by Hafner et al. (2017) .

To conclude, in this work we have given a step -and we hope that more will follow-in providing neural networks with flexible memory structures.

We expect that further pursuing this goal will pave the way towards developing more general learning systems and, fundamentally, that in the future neural networks will no longer need to be excused from class just because their weights are full.

@highlight

We introduce a continual learning setup based on language modelling where no explicit task segmentation signal is given and propose a neural network model with growing long term memory to tackle it.