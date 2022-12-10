Temporal point processes are the dominant paradigm for modeling sequences of events happening at irregular intervals.

The standard way of learning in such models is by estimating the conditional intensity function.

However, parameterizing the intensity function usually incurs several trade-offs.

We show how to overcome the limitations of intensity-based approaches by directly modeling the conditional distribution of inter-event times.

We draw on the literature on normalizing flows to design models that are flexible and efficient.

We additionally propose a simple mixture model that matches the flexibility of flow-based models, but also permits sampling and computing moments in closed form.

The proposed models achieve state-of-the-art performance in standard prediction tasks and are suitable for novel applications, such as learning sequence embeddings and imputing missing data.

Visits to hospitals, purchases in e-commerce systems, financial transactions, posts in social media -various forms of human activity can be represented as discrete events happening at irregular intervals.

The framework of temporal point processes is a natural choice for modeling such data.

By combining temporal point process models with deep learning, we can design algorithms able to learn complex behavior from real-world data.

Designing such models, however, usually involves trade-offs along the following dimensions: flexibility (can the model approximate any distribution?), efficiency (can the likelihood function be evaluated in closed form?), and ease of use (is sampling and computing summary statistics easy?).

Existing methods (Du et al., 2016; Mei & Eisner, 2017; Omi et al., 2019) that are defined in terms of the conditional intensity function typically fall short in at least one of these categories.

Instead of modeling the intensity function, we suggest treating the problem of learning in temporal point processes as an instance of conditional density estimation.

By using tools from neural density estimation (Bishop, 1994; Rezende & Mohamed, 2015) , we can develop methods that have all of the above properties.

To summarize, our contributions are the following:

• We connect the fields of temporal point processes and neural density estimation.

We show how normalizing flows can be used to define flexible and theoretically sound models for learning in temporal point processes.

• We propose a simple mixture model that performs on par with the state-of-the-art methods.

Thanks to its simplicity, the model permits closed-form sampling and moment computation.

• We show through a wide range of experiments how the proposed models can be used for prediction, conditional generation, sequence embedding and training with missing data.

as a sequence of strictly positive inter-event times τ i = t i − t i−1 ∈ R + .

Representations in terms of t i and τ i are isomorphic -we will use them interchangeably throughout the paper.

The traditional way of specifying the dependency of the next arrival time t on the history H t = {t j ∈ T : t j < t} is using the conditional intensity function λ * (t) := λ(t|H t ).

Here, the * symbol reminds us of dependence on H t .

Given the conditional intensity function, we can obtain the conditional probability density function (PDF) of the time τ i until the next event by integration (Rasmussen, 2011) as p * (τ i ) := p(τ i |H ti ) = λ * (t i−1 + τ i ) exp − τi 0 λ * (t i−1 + s)ds .

Learning temporal point processes.

Conditional intensity functions provide a convenient way to specify point processes with a simple predefined behavior, such as self-exciting (Hawkes, 1971 ) and self-correcting (Isham & Westcott, 1979) processes.

Intensity parametrization is also commonly used when learning a model from the data: Given a parametric intensity function λ * θ (t) and a sequence of observations T , the parameters θ can be estimated by maximizing the log-likelihood: θ * = arg max θ i log p * θ (τ i ) = arg max θ i log λ *

The main challenge of such intensity-based approaches lies in choosing a good parametric form for λ

We develop several approaches for modeling the distribution of inter-event times.

First, we assume for simplicity that each inter-event time τ i is conditionally independent of the history, given the model parameters (that is, p * (τ i ) = p(τ i )).

In Section 3.1, we show how state-of-the-art neural density estimation methods based on normalizing flows can be used to model p(τ i ).

Then in Section 3.2, we propose a simple mixture model that can match the performance of the more sophisticated flowbased models, while also addressing some of their shortcomings.

Finally, we discuss how to make p(τ i ) depend on the history H ti in Section 3.3.

The core idea of normalizing flows (Tabak & Turner, 2013; Rezende & Mohamed, 2015) is to define a flexible probability distribution by transforming a simple one.

Assume that z has a PDF q(z).

Let x = g(z) for some differentiable invertible transformation g : Z → X (where Z, X ⊆ R) 2 .

We can obtain the PDF p(x) of x using the change of variables formula as p(x) = q(g −1 (x))

∂ ∂x g −1 (x) .

By stacking multiple transformations g 1 , ..., g M , we obtain an expressive probability distribution p(x).

To draw a sample x ∼ p(x), we need to draw z ∼ q(z) and compute the forward transformation x = (g M • · · · • g 1 )(z).

To get the density of an arbitrary point x, it is necessary to evaluate the inverse transformation z = (g −1 1 • · · · • g −1 M )(x) and compute q(z).

Modern normalizing flows architectures parametrize the transformations using extremely flexible functions f θ , such as polynomials (Jaini et al., 2019) or neural networks (Krueger et al., 2018) .

The flexibility of these functions comes at a cost -while the inverse f −1 θ exists, it typically doesn't have a closed form.

That is, if we use such a function to define one direction of the transformation in a flow model, the other direction can only be approximated numerically using iterative root-finding methods (Ho et al., 2019) .

In this work, we don't consider invertible normalizing flows based on dimension splitting, such as RealNVP (Dinh et al., 2017) , since they are not applicable to 1D data.

In the context of TPPs, our goal is to model the distribution p(τ ) of inter-event times.

In order to be able to learn the parameters of p(τ ) using maximum likelihood, we need to be able to evaluate the density at any point τ .

For this we need to define the inverse transformation g −1 := (g

M (τ ) = log τ to convert a positive τ ∈ R + into z M ∈ R. Then, we stack multiple layers of parametric functions f θ : R → R that can approximate any transformation.

We consider two choices for f θ : deep sigmoidal flow (DSF) from Krueger et al. (2018) and sum-ofsquares (SOS) polynomial flow from Jaini et al.

where a, w, s, µ are the transformation parameters, K is the number of components, R is the polynomial degree, and σ(x) = 1/(1 + e −x ).

We denote the two variants of the model based on f

and f SOS building blocks as DSFlow and SOSFlow respectively.

Finally, after stacking multiple g −1 m = f θm , we apply a sigmoid transformation g −1 1 = σ to convert z 2 into z 1 ∈ (0, 1).

For both models, we can evaluate the inverse transformations (g

, which means the model can be efficiently trained via maximum likelihood.

The density p(τ ) defined by either DSFlow or SOSFlow model is extremely flexible and can approximate any distribution (Section 3.4).

However, for some use cases, this is not sufficient.

For example, we may be interested in the expected time until the next event, E p [τ ] .

In this case, flow-based models are not optimal, since for them E p [τ ] does not in general have a closed form.

Moreover, the forward transformation (g M • · · · • g 1 ) cannot be computed in closed form since the functions f DSF and f SOS cannot be inverted analytically.

Therefore, sampling from p(τ ) is also problematic and requires iterative root finding.

This raises the question: Can we design a model for p(τ ) that is as expressive as the flow-based models, but in which sampling and computing moments is easy and can be done in closed form?

Model definition.

While mixture models are commonly used for clustering, they can also be used for density estimation.

Mixtures work especially well in low dimensions (McLachlan & Peel, 2004) , which is the case in TPPs, where we model the distribution of one-dimensional inter-event times τ .

Since the inter-event times τ are positive, we choose to use a mixture of log-normal distributions to model p(τ ).

The PDF of a log-normal mixture is defined as

2 All definitions can be extended to R D for D > 1.

We consider the one-dimensional case since our goal is to model the distribution of inter-event times τ ∈ R+.

Figure 1 : Model architecture.

Parameters of p * (τ i |θ i ) are generated based on the conditional information c i .

where w are the mixture weights, µ are the mixture means, and s are the standard deviations.

Because of its simplicity, the log-normal mixture model has a number of attractive properties.

Moments.

Since each component k has a finite mean, the mean of the entire distribution can be computed as

, a weighted average of component means.

Higher moments can be computed based on the moments of each component (Frühwirth-Schnatter, 2006) .

Sampling.

While flow-based models from Section 3.1 require iterative root-finding algorithms to generate samples, sampling from a mixture model can be done in closed form:

where z is a one-hot vector of size K. In some applications, such as reinforcement learning (Upadhyay et al., 2018), we might be interested in computing gradients of the samples w.r.t.

the model parameters.

The samples τ drawn using the procedure above are differentiable with respect to the means µ and scales s.

By using the Gumbel-softmax trick (Jang et al., 2017) when sampling z, we can obtain gradients w.r.t.

all the model parameters (Appendix D.6).

Such reparametrization gradients have lower variance and are easier to implement than the score function estimators typically used in other works (Mohamed et al., 2019) .

Other flexible models (such as multi-layer flow models from Section 3.1) do not permit sampling through reparametrization, and thus are not well-suited for the above-mentioned scenario.

In Section 5.4, we show how reparametrization sampling can also be used to train with missing data by performing imputation on the fly.

History.

A crucial feature of temporal point processes is that the time τ i = (t i − t i−1 ) until the next event may be influenced by all the events that happened before.

A standard way of capturing this dependency is to process the event history H ti with a recurrent neural network (RNN) and embed it into a fixed-dimensional vector h i ∈ R H (Du et al., 2016) .

Conditioning on additional features.

The distribution of the time until the next event might depend on factors other than the history.

For instance, distribution of arrival times of customers in a restaurant depends on the day of the week.

As another example, if we are modeling user behavior in an online system, we can obtain a different distribution p * (τ ) for each user by conditioning on their metadata.

We denote such side information as a vector y i .

Such information is different from marks (Rasmussen, 2011), since (a) the metadata may be shared for the entire sequence and (b) y i only influences the distribution p * (τ i |y i ), not the objective function.

In some scenarios, we might be interested in learning from multiple event sequences.

In such case, we can assign each sequence T j a learnable sequence embedding vector e j .

By optimizing e j , the model can learn to distinguish between sequences that come from different distributions.

The learned embeddings can then be used for visualization, clustering or other downstream tasks.

Obtaining the parameters.

We model the conditional dependence of the distribution p * (τ i ) on all of the above factors in the following way.

The history embedding h i , metadata y i and sequence embedding e j are concatenated into a context vector c i = [h i ||y i ||e j ].

Then, we obtain the parameters of the distribution p * (τ i ) as an affine function of c i .

For example, for the mixture model we have

where the softmax and exp transformations are applied to enforce the constraints on the distribution parameters, and {V w , V s , V µ , b w , b s , b µ } are learnable parameters.

Such model resembles the mixture density network architecture (Bishop, 1994) .

The whole process is illustrated in Figure 1 .

We obtain the parameters of the flow-based models in a similar way (see Appendix D).

Universal approximation.

The SOSFlow and DSFlow models can approximate any probability density on R arbitrarily well (Jaini et al., 2019, Theorem 3), (Krueger et al., 2018, Theorem 4) .

It turns out, a mixture model has the same universal approximation (UA) property.

Theorem 1 (DasGupta, 2008, Theorem 33.2) .

Let p(x) be a continuous density on R. If q(x) is any density on R and is also continuous, then, given ε > 0 and a compact set S ⊂ R, there exist number of components K ∈ N, mixture coefficients w ∈ ∆ K−1 , locations µ ∈ R K , and scales

This results shows that, in principle, the mixture distribution is as expressive as the flow-based models.

Since we are modeling the conditional density, we additionally need to assume for all of the above models that the RNN can encode all the relevant information into the history embedding h i .

This can be accomplished by invoking the universal approximation theorems for RNNs (Siegelmann & Sontag, 1992; Schäfer & Zimmermann, 2006) .

Note that this result, like other UA theorems of this kind (Cybenko, 1989; Daniels & Velikova, 2010) , does not provide any practical guarantees on the obtained approximation quality, and doesn't say how to learn the model parameters.

Still, UA intuitively seems like a desirable property of a distribution.

This intuition is supported by experimental results.

In Section 5.1, we show that models with the UA property consistently outperform the less flexible ones.

Interestingly, Theorem 1 does not make any assumptions about the form of the base density q(x).

This means we could as well use a mixture of distribution other than log-normal.

However, other popular distributions on R + have drawbacks: log-logistic does not always have defined moments and gamma distribution doesn't permit straightforward sampling with reparametrization.

Intensity function.

For both flow-based and mixture models, the conditional cumulative distribution function (CDF) F * (τ ) and the PDF p * (τ ) are readily available.

This means we can easily compute the respective intensity functions (see Appendix A).

However, we should still ask whether we lose anything by modeling p * (τ ) instead of λ * (t).

The main arguments in favor of modeling the intensity function in traditional models (e.g. self-exciting process) are that it's intuitive, easy to specify and reusable (Upadhyay & Rodriguez, 2019) .

"

Intensity function is intuitive, while the conditional density is not." -While it's true that in simple models (e.g. in self-exciting or self-correcting processes) the dependence of λ * (t) on the history is intuitive and interpretable, modern RNN-based intensity functions (as in Du et al. (2016) ; Mei & Eisner (2017); Omi et al. (2019) ) cannot be easily understood by humans.

In this sense, our proposed models are as intuitive and interpretable as other existing intensity-based neural network models.

"λ * (t) is easy to specify, since it only has to be positive.

On the other hand, p * (τ ) must integrate to one." -As we saw, by using either normalizing flows or a mixture distribution, we automatically enforce that the PDF integrates to one, without sacrificing the flexibility of our model.

"Reusability: If we merge two independent point processes with intensitites λ * 1 (t) and λ * 2 (t), the merged process has intensity λ * (t) = λ * 1 (t) + λ * 2 (t)." -An equivalent result exists for the CDFs F * 1 (τ ) and F * 2 (τ ) of the two independent processes.

The CDF of the merged process is obtained as

2 (τ ) (derivation in Appendix A).

As we just showed, modeling p * (τ ) instead of λ * (t) does not impose any limitation on our approach.

Moreover, a mixture distribution is flexible, easy to sample from and has well-defined moments, which favorably compares it to other intensity-based deep learning models.

Neural temporal point processes.

Fitting simple TPP models (e.g. self-exciting (Hawkes, 1971) or self-correcting (Isham & Westcott, 1979 ) processes) to real-world data may lead to poor results because of model misspecification.

Multiple recent works address this issue by proposing more flexible neural-network-based point process models.

These neural models are usually defined in terms of the conditional intensity function.

For example, Mei & Eisner (2017) propose a novel RNN architecture that can model sophisticated intensity functions.

This flexibility comes at the cost of inability to evaluate the likelihood in closed form, and thus requiring Monte Carlo integration.

Du et al. (2016) suggest using an RNN to encode the event history into a vector h i .

The history embedding h i is then used to define the conditional intensity, for example, using the constant intensity model λ et al., 2018; Huang et al., 2019) or the more flexible exponential intensity model λ et al., 2016; Upadhyay et al., 2018) .

By considering the conditional distribution p * (τ ) of the two models, we can better understand their properties.

Constant intensity corresponds to an exponential distribution, and exponential intensity corresponds to a Gompertz distribution (see Appendix B).

Clearly, these unimodal distributions cannot match the flexibility of a mixture model (as can be seen in Figure 8 ).

Omi et al. (2019) introduce a flexible fully neural network (FullyNN) intensity model, where they model the cumulative intensity function Λ * (τ ) with a neural net.

The function Λ * converts τ into an exponentially distributed random variable with unit rate (Rasmussen, 2011), similarly to how normalizing flows model p * (τ ) by converting τ into a random variable with a simple distribution.

However, due to a suboptimal choice of the network architecture, the PDF of the FullyNN model does not integrate to 1, and the model assigns non-zero probability to negative inter-event times (see Appendix C).

In contrast, SOSFlow and DSFlow always define a valid PDF on R + .

Moreover, similar to other flow-based models, sampling from the FullyNN model requires iterative root finding.

Several works used mixtures of kernels to parametrize the conditional intensity function (Taddy et al., 2012; Tabibian et al., 2017; Okawa et al., 2019) .

Such models can only capture self-exciting influence from past events.

Moreover, these models do not permit computing expectation and drawing samples in closed form.

Recently, Biloš et al. (2019) and Türkmen et al. (2019) proposed neural models for learning marked TPPs.

These models focus on event type prediction and share the limitations of other neural intensity-based approaches.

Other recent works consider alternatives to the maximum likelihood objective for training TPPs.

Examples include noise-contrastive estimation (Guo et al., 2018) , Wasserstein distance (Xiao et al., 2017; 2018; Yan et al., 2018) , and reinforcement learning (Li et al., 2018; Upadhyay et al., 2018) .

This line of research is orthogonal to our contribution, and the models proposed in our work can be combined with the above-mentioned training procedures.

Neural density estimation.

There exist two popular paradigms for learning flexible probability distributions using neural networks: In mixture density networks (Bishop, 1994) , a neural net directly produces the distribution parameters; in normalizing flows (Tabak & Turner, 2013; Rezende & Mohamed, 2015) , we obtain a complex distribution by transforming a simple one.

Both mixture models (Schuster, 2000; Eirola & Lendasse, 2013; Graves, 2013) and normalizing flows (Oord et al., 2016; Ziegler & Rush, 2019 ) have been applied for modeling sequential data.

However, surprisingly, none of the existing works make the connection and consider these approaches in the context of TPPs.

We evaluate the proposed models on the established task of event time prediction (with and without marks) in Sections 5.1 and 5.2.

In the remaining experiments, we show how the log-normal mixture model can be used for incorporating extra conditional information, training with missing data and learning sequence embeddings.

We use 6 real-world datasets containing event data from various domains: Wikipedia (article edits), MOOC (user interaction with online course system), Reddit (posts in social media) (Kumar et al., 2019) , Stack Overflow (badges received by users), LastFM (music playback) (Du et al., 2016) , and Yelp (check-ins to restaurants).

We also generate 5 synthetic datasets (Poisson, Renewal, Self-correcting, Hawkes1, Hawkes2), as described in Omi et al. (2019) .

Detailed descriptions and summary statistics of all the datasets are provided in Appendix E.

Setup.

We consider two normalizing flow models, SOSFlow and DSFlow (Equation 1), as well a log-normal mixture model (Equation 2), denoted as LogNormMix.

As baselines, we consider RMTPP (i.e. Gompertz distribution / exponential intensity from Du et al. (2016)) and FullyNN model by Omi et al. (2019) .

Additionally, we use a single log-normal distribution (denoted LogNormal) to highlight the benefits of the mixture model.

For all models, an RNN encodes the history into a vector h i .

The parameters of p * (τ ) are then obtained using h i (Equation 3).

We exclude the NeuralHawkes model from our comparison, since it is known to be inferior to RMTPP in time prediction (Mei & Eisner, 2017) , and, unlike other models, doesn't have a closed-form likelihood.

Each dataset consists of multiple sequences of event times.

The task is to predict the time τ i until the next event given the history H ti .

For each dataset, we use 60% of the sequences for training, 20% for validation and 20% for testing.

We train all models by minimizing the negative log-likelihood (NLL) of the inter-event times in the training set.

To ensure a fair comparison, we try multiple hyperparameter configurations for each model and select the best configuration using the validation set.

Finally, we report the NLL loss of each model on the test set.

All results are averaged over 10 train/validation/test splits.

Details about the implementation, training process and hyperparameter ranges are provided in Appendix D. For each real-world dataset, we report the difference between the NLL loss of each method and the LogNormMix model (Figure 3) .

We report the differences, since scores of all models can be shifted arbitrarily by scaling the data.

Absolute scores (not differences) in a tabular format, as well as results for synthetic datasets are provided in Appendix F.1.

Results.

Simple unimodal distributions (Gompertz/RMTPP, LogNormal) are always dominated by the more flexible models with the universal approximation property (LogNormMix, DSFlow, SOSFlow, FullyNN).

Among the simple models, LogNormal provides a much better fit to the data than RMTPP/Gompertz.

The distribution of inter-event times in real-world data often has heavy tails, and the Gompertz distributions fails to capture this behavior.

We observe that the two proposed models, LogNormMix and DSFlow consistently achieve the best loss values.

Setup.

We apply the models for learning in marked temporal point processes.

Marks are known to improve performance of simpler models (Du et al., 2016) , we want to establish whether our proposed models work well in this setting.

We use the same setup as in the previous section, except for two differences.

The RNN takes a tuple (τ i , m i ) as input at each time step, where m i is the mark.

Moreover, the loss function now includes a term for predicting the next mark:

Results.

Figure 3 (right) shows the time NLL loss (i.e. − i log p * (τ i )) for Reddit and MOOC datasets.

LogNormMix shows dominant performance in the marked case, just like in the previous experiment.

Like before, we provide the results in tabular format, as well as report the marks NLL loss in Appendix F.

Setup.

We investigate whether the additional conditional information (Section 3.3) can improve performance of the model.

In the Yelp dataset, the task is predict the time τ until the next check-in for a given restaurant.

We postulate that the distribution p * (τ ) is different, depending on whether it's a weekday and whether it's an evening hour, and encode this information as a vector y i .

We consider 4 variants of the LogNormMix model, that either use or don't use y i and the history embedding h i .

Results.

Figure 5 shows the test set loss for 4 variants of the model.

We see that additional conditional information boosts performance of the LogNormMix model, regardless of whether the history embedding is used.

In practical scenarios, one often has to deal with missing data.

For example, we may know that records were not kept for a period of time, or that the data is unusable for some reason.

Since TPPs are a generative model, they provide a principled way to handle the missing data through imputation.

Setup.

We are given several sequences generated by a Hawkes process, where some parts are known to be missing.

We consider 3 strategies for learning from such a partially observed sequence: (a) ignore the gaps, maximize log-likelihood of observed inter-event times (b) fill the gaps with the average τ estimated from observed data, maximize log-likelihood of observed data, and (c) fill the gaps with samples generated by the model, maximize the expected log-likelihood of the observed points.

The setup is demonstrated in Figure 4 .

Note that in case (c) the expected value depends on the parameters of the distribution, hence we need to perform sampling with reparametrization to optimize such loss.

A more detailed description of the setup is given in Appendix F.4.

Results.

The 3 model variants are trained on the partially-observed sequence.

Figure 4 shows the NLL of the fully observed sequence (not seen by any model at training time) produced by each strategy.

We see that strategies (a) and (b) overfit the partially observed sequence.

In contrast, strategy (c) generalizes and learns the true underlying distribution.

The ability of the LogNormMix model to draw samples with reparametrization was crucial to enable such training procedure.

Different sequences in the dataset might be generated by different processes, and exhibit different distribution of inter-event times.

We can "help" the model distinguish between them by assigning a trainable embedding vector e j to each sequence j in the dataset.

It seems intuitive that embedding vectors learned this way should capture some notion of similarity between sequences.

Learned sequence embeddings.

We learn a sequence embedding for each of the sequences in the synthetic datasets (along with other model parameters).

We visualize the learned embeddings using t-SNE (Maaten & Hinton, 2008) in Figure 7 colored by the true class.

As we see, the model learns to differentiate between sequences from different distributions in a completely unsupervised way.

Generation.

We fit the LogNormMix model to two sequences (from self-correcting and renewal processes), and, respectively, learn two embedding vectors e SC and e RN .

After training, we generate 3 sequences from the model, using e SC , 1 /2(e SC + e RN ) and e RN as sequence embeddings.

Additionally, we plot the learned conditional intensity function of our model for each generated sequence ( Figure 6 ).

The model learns to map the sequence embeddings to very different distributions.

We use tools from neural density estimation to design new models for learning in TPPs.

We show that a simple mixture model is competitive with state-of-the-art normalizing flows methods, as well as convincingly outperforms other existing approaches.

By looking at learning in TPPs from a different perspective, we were able to address the shortcomings of existing intensity-based approaches, such as insufficient flexibility, lack of closed-form likelihoods and inability to generate samples analytically.

We hope this alternative viewpoint will inspire new developments in the field of TPPs.

CDF and conditional intensity function of proposed models.

The cumulative distribution function (CDF) of a normalizing flow model can be obtained in the following way.

If z has a CDF Q(z) and τ = g(z), then the CDF F (τ ) of τ is obtained as

Since for both SOSFlow and DSFlow we can evaluate g −1 in closed form, F (τ ) is easy to compute.

For the log-normal mixture model, CDF is by definition equal to

where Φ(·) is the CDF of a standard normal distribution.

Given the conditional PDF and CDF, we can compute the conditional intensity λ * (t) and the cumulative intensity Λ * (τ ) for each model as

where t i−1 is the arrival time of most recent event before t (Rasmussen, 2011).

Merging two independent processes.

We replicate the setup from Upadhyay & Rodriguez (2019) and consider what happens if we merge two independent TPPs with intensity functions λ * 1 (t) and λ * 2 (t) (and respectively, cumulative intensity functions Λ * 1 (τ ) and Λ * 2 (τ )).

According to Upadhyay & Rodriguez (2019) , the intensity function of the new process is λ * (t) = λ * 1 (t) + λ * 2 (t).

Therefore, the cumulative intensity function of the new process is

Using the previous result, we can obtain the CDF of the merged process as

The PDF of the merged process is obtained by simply differentiating the CDF w.r.t.

τ .

This means that by using either normalizing flows or mixture distributions, and thus directly modeling PDF / CDF, we are not losing any benefits of the intensity parametrization.

Constant intensity model as exponential distribution.

The conditional intensity function of the constant intensity model (Upadhyay et al., 2018 ) is defined as λ

H is the history embedding produced by an RNN, and b ∈ R is a learnable parameter.

By setting c = exp(v T h i + b), it's easy to see that the PDF of the constant intensity model p * (τ ) = c exp(−c) corresponds to an exponential distribution.

PDF of a Gompertz distribution (Wienke, 2010) is defined as The conditional intensity function of the exponential intensity model (Du et al., 2016 ) is defined as λ

, where h i ∈ R H is the history embedding produced by an RNN, and v ∈ R H , b ∈ R, w ∈ R + are learnable parameters.

By defining d = v T h i + b, we obtain the PDF of the exponential intensity model (Du et al., 2016, Equation 12 ) as

By setting α = exp(d) and β = w we see that the exponential intensity model is equivalent to a Gompertz distribution.

Discussion.

Figure 8 shows densities that can be represented by exponential and Gompertz distributions.

Even though the history embedding h i produced by an RNN may capture rich information, the resulting distribution p * (τ i ) for both models has very limited flexibility, is unimodal and light-tailed.

In contrast, a flow-based or a mixture model is significantly more flexible and can approximate any density.

Summary The main idea of the approach by Omi et al. (2019) is to model the integrated conditional intensity function

using a feedforward neural network with non-negative weights

are non-negative weight matrices, and

(3) ∈ R are the remaining model parameters.

FullyNN as a normalizing flow Let z ∼ Exponential(1), that is

We can view f : R + → R + as a transformation that maps τ to z

We can now use the change of variables formula to obtain the conditional CDF and PDF of τ .

Alternatively, we can obtain the conditional intensity as

and use the fact that p

Both approaches lead to the same conclusion

However, the first approach also provides intuition on how to draw samplesτ from the resulting distribution p * (τ ) -an approach known as the inverse method (Rasmussen, 2011)

1.

Samplez ∼ Exponential(1) 2.

Obtainτ by solving f (τ ) −z = 0 for τ (using e.g. bisection method)

Similarly to other flow-based models, sampling from the FullyNN model cannot be done exactly and requires a numerical approximation.

1.

The PDF defined by the FullyNN model doesn't integrate to 1.

By definition of the CDF, the condition that the PDF integrates to 1 is equivalent to lim τ →∞ F * (τ ) = 1, which in turn is equivalent to lim τ →∞

Λ * (τ ) = ∞. However, because of saturation of tanh activations (i.e. sup x∈R | tanh(

Therefore, the PDF doesn't integrate to 1.

2.

The FullyNN model assigns a non-zero amount of probability mass to the (−∞, 0) interval, which violates the assumption that inter-event times are strictly positive.

Since the inter-event times τ are assumed to be strictly positive almost surely, it must hold that Prob(τ ≤ 0) = F * (0) = 0, or equivalently Λ * (0) = 0.

However, we can see that

which means that the FullyNN model permits negative inter-event times.

We implement SOSFlow, DSFlow and LogNormMix, together with baselines: RMTPP (Gompertz distribution), exponential distribution and a FullyNN model.

All of them share the same pipeline, from the data preprocessing to the parameter tuning and model selection, differing only in the way we calculate p * (τ ).

This way we ensure a fair evaluation.

Our implementation uses Pytorch.

From arival times t i we calculate the inter-event times τ i = t i − t i−1 .

Since they can contain very large values, RNN takes log-transformed and centered inter-event time and produces h i ∈ R H .

In case we have marks, we additionally input m i -the index of the mark class from which we get mark embedding vector m i .

In some experiments we use extra conditional information, such as metadata y i and sequence embedding e j , where j is the index of the sequence.

As illustrated in Section 3.3 we generate the parameters θ of the distribution p * (τ i ) from [h i ||y i ||e j ] using an affine layer.

We apply a transformation of the parameters to enforce the constraints, if necessary.

All decoders are implemented using a common framework relying on normalizing flows.

By defining the base distribution q(z) and the inverse transformation (g

) we can evaluate the PDF p * (τ ) at any τ , which allows us to train with maximum likelihood (Section 3.1).

The log-normal mixture distribution is defined in Equation 2.

We generate the parameters of the distribution w ∈ R K , µ ∈ R K , s ∈ R K (subject to k w k = 1, w k ≥ 0 and s k > 0), using an affine transformation (Equation 3).

The log-normal mixture is equivalent to the following normalizing flow model

By using the affine transformation z 2 = az 1 + b before the exp transformation, we obtain a better initialization, and thus faster convergence.

This is similar to the batch normalization flow layer (Dinh et al., 2017)

(log τ i − b) are estimated using the entire dataset, not using batches.

Forward direction samples a value from a Gaussian mixture, applies an affine transformation and applies exp.

In the bacward direction we apply log-transformation to an observed data, center it with an affine layer and compute the density under the Gaussian mixture.

We implement FullyNN model (Omi et al., 2019) as described in Appendix C, using the official implementation as a reference 4 .

The model uses feed-forward neural network with non-negative weights (enforced by clipping values at 0 after every gradient step).

Output of the network is a cumulative intensity function Λ * (τ ) from which we can easily get intensity function λ * (τ ) as a derivative w.r.t.

τ using automatic differentiation in Pytorch.

We get the PDF as p

We implement RMTPP / Gompertz distribution (Du et al., 2016) 5 and the exponential distribution (Upadhyay et al., 2018) models as described in Appendix B.

All of the above methods define the distribution p * (τ ).

Since the inter-event times may come at very different scales, we apply a linear scalingτ = aτ , where a = 1 N N i=1 τ i is estimated from the data.

This ensures a good initialization for all models and speeds up training.

A single layer of DSFlow model is defined as

.

We obtain the parameters of each layer using Equation 3.

We define p(τ ) through the inverse transformation (g

We use the the batch normalization flow layer (Dinh et al., 2017) between every pair of consecutive layers, which significantly speeds up convergence.

A single layer of SOSFlow model is defined as

There are no constraints on the polynomial coefficients a ∈ R (R+1)×K .

We obtain a similarly to Equation 3 as a = V a c + b a , where c is the context vector.

We define p(τ ) by through the inverse transformation (g

Same as for DSFlow, we use the the batch normalization flow layer between every pair of consecutive layers.

When implementing SOSFlow, we used Pyro 6 for reference.

Using a log-normal mixture model allows us to sample with reparametrization which proves to be useful, e.g. when imputing missing data (Section 5.4).

In a score function estimator (Williams, 1992) given a random variable x ∼ p θ (x), where θ are parameters, we can compute

.

This is an unbiased estimator of the gradients but it often suffers from high variance.

If the function f is differentiable, we can obtain an alternative estimator using the reparametrization trick: ∼ q( ), x = g θ ( ).

Thanks to this reparametrization, we can compute

.

Such reparametrization estimator typically has lower variance than the score function estimator (Mohamed et al., 2019) .

In both cases, we estimate the expectation using Monte Carlo.

To sample with reparametrization from the mixture model we use the Straight-Through Gumbel Estimator (Jang et al., 2017) .

We first obtain a relaxed sample z * = softmax((log w + o)/T ), where each o i is sampled i.i.d.

from a Gumbel distribution with zero mean and unit scale, and T is the temperature parameter.

Finally, we get a one-hot sample z = onehot(arg max k z * k ).

While a discrete z is used in the forward pass, during the backward pass the gradients will flow through the differentiable z * .

The gradients obtained by the Straight-Through Gumbel Estimator are slightly biased, which in practice doesn't have a significant effect on the model's performance.

There exist alternatives (Tucker et al., 2017; Grathwohl et al., 2018 ) that provide unbiased gradients, but are more expensive to compute.

E DATASET STATISTICS E.1 SYNTHETIC DATA Synthetic data is generated according to Omi et al. (2019) using well known point processes.

We sample 64 sequences for each process, each sequence containing 1024 events.

Poisson.

Conditional intensity function for a homogeneous (or stationary) Poisson point process is given as λ * (t) = 1.

Constant intensity corresponds to exponential distribution.

Renewal.

A stationary process defined by a log-normal probability density function p(τ ), where we set the parameters to be µ = 1.0 and σ = 6.0.

Sequences appear clustered.

LastFM  929  1268385  Reddit  10000  672350  Stack Overflow  6633  480414  MOOC  7047  396633  Wikipedia  1000  157471  Yelp  300  215146   Table 2 : Dataset statistics.

Self-correcting.

Unlike the previous two, this point process depends on the history and is defined by a conditional intensity function λ * (t) = exp(t − ti<t 1).

After every new event the intensity suddenly drops, inhibiting the future points.

The resulting point patterns appear regular.

Hawkes.

We use a self-exciting point process with a conditional intensity function given as λ

In addition we use real-world datasets that are described bellow.

Table 2 shows their summary.

All datasets have a large amount of unique sequences and the number of events per sequence varies a lot.

Using marked temporal point processes to predict the type of an event is feasible for some datasets (e.g. when the number of classes is low), and is meaningless for other.

7 The dataset contains sequences of songs that selected users listen over time.

Artists are used as an event type.

8 On this social network website users submit posts to subreddits.

In the dataset, most active subreddits are selected, and posts from the most active users on those subreddits are recodered.

Each sequence corresponds to a list of submissions a user makes.

The data contains 984 unique subreddits that we use as classes in mark prediction.

9 Users of a question-answering website get rewards (called badges) over time for participation.

A sequence contains a list of rewards for each user.

Only the most active users are selected and only those badges that users can get more than once.

8 Contains the interaction of students with an online course system.

An interaction is an event and can be of various types (97 unique types), e.g. watching a video, solving a quiz etc.

8 A sequence corresponds to edits of a Wikipedia page.

The dataset contains most edited pages and users that have an activity (number of edits) above a certain threshold.

10 We use the data from the review forum and consider the reviews for the 300 most visited restaurants in Toronto.

Each restaurant then has a corresponding sequence of reviews over time.

After splitting the data into the 3 sets, we break down long training sequences into sequences of length at most 128.

Optimization is performed using Adam (Kingma & Ba, 2015) with learning rate 10 −3 .

We perform training using mini-batches of 64 sequences.

We train for up to 2000 epochs (1 epoch = 1 full pass through all the training sequences).

For all models, we compute the validation loss at every epoch.

If there is no improvement for 100 epochs, we stop optimization and revert to the model parameters with the lowest validation loss.

We select hyperparameter configuration for each model that achieves the lowest average loss on the validation set.

For each model, we consider different values of L 2 regularization strength C ∈ {0, 10 −5 , 10 −3 }.

Additionally, for SOSFlow we tune the number of transformation layers M ∈ {1, 2, 3} and for DSFlow M ∈ {1, 2, 3, 5, 10}. We have chosen the values of K such that the mixture model has approximately the same number of parameters as a 1-layer DSFlow or a 1-layer FullyNN model.

More specifically, we set K = 64 for LogNormMix, DSFlow and FullyNN.

We found all these models to be rather robust to the choice of K, as can be seen in Table 3 for LogNormMix.

For SOSFlow we used K = 4 and R = 3, resulting in a polynomial of degree 7 (per each layer).

Higher values of R led to unstable training, even when using batch normalization.

Additional discussion.

In this experiment, we only condition the distribution p * (τ i ) on the history embedding h i .

We don't learn sequence embeddings e j since they can only be learned for the training sequences, and not fore the validation/test sets.

There are two important aspects related to the NLL loss values that we report.

First, the absolute loss values can be arbitrarily shifted by rescaling the data.

Assume, that we have a distribution p(τ ) that models the distribution of τ .

Now assume that we are interested in the distribution q(x) of x = aτ (for a > 0).

Using the change of variables formula, we obtain log q(x) = log p(τ ) + log a. This means that by simply scaling the data we can arbitrarily offset the log-likelihood score that we obtain.

Therefore, the absolute values of of the (negative) log-likelihood L for different models are of little interest -all that matters are the differences between them.

The loss values are dependent on the train/val/test split.

Assume that model 1 achieves loss values L 1 = {1.0, 3.0} on two train/val/test splits, and model 2 achieves L 2 = {2.0, 4.0} on the same splits.

If we first aggregate the scores and report the averageL 1 = 2.0 ± 1.0,L 2 = 3.0 ± 1.0, it may seem that the difference between the two models is not significant.

However, if we first compute the differences and then aggregate (L 2 − L 1 ) = 1.0 ± 0.0 we see a different picture.

Therefore, we use the latter strategy in Figure 3 .

For completeness, we also report the numbers obtained using the first strategy in Table 4 .

As a baseline, we also considered the constant intensity / exponential distribution model (Upadhyay et al., 2018) .

However, we excluded the results for it from Figure 3 , since it consistently achieved the worst loss values and had high variance.

We still include the results for the constant intensity model in Table 4 .

We also performed all the experiments on the synthetic datasets (Appendix E.1).

The results are shown in Table 5 , together with NLL scores under the true model.

We see that LogNormMix and DSFlow, besides achieving the best results, recover the true distribution.

Finally, in Figure 9 we plot the conditional distribution p(τ |H) with models trained on Yelp dataset.

The events represent check-ins into a specific restaurant.

Since check-ins mostly happen during the opening hours, the inter-event time is likely to be on the same day (0h), next day (24h), the day after (48h), etc.

LogNormMix can fully recover this behavior from data while others either cannot learn multimodal distributions (e.g. RMTPP) or struggle to capture it (e.g. FullyNN).

Table 4 : Time prediction test NLL on real-world data.

Detailed setup.

We use the same setup as in Section F.1, except two differences.

For learning in a marked temporal point process, we mimic the architecture from Du et al. (2016) .

The RNN takes a tuple (τ i , m i ) as input at each time step, where m i is the mark.

Moreover, the loss function now includes a term for predicting the next mark:

The next mark m i at time t i is predicted using a categorical distribution p * (m i ).

The distribution is parametrized by the vector π i , where π i,c is the probability of event m i = c. We obtain π i using the history embedding h i passed through a feedforward neural network π i = softmax V π are the parameters of the neural network.

Additional discussion.

In Figure 3 (right) we reported the differences in time NLL between different models L time (θ) = − 1 N N i=1 log p * θ (τ i ).

In Table 6 we additionally provide the total NLL L total (θ) = − 1 N N i=1 [log p * θ (τ i ) + log p * θ (m i )] averaged over multiple splits.

Using marks as input to the RNN improves time prediction quality for all the models.

However, since we assume that the marks are conditionally independent of the time given the history (as was done in earlier works), all models have similar mark prediction accuracy.

Detailed setup.

In the Yelp dataset, the task is to predict the time τ i until the next customer checkin, given the history of check-ins up until the current time t i−1 .

We want to verify our intuition that the distribution p * (τ i ) depends on the current time t i−1 .

For example, p * (τ i ) might be different depending on whether it's a weekday and / or it's an evening hour.

Unfortunately, a model that processes the history with an RNN cannot easily obtain this information.

Therefore, we provide this information directly as a context vector y i when modeling p * (τ i ).

The first entry of context vector y i ∈ {0, 1} 2 indicates whether the previous event t i−1 took place on a weekday or a weekend, and the second entry indicates whether t i−1 was in the 5PM-11PM time Table 6 : Time and total NLL and mark accuracy when learning a marked TPP.

window.

To each of the four possibilities we assign a learnable 64-dimensional embedding vector.

The distribution of p * (τ i ) until the next event depends on the embedding vector of the time stamp t i−1 of the most recent event.

Detailed setup.

The dataset for the experiment is generated as a two step process: 1) We generate a sequence of 100 events from the model used for Hawkes1 dataset (Appendix E.1) resulting in a sequence of arrival times {t 1 , . . .

t N }, 2) We choose random t i and remove all the events that fall inside the interval [t i , t i+k ] where k is selected such that the interval length is approximately t N /3.

We consider three strategies for learning with missing data (shown in Figure 4 (left)): a) No imputation.

The missing block spans the time interval [t i , t i+k ].

We simply ignore the missing data, i.e. training objective L time will include an inter-event time τ = t i+k − t i .

b) Mean imputation.

We estimate the average inter-event timeτ from the observed data, and impute events at times {t i + nτ for n ∈ N, such that t i + nτ < t i+k }.

These imputed events are fed into the history-encoding RNN, but are not part of the training objective.

c) Sampling .

The RNN encodes the history up to and including t i and produces h i that we use to define the distribution p * (τ |h i ).

We draw a sample τ (imp) j form this distribution and feed it into the RNN.

We keep repeating this procedure until the samples get past the point t i+k .

The imputed inter-event times τ ).

We sample multiple such sequences in order to approximate the expected log-likelihood of the observed inter-event times E τ (imp) ∼p * i log p * (τ

) .

Since this objective includes an expectation that depends on p * , we make use of reparametrization sampling to obtain the gradients w.r.t.

the distribution parameters (Mohamed et al., 2019) .

Detailed setup.

When learning sequence embeddings, we train the model as described in Appendix F.1, besides one difference.

First, we pre-train the sequence embeddings e j by disabling the his-tory embedding h i and optimizing − 1 N i log p θ (τ i |e j ).

Afterwards, we enable the history and minimize − 1 N i log p θ (τ i |e j , h i ).

In Figure 6 the top row shows samples generated using e SC , embedding of a self-correcting sequence, the bottom row was generated using e SC , embedding of a renewal sequence, and the middle row was generated using 1 /2(e SC + e RN ), an average of the two embeddings.

@highlight

Learn in temporal point processes by modeling the conditional density, not the conditional intensity.