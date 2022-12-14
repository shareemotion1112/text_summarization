We elaborate on using importance sampling for causal reasoning, in particular for counterfactual inference.

We show how this can be implemented natively in probabilistic programming.

By considering the structure of the counterfactual query, one can significantly optimise the inference process.

We also consider design choices to enable further optimisations.

We introduce MultiVerse, a probabilistic programming prototype engine for approximate causal reasoning.

We provide experimental results and compare with Pyro, an existing probabilistic programming framework with some of causal reasoning tools.

Machine learning has renewed interest in causal tools to aid reasoning (Pearl, 2018) .

Counterfactuals are particularly special causal questions as they involve the full suite of causal tools: posterior 1 inference and interventional reasoning (Pearl, 2000) .

Counterfactuals are probabilistic in nature and difficult to infer, but are powerful for explanation (Wachter et al., 2017; Sokol and Flach, 2018; Guidotti et al., 2018; Pedreschi et al., 2019) , fairness Kusner et al. (2017) ; Zhang and Bareinboim (2018) ; Russell et al. (2017) , policy search (e.g. Buesing et al. (2019) ) and are also quantities of interest on their own (e.g. Johansson et al. (2016) ).

This has seen counterfactuals applied to medicine (Constantinou et al., 2016; Schulam and Saria, 2017; Richens et al., 2019; Oberst and Sontag, 2019) , advertisement and search (Bottou et al., 2013; Swaminathan and Joachims, 2015; Li et al., 2015; Gilotte et al., 2018) , translation (Lawrence et al., 2017) , and reinforcement learning (Foerster et al., 2018; Forney et al., 2017; Buesing et al., 2019) .

Consequently, counterfactual inference generally requires enhanced tools and inference procedures to incorporate both observation and inter-vention.

Existing frameworks are not fully equipped to handle them naturally, preventing both easy interventional reasoning, as well as optimizations that emerge when considering the full counterfactual query.

A counterfactual query is a what-if ? question: what would have been the outcome had I changed an input?

More formally: what would have happened in the posterior representation of a world (given observations) if in that posterior world one or more things had been forced to change?

This is different from observational inference as it (1) fixes the context of the model to a "posterior world" using observational inference, but (2) then intervenes on one or more variables in that world by forcing each of them take a value.

Interventions in the "posterior" world can cause variables -previously observed or otherwise -to take new values (i.e. "counter" to their observed value, or their distribution).

A counterfactual inference query can be expressed as query over P (K | Y = e; do(D = d)) 2 , given a probabilistic model M = P (X, Y ) that consists of observed variables Y and latent variables X, evidence values e and intervention values d such that E, D ??? X ??? Y .

Variables K are to be predicted after we intervene in the "posterior world" (explained below), and they correspond to variables K ??? X ??? Y in the original world.

Most often, variables K are just variables Y , and so the query becomes P (Y | Y = e; do(D = d)).

Following Pearl (2000), we evaluate this query in three steps:

1.

Abduction (observational inference) -perform the query P (X | Y = e) to receive the joint posterior over the latent variables given the evidence.

This defines the "posterior world".

The result is model M , which has the same structure as model M but where X has been replaced by the joint posterior X | Y = e. In the new model, the previously observed variables Y are not observed anymore.

We denote this step via the do-operator (Pearl, 2000) .

3.

Prediction -we predict the quantities of interest K in M .

Any direct or indirect descendants of D need to be updated prior to estimating a value of interest.

Abduction is the hardest part of the counterfactual procedure as it requires full inference over the joint distribution of the latents.

Abduction by exact inference is possible, but is usually difficult or intractable.

Hence, approximate inference is crucial for counterfactual inference.

There are several approaches for counterfactual inference, including the twin network approach (where approximate inference, e.g. loopy belief propagation, is usually used as in Balke and Pearl (1994) ), single-world intervention graphs (Richardson and Robins, 2013) , matching (Li, 2013) , and more.

We use the standard approach to counterfactual inference as defined above, with its three steps: abduction, intervention, and prediction.

2.

We use the order of operations on the right side of the query (i.e. firstly providing the evidence, and only then performing do), to emphasise that it is a counterfactual query rather than an observational query with an intervention.

Counterfactual notation may seem contradictory, which we discuss in Section C.5.

Probabilistic programming systems (Gordon et al., 2014; Perov, 2016; van de Meent et al., 2018 ) allow a user to: (a) write and iterate over generative probabilistic models as programs easily, (b) set arbitrary evidence for observed variables, and (c) use out-of-the-box, mostly approximate, efficient inference methods to perform queries on the models.

In most cases, probabilistic programming frameworks natively support only observational inference.

Importance sampling is an approximate inference technique that calculates the posterior P (X | Y ) by drawing N samples {s i } from a proposal distribution Q and accumulating the prior, proposal, and likelihood probabilities into weights {w i }.

Given this information, we can compute statistics of interests.

For more details, see Section C.1.

To the best of the authors' knowledge, no major probabilistic programming engine natively supports counterfactual inference.

However, there are (at least) three related directions of work to performing probabilistic causal inference in the settings of probabilistic programming.

We provide a brief overview of these directions below, and we expand on them a little more in Section D. First, it has been shown (Ness, 2019c,b) that in probabilistic programming languages, which support the intervention operation, such as Pyro (Bingham et al., 2018 ) (which has the do operator; or as it can be implemented in Edward (Tran et al., 2018) ), it is possible to write the abduction-intervention-prediction steps in a compositional fashion to perform counterfactual inference (or causal inference, using only the intervention step).

Second, an entirely new probabilistic programming language Omega C (Tavares et al., 2018) for performing counterfactual inference has been recently presented, focusing on carefully considered syntax and semantics.

Third, the use of causal and counterfactual reasoning has been explored in the field of probabilistic logic programming (Baral and Hunsaker, 2007; Baral et al., 2009; Vennekens et al., 2009 ) for probabilistic logic programs, e.g. by the use of the model/query "encoding".

The design, interface and inference approaches presented in our paper can be employed and implemented in almost any probabilistic programming system and are mostly languageindependent.

The presented MultiVerse engine prototype, which is built upon existing probabilistic programming ideas and implementations, employs a single, immutable model approach for causal inference.

This makes counterfactual inference more efficient by changing the inference process, rather than changing the way in which a model is expressed or modified into other derived models.

We can perform inference for the counterfactual query P (K | Y = e, do(D = d)) using importance sampling by modifying the three steps of abduction, intervention, and prediction:

1.

Use importance sampling to obtain a representation of the posterior distribution P (X | Y = e).

The key idea is that the posterior distribution is approximately represented in the form of N samples, s 1 , . . .

, s N , and their weights, w 1 , . . . , w N .

That is, the set of tuples {s i , w i } N i=1 is a valid approximate representation of the posterior distribution P (X | Y = e).

More details on implementing this algorithm for inference for probabilistic programming and its brief computational/memory cost analysis is given in Section C.3.

A key contribution of this paper is MultiVerse: a probabilistic programming system for approximate counterfactual inference that exploits several speed and memory optimisations as a result of considering the counterfactual query and inference scheme.

We have designed MultiVerse to be a fully "native" probabilistic programming engine for causal reasoning in the sense that you define all elements abstractly and independently of each other: a model, observations and interventions, and an inference query, e.g. a counterfactual one.

MultiVerse can perform observational and interventional queries, if chosen, on the same model as well.

In our system there is no double-sampling: for counterfactual inference, we draw a sample and calculate a weight from the proposal distribution given prior and observation likelihood, then we intervene on each sample itself, and predict and estimate its values of interest.

On the other hand, to the best of our understanding, in Pyro 3 one might generally need to redraw samples from the posterior distribution representation and one needs to manually force values of all variables (except intervened ones) in the model to their posterior values per each sample (unless using the internals of Pyro traces).

The latter resampling from already approximated representation introduces an additional approximation error.

Also, in our implementation we save on memory and time as we don't need to define any new model or inference objects beyond the original and as we don't need to pass any models between stages of the inference.

In addition, further optimisations to counterfactual queries can be done by evaluating only those parts of the probabilistic program execution trace that must be evaluated per each step of counterfactual inference.

Because MultiVerse (MV) allows a version of "lazy" evaluation, our tests include prototypical experiments with "Optimised MV" where we only evaluate the necessary parts of the trace per each step.

For more info, see Section A.2.

Probabilistic programming frameworks handle well OBSERVE(variable, value) statements for observational inference: they incorporate the likelihood of the variables given observations.

However, for counterfactual inference, it is generally necessary to represent the noise variables as explicit random variables in the trace because the noise variables should be part of the joint posterior that is received after the abduction step.

In MultiVerse, we introduce "Observable" Random Procedures that are similar to regular Random Procedures but also (a) have an explicit noise variable that is the part of the program trace, and (b) have an inverse function that proposes that variable to a specific value to "match" the hyperparameters of the random procedure and the observation.

For more details, see Section E.

To evaluate the performance of different versions of MultiVerse (i.e. non-optimised and optimised) in terms of speed of sampling and convergence to the true counterfactual query values, we ran counterfactual queries for 1,000 Structural Causal Models which we generated for this experiment.

For comparison, we also ran the same queries in Pyro.

In total, we compared four approximate inference systems: "MultiVerse", "MultiVerse Optimised", Pyro without a smart proposal ("guide"), and Pyro with a smart proposal.

As discussed in Sections 2.3 and E, the smart proposal forces the inference procedure to set each "noise" exogenous variable to a particular value such that the predicted value for an observed endogenous variable -which follows a Delta distribution -is the same as the observed value.

This characteristic is important to prevent significant rejection and is a consequence of the nature of the Structural Causal Model paradigm.

Our experiments show that each system converges but MultiVerse converges in less time (in terms of the constant factor) and more efficiently (in terms of the per-sample inference efficiency) than Pyro (see Section A.2 for more info).

MultiVerse also has the benefit of not needing to pass or store any new model objects.

As expected, Pyro with a smart proposal converges more efficiently than Pyro without one.

Additionally, optimised MultiVerse performs faster than regular MultiVerse.

See Section A for more details on experiments and figures.

In this paper we discuss how to perform counterfactual queries using importance sampling.

Further, we introduce MultiVerse, a probabilistic programming system for causal reasoning that optimises approximate counterfactual inference.

For future work, we aim towards an approximate causal inference engine for any counterfactual query expressed in a probabilistic program, taking advantage of the structure of counterfactual queries to optimise the process of the inference and to choose one of many approximate inference methods.

As causal queries become more used in machine learning, we believe so will flexible and optimised tools that perform these types of inference.

We randomly generated 1,000 Structural Causal Models (Pearl, 2000) (with 15 probabilistic procedures each not counting Delta-distribution procedures), their corresponding Bayesian networks in the form of probabilistic programs, as well as a counterfactual query of the form {Y, D, K } for each model.

On a 16-core EC2 instance m4.4xlarge, we calculated the exact counterfactual value of interest using enumeration, and then compared four approximate systems: "MultiVerse" (i.e. not optimised), "MultiVerse Optimised", Pyro without a smart proposal ("guide"), and Pyro with a smart proposal.

In the experiments, each system converges but both versions of MultiVerse experiments produce the same number of samples in less time than Pyro; for example in the experiments for 5,000 samples, "MultiVerse" produces 5,000 samples, on average, 92.8% faster 4 than Pyro.

Additionally, "MultiVerse Optimised" performs computationally (i.e. in terms of the speed of producing the same number of samples) 26.1% faster 5 than regular "MultiVerse" on average, when compared on generating 1,000,000 samples per run.

In terms of statistical inference convergence quality (i.e. in terms of how well a sampler approximates a statistic of interest), both "MultiVerse" experiments have better inference convergence as well: for example, in the experiments for 5,000 samples, the mean absolute error of predicted values, when compared to the ground truth, for "MultiVerse Optimised" 6 is 0.00539, while for Pyro it is 0.00723; hence "MultiVerse Optimised" inference convergence performance is 25.4% more efficient 7 .

See Section A.3 for figures and details.

The 1,000 Structural Causal Models and their corresponding Bayesian networks with binary variables in the form of probabilistic programs were generated similar to the randomDAG procedure in Kalisch et al. (2012) .

Each Bayesian network contains 15 blocks, where each block can be of two types: 1.

A "prior" exogenous Bernoulli variable with a constant hyperparameter p ??? Uniform- Continuous[0.3, 0.7] that is randomly chosen during network generation.

2.

A dependent endogenous Delta-distribution variable j that has a binary functional output g(f (. . .), ?? j ) and a related "noise" exogenous Bernoulli variable ?? j .

Func-

The exogenous noise variable ?? j with predefined probability q flips (i.e. maps from 1 to 0 and vice versa) the value of f if ?? j = 1.

Probability q is sampled such that q ??? UniformContinuous[0.3, 0.7] during network generation.

Vector ?? j is a vector with elements ?? j,k ??? Beta(5, 5) sampled randomly during network generation and then unit normalised.

Note that every block type contains one and only one "non-trivial" probabilistic procedure (i.e. excluding Delta-distribution procedures).

An example of a network with similar structure is provided in Figure 1 .

Figure 1 : An example of a network with similar structure to the networks which were used for the experiments.

Nodes N 1 and N 2 are "prior" exogenous Bernoulli variables.

Nodes N 3 and N 4 are "dependent" endogenous Delta-distribution variables, where parents of those variable include its own "exogenous noise variable" ?? j .

A structure of a corresponding Structural Causal Model is the same, except there is one more intermediate endogenous variable that deterministically propagates, as an identity function, the values of "prior" exogenous Bernoulli variables.

Each counterfactual query consists of a model G as defined above on which to run the query, which consists of: an evidence set Y (a set of approximately 30% of all nodes, chosen randomly) with evidence value e set to a random value in {0, 1}), a single-node intervention for one node D (chosen randomly) with value d set to a random value in {0, 1} or the flip of

, and a node of interest K (chosen randomly from all nodes except for the first two in the topological order) such that there is an active path between K and D with D being before K in the topological order.

We ran experiments with networks with only binary nodes because it simplifies the computation of the exact value of the counterfactual query result (i.e. the ground truth of it).

However, it is without any loss of generality and can be extended to continuous spaces.

Both Pyro and MultiVerse support both discrete and continuous variables.

An example of a Gaussian model with continuous variables and code for it can be found in Section B.

We run four different versions of experiments:

1. "MultiVerse", which runs the counterfactual query as described in Section C.3.

2. "Optimised MultiVerse", where we calculate only variables that needs to be evaluated for abduction, intervention and prediction steps.

We define lazy evaluations in our probabilistic model in Python using recursive calls.

That is, we start from the variables that we must predict, observe or intervene, and evaluate only those variables and their ancestors recursively.

Note that for intervened variables, we rely on MultiVerse to replace those variables with Delta-distributions with the intervened values, and we don't need to evaluate the parents of those intervened variables during the prediction step.

An illustrative example of using MultiVerse methods for such optimisations with more details is provided in Section F.1.

3.

"Pyro without guide", in which we define a model as a probabilistic program but we don't define any guide.

Because we have observations on Delta variables, this implementation leads to a lot of samples rejected.

4.

"Pyro with guide", in which we define a guide (proposal model) for "noise" Bernoulli variables {?? i }.

That guide forces the values of the noise variables to ensure that each observed variable is flipped or not flipped accordingly given the other parents of the observed variable and given its realised observations (i.e. to match those observations).

Note that for Pyro we:

1. Used Python Pyro package pyro-ppl==0.4.1.

2.

Performed two sampling steps: one for the abduction step, and another one for the intervention step where samples are drawn from the posterior.

That approach of doing counterfactual inference in Pyro was also suggested in (Ness, 2019a) , to the best of our understanding.

Another, more efficient way, would be to re-use Pyro traces directly; that way we can avoid the second sampling step (e.g. by using vectorised importance weights which might make it significantly more efficient computationally as well).

The latter approach would be then similar to the counterfactual importance sampling inference that we suggest in this paper and that is defined in Section C.3.

3. Used one processor, as to the best of our knowledge, parallelisation is not natively supported for importance sampling in Pyro.

4.

Pass the posterior presentation of the abducted model for intervention step as an EmpiricalMarginal object.

In general, for very large models this might involve extra computational time/network/memory costs.

5.

For "Pyro without guide", if all samples for the abduction step had zero weights, we repeated that step again until at least one sample had non-zero weight.

As shown in Figure 2 , both MultiVerse (MV) and Pyro implementations seem to converge to the ground truth that has been calculated using exact enumeration.

Implementations "MV", "MV optimised" have the same inference schema and hence are expected to converge similarly, inference-wise, with the same number of samples, on average.

"Pyro with guide" is expected to converge, on average, slightly slower because of the double sampling in abduction and prediction steps as discussed in Section A.2; the experimental results confirm that.

Note that "Pyro without guide" converges, inference-wise, much slower than all three other implementations; that is expected because without a proposal (guide), a lot of samples are rejected since the observed variables don't match their observations during the abduction step.

Both MultiVerse implementations are significantly more efficient in terms of speed per sample than Pyro 9 (see Figure 3) .

The "Pyro with guide" takes slightly longer (but not 9.

Note that potential gains in computational efficiency might be explored in the future work by using vectorised sampling in Pyro.

In our experiments, however, we aimed to use both Pyro and MultiVerse similarly to how a basic user of probabilistic programming would use them: i.e. by just writing their model as a Python program without vectorisation, at least for the first iteration over that model.

significantly longer) than "Pyro without guide", although the former is superior in terms of inference efficiency as mentioned above.

Both MultiVerse implementations support parallel importance sampling, and so both of them benefited from the fact that experiments were run on a Amazon Web Services EC2 machine with 16 cores.

At the same time, as mentioned earlier, we could not find a simple way to run importance sampling in parallel in Pyro 10 .

However, if we compute average persample time and take into the account the number of cores (i.e. by dividing Pyro's time by 16), MultiVerse is still faster: based on experiments with 5,000 samples 11 , the average time to run 1 sample for "Pyro w/ guide" is 1.03833 milliseconds (already divided by 16), while for "MultiVerse" it is 0.07431 milliseconds and for "MultiVerse Optimised" it is 0.05692 milliseconds.

For 1,000,000 samples, the average time to run 1 sample for "MV" is 0.06616 milliseconds and for "MV" Optimised' it is 0.04890 milliseconds.

In this Section we provide code snippets in MultiVerse and Pyro for an example model and a counterfactual query for it.

The model is a simple Gaussian model with two latent variables X and Z and one emission variable Y with additional Gaussian noise ??.

It is the same as in Figure 4b .

The counterfactual query is query E(Y | Y = 1.2342, do(Z = ???2.5236)).

That is, this counterfactual query answers the question: what would be the expected value of Y if we observed Y to be equal to 1.2342 and, in that world, we have intervened Z to be equal to ???2.5236?

B.1.

Counterfactual query model example with MultiVerse import t i m e i t 10.

We came to this conclusion based on the available documentation (Uber AI Labs, 2019).

It appears there is a way to run parallel chains using MCMC but not importance sampling, to the best of our understanding.

Note that someone in principal might run importance sampling samplers in parallel, but that requires additional wrappers/helpers to be implemented.

11.

As discussed earlier in the paper, for Pyro this number of samples is used twice, once for abduction step and another for prediction step.

s t a r t = t i m e i t .

d e f a u l t t i m e r ( ) r e s u l t s = r u n i n f e r e n c e ( p r o g r a m w i t h d a t a , NUM SAMPLES) s t o p = t i m e i t .

d e f a u l t t i m e r ( ) print ( ' Time : ' , s t o p ??? s t a r t ) r e s u l t = c a l c u l a t e e x p e c t a t i o n ( r e s u l t s ) print ( ' P r e d i c t i o n : ' , r e s u l t )

The code above in MultiVerse illustrates that MultiVerse allows the user to use "native" probabilistic programming in the sense that it allows him/her to "abstractly" and independently define a model, provide observations and interventions and perform the desired inference query (be it an observational, interventional or counterfactual query).

After all of that is defined, the probabilistic programming implementation will do all inference for the user automatically.

There is no need to define a new model M or store and pass a posterior P (X | Y = e).

This is a general way to perform counterfactual inference in Pyro using its interventional do operator, similar to that suggested in Ness (2019a)

i f var name in i n t e r v e n t i o n : # We must e n s u r e t h a t we don ' t # use i n t e r v e n e d v a r i a b l e s # from i t s p o s t e r i o r : pass e l s e :

p

Note that we have to implement the counterfactual inference using a combination of the different Pyro tools, rather than just rely on the engine and its abstractions to do it.

While that is not necessarily a disadvantage, it does require a user to implement a counterfactual inference query themselves.

It also requires a model to be modified and provided with the posterior values in prediction step.

Further, implementing it as described above in Pyro enforces modularity and the abduction, intervention, and prediction steps are done in isolation; they are prevented from communicating optimisations because they are individually ignorant of the full query.

Also, as mentioned in Section A.2, two sampling steps are required: one for the abduction sampling step, and one for prediction step.

In addition, note that in the example above, we have to operate in a discretised continuous space for the emission variable and its observations.

We achieve this with the function rounder and global variable ROUND DIGIT APPR that defines how many digits should be used when rounding the number.

We have to discretise the space because otherwise it is almost impossible for the sampled value of the emission variable to match its observation.

There is an alternative to the discretisation of the space.

The alternative is to force the explicit noise variable ?? to a value that allows emission variable Y = X + Z + ?? to match its observation.

12 This is very similar to the idea of implemented ObserverableERPs in MultiVerse (see Section E).

An example of such a guide for Pyro for this Gaussian model is provided below:

def my guide (

12.

In this example, ?? is additive.

In reality, it can enter in the Structural Causal Model in any form so long as there is a function observed noise invert function f?? such that f??(Y, X, Z) = ?? (it can also return none (i.e. leading to rejection) or multiple (i.e. additional sampling must be made) appropriate ??-s).

In the case of this example, ?? = f??(Y, X, Z) = Y ??? X ??? Z.

return X, Z , Y e p s i l o n , Y Finally, to enable the guide (and disable the rounding over float values since with the perfect guide we have we don't need it anymore), we need to set configuration variables as follows:

ROUND DIGIT APPR = None GUIDE TO USE = my guide

To compare the different options described above, we computed the effective sample size estimator (Kish, 1965) of the sample weights

for Pyro without and with a guide, as well as for MultiVerse.

We ran 100 runs, each with 1,000 samples, for each option of three.

The results are provided in the Appendix C. Details on importance sampling for counterfactual queries in probabilistic programming

With importance sampling, we can approximate the observational posterior queries P (X | Y ) by generating N samples {s i } from a proposal distribution Q and accumulating the prior, proposal and likelihood probabilities into weights {w i }:

In most cases, X is a vector and it can be sampled forward from the model element by element.

Finally, we calculate statistics of interest using the samples and their weights.

For example, we can compute the expected value of arbitrary function f using the selfnormalised importance sampling:

Similarly, we can do this in probabilistic programming settings, where a probabilistic program is a procedure which represents a generative model.

Each variable x i ??? X and y i ??? Y is represented as a random procedure.

To generate a sample s i of the whole probabilistic program, we evaluate the program forward and in the process: (a) for latent (unobserved) variables, we sample each variable value from its proposal, and incorporate the likelihood of prior and proposal into the weight; (b) for observed variables, we incorporate the likelihood into the weight.

Finally, we can compute the statistics of interest given the samples and weights.

Note that, as mentioned in Section 1.1, the most complex step of counterfactual inference is generally the abduction step, which involves the standard joint posterior inference.

The standard inference techniques for amortised approximate inference (Gu et al., 2015; Germain et al., 2015; Perov et al., 2015; Paige and Wood, 2016a; Perov, 2016; Le et al., 2016; Morris, 2001; Paige and Wood, 2016b; Ritchie et al., 2016; Le et al., 2017) and in particular importance sampling inference (Douglas et al., 2017; Walecki et al., 2018) can be used to facilitate the posterior inference problem.

To compute N samples from counterfactual query et al., 2011) .

Note that in this work in MultiVerse, by default, it is assumed that the structure of the program (including dependencies between variables) and the number of random choices is fixed and therefore the addresses of random variables do not change.

That is because MultiVerse is based on the Python language, and in Python it is hard to track execution traces inside the program itself.

Optionally, in MultiVerse, a user can also specify their own addressing scheme to account for complex control flows of the program execution and complex structure of the program.

1.

Execute that program N more times without any intervention.

As usual with importance sampling in probabilistic programming, we need to sample X from a proposal, incorporate the prior and proposal likelihoods of X into the weights, as well as the likelihoods of observed variables Y .

Note that in this step, weights don't need to be updated because they already represent the posterior space along with the samples.

The "counterfactual's intervention" is intended to operate in exactly that posterior space defined by the samples and the weights 13,14 .

(Let us remark that the "counterfactual's intervention" is neither an inference proposal nor an observation.)

3.

Predict the counterfactual variable(s) of interest.

For example, for expected values it means taking the normalised weighted averages for the variable(s) of interest in the samples, as described in Clause 3 in Section 2.1.

Based on the algorithm above, we can note that we will need 2N + 1 evaluations of the program.

Hence, the memory and computational complexity of importance sampling for counterfactual queries is the same in terms of O-notation as the complexity of importance sampling for "observational" queries as we need to evaluate each program twice.

However, it takes two times more in terms of constant factor.

We also can calculate s i immediately after we calculate s i rather than firstly calculating all {s i } and only then calculating {s i }; that way the memory asymptotic complexity should be the same as for "observational" queries.

As with most sampling methods, we can carry out the counterfactual queries in parallel.

As for the memory optimisations, instead of keeping full samples in the memory, we can discard all but predicted values for K (or, even further, we can only accumulate requested statistics of interest, e.g. a mean or variance).

We employed the optimisations mentioned in Section 2.2 by using "lazy" variable evaluation in "MultiVerse Optimised" in the experiments as discussed in Sections A.2 and illustrated in Section F.1.

Similar to other probabilistic programming language implementations (e.g. similar to optimisations for Church (Perov and Mansinghka, 2012) and Venture (Mansinghka et al., 2014) ), further/alternative optimisations can be made by tracking the dependency of a complex model graph to make sure that for computing {s i } and {s i } we only evaluate/re-evaluate the necessary sub-parts of a program/model.

That is, a more "intelligent" probabilistic programming engine can perform static/dynamic analysis on the computation graph (maybe even potentially a form of "just-in-time" compilation for inference) such that a user even don't need to make the variable evaluation "lazy" themselves in the model but rather the engine can determine itself what parts of the graph needs to be evaluated and when.

Also note that for efficient memory usage, "copy-on-write" strategies (Wikipedia, 2019; Paige and Wood, 2014) can be applied when making intervention/prediction on parts of the sample s i in the process of producing and using for prediction the intervened (i.e. modified) sample s i .

13.

Note that if we were doing a full, exact enumeration over the posterior space (rather than importance sampling), all samples {si} are expected to be unique (i.e. different from each other).

However, after the intervention some samples can become identical like si == sj for i = j; if we want to represent the "counterfactual" probability space, their weights should be summed (like wi + wj) for the final representation of that probability space.

For importance sampling, it does not matter too much because we usually enumerate and calculate statistics of interest over all samples s i like N i=1 s i wi, no matter whether they are unique or not.

14.

Note that if we are to do "counterfactual conditioning", which involves taking into the account additional observations on the counterfactual world after abduction and intervention, and before the prediction step, then the weights should be updated accordingly in that step as well (e.g. they should be set to 0 for the points of the counterfactual space that do not satisfy "counterfactual conditioning" "observations").

In this paper, we consider importance sampling.

To be clear, counterfactual inference can be performed with any appropriate inference scheme, whether it is sampling or optimisationbased, as the abduction step can always be performed separately on its own.

We focus on importance sampling for the first version of MultiVerse.

Importance sampling is a conceptually straightforward 15 way of showing what is perhaps one of the key insights of this paper: designing a probabilistic programming system with counterfactual and causal reasoning in mind enables further optimisations that existing general-purpose probabilistic programming system might not achieve right now (but they can be improved in the future with similar ideas).

Some of these optimisations are described in Section 2.2.

Note that using basic optimisation-based approaches instead, e.g. basic variational inference approaches, might not be efficient because one of the important considerations for counterfactual inference is the necessity to operate on the joint posterior distribution, where the joint is over all hidden variables including noise variables.

Hence, approaches such as mean-field approximations are not particularly suitable, if a good precision of the final counterfactual query is desired, and more sophisticated optimisation-based approaches that preserve the relation between variables in the joint need to be used.

The notation P (K | Y = e; do(D = d)) may seem contradictory at first glance, as it could be that D ??? Y (if we intervene on a variable that we've already observed) or/and K ??? Y (if we are interested in a variable we've already observed).

In reality, if D ??? Y or K ??? Y , they are variables in model M once we have replaced the distribution P (X) by P (X | Y = e).

Hence, no contradiction occurs, but it does highlight the limited power of a standard probabilistic expression to express the intuition of the counterfactual.

This contradiction explains the name "counterfactual" and necessitates the three-part inference procedure.

In this short paper, we employ this "abused" notation to denote (left to right) first the abduction, then the intervention, in service of prediction.

Pearl (2000) offers one notational resolution by denoting K Y as the distribution of K with X already updated and replaced by P (X | Y = e).

Balke and Pearl (1994) offers another resolution via Twin Networks, where all endogenous nodes in the Structural Causal Model are copied but share the same noise values, thus creating endogenous counterfactual counterpart variables that are separate.

Richardson and Robins (2013) offers Single World Intervention Graphs, a resolution similar to Balke and Pearl (1994) that changes the graphical representation of counterfactual and intervened variables.

Work for Counterfactual Inference

Given an intervention mechanism such as exists natively in Pyro (Bingham et al., 2018) (or as can be implemented in Edward as in Tran et al. (2018)), one can write the steps of abduction, intervention and prediction, as it has been independently shown in (Ness, 2019c,b) using sampling 16 .

However, the complex usage of existing methods introduces redundancy, requires model modifications, creates multiple models, and doesn't optimise inference for counterfactuals.

D.2.

Related Language: Omega C probabilistic programming language, syntax and semantics

A new paper 17 by Tavares et al. (2018) proposes Omega C , a causal probabilistic programming language for inference in counterfactual generative models.

This commendable work develops its own syntax and semantics for a new language for counterfactual inference.

For future work, we are interested in: (a) how different approximate counterfactual inference techniques operate and can be optimised in Omega C ; (b) comparing the semantics and syntax of counterfactuals with Omega C , Pyro, MultiVerse and other languages, and identifying ones that are optimal for counterfactuals; and (c) how to extend the insights from Omega C to other probabilistic languages and engines in order to make them more expressible and/or more efficient for counterfactuals.

There is another important set of related work 18 on causal reasoning and probabilistic modelling/programming, specifically in the field of probabilistic logic programming, which "combines logic programming with probability theory as well as algorithms that operate over programs in these formalisms" (Organisers of the 6th Workshop on Probabilistic Logic Programming, 2019).

In particular, Baral and Hunsaker (2007) show how the probabilistic logic programming language P-log (Baral et al., 2009) can be used for causal and counterfactual reasoning e.g. with the help of special variable indexing and related encoding of a model and a query.

In another paper, Vennekens et al. (2009) develop CP-logic, a logical language for representing probabilistic causal laws in the settings of probabilistic logic programming.

16.

The suggested methodology in Ness (2019b) for Pyro explicitly requires resampling from the posterior to calculate counterfactual queries.

For ideas on other approaches to be explored see Section A.2.

17.

The authors of this paper discovered the OmegaC paper a few days before the submission of this paper.

18.

We thank our reviewers for the Second Approximate Inference Symposium (see http:// approximateinference.org/), to which our work has been accepted, for bringing this to our attention.

Appendix E. Model design choices for counterfactual inference in probabilistic programs

In counterfactual settings it is generally expected that all latent variables, including noise variables, should be represented explicitly.

That is, the joint posterior distribution P (X | Y = e) in the "abduction" step must account for the joint of all sources of randomness.

Moreover, it is one of the requirements of structural causal models that all exogenous variables are explicitly represented.

On the other hand, the noise variables in probabilistic programming are often represented implicitly.

Furthermore, often implementations of probabilistic programming systems force a user to represent them only in that implicit way.

For example, an observation with the normal noise is usually absorbed into the likelihood as in the example below:

rather than with explicit representation of the separate, latent noise variable:

There is a good reason, in general, for the implicit representation of noise variables in existing probabilistic programming frameworks (which mostly perform only observational inference) as it allows the OBSERVE statement to "absorb" the observation into the likelihood without sampling the observed variable and without intractable rejection sampling.

To preserve the same benefit in our implementation but also to allow for proper counterfactual inference, we suggest to define versions of "observable probabilistic procedures" (OPP).

For example, a Normal procedure Normal(??, ??) can have a sibling procedure ObservableNormal(??, ??).

An OPP behaves in the similar way as any PP but it has two special considerations:

1.

An OPP must sample its noise explicitly into the program trace as an additional random variable.

2.

If an OPP is being OBSERVEd, then it must have a method of calculating an inverse transformation and observing that noise variable into the appropriate value, as well as calculating the marginalised likelihood of that for the trace weight.

It is also a possible design choice to make all PPs OPPs by default, if desired.

When writing an implementation of an OPP, including its sampling and inverse methods, the similar considerations that are used for writing good proposals (i.e. "guides" in Pyro) can be used:

1.

It is mandatory to ensure that no non-zero parts of the posterior are skipped due to a proposal.

For example, if there are two values of the noise variable that make the emission variable match the observation, both of those values should be sampled with non-zero probability.

2. Note that it is okay if given some specific values of the parents of an OPP, there is no possible value that a noise variable can take to make the OPP match its observed value; in that case, that sample just should be rejected.

For our experiments in Pyro, and generally, it is possible to use Pyro "guide" 19 to force the noise variables, which must be represented explicitly, to their inversed values.

An example of that is provided in Section B.3.

Figure 4 In Figure 4 there are three different but very similar representations/models.

In all of them, there are two latent variables X and Z, which are a priori independent and both follow prior distribution N ormal(0, 1).

The emission variables have extra Gaussian noise N ormal(0, 2).

Figure 4a illustrates a representation of such model with one emission variable; that is common to represent it in such way in probabilistic programming and generally for "observational" inference.

Figure 4b is a representation of the same model but it explicitly represents the exogenous noise variable as an independent variable (highlighted with gradient) such that the emission variable Y is a deterministic function of X, Z and ?? (that way, it is aligned with the general requirements of structural causal models).

That way, variable Y is just a sum of three variables.

For the purpose of "observational" inference both representations have the same joint posterior P (X, Z | Y =??).

However, counterfactual query P (Y | Y =??, do(Z =z)) will be different for Figures 4a and 4b , because in the former case the randomness over the noise has not been recorded in the joint and cannot be used for the counterfactual prediction.

In other words, "by design", in the former case variable Y has to be resampled from its prior given the posterior and intervention over its 19.

"Guide" is Pyro terminology for a model that defines a proposal or variational distribution for more efficient inference.

hyperparameters.

Note that if anyone tries to, "technically", compute the "counterfactual" query P (Y | Y =??, do(Z =z)) given the model representation in Figure 4a , it will be the same as counterfactual query P (Y 2 | Y 1 =??, do(Z =z)) in Figure 4c rather than in Figure 4b as it might had been expected.

We could argue that a choice of a representation and a query should be based on an informed and mindful decision of a user (otherwise, someone accidentally would run a counterfactual query on a model as in Figure 4c if they implement a model as in Figure 4a , although their aim might had been to run a query on a model as in Figure 4b) .

Note a nuance about re-evaluating the stochastic (i.e. non-deterministic) variables that are descendants of any variables in set D. Following the convention suggested by Pearl et al. (e.g. see (Pearl, 2000) ) for Structural Causal Models, to the best of our understanding, it only makes sense for a do operation to entail an intervention on endogenous variables (Pearl, 2000) .

(It is however technically possible to intervene on exogenous variables.)

Following a similar principle in that convention, any variable that is a descendent (direct or indirect) of an intervened variable should also be an endogenous variable.

That is one of the requirements of working with structural causal models.

Note that in general in probabilistic programming a variable that is a descendent of any variable in D can be a random variable with some hyperparameters; in other words such a variable is both an exogenous variable (defined by its own randomness (e.g. noise) that is not expressed as a separate part of the model and hence breaks the assumptions of structural causal models) and an endogenous variable (by virtue of having hyperparameters that depend on other variables in the model).

There are at least three management strategies for this scenario:

1. Be very careful when you are performing the modelling and formulate queries by ensuring you have strict structural causal models and all your queries are appropriate.

the noise is defined by its prior distribution even in the counterfactual's intervention step.

However, it is only a "hack" due to the implementation, and if someone would like to model something like that, it might be the best to introduce proper language syntax constructions for that (e.g. by specifying what variables should be resampled from their prior (or follow a completely different distribution) in the interventional part of a counterfactual query 20 ; or by adjusting the model and doing "partial" counterfactual queries as shown in one of the examples in Figure 4c ) in Section E.3.

Figure 6 illustrates the Observable Normal ERP with its inverse function (for the proposal) ?? := ObservedV alue???f (X1, . . .

, XN ), if there is an observation.

That way, the observation is satisfied.

Figure 6 Figure 7 shows the Observable Bernoulli ERP where its noise variable ?? flips the output of the binary function f (X1, . . .

, XN ) if ?? = 1; however, if ?? = 0, then the emission variable Y just returns the value of function f (X1, . . .

, XN ).

The inverse function works as follows: if the output of function f (. . .) matches the observed value for Y , then the noise variable value is set (proposed with probability 1.0) to value 0 (because no flipping is required); otherwise, the noise variable value is set (proposed with probability 1.0) to value 1 to enforce the flip.

This helps to satisfy the observed value.

E.7.

More sophisticated example of an Observable ERP:

the Observable Noisy OR A popular stochastic "gate"/procedure is a noisy OR procedure (Pearl, 2014) .

One of the definitions of the noisy-OR variable Y given its N parents X1, . . .

, XN is as follows:

20.

One way to think about that is to say that do operator might intervene on a variable to define a new distribution for it rather than just one value.

Figure 8 illustrates a noisy-OR gate.

Each parent j, if active, can switch the noisy-OR variable Y but with only probability 1 ??? ?? j .

In other words, there is a noisy variable associated with each parent: only if both the parent j is T rue and the associated noise variable ?? j is T rue, then the noisy-OR variable Y becomes T rue.

There is also one more, independent, "leak" cause for the noisy-OR variable Y to be switched on: that is if the noise (leak) variable ?? 0 = Bernoulli(1.0 ??? ?? 0 ) is T rue.

By that definition, the noisy-OR variable Y is F alse only if all ?? j , such that j includes j = 0 and j includes all parents that have state T rue, are F alse; that is exactly what the equation above calculates.

For the Observable Noisy-OR procedure, the proposal for noise variables ?? j is more sophisticated.

For example, a proposal might be as follows:

1.

If the observed value is F alse, then: (a) for any Xi that is T rue, the associated noise variables ?? j should be set (i.e. proposed with probability 1.0) to F alse; (b) the noise variable ?? 0 should be set to F alse; (c) all other noise variables ?? j can be sampled from any non-degenerated proposal (i.e. such that both T rue and F alse states have non-zero probability).

2.

If the observed value is T rue, then: (a) all ?? j s.t.

j >= 1 can be sampled from any non-generative proposal; then (b-1) if that is enough, with the states of the parents, to enable Y to be T rue, then variable ?? 0 can be sampled from any non-generative distribution; alternatively, (b-2) if that is not enough, variable ?? 0 must be set to T rue.

Note that the proposal describe just above is one of many possible proposals.

Another proposal might be that if the observed value is T rue, then absolutely all ?? j including j = 0 are sampled from some non-generative proposal, and if that does not result in variable Y being T rue, then that sample is just rejected (in other words, its weight will be 0).

Note that, to the best of our understanding, for counterfactual inference, if an intervention (that happens after abduction) provokes an execution control flow change in a program, then it by default leads to the new control flow sub-trace part to be resampled from its prior.

For Figure 8 example if in probabilistic program 21 (if (Bernoulli 0.1) (Normal 1 1) (Bernoulli 0.5)) the value of the predicate expression (Bernoulli 0.1) has been intervened from 1 to 0, then the value of the alternative if-branch expression Bernoulli(0.5) by default will be resampled from its prior.

That resampling is similar to the resampling of the noise from its prior as discussed in Section E.5 and might have similar implications as discussed in Section E.3, in particular similarly to the model and query for Figure 4a .

In the code above, we define latent variables X1, X2, X3.

We then define emission variables Y 1 and Y 2, which have Gaussian noise.

Because we use ObservableNormalERP, that Gaussian noise will be represented explicitly in the trace and it will be part of the joint model posterior for further counterfactual prediction.

Proposal parameters can be provided to probabilistic procedure object calls.

We then put an observation of Y 1 for value param1, which is passed as an input argument to the probabilistic program.

We also do an intervention for variable X2 to force it to value param2.

At the end, we predict the values of Y 1 and Y 2.

Since in Python it is not easy to automatically track dependencies between objects (which are mutable in general), we have to explicitly specify the dependencies of each probabilistic procedure.

22, 23 Observations, similarly to other probabilistic programming languages, are provided with instruction observe(erp, value).

24 By default, instruction do(erp, value, do_type=DOTYPE_CF) performs a counterfactual intervention and instruction predict(expr, predict_counterfactual=True) performs a counterfactual prediction.

It is also possible to perform a "simple" intervention on the original model by calling instruction do(erp, value, do_type=DOTYPE_IV), which is equivalent to modifying the original model.

It is also possible to perform an "observational" (i.e. not counterfactual) prediction predict(expr, predict_counterfactual=False).

Of course, it is possible to combine all of these instructions in one model/query if desired.

Also note that without any counterfactual interventions do(..., do_type=DOTYPE_CF), the counterfactual prediction is equivalent to the "observational" prediction.

Performing inference is as simple as providing the evidence and the interventions, and calling run_inference method with the selected number of samples: r e s u l t s = r u n i n f e r e n c e ( lambda : m y m o d e l a s p r o b a b i l i s t i c p r o g r a m ( 3 . 5 , 2 . 5 ) , num samples , )

The output of run_inference contains all predictions per each sample and samples' weights.

Those can be used to compute any statistics of interest, e.g. the expected values.

Method run_inference can run inference in parallel using multiple cores.

Each ERP object creation call can be provided with its trace address by a user, e.g. X2 = NormalERP(0, 1, trace_address="X2").

Providing such an address is optional 22.

Note that that is required for counterfactual inference.

It is not a requirement for observational inference.

23.

In the future implementations in other languages, e.g. in the subset of Clojure, the dependencies can be tracked automatically.

24. Note that currently MultiVerse allows observations and interventions only on statically defined random procedures; those procedures can't be determined based on the stochastic trace execution.

It is the future work to explore that.

because by default the engine uses a simple incremental addressing scheme.

However if a probabilistic program has a changing control flow (e.g. there is a statement like if predicate: X2 = X1 + Normal(0, 1, ...); else: ... such that if statement predicate is not deterministic), then the user must use their own probabilistic procedure addressing scheme to ensure consistency for book keeping of probabilistic procedures and their values.

We chose Python for our prototype implementation because Python is a very popular language.

This way, our implementation allows anyone, who knows Python, to run counterfactual queries for any probabilistic program written in Python.

On the other hand, the most of the optimisations that we mentioned in Section 2.2 are harder to implement and, further, fully automatise in Python.

There might be implementations of a similar engine in a restricted subset of Python or in languages like Clojure.

Also, similar ideas can be implemented in existing probabilistic programming languages like Pyro, Anglican (Wood et al., 2014; Tolpin et al., 2015 Tolpin et al., , 2016 , Venture, Church (Goodman et al., 2012) language engines, Gen (Cusumano-Towner and Mansinghka, 2018) , and others (Probabilistic Programming Wiki Authors, 2019).

For "MultiVerse Optimised" experiments as discussed in Section A.2, we redefined the model such that all variables are computed in "a lazy way" (hence computed only if necessary), and we used some methods of MultiVerse engine that allowed us to skip computations unless they are necessary.

That is, instead of computing all variables in the model (as we did for "MV" and for "Pyro") in all steps as follows: we rather compute variables only if required as shown in code snippet below.

That is, if we know that we want to compute the variable of interest VAR TO PREDICT, so we shall compute it.

Our method compute var helper can compute any variable, but first it will compute all its parents (and it will do so recursively for the parents of the parents, etc.).

Also, we wrap calls to compute var helper in our method compute var, in which we rely on MultiVerse's method compute procedure if necessary to check whether we really need to compute a variable, or it has been intervened and we don't need to compute it (and hence its parents as well unless they should be computed for other reasons).

MultiVerse's method compute procedure if necessary(trace, procedure caller) takes a variable trace and if that trace has been intervened, MultiVerse will return a Delta-distribution with the intervened variable's value instead; if it was not intervened, MultiVerse will compute that variable by calling function procedure caller which is provided by us.

The similar logic is used for the variables that needs to be observed (i.e. EVIDENCE variables) or intervened (i.e. INTERVENTION variables).

Finally, we wrap all observations in block IF OBSERVE BLOCK and all interventions in block IF DO BLOCK; that way, MultiVerse will execute those blocks only when required (e.g. we don't need to compute any observation-related parts of the program after we already did abduction step; similarly, we need to record intervention only during the initial run of the program when we record all variables that have been intervened).

@highlight

Probabilistic Programming that Natively Supports Causal, Counterfactual Inference