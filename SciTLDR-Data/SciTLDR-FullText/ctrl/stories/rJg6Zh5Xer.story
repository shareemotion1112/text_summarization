Recent progress in hardware and methodology for training neural networks has ushered in a new generation of large networks trained on abundant data.

These models have obtained notable gains in accuracy across many NLP tasks.

However, these accuracy improvements depend on the availability of exceptionally large computational resources that necessitate similarly substantial energy consumption.

As a result these models are costly to train and develop, both financially, due to the cost of hardware and electricity or cloud compute time, and environmentally, due to the carbon footprint required to fuel modern tensor processing hardware.

In this paper we bring this issue to the attention of NLP researchers by quantifying the approximate financial and environmental costs of training a variety of recently successful neural network models for NLP.

Based on these findings, we propose actionable recommendations to reduce costs and improve equity in NLP research and practice.

Advances in techniques and hardware for training deep neural networks have recently enabled impressive accuracy improvements across many fundamental NLP tasks BID1 BID12 BID8 BID18 , with the most computationally-hungry models obtaining the highest scores BID13 BID7 BID14 BID16 .

As a result, training a state-of-the-art model now requires substantial computational resources which demand considerable energy, along with the associated financial and environmental costs.

Research and development of new models multiplies these costs by thousands of times by requiring retraining to experiment with model architectures and hyperparameters.

Whereas a decade ago most NLP models could be trained and developed on a commodity laptop or server, many now require multiple instances of specialized hardware such as GPUs or TPUs, therefore limiting access to these highly accurate models on the basis of finances.

Even when these expensive computational resources are available, model training also incurs a substantial cost to the environment due to the energy required to power this hardware for weeks or months at a time.

Though some of this energy may come from renewable or carbon credit-offset resources, the high energy demands of these models are still a concern since (1) energy is not currently derived from carbon-neural sources in many locations, and (2) when renewable energy is available, it is still limited to the equipment we have to produce and store it, and energy spent training a neural network might better be allocated to heating a family's home.

It is estimated that we must cut carbon emissions by half over the next decade to deter escalating rates of natural disaster, and based on the estimated CO 2 emissions listed in TAB1 , model training and development likely make up a substantial portion of the greenhouse gas emissions attributed to many NLP researchers.

To heighten the awareness of the NLP community to this issue and promote mindful practice and policy, we characterize the dollar cost and carbon emissions that result from training the neural networks at the core of many state-of-the-art NLP models.

We do this by estimating the kilowatts of energy required to train a variety of popular off-the-shelf NLP models, which can be converted to approximate carbon emissions and electricity costs.

To estimate the even greater resources required to transfer an existing model to a new task or develop new models, we perform a case study of the full computational resources required for the development and tuning of a recent state-of-the-art NLP pipeline BID17 .

We conclude with recommendations to the community based on our findings, namely: (1) Time to retrain and sensitivity to hyperparameters should be reported for NLP machine learning models; (2) academic researchers need equitable access to computational resources; and (3) researchers should prioritize developing efficient models and hardware.

To quantify the computational and environmental cost of training deep neural network models for NLP, we perform an analysis of the energy required to train a variety of popular offthe-shelf NLP models, as well as a case study of the complete sum of resources required to develop LISA BID17 , a state-of-the-art NLP model from EMNLP 2018, including all tuning and experimentation.

We measure energy use as follows.

We train the models described in §2.1 using the default settings provided, and sample GPU and CPU power consumption during training.

Each model was trained for a maximum of 1 day.

We train all models on a single NVIDIA Titan X GPU, with the exception of ELMo which was trained on 3 NVIDIA GTX 1080 Ti GPUs.

While training, we repeatedly query the NVIDIA System Management Interface 2 to sample the GPU power consumption and report the average over all samples.

We estimate the total time expected for models to train to completion using training times and hardware reported in the original papers.

We then calculate the power consumption in kilowatt-hours (kWh) as follows.

Let p c be the average power draw (in watts) from all CPU sockets during training, let p r be the average power draw from all DRAM (main memory) sockets, let p g be the average power draw of a GPU during training, and let g be the number of GPUs used to train.

We estimate total power consumption as combined GPU, CPU and DRAM consumption, then multiply this by Power Usage Effectiveness (PUE), which accounts for the additional energy required to support the compute infrastructure (mainly cooling).

We use a PUE coefficient of 1.58, the 2018 global average for data centers BID0 .

Then the total power p t required at a given instance during training is given by: DISPLAYFORM0 The U.S. Environmental Protection Agency (EPA) provides average CO 2 produced (in pounds per kilowatt-hour) for power consumed in the U.S. (EPA, 2018), which we use to convert power to estimated CO 2 emissions: DISPLAYFORM1 This conversion takes into account the relative proportions of different energy sources (primarily natural gas, coal, nuclear and renewable) consumed to produce energy in the United States.

TAB3 lists the relative energy sources for China, Germany and the United States compared to the top three cloud service providers.

The U.S. breakdown of energy is comparable to that of the most popular cloud compute service, Amazon Web Services, so we believe this conversion to provide a reasonable estimate of CO 2 emissions per kilowatt hour of compute energy used.

We analyze four models, the computational requirements of which we describe below.

All models have code freely available online, which we used out-of-the-box.

For more details on the models themselves, please refer to the original papers.

Transformer.

The Transformer model BID18 is an encoder-decoder architecture primarily recognized for efficient and accurate machine translation.

The encoder and decoder each consist of 6 stacked layers of multi-head selfattention.

BID18 report that the Transformer base model (65M parameters) was trained on 8 NVIDIA P100 GPUs for 12 hours, and the Transformer big model (213M parameters) was trained for 3.5 days (84 hours; 300k steps).

This model is also the basis for recent work on neural architecture search (NAS) for machine translation and language modeling BID16 , and the NLP pipeline that we study in more detail in §4.2 BID17 .

BID16 report that their full architecture search ran for a total of 979M training steps, and that their base model requires 10 hours to train for 300k steps on one TPUv2 core.

This equates to 32,623 hours of TPU or 274,120 hours on 8 P100 GPUs.

ELMo.

The ELMo model BID13 is based on stacked LSTMs and provides rich word representations in context by pre-training on a large amount of data using a language modeling objective.

Replacing context-independent pretrained word embeddings with ELMo has been shown to increase performance on downstream tasks such as named entity recognition, semantic role labeling, and coreference.

BID13 report that ELMo was trained on 3 NVIDIA GTX 1080 GPUs for 2 weeks (336 hours).BERT.

The BERT model BID7 provides a Transformer-based architecture for building contextual representations similar to ELMo, but trained with a different language modeling objective.

BERT substantially improves accuracy on tasks requiring sentence-level representations such as question answering and natural language inference.

BID7 report that the BERT base model (110M parameters) was trained on 16 TPU chips for 4 days (96 hours).

NVIDIA reports that they can train a BERT model in 3.3 days (79.2 hours) using 4 DGX-2H servers, totaling 64 Tesla V100 GPUs BID10 .

GPT-2.

This model is the latest edition of OpenAI's GPT general-purpose token encoder, also based on Transformer-style self-attention and trained with a language modeling objective (Radford et al., 2019).

By training a very large model on massive data, BID14 show high zero-shot performance on question answering and language modeling benchmarks.

The large model described in BID14 has 1542M parameters and is reported to require 1 week (168 hours) of training on 32 TPU v3 chips.

6

There is some precedent for work characterizing the computational requirements of training and inference in modern neural network architectures in the computer vision community.

BID11 present a detailed study of the energy use required for training and inference in popular convolutional models for image classification in computer vision, including fine-grained analysis comparing different neural network layer types.

BID5 assess image classification model accuracy as a function of model size and gigaflops required during inference.

They also measure average power draw required during inference on GPUs as a function of batch size.

Neither work analyzes the recurrent and self-attention models that have become commonplace in NLP, nor do they extrapolate power to estimates of carbon and dollar cost of training.

Analysis of hyperparameter tuning has been performed in the context of improved algorithms for hyperparameter search BID3 BID2 BID15 .

To our knowledge there exists to date no analysis of the computation required for R&D and hyperparameter tuning of neural network models in NLP.

6 Via the authors on Reddit.

7 GPU lower bound computed using pre-emptible P100/V100 U.S. resources priced at $0.43-$0.74/hr, upper bound uses on-demand U.S. resources priced at $1.46-$2.48/hr.

We similarly use pre-emptible ($1.46/hr-$2.40/hr) and on-demand ($4.50/hr-$8/hr) pricing as lower and upper bounds for TPU v2/3; cheaper bulk contracts are available.

Table 3 : Estimated cost of training a model in terms of CO 2 emissions (lbs) and cloud compute cost (USD).

7 Power and carbon footprint are omitted for TPUs due to lack of public information on power draw for this hardware.

Table 3 lists CO 2 emissions and estimated cost of training the models described in §2.1.

Of note is that TPUs are more cost-efficient than GPUs on workloads that make sense for that hardware (e.g. BERT).

We also see that models emit substantial carbon emissions; training BERT on GPU is roughly equivalent to a trans-American flight.

BID16 report that NAS achieves a new stateof-the-art BLEU score of 29.7 for English to German machine translation, an increase of just 0.1 BLEU at the cost of at least $150k in on-demand compute time and non-trivial carbon emissions.

To quantify the computational requirements of R&D for a new model we study the logs of all training required to develop LinguisticallyInformed Self-Attention BID17 ), a multi-task model that performs part-of-speech tagging, labeled dependency parsing, predicate detection and semantic role labeling.

This model makes for an interesting case study as a representative NLP pipeline and as a Best Long Paper at EMNLP.Model training associated with the project spanned a period of 172 days (approx.

6 months).

During that time 123 small hyperparameter grid searches were performed, resulting in 4789 jobs in total.

Jobs varied in length ranging from a minimum of 3 minutes, indicating a crash, to a maximum of 9 days, with an average job length of 52 hours.

All training was done on a combination of NVIDIA Titan X (72%) and M40 (28%) GPUs.

8 The sum GPU time required for the project totaled 9998 days (27 years).

This averages to 8 We approximate cloud compute cost using P100 pricing.

about 60 GPUs running constantly throughout the 6 month duration of the project.

TAB6 lists upper and lower bounds of the estimated cost in terms of Google Cloud compute and raw electricity required to develop and deploy this model.

9 We see that while training a single model is relatively inexpensive, the cost of tuning a model for a new dataset, which we estimate here to require 24 jobs, or performing the full R&D required to develop this model, quickly becomes extremely expensive.are compatible with their setting.

More explicit characterization of tuning time could also reveal inconsistencies in time spent tuning baseline models compared to proposed contributions.

Realizing this will require: (1) a standard, hardwareindependent measurement of training time, such as gigaflops required to convergence, and (2) a standard measurement of model sensitivity to data and hyperparameters, such as variance with respect to hyperparameters searched.

Academic researchers need equitable access to computation resources.

Recent advances in available compute come at a high price not attainable to all who desire access.

Most of the models studied in this paper were developed outside academia; recent improvements in state-of-the-art accuracy are possible thanks to industry access to large-scale compute.

Limiting this style of research to industry labs hurts the NLP research community in many ways.

First, it stifles creativity.

Researchers with good ideas but without access to large-scale compute will simply not be able to execute their ideas, instead constrained to focus on different problems.

Second, it prohibits certain types of research on the basis of access to financial resources.

This even more deeply promotes the already problematic "rich get richer" cycle of research funding, where groups that are already successful and thus well-funded tend to receive more funding due to their existing accomplishments.

Third, the prohibitive start-up cost of building in-house resources forces resource-poor groups to rely on cloud compute services such as AWS, Google Cloud and Microsoft Azure.

While these services provide valuable, flexible, and often relatively environmentally friendly compute resources, it is more cost effective for academic researchers, who often work for nonprofit educational institutions and whose research is funded by government entities, to pool resources to build shared compute centers at the level of funding agencies, such as the U.S. National Science Foundation.

For example, an off-the-shelf GPU server containing 8 NVIDIA 1080 Ti GPUs and supporting hardware can be purchased for approximately $20,000 USD.

At that cost, the hardware required to develop the model in our case study (approximately 58 GPUs for 172 days) would cost $145,000 USD plus electricity, about half the estimated cost to use on-demand cloud GPUs.

Unlike money spent on cloud compute, however, that invested in centralized resources would continue to pay off as resources are shared across many projects.

A government-funded academic compute cloud would provide equitable access to all researchers.

Researchers should prioritize computationally efficient hardware and algorithms.

We recommend a concerted effort by industry and academia to promote research of more computationally efficient algorithms, as well as hardware that requires less energy.

An effort can also be made in terms of software.

There is already a precedent for NLP software packages prioritizing efficient models.

An additional avenue through which NLP and machine learning software developers could aid in reducing the energy associated with model tuning is by providing easyto-use APIs implementing more efficient alternatives to brute-force grid search for hyperparameter tuning, e.g. random or Bayesian hyperparameter search techniques BID3 BID2 BID15 .

While software packages implementing these techniques do exist, 10 they are rarely employed in practice for tuning NLP models.

This is likely because their interoperability with popular deep learning frameworks such as PyTorch and TensorFlow is not optimized, i.e. there are not simple examples of how to tune TensorFlow Estimators using Bayesian search.

Integrating these tools into the workflows with which NLP researchers and practitioners are already familiar could have notable impact on the cost of developing and tuning in NLP.

<|TLDR|>

@highlight

We quantify the energy cost in terms of money (cloud credits) and carbon footprint of training recently successful neural network models for NLP. Costs are high.