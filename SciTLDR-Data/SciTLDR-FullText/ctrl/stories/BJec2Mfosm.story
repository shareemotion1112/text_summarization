Although deep convolutional networks have achieved improved performance in many natural language tasks, they have been treated as black boxes because they are difficult to interpret.

Especially, little is known about how they represent language in their intermediate layers.

In an attempt to understand the representations of deep convolutional networks trained on language tasks, we show that individual units are selectively responsive to specific morphemes, words, and phrases, rather than responding to arbitrary and uninterpretable patterns.

In order to quantitatively analyze such intriguing phenomenon, we propose a concept alignment method based on how units respond to replicated text.

We conduct analyses with different architectures on multiple datasets for classification and translation tasks and provide new insights into how deep models understand natural language.

The contributions of this work can be summarized as follows:

• We show that the units of deep CNNs learned in NLP tasks could act as a natural language 49 concept detector.

Without any additional labeled data or re-training process, we can discover, Finally, for each unit u, we define a set of its aligned concepts C * u = {c * selectivity of a unit u, to which a set of concepts C * u that our alignment method detects, as follows: DISPLAYFORM0 where S denotes all sentences in training set, and µ + = 1 |S+| s∈S+ a u (s) is the average value of 133 unit activation when forwarding a set of sentences S + , which is defined as one of the following:

• replicate: S + contains the sentences created by replicating each concept in C * u .

As before, 135 the sentence length is set as the average length of all training sentences for fair comparison.

• inclusion: S + contains the training sentences that include at least one concept in C * u .

• random: S + contains randomly sampled sentences from the training data.

Intuitively, if unit u's activation is highly sensitive to C * u (i.e. those found by our alignment method) 141 and if it is not to other factors, then Sel u gets large; otherwise, Sel u is near 0.142 FIG0 shows the mean and variance of selectivity values for all units learned in each dataset for 143 the three S + categories.

Consistent with our intuition, in all datasets, the mean selectivity of the 144 replicate set is the highest with a significant margin, that of inclusion set is the runner-up, and that of 145 the random set is the lowest.

These results support our claim that our method is successful to align 146 concepts in which the unit responds selectively.

show that individual units are selectively responsive to specific natural language concepts.

More interestingly, we discover that many units could capture specific meanings or syntactic roles • This is the first time, it is the first exercise.• These, however, are just the first steps.• This ought to be the first step forward.• That will be just the first step.• We can already see the first results.

Layer 14, Unit 360: the first step, first, be the first step• That is not the subject of this communication.• That is the purpose of this communication.• I would like to ask the Commissioner for a reply.• This is impossible without increasing efficiency.•

Will we be able to achieve this, Commissioner?Layer 06, Unit 396: of this communication, will, communication• qualcomm has inked a licensing agreement with Microsoft • peoplesoft wants its customers to get aggressive with software upgrades to increase efficiency.• provide its customers with access to wi-fi hotspots around the world.• realnetworks altered the software for market-leading ipod.• apple lost one war to microsoft by not licensing its mac…

•

They know that and we know that.•

I am sure you will understand.•

I am sure you will do this.•

I am confident that we will find a solution.

• one of the best restaurants and the best meat in town… • friendly service sweet tomatoes is a great place.• the margaritas are fantastic, the service was great… • love love love this place!...

• Several units in the CNN learned on NLP tasks respond selectively to specific natural language concepts, rather than getting activated in an uninterpretable way.

This means that 179 these units can serve as detectors for specific natural language concepts.

•

<|TLDR|>

@highlight

We show that individual units in CNN representations learned in NLP tasks are selectively responsive to specific natural language concepts.

@highlight

Uses grammatical units of natural language that preserve meanings to show that the units of deep CNNs learned in NLP tasks could act as a natural language concept detector.