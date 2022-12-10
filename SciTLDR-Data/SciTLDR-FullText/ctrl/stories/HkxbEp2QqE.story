Recent work on explanation generation for decision-making problems has viewed the explanation process as one of model reconciliation where an AI agent brings the human mental model (of its capabilities, beliefs, and goals) to the same page with regards to a task at hand.

This formulation succinctly captures many possible types of explanations, as well as explicitly addresses the various properties -- e.g. the social aspects, contrastiveness, and selectiveness -- of explanations studied in social sciences among human-human interactions.

However, it turns out that the same process can be hijacked into producing "alternative explanations" -- i.e. explanations that are not true but still satisfy all the properties of a proper explanation.

In previous work, we have looked at how such explanations may be perceived by the human in the loop and alluded to one possible way of generating them.

In this paper, we go into more details of this curious feature of the model reconciliation process and discuss similar implications to the overall notion of explainable decision-making.

One of the root causes 1 for the need of an explanation is that of model differences between the human and the AI agent.

This is because, even if an agent makes the best decisions possible given its model, they may appear to be suboptimal or inexplicable if the human has a different mental model of its capabilities, beliefs and goals.

Thus, it follows that the explanation process, whereby the AI agent justifies its behavior to the human in the loop, is one of model reconciliation.

The Model Reconciliation Process M R , M R h , π takes in the agent model M R , the human mental model of it M R h , and the agent decision π which is optimal in M R as inputs and produces a modelM R h where π is also optimal.• An Explanation is the model differenceM R h ∆M R h .

Thus, by setting the mental modelM R h ← M R h + (through means of some form of interaction / communication), the human cannot come up with a better foil or decisionπ, and hence we say that the original decision π has been explained.

This is referred to as the contrastive property of an explanation.

This property is also the basis of persuasion since the human, given this information, cannot come up with any other alternative to what was done.

So how do we compute this model update?

It turns out that there are several possibilities BID2 ), many of which have the contrastive property.

Minimal Explanations These minimize the size of an explanation and ensure that the human cannot find a better foil using the fewest number of model updates.

These are referred to as minimally complete explanations or MCEs.

DISPLAYFORM0 Monotonic Explanations It turns out that MCEs can become invalid on updating the mental model further, while explaining a later decision.

Minimally monotonic explanations or MMEs, on the other hand, maintain the notion of minimality as before but also ensure that the given decision π never becomes invalid with further explanations.

DISPLAYFORM1

So far, the agent was only explaining its decision (1) with respect to and (2) in terms of what it knows to be true.

Constraint (1) refers to the fact that valid model updates considered during the search for an explanation were always towards the target model M R which is, of course, the agent's belief of the ground truth.

This means that (2) the content of the model update is also always grounded in (the agent's belief of) reality.

In the construction of lies or "alternative facts" to explain, we start stripping away at these two considerations.

There may be many reasons to favor them over traditional explanations: -One could consider cases where team utility is improved because of a lie.

Indeed, authors in BID6 ) discuss how such considerations makes it not only preferable but also necessary that agents learn to deceive.

-A specific case of the above can be seen in terms of difficulty of explanations -a lie can lead to an explanation that is shorter and/or easier to explain... or are more likely to be accepted by the human.

These deal with cases when the agent provides a model update that negates parts of its ground truth -e.g.

saying it does not have a capability it actually has.

This is, in fact, a curious outcome of the non-monotonicity of the model reconciliation process.

Consider the case where the initial estimate of the mental model is empty or φ -i.e.

we start by assuming that the human has no expectations of the agent.

Furthermore, let the minimally complete and minimally monotonic explanations for the model reconciliation process M R , φ, π produce intermediate models M DISPLAYFORM0 which involves the agent stating that its model does not contain parts which it actually does.• A Lie of Omission can emerge from the model reconciliation process φ, M However, they happen to be the easiest to compute due to the fact that they are constrained by a target model (which is empty) and do not requite any "imagination".

More on this when we discuss lies of commission.

In lies of omission, the agent omitted constraints in its model that actually existed.

It did not make up new things (and having the target model as M R in the original model reconciliation process prevented that).

In lies of commission, the agent can make up new aspects of its decision-making model that do not belong to its ground truth model.

Let M be the space of models induced by M R and M R h .

3 Then:• A Lie of Commission can emerge from the model reconciliation process M, M R h , π where M ∈ M. We have dropped the target here from being M R to any possible model.

Immediately, the computational problem arises: the space of models was rather large to begin with -O(2 DISPLAYFORM0 -and now we have an exponentially larger number of models to search through without a target -O(2 DISPLAYFORM1 .

This should be expected: after all, even for humans, computationally it is always much easier to tell the truth rather than think of possible lies.

2 As per the definition of an MME, if the mental model is between the MME and the agent model, then there is no need for an explanation since optimal decisions in those models are equivalent.3 This consists of the union of the power sets of the set representation of models M R and M R h following BID2 .

4 "A lie is when you say something happened with didn't happen.

But there is only ever one thing which happened at a particular time and a particular place.

And there are an infinite number of things which didn't happen at that time and that place.

And if I think about something which didn't happen I start thinking about all the other things which didn't happen." BID5 The problem becomes more interesting when the agent can expand on M to conceive of lies that are beyond its current understanding of reality.

This requires a certain amount of imagination from the agent: -One simple way to expand the space of models is by defining a theory of what makes a sound model and how models can evolve.

Authors in (Bryce, Benton, and Boldt 2016) explore one such technique in a different context of tracking a drifting model of the user.-A more interesting technique of model expansion can borrow from work in the space of storytelling BID8 in imagining lies that are likely to be believable -here, the system extends a given model of decisionmaking by using word similarities and antonyms from a knowledge base like WordNet to think about actions that are not defined in the model but may exist, or are at least plausible, in the real world.

Originally built for the purpose of generating new storylines, one could imagine similar techniques being used to come up with false explanations derived from the current model.

In all the discussion so far, the objective has been still the same as the original model reconciliation work: the agent is trying to justify the optimality of its decision, i.e. persuade the human that this was the best possible decision that could have been made.

At this point, it is easy to see that in general, the starting point of this process may not require a decision that is optimal in the robot model at all, as long as the intermediate model preserves its optimality so that the human in the loop cannot come up with a better foil (or negates the specific set of foils given by the human (Sreedharan, Srivastava, and Kambhampati 2018)).The Persuasion Process M R h , π takes in the human mental model M R h of a decision-making task and the agent's decision π and produces a modelM R h where π is optimal.

Note here that, in contrast to the original model reconciliation setup, we have dropped the agent's ground truth model from the definition, as well as the requirement that the agent's decision be optimal in that model to begin with.

The content ofM R h is left to the agent's imaginationfor the original model reconciliation work for explanations BID2 ) these updates were consistent with the agent model.

In this paper, we saw what happens to the reconciliation process when that constraint is relaxed.

So far we have only considered explicit cases of deception.

Interestingly, existing approaches in model reconciliation already tend to allow for misconceptions to be ignored if not actively induced by the agent.

In trying to minimize the size of an explanation, the agent omits a lot of details of the agent model that were actually used in coming up with the decision, as well as decided to not rectify known misconceptions of the human, since the optimality of the decision holds irrespective of them being there.

Such omissions can have impact on the the human going forward, who will base their decisions on M R h which is only partially true.

5 Humans, in fact, make such decision all the time while explaining -this is known as the selective property of an explanation BID8 .Furthermore, MCEs and MMEs are not unique.

Even without consideration of omitted facts about the model, the agent must consider the relative importance BID11 ) of model differences to the human in the loop.

Is it okay then to exploit these preferences towards generating "preferred explanations" even if that means departing from a more valid explanation?It is unclear what the prescribed behavior of the agent should be in these cases.

Indeed, a variant of model reconciliation -contingent explanations -that engages the human in dialogue to better figure out the mental model can explicitly figure out gaps in the human knowledge and exploit that to shorten explanations.

On the face of it, this sounds worrisome, though perfectly legitimate in so far as preserving the various well-studied properties of explanations go.

In this paper we have only considered cases of deception where the agent explicitly changes the mental model.

Interestingly, in this multi-model setup, it is also possible to deceive the human without any model updates at all.

A parallel idea, in dealing with model differences, is that of explicability ) - DISPLAYFORM0 Thus, the agent, instead of trying to explain its decision, sacrifices optimality and instead conforms to the human expectation (if possible).

Indeed, the notion of explanations and explicability can be considered under the same framework where the agent gets to trade off the cost (e.g. length) of an explanation versus the cost of being explicable (i.e. departure from optimality).

Unfortunately, this criterion only ensures that the decision the agent makes is equivalent to one that the human would expect though not necessarily for the same reasons.

For example, it is quite conceivable that the agent's goal is different to what the human expects though the optimal decisions for both the goals coincide.

Such decisions may be explicable for the wrong reasons, even though the current formulation allows it.

Similar notions can apply to other forms of explainable behavior as well, as we discuss in .

Indeed, authors in BID7 explore how an unified framework of decision-making can produce both legible as well as obfuscated behavior.

<|TLDR|>

@highlight

Model Reconciliation is an established framework for plan explanations, but can be easily hijacked to produce lies.