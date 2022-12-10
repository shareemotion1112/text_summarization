Over the last few years exciting work in deep generative models has produced models able to suggest new organic molecules by generating strings, trees, and graphs representing their structure.

While such models are able to generate molecules with desirable properties, their utility in practice is limited due to the difficulty in knowing how to synthesize these molecules.

We therefore propose a new molecule generation model, mirroring a more realistic real-world process, where reactants are selected and combined to form more complex molecules.

More specifically, our generative model proposes a bag of initial reactants (selected from a pool of commercially-available molecules) and uses a reaction model to predict how they react together to generate new molecules.

Modeling the entire process of constructing a molecule during generation offers a number of advantages.

First, we show that such a model has the ability to generate a wide, diverse set of valid and unique molecules due to the useful inductive biases of modeling reactions.

Second, modeling synthesis routes rather than final molecules offers practical advantages to chemists who are not only interested in new molecules but also suggestions on stable and safe synthetic routes.

Third, we demonstrate the capabilities of our model to also solve one-step retrosynthesis problems, predicting a set of reactants that can produce a target product.

@highlight

A deep generative model for organic molecules that first generates reactant building blocks before combining these using a reaction predictor.

@highlight

A molecular generative model that generates molecules via a two-step process that provides synthesis routes of the generated molecules, allowing users to examine the synthetic accessibility of generated compounds.