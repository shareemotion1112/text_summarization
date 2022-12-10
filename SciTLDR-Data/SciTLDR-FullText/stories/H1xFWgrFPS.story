As machine learning methods see greater adoption and implementation in high stakes applications such as medical image diagnosis, the need for model interpretability and explanation has become more critical.

Classical approaches that assess feature importance (eg saliency maps) do not explain how and why a particular region of an image is relevant to the prediction.

We propose a method that explains the outcome of a classification black-box by gradually exaggerating the semantic effect of a given class.

Given a query input to a classifier, our method produces a progressive set of plausible variations of that query, which gradually change the posterior probability from its original class to its negation.

These counter-factually generated samples preserve features unrelated to the classification decision, such that a user can employ our method as a ``tuning knob'' to traverse a data manifold while crossing the decision boundary.

Our method is model agnostic and only requires the output value and gradient of the predictor with respect to its input.

@highlight

A method to explain a classifier, by generating visual perturbation of an image by exaggerating  or diminishing the semantic features that the classifier associates with a target label.

@highlight

A model that when given a query input to a black-box, aims to explain the outcome by providing plausible and progressive variations to the query that can result in a change to the output.

@highlight

A method for explaining the output of black box classification of images, that generates gradual perturbation of outputs in response to gradually perturbed input queries.