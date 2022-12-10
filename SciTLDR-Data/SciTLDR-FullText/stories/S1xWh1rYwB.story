Attribution methods provide insights into the decision-making of machine learning models like artificial neural networks.

For a given input sample, they assign a relevance score to each individual input variable, such as the pixels of an image.

In this work we adapt the information bottleneck concept for attribution.

By adding noise to intermediate feature maps we restrict the flow of information and can quantify (in bits) how much information image regions provide.

We compare our method against ten baselines using three different metrics on VGG-16 and ResNet-50, and find that our methods outperform all baselines in five out of six settings.

The method’s information-theoretic foundation provides an absolute frame of reference for attribution values (bits) and a guarantee that regions scored close to zero are not necessary for the network's decision.

@highlight

We apply the informational bottleneck concept to attribution.

@highlight

The paper proposes a novel perturbation-based method for computing attribution/saliency maps for deep neural network based image classifiers, by injecting crafted noise into an early layer of the network.