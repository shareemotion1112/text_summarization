Much of the focus in the design of deep neural networks had been on improving accuracy, leading to more powerful yet highly complex network architectures that are difficult to deploy in practical scenarios.

As a result, there has been a recent interest in the design of quantitative metrics for evaluating deep neural networks that accounts for more than just model accuracy as the sole indicator of network performance.

In this study, we continue the conversation towards universal metrics for evaluating the performance of deep neural networks for practical on-device edge usage by introducing NetScore, a new metric designed specifically to provide a quantitative assessment of the balance between accuracy, computational complexity, and network architecture complexity of a deep neural network.

In what is one of the largest comparative analysis between deep neural networks in literature, the NetScore metric, the top-1 accuracy metric, and the popular information density metric were compared across a diverse set of 60 different deep convolutional neural networks for image classification on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC 2012) dataset.

The evaluation results across these three metrics for this diverse set of networks are presented in this study to act as a reference guide for practitioners in the field.

There has been a recent urge in both research and industrial interests in deep learning [4] , with deep performance across a wide variety of applications BID22 22, 11] .

However, the practical industrial BID24 deployment bottlenecks associated with the powerful yet highly complex deep neural networks in 22 research literature has become even increasingly visible, and as a result, the design of deep neural 23 networks that strike a strong balance between accuracy and complexity become a very hot area of 24 research focus [18, 14, 34, 33, 26, BID31 36] .

One of the key challenges in designing practical deep 25 neural networks lies in the difficulties with assessing how well a particular network architecture 26 is striking that balance.

One of the most widely cited metrics is the information density metric 27 proposed by BID0 , which attempts to measure the relative amount of accuracy given network size.

However, information density does not account for computational requirements for performing 29 network inference (e.g., MobileNet [14] has more parameters than SqueezeNet [18] but has lower 30 computational requirements for network inference).

Therefore, the exploration and investigation 31 towards universal performance metrics that account for accuracy, architectural complexity, and 32 computational complexity is highly desired as it has the potential to improve network model search 33 and design.

In this study, we introduce NetScore, a new metric designed specifically to provide a 34 quantitative assessment of the balance between accuracy, computational complexity, and network low accuracy remain unusable in practical scenarios, regardless how small or fast the network is.

Furthermore, we set β = 0.5 and γ = 0.5 since, while architectural and computational complexity are the results presented in this study can act as a reference guide for practitioners in the field.

The set of deep convolutional neural networks being evaluated in this study are: AlexNet BID22 , shown in Fig. 1(right) .

Similar to the trend observed in Fig. 1

@highlight

We introduce NetScore, new metric designed to provide a quantitative assessment of the balance between accuracy, computational complexity, and network architecture complexity of a deep neural network.