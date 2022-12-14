Foveation is an important part of human vision, and a number of deep networks have also used foveation.

However, there have been few systematic comparisons between foveating and non-foveating deep networks, and between different variable-resolution downsampling methods.

Here we define several such methods, and compare their performance on ImageNet recognition with a Densenet-121 network.

The best variable-resolution method slightly outperforms uniform downsampling.

Thus in our experiments, foveation does not substantially help or hinder object recognition in deep networks.

To help fill this gap, we compare several foveated downsampling approaches to uniform downsampling 23 in object recognition.

In this context, the different foveated methods perform fairly similarly to each 24 other, and the best performs slightly better than uniform downsampling (top-1 validation accuracy 25 48.95% vs. 47.72%; Table 1 ).

Therefore, foveation does not seem to be important for object 26 recognition (which is unsurprising given the good performance of standard deep networks), but it 27 does not greatly interfere either.

This suggests that foveation could be incorporated into more general 28 vision systems that perform multiple tasks, such as in robots that must recognize objects and also 29 read text in the environment.

We trained deep networks on the ImageNet recognition task, with various kinds of downsampled 33 images as input.

In each case we used a DenseNet-121 [1] network with original hyperparameters Figure 1 : Estimate of retinal ganglion cell (RGC) density as a function of degrees from the fovea.

We use estimates from [3] , which provides data along the nasal-temporal axis.

[4] shows that density is similar in temporal, dorsal, and ventral directions, but higher in the nasal direction.

To calculate radially symmetric mean values, we sum nasal and temporal fits from [3] with weights 0.25 and 0.75 (to account for the fact that nasal density is atypical). and training procedure, including random horizontal flips, batch size etc.

We trained each network 35 for 90 epochs, using SGD (initial learning rate 0.1, reduced by 10x every 30 epochs).

resulting crop went outside the image boundaries, we extrapolated by copying edge pixels.

We sometimes chose multiple points in a single image.

The highest-saliency points are typically close 61 together, and contain similar information.

To avoid selecting multiple similar points, we modified the 62 saliency maps after each selection.

Specifically, we subtracted a square-gaussian function from the 63 saliency map, with a peak equal to the saliency at the chosen point, and a width of 60 pixels.

interesting to explore potential benefits within more general vision systems.

@highlight

We compare object recognition performance on images that are downsampled uniformly and with three different foveation schemes.