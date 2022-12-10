Many imaging tasks require global information about all pixels in an image.

Conventional bottom-up classification networks globalize information by decreasing resolution; features are pooled and down-sampled into a single output.

But for semantic segmentation and object detection tasks, a network must provide higher-resolution pixel-level outputs.

To globalize information while preserving resolution, many researchers propose the inclusion of sophisticated auxiliary blocks, but these come at the cost of a considerable increase in network size and computational cost.

This paper proposes stacked u-nets (SUNets), which iteratively combine features from different resolution scales while maintaining resolution.

SUNets leverage the information globalization power of u-nets in a deeper net- work architectures that is capable of handling the complexity of natural images.

SUNets perform extremely well on semantic segmentation tasks using a small number of parameters.

@highlight

Presents new architecture which leverages information globalization power of u-nets in a deeper networks and performs well across tasks without any bells and whistles.

@highlight

A network architecture for semantic image segmentation, based on composing a stack of basic U-Net architectures, that reduces the number of parameters and improves results.

@highlight

This proposes a stacked U-Net architecture for image segmentation.