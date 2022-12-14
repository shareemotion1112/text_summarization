Deep learning has found numerous applications thanks to its versatility and accuracy on pattern recognition problems such as visual object detection.

Learning and inference in deep neural networks, however, are memory and compute intensive and so improving efficiency is one of the major challenges for frameworks such as PyTorch, Tensorflow, and Caffe.

While the efficiency problem can be partially addressed with specialized hardware and its corresponding proprietary libraries, we believe that neural network acceleration should be transparent to the user and should support all hardware platforms and deep learning libraries.



To this end, we introduce a transparent middleware layer for neural network acceleration.

The system is built around a compiler for deep learning, allowing one to combine device-specific libraries and custom optimizations while supporting numerous hardware devices.

In contrast to other projects, we explicitly target the optimization of both prediction and training of neural networks.

We present the current development status and some preliminary but encouraging results: on a standard x86 server, using CPUs our system achieves a 11.8x speed-up for inference and a 8.0x for batched-prediction (128); on GPUs we achieve a 1.7x and 2.3x speed-up respectively.

The limitations of today's general purpose hardware and the extreme parallelism that neural network 19 processing can exploit has led to a large range of specialized hardware from manufacturers such as 20 NVIDIA [13], Google [7] , ARM BID0 and PowerVR [15] , to name but a few.

Most of these platforms hardware makes it necessary to transform neural network models from one framework to another in 24 order to utilize different hardware architectures.

While standardized formats [6, 11] try to bridge this 25 gap, they cannot guarantee that an exported network behaves identically in all frameworks.

In addition to the hardware support for deep learning frameworks, the usage model itself can differ.

For example, PyTorch is known to be very flexible thanks to its dynamic graph structure, while

TensorFlow uses a static graph that is more restricted, but usually yields better performance.

These 29 differences are dealt with through different strategies.

The big hardware manufacturers such as Intel having to deal with framework or hardware specific issues.

To use our system, the user simply adds a 43 line of code of the form optimizedNN = optimize(myNN).

Finally, our middleware can be easily 44 extended to interface with other AI frameworks and hardware platforms.

In the following we will introduce our optimization cycle, followed by our system architecture, some 46 preliminary results and close with a description of our future development plans.

transformations to merge these nested loops; this step is generic and identical for all target devices.

Next, we use hardware characteristics (e.g., number of cores, SIMD units per core and cache sizes) to 75 generate specific mappings of loops onto compute resources.

FIG2 illustrates a merging operation 76 for a small neural network.

Depending on the hardware, we further exploit device-specific characteristics (shared memory,

Currently, our system can run prediction tasks on both CPUs and GPUs.

To test its performance, we 93 use a server with 2x Intel E5-2637 v4 CPUs, 128GB DDR4, an NVIDIA GTX 1080 Ti card, Debian Overall, we achieve a peak improvement of 11.8x for inference and 8.0x for batched-prediction (128)

on CPUs; and a 1.7x and 2.3x speed-up respectively on GPUs.

Figure 2 further shows that the DFP 102 method can significantly reduce neural network peak memory consumption.

@highlight

We introduce a transparent middleware for neural network acceleration, with own compiler engine, achieving up to 11.8x speed up on CPUs and 2.3x on GPUs.

@highlight

This paper proposes a transparent middleware layer for neural network acceleration and obtains some acceleration results on basic CPU and GPU architectures