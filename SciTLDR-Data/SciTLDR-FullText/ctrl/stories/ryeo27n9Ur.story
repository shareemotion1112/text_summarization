Computational imaging systems jointly design computation and hardware to retrieve information which is not traditionally accessible with standard imaging systems.

Recently, critical aspects such as experimental design and image priors are optimized through deep neural networks formed by the unrolled iterations of classical physics-based reconstructions (termed physics-based networks).

However, for real-world large-scale systems, computing gradients via backpropagation restricts learning due to memory limitations of graphical processing units.

In this work, we propose a memory-efficient learning procedure that exploits the reversibility of the network’s layers to enable data-driven design for large-scale computational imaging.

We demonstrate our methods practicality on two large-scale systems: super-resolution optical microscopy and multi-channel magnetic resonance imaging.

Computational imaging systems (tomographic systems, computational optics, magnetic resonance imaging, to name a few) jointly design software and hardware to retrieve information which is not traditionally accessible on standard imaging systems.

Generally, such systems are characterized by how the information is encoded (forward process) and decoded (inverse problem) from the measurements.

The decoding process is typically iterative in nature, alternating between enforcing data consistency and image prior knowledge.

Recent work has demonstrated the ability to optimize computational imaging systems by unrolling the iterative decoding process to form a differentiable Physicsbased Network (PbN) (1; 2; 3) and then relying on a dataset and training to learn the system's design parameters, e.g. experimental design (3; 4; 5) , image prior model (1; 2; 6; 7).

PbNs are constructed from the operations of reconstruction, e.g. proximal gradient descent algorithm.

By including known structures and quantities, such as the forward model, gradient, and proximal updates, PbNs can be efficiently parameterized by only a few learnable variables, thereby enabling an efficient use of training data (6) while still retaining robustness associated with conventional physics-based inverse problems.

Training PbNs relies on gradient-based updates computed using backpropagation (an implementation of reverse-mode differentiation (8) ).

Most modern imaging systems seek to decode ever-larger growing quantities of information (gigabytes to terabytes) and as this grows, memory required to perform backpropagation is limited by the memory capacity of modern graphical processing units (GPUs).

Methods to save memory during backpropagation (e.g. forward recalculation, reverse recalculation, and checkpointing) trade off spatial and temporal complexity (8) .

For a PbN with N layers, standard backpropagation achieves O(N ) temporal and spatial complexity.

Forward recalculation achieves O(1) memory complexity, but has to recalculate unstored variables forward from the input of the network when needed, yielding O(N 2 ) temporal complexity.

Forward checkpointing smoothly trades off temporal, O(N K), and spatial, O(N/K), complexity by saving variables every K layers and forward-recalculating unstored variables from the closest checkpoint.

Reverse recalculation provides a practical solution to beat the trade off between spatial vs. temporal complexity by calculating unstored variables in reverse from the output of the network, yielding O(N ) temporal and O(1) spatial complexities.

Recently, several reversibility schemes have been proposed for residual networks (9), learning ordinary differential equations (10) , and other specialized network architectures (11; 12) .

In this work, we propose a memory-efficient learning procedure for backpropagation for the PbN formed from proximal gradient descent, thereby enabling learning for many large-scale computational imaging systems.

Based on the concept of invertibility and reverse recalculation, we detail how backpropagation can be performed without the need to store intermediate variables for networks composed of gradient and proximal layers.

We highlight practical restrictions on the layers and introduce a hybrid scheme that combines our reverse recalculation methods with checkpointing to mitigate numerical error accumulation.

Finally, we demonstrate our method's usefulness to learn the design for two practical large-scale computational imaging systems: superresolution optical microscopy (Fourier Ptychography) and multi-channel magnetic resonance imaging.

Computational imaging systems are described by how sought information is encoded to and decoded from a set of measurements.

The encoding of information, x into measurements, y, is given by

where A is the forward model that characterizes the measurement system physics and n is random system noise.

The forward model is a continuous process, but is often approximated by a discrete representation.

The retrieval of information from a set of measurements, i.e. decoding, is commonly structured using an inverse problem formulation,

where D(·) is a data fidelity penalty and P(·) is a prior penalty.

When n is governed by a known noise model, the data consistency penalty can be written as the negative log-likelihood of the appropriate distribution.

When P(·) is a non-smooth prior (e.g. 1 , total variation), proximal gradient descent (PGD) and its accelerated variants are often efficient algorithms to minimize the objective in Eq. 2 and are composed of the following alternating steps:

where α is the gradient step size, ∇ x is the gradient operator, prox P is a proximal function that enforces the prior (13), and x (k) and z (k) are intermediate variables for the k th iteration.

The structure of the PbN is determined by unrolling N iterations of the optimizer to form the N layers of a network (Eq. 3 and Eq. 4 form a single layer).

Specifically, the input to the network is the initialization of the optimization, x (0) , and the output is the resultant, x (N ) .

The learnable parameters are optimized using gradient-based methods.

Common machine learning toolboxes' (e.g. PyTorch, Tensor Flow, Caffe) auto-differentiation functionalities are used to compute gradients for backpropagation.

Auto-differentiation accomplishes this by creating a graph composed of the PbN's operations and storing intermediate variables in memory.

Our main contribution is to improve the spatial complexity of backpropagation for PbNs by treating the larger single graph for auto-differentiation as a series of smaller graphs.

Specifically, consider a PbN, F, composed of a sequence of layers,

where x (k) and x (k+1) are the k th layer input and output, respectively, and θ (k) are its learnable parameters.

When performing reverse-mode differentiation, our method treats a PbN of N layers as N separate smaller graphs, processed one at a time, rather than as a single large graph, thereby saving a factor N in memory.

As outlined in Alg.

1, we first recalculate the current layer's input,

inverse , and then form one of the smaller graphs by recomputing the output of the layer, v (k) , from the recalculated input.

To compute gradients, we then rely on auto-differentiation of each layer's smaller graph to compute the gradient of the loss, L, with respect to

The procedure is repeated for all N layers in reverse order.

Algorithm 1 Memory-efficient learning for physics-based networks 1: procedure MEMORY-EFFICIENT BACKPROPAGA-

for k > 0 do 4:

end for 10:

11: end procedure

In order to perform the reverse-mode differentiation efficiently, we must be able to compute each layer's inverse operation,

inverse .

The remainder of this section overviews the procedures to invert gradient and proximal update layers.

A common interpretation of gradient descent is as a forward Euler discretization of a continuous-time ordinary differential equation.

As a consequence, the inverse of the gradient step layer (Eq. 3) can be viewed as a backward Euler step,

This implicit equation can be solved iteratively via the backward Euler method using the fixed point algorithm (Alg.

2).

Convergence is guaranteed if

where Lip(·) computes the Lipschitz constant of its argument (14) .

In the setting when D(x; y) = Ax − y 2 and A is linear this can be ensured if α <

, where σ max (·) computes the largest singular value of its argument.

Finally, as given by Banach Fixed Point Theorem, the fixed point algorithm (Alg.

2) will have an exponential rate of convergence (14) .

Algorithm 2 Inverse for gradient layer 1: procedure FIXED POINT METHOD(z, T )

2:

x ← z 3:

for t < T do 4:

x ← z + α∇ x D(x; y) 5:

end for

return x 8: end procedure

The proximal update (Eq. 4) is defined by the following optimization problem (13):

For differentiable P(·), the optimum of which is,

In contrast to the gradient update layer, the proximal update layer can be thought of as a backward Euler step (13) .

This allows its inverse to be expressed as a forward Euler step,

when the proximal function is bijective (e.g. prox

).

If the proximal function is not bijective (e.g. prox 1 ) the inversion is not straight forward.

However, in many cases it is possible to substitute it with a bijective function with similar behavior.

Reverse recalculation of the unstored variables is non-exact as the operations to calculate the variables are not identical to forward calculation.

The result is numerical error between the original forward and reverse calculated variables and as more iterations are unrolled, numerical error can accumulate.

To mitigate these effects, some of the intermediate variables can be stored from forward calculation, referred to as checkpoints.

Memory permitting, as many checkpoints should be stored as possible to ensure accuracy while performing reverse recalculation.

While most PbNs cannot afford to store all variables required for reverse-mode differentiation, it is often possible to store a few.

Standard bright-field microscopy offers a versatile system to image in vitro biological samples, however, is restricted to imaging either a large field of view or a high resolution.

Fourier Ptychographic Microscopy (FPM) (15) is a super resolution (SR) method that can create gigapixel-scale images beating this trade off on a standard optical microscope by acquiring a series of measurements (up to hundreds) under various illumination settings on an LED array microscopy (16) and combining them via a phase retrieval based optimization.

The system's dependence on many measurements inhibits its ability to image live fast-moving biology.

Reducing the number of measurements is possible using linear multiplexing (17) and state of the art performance is achieved by forming a PbN and learning its experimental design (4; 3), however, is currently limited in scale due to GPU memory constraints (terabyte-scale memory is required for learning the full measurement system).

With our proposed memory-efficient learning framework, we reduce the required memory to only a few gigabytes, thereby enabling the use of consumer-grade GPU hardware.

To evaluate accuracy we compare standard learning with our proposed memory-efficient learning on a problem that fits in standard GPU memory.

We reproduce results in (4) where the number of measurements are reduced by a factor of 10 using 6.26GB of memory using only 0.627GB and time is only increased by a factor of 2.

To perform memory-efficient learning, we set T = 4 and checkpoint every 10 unrolled iterations.

The testing loss between our method and standard learning are comparable (Fig. 1a) .

In addition, we qualitatively highlight equivalence of the two methods, displaying SR reconstructions with learned design using standard (Fig. 1d ) and memory-efficient (Fig. 1e) methods.

For relative comparison, we display a single low resolution measurement (Fig. 1b) and the ground truth SR reconstruction using all measurements (Fig. 1c) .

MRI is a powerful Fourier-based medical imaging modality that non-invasively captures rich biophysical information without ionizing radiation.

Since MRI acquisition time is directly proportional to the number of acquired measurements, reducing measurements leads to immediate impact on patient throughput and enables capturing fast-changing physiological dynamics.

Multi-channel MRI is the standard of care in clinical systems and uses multiple receive coils distributed around the body to acquire measurements in parallel, thereby reducing the total number of required acquisition frames for decoding (18) .

By additionally modifying the measurement pattern to take advantage of image prior knowledge, e.g. through compressed sensing (19) , it is possible to dramatically reduce scan times.

As with experimental design, PbNs with learned deep image priors have demonstrated state-of-the-art performance for multi-channel MRI (20; 6), but are limited in network size and number of unrolled iterations due to memory required for training.

Our memory-efficient learning reduces memory footprint at training time, thereby enabling learning for larger problems.

To evaluate our proposed memory-efficient learning, we reproduce the results in (6) for the "SD-ET-WD" PbN, which is equivalent to PGD (10 unrolled iterations) where the proximal update is replaced with a learned invertible residual convolutional neural network (RCNN) (21; 11; 9).

We compare training with full backpropagation, requiring 10.77GB of memory and 3:50 hours, versus memory-efficient learning, requiring 2.11GB and 8:25 hours.

We set T = 6 and do not use checkpointing.

As Fig. 2 shows, the training loss is comparable across epochs, and inference results are similar on one image in the training set, with normalized root mean-squared error of 0.03 between conventional and memory-efficient learning.

Discussion: Our proposed memory-efficient learning opens the door to applications that are not otherwise possible to train due to GPU memory constraints, without a large increase in training time.

While we specialized the procedure to PGD networks, similar approaches can be taken to invert other PbNs with more complex subroutines such as solving linear systems of equations.

However, sufficient conditions for invertibility must be met.

This limitation is clear in the case of a gradient descent block with an evolving step size, as the Lipschitz constant may no longer satisfy Eq. 7.

Furthermore, the convergent behavior of optimization to minima makes accurate reverse recalculation of unstored variables severely ill-posed and can cause numerical error accumulation.

Checkpoints can be used to improve the accuracy of reverse recalculated variables, though most PbN are not deep enough for numerical convergence to occur.

In this communication, we presented a practical memory-efficient learning method for large-scale computational imaging problems without dramatically increasing training time.

Using the concept of reversibility, we implemented reverse-mode differentiation with favorable spatial and temporal complexities.

We demonstrated our method on two representative applications: SR optical microscopy and multi-channel MRI.

We expect other computational imaging systems to nicely fall within our framework.

<|TLDR|>

@highlight

We propose a memory-efficient learning procedure that exploits the reversibility of the network’s layers to enable data-driven design for large-scale computational imaging.