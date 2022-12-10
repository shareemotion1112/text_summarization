We study the problem of learning and optimizing through physical simulations via differentiable programming.

We present DiffSim, a new differentiable programming language tailored for building high-performance differentiable physical simulations.

We demonstrate the performance and productivity of our language in gradient-based learning and optimization tasks on 10 different physical simulators.

For example, a differentiable elastic object simulator written in our language is 4.6x faster than the hand-engineered CUDA version yet runs as fast, and is 188x faster than TensorFlow.

Using our differentiable programs, neural network controllers are typically optimized within only tens of iterations.

Finally, we share the lessons learned from our experience developing these simulators, that is, differentiating physical simulators does not always yield useful gradients of the physical system being simulated.

We systematically study the underlying reasons and propose solutions to improve gradient quality.

Figure 1: Left: Our language allows us to seamlessly integrate a neural network (NN) controller and a physical simulation module, and update the weights of the controller or the initial state parameterization (blue).

Our simulations typically have 512 ∼ 2048 time steps, and each time step has up to one thousand parallel operations.

Right: 10 differentiable simulators built with DiffSim.

Differentiable physical simulators are effective components in machine learning systems.

For example, de Avila Belbute- Peres et al. (2018a) and Hu et al. (2019b) have shown that controller optimization with differentiable simulators converges one to four orders of magnitude faster than model-free reinforcement learning algorithms.

The presence of differentiable physical simulators in the inner loop of these applications makes their performance vitally important.

Unfortunately, using existing tools it is difficult to implement these simulators with high performance.

We present DiffSim, a new differentiable programming language for high performance physical simulations on both CPU and GPU.

It is based on the Taichi programming language (Hu et al., 2019a) .

The DiffSim automatic differentiation system is designed to suit key language features required by physical simulation, yet often missing in existing differentiable programming tools, as detailed below: Megakernels Our language uses a "megakernel" approach, allowing the programmer to naturally fuse multiple stages of computation into a single kernel, which is later differentiated using source code transformations and just-in-time compilation.

Compared to the linear algebra operators in TensorFlow (Abadi et al., 2016) and PyTorch (Paszke et al., 2017) , DiffSim kernels have higher arithmetic intensity and are therefore more efficient for physical simulation tasks.

Imperative Parallel Programming In contrast to functional array programming languages that are popular in modern deep learning (Bergstra et al., 2010; Abadi et al., 2016; Li et al., 2018b) , most traditional physical simulation programs are written in imperative languages such as Fortran and C++.

DiffSim likewise adopts an imperative approach.

The language provides parallel loops and control flows (such as "if" statements), which are widely used constructs in physical simulations: they simplify common tasks such as handling collisions, evaluating boundary conditions, and building iterative solvers.

Using an imperative style makes it easier to port existing physical simulation code to DiffSim.

Flexible Indexing Existing parallel differentiable programming systems provide element-wise operations on arrays of the same shape, e.g. can only be expressed with unintuitive scatter/gather operations in these existing systems, which are not only inefficient but also hard to develop and maintain.

On the other hand, in DiffSim, the programmer directly manipulates array elements via arbitrary indexing, thus allowing partial updates of global arrays and making these common simulation patterns naturally expressible.

The explicit indexing syntax also makes it easy for the compiler to perform access optimizations (Hu et al., 2019a) .

The three requirements motivated us to design a tailored two-scale automatic differentiation system, which makes DiffSim especially suitable for developing complex and high-performance differentiable physical simulators, possibly with neural network controllers ( Fig. 1, left) .

Using our language, we are able to quickly implement and automatically differentiate 10 physical simulators 1 , covering rigid bodies, deformable objects, and fluids ( Fig. 1, right) .

A comprehensive comparison between DiffSim and other differentiable programming tools is in Appendix A.

DiffSim is based on the Taichi programming language (Hu et al., 2019a) .

Taichi is an imperative programming language embedded in C++14.

It delivers both high performance and high productivity on modern hardware.

The key design that distinguishes Taichi from other imperative programming languages such as C++/CUDA is the decoupling of computation from data structures.

This allows programmers to easily switch between different data layouts and access data structures with indices (i.e. x[i, j, k]), as if they are normal dense arrays, regardless of the underlying layout.

The Taichi compiler then takes both the data structure and algorithm information to apply performance optimizations.

Taichi provides "parallel-for" loops as a first-class construct.

These designs make Taichi especially suitable for writing high-performance physical simulators.

For more details, readers are referred to Hu et al. (2019a) .

The DiffSim language frontend is embedded in Python, and a Python AST transformer compiles DiffSim code to Taichi intermediate representation (IR) .

Unlike Python, the DiffSim language is compiled, statically-typed, parallel, and differentiable.

We extend the Taichi compiler to further compile and automatically differentiate the generated Taichi IR into forward and backward executables.

We demonstrate the language using a mass-spring simulator, with three springs and three mass points, as shown right.

In this section we introduce the forward simulator using the DiffSim frontend of Taichi, which is an easier-to-use wrapper of the Taichi C++14 frontend.

Allocating Global Variables Firstly we allocate a set of global tensors to store the simulation state.

These tensors include a scalar loss of type float32, 2D tensors x, v, force of size steps ×n_springs and type float32x2, and 1D arrays of size n_spring for spring properties: spring_anchor_a (int32), spring_anchor_b (int32), spring_length (float32).

Defining Kernels A mass-spring system is modeled by Hooke's law

where k is the spring stiffness, F is spring force, x a and x b are the positions of two mass points, and l 0 is the rest length.

The following kernel loops over all the springs and scatters forces to mass points:

For each particle i, we use semi-implicit Euler time integration with damping:

, m i are the velocity, position and mass of particle i at time step t, respectively.

α is a damping factor.

The kernel is as follows: The main goal of DiffSim's automatic differentiation (AD) system is to generate gradient simulators automatically with minimal code changes to the traditional forward simulators.

Design Decision Source Code Transformation (SCT) (Griewank & Walther, 2008) and Tracing (Wengert, 1964) are common choices when designing AD systems.

In our setting, using SCT to differentiate a whole simulator with thousands of time steps, results in high performance yet poor flexibility and long compilation time.

On the other hand, naively adopting tracing provides flexibility yet poor performance, since the "megakernel" structure is not preserved during backpropagation.

To get both performance and flexibility, we developed a two-scale automatic differentiation system ( Figure 2 ): we use SCT for differentiating within kernels, and use a light-weight tape that only stores function pointers and arguments for end-to-end simulation differentiation.

The global tensors are natural checkpoints for gradient evaluation.

Assumption Unlike functional programming languages where immutable output buffers are generated, imperative programming allows programmers to freely modify global tensors.

To make automatic differentiation well-defined under this setting, we make the following assumption on imperative kernels: Figure 2 : Left: The DiffSim system.

We reuse some infrastructure (white boxes) from Taichi, while the blue boxes are our extensions for differentiable programming.

Right: The tape records kernel launches and replays the gradient kernels in reverse order during backpropagation.

Global Data Access Rules: 1) If a global tensor element is written more than once, then starting from the second write, the write must come in the form of an atomic add ("accumulation").

2) No read accesses happen to a global tensor element, until its accumulation is done.

In forward simulators, programmers may make subtle changes to satisfy the rules.

For instance, in the mass-spring simulation example, we record the whole history of x and v, instead of keeping only the latest values.

The memory consumption issues caused by this can be alleviated via checkpointing, as discussed later in Appendix D.

With these assumptions, kernels will not overwrite the outputs of each other, and the goal of AD is clear: given a primal kernel f that takes as input X 1 , X 2 , . . .

, X n and outputs (or accumulates to)

Users can specify the storage of adjoint tensors using the Taichi data structure description language (Hu et al., 2019a) , as if they are primal tensors.

We also provide ti.root.lazy_grad() to automatically place the adjoint tensors following the layout of their primals.

A typical Taichi kernel consists of multiple levels of for loops and a body block.

To make later AD easier, we introduce two basic code transforms to simplify the loop body, as detailed below.

Flatten Branching In physical simulation branches are common, e.g. when implementing boundary conditions and collisions.

To simplify the reverse-mode AD pass, we first flatten "if" statements by replacing every instruction that leads to side effects with the ternary operator select(cond , value_if_true, value_if_false) -whose gradient is clearly defined -and a store instruction (Fig. 3, middle) .

This is a common transformation in program vectorization (e.g. Karrenberg & Hack (2011); Pharr & Mark (2012) ).

Eliminate Mutable Local Variables After removing branching, we end up with straight-line loop bodies.

To further simplify the IR and make the procedure truly single-assignment, we apply a series of local variable store forwarding transforms, until the mutable local variables can be fully eliminated (Fig. 3 , right).

After these two custom IR simplification transforms, DiffSim only has to differentiate the straightline code without mutable variables, which it achieves with reverse-mode AD, using a standard source code transformation (Griewank & Walther, 2008) .

More details on this transform are in Appendix B.

Loops Most loops in physical simulation are parallel loops, and during AD we preserve the parallel loop structures.

For loops that are not explicitly marked as parallel, we reverse the loop order during AD transforms.

We do not support loops that carry a mutating local variable since that would require a complex and costly run-time stack to maintain the history of local variables.

Instead, users are instructed to employ global variables that satisfy the global data access rules.

Parallelism and Thread Safety For forward simulation, we inherit the "parallel-for" construct from Taichi to map each loop iteration onto CPU/GPU threads.

Programmers use atomic operations for thread safety.

Our system can automatically differentiate these atomic operations.

Gradient contributions in backward kernels are accumulated to the adjoint tensors via atomic adds.

We construct a tape (Fig. 2 , right) of the kernel execution so that gradient kernels can be replayed in a reversed order.

The tape is very light-weight: since the intermediate results are stored in global tensors, during forward simulation the tape only records kernel names and the (scalar) input parameters, unlike other differentiable functional array systems where all the intermediate buffers have to be recorded by the tape.

Whenever a DiffSim kernel is launched, we append the kernel function pointer and parameters to the tape.

When evaluating gradients, we traverse the reversed tape, and invoke the gradient kernels with the recorded parameters.

Note that DiffSim AD is evaluating gradients with respect to input global tensors instead of the input parameters.

Learning/Optimization with Gradients Now we revisit the mass-spring example and make it differentiable for optimization.

Suppose the goal is to optimize the rest lengths of the springs so that the triangle area formed by the three springs becomes 0.2 at the end of the simulation.

We first define the loss function: Taichi Complex Kernels Sometimes the user may want to override the gradients provided by the compiler.

For example, when differentiating a 3D singular value decomposition done with an iterative solver, it is better to use a manually engineered SVD derivative subroutine for better stability.

We provide two more decorators ti.complex_kernel and ti.complex_kernel_grad to overwrite the default automatic differentiation, as detailed in Appendix C. Apart from custom gradients, complex kernels can also be used to implement checkpointing, as detailed in Appendix D.

We evaluate DiffSim on 10 different physical simulators covering large-scale continuum and smallscale rigid body simulations.

All results can be reproduced with the provided script.

The dynamic/optimization processes are visualized in the supplemental video.

In this section we focus our discussions on three simulators.

More details on the simulators are in Appendix E.

First, we build a differentiable continuum simulation for soft robotics applications.

The physical system is governed by momentum and mass conservation, i.e. ρ

We follow ChainQueen's implementation (Hu et al., 2019b) and use the moving least squares material point method (Hu et al., 2018) to simulate the system.

We were able to easily translate the original CUDA simulator into DiffSim syntax.

Using this simulator and an open-loop controller, we can easily train a soft robot to move forward (Fig. 1, diffmpm) .

Performance and Productivity Compared with manual gradient implementations in (Hu et al., 2019b) , getting gradients in DiffSim is effortless.

As a result, the DiffSim implementation is 4.2× shorter in terms of lines of code, and runs almost as fast; compared with TensorFlow, DiffSim code is 1.7× shorter and 188× faster ( Table 1) .

The Tensorflow implementation is verbose due to the heavy use of tf.gather_nd/scatter_nd and array transposing and broadcasting.

We implemented a smoke simulator (Fig. 1, smoke) with semi-Lagrangian advection (Stam, 1999) and implicit pressure projection, following the example in Autograd (Maclaurin et al., 2015) .

Using gradient descent optimization on the initial velocity field, we are able to find a velocity field that changes the pattern of the fluid to a target image ( Fig. 7a in Appendix).

We compare the performance of our system against PyTorch, Autograd, and JAX in Table 2 .

Note that as an example from the Autograd library, this grid-based simulator is intentionally simplified to suit traditional array-based programs.

For example, a periodic boundary condition is used so that Autograd can represent it using numpy.roll, without any branching.

Still, Taichi delivers higher performance than these arraybased systems.

The whole program takes 10 seconds to run in DiffSim on a GPU, and 2 seconds are spent on JIT.

JAX JIT compilation takes 2 minutes.

We built an impulse-based (Catto, 2009) differentiable rigid body simulator (Fig. 1, rigid_body) for optimizing robot controllers.

This simulator supports rigid body collision and friction, spring forces, joints, and actuation.

The simulation is end-to-end differentiable except for a countable number of discontinuities.

Interestingly, although the forward simulator works well, naively differentiating it with DiffSim leads to completely misleading gradients, due to the rigid body collisions.

We discuss the cause and solution of this issue below.

Improving collision gradients Consider the rigid ball example in Fig. 4 (left) , where a rigid ball collides with a friction-less ground.

Gravity is ignored, and due to conservation of kinetic energy the ball keeps a constant speed even after this elastic collision.

In the forward simulation, using a small ∆t often leads to a reasonable result, as done in many physics simulators.

Lowering the initial ball height will increase the final ball height, since there is less distance to travel before the ball hits the ground and more after (see the loss curves in Fig.4 , middle right).

However, using a naive time integrator, no matter how small ∆t is, the evaluated gradient of final height w.r.t.

initial height will be 1 instead of −1.

This counter-intuitive behavior is due to the fact that time discretization itself is not differentiated by the compiler.

We propose a simple solution of adding continuous collision resolution (see, for example, Redon et al. (2002)), which considers precise time of impact (TOI), to the forward program (Fig. 4 , middle left).

Although it barely improves the forward simulation (Fig. 4, middle right) , the gradient will be corrected effectively (Fig. 4, right) .

The details of continuous collision detection are in Appendix F. In real-world simulators, we find the TOI technique leads to significant improvement in gradient quality in controller optimization tasks (Fig. 5) .

Having TOI or not barely affects forward simulation: in the supplemental video, we show that a robot controller optimized in a simulator with TOI, actually works well in a simulator without TOI.

The takeaway is, differentiating physical simulators does not always yield useful gradients of the physical system being simulated, even if the simulator does forward simulation well.

In Appendix G, we discuss some additional gradient issues we have encountered. (Paszke et al., 2017) .

However, physical simulation requires complex and customizable operations due to the intrinsic computational irregularity.

Using the aforementioned frameworks, programmers have to compose these coarse-grained basic operations into desired complex operations.

Doing so often leads to unsatisfactory performance.

Earlier work on automatic differentiation focuses on transforming existing scalar code to obtain derivatives (e.g. (Kato et al., 2018) , redner (Li et al., 2018a) , Mitsuba 2 (Nimier-David et al., 2019)) to learn from 3D scenes.

We have presented DiffSim, a new differentiable programming language designed specifically for building high-performance differentiable physical simulators.

Motivated by the need for supporting megakernels, imperative programming, and flexible indexing, we developed a tailored two-scale automatic differentiation system.

We used DiffSim to build 10 simulators and integrated them into deep neural networks, which proved the performance and productivity of DiffSim over existing systems.

We hope our programming language can greatly lower the barrier of future research on differentiable physical simulation in the machine learning and robotics communities.

Workload differences between deep learning and differentiable physical simulation Existing differentiable programming tools for deep learning are typically centered around large data blobs.

For example, in AlexNet, the second convolution layer has size 27 × 27 × 128 × 128.

These tools usually provide users with both low-level operations such as tensor add and mul, and high-level operations such as convolution.

The bottleneck of typical deep-learning-based computer vision tasks are convolutions, so the provided high-level operations, with very high arithmetic intensity 2 , can fully exploit hardware capability.

However, the provided operations are "atoms" of these differentiable programming tools, and cannot be further customized.

Users often have to use low-level operations to compose their desired high-level operations.

This introduces a lot of temporary buffers, and potentially excessive GPU kernel launches.

As shown in Hu et al. (2019b) , a pure TensorFlow implementation of a complex physical simulator is 132× slower than a CUDA implementation, due to excessive GPU kernel launches and the lack of producer-consumer locality 3 .

The table below compares DiffSim with existing tools for build differentiable physical simulators.

Primal and adjoint kernels Recall that in DiffSim, (primal) kernels are operators that take as input multiple tensors (e.g., X, Y ) and output another set of tensors.

Mathematically, kernel f has the form

Kernels usually execute uniform operations on these tensors.

When it comes to differentiable programming, a loss function is defined on the final output tensors.

The gradients of the loss function "L" with respect to each tensor are stored in adjoint tensors and computed via adjoint kernels.

The adjoint tensor of (primal) tensor X ijk is denoted as X * ijk .

Its entries are defined by X * ijk = ∂L/∂X ijk .

At a high level, our automatic differentiation (AD) system transforms a primal kernel into its adjoint form.

Mathematically,

In this section we demonstrate how to use checkpointing via complex kernels.

The goal of checkpointing is to use recomputation to save memory space.

We demonstrate this using the diffmpm example, whose simulation cycle consists of particle to grid transform (p2g), grid boundary conditions (grid_op), and grid to particle transform (g2p).

We assume the simulation has O(n) time steps.

A naive implementation without checkpointing allocates O(n) copied of the simulation grid, which can cost a lot of memory space.

Actually, if we recompute the grid states during the backward simulation time step by redoing p2g and grid_op, we can reused the grid states and allocate only one copy.

This checkpointing optimization is demonstrated in the code below:

Given a simulation with O(n) time steps, if all simulation steps are recorded, the space consumption is O(n).

This linear space consumption is sometimes too large for high-resolution simulations with long time horizon.

Fortunately, we can reduce the space consumption using a segment-wise checkpointing trick: We split the simulation into segments of S steps, and in forward simulation store only the first simulation state in each segment.

During backpropagation when we need the remaining simulation states in a segment, we recompute them based on the first state in that segment.

Note that if the segment size is O(S), then we only need to store O(n/S) simulation steps for checkpoints and O(S) reusable simulation steps for backpropagation within segments.

The total space consumption is O(S + n/S).

We follow Tampubolon et al. (2017) and implemented a 3D differentiable liquid simulator.

Our liquid simulation can be two-way coupled with elastic object simulation [diffmpm] (Figure 6 , right).

(a) smoke (b) wave Backpropagating Through Pressure Projection We followed the baseline implementation in Autograd, and used 10 Jacobi iterations for pressure projection.

Technically, 10 Jacobi iterations are not sufficient to make the velocity field fully divergence-free.

However, in this example, it does a decent job, and we are able to successfully backpropagate through the unrolled 10 Jacobi iterations.

In larger-scale simulations, 10 Jacobi iterations are likely not sufficient.

Assuming the Poisson solve is done by an iterative solver (e.g. multigrid preconditioned conjugate gradients, MGPCG) with 5 multigrid levels and 50 conjugate gradient iterations, then automatic differentiation will likely not be able to provide gradients with sufficient numerical accuracy across this long iterative process.

The accuracy is likely worse with conjugate gradients present, as they are known to numerically drift as the number of iterations increases.

In this case, the user can still use DiffSim to implement the forward MGPCG solver, while implementing the backward part of the Poisson solve manually, likely using adjoint methods (Errico, 1997) .

DiffSim provides "complex kernels" to override the built-in AD system, as shown in Appendix C.

We adopt the wave equation in to model shallow water height field evolution:

where u is the height of shallow water, c is the "speed of sound" and α is a damping coefficient.

We use theu andü notations for the first and second order partial derivatives of u w.r.t time t respectively.

used the finite different time-domain (FDTD) method (Larsson & Thomée, 2008) to discretize Eqn.

1, yielding an update scheme:

We implemented this wave simulator in DiffSim to simulate shallow water.

We used a grid of resolution 128 × 128 and 256 time steps.

The loss function is defined as

where T is the final time step, andû is the target height field.

200 gradient descent iterations are then used to optimize the initial height field.

We setû to be the pattern "Taichi", and Fig. 7b shows the unoptimized and optimized wave evolution.

We set the "Taichi" symbol as the target pattern.

Fig. 7b shows the unoptimized and optimized final wave patterns.

More details on discretization is in Appendix E.

We extend the mass-spring system in the main text with ground collision and a NN controller.

The optimization goal is to maximize the distance moved forward with 2048 time steps.

We designed three mass-spring robots as shown in Fig. 8 (left) .

A differentiable rigid body simulator is built for optimizing a billiards strategy (Fig. 8, middle) .

We used forward Euler for the billiard ball motion and conservation of momentum and kinetic energy for collision resolution.

E.7 DIFFERENTIABLE RIGID BODY SIMULATOR [rigid_body]

Are rigid body collisions differentiable?

It is worth noting that discontinuities can happen in rigid body collisions, and at a countable number of discontinuities the objective function is nondifferentiable.

However, apart from these discontinuities, the process is still differentiable almost everywhere.

The situation of rigid body collision is somewhat similar to the "ReLU" activation function in neural networks: at point x = 0, ReLU is not differentiable (although continuous), yet it is still widely adopted.

The rigid body simulation cases are more complex than ReLU, as we have not only non-differentiable points, but also discontinuous points.

Based on our experiments, in these impulse-based rigid body simulators, we still find the gradients useful for optimization despite the discontinuities, especially with our time-of-impact fix.

We implemented differentiable renderers to visualize the refracting water surfaces from wave.

We use finite differences to reconstruct the water surface models based on the input height field and refract camera rays to sample the images, using bilinear interpolation for meaningful gradients.

To show our system works well with other differentiable programming systems, we use an adversarial optimization goal: fool VGG-16 into thinking that the refracted squirrel image is a goldfish (Fig. 9 ).

E.9 DIFFERENTIABLE VOLUME RENDERER [volume_renderer] We implemented a basic volume renderer that simply uses ray marching (we ignore light, scattering, etc.) to integrate a density field over each camera ray.

In this task, we render a number of target images from different viewpoints, with the camera rotated around the given volume.

The goal is then to optimize for the density field of the volume that would produce these target images: we render candidate images from the same viewpoints and compute an L2 loss between them and the target images, before performing gradient descent on the density field (Fig. 10 ).

Recall Coulomb's law: F = k q1q2 r 2r .

In the right figure, there are eight electrodes carrying static charge.

The red ball also carries static charge.

The controller, which is a two-layer neural network, tries to manipulate the electrodes so that the red ball follows the path of the blue ball.

The bigger the electrode, the more positive charge it carries.

# Note that with time of impact, dt is divided into two parts, # the first part using old_v, and second part using new_v new_x = old_x + toi * old_v + (dt -toi) * new_v

In rigid body simulation, the implementation follows the same idea yet is slightly more complex.

Please refer to rigid_body.py for more details.

Initialization matters: flat lands and local minima in physical processes A trivial example of objective flat land is in billiards.

Without proper initialization, gradient descent will make no progress since gradients are zero (Fig. 11) .

Also note the local minimum near (−5, 0.03).

In mass_spring and rigid_body, once the robot falls down, gradient descent will quickly become trapped.

A robot on the ground will make no further progress, no matter how it changes its controller.

This leads to a more non-trivial local minimum and zero gradient case.

Ideal physical models are only "ideal": discontinuities and singularities Real-world macroscopic physical processes are usually continuous.

However, building upon ideal physical models, even in the forward physical simulation results can contain discontinuities.

For example, in a rigid body model with friction, changing the initial rotation of the box can lead to different corners hitting the ground first, and result in a discontinuity (Fig. 12) .

In electric and mass_spring, due to the 1 r 2 and 1 r terms, when r → 0, gradients can be very inaccurate due to numerical precision issues.

Note that d(1/r)/dr = −1/r 2 , and the gradient is more numerically problematic than the primal for a small r. Safeguarding r is critically important for gradient stability.

Figure 12: Friction in rigid body with collision is a common source of discontinuity.

In this scene a rigid body hits the ground.

Slightly rotating the rigid body changes which corner (A/B) hits the ground first, and different normal/friction impulses will be applied to the rigid body.

This leads to a discontinuity in its final position (loss=final y coordinate). [Reproduce: python3 rigid_body_discontinuity.py]

Please see our supplemental video for more details.

<|TLDR|>

@highlight

We study the problem of learning and optimizing through physical simulations via differentiable programming, using our proposed DiffSim programming language and compiler.