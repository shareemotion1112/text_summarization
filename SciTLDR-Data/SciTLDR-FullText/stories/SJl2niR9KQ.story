Many machine learning image classifiers are vulnerable to adversarial attacks, inputs with perturbations designed to intentionally trigger misclassification.

Current adversarial methods directly alter pixel colors and evaluate against pixel norm-balls: pixel perturbations smaller than a specified magnitude, according to a measurement norm.

This evaluation, however, has limited practical utility since perturbations in the pixel space do not correspond to underlying real-world phenomena of image formation that lead to them and has no security motivation attached.

Pixels in natural images are measurements of light that has interacted with the geometry of a physical scene.

As such, we propose a novel evaluation measure, parametric norm-balls, by directly perturbing physical parameters that underly image formation.

One enabling contribution we present is a physically-based differentiable renderer that allows us to propagate pixel gradients to the parametric space of lighting and geometry.

Our approach enables physically-based adversarial attacks, and our differentiable renderer leverages models from the interactive rendering literature to balance the performance and accuracy trade-offs necessary for a memory-efficient and scalable adversarial data augmentation workflow.

Research in adversarial examples continues to contribute to the development of robust (semi-)supervised learning (Miyato et al., 2018) , data augmentation BID14 Sun et al., 2018) , and machine learning understanding BID24 .

One important caveat of the approach pursued by much of the literature in adversarial machine learning, as discussed recently (Goodfellow, original image parametric (lighting) texture color [Athalye 17] multi-step pixel [Moosavi Dezfooli 16] parametric (geometry) one-step pixel [Goodfellow 14] Figure 1: Traditional pixel-based adversarial attacks yield unrealistic images under a larger perturbation (L ∞ -norm ≈ 0.82), however our parametric lighting and geometry perturbations output more realistic images under the same norm (more results in Appendix A).

Figure 2: Parametrically-perturbed images remain natural, whereas pixel-perturbed ones do not.2018; BID12 , is the reliance on overly simplified attack metrics: namely, the use of pixel value differences between an adversary and an input image, also referred to as the pixel norm-balls.

The pixel norm-balls game considers pixel perturbations of norm-constrained magnitude BID14 , and is used to develop adversarial attackers, defenders and training strategies.

The pixel norm-ball game is attractive from a research perspective due to its simplicity and well-posedness: no knowledge of image formation is required and any arbitrary pixel perturbation remains eligible (so long as it is "small", in the perceptual sense).

Although the pixel norm-ball is useful for research purposes, it only captures limited real-world security scenarios.

Despite the ability to devise effective adversarial methods through the direct employment of optimizations using the pixel norm-balls measure, the pixel manipulations they promote are divorced from the types of variations present in the real world, limiting their usefulness "in the wild".

Moreover, this methodology leads to defenders that are only effective when defending against unrealistic images/attacks, not generalizing outside of the space constrained by pixel norm-balls.

In order to consider conditions that enable adversarial attacks in the real world, we advocate for a new measurement norm that is rooted in the physical processes that underly realistic image synthesis, moving away from overly simplified metrics, e.g., pixel norm-balls.

Our proposed solution -parametric norm-balls -rely on perturbations of physical parameters of a synthetic image formation model, instead of pixel color perturbations (Figure 2 ).

To achieve this, we use a physically-based differentiable renderer which allows us to perturb the underlying parameters of the image formation process.

Since these parameters indirectly control pixel colors, perturbations in this parametric space implicitly span the space of natural images.

We will demonstrate two advantages that fall from considering perturbations in this parametric space: (1) they enable adversarial approaches that more readily apply to real-world applications, and (2) they permit the use of much more significant perturbations (compared to pixel norms), without invalidating the realism of the resulting image ( Figure 1 ).

We validate that parametric norm-balls game playing is critical for a variety of important adversarial tasks, such as building defenders robust to perturbations that can occur naturally in the real world.

We perform perturbations in the underlying image formation parameter space using a novel physicallybased differentiable renderer.

Our renderer analytically computes the derivatives of pixel color with respect to these physical parameters, allowing us to extend traditional pixel norm-balls to physicallyvalid parametric norm-balls.

Notably, we demonstrate perturbations on an environment's lighting and on the shape of the 3D geometry it shades.

Our differentiable renderer achieves state-of-the-art performance in speed and scalability (Section 3) and is fast enough for rendered adversarial data augmentation (Section 5): training augmented with adversarial images generated with a renderer.

Existing differentiable renders are slow and do not scalable to the volume of high-quality, highresolutions images needed to make adversarial data augmentation tractable (Section 2).

Given our analytically-differentiable renderer (Section 3), we are able to demonstrate the efficacy of parametric space perturbations for generating adversarial examples.

These adversaries are based on a substantially different phenomenology than their pixel norm-balls counterparts (Section 4).

Ours is among the first steps towards the deployment of rendered adversarial data augmentation in real-world applications: we train a classifier with computer-generated adversarial images, evaluating the performance of the training against real photographs (i.e., captured using cameras; Section 5).

We test on real photos to show the parametric adversarial data augmentation increases the classifier's robustness to "deformations" happened in the real world.

Our evaluation differs from the majority of existing literature which evaluates against computer-generated adversarial images, since our parametric space perturbation is no-longer a wholly idealized representation of the image formation model but, instead, modeled against of theory of realistic image generation.

Our work is built upon the fact that simulated or rendered images can participate in computer vision and machine learning on real-world tasks.

Many previous works use rendered (simulated) data to train deep networks, and those networks can be deployed to real-world or even outperform the state-of-the-art networks trained on real photos (Movshovitz-Attias et al., 2016; BID6 Varol et al., 2017; BID4 BID22 Veeravasarapu et al., 2017b; Sadeghi & Levine, 2016; BID21 .

For instance, Veeravasarapu et al. (2017a) show that training with 10% real-world data and 90% simulation data can reach the level of training with full real data.

Tremblay et al. (2018) even demonstrate that the network trained on synthetic data yields a better performance than using real data alone.

As rendering can cheaply provide a theoretically infinite supply of annotated input data, it can generate data which is orders of magnitude larger than existing datasets.

This emerging trend of training on synthetic data provides an exciting direction for future machine learning development.

Our work complements these works.

We demonstrate the utility of rendering can be used to study the potential danger lurking in misclassification due to subtle changes to geometry and lighting.

This provides a future direction of combining with synthetic data generation pipelines to perform physically based adversarial training on synthetic data.

Szegedy et al. (2014) expose the vulnerability of modern deep neural nets using purposefully-manipulated images with human-imperceptible misclassification-inducing noise.

BID14 introduce a fast method to harness adversarial examples, leading to the idea of pixel norm-balls for evaluating adversarial attackers/defenders.

Since then, many significant developments in adversarial techniques have been proposed BID1 Szegedy et al., 2014; Rozsa et al., 2016; BID29 Moosavi Dezfooli et al., 2016; BID7 Papernot et al., 2017; Moosavi-Dezfooli et al., 2017; BID5 Su et al., 2017) .

Our work extends this progression in constructing adversarial examples, a problem that lies at the foundation of adversarial machine learning.

BID28 study the transferability of attacks to the physical world by printing then photographing adversarial images.

BID2 and BID10 propose extensions to non-planar (yet, still fixed) geometry and multiple viewing angles.

These works still rely fundamentally on the direct pixel or texture manipulation on physical objects.

Since these methods assume independence between pixels in the image or texture space they remain variants of pixel norm-balls.

This leads to unrealistic attack images that cannot model real-world scenarios BID13 BID18 BID12 .

Zeng et al. (2017) generate adversarial examples by altering physical parameters using a rendering network BID19 trained to approximate the physics of realistic image formation.

This data-driven approach leads to an image formation model biased towards the rendering style present in the training data.

This method also relies on differentiation through the rendering network in order to compute adversaries, which requires high-quality training on a large amount of data.

Even with perfect training, in their reported performance, it still requires 12 minutes on average to find new adversaries, we only take a few seconds Section 4.1.

Our approach is based on a differentiable physically-based renderer that directly (and, so, more convincingly) models the image formation process, allowing us to alter physical parameters -like geometry and lighting -and compute Table 1 : Previous non-pixel attacks fall short in either the parameter range they can take derivatives or the performance.

Perf.

Color Normal Material Light Geo.

Zeng 17 Ours derivatives (and adversarial examples) much more rapidly compared to the (Zeng et al., 2017) .

We summarize the difference between our approach and the previous non-image adversarial attacks in Table 1 .

BID2 Zeng et al., 2017) , and in generalizing neural style transfer to a 3D context BID25 Liu et al., 2018 TAB0 ).

Our renderer explicitly models the physics of the image formation processes, and so the images it generates are realistic enough to illicit correct classifications from networks trained on real-world photographs.

Adversarial attacks based on pixel norm-balls typically generate adversarial examples by defining a cost function over the space of images C : I → R that enforces some intuition of what failure should look like, typically using variants of gradient descent where the gradient ∂C /∂I is accessible by differentiating through networks (Szegedy et al., 2014; BID14 Rozsa et al., 2016; BID29 Moosavi Dezfooli et al., 2016; BID7 .The choices for C include increasing the cross-entropy loss of the correct class BID14 , decreasing the cross-entropy loss of the least-likely class BID29 , using a combination of cross-entropies (Moosavi Dezfooli et al., 2016) , and more (Szegedy et al., 2014; Rozsa et al., 2016; BID7 Tramèr et al., 2017) .

We combine of cross-entropies to provide flexibility for choosing untargeted and targeted attacks by specifying a different set of labels: DISPLAYFORM0 where I is the image, f (I) is the output of the classifier, L d , L i are labels which a user wants to decrease and increase the predicted confidences respectively.

In our experiments, L d is the correct class and L i is either ignored or chosen according to user preference.

Our adversarial attacks in the parametric space consider an image I(U, V ) is the function of physical parameters of the image formation model, including the lighting U and the geometry V .

Adversarial examples constructed by perturbing physical parameters can then be computed via the chain rule DISPLAYFORM1 where ∂I /∂U, ∂I /∂V are derivatives with respect to the physical parameters and we evaluate using our physically based differentiable renderer.

In our experiments, we use gradient descent for finding parametric adversarial examples where the gradient is the direction of ∂I /∂U, ∂I /∂V .

Rendering is the process of generating a 2D image from a 3D scene by simulating the physics of light.

Light sources in the scene emit photons that then interact with objects in the scene.

At each interaction, photons are either reflected, transmitted or absorbed, changing trajectory and repeating until arriving at a sensor such as a camera.

A physically based renderer models the interactions mathematically (Pharr et al., 2016) , and our task is to analytically differentiate the physical process.

Top 5: miniskirt 28% t-shirt 21% boot 6% crutch 5% sweatshirt 5% t-shirt 86%Top 5: water tower 48% street sign 18% mailbox 9% gas pump 3% barn 3% street sign 57% Figure 4 : By changing the lighting, we fool the classifier into seeing miniskirt and water tower, demonstrating the existence of adversarial lighting.boot 100% boot 98% boot 98% sleeping bag 98% watter bottle 15% cannon 20% We develop our differentiable renderer with common assumptions in real-time rendering (AkenineMoller et al., 2008) -diffuse material, local illumination, and distant light sources.

Our diffuse material assumption considers materials which reflect lights uniformly for all directions, equivalent to considering non-specular objects.

We assume that variations in the material (texture) are piece-wise constant with respect to our triangle mesh discretization.

The local illumination assumption only considers lights that bounce directly from the light source to the camera.

Lastly, we assume light sources are far away from the scene, allowing us to represent lighting with one spherical function.

For a more detailed rationale of our assumptions, we refer readers to Appendix B).

These assumptions simplify the complicated integral required for rendering BID23 and allow us to represent lighting in terms of spherical harmonics, an orthonormal basis for spherical functions analogous to Fourier transformation.

Thus, we can analytically differentiate the rendering equation to acquire derivatives with respect to lighting, geometry, and texture (derivations found in Appendix C).Using analytical derivatives avoids pitfalls of previous differentiable renderers (see Section 2) and make our differentiable renderer orders of magnitude faster than the previous fully differentiable renderer OPENDR (Loper & Black, 2014 ) (see FIG1 ).

Our approach is scalable to handle problems with more than 100,000 variables, while OPENDR runs out of memory for problems with more than 3,500 variables.

Adversarial lighting denotes adversarial examples generated by changing the spherical harmonics lighting coefficients U BID15 .

As our differentiable renderer allows us to compute ∂I /∂U analytically (derivation is provided in Appendix C.4), we can simply apply the chain rule: DISPLAYFORM0 where ∂C /∂I is the derivative of the cost function with respect to pixel colors and can be obtained by differentiating through the network.

Spherical harmonics act as an implicit constraint to prevent unrealistic lighting because natural lighting environments everyday life are dominated by lowfrequency signals.

For instance, rendering of diffuse materials can be approximated with only 1% pixel intensity error by the first 2 orders of spherical harmonics (Ramamoorthi & Hanrahan, 2001) .

As computers can only represent a finite number of coefficients, using spherical harmonics for lighting implicitly filters out high-frequency, unrealistic lightings.

Thus, perturbing the parametric space of spherical harmonics lighting gives us more realistic compared to image-pixel perturbations Figure 1 .jaguar 61% jaguar 80% Egyptian cat 90% hunting dog 93% Figure 6 : By specifying different target labels, we can create an optical illusion: a jaguar is classified as cat and dog from two different views after geometry perturbations.

Adversarial geometry is an adversarial example computed by changes the position of the shape's surface.

The shape is encoded as a triangle mesh with |V | vertices and |F | faces, surface points are vertex positions V ∈ R |V |×3 which determine per-face normals N ∈ R |F |×3 which in turn determine the shading of the surface.

We can compute adversarial shapes by applying the chain rule: DISPLAYFORM1 ni hij vj where ∂I /∂N is computed via a derivation in Appendix E. Each triangle only has one normal on its face, making ∂N /∂V computable analytically.

In particular, the 3 × 3 Jacobian of a unit face normal vector n i ∈ R 3 of the jth face of the triangle mesh V with respect to one of its corner vertices v j ∈ R 3 is DISPLAYFORM2 where h ij ∈ R 3 is the height vector: the shortest vector to the corner v j from the opposite edge.

We have described how to compute adversarial examples by parametric perturbations, including lighting and geometry.

In this section, we show that adversarial examples exist in the parametric spaces, then we analyze the characteristics of those adversaries and parametric norm-balls.

We use 49 × 3 spherical harmonics coefficients to represent environment lighting, with an initial realworld lighting condition (Ramamoorthi & Hanrahan, 2001) .

Camera parameters and the background images are empirically chosen to have correct initial classifications and avoid synonym sets.

In Figure 4 we show that single-view adversarial lighting attack can fool the classifier (pre-trained ResNet-101 on ImageNet BID17 ).

FIG0 shows multi-view adversarial lighting, which optimizes the summation of the cost functions for each view, thus the gradient is computed as the summation over all camera views: DISPLAYFORM0 missile 49% wing 33%Figure 7: Even if we further constrain to a lighting subspace, skylight, we can still find adversaries.

If one is interested in a more specific subspace, such as outdoor lighting conditions governed by sunlight and weather, our adversarial lighting can adapt to it.

In Figure 7 , we compute adversarial lights over the space of skylights by applying one more chain rule to the Preetham skylight parameters (Preetham et al., 1999; BID16 .

Details about taking these derivatives are provided in Appendix D. Although adversarial skylight exists, its low degrees of freedom (only three parameters) makes it more difficult to find adversaries.

In FIG2 and Figure 9 we show the existence of adversarial geometry in both single-view and multi-view cases.

Note that we upsample meshes to have >10K vertices as a preprocessing step to increase the degrees of freedom available for perturbations.

Multiview adversarial geometry enables us to perturb the same 3D shape from different viewing directions, which enables us to construct a deep optical illusion: The same 3D shape are classified differently from different angles.

To create the optical illusion in Figure 6 , we only need to specify the L i in Equation FORMULA0 to be a dog and a cat for two different views.

street sign 86% street sign 99% street sign 91% mailbox 71% mailbox 61% mailbox 51% Figure 9 : We construct a single adversarial geometry that fools the classifier seeing a mailbox from different angles.

To further understand parametric adversaries, we analyze how do parametric adversarial examples generalize to black-box models.

In TAB2 , we test 5,000 ResNet parametric adversaries on unseen networks including AlexNet BID27 , DenseNet BID19 , SqueezeNet (Iandola et al., 2016) , and VGG (Simonyan & Zisserman, 2014) .

Our result shows that parametric adversarial examples also share across models.

In addition to different models, we evaluate parametric adversaries on black-box viewing directions.

This evaluation mimics the real-world scenario that a self-driving car would "see" a stop sign from different angles while driving.

In TAB3 , we randomly sample 500 correctly classified views for a given shape and perform adversarial lighting and geometry algorithms only on a subset of views, then evaluate the resulting adversarial lights/shapes on all the views.

The results show that adversarial lights are more generalizable to fool unseen views; adversarial shapes, yet, are less generalizable.init.

light adv.

light init. geo.

adv. geo.

9.1%

17.5% Figure 10 : A quantitative comparison using parametric norm-balls shows the fact that adversarial lighting/geometry perturbations have a higher success rate (%) in fooling classifiers comparing to random perturbations in the parametric spaces.

Switching from pixel norm-balls to parametric norm-balls only requires to change the normconstraint from the pixel color space to the parametric space.

For instance, we can perform a quantitative comparison between parametric adversarial and random perturbations in Figure 10 .

We use L ∞ -norm = 0.1 to constraint the perturbed magnitude of each lighting coefficient, and L ∞ -norm = 0.002 to constrain the maximum displacement of surface points along each axis.

The results show how many parametric adversaries can fool the classifier out of 10,000 adversarial lights and shapes respectively.

Not only do the parametric norm-balls show the effectiveness of adversarial perturbation, evaluating robustness using parametric norm-balls has real-world implications.

Adversarial Geometry #pixel Runtime The inset presents our runtime per iteration for computing derivatives.

An adversary normally requires less than 10 iterations, thus takes a few seconds.

We evaluate our CPU PYTHON implementation and the OPENGL rendering, on an Intel Xeon 3.5GHz CPU with 64GB of RAM and an NVIDIA GeForce GTX 1080.

Our runtime depends on the number of pixels requiring derivatives.

We inject adversarial examples, generated using our differentiable renderer, into the training process of modern image classifiers.

Our goal is to increase the robustness of these classifiers to real-world perturbations.

Traditionally, adversarial training is evaluated against computer-generated adversarial images BID29 Madry et al., 2018; Tramèr et al., 2017) .

In contrast, our evaluation differs from the majority of the literature, as we evaluate performance against real photos (i.e., images captured using a camera), and not computer-generated images.

This evaluation method is motivated by our goal of increasing a classifier's robustness to "perturbations" that occur in the real world and result from the physical processes underlying real-world image formation.

We present preliminary steps towards this objective, resolving the lack of realism of pixel norm-balls and evaluating our augmented classifiers (i.e., those trained using our rendered adversaries) against real photographs.

Training We train the WideResNet (16 layers, 4 wide factor) (Zagoruyko & Komodakis, 2016) on CIFAR-100 BID26 ) augmented with adversarial lighting examples.

We apply a common adversarial training method that adds a fixed number of adversarial examples each epoch BID14 BID29 .

We refer readers to Appendix F for the training detail.

In our experiments, we compare three training scenarios: (1) CIFAR-100, (2) CIFAR-100 + 100 images under random lighting, and (3) CIFAR-100 + 100 images under adversarial lighting.

Comparing to the accuracy reported in (Zagoruyko & Komodakis, 2016), WideResNets trained on these three cases all have comparable performance (≈ 77%) on the CIFAR-100 test set.

Figure 11: Unlike much of the literature on adversarial training, we evaluate against real photos (captured by a camera), not computergenerated images.

This figure illustrates a subset of our test data.

Testing We create a test set of real photos, captured in a laboratory setting with controlled lighting and camera parameters: we photographed oranges using a calibrated Prosilica GT 1920 camera under different lighting conditions, each generated by projecting different lighting patterns using an LG PH550 projector.

This hardware lighting setup projects lighting patterns from a fixed solid angle of directions onto the scene objects.

Figure 11 illustrates samples from the 500 real photographs of our dataset.

We evaluate the robustness of our classifier models according to test accuracy.

Of note, average prediction accuracies over five trained WideResNets on our test data under the three training cases are (1) 4.6%, (2) 40.4%, and (3) 65.8%.

This result supports the fact that training on rendered images can improve the networks' performance on real photographs.

Our preliminary experiments motivate the potential of relying on rendered adversarial training to increase the robustness to visual phenomena present in the real-world inputs.

Using parametric norm-balls to remove the lack of realism of pixel norm-balls is only the first step to bring adversarial machine learning to real-world.

More evaluations beyond the lab experimental data could uncover the potential of the rendered adversarial data augmentation.

Coupling the differentiable renderer with methods for reconstructing 3D scenes, such as (Veeravasarapu et al., 2017b; Tremblay et al., 2018) , has the potential to develop a complete pipeline for rendered adversarial training.

We can take a small set of real images, constructing 3D virtual scenes which have real image statistics, using our approach to manipulate the predicted parameters to construct the parametric adversarial examples, then perform rendered adversarial training.

This direction has the potential to produce limitless simulated adversarial data augmentation for real-world tasks.

Our differentiable renderer models the change of realistic environment lighting and geometry.

Incorporating real-time rendering techniques from the graphics community could further improve the quality of rendering.

Removing the locally constant texture assumption could improve our results.

Extending the derivative computation to materials could enable "adversarial materials".

Incorporating derivatives of the visibility change and propagating gradient information to shape skeleton could also create "adversarial poses".

These extensions offer a set of tools for modeling real security scenarios.

We extend our comparisons against pixel norm-balls methods (Figure 1 ) by visualizing the results and the generated perturbations (Figure 12 ).

We hope this figure elucidates that our parametric perturbation are more realistic several scales of perturbations.original image parametric (lighting) texture color [Athalye 17] one-step pixel [Goodfellow 14] multi-step pixel [Moosavi Dezfooli 16]

Physically based rendering (PBR) seeks to model the flow of light, typically the assumption that there exists a collection of light sources that generate light; a camera that receives this light; and a scene that modulates the flow light between the light sources and camera (Pharr et al., 2016) .

What follows is a brief discussion of the general task of rendering an image from a scene description and the approximations we take in order to make our renderer efficient yet differentiable.

Computer graphics has dedicated decades of effort into developing methods and technologies to enable PBR to synthesize of photorealistic images under a large gamut of performance requirements.

Much of this work is focused around taking approximations of the cherished Rendering equation BID23 , which describes the propagation of light through a point in space.

If we let u o be the output radiance, p be the point in space, ω o be the output direction, u e be the emitted radiance, u i be incoming radiance, ω i be the incoming angle, f r be the way light be reflected off the material at that given point in space we have: DISPLAYFORM0 From now on we will ignore the emission term u e as it is not pertinent to our discussion.

Furthermore, because the speed of light is substantially faster than the exposure time of our eyes, what we perceive is not the propagation of light at an instant, but the steady state solution to the rendering equation evaluated at every point in space.

Explicitly computing this steady state is intractable for our applications and will mainly serve as a reference for which to place a plethora of assumptions and simplifications we will make for the sake of tractability.

Many of these methods focus on ignoring light with nominal effects on the final rendered image vis a vis assumptions on the way light travels.

For instance, light is usually assumed to have nominal interacts with air, which is described as the assumption that the space between objects is a vacuum, which constrains the interactions of light to the objects in a scene.

Another common assumption is that light does not penetrate objects, which makes it difficult to render objects like milk and human skin 1 .

This constrains the complexity of light propagation to the behavior of light bouncing off of object surfaces.

Figure 14 : Rasterization converts a 3D scene into pixels.

It is common to see assumptions that limit number of bounces light is allowed.

In our case we chose to assume that the steady state is sufficiently approximated by an extremely low number of iterations: one.

This means that it seems sufficient to model the lighting of a point in space by the light sent to it directly by light sources.

Working with such a strong simplification does, of course, lead to a few artifacts.

For instance, light occluded by other objects is ignored so shadows disappear and auxiliary techniques are usually employed to evaluate shadows (Williams, 1978; Miller, 1994) .When this assumption is coupled with a camera we approach what is used in standard rasterization systems such as OPENGL (Shreiner & Group, 2009) , which is what we use.

These systems compute the illumination of a single pixel by determining the fragment of an object visible through that pixel and only computing the light that traverses directly from the light sources, through that fragment, to that pixel.

The lighting of a fragment is therefore determined by a point and the surface normal at that point, so we write the fragment's radiance as R(p, n, DISPLAYFORM0

Lambertian Non-Lambertian Each point on an object has a model approximating the transfer of incoming light to a given output direction f r , which is usually called the material.

On a single object the material parameters may vary quite a bit and the correspondence between points and material parameters is usually called the texture map which forms the texture of an object.

There exists a wide gamut of material models, from mirror materials that transport light from a single input direction to a single output direction, to materials that reflect light evenly in all directions, to materials liked brushed metal that reflect differently along different angles.

For the sake of document we only consider diffuse materials, also called Lambertian materials, where we assume that incoming light is reflected uniformly, i.e f r is a constant function with respect to angle, which we denote f r (p, DISPLAYFORM0 This function ρ is usually called the albedo, which can be perceived as color on the surface for diffuse material, and we reduce our integration domain to the upper hemisphere Ω(n) in order to model light not bouncing through objects.

Furthermore, since only the only ω and u are the incoming ones we can now suppress the "incoming" in our notation and just use ω and u respectively.

The illumination of static, distant objects such as the ground, the sky, or mountains do not change in any noticeable fashion when objects in a scene are moved around, so u can be written entirely in terms of ω, u(p, ω) = u(ω).

If their illumination forms a constant it seems prudent to pre-compute or cache their contributions to the illumination of a scene.

This is what is usually called environment mapping and they fit in the rendering equation as a representation for the total lighting of a scene, i.e the total incoming radiance u i .

Because the environment is distant, it is common to also assume that the position of the object receiving light from an environment map does not matter so this simplifies u i to be independent of position: DISPLAYFORM0

Despite all of our simplifications, the inner integral is still a fairly generic function over S 2 .

Many techniques for numerically integrating the rendering equation have emerged in the graphics community and we choose one which enables us to perform pre-computation and select a desired spectral accuracy: spherical harmonics.

Spherical harmonics are a basis on S 2 so, given a spherical harmonics expansion of the integrand, the evaluation of the above integral can be reduced to a weighted product of coefficients.

This particular basis is chosen because it acts as a sort of Fourier basis for functions on the sphere and so the bases are each associated with a frequency, which leads to a convenient multi-resolution structure.

In fact, the rendering of diffuse objects under distant lighting can be 99% approximated by just the first few spherical harmonics bases (Ramamoorthi & Hanrahan, 2001 ).We will only need to note that the spherical harmonics bases Y m l are denoted with the subscript with l as the frequency and that there are 2l + 1 functions per frequency, denoted by superscripts m between −l to l inclusively.

For further details on them please take a glance at Appendix C.If we approximate a function f in terms of spherical harmonics coefficients f ≈ lm f l,m Y m l the integral can be precomputed as DISPLAYFORM0 Thus we have defined a reduced rendering equation that can be efficiently evaluated using OPENGL while maintaining differentiability with respect to lighting and vertices.

In the following appendix we will derive the derivatives necessary to implement our system.

Rendering computes an image of a 3D shape given lighting conditions and the prescribed material properties on the surface of the shape.

Our differentiable renderer assumes Lambertian reflectance, distant light sources, local illumination, and piece-wise constant textures.

We will discuss how to explicitly compute the derivatives used in the main body of this text.

Here we give a detailed discussion about spherical harmonics and their advantages.

Spherical harmonics are usually defined in terms of the Legendre polynomials, which are a class of orthogonal polynomials defined by the recurrence relation DISPLAYFORM0 The associated Legendre polynomials are a generalization of the Legendre polynomials and can be fully defined by the relations DISPLAYFORM1 Using the associated Legendre polynomials P m l we can define the spherical harmonics basis as DISPLAYFORM2 where DISPLAYFORM3 We will use the fact that the associated Legendre polynomials correspond to the spherical harmonics bases that are rotationally symmetric along the z axis (m = 0).In order to incorporate spherical harmonics into Equation 8, we change the integral domain from the upper hemisphere Ω(n) back to S 2 via a max operation DISPLAYFORM4 We see that the integral is comprised of two components: a lighting component u(ω) and a component that depends on the normal max(ω · n, 0).

The strategy is to pre-compute the two components by projecting onto spherical harmonics, and evaluating the integral via a dot product at runtime, as we will now derive.

Approximating the lighting component u(ω) in Equation 19 using spherical harmonics Y m l up to band n can be written as DISPLAYFORM0 where U l,m ∈ R are coefficients.

By using the orthogonality of spherical harmonics we can use evaluate these coefficients as an integral between u(ω) and Y DISPLAYFORM1 So far we have only considered the shading of a specific point p with surface normal n. If we consider the rendered image I given a shape V , lighting U , and camera parameters η, the image I is the evaluation of the rendering equation R of each point in V visible through each pixel in the image.

This pixel to point mapping is determined by η.

Therefore, we can write I as DISPLAYFORM2 where N (V ) is the surface normal.

We exploit the notation and use ρ(V, η) to represent the texture of V mapped to the image space through η.

For our applications we must differentiate Equation 25 with respect to lighting and material parameters.

The derivative with respect to the lighting coefficients U can be obtained by DISPLAYFORM0 This is the Jacobian matrix that maps from spherical harmonics coefficients to pixels.

The term ∂F /∂U l,m can then be computed as DISPLAYFORM1 The derivative with respect to texture is defined by DISPLAYFORM2 Note that we assume texture variations are piece-wise constant with respect to our triangle mesh discretization.

To model possible outdoor daylight conditions, we use the analytical Preetham skylight model (Preetham et al., 1999) .

This model is calibrated by atmospheric data and parameterized by two intuitive parameters: turbidity τ , which describes the cloudiness of the atmosphere, and two polar angles θ s ∈ [0, π/2], φ s ∈ [0, 2π], which are encode the direction of the sun.

Note that θ s , φ s are not the polar angles θ, φ for representing incoming light direction ω in u(ω).

The spherical harmonics representation of the Preetham skylight is presented in BID16 as DISPLAYFORM0 This is derived by first performing a non-linear least squares fit to write U l,m as a polynomial of θ s and τ which lets them solve forŨ l,m (θ DISPLAYFORM1 where (p l,m ) i,j are scalar coefficients, then U l,m (θ s , φ s , τ ) can be computed by applying a spherical harmonics rotation with φ s using DISPLAYFORM2 We refer the reader to (Preetham et al., 1999) for more detail.

For the purposes of this article we just need the above form to compute the derivatives.

The derivatives of the lighting with respect to the skylight parameters (θ s , φ s , τ ) are DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 We assume the texture variations are piece-wise constant with respect to our triangle mesh discretization and omit the first term ∂ρ/∂V as the magnitude is zero.

Computing ∂N /∂V is provided in Section 3.2.

Computing ∂F /∂Ni on face i is DISPLAYFORM3 where the ∂Y m l /∂Ni is the derivative of the spherical harmonics with respect to the face normal N i .

To begin this derivation recall the relationship between a unit normal vector n = (n x , n y , n z ) and its corresponding polar angles θ, φ θ = cos other experiments, using the real-world lighting data provided in (Ramamoorthi & Hanrahan, 2001 ).

Our stepsize for computing adversaries is 0.05 along the direction of lighting gradients.

We run our adversarial lighting iterations until fooling the network or reaching the maximum 30 iterations to avoid too extreme lighting conditions, such as turning the lights off.

Our random lighting examples are constructed at each epoch by randomly perturb the lighting coefficients ranging from -0.5 to 0.5.When training the 16-layers WideResNet (Zagoruyko & Komodakis, 2016) with wide-factor 4, we use batch size 128, learning rate 0.125, dropout rate 0.3, and the standard cross entropy loss.

We implement the training using PYTORCH (Paszke et al., 2017) , with the SGD optimizer and set the Nesterov momentum 0.9, weight decay 5e-4.

We train the model for 150 epochs and use the one with best accuracy on the validation set.

FIG7 shows examples of our adversarial lights at different training stages.

In the early stages, the model is not robust to different lighting conditions, thus small lighting perturbations are sufficient to fool the model.

In the late stages, the network becomes more robust to different lightings.

Thus it requires dramatic changes to fool a model or even fail to fool the model within 30 iterations.

G EVALUATE RENDERING QUALITY We evaluated our rendering quality by whether our rendered images are recognizable by models trained on real photographs.

Although large 3D shape datasets, such as ShapeNet BID4 , are available, they do not have have geometries or textures at the resolutions necessary to create realistic renderings.

We collected 75 high-quality textured 3D shapes from cgtrader.com and turbosquid.com to evaluate our rendering quality.

We augmented the shapes by changing the field of view, backgrounds, and viewing directions, then keep the configurations that were correctly classified by a pre-trained ResNet-101 on ImageNet.

Specifically, we place the centroid, calculated as the weighted average of the mesh vertices where the weights are the vertex areas, at the origin and normalize shapes to range -1 to 1; the field of view is chosen to be 2 and 3 in the same unit with the normalized shape; background images include plain colors and real photos, which have small influence on model predictions; viewing directions are chosen to be 60 degree zenith and uniformly sampled 16 views from 0 to 2π azimuthal angle.

In FIG8 , we show that the histogram of model confidence on the correct labels over 10,000 correctly classified rendered images from our differentiable renderer.

The confidence is computed using softmax function and the results show that our rendering quality is faithful enough to be recognized by models trained on natural images.

@highlight

Enabled by a novel differentiable renderer, we propose a new metric that has real-world implications for evaluating adversarial machine learning algorithms, resolving the lack of realism of the existing metric based on pixel norms.