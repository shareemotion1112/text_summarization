We focus on temporal self-supervision for GAN-based video generation tasks.

While adversarial training successfully yields generative models for a variety of areas, temporal relationship in the generated data is much less explored.

This is crucial for sequential generation tasks, e.g. video super-resolution and unpaired video translation.

For the former, state-of-the-art methods often favor simpler norm losses such as L2 over adversarial training.

However, their averaging nature easily leads to temporally smooth results with an undesirable lack of spatial detail.

For unpaired video translation, existing approaches modify the generator networks to form spatio-temporal cycle consistencies.

In contrast, we focus on improving the learning objectives and propose a temporally self-supervised algorithm.

For both tasks, we show that temporal adversarial learning is key to achieving temporally coherent solutions without sacrificing spatial detail.

We also propose a novel Ping-Pong loss to improve the long-term temporal consistency.

It effectively prevents recurrent networks from accumulating artifacts temporally without depressing detailed features.

We also propose a first set of metrics to quantitatively evaluate the accuracy as well as the perceptual quality of the temporal evolution.

A series of user studies confirms the rankings computed with these metrics.

Generative adversarial models (GANs) have been extremely successful at learning complex distributions such as natural images (Zhu et al., 2017; Isola et al., 2017) .

However, for sequence generation, directly applying GANs without carefully engineered constraints typically results in strong artifacts over time due to the significant difficulties introduced by the temporal changes.

In particular, conditional video generation tasks are very challenging learning problems where generators should not only learn to represent the data distribution of the target domain, but also learn to correlate the output distribution over time with conditional inputs.

Their central objective is to faithfully reproduce the temporal dynamics of the target domain and not resort to trivial solutions such as features that arbitrarily appear and disappear over time.

In our work, we propose a novel adversarial learning method for a recurrent training approach that supervises both spatial content as well as temporal relationships.

We apply our approach to two video-related tasks that offer substantially different challenges: video super-resolution (VSR) and unpaired video translation (UVT).

With no ground truth motion available, the spatio-temporal adversarial loss and the recurrent structure enable our model to generate realistic results while keeping the generated structures coherent over time.

With the two learning tasks we demonstrate how spatio-temporal adversarial training can be employed in paired as well as unpaired data domains.

In addition to the adversarial network which supervises the short-term temporal coherence, long-term consistency is self-supervised using a novel bi-directional loss formulation, which we refer to as "Ping-Pong" (PP) loss in the following.

The PP loss effectively avoids the temporal accumulation of artifacts, which can potentially benefit a variety of recurrent architectures.

The central contributions of our work are: a spatio-temporal discriminator unit together with a careful analysis of training objectives for realistic and coherent video generation tasks, a novel PP loss supervising long-term consistency, in addition to a set of metrics for quantifying temporal coherence based on motion estimation and perceptual distance.

Together, our contributions lead to models that outperform previous work in terms of temporally-coherent detail, which we quantify with a wide range of metrics and user studies.

features, but collapses to essentially static outputs of Obama.

It manages to transfer facial expressions back to Trump using tiny differences encoded in its Obama outputs, instead of learning a meaningful mapping.

Being able to establish the correct temporal cycle-consistency between domains, ours and RecycleGAN can generate correct blinking motions.

Our model outperforms the latter in terms of coherent detail that is generated.

Deep learning has made great progress for image generation tasks.

While regular losses such as L 2 (Kim et al., 2016; Lai et al., 2017) offer good performance for image super-resolution (SR) tasks in terms of PSNR metrics, GAN researchers found adversarial training (Goodfellow et al., 2014) to significantly improve the perceptual quality in multi-modal problems including image SR (Ledig et al., 2016) , image translations (Zhu et al., 2017; Isola et al., 2017) , and others.

Perceptual metrics (Zhang et al., 2018; Prashnani et al., 2018) are proposed to reliably evaluate image similarity by considering semantic features instead of pixel-wise errors.

Video generation tasks, on the other hand, require realistic results to change naturally over time.

Recent works in VSR improve the spatial detail and temporal coherence by either using multiple low-resolution (LR) frames as inputs (Jo et al., 2018; Tao et al., 2017; Liu et al., 2017) , or recurrently using previously estimated outputs .

The latter has the advantage to re-use high-frequency details over time.

In general, adversarial learning is less explored for VSR and applying it in conjunction with a recurrent structure gives rise to a special form of temporal mode collapse, as we will explain below.

For video translation tasks, GANs are more commonly used but discriminators typically only supervise the spatial content.

E.g., Zhu et al. (2017) does not employ temporal constrains and generators can fail to learn the temporal cycle-consistency.

In order to learn temporal dynamics, RecycleGAN (Bansal et al., 2018) proposes to use a prediction network in addition to a generator, while a concurrent work (Chen et al., 2019) chose to learn motion translation in addition to spatial content translation.

Being orthogonal to these works, we propose a spatiotemporal adversarial training for both VSR and UVT and we show that temporal self-supervision is crucial for improving spatio-temporal correlations without sacrificing spatial detail.

While L 2 temporal losses based on warping are used to enforce temporal smoothness in video style transfer tasks (Ruder et al., 2016; Chen et al., 2017) , concurrent GAN-based VSR work (Pérez-Pellitero et al., 2018) and UVT work (Park et al., 2019) , it leads to an undesirable smooth over spatial detail and temporal changes in outputs.

Likewise, the L 2 temporal metric represents a sub-optimal way to quantify temporal coherence and perceptual metrics that evaluate natural temporal changes are unavailable up to now.

We work on this open issue, propose two improved temporal metric and demonstrate the advantages of temporal self-supervision over direct temporal losses.

Previous work, e.g. tempoGAN (Xie et al., 2018) and vid2vid (Wang et al., 2018b) , have proposed adversarial temporal losses to achieve time consistency.

While tempoGAN employs a second temporal discriminator with multiple aligned frames to assess the realism of temporal changes, it is not suitable for videos, as it relies on ground truth motions and employs a single-frame processing that is sub-optimal for natural images.

On the other hand, vid2vid focuses on paired video translations and proposes a video discriminator based on a conditional motion input that is estimated from the paired ground-truth sequences.

We focus on more difficult unpaired translation tasks instead, and demonstrate the gains in quality of our approach in the evaluation section.

For tracking and optical flow estimation, L2-based time-cycle losses (Wang et al., 2019) were proposed to constrain motions and tracked correspondences using symmetric video inputs.

By optimizing indirectly via motion compensation or tracking, this loss improves the accuracy of the results.

For video generation, we propose a PP loss that also makes use of symmetric sequences.

However, we directly constrain the PP loss via the generated video content, which successfully improves the long-term temporal consistency in the video results.

Generative Network Before explaining the temporal self-supervision in more detail, we outline the generative model to be supervised.

Our generator networks produce image sequences in a frame-recurrent manner with the help of a recurrent generator G and a flow estimator F .

We follow previous work , where G produces output g t in the target domain B from conditional input frame a t from the input domain A, and recursively uses the previous generated output g t−1 .

F is trained to estimate the motion v t between a t−1 and a t , which is then used as a motion compensation that aligns g t−1 to the current frame.

This procedure, also shown in Fig. 2a ), can be summarized as: g t = G(a t , W (g t−1 , v t )), where v t = F(a t−1 , a t ) and W is the warping operation.

While one generator is enough to map data from A to B for paired tasks such as VSR, unpaired generation requires a second generator to establish cycle consistency. (Zhu et al., 2017 ).

In the UVT task, we use two recurrent generators, mapping from domain A to B and back.

As shown in Fig. 2b , vt)) to enforce consistency.

A ResNet architecture is used for the VSR generator G and a encoder-decoder structure is applied to UVT generators and F .

We intentionally keep generators simple and in line with previous work, in order to demonstrate the advantages of the temporal self-supervision that we will explain in the following paragraphs.

Spatio-Temporal Adversarial Self-Supervision The central building block of our approach is a novel spatio-temporal discriminator D s,t that receives triplets of frames.

This contrasts with typically used spatial discriminators which supervise only a single image.

By concatenating multiple adjacent frames along the channel dimension, the frame triplets form an important building block for learning because they can provide networks with gradient information regarding the realism of spatial structures as well as short-term temporal information, such as first-and second-order time derivatives.

We propose a D s,t architecture, illustrated in Fig. 3 and Fig. 4 , that primarily receives two types of triplets: three adjacent frames and the corresponding warped ones.

We warp later frames backward and previous ones forward.

While original frames contain the full spatio-temporal infor-mation, warped frames more easily yield temporal information with their aligned content.

For the input variants we use the following notation: Ig = {gt−1, gt, gt+1}, I b = {bt−1, bt, bt+1};

For VSR tasks, D s,t should guide the generator to learn the correlation between LR inputs and highresolution (HR) targets.

Therefore, three LR frames I a = {a t−1 , a t , a t+1 } from the input domain are used as a conditional input.

The input of D s,t can be summarized as I b s,t = {I b , I wb , I a } labelled as real and the generated inputs I g s,t = {I g , I wg , I a } labelled as fake.

In this way, the conditional D s,t will penalize G if I g contains less spatial details or unrealistic artifacts according to I a , I b .

At the same time, temporal relationships between the generated images I wg and those of the ground truth I wb should match.

With our setup, the discriminator profits from the warped frames to classify realistic and unnatural temporal changes, and for situations where the motion estimation is less accurate, the discriminator can fall back to the original, i.e. not warped, images.

For UVT tasks, we demonstrate that the temporal cycleconsistency between different domains can be established using the supervision of unconditional spatio-temporal discriminators.

This is in contrast to previous work which focuses on the generative networks to form spatio-temporal cycle links.

Our approach actually yields improved results, as we will show below, and Fig. 1 shows a preview of the quality that can be achieved using spatio-temporal discriminators.

In practice, we found it crucial to ensure that generators first learn reasonable spatial features, and only then improve their temporal correlation.

Therefore, different to the D s,t of VST that always receives 3 concatenated triplets as an input, the unconditional D s,t of UVT only takes one triplet at a time.

Focusing on the generated data, the input for a single batch can either be a static triplet of Isg = {gt, gt, gt}, the warped triplet I wg , or the original triplet I g .

The same holds for the reference data of the target domain, as shown in Fig. 4 .

With sufficient but complex information contained in these triplets, transition techniques are applied so that the network can consider the spatio-temporal information step by step, i.e., we initially start with 100% static triplets I sg as the input.

Then, over the course of training, 25% of them transition to I wg triplets with simpler temporal information, with another 25% transition to I g afterwards, leading to a (50%,25%,25%) distribution of triplets.

Details of the transition calculations are given in Appendix D. Here, the warping is again performed via F .

While non-adversarial training typically employs loss formulations with static goals, the GAN training yields dynamic goals due to discriminative networks discovering the learning objectives over the course of the training run.

Therefore, their inputs have strong influence on the training process and the final results.

Modifying the inputs in a controlled manner can lead to different results and substantial improvements if done correctly, as will be shown in Sec. 4.

Although the proposed concatenation of several frames seems like a simple change that has been used in a variety of projects, it is an important operation that allows discriminators to understand spatio-temporal data distributions.

As will be shown below, it can effectively reduce temporal problems encountered by spatial GANs.

While L 2 −based temporal losses are widely used in the field of video generation, the spatiotemporal adversarial loss is crucial for preventing the inference of blurred structures in multi-modal data-sets.

Compared to GANs using multiple discriminators, the single D s,t network can learn to balance the spatial and temporal aspects from the reference data and avoid inconsistent sharpness as well as overly smooth results.

Additionally, by extracting shared spatio-temporal features, it allows for smaller network sizes.

Self-Supervision for Long-term Temporal Consistency When relying on a previous output as input, i.e., for frame-recurrent architectures, generated structures easily accumulate frame by frame.

In an adversarial training, generators learn to heavily rely on previously generated frames and can easily converge towards strongly reinforcing spatial features over longer periods of time.

For videos, this especially occurs along directions of motion, and these solutions can be seen as a special form of temporal mode collapse.

We have noticed this issue in a variety of recurrent architectures, examples are shown in Fig. 5 a) and the Dst in Fig. 1 .

While this issue could be alleviated by training with longer sequences, we generally want generators to be able to work with sequences of arbitrary length for inference.

To address this inherent problem of recurrent generators, we propose a new Result trained with PP loss.

These artifacts are removed successfully for the latter.

c) The ground-truth image.

With our PP loss (shown on the right), the L 2 distance between gt and g t is minimized to remove drifting artifacts and improve temporal coherence.

bi-directional "Ping-Pong" loss.

For natural videos, a sequence with forward order as well as its reversed counterpart offer valid information.

Thus, from any input of length n, we can construct a symmetric PP sequence in form of a 1 , ...a n−1 , a n , a n−1 , ...a 1 as shown in Fig. 5 .

When inferring this in a frame-recurrent manner, the generated result should not strengthen any invalid features from frame to frame.

Rather, the result should stay close to valid information and be symmetric, i.e., the forward result gt = G(at, gt−1) and the one generated from the reversed part, g t = G(at, g t+1 ), should be identical.

Based on this observation, we train our networks with extended PP sequences and constrain the generated outputs from both "legs" to be the same using the loss:

Note that in contrast to the generator loss, the L 2 norm is a correct choice here: We are not faced with multi-modal data where an L 2 norm would lead to undesirable averaging, but rather aim to constrain the recurrent generator to its own, unique version over time.

The PP terms provide constraints for short term consistency via gn−1 − gn−1 2 , while terms such as g1 − g1 2 prevent long-term drifts of the results.

As shown in Fig. 5 (b), this PP loss successfully removes drifting artifacts while appropriate high-frequency details are preserved.

In addition, it effectively extends the training data set, and as such represents a useful form of data augmentation.

A comparison is shown in Appendix E to disentangle the effects of the augmentation of PP sequences and the temporal constrains.

The results show that the temporal constraint is the key to reliably suppressing the temporal accumulation of artifacts, achieving consistency, and allowing models to infer much longer sequences than seen during training.

Perceptual Loss Terms As perceptual metrics, both pre-trained NNs (Johnson et al., 2016; Wang et al., 2018a) and in-training discriminators (Xie et al., 2018) were successfully used in previous work.

Here, we use feature maps from a pre-trained VGG-19 network (Simonyan & Zisserman, 2014) , as well as D s,t itself.

In the VSR task, we can encourage the generator to produce features similar to the ground truth ones by increasing the cosine similarity between their feature maps.

In UVT tasks without paired ground truth data, we still want the generators to match the distribution of features in the target domain.

Similar to a style loss in traditional style transfer (Johnson et al., 2016) , we here compute the D s,t feature correlations measured by the Gram matrix instead.

The feature maps of D s,t contain both spatial and temporal information, and hence are especially well suited for the perceptual loss.

We now explain how to integrate the spatio-temporal discriminator into the paired and unpaired tasks.

We use a standard discriminator loss for the D s,t of VSR and a least-square discriminator loss for the D s,t of UVT.

Correspondingly, a non-saturated L adv is used for the G and F of VSR, and a least-squares one is used for the UVT generators.

As summarized in Table 1 , G and F are trained with the mean squared loss L content , adversarial losses L adv , perceptual losses L φ , the PP loss L PP , and a warping loss L warp , where again g, b and Φ stand for generated samples, ground truth images and feature maps of VGG-19 or D s,t .

We only show losses for the mapping from A to B for UVT tasks, as the backward mapping simply mirrors the terms.

We refer to our full model for both tasks as TecoGAN below.

1 Training parameters and details are given in Appendix G.

In the following, we illustrate the effects of temporal supervision using two ablation studies.

In the first one, models trained with ablated loss functions show how L adv and L PP change the overall learning objectives.

Next, full UVT models are trained with different D s,t inputs.

This highlights how differently the corresponding discriminators converge to different spatio-temporal equilibriums, and the general importance of providing suitable data distributions from the target domain.

While we provide qualitative and quantitative evaluations in the following, we also refer the reader to our supplemental html document 2 , with video clips that more clearly highlight the temporal differences.

Loss Ablation Study Below we compare variants of our full TecoGAN model to EnhanceNet (ENet) (Sajjadi et al., 2017) , FRVSR (Sajjadi et al., 2018) , and DUF (Jo et al., 2018) for VSR, and CycleGAN (Zhu et al., 2017) and RecycleGAN (Bansal et al., 2018) for UVT.

Specifically, ENet and CycleGAN represent state-of-the-art single-image adversarial models without temporal information, FRVSR and DUF are state-of-the-art VSR methods without adversarial losses, and RecycleGAN is a spatial adversarial model with a prediction network learning the temporal evolution.

For VSR, we first train a DsOnly model that uses a frame-recurrent G and F with a VGG-19 loss and only the regular spatial discriminator.

Compared to ENet, which exhibits strong incoherence due to the lack of temporal information, DsOnly improves temporal coherence thanks to the framerecurrent connection, but there are noticeable high-frequency changes between frames.

The temporal profiles of DsOnly in Fig. 6 and 8, correspondingly contain sharp and broken lines.

When adding a temporal discriminator in addition to the spatial one (DsDt), this version generates more coherent results, and its temporal profiles are sharp and coherent.

However, DsDt often produces the drifting artifacts discussed in Sec. 3, as the generator learns to reinforce existing details from previous frames to fool D s with sharpness, and satisfying D t with good temporal coherence in the form of persistent detail.

While this strategy works for generating short sequences during training, the strengthening effect can lead to very undesirable artifacts for long-sequence inferences.

By adding the self-supervision for long-term temporal consistency L pp , we arrive at the DsDtPP model, which effectively suppresses these drifting artifacts with an improved temporal coherence.

In Fig. 6 and Fig. 8 , DsDtPP results in continuous yet detailed temporal profiles without streaks from temporal drifting.

Although DsDtPP generates good results, it is difficult in practice to balance the generator and the two discriminators.

The results shown here were achieved only after numerous runs manually tuning the weights of the different loss terms.

By using the proposed D s,t discriminator instead, we get a first complete model for our method, denoted as TecoGAN .

This network is trained with a discriminator that achieves an excellent quality with an effectively halved network size, as illustrated on the right of Fig. 7 .

The single discriminator correspondingly leads to a significant reduction in resource usage.

Using two discriminators requires ca.

70% more GPU memory, and leads to a reduced training performance by ca.

20%.

The TecoGAN model yields similar perceptual and temporal quality to DsDtPP with a significantly faster and more stable training.

Since the TecoGAN model requires less training resources, we also trained a larger generator with 50% more weights.

In the following we will focus on this larger single-discriminator architecture with PP loss as our full TecoGAN model for VSR.

Compared to the TecoGAN model, it can generate more details, and the training process is more stable, indicating that the larger generator and D s,t are more evenly balanced.

Result images and temporal profiles are shown in Fig. 6 and Fig. 8 .

Video results are shown in Sec. 4 of the supplemental material.

We also carry out a similar ablation study for the UVT task.

Again, we start from a single-image GAN-based model, a CycleGAN variant which already has two pairs of spatial generators and discriminators.

Then, we train the DsOnly variant by adding flow estimation via F and extending the spatial generators to frame-recurrent ones.

By augmenting the two discriminators to use the triplet inputs proposed in Sec. 3, we arrive at the Dst model with spatio-temporal discriminators, which

does not yet use the PP loss.

Although UVT tasks are substantially different from VSR tasks, the comparisons in Fig. 1 and Sec. 4.6 of our supplemental material yield similar conclusions.

In these tests, we use renderings of 3D fluid simulations of rising smoke as our unpaired training data.

These simulations are generated with randomized numerical simulations using a resolution of 64 3 for domain A and 256 3 for domain B, and both are visualized with images of size 256 2 .

Therefore, video translation from domain A to B is a tough task, as the latter contains significantly more turbulent and small-scale motions.

With no temporal information available, the CycleGAN variant generates HR smoke that strongly flickers.

The DsOnly model offers better temporal coherence by relying on its frame-recurrent input, but it learns a solution that largely ignores the current input and fails to keep reasonable spatio-temporal cycle-consistency links between the two domains.

On the contrary, our D s,t enables the Dst model to learn the correlation between the spatial and temporal aspects, thus improving the cycle-consistency.

However, without L pp , the Dst model (like the DsDt model of VSR) reinforces detail over time in an undesirable way.

This manifests itself as inappropriate smoke density in empty regions.

Using our full TecoGAN model which includes L pp , yields the best results, with detailed smoke structures and very good spatio-temporal cycle-consistency.

For comparison, a DsDtPP model involving a larger number of separate networks, i.e. four discriminators, two frame-recurrent generators and the F , is trained.

By weighting the temporal adversarial losses from Dt with 0.3 and the spatial ones from Ds with 0.5, we arrived at a balanced training run.

Although this model performs similarly to the TecoGAN model on the smoke dataset, the proposed spatio-temporal D s,t architecture represents a more preferable choice in practice, as it learns a natural balance of temporal and spatial components by itself, and requires fewer resources.

Continuing along this direction, it will be interesting future work to evaluate variants, such as a shared D s,t for both domains, i.e. a multi-class classifier network.

Besides the smoke dataset, an ablation study for the Obama and Trump dataset from Fig. 1 shows a very similar behavior, as can be seen in the supplemental material.

Spatio-temporal Adversarial Equilibriums Our evaluation so far highlights that temporal adversarial learning is crucial for achieving spatial detail that is coherent over time for VSR, and for enabling the generators to learn the spatio-temporal correlation between domains in UVT.

Next, we will shed light on the complex spatio-temporal adversarial learning objectives by varying the information provided to the discriminator network.

The following tests D s,t networks that are identical apart from changing inputs, and we focus on the smoke dataset.

In order to learn the spatial and temporal features of the target domain as well as their correlation, the simplest input for D s,t consists of only the original, unwarped triplets, i.e. {I g or I b }.

Using these, we train a baseline model, which yields a sub-optimal quality: it lacks sharp spatial structures, and contains coherent but dull motions.

Despite containing the full information, these input triplets prevent D s,t from providing the desired supervision.

For paired video translation tasks, the vid2vid network achieves improved temporal coherence by using a video discriminator to supervise the output sequence conditioned with the ground-truth motion.

With no ground-truth data available, we train a vid2vid variant by using the estimated motions and original triplets, i.e {I g + F (g t−1 , g t ) + F (g t+1 , g t ) or I b + F (b t−1 , b t ) + F (b t+1 , b t )}, as the input for D s,t .

However, the result do not significantly improve.

The motions are only partially reliable, and hence don't help for the difficult unpaired translation task.

Therefore, the discriminator still fails to fully correlate spatial and temporal features.

We then train a third model, concat, using the original triplets and the warped ones, i.e. {I g +I wg or I b +I wb }.

In this case, the model learns to generate more spatial details with a more vivid motion.

I.e., the improved temporal information from the warped triplets gives the discriminator important cues.

However, the motion still does not fully resemble the target domain.

We arrive at our final TecoGAN model for UVT by controlling the composition of the input data: as outlined above, we first provide only static triplets {I sg or I sb }, and then apply the transitions of warped triplets {I wg or I wb }, and original triplets {I g or I b } over the course of training.

In this way, the network can first learn to extract spatial features, and build on them to establish temporal features.

Finally, discriminators learn features about the correlation of spatial and temporal content by analyzing the original triplets, and provide gradients such that the generators learn to use the motion information from the input and establish a correlation between the motions in the two unpaired domains.

Consequently, the discriminator, despite receiving only a single triplet at once, can guide the generator to produce detailed structures that move coherently.

Video comparisons are shown in Sec 5.

of the supplemental material.

Results and Metric Evaluation While the visual results discussed above provide a first indicator of the quality our approach achieves, quantitative evaluations are crucial for automated evaluations across larger numbers of samples.

Below we focus on the VSR task as ground-truth data is available in this case.

We conduct user studies and present evaluations of the different models w.r.t.

established spatial metrics.

We also motivate and propose two novel temporal metrics to quantify temporal coherence.

A visual summary is shown in Fig. 7 .

For evaluating image SR, Blau & Michaeli (2018) demonstrated that there is an inherent trade-off between the perceptual quality of the result and the distortion measured with vector norms or lowlevel structures such as PSNR and SSIM.

On the other hand, metrics based on deep feature maps such as LPIPS (Zhang et al., 2018) can capture more semantic similarities.

We measure the PSNR and LPIPS using the Vid4 scenes.

With a PSNR decrease of less than 2dB over DUF which has twice the model size of ours, TecoGAN outperforms all methods by more than 40% on LPIPS.

While traditional temporal metrics based on vector norm differences of warped frames, e.g. T-diff, can be easily deceived by very blurry results, e.g. bi-cubic interpolated ones, we propose to use a tandem of two new metrics, tOF and tLP, to measure the consistence over time.

tOF measures the pixel-wise difference of motions estimated from sequences, and tLP measures perceptual changes over time using deep feature map:

where OF represents an optical flow estimation with LucasKanade (1981) and LP is the perceptual LPIPS metric.

In tLP, the behavior of the reference is also considered, as natural videos exhibit a certain degree of changes over time.

In conjunction, both pixel-wise differences and perceptual changes are crucial for quantifying realistic temporal coherence.

While they could be combined into a single score, we list both measurements separately, as their relative importance could vary in different application settings.

Our evaluation with these temporal metrics in Table 2 shows that all temporal adversarial models outperform spatial adversarial ones, and the full TecoGAN model performs very well: With a large amount of spatial detail, it still achieves good temporal coherence, on par with non-adversarial methods such as DUF and FRVSR.

For VSR, we have confirmed these automated evaluations with several user studies.

Across all of them, we find that the majority of the participants considered the TecoGAN results to be closest to the ground truth.

For the UVT tasks, where no ground-truth data is available, we can still evaluate tOF and tLP metrics by comparing the motion and the perceptual changes of the output data w.r.t.

the ones from the input data , i.e., tOF = OF (at−1, at) − OF (g Table 3 , although it is worth to point out that the tOF is less informative in this case, as the motion in the target domain is not necessarily pixel-wise aligned with the input.

Overall, TecoGAN achieves good tLP scores thanks to its temporal coherence, on par with RecycleGAN, and its spatial detail is on par with CycleGAN.

As for VSR, a perceptual evaluation by humans in the right column of Table 3 confirms our metric evaluations for the UVT task (details in Appendix C).

In paired as well as unpaired data domains, we have demonstrated that it is possible to learn stable temporal functions with GANs thanks to the proposed discriminator architecture and PP loss.

We have shown that this yields coherent and sharp details for VSR problems that go beyond what can be achieved with direct supervision.

In UVT, we have shown that our architecture guides the training process to successfully establish the spatio-temporal cycle consistency between two domains.

These results are reflected in the proposed metrics and user studies.

While our method generates very realistic results for a wide range of natural images, our method can generate temporally coherent yet sub-optimal details in certain cases such as under-resolved faces and text in VSR, or UVT tasks with strongly different motion between two domains.

For the latter case, it would be interesting to apply both our method and motion translation from concurrent work (Chen et al., 2019) .

This can make it easier for the generator to learn from our temporal self supervision.

In our method, the interplay of the different loss terms in the non-linear training procedure does not provide a guarantee that all goals are fully reached every time.

However, we found our method to be stable over a large number of training runs, and we anticipate that it will provide a very useful basis for wide range of generative models for temporal data sets.

In the following, we first provide qualitative analysis(Appendix A) using multiple results that are mentioned but omitted in our main document due to space constraints.

We then explain details of the metrics and present the quantitative analysis based on them(Appendix B).

The conducted user studies are in support of our TecoGAN network and proposed temporal metrics (Appendix C).

Then, we give technical details of our spatio-temporal discriminator (Sec. D), details of network architectures and training parameters (Appendix F, Appendix G).

In the end, we discuss the performance of our approach in Appendix H.

For the VSR task, we test our model on a wide range of video data, including the generally used Vid4 dataset shown in Fig. 8 and 12 , detailed scenes from the movie Tears of Steel (ToS, 2011) shown in Fig. 12 , and others shown in Fig. 9 .

As mentioned in the main document, the TecoGAN model is trained with down-sampled inputs and it can similarly work with original images that were not down-sampled or filtered, such as a data-set of real-world photos (Liao et al., 2015) .

In Fig. 10 , we compared our results to two other methods (Liao et al., 2015; Tao et al., 2017 ) that have used the same dataset.

With the help of adversarial learning, our model is able to generate improved and realistic details in down-sampled images as well as captured images.

For UVT tasks, we train models for Obama and Trump translations, LR-and HR-smoke simulation translations, as well as translations between smoke simulations and real-smoke captures.

While smoke simulations usually contain strong numerical viscosity with details limited by the simulation resolution, the real smoke, captured using the setup from Eckert et al. (2018) , contains vivid fluid motions with many vortices and high-frequency details.

As shown in Fig. 11 , our method can be used to narrow the gap between simulations and real-world phenomenon.

Spatial Metrics We evaluate all VSR methods with PSNR together with the human-calibrated LPIPS metric (Zhang et al., 2018) .

While higher PSNR values indicate a better pixel-wise accuracy, lower LPIPS values represent better perceptual quality and closer semantic similarity.

Mean values of the Vid4 scenes Liu & Sun (2011) are shown on the top of Table 4 .

Trained with direct vector norms losses, FRVSR and DUF achieve high PSNR scores.

However, the undesirable smoothing induced by these losses manifests themselves in larger LPIPS distances.

ENet, on the other hand, with no information from neighboring frames, yields the lowest PSNR and achieves an LPIPS score that is only slightly better than DUF and FRVSR.

TecoGAN model with adversarial training achieves an excellent LPIPS score, with a PSNR decrease of less than 2dB over DUF, which is very reasonable, since PSNR and perceptual quality were shown to be anti-correlated (Blau & Michaeli, 2018) , especially in regions where PSNR is very high.

Based on good perceptual quality and reasonable pixel-wise accuracy, TecoGAN outperforms all other methods by more than 40% for LPIPS.

Temporal Metrics For both VSR and UVT, evaluating temporal coherence without ground-truth motion is a very challenging problem.

The metric T-diff = gt − W (gt−1, vt) 1 was used by Chen et al. (2017) as a rough assessment of temporal differences.

As shown on bottom of Table 4 , T-diff, due to its local nature, is easily deceived by blurry method such as the bi-cubic interrelation and can not correlate well with visual assessments of coherence.

By measuring the pixel-wise motion difference using tOF in together with the perceptual changes over time using tLP, we show the temporal evaluations for the VSR task in the middle of Table 4 .

Not surprisingly, the results of ENet show larger errors for all metrics due to their strongly flickering content.

Bi-cubic up-sampling, DUF, and FRVSR achieve very low T-diff errors due to their smooth results, representing an easy, but undesirable avenue for achieving coherency.

However, the overly smooth changes of the former two are identified by the tLP scores.

While our DsOnly model generates sharper results at the expense of temporal coherence, it still outperforms ENet there.

By adding temporal information to discriminators, our DsDt, DsDt+PP, TecoGAN and TecoGAN improve in terms of temporal metrics.

Especially the full TecoGAN model stands out here.

For the UVT tasks, temporal motions are evaluated by comparing to the input sequence.

With sharp spatial features and coherent motion, TecoGAN outperforms previous work on the Obama&Trump dataset, as shown in Table 3 .

Since temporal metrics can trivially be reduced for blurry image content, we found it important to evaluate results with a combination of spatial and temporal metrics.

Given that perceptual metrics are already widely used for image evaluations, we believe it is the right time to consider perceptual changes in temporal evaluations, as we did with our proposed temporal coherence metrics.

Although not perfect, they are not easily deceived.

Specifically, tOF is more robust than a direct pixel-wise metric as it compares motions instead of image content.

In the supplemental material, we visualize the motion difference and it can well reflect the visual inconsistencies.

On the other hand, we found that our calculation of tLP is a general concept that can work reliably with different perceptual metric: When repeating the tLP evaluation with the PieAPP metric (Prashnani et al., 2018) instead of LP , i.e., tPieP = f (yt−1, yt) − f (gt−1, gt) 1 , where f(·) indicates the perceptual error function of PieAPP, we get close to identical results, listed in Fig. 13 .

The conclusions from tPieP also closely match the LPIPS-based evaluation: our network architecture can generate realistic and temporally coherent detail, and the metrics we propose allow for a stable, automated evaluation of the temporal perception of a generated video sequence.

Besides the previously evaluated the Vid4 dataset, with graphs shown in Fig. 14, 15 , we also get similar evaluation results on the Tears of Steel data-sets (room, bridge, and face, in the following referred to as ToS scenes) and corresponding results are shown in Table 5 and Fig. 16 .

In all tests, we follow the procedures of previous work (Jo et al., 2018; to make the outputs of all methods comparable, i.e., for all result images, we first exclude spatial borders with a distance of 8 pixels to the image sides, then further shrink borders such that the LR input image is divisible by 8 and for spatial metrics, we ignore the first two and the last two frames, while for temporal metrics, we ignore first three and last two frames, as an additional previous frame is required for inference.

In the following, we conduct user studies for the Vid4 scenes.

By comparing the user study results and the metric breakdowns shown in Table 4 , we found our metrics to reliably capture the human temporal perception, as shown in Appendix C.

We conduct several user studies for the VSR task using five different methods, namely bi-cubic interpolation, ENet, FRVSR, DUF and our TecoGAN.

The established 2AFC design (Fechner & Wundt, 1889; Um et al., 2017) is applied, i.e., participants have a pair-wise choice, with the groundtruth video shown as reference.

One example can be seen in Fig. 17 .

The videos are synchronized and looped until user made the final decision.

With no control to stop videos, users Participants cannot stop or influence the playback, and hence can focus more on the whole video, instead of specific spatial details.

Videos positions (left/A or right/B) are randomized.

After collecting 1000 votes from 50 users for every scene, i.e. twice for all possible pairs (5 × 4/2 = 10 pairs), we follow common procedure and compute scores for all models with the Bradley-Terry model (1952) .

The outcomes for the Vid4 scenes can be seen in Fig. 18 (overall scores are listed in Table 2 of the main document).

From the Bradley-Terry scores for the Vid4 scenes we can see that the TecoGAN model performs very well, and achieves the first place in three cases, as well as a second place in the walk scene.

The latter is most likely caused by the overall slightly smoother images of the walk scene, in conjunction with the presence of several human faces, where our model can lead to the generation of unexpected details.

However, overall the user study shows that users preferred the TecoGAN output over the other two deep-learning methods with a 63.5% probability.

This result also matches with our metric evaluations.

In Table 4 , while TecoGAN achieves spatial (LPIPS) improvements in all scenes, DUF and FRVSR are not far behind in the walk scene.

In terms of temporal metrics tOF and tLP, TecoGAN achieves similar or lower scores compared to FRVSR and DUF for calendar, foliage and city scenes.

The lower performance of our model for the walk scene is likewise captured by higher tOF and tLP scores.

Overall, the metrics confirm the performance of our TecoGAN approach and match the results of the user studies, which indicate that our proposed temporal metrics successfully capture important temporal aspects of human perception.

For UVT tasks which have no ground-truth data, we carried out two sets of user studies: One uses an arbitrary sample from the target domain as the reference and the other uses the actual input from the source domain as the reference.

On the Obama&Trump data-sets, we evaluate results from CycleGAN, RecycleGAN, and TecoGAN following the same modality, i.e. a 2AFC design with 50 users for each run.

E.g., on the left of Fig. 19 , users evaluate the generated Obama in reference with the input Trump on the y-axis, while an arbitrary Obama video is shown as the reference on the x-axis.

Effectively, the y-axis is more important than the x-axis as it indicates whether the translated result preserves the original expression.

A consistent ranking of TecoGAN > RecycleGAN > CycleGAN is shown on the y-axis with clear separations, i.e. standard errors don't overlap.

The x-axis indicates whether the inferred result matches the general spatio-temporal content of the target domain.

Our TecoGAN model also receives the highest scores here, although the responses are slightly more spread out.

On the right of Fig. 19 , we summarize both studies in a single graph highlighting that the TecoGAN model is consistently preferred by the participants of our user studies.

Figure 20: Near image boundaries, flow estimation is less accurate and warping often fails to align well.

First two columns show original and warped frames and the third one shows differences after warping (ideally all black).

The top row shows things move into the view with problems near lower boundaries, while the second row has objects moving out of the view.

with the help of the flow estimation network F. However, at the boundary of images, the output of F is usually less accurate due to the lack of reliable neighborhood information.

There is a higher chance that objects move into the field of view, or leave suddenly, which significantly affects the images warped with the inferred motion.

An example is shown in Fig. 20

Figure 18 : Tables and bar graphs to warp the previous frame in accordance with the detail that G can synthesize.

However, F does not adjust the motion estimation only to reduce the adversarial loss.

Curriculum Learning for UVT Discriminators As mentioned in the main part, we train the UVT D s,t with 100% spatial triplets at the very beginning.

During training, 25% of them gradually transfer into warped triplets and another 25% transfer into original triplets.

The transfer of the warped triplets can be represented as: (1−α)I cg +αI wg , with α growing form 0 to 1.

For the original triplets, we additionally fade the "warping" operation out by using (1 − α)I cg + α{W (g t−1 , v t * β), g t , W (g t+1 , v t * β)}, again with α growing form 0 to 1 and β decreasing from 1 to 0.

We found this smooth transition to be helpful for a stable training.

Since training with sequences of arbitrary length is not possible with current hardware, problems such as the streaking artifacts discussed above generally arise for recurrent models.

In the proposed PP loss, both the Ping-Pang data augmentation and the temporal consistency constraint contribute to solving these problems.

In order to show their separated contributions, we trained another TecoGAN variant that only employs the data augmentation without the constraint (i.e., λ p = 0 in Table 1

In this section, we use the following notation to specify all network architectures used: conc() represents the concatenation of two tensors along the channel dimension; C/CT (input, kernel size, output channel, stride size) stands for the convolution and transposed convolution operation, respectively; "+" denotes element-wise addition; BilinearUp2 up-samples input tensors by a factor of 2 using bi-linear interpolation; BicubicResize4(input) increases the resolution of the input tensor to 4 times higher via bi-cubic up-sampling; Dense(input, output size) is a densely-connected layer, which uses Xavier initialization for the kernel weights.

The architecture of our VSR generator G is:

conc(x t , W (g t−1 , v t )) → l in ; C(l in , 3, 64, 1), ReLU → l 0 ; ResidualBlock(l i ) → l i+1 with i = 0, ..., n − 1; CT (l n , 3, 64, 2), ReLU → l up2 ; CT (l up2 , 3, 64, 2), ReLU → l up4 ; C(l up4 , 3, 3, 1), ReLU → l res ; BicubicResize4(x t ) + l res → g t .

In TecoGAN , there are 10 sequential residual blocks in the generator ( l n = l 10 ), while the TecoGAN generator has 16 residual blocks ( l n = l 16 ).

Each ResidualBlock(l i ) contains the following operations: C(l i , 3, 64, 1), ReLU → r i ; C(r i , 3, 64, 1) + l i → l i+1 .

The VSR D s,t 's architecture is: IN g s,t or IN y s,t → l in ; C(l in , 3, 64, 1), Leaky ReLU → l 0 ; C(l 0 , 4, 64, 2), BatchNorm, Leaky ReLU → l 1 ; C(l 1 , 4, 64, 2), BatchNorm, Leaky ReLU → l 2 ; C(l 2 , 4, 128, 2), BatchNorm, Leaky ReLU → l 3 ; C(l 3 , 4, 256, 2), BatchNorm, Leaky ReLU → l 4 ;

Dense(l 4 , 1), sigmoid → l out .

VSR discriminators used in our variant models, DsDt, DsDtPP and DsOnly, have a similar architecture as D s,t .

They only differ in terms of their inputs.

The flow estimation network F has the following architecture:

conc(x t , x t−1 ) → l in ; C(l in , 3, 32, 1), Leaky ReLU → l 0 ; C(l 0 , 3, 32, 1), Leaky ReLU, MaxPooling → l 1 ; C(l 1 , 3, 64, 1), Leaky ReLU → l 2 ; C(l 2 , 3, 64, 1), Leaky ReLU, MaxPooling → l 3 ; C(l 3 , 3, 128, 1), Leaky ReLU → l 4 ; C(l 4 , 3, 128, 1), Leaky ReLU, MaxPooling → l 5 ; C(l 5 , 3, 256, 1), Leaky ReLU → l 6 ; C(l 6 , 3, 256, 1), Leaky ReLU, BilinearUp2 → l 7 ; C(l 7 , 3, 128, 1), Leaky ReLU → l 8 ; For all UVT tasks, we use a learning rate of 10 −4 to train the first 90k batches and the last 10k batches are trained with the learning rate decay from 10 −4 to 0.

Images of the input domain are cropped into a size of 256 × 256 when training, while the original size is 288 × 288.

While the Additional training parameters are also listed in Table 6 .

For UVT, L content and L φ are only used to improve the convergence of the training process.

We fade out the L content in the first 10k batches and the L φ is used for the first 80k and faded out in last 20k.

TecoGAN is implemented in TensorFlow.

While generator and discriminator are trained together, we only need the trained generator network for the inference of new outputs after training, i.e., the whole discriminator network can be discarded.

We evaluate the models on a Nvidia GeForce GTX 1080Ti GPU with 11G memory, the resulting VSR performance for which is given in Table 2 .

The VSR TecoGAN model and FRVSR have the same number of weights (843587 in the SRNet, i.e. generator network, and 1.7M in F), and thus show very similar performance characteristics with around 37 ms spent for one frame.

The larger VSR TecoGAN model with 1286723 weights in the generator is slightly slower than TecoGAN , spending 42 ms per frame.

In the UVT task, generators spend around 60 ms per frame with a size of 512 × 512.

However, compared with the DUF model, with has more than 6 million weights in total, the TecoGAN performance significantly better thanks to its reduced size.

@highlight

We propose temporal self-supervisions for learning stable temporal functions with GANs.