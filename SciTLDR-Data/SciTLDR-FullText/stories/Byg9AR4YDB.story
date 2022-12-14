Cell-cell interactions have an integral role in tumorigenesis as they are critical in governing immune responses.

As such, investigating specific cell-cell interactions has the potential to not only expand upon the understanding of tumorigenesis, but also guide clinical management of patient responses to cancer immunotherapies.

A recent imaging technique for exploring cell-cell interactions, multiplexed ion beam imaging by time-of-flight (MIBI-TOF), allows for cells to be quantified in 36 different protein markers at sub-cellular resolutions in situ as high resolution multiplexed images.

To explore the MIBI images, we propose a GAN for multiplexed data with protein specific attention.

By conditioning image generation on cell types, sizes, and neighborhoods through semantic segmentation maps, we are able to observe how these factors affect cell-cell interactions simultaneously in different protein channels.

Furthermore, we design a set of metrics and offer the first insights towards cell spatial orientations, cell protein expressions, and cell neighborhoods.

Our model, cell-cell interaction GAN (CCIGAN), outperforms or matches existing image synthesis methods on all conventional measures and significantly outperforms on biologically motivated metrics.

To our knowledge, we are the first to systematically model multiple cellular protein behaviors and interactions under simulated conditions through image synthesis.

Cell-cell interactions within the tumor microenvironment have been implicated in many facets of cancer pathogenesis and treatment.

Most prominently, tumor cell evasion of immune surveillance (Jiang et al., 2019) , tumor metastasis (Nishida-Aoki & Gujral, 2019) , and efficacy of cancer immunotherapies have all been closely linked to the relationships between immune and cancer cells.

These types of cell-cell relationships are generally governed by the interactions of cell surface proteins which drive cell behavior, gene expression, and survival.

One of the most prominent examples of cellular proteins influencing disease progression is the case of PD-1/PD-L1.

PD-L1 is a protein often overexpressed on tumor cells and has the capacity to bind PD-1 on local T cells to downregulate their anti-tumor immune responses (Iwai et al., 2002) .

Antibodies which interrupt the interaction of PD-L1 and PD-1 and allow the immune system to attack tumor cells, have become clinically influential treatments for a variety of cancers (Pardoll, 2012) .

This example highlights the value of accurately predicting cellular protein patterns which play key roles in disease processes.

Exploring protein localizations in a multi-cellular system represents a challenge for which deep learning models are uniquely suited.

However, to our knowledge no image-based deep generative models have utilized semantic image synthesis to produce accurate predictions of these biological phenomena.

A meaningful exploration of cell-cell interactions, particularly in the tumor microenvironment, requires a thorough understanding of the proteins expressed on and within a cell and its neighborhood.

Multiplexed ion beam imaging by time-of-flight (MIBI-TOF) represents a novel technology that can accurately quantify and spatially resolve cellular protein expressions at the single cell level within tissue samples.

Given a tissue sample that is first stained with protein-specific antibodies tethered to elemental metals, MIBI-TOF bombards the sample with atomic ions (i.e. O + 2 ) from a primary ion beam.

This causes the release of elemental isotopes and tissue-specific material which can be quantified in a mass spectrometer (Angelo et al., 2014) .

The cellular proteins characterized by this technique indicate specific cell types (i.e. immune cells, tumor cells), cell status (i.e. markers of proliferation), and immunomodulation.

Figure 1 (A) shows an example spatial orientation of cell types and some selected cellular protein expressions.

Here, we propose a novel protein based attention mechanism for a convolutional Generative Adversarial Network (GAN) with the capacity to provide accurate conditioned predictions of cellular protein localizations.

The model, Cell-Cell Interaction GAN (CCIGAN), learns a many to many mapping between different cell types and different protein markers.

CCIGAN is trained on semantic segmentation maps of cell tissue samples to identify and associate cell type, shape, and the identity of its cellular neighbors with cellular protein expressions for each cell.

The network was then applied to simulated segmentation maps to understand how cells vary their protein localizations.

CCIGAN demonstrates its ability to probe elements of oncologic processes by:

1.

Independently recapitulating established patterns of biological phenomena, thereby demonstrating the accuracy of its predictive power.

2.

Yielding quantitative and spatial information on specific cellular protein modulations as a result of immune cell -cancer cell interactions.

3.

Allowing for biological data to be generalized beyond traditional in vitro scenarios where specific, rare, or hypothetical cell-cell events can be posed in counterfactual scenarios and the results of their interactions predicted.

The ability for CCIGAN to generate subcellular protein predictions represents a step forward in the ability to understand cellular relationships within a microenvironment.

Rather than assessing in vitro incidence of cell interaction phenomena as is done by MIBI-TOF, CCIGAN allows for hypothetical biological situations to be generated.

Figure 1 (B) illustrates examples how a segmentation map can be directly manipulated to pose various biological scenarios.

In other words, individual cellular responses to challenge or proximity of cells of any identity can be assessed without having to seek this specific occurrence within the available biological tissue sample.

These capabilities allow the model to provide insight on hypothetical interaction events which shed insight on disease pathogenesis and which cannot be feasibly achieved by biological investigation alone.

MIBI-TOF is a labor intensive process which can make the data collection process cumbersome.

Deep learning models such as CCIGAN have the potential to be useful tools which can extend on MIBI-TOF capabilities and result in more rapidly synthesized data.

CCIGAN, once trained on MIBI-TOF data sets and properly replicating known cell-cell interaction patterns, can be used as a robust means to provide high throughput readouts on myriad single cell scenarios which may normally take many iterations of MIBI-TOF experiments to investigate.

Recent work by (Wu et al., 2019) demonstrated the utility of conditional GANs for adding a depth dimension to images; we demonstrate learning contextual cell-cell interactions, facilitating rapid hypothesis testing to assess biological environments.

This provides valuable insight into a complex system that normally would have taken resource-intensive wet lab experiments to interrogate.

We are interested in the task of generating biologically consistent expression patterns of cellular proteins given a segmentation map of cell neighborhoods.

Specifically, we want to learn a generative model that simultaneously produces high quality maps of protein expression for individual cells that are probabilistically consistent when conditioned on the same factors, e.g. similar cell neighborhoods should produce similar expression patterns.

Such generative models typically take the framework of a generative adversarial network (GAN) (Goodfellow et al., 2014; Brock et al., 2018) or a variational autoencoder (VAE) (Kingma & Welling, 2013) .

Within these generative modeling techniques, image translation focuses on learning a many to many mapping to transform data from one domain to another.

One approach is to model the translation task, known as image synthesis, with conditioning such as Pix2Pix (Isola et al., 2016) , Pix2PixHD (Wang et al., 2018a) and CycleGAN (Zhu et al., 2017) .

By using image synthesis, a model is able to distill more information from a segmentation map to various protein channels.

As MIBI-TOF measures information such as protein localizations at a subcellular resolution, framing the problem through an image synthesis framework allows a model to see how predictive neighborhoods and cell types are of a cell's phenotype.

Recently, Park et al. (2019) proposed spatial adaptive normalization (SPADE) to synthesize images using segmentation maps by learning fully convolutional normalization parameters based on their segmentation conditioning.

For each layer, semantic information is retained by allowing the network to learn from the segmentation map directly and modulate the current layer.

They demonstrate impressive quality and diversity in generated images, especially modeling objects in context.

Using SPADE normalization, CCIGAN is able to condition on surrounding cell neighbors and capture cellcell interactions in a local receptive field.

Despite image generation, there has been no work done towards understanding such generations in context, particularly in biological images.

Attention layers (Vaswani et al., 2017; Wang et al., 2018b; Zhang et al., 2018) have also been added to generative models with great success.

Self-attention was initially proposed in machine translation tasks to help model long distance dependencies that occur frequently in language (Sutskever et al., 2014) , which led to the idea of external memories as a persistent state to model long range dependencies (Sukhbaatar et al., 2015) .

While convolutional neural networks are apt at exploiting local structures of patches, they may struggle to model global structures.

For this reason visual attention allows the network to enhance activations in interesting parts of an image.

Goodfellow et al. (2014) and Brock et al. (2018) both achieve state-of-the-art unconditional generative modeling using self attention GAN (Zhang et al., 2018) .

We propose a specialized attention module conditioned on different proteins to mimic real world protein markers.

All experiments are performed on data obtained through MIBI-TOF characterized tissue samples collected from triple-negative breast cancer (TNBC) patients.

By simultaneously imaging over 36 protein markers, MIBI-TOF is able to identify cell type as well as provide detailed information of sub-cellular structure, cell neighbors, and interactions in the tumor microenvironment across these different marker settings.

Each of these markers m ??? {1, ..., M }, given as a channel taking on real values continuous in [0, 1] at each (x, y) coordinate, demarcates a different cellular subtype and furthermore, is indicative of the functional properties of a cell.

While MIBI-TOF is capable of 36 different markers, we chose markers exhibiting cellular protein localization and disregarded markers that were blank or used simply for cell-typing 1 resulting in M = 24.

A full list of these markers is given in A.8.

The MIBI-TOF data collected from the TNBC tissue allows for the differentiation between a wide variety of cell types within the samples.

For instance, cells positive for the marker CD3 could be identified as T cells, and then subdivided into cytotoxic or helper lineages by the presence of markers CD8 or CD4, respectively.

Tumor cells could be identified by markers such as pan-keratin and overexpressed beta-catenin.

Along these lines, a wide variety of cellular proteins identified by MIBI-TOF could characterize cell interactions as well as immunomodulatory processes occurring within the microenvironment.

MIBI-TOF data is fundamentally different than typical RGB images.

This poses unique challenges in image modeling and characterization.

Such challenges stem from each marker being conditionally expressed on the cell type in its respective channel m. Using a simplified 3 channel multiplex setting, a T cell expresses signals in the CD3, CD8 channels (indicators for immune cells) but not in a pankeratin channel (indicator for tumor cells).

Another problem is the sparsity of the data, meaning either some expressions for rare cell types are rarely observed or have weak signals.

For example, Figure 2 includes a segmentation of a CD8 T cell (orange, top right corner), where other models fail in generating correct CD3 expression, if at all 2 .

Lastly, the noisy nature of the data leads to inaccurate cell type classifications, creating inconsistent pairs of labels and outputs during training.

These issues make it especially difficult for an RGB multihead decoder to output multiple channels in a biologically accurate manner.

Without addressing multiplexed data, a decoder would equally attend to every location of the current latent representation, even if it is irrelevant to the current protein.

Furthermore each protein channel in the output has its own sensitivities to signal intensities and noise, suggesting each channel requires a unique prior and that equal attention would be problematic.

It follows that special care must be given towards modeling specific channels and the multiplexed nature of MIBI-TOF images.

3.2 DATA PROCESSING MIBI-TOF images are represented as a high dimensional tensor T ??? R (M,2048,2048) .

These images are then further processed at a cell by cell basis into Y ??? R (M,64,64) patches, where a cell is at the center of the patch along with its neighbors.

Next, we construct semantic segmentation maps S ??? R (C+1,64,64) , where a vector S :,i,j is one-hot encoded based on a cell type C = 17, and the C + 1-th channel denotes empty segmentation space.

The data is train-test split at a 9:1 ratio at the MIBI-TOF image level to avoid cell neighborhood bias.

We also use a synthetic test set where cells and their neighbors are sequentially modified to observe how varying cell type, position, and size affects the progressive changes in protein localizations 3 .

We use SPADE residual blocks (Park et al., 2019) as our generative backbone and DCGAN's discriminator's architecture (Figure 3 , A.1) (Radford et al., 2015) .

Park et al. (2019) have shown SPADE to be an effective way to inject conditioning into a generative model.

The SPADE normalization layer serves as a replacement for previous layer normalization techniques.

Instead of learning a universally shared per channel affine transformation, like in Batch Normalization (Ioffe & Szegedy, 2015) or Instance Normalization (Ulyanov et al., 2016) , SPADE learns to predict affine transformations based on segmentation maps; each feature is uniquely transformed based on its cell type, size, and neighboring cells.

The ability for SPADE to modulate activations based on the context of adjacent cell segmentations allows the network to effectively model the behaviors and interactions of cells.

The input of CCIGAN is a noise vector z ??? R 128 and a segmentation map S. f denotes a linear layer R 128 ??? R 2048 .

R i are feature map representations from SPADE resblocks and X denotes the final output of M cell expressions.

Below, each layer's output dimensions are given next to their respective equations.

Our architectural contribution is a protein marker dependent attention module in the final output layer.

The goal of the attention module is to condition the final output of a channel on a protein marker m and S's cell types.

For example the protein marker, pan-keratin m pk , is expressed exclusively in tumor cells but not in other cells.

Appropriately, an attention mechanism should attend to tumor cells and ignore irrelevant cells in S for m pk .

To replicate a marker searching for specific cell types that express it, we define a learned persistent vector for each marker denoted by s m???M ??? R 8 that undergo a series of operations ( Figure 4 ) with the final feature map representation attending to m's specific cell types.

It is also worthwhile to note that these persistent vectors s m offer a degree of model interpretability that mimic real world markers.

The current input dimensions to the attention module are R (128, 64, 64) following the last resblock R 4 and m indexes from 1, .., M .

Shown in Figure 4 , after R 4 , a bottleneck convolution is applied to match the original data's dimension as O (step 1), which is used in a residual manner with the final output.

Intuitively at this stage, O's feature maps resemble the target Y, but we wish to further refine the output channels.

We convolve O into M K channeled features for each protein marker where K = 8.

Considering each C i where i ??? {1, ..., M } as a group of K channels, the model spatially adaptive normalizes each C i and computes an outer product with the corresponding persistent vector s i and C i .

The resulting matrix is flattened and convolved (with a kernel size of 1 on the pixel level) from A i ??? R (|s|??K,64,64) ??? R (1,64,64) followed by a sigmoid ??(??) activation.

Lastly, the attentions B 1,...,M are added to O to obtain the output X. Initially, the model has no priors over the interaction of protein markers and cell types.

The proposed outer product attention layer (outer product and 1 ?? 1 convolution) excels at modeling these relationships and interactions between specific markers and cell types.

By using an outer product, the model forces attention at a pairwise pixel level comparison for all combinations of elements between s m and A i .

As training progresses, both the learned features over segmentation patches and the learned persistent vectors s m improve, in turn allowing the learned 1 ?? 1 convolution to reason about positive or negative relationships from the pairwise pixel combinations.

Our implementation (A.1) of the generator applies Spectral Norm to all layers (Miyato et al., 2018) .

The discriminator's input is the output of the generator concatenated with the segmentation patch

To conduct fair experiments, all models were optimized, tuned, and set with similar parameters.

They were also taken from their official online implementations and trained for 120 epochs or until convergence (max 150).

CCIGAN is identical to our designed SPADE comparison baseline with the exception of the attention module.

Three experiments were conducted to validate the trained model's utility in generating biologically meaningful cellular proteins in the tumor microenvironment and ability to recapitulate and quantify previously established biological phenomena.

Each subsection describes the experiment and the relevant metrics used in evaluation.

Full mathematical definitions are given in section A.3.

First, we use the following evaluation metrics in order to compare with baseline results: adjusted L 1 and MSE score, L 1 and MSE score, structural similarity (SSIM) index (Wang et al., 2004) In the first step of biologically verifying the accuracy of CCIGAN, PD-1/PD-L1 expression relationships between CD8 T cells and tumor cells were assessed.

Many T cells located within the tumor microenvironment have upregulated expression of PD-1 suggesting that the tumor milieu exerts influence on the protein localization and expression of infiltrating T cells (Ahmadzadeh et al., 2009) .

We assessed if increased expression of PD-L1 on neighboring tumor cells would result in increased directional PD-1 expression in a CD8 T cell ( Figure 5 ).

We also determined if there was a shift in the cell surface localization of the PD-1.

To do so, we computed the Earth Mover's Distance between a CD8 T cell's PD-1 expression (represented as a histogram on a cell's polar coordinates) and itself before and after in different tumor scenarios.

In addition to the mass, we computed the expected center of mass (COM) of PD-1 in a CD8 T cell with respect to the neighboring tumor PD-L1 expressions.

Equations and explanations are given in A.3.2 and A.3.3.

Additional figures illustrating this process are given in A.4.

We expected that the properties of the PD-L1 expressing tumor cell would result in an increased directional PD-1 expression in a neighboring T cell.

As a control, a PD-1 expressing T Cell was surrounded with endothelial cells (a normal tissue lining cell).

As shown in Table 2 , our initial hypothesis was confirmed in CCIGAN's predictions, wherein the presence of a PD-L1 expressing tumor cell, PD-1 expression in the CD8 T cell increased and moved towards the PD-L1 COM.

On the other hand, the endothelial cell presence yielded no effect on the T cell's PD-1 expression, a result which is biologically expected.

While both tumor and endothelial cells express PD-L1, the tumor microenvironment exercises more complex suppressive effects on CD8 T cells, which are more likely to modulate the T cell expression than endothelial cells.

These findings verify CCIGAN captures the biological relationship of tumor cells inducing immunomodulatory changes on a neighboring T cell.

None of the other models succeeded in capturing this protein relationship.

This serves to highlight the capacity of CCIGAN to recapitulate established cell interaction phenomena within the tumor microenvironment.

Secondly, we assessed the effect of immune cell presence on tumor cell status markers.

Keratins are a class of intracellular proteins that play an important role in ensuring cell structure.

Furthermore, when CD8 T cells mediate tumor killing, they release enzymes which cleave the tumor cell's pan-keratin, disrupting the tumor cell structure (Oshima, 2002) .

We explored how the presence of neighboring CD8 T cells to a tumor cell would affect pan-keratin levels within the tumor cell, hypothesizing that T cell mediated tumor killing would result in a drop in pan-keratin tumor expression.

In this experiment, we used a Student's t-test as the statistical hypothesis test to evaluate the correlations between the pan-keratin expression in tumor cells and the area/number of CD8 T cells in contact with the tumor cell surface.

Given a generated pan-keratin channel X i ??? R (H,W ) and the segmentation map channel for CD8 T cells S i ??? R (H,W ) , we compute the total area of the cells

W w=1 S i , and the total expression level of pan-keratin

and assess significance of the slope using a t-test against the null of no change in pan-keratin expression as a function of CD8 T cell-tumor contact.

CD8 T cells were placed adjacent to a tumor cell in increasing number, similar to Figure 1 (B)'s first row and Figure 5 's experiment.

It was observed that the presence of neighboring CD8 T cells to a tumor cell tended to decrease the pan-keratin expression in the tumor cell.

This effect became more dramatic as the number of surrounding CD8 T cells was increased (A.5).

As a control, when the tumor cell of interest was surrounded with adjacent tumor cell(s) instead of CD8 T cells, no change was noted in that tumor cell's pan keratin for CCIGAN.

Table 2 's fourth row shows the result of the t-test where the slope of the linear regression of pan-keratin expression with respect to the number of surrounding cells is compared to a y = 0 flat baseline.

A statistically significant difference between the slope of the linear regression for the T cell scenario vs. the baseline was found in the main experiment, but was not present in the control scenario.

These results indicate that the T cell presence mitigates a decrease in pan-keratin expression in the tumor cell, which is suggestive of T cell mediated tumor killing.

Other models reported contrasting results where a significant change in the tumor cell pan-keratin expression was incorrectly reported under the control conditions.

The resulting graphs are given in A.5.

Further protein interaction patterns identified by CCIGAN are also reported in A.7.

The findings from this experiment serve to highlight the robustness and accuracy of CCIGAN compared to existing image synthesis techniques in probing cell-cell interactions.

As a final biological evaluation, we consider the variability of PD-L1 expression in immune and tumor cells across TNBC patient groups.

Keren et al. (2018) determined that in situations of mixed tumor-immune architecture, where immune cells freely infiltrated the tumor, the tumor cells predominantly expressed PD-L1.

Conversely, in situations of compartmentalized tumors, where there is a greater degree of delineation between immune and tumor cells, macrophages were the predominant source of expressed PD-L1, particularly at the tumor boundary.

Figure 1 (B)'s row 2, 3 show examples of how directly manipulated segmentation maps can simulate the two microenvironments.

We used CCIGAN to predict on 200 directly manipulated mixed and non-mixed tumor environment segmentation patches.

Similar to Section 5.3's experimental settings, we then compute the average expression of a specific marker for the cells of interest for all patches.

For each experiment, we use endothelial cells as control cells to show our result has biological significance.

These findings were recapitulated by CCIGAN in Table 3 .

For a patient with a mixed tumor environment, when trained with mixed patient samples, CCIGAN reported increased PD-L1 expression on tumor cells.

Furthermore, CCIGAN was able to quantify this difference in expression at the single cell level, reporting a tumor to macrophage PD-L1 expression ratio (bolded) of approximately 3.2 and 1.75 for patients A and B respectively.

Conversely, when trained with compartmentalized patient samples, CCIGAN reported increased PD-L1 expression on macrophages adjacent to tumor cells as compared to macrophages adjacent to normal endothelial (inert) cells for patients C and D. This difference was quantified as a ratio of PD-L1 expression of tumor-adjacent macrophages to endothelial-adjacent macrophages, approximately 1.85 and 2.7 for patient C and patient D respectively.

Moreover, using the trained compartmentalized model to predict on mixed segmentation patches, CCIGAN still reports a 26% (patient C) and 19% (patient D) increase of macrophage PD-L1 expression when compared to mixed microenvironments (Table 4) .

Table 4 : Average PDL1 expression of macrophages/monocytes on the compartmentalized tumor environment.

The PD-L1 ratios for the above two scenarios indicate that CCIGAN has appropriately captured previously reported biological outcomes and is capable of quantifying these phenomena at single cell levels.

Furthermore, the model is adaptable to various different types of tumor architecture depending on its training set to produce different hypothesis testing environments.

The agreement between CCIGAN data and those reported in Keren et al. (2018) serve as an important control and demonstrate the fidelity of the CCIGAN output towards true biological results.

More importantly, this provides strong evidence to support the accuracy of the predictions made by CCIGAN in the assessment of hypothetical cellular scenarios which cannot be tested via in vitro tissue study alone.

Examining the model's persistent vectors s m , we can try to understand if there is a match between real world protein markers and the representations of s m .

For example, the vector s pk for pankeratin attends to tumor cells and s CD8 attends to CD8 T cells at pixel pairwise levels.

It follows that in a simple experiment where corresponding s CD8 ??? s pk vectors are exchanged internally in the attention module (Eq. 10, Figure 4 Step 3, outer product) we may observe a lower expression for tumor cells in channel m pk and a lower expression for CD8 T cells in channel m CD8 since tumor cells do not express CD8 and CD8 T cells do not express pan-keratin.

As a control, we also switch surface membrane markers HLA Class 1 and dsDNA markers as they are present in all cells and have very similar average expression values (s HLAc1 ??? s dsDNA ).

Accordingly, for our control, we expect to see negligble changes.

We define the expression ratio as In Table 5 , we can see a larger magnitude decrease of the expression ratios in the s CD8 ??? s pk experiment and a minute difference in the s HLAc1 ??? s dsDNA .

Further visualizations ( Figure 13 ) and discussion (model generativeness, Figure 14) are given in A.6.

We introduced the idea of applying image synthesis to understanding and exploring cell-cell interactions in various and different contexts.

To do so we use a protein attention based GAN, CCIGAN, which can provide accurate characterizations of cellular protein localization phenomena from conditioned counterfactual cell-cell scenarios.

Additionally, the architecture of the attention module we propose can be generalized to other multiplexed datasets that require real world priors.

Furthermore, CCIGAN outperforms a variety of current methods in biological modeling.

We demonstrate this through biological consistency where CCIGAN recapitulates, discovers, and quantifies meaningful cellular interactions through 3 different experiments in a tumor environment unrecognized by other models.

This highlights the potential for CCIGAN to identify cellular protein interactions which account for variation in patient responses to cancer therapy, providing a framework for biological hypotheses which explain clinical outcomes on a cellular level.

where ResBlk is the residual block with skip connection used in ResNet (He et al., 2016) , and SPADE is the spatially-adaptive normalization layer.

The detailed architecture of our discriminator is shown on Table 7 .

For all baseline models, we use the architecture based on the their original implementation.

Due to the size of the cell patch is (64, 64), we reduce the size of hidden layers to fit our dataset.

For fair comparison, we use the same reduction of hidden layers and the same discriminator architecture for SPADE, pix2pixHD, and CCIGAN.

G is the generator and D is the discriminator for CCIGAN.

Given segmentation map S, ground truth Y and noise ??, the generated image is X = G(S, ??).

The input of the discriminator is the cell image conditioned on the segmentation map S. We use LSGAN loss (Mao et al., 2017) in CCIGAN, which is defined as follows:

In addition to GAN loss, we also use feature matching loss (Wang et al., 2018a) during training expressed as:

where D j is j-th layer feature map of the discriminator for j ??? {1, ..., J}, and N j is the number of elements in j-th layer.

Consequently, the objective function for training is given as follows:

where ?? = 10.

Due to the size of cell patch is (64, 64), we do not use multi-scale discriminators and perceptual loss in CCIGAN and other baseline models e.g. SPADE and pix2pixHD.

In training, we use ADAM as the optimizer.

The generator learning rate is lr G = 0.0004 and the discriminator learning rate is lr D = 0.0001.

We train CCIGAN 120 epochs with a training set of 5648 cell patches.

We train other baseline models for 120 epochs or until they converge (max 150).

The full details of training of CCIGAN and baselines are shown as Table 8 .

The hyperparameters of each model are fine-tuned to get better performance.

The training time was roughly equal for all models.

In particular, CCIGAN was around 1.2 times slower than the SPADE baseline on a single Tesla V100 GPU.

Given the generated image set X = {X i } N i=1 and the ground truth set

where ?? * can be either L 1 or L 2 norm, is the element-wise product, X i,m and Y i,m are the m-th channel of the i-th cell patch, U i ??? {0, 1} (H,W ) is the mask matrix which masks all the cells in i-th patch.

For any matrix A, sort(A) is the sort function that sorts all entries of A. The sorting function ensures our metrics are position independent and only measures the intensity of the generated image and ground truth.

The score function L(X , Y) only computes the loss of sorted expression inside of the cells.

Then we add penalization for expression outside of cells.

The adjusted L 1 /MSE score is introduced as follows,

where 1 d is the matrix with all entries equal to 1.

The (adjusted) L 1 /MSE scores of CCIGAN and baseline models are shown on Table 1 .

A smaller score means a better result.

For any two images X, Y ??? [0, 1] (H,W ) , the SSIM and MI are defined as:

where H(??) is entropy, ?? X and ?? X are the mean and standard deviation of X, c 1 , c 2 are constants.

In cell based MI, test patches are processed at a cell-cell basis where their mutual information is computed with the corresponding cell in the ground truth.

For the generated image X i of the i-th patch, we assume there are T i cells in the i-th patch.

Then for each cell t, the pixels of m-th channel of the t-th cell in the i-th patch can be expressed as a vector x t i,m .

Hence, the cell based MI is formulated as:

We report I(X ; Y) on Table 1 .

The SSIM measures the similarity between the generated image and the ground truth.

For SSIM, we use HLA Class 1 and dsDNA due to the their expressions in all cells.

If all channels were considered, the SSIM would be uninformative due to the majority of the channels being blank or sparse.

The MI measures the information shared between generated image and ground truth at a cell by cell basis where we consider all channels.

Consider the example where a model generates no expression in marker m but the real data has expression in m, the MI would be 0 and vice versa.

Higher SSIM and MI values mean better results.

Table 1 demonstrates that CCIGAN outperforms or matches all other baselines on all reconstruction metrics.

For a generated cell image, its centroid is the mean position of all the points in the cell.

We use pixel values as weights in computing the weighted centroid, now referred to as center of mass (COM).

Given a cell image X ??? R (H,W ) , with indices of the segmented cell V ??? {1, . . .

, H}??{1, . . .

, W }, the COMp = (x,??) is defined asx = (x,y)???V xXx,y (x,y)???V Xx,y and?? = (x,y)???V yXx,y (x,y)???V Xx,y .

In the PD-1/PD-L1 experiment, we compute the COM of the CD8 T cell (cell of interest) weighted by PD-1 expression, given asp CD8 , and the COM of all tumor cells weighted by PD-L1 expression, given asp

Tumor .

Since T cells located within the tumor microenvironment often have upregulated expression of PD-L1, we assume thatp CD8 should have the same COM as all of its surrounding tumor cellsp

Tumor .

The center of mass score is defined below as the relative distance betweenp CD8 andp Tumor , where N is defined as the number of patches:

The projection function Proj(??) is used to projectp Tumor onto the CD8 T cell to ensure the expected COM of the tumor cells is inside of the CD8 T cell.

As a reference we choose a random position p Random in the CD8 T cell (PD-1) which replacesp CD8 in Eq. 22 and compute the random COM score to show the effectiveness of the result.

An example illustration is given in Figure 6 A.4.

For each segmentation map i, we iteratively add its T i tumor cells around one CD8 T cell.

The COM is defined for the t-th tumor cell asp Tumor t,i for t ??? {1, ..., T i }.

We omit the subscript i when it is clear from context.

The subsequent instances of the PD-1 COMs in the CD8 T cell by adding the t-th tumor are given byp CD8 t .

Initially when there are no tumor cells, we definep CD8 0 as the centroid of the CD8 T cell.

Based on the above setting, we define a vector v t which points from the centroid of the CD8 T cellp CD8 0 , to the COM of the t-th tumor cellp Tumor t .

We define vector u t which points from the previous COMp CD8 t???1 to the current COMp CD8 t of the CD8 T cell.

We define ?? t as the angle between v t , u t .

If cos ?? t > 0, that is to say if the cosine similarity is positive, the COM of a CD8 T cellp CD8 t , moves correctly towards the COM of the added tumor cellp Tumor t .

An illustration of the points and vectors is given in Figure 8 .

Formally:

After obtaining the directional information, we use Earth Mover's Distance (EMD) (Rubner et al., 2000) to evaluate the changes in PD-1 expression of the CD8 T cell.

The EMD, which measures the dissimilarity of two distributions, is used in this context to measure the protein localization shifts in PD-1 before and after adding a tumor cell.

We consider each cell X in polar coordinates (r, ??) with respect to its centroid, integrate its expression along the radius coordinates, and evaluate the resulting histogram hist(X) along the angle coordinate.

This allows for the definition of distance for moving one histogram to another, i.e. em(

)), for the generated PD-1 expression of the CD8 T cell X t i when adding the t-th tumor cell.

The final EMD score is defined as:

where the indicator function 1(??) = 1 if and only if X , implying the shift in PD-1 expression is correct, and in turn increases the EMD.

By contrast, the EMD score decreases whenp CD8 t moves in the opposite direction.

Example figures and illustrations showing this process are given in Figure 7 A.4.

Using the EMD we define a randomized search algorithm for discovering other cell-cell interactions; their results and discussion are given in A.7.

Based on the definition of EMD score, the positive EMD score is defined as:

The positive EMD score only evaluates the change in PD-1 expression when the COM of a CD8 T cell moves towards the COM of the added tumor cell.

The projected EMD score is defined as:

The projected EMD score is the EMD score weighted by u t,i , i.e. the shift from the previous COM to the current COM of the CD8 T cell.

Here we provide some example visualizations and illustrations center of mass nomenclature, mass movement of PD-1 as a function of neighboring PD-L1, and process of computing EM distance.

We can observe in Figure 7 that the mass inside of the T cell in the PD-1 channel shifts as a response to surrounding tumor cell expressions of PD-L1.

The surrounding tumor PD-L1 expressionsp Tumor t are shown in the third row on a cell by cell basis for t ??? {1, ..., T i = 4}. Note that the 3rd column in PD-L1 has sparse expression.

Finally the last row shows the PD-1 and PD-L1 channels superimposed into one channel.

We give an example in Figure 8 to illustrate the vectors v t and u t after adding t-th tumor cell for t = 1, 2 in computing EMD score, where v t and u t defined in Eq. 23.

The pan-keratin/CD8 experiment is similar to Figure 7 's orientation except the center cell (cell of interest) is a tumor cell (red) and the adjacent neighboring cells are CD8 T cells (orange).

CCIGAN predicted a decrease in tumor cell pan-keratin expression with respect to increasing CD8 T cell area/number ( Figure 9 ).

This is juxtaposed to the tumor cell control where there is no change in the pan-keratin level as the number of neighboring tumor cells is increased.

SPADE does not predict a decrease in tumor cell pan-keratin expression with respect to increasing CD8 T cell area/number and shows no difference in pan-keratin expression trends between the T cell and control groups (Figure 10 ).

pix2pixHD erroneously predicts an increase in tumor cell pan-keratin expression with respect to increasing CD8 T cell area/number and shows no difference in pan-keratin expression trends between the T cell and control groups (Figure 11 ).

CycleGAN fails to predict a decrease in tumor cell pan-keratin expression with respect to increasing CD8 T cell area/number and shows no difference in pan-keratin expression trends between the T cell and control groups (Figure 12 ).

A.6 MODEL INTERPRETABILITY AND GENERATIVENESS Figure 13 shows the persistent vectors s i for all proteins.

Note the similarity between CD3 and CD8 T cell protein markers and the similarity between dsDNA and HLA Class 1 surface membrane proteins (expressed in all cells).

It is also important to make the distinction that sparse markers (while different) are similar in state.

This is due to the lack of training data for rare cell types, making it difficult for the model to reason on such a small sample size.

Figure 14 shows the generativeness of CCIGAN through an uncertainty map over 100 instances (random noise).

An uncertainty map shows the differences per pixel (x, y) location.

The higher intensities indicate a higher probability of changing at the specified (x, y) location.

Table 9 : Additional cell interaction trends captured by CCIGAN.

The experimental findings in Indices 4, 5 and 6 support those reported in 6.2.2, demonstrating that immune cell presence adjacent to tumor cells causes a decrease in the tumor's pan-keratin, regardless of immune cell identity.

In addition to confirming the results of 6.2.1, findings in indices 1 and 3 also indicate that tumor cells increase the expression of the T cell co-receptor, although this is of unclear functional significance.

Index 2 suggests T cell clustering reduces the expression of the immune suppressive PD-1 marker.

Lastly, Index 7 demonstrates an increase in macrophage expressed vimentin when macrophages are placed adjacent to tumor cells.

Since vimentin is secreted as a pro-inflammatory marker in macrophages, this suggests an early macrophage inflammatory response to its tumor neighbor.

The markers we used (total 24) in our experiments are: Pan-Keratin, EGFR, Beta catenin, dsDNA, Ki67, CD3, CD8, CD4, FoxP3, MPO, HLA-DR, HLA-Class-1, CD209, CD11b, CD11c, CD68, CD63, Lag3, PD1, PD-L1, IDO, Vimentin, SMA, CD31.

The markers we didn't use (total 12) in our experiments are: CD16, B7H3, CD45, CD45RO, Keratin17, CD20, CD163, CD56, Keratin6, CSF-1R, p53, CD138

@highlight

We explore cell-cell interactions across tumor environment contexts observed in highly multiplexed images, by image synthesis using a novel attention GAN architecture.

@highlight

A new method to model the data generated by multiplexed ion beam imaging by time-of-flight (MIBI-TOF) by learning the many-to-many mapping between cell types and protein markers' expression levels.