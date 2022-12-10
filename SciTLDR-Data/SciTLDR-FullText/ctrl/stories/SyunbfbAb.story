We introduce FigureQA, a visual reasoning corpus of over one million question-answer pairs grounded in over 100,000 images.

The images are synthetic, scientific-style figures from five classes: line plots, dot-line plots, vertical and horizontal bar graphs, and pie charts.

We formulate our reasoning task by generating questions from 15 templates; questions concern various relationships between plot elements and examine characteristics like the maximum, the minimum, area-under-the-curve, smoothness, and intersection.

To resolve, such questions often require reference to multiple plot elements and synthesis of information distributed spatially throughout a figure.

To facilitate the training of machine learning systems, the corpus also includes side data that can be used to formulate auxiliary objectives.

In particular, we provide the numerical data used to generate each figure as well as bounding-box annotations for all plot elements.

We study the proposed visual reasoning task by training several models, including the recently proposed Relation Network as strong baseline.

Preliminary results indicate that the task poses a significant machine learning challenge.

We envision FigureQA as a first step towards developing models that can intuitively recognize patterns from visual representations of data.

Scientific figures compactly summarize valuable information.

They depict patterns like trends, rates, and proportions, and enable humans to understand these concepts intuitively at a glance.

Because of these useful properties, scientific papers and other documents often supplement textual information with figures.

Machine understanding of this structured visual information could assist human analysts in extracting knowledge from the vast documentation produced by modern science.

Besides immediate applications, machine understanding of plots is interesting from an artificial intelligence perspective, as most existing approaches simply revert to inverting the visualization pipeline (i.e. by reconstructing the source data).

Mathematics exams, e.g. Graduate Records Examinations (GREs), often include questions about a plot regarding relations between the plot elements.

When solving these exam questions, humans don't always build a table of coordinates for all data points, but judge by visual intuition.

Thus motivated, and inspired by recent research in Visual Question Answering (VQA) BID1 Goyal et al., 2016) and relational reasoning BID6 BID20 , we introduce FigureQA.

FigureQA is a corpus of over one million question-answer pairs grounded in over 100, 000 figures, devised to study aspects of comprehension and reasoning in machines.

There are five common figure types represented in the corpus, which model both continuous and categorical information: line, dot-line, vertical and horizontal bar, and pie plots.

Questions concern one-to-all and one-to-one relations among plot elements: for example, Is X the low median?, Does X intersect Y?.

Their successful resolution requires inference over multiple plot elements.

There are 15 question types in total, which address properties like magnitude, maximum, minimum, median, area-under-the-curve, smoothness, and intersections.

Each question is posed such that its answer is either yes or no.

FigureQA is a synthetic corpus, like the related CLEVR dataset for visual reasoning BID6 .

While this means that the data may not exhibit the same richness as figures "in the wild", it permits greater control over the task's complexity, enables auxiliary supervision signals and most importantly provides reliable ground-truth answers.

Further, by analyzing the performance on real figures of models trained on FigureQA it will be possible to extend the corpus to address limitations not considered during generation.

The FigureQA corpus can be extended iteratively, each time raising the task complexity, as model performance increases.

This is reminiscent of curriculum learning BID2 ) allowing iterative pretraining on increasingly challenging versions of the data.

By releasing the data now, we want to gauge the interest in the research community and adapt future versions based on feedback, hopefully accelerating research in this direction.

Additional annotation is provided to allow researchers to define other tasks than the one we introduce in this manuscript.

The corpus is built using a two-stage generation process.

First, we sample numerical data according to a carefully tuned set of constraints and heuristics designed to make sampled figures appear natural.

Next we use the Bokeh open-source plotting library (Bokeh Development Team, 2014) to plot the data in an image.

This process necessarily gives us access to the quantitative data presented in the figure.

We also modify the Bokeh backend to output bounding boxes for all plot elements: data points, axes, axis labels and ticks, legend tokens, etc.

We provide the underlying numerical data and the set of bounding boxes as supplementary information with each figure, which may be useful in formulating auxiliary tasks like reconstructing quantitative data given only a figure.

Bounding box targets of plot elements relevant for answering a question might be useful for supervising an attention mechanism to ignore potential distractors.

Experiments in that direction are outside of the scope of this manuscript, but we want to facilitate research of such approaches by releasing these annotations.

As part of the generation process we balance the ratio of yes and no answers for each question type and each figure.

This makes it more difficult for models to exploit biases in answer frequencies while ignoring visual content.

We review related work in Section 2.

In Section 3 we describe the FigureQA dataset and the visualreasoning task in detail.

Section 4 describes and evaluates four neural baseline models trained on the corpus: a text-only Long Short-Term Memory (LSTM) model (Hochreiter & Schmidhuber, 1997) as a sanity check for biases, the same LSTM model with added Convolutional Neural Network (CNN) image features BID9 Fukushima, 1988) , and a Relation Network (RN) BID16 , a strong baseline model for relational reasoning.

The RN respectively achieves accuracies of 72.40% and 76.52% on the FigureQA test set with alternated color scheme (described in Section 3.1) and the test set without swapping colors.

An "official" version of the corpus is publicly available as a benchmark for future research.

1 We also provide our generation scripts, which are easily configurable, enabling researchers to tweak generation parameters to produce their own variations of the data.

Machine learning tasks that pose questions about visual scenes have received great interest of late.

For example, BID1 proposed the VQA challenge, in which a model seeks to output a correct natural-language answer a to a natural-language question q concerning image I.

An example is the question "Who is wearing glasses?" about an image of a man and a woman, one of whom is indeed wearing glasses.

Such questions typically require capabilities of vision, language, and common-sense knowledge to answer correctly.

Several works tackling the VQA challenge observe that models tend to exploit strong linguistic priors rather than learning to understand visual content.

To remedy this problem, Goyal et al. (2016) introduced the balanced VQA task.

This features triples (I , q, a ) to supplement each image-question-answer triple (I, q, a), such that I is similar to I but the answer given I and the same q is a rather than a.

Beyond linguistic priors, another potential issue with the VQA challenges stems from their use of real images.

Images of the real world entangle visual-linguistic reasoning with common-sense concepts, where the latter may be too numerous to learn from VQA corpora alone.

On the other hand, synthetic datasets for visual-linguistic reasoning may not require common sense and may permit the reasoning challenge to be studied in isolation.

CLEVR BID6 and NLVR BID20 are two such corpora.

They present scenes of simple geometric objects along with questions concerning their arrangement.

To answer such questions, machines should be capable of spatial and relational reasoning.

These tasks have instigated rapid improvement in neural models for visual understanding BID16 BID13 Hu et al., 2017) .

FigureQA takes the synthetic approach of CLEVR and NLVR for the same purpose, to contribute to advances in figure-understanding algorithms.

The figure-understanding task has itself been studied previously.

For example, BID17 present a smaller dataset of figures extracted from research papers, along with a pipeline model for analyzing them.

As in FigureQA, they focus on answering linguistic questions about the underlying data.

Their FigureSeer corpus contains 60, 000 figure images annotated by crowdworkers with the plot-type labels.

A smaller set of 600 figures comes with richer annotations of axes, legends, and plot data, similar to the annotations we provide for all 140, 000 figures in our corpus.

The disadvantage of FigureSeer as compared with FigureQA is its limited size; the advantage is that its plots come from real data.

The questions posed in FigureSeer also entangle reasoning about figure content with several detection and recognition tasks, such as localizing axes and tick labels or matching line styles with legend entries.

Among other capabilities, models require good performance in optical character recognition (OCR).

Accordingly, the model presented by BID17 comprises a pipeline of disjoint, off-the-shelf components that are not trained end-to-end.

BID14 propose the related task of recovering visual encodings from chart images.

This entails detection of legends, titles, labels, etc., as well as classification of chart types and text recovery via OCR.

Several works focus on data extraction from figures.

BID21 use convolutional networks to detect boundaries of subfigures and extract these from compound figures; BID7 propose a system for processing chart images, which consists of figuretype classification followed by type-specific interactive tools for data extraction.

Also related to our work is the corpus of Cliche et al. (2017) .

There, the goal is automated extraction of data from synthetically generated scatter plots.

This is equivalent to the data-reconstruction auxiliary task available with FigureQA.FigureQA is designed to focus specifically on reasoning, rather than subtasks that can be solved with high accuracy by existing tools for OCR.

It follows the general VQA setup, but additionally provides rich bounding-box annotations for each figure along with underlying numerical data.

It thus offers a setting in which existing and novel visual-linguistic models can be trained from scratch and may take advantage of dense supervision.

Its questions often require reference to multiple plot elements and synthesis of information distributed spatially throughout a figure.

The task formulation is aimed at achieving an "intuitive" figure-understanding system, that does not resort to inverting the visualization pipeline.

This is in line with the recent trend in visual-textual datasets, such as those for intuitive physics and reasoning (Goyal et al., 2017; BID11 .The majority of recent methods developed for VQA and related vision-language tasks, such as image captioning BID23 Fang et al., 2015) , video-captioning BID26 , phrase localization (Hu et al., 2016) , and multi-modal machine translation BID5 , employ a neural encoder-decoder framework.

These models typically encode the visual modality with pretrained CNNs, such as VGG BID18 or ResNet (He et al., 2016) , and may extract additional information from images using pretrained object detectors BID15 .

Language encoders based on bag-of-words or LSTM approaches are typically either trained from scratch BID5 or make use of pretrained word embeddings BID25 .

Global or local image representations are typically combined with the language encodings through attention BID22 BID10 and pooling (Fukui et al., 2016) mechanisms, then fed to a decoder that outputs a final answer in language.

In this work we evaluate a standard CNN-LSTM encoder model as well as a more recent architecture designed expressly for relational reasoning BID16 .

FigureQA consists of common scientific-style plots accompanied by questions and answers concerning them.

The corpus is synthetically generated at large scale: its training set contains 100, 000 images with 1.3 million questions; the validation and test sets each contain 20, 000 images with over 250, 000 questions.

The corpus represents numerical data according to five figure types commonly found in analytical documents, namely, horizontal and vertical bar graphs, continuous and discontinuous line charts, and pie charts.

These figures are produced with white background and the colors of plot elements (lines, bars and pie slices) are chosen from a set of 100 colors (see Section 3.1).

Figures also contain common plot elements such as axes, gridlines, labels, and legends.

We generate questionanswer pairs for each figure from its numerical source data according to predefined templates.

We formulate 15 questions types, given in TAB0 , that compare quantitative attributes of two plot elements or one plot element versus all others.

In particular, questions examine properties like the maximum, minimum, median, roughness, and greater than/less than relationships.

All are posed as a binary choice between yes and no. In addition to the images and question-answer pairs, we provide both the source data and bounding boxes for all figure elements, and supplement questions with the names, RGB codes, and unique identifiers of the featured colors.

These are for optional use in analysis or to define auxiliary training objectives.

In the following section, we describe the corpus and its generation process in depth.

The many parameters we use to generate our source data and figures are summarized in Table 1 .

These constrain the data-sampling process to ensure consistent, realistic plots with a high degree of variation.

Generally, we draw data values from uniform random distributions within parameterlimited ranges.

We further constrain the "shape" of the data using a small set of commonly observed functions (linear, quadratic, bell curve) with additive perturbations.

A figure's data points are identified visually by color; textually (on axes and legends and in questions), we identify data points by the corresponding color names.

For this purpose we chose 100 unique colors from the X11 named color set 3 , selecting those with a large color distance from white (the figure background color).We construct FigureQA's training, validation, and test sets such that all 100 colors are observed during training, while validation and testing are performed on unseen color-plot combinations.

This we accomplish by a methodology consistent with that of the CLEVR dataset BID6 , as follows.

We divide our 100 colors into two disjoint, equally-sized subsets (denoted A and B).

In the training set, we color a particular figure type by drawing from one, and only one, of these subsets (see Table 1 ).

When generating the validation and test sets, we draw from a given plot type's opposite subset, i.e., if subset A was used for training, then subset B is used for validation and testing.

We refer to this as the "alternated color scheme." 4 We define the appearance of several other aspects during data generation, randomizing these too to encourage variation.

The placement of the legend within or outside the plot area is determined by a coin flip, and we select its precise location and orientation to cause minimal obstruction by counting the occupancy of cells in a 3 × 3 grid.

Figure width is constrained to within one to two times the height, there are four font sizes available, and grid lines may be rendered or not -all with uniform probability.

Table 1 : Synthetic Data Parameters, the color set column indicates the set used for training.

We generate questions and their answers by referring to a figure's source data and applying the templates given in TAB0 .

One yes and one no question is generated for each template that applies.

Once all question-answer pairs have been generated, we filter them to ensure an equal number of yes and no answers by discarding question-answer pairs until the answers per question type are balanced.

This removes bias from the dataset to prevent models from learning summary statistics of the question-answer pairs.

Note that since we provide source data for all the figures, arbitrary additional questions may be synthesized.

This makes the dataset extensible for future research.

To measure the smoothness of curves for question templates 9 and 10, we devised a roughness metric based on the sum of absolute pairwise differences of slopes, computed via finite differences.

Concretely, for a curve with n points defined by series x and y, DISPLAYFORM0

We generate figures from the synthesized source data using the open-source plotting library Bokeh.

Bokeh was selected for its ease of use and modification and its expressiveness.

We modified the library's web-based rendering component to extract and associate bounding boxes for all figure elements.

Figures are encoded in three channels (RGB) and saved in Portable Network Graphics (PNG) format.

In all experiments we use training, validation, and test sets with the alternated color scheme (see Section 3.1).

The results of an experiment with the RN baseline trained and evaluated with different schemes is provided in Section C.

We train all models using the Adam optimizer BID8 ) on the standard cross-entropy loss with learning rate 0.00025.Preprocessing We resize the longer side of each image to 256 pixels, preserving the aspect ratio; images are then padded with zeros to size 256 × 256.

For data augmentation, we use the common scheme of padding images (to size 264×264) and then randomly cropping them back to the previous size (256 × 256).Text-only baseline Our first baseline is a text-only model that uses an LSTM 7 to read the question word by word.

Words are represented by a learned embedding of size 32 (our vocabulary size is only 85, not counting default tokens such as those marking the start and end of a sentence).

The LSTM has 256 hidden units.

A Multi-Layer Perceptron (MLP) classifier passes the last LSTM state through two hidden layers with 512 Rectified Linear Units (ReLUs) BID12 to produce an output.

The second hidden layer uses dropout at a rate of 50% BID19 .

This model was trained with batch size 64.

CNN+LSTM In this model the MLP classifier receives the concatenation of the question encoding with a learned visual representation.

The visual representation comes from a CNN with five convolutional layers, each with 64 kernels of size 3 × 3, stride 2, zero padding of 1 on each side and batch normalization (Ioffe & Szegedy, 2015) , followed by a fully connected layer of size 512.

All layers use the ReLU activation function.

The LSTM producing the question encoding has the same architecture as in the text-only model.

This baseline was trained using four parallel workers each computing gradients on batches of size 160 which are then averaged and used for updating parameters.

CNN+LSTM on VGG-16 features In our third baseline we extract features from layer pool5 of an ImageNet-pretrained VGG-16 network BID18 CNN with four convolutional layers, all with 3×3 kernels, ReLU activation and batch normalization.

The first two convolutional layers both have 128 output channels, the third and fourth 64 channels, each.

The convolutional layers are followed by one fully-connected layer of size 512.

This model was trained using batch size 64.

BID16 introduced a simple yet powerful neural module for relational reasoning.

It takes as input a set of N "object" representations o i ∈ R C , i = 1, . . .

, N and computes a representation of relations between objects according to

where O ∈ R N ×C is the matrix containing N C-dimensional object representations o i,· stacked row-wise.

Both f φ and g θ are implemented as MLPs, making the relational module fullydifferentiable.

In our FigureQA experiments, we follow the overall architecture used by BID16 in their experiments on CLEVR from pixels, adding one convolutional layer to account for the higher resolution of our input images and increasing the number of channels.

We also don't use random rotations for data augmentation, to avoid distortions that might change the correct response to a question.

The object representations are provided by a CNN with the same architecture as the one in the previous baseline, only dropping the fully-connected layer at the end.

Each pixel of the CNN output (64 feature maps of size 8 × 8) corresponds to one "object" DISPLAYFORM0 , where H and W , denote the height and width, respectively.

To also encode the location of objects inside the feature map, the row and column coordinates are concatenated to that representation: DISPLAYFORM1 The RN takes as input the stack of all pairs of object representations, concatenated with the question; here the question encoding is once again produced by an LSTM with 256 hidden units.

Object pairs are then separately processed by g θ to produce a feature representation of the relation between the corresponding objects.

The sum over all relational features is then processed by f φ , yielding the predicted outputs.

The MLP implementing g θ has four layers, each with 256 ReLU units.

The MLP classifier f φ processing the overall relational representation, has two hidden layers, each with 256 ReLU units, the second layer using dropout with a rate of 50%.

An overall sketch of the RN's structure is shown in FIG3 .

The model was trained using four parallel workers, each computing gradients on batches of size 160, which are then averaged for updating parameters.

All model baselines are trained and evaluated using the alternated color scheme.

At each training step, we compute the accuracy on one randomly selected batch from validation set and keep an exponential moving average with decay 0.9.

Starting from the 100th update, we perform earlystopping using the described moving average.

The best performing model using this approximate validation performance measure is evaluated on the whole test set.

FIG4 shows the training and validation accuracy over updates for the RN model.

Our editorial team answered a subset from our test set, containing 16, 876 questions, corresponding to 1, 275 randomly selected figures (roughly 250 per figure type).

The results are reported in TAB2 and compared with the CNN+LSTM and RN baselines evaluated on the same subset.

The comparison between text only and CNN+LSTM models shows that the visual modality contributes to learning.

However, due to the relational structure of the questions, the RN significantly outperforms the simpler CNN+LSTM model.

Our human baseline shows that the problem is challenging, but leads by a significant margin.

TAB3 show the performances of the CNN+LSTM and RN baselines compared to the performances of our editorial staff by figure type and by question type, respectively.

More details on the human baseline and an analysis of results are provided in the supplementary material.

over 100, 000 synthetic figure images.

Questions examine plot characteristics like the extrema, areaunder-the-curve, smoothness, and intersection, and require integration of information distributed spatially throughout a figure.

The corpus comes bundled with side data to facilitate the training of machine learning systems.

This includes the numerical data used to generate each figure and bounding-box annotations for all plot elements.

We studied the visual reasoning task by training four baseline neural models on our data, analyzing their test-set performance, and comparing it with that of humans.

Results indicate that more powerful models must be developed to reach human-level performance.

In future work, we plan to test the transfer of models trained on FigureQA to question-answering on real scientific figures, and to iteratively extend the dataset either by significantly increasing the number of templates or by crowdsourcing natural-language questions.

We envision FigureQA as a first step to developing models that intuitively extract knowledge from the numerous figures produced by modern science.

A DATA SAMPLESHere we present a sample figures of each plot type (vertical bar graph, horizontal bar graph, line graph, dot line graph and pie chart) from our dataset along with the corresponding question-answer pairs and some of the bounding boxes.

A.1 VERTICAL BAR GRAPH

To assess FigureQA's difficulty and to set a benchmark for model performance, we measured human accuracy on a sample of the test set with the alternated color scheme.

Our editorial staff answered 16, 876 questions corresponding to 1, 275 randomly selected figures (roughly 250 per type), providing them in each instance with a figure image, a question, and some disambiguation guidelines.

Our editors achieved an accuracy of 91.21%, compared with 72.18% for the RN Santoro et al. (2017) baseline.

We provide further analysis of the human results below.

We stratify human accuracy by figure type in Table 5 .

People performed exceptionally well on bar graphs, though worse on line plots, dot-line plots, and pie charts.

Analyzing the results and plot images from these figure categories, we learned that pie charts with similarly sized slices led most frequently to mistakes.

Accuracy on dot-line plots was lower because plot elements sometimes obscure each other as FIG0 shows.

TAB3 shows how human accuracy varies across question types, with people performing best on minimum, maximum, and greater/less than queries.

Accuracy is generally higher on question types for categorical figures compared to continuous figures.

It is noticeably lower for questions concerning the median and curve smoothness.

Analysis indicates that many wrong answers to median questions occurred when plots had a larger number of (unordered) elements, which increases the difficulty of the task and may also induce an optical illusion.

In the case of smoothness, annotators struggled to consider both the number of deviations in a curve and the size of deviations.

This was particularly evident when comparing one line with more deviations to another with larger ones.

Additionally, ground truth answers for smoothness were determined with computational or numerical precision that is beyond the capacity of human annotators.

In some images, smoothness differences were too small to notice accurately with the naked eye.

We provided our annotators with a third answer option, unknown, for cases where it was difficult or impossible to answer a question unambiguously.

Note that we instructed our annotators to select unknown as a last resort.

Only 0.34% of test questions were answered with unknown, and this accounted for 3.91% of all incorrect answers.

Looking at the small number of such responses, we observe that generally, annotators selected unknown in cases where two colors were difficult to Which bar is the median: Light Gold or Royal Blue?

Which curve is rougher?One seems 'noisier' while another seems more 'jagged'.

distinguish from each other, when one plot element was covered by another, or when a line plot's region of interest was obscured by a legend.

C PERFORMANCE OF THE RELATION NETWORK WITH AND WITHOUT ALTERNATED COLOR SCHEMEIn this experiment we trained the RN baseline using both the validation set with swapped colors and without, saving parameters for both.

We then evaluated both models on the test sets with and without swapped colors.

Table 7 compares the results.

Table 7 : Performance of our RN baselines trained with early stopping on val1 (using the same color-to-plot-type assignments as in the training set) and with early stopping on val2 (with swapped colors).

We show performances of both on test1 (with the same color assignments) and test2 (with swapped colors).

<|TLDR|>

@highlight

We present a question-answering dataset, FigureQA, as a first step towards developing models that can intuitively recognize patterns from visual representations of data.

@highlight

This paper introduces a dataset of templated question answering on figures, involving reasoning about figure elements.

@highlight

The paper introduces a new visual reasoning dataset called Figure-QA which consists of 140K figure images and 1.55M QA pairs, which can help in developing models that can extract useful information from visual representations of data.