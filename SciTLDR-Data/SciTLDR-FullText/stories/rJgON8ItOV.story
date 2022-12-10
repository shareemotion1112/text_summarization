We present an analytic framework to visualize and understand GANs at the unit-, object-, and scene-level.

We first identify a group of interpretable units that are closely related to object concepts with a segmentation-based network dissection method.

Then, we examine the causal effect of interpretable units by measuring the ability of interventions to control objects in the output.

Finally, we examine the contextual relationship between these units and their surrounding by inserting the discovered object concepts into new images.

We show several practical applications enabled by our framework, from comparing internal representations across different layers and models, to improving GANs by locating and removing artifact-causing units, to interactively manipulating objects in the scene.

We first identify a group of interpretable units that are related to semantic classes (Figure 1a, b) .

These units' featuremaps closely match the semantic segmentation of a particular object class (e.g., trees).

Then, we intervene in units in the network to cause a type of object to disappear or appear (Figure 1c, d) .

Finally, we study contextual relationships by observing where we can insert the object concepts in new images and how this intervention interacts with other objects in the image (Figure 4 ).

This framework allows us to compare representations across different layers, GAN variants, and datasets; to debug and improve GANs by locating artifact-causing units ( Figure 1e

We analyze the internal GAN representations by decomposing the featuremap r at a layer into positions P ⊂ P and unit channels u ∈ U. To identify a unit u with semantic behavior, we upsample and threshold the unit, and measure how well it matches an object class c in the image x as identified by a supervised semantic segmentation network s c (x) BID5 IoU DISPLAYFORM0 , where t u,c = arg max DISPLAYFORM1 This approach is inspired by the observation that many units in classification networks locate emergent object classes when upsampled and thresholded BID0 .

Here, the threshold t u,c is chosen to maximize the information quality ratio, that is, the portion of the joint entropy H which is mutual information I BID4 .To identify a sets of units U ⊂ U that cause semantic effects, we intervene in the network G(z) = f (h(z)) = f (r) by decomposing the featuremap r into two parts (r U,P , r U,P ), and forcing the components r U,P on and off.

Given an original image x = G(z) ≡ f (r) ≡ f (r U,P , r U,P ), we can intervene in the network and generate an image with units U ablated at pixels P: DISPLAYFORM2 Or an image with units U activated to a high level c at pixels P: DISPLAYFORM3 We measure the average causal effect (ACE) BID2 of units U on class c as: DISPLAYFORM4 2 RESULTS AND DISCUSSION Analysis of the semantics and causal behavior of the internal units of a GAN reveals several new findings.

Units matching diverse objeccts emerge on more diverse models.

Internal units for more object classes emerge as the architecture becomes more diverse.

FIG2 compares three models BID3 ) that introduce two innovations on baseline Progressive GANs.

The number of types of objects, parts, and materials matching units increases by more than 40% as minibatch-stdev is introduced; and pixelwise normalization increase units that match semantic classes by 19%.

Interpretable units emerge in the middle layers, not at the initial layers.

In classifier networks, units matching high-level concepts appear in layers furthest from the pixels BID6 .

In contrast, in a GAN, it is mid-level layers 4 to 7 that have the largest number of units that match semantic objects and object parts.

A selection of layers is shown in Figure 3 .

Our framework can also analyze the causes of failures and repair some GAN artifacts.

Figure 1e shows several annotated units that are responsible for typical artifacts consistently appearing across different images.

We can fix these errors by ablating 20 artifact-causing units.

Figure 1g shows that artifacts are successfully removed and the artifact-free pixels stay the same, improving the generated results.

TAB2 summarizes quality improvements: we compute the Fréchet Inception Distance BID1 between the generated images and real images using 50 000 real images and 10 000 generated images with high activations on these units.

We also collect 20 000 annotations of realism on Amazon MTurk, with 1 000 images per method.

An identical "door" intervention at layer4 of each pixel in the featuremap has a different effect on final convolutional feature layer, depending on the location of the intervention.

In the heatmap, brighter colors indicate a stronger effect on the layer14 feature.

A request for a door has a larger effect in locations of a building, and a smaller effect near trees and sky.

At right, the magnitude of feature effects at every layer is shown, measured by mean normalized feature changes.

In the line plot, feature changes for interventions that result in human-visible changes are separated from interventions that do not result in noticeable changes in the output.

Characterizing contextual relationships using insertion We can also learn about the operation of a GAN by forcing units on and inserting these features into specific locations in scenes.

Figure 4 shows the effect of inserting 20 layer4 causal door units in church scenes.

We insert units by setting their activation to the mean activation level at locations at which doors are present.

Although this intervention is the same in each case, the effects vary widely depending on the context.

The doors added to the five buildings in Figure 4 appear with a diversity of visual attributes, each with an orientation, size, material, and style that matches the building.

We also observe that doors cannot be added in most locations.

The locations where a door can be added are highlighted by a yellow box.

The bar chart in Figure 4 shows average causal effects of insertions of door units, conditioned on the object class at the location of the intervention.

Doors can be created in buildings, but not in trees or in the sky.

A particularly good location for inserting a door is one where there is already a window.

Tracing the causal effects of an intervention To investigate the mechanism for suppressing the visible effects of some interventions, we perform an insertion of 20 door-causal units on a sample of locations and measure the changes in later layer featuremaps caused by interventions at layer 4.

To quantify effects on downstream features, and the effect on each each feature channel is normalized by its mean L1 magnitude, and we examine the mean change in these normalized featuremaps at each layer.

In FIG4 , these effects that propagate to layer14 are visualized as a heatmap: brighter colors indicate a stronger effect on the final feature layer when the door intervention is in the neighborhood of a building instead of trees or sky.

Furthermore, we graph the average effect on every layer at right in FIG4 , separating interventions that have a visible effect from those that do not.

A small identical intervention at layer4 is amplified to larger changes up to a peak at layer12.Interventions provide insight on how a GAN enforces relationships between objects.

We find that even if we try to add a door in layer4, that choice can be vetoed by later layers if the object is not appropriate for the context.

@highlight

GAN representations are examined in detail, and sets of representation units are found that control the generation of semantic concepts in the output.