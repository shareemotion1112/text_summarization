Feedforward convolutional neural network has achieved a great success in many computer vision tasks.

While it validly imitates the hierarchical structure of biological visual system, it still lacks one essential architectural feature: contextual recurrent connections with feedback, which widely exists in biological visual system.

In this work, we designed a Contextual Recurrent Convolutional Network with this feature embedded in a standard CNN structure.

We found that such feedback connections could enable lower layers to ``rethink" about their representations given the top-down contextual information.

We carefully studied the components of this network, and showed its robustness and superiority over feedforward baselines in such tasks as noise image classification, partially occluded object recognition and fine-grained image classification.

We believed this work could be an important step to help bridge the gap between computer vision models and real biological visual system.

It has been long established that the primate's ventral visual system has a hierarchical structure BID5 including early (V1, V2), intermediate (V4), and higher (IT) visual areas.

Modern deep convolutional neural networks (CNNs) for image recognition BID10 BID18 trained on large image data sets like ImageNet (Russakovsky et al., 2015) imitate this hierarchical structure with multiple layers.

There is a hierarchical correspondence between internal feature representations of a deep CNN's different layers and neural representations of different visual areas BID3 BID25 ; lower visual areas (V1, V2) are best explained by a deep CNN's internal representations from lower layers (Cadena et al., 2017; Khaligh-Razavi & Kriegeskorte, 2014) and higher areas (IT, V4) are best explained by its higher layers (Khaligh-Razavi & Kriegeskorte, 2014; BID24 .

Deep CNNs explain neuron responses in ventral visual system better than any other model class BID25 BID9 , and this success indicates that deep CNNs share some similarities with the ventral visual system, in terms of architecture and internal feature representations BID25 .However, there is one key structural component that is missing in the standard feedforward deep CNNs: contextual feedback recurrent connections between neurons in different areas BID5 .

These connections greatly contribute to the complexity of the visual system, and may be essential for the success of the visual systems in reality; for example, there are evidences that recurrent connections are crucial for object recognition under noise, clutter, and occlusion BID14 BID19 BID15 .In this paper, we explored a variety of model with different recurrent architectures, contextual modules, and information flows to understand the computational advantages of feedback circuits.

We are interested in understanding what and how top-down and bottom-up contextual information can be combined to improve in performance in visual tasks.

We investigated VGG16 BID18 , a standard CNN that coarsely approximate the ventral visual hierarchical stream, and its recurrent variants for comparison.

To introduce feedback recurrent connections, we divided VGG16's layers into stages and selectively added feedback connections from the groups' highest layers to their lowest layers.

At the end of each feedback connection, there is a contextual module (Section 3.2) that refines the bottom-up input with gated contextual information.

We tested and compared several networks with such contextual modules against VGG16 in several standard image classification task, as well as visual tasks in which refinement under feedback guidance is more likely to produce some beneficial effects, such as object recognition under degraded conditions (noise, clutter and occlusion) and fine-grained recognition.

We found that our network could outperform all the baseline feedforward networks and surpassed them by a large margin in finegrained and occlusion tasks.

We also studied the internal feature representations of our network to illustrate the effectiveness of the structure.

While much future work has to be done, our work can still be an important step to bridge the gap between biological visual systems and state-of-the-art computer vision models.

Although recurrent network modules including LSTM BID6 and Gated Recurrent Unit BID2 have been widely used in temporal prediction BID23 and processing of sequential data (e.g. video classification BID4 ), few studies have been done to augment feedforward CNNs with recurrent connections in image-based computer vision tasks.

Image classification.

Standard deep CNNs for image classification suffer from occlusion and noise BID21 b; BID27 , since heavy occlusion and noise severely corrupt feature representations at lower layers and therefore cause degradation of higher semantic layers.

With the inclusion of feedback connections, a model can "rethink" or refine its feature representations at lower layers using feedback information from higher layers BID11 ; after multiple rounds of feedback and refinement, input signals from distracting objects (noise, irrelevant objects, etc.) will be suppressed in the final feature representation BID1 .

BID11 used the output posterior possibilities of a CNN to refine its intermediate feature maps; however, their method requires posterior possibilities for refinement and thus cannot be applied in scenarios where supervision is absent.

BID7 used more global and semantic features at higher convolutional layers to sharpen more local feature maps at lower layers for image classification on CIFAR datasets; however, our own experimentation suggests that this method only works when the higher and lower layers have a relatively small semantic gap (similarly sized receptive fields); on high- Other computer vision tasks.

BID12 designed a model with explicit horizontal recurrent connections to solve contour detection problems, and BID19 evaluated the performance of various models with recurrent connections on digit recognition tasks under clutter.

The tasks evaluated in these studies are rather simple and contrived, and it remains to be seen whether their models and conclusions can apply to real world computer vision problems.

BID11 uses posterior possibilities at the last fully connected layer to select intermediate feature map representations; however, the posterior possibility vector is not informative enough and the input of the feedback connection is totally fixed, which makes it less flexible to fully mimic the recurrent connections in the visual system.

Overall, feedback and recurrent connections are present in multiple layers of the visual hierarchy, and this study constrains feedback connections to the output classification layer only.

It is worth noting that a recent study BID13 is motivated by recurrent connections in the brain as well; however, their work focuses on exploring the computational benefits of local recurrent connections while ours focuses on feedback recurrent ones.

Thus, we believe that our work is complementary to theirs.

In this section, we will describe the overall architecture of our proposed model and discuss some design details.

The main structure of our Contextual Recurrent Convolutional Network (CRCN) is shown in Figure 1.

A CRCN model is a standard feedforward convolutional network augmented with feedback connections attached to some layers.

At the end of each feedback connection, a contextual module fuses top-down and bottom-up information (dashed red lines in FIG0 ) to provide refined and sharpened input to the augmented layer.

Given an input image, the model generates intermediate feature representations and output responses in multiple time steps.

At the first time step (t = 0 in FIG0 , the model passes the input through the feedforward route (black arrows in FIG0 ) as in a standard CNN.

At later time steps (t > 0 in FIG0 ), each contextual module fuses output representations of lower and higher layers at the previous step (dashed red lines in FIG0 ) to generate the refined input at the current time step (red lines in FIG0 ).

Mathematically, we have DISPLAYFORM0 where S G is the index set of layers augmented with feedback connections and contextual modules, c k (??, ??) (detailed in Eqs. (2) ) is the contextual module for layer k, Ok denotes the output of layer k at time t, h(??) is a function that maps the index of an augmented layer to that of its higher feedback BID7 , VGG-LR-2 means the "rethinking" one-FC-layer VGG model with 2 unrolling times proposed in BID11 .

CRCN-n means our 2-recurrentconnection model with n unrolling times.layer, and f k (??) denotes the (feedforward) operation to compute the output of layer k given some input.

The key part of the Contextual Recurrent Convolutional Network model is the contextual module at the end of each feedback connection.

FIG2 shows one possible design of the contextual module, which is inspired by traditional RNN modules including LSTM BID6 and Gated Recurrent Unit BID2 .

In this scheme, a gate map is generated by the concatenation of the bottom-up and the (upsampled) top-down feature map passing through a 3 ?? 3 convolution (black circle with "C" and black arrows with circle).

Then a tanh function is applied to the map to generate a gate map.

The gate map then controls the amount of contextual information that can go through by a point-wise multiplication (red lines).

To make the information flow more stable, we add it with bottom-up feature map (black circle with "+").

The equations are presented in Eqs. (2).

Then we use this new feature representation to replace the old one and continue feedforward calculation as described in Section 3.1.

DISPLAYFORM0

Since there exists a gap between the semantic meanings of feature representations of bottom-up and top-down layers, we argue that recurrent connection across too many layers can do harm to the performance.

Therefore, we derive three sets of connections, conv3 2 to conv2 2, conv4 2 to conv3 3, and conv5 2 to conv4 3 respectively.

It is worth noting that all these connections go across pooling layers, for pooling layers can greatly enlarge the receptive field of neurons and enrich the contextual information of top-down information flow.

For information flow in networks with multiple recurrent connections, take the network structure in FIG1 as an example.

The part between conv2 2 and conv5 2 will be unrolled for a certain number of times.

To make the experiments setting consistent, we used model with two recurrent connections(loop1 + loop2) in all the tasks.

We first tested the Contextual Recurrent Convolutional model on standard image classification task including CIFAR-10, CIFAR-100, ImageNet and fine-grained image classification dataset CUB-200.

BID11 .

As the attack gets stronger, our model shows more robustness.

VGG-small 64.88 VGG-ATT BID7 73.19 VGG-LR-2 BID11 72.99 VGG-CRCN-2 74.90 Table 2 : Top-1 accuracy on CUB-200 datasets.

Model Occlusion VGG-small 34.50 VGG-ATT BID7 46.57 VGG-LR-2 BID11 45.88 VGG-CRCN-2 50.70 Table 3 : Top-1 accuracy on Occlusion datasets.

To display the robustness of our model, we showed its performance on noise image classification, adversarial attack and occluded images.

We found that our model achieved considerate performance gain compared with the standard feedforward model on all these tasks.

Notice that our proposed models are based on VGG16 with 2 recurrent connection(loop1+loop2 in FIG1 ) in all the tasks.

CIFAR-10: Because CIFAR-10 and CIFAR-100 datasets only contain tiny images, the receptive fields of neurons in layers beyond conv3 2 already cover an image entirely.

Although the real power of contextual modulation is hindered by this limitation, our model can still beat the baseline VGG16 network by a large margin (Second column in Table 1 ).

Our model also compared favorably to two other recent models with recurrent connections.

Again, our models showed better results.

Based on the assumption that contextual modulation can help layers capture more detailed information, we also tested our model on CIFAR-100 dataset, which is a 100-category version of CIFAR-10.

Our model got a larger improvement compared with feedforward and other models (The third column in Table 5 : Noise image classification top-1 accuracy on different loop locations.

Loop1 corresponds to the first feedback connection in FIG1 .

The same for Loop2, 3, 1+2, 2+3 and 1+2+3.

ImageNet: ImageNet BID10 is the commonly used large-scale image classification dataset.

It contains over 1 million images with 1000 categories.

In this task, to test the robustness of our model, we added different levels of Gaussian noise on the 224px??224px images in the validation set and calculated the performance drop.

In detail, we used the two recurrent connection model for this task(loop1+loop2 in FIG1 ).

Notice that all models are not trained on noise images.

The result of top1 error without any noise is shown in Table 7 .

We found that the performance gap between our model and feedforward VGG model got larger as the noise level increased.

Results are shown in FIG3 .

Also, we showed the noise ImageNet top-1 accuracy of our model, BID11 's model and feed-forward model in Table 8 .Additionally, we also tested adversarial attacks on our model.

FIG3 shows the results with different L ??? norm coefficient.

We also found that our model had much lower fooling rates than feedforward model and BID11 's model with the increasing of the norms, which successfully proved our model's robustness.

We argued that the contextual module can help the network to preserve more fine-grained details in feature representations, and thus we tested our model on CUB-200 fine-grained bird classification dataset BID20 .

We used the same model as ImageNet classification task which indicates that our model contains two recurrent connection(loop1+loop2 in FIG1 .

As a result, our model can outperform much better than the feed-forward VGG model BID26 and other similar models with the same experimental settings.

The result is shown in 2.

To further prove the robust ability of our model, we tested our model on VehicleOcclusion dataset BID22 , which contains 4549 training images and 4507 testing images covering six types of vehicles, i.e., airplane, bicycle, bus, car, motorbike and train.

For each test image in dataset, some randomly-positioned occluders (irrelevant to the target object) are placed onto the target object, and make sure that the occlusion ratio of the target object is constrained.

One example is shown in FIG5 .

In this task, we used multi-recurrent model which is similar with the model mentioned in Imagenet task.

Here, we found that our model can achieve a huge improvement, which is shown in 3.

We implemented all the possible combinations of recurrent connections listed in FIG1 .

We denote connection from conv3 2 to conv2 2, conv4 2 to conv3 3, and conv5 2 to conv4 3 as Loop 1, Loop 2 and Loop 3, respectively.

The same naming scheme goes for Loop 1+2 and Loop 1+2+3, etc.

We tested altogether 6 different models on the noise classification experiment, the settings of which were completely the same.

In Table 5 , by comparing the corresponding columns where one more recurrent connection is added, we can find that having more loops yields better classification accuracy and robustness, consistent with the reciprocal loops between successive layers in the hierarchical visual cortex.

Especially, we can also find that the importance of Loop 1 is slightly better than Loop 2 and Loop 3, indicating the early layers may benefit more from the additional contextual information as an aid.

In additional to the original contextual module in FIG2 , we implemented three other structures that we thought were all reasonable, so as to further study the effect and importance of top-down information and contextual modulation.

Briefly, we refer Module 1 to the scheme that top-down feature map gating contextual map, Module 2 to contextual map gating contextual map itself, Module 3 to the scheme that top-down feature map gating contextual map, as well as contextual map gating top-down feature map, and afterwards the two gating results are added together.

The final output of all three modules are the gating output added by bottom-up feature map.

By "contextual map", we mean the concatenation of top-down and bottom-up feature map undergone a 3??3 convolution layer.

By "gating", we mean the gated map element-wisely multiplied with the Sigmoid responses of the gate map.

For formulas and further details of the three module structures, we guide readers to read the supplementary materials.

Models VGG16 We did the same noise image classification experiments on these different contextual modules to give a comparison.

We use the Loop 1+2 model as the remaining fixed part.

The performance of these modules are listed in FIG3 .

The differences among these contextual modules lie in how the gate map is generated and what information is to be gated.

The best model is obtained by generating the gate map from contextual map and then use it to gate top-down information.

By comparing it with Module 1, we find that using only top-down information to generate the map and control total data flow is not adequate, possibly because top-down information is too abstract and coarse.

By comparing the best module with Module 2, we find that only top-down information is necessary to be gated.

A direct addition of bottom-up map with the output of the gate is adequate to keep all the details in lower level feature maps.

We drew t-SNE visualization of feature representations of both final fully connected layers and layers with recurrent connections attached (e.g. conv2 2, conv3 3, conv4 3).

We selected 5 out of 1000 categories from ImageNet validation set.

To effectively capture the changes of feature representations of intermediate convolutional layers, we used ImageNet bounding box annotations and did an average pooling of all the feature responses corresponding to the object bounding box.

By comparing the representations of both networks, we can find that the Contextual Recurrent Network is able to form a more distinct clustering than VGG16 network.

Notice that we also tested the presentation when a high noise (standard deviation equal to 30) is added to the images.

We can find a consistent improvement over VGG16 network in both intermediate representations and representations directly linked to the final classification task.

The results are shown in FIG4 .

There is another finding that the contextual module dynamics in recurrent connections not only helps to refine the low-level feature representation during inference, it can also refine the feedforward weights, resulting in better performance in computer vision tasks even in the first iteration, acting as a regularizer.

The results are shown in Table 6 .

In this paper, we proposed a novel Contextual Recurrent Convolutional Network.

Based on the recurrent connections between layers in the hierarchy of a feedforward deep convolutional neural network, the new network can show some robust properties in some computer vision tasks compared with its feedforward baseline.

Moreover, the network shares many common properties with biological visual system.

We hope this work will not only shed light on the effectiveness of recurrent connections in robust learning and general computer vision tasks, but also give people some inspirations to bridge the gap between computer vision models and real biological visual system.

VGG16 BID18 71.076 VGG-LR-2 BID11 71.550 VGG-CRCN-2 71.632 Table 7 : ImageNet classification top-1 accuracy.6 SUPPLEMENTARY MATERIALS

We tested three other possible contextual modules in Section 4.

Here are the detailed formulations of the three modules.

DISPLAYFORM0 In the module described by Eqs. (3), we first generated the gate by the top-down layer.

Then we used the gate to control the contextual information generated by concatenating bottom-up layer and top-down layer.

To stable the information flow, we added it with the bottom-up layer.

In the module described by Eqs. (4), we first generated the gate by contextual information which is the same as our proposed module.

Then we used the gate to control the contextual information itself which we thought was a feasible way to store the largest information.

To stable the information flow, we also added it with the bottom-up layer.

We generated two gates by both contextual information and top-down layer in the module described by Eqs. (5).

Then we used the gate contextual to control the top-down information and used the gate to control the contextual information.

To stable the information flow, we also added it with the bottom-up layer.

In this section, we showed some examples of image occlusion task and adversarial noise task.

In the left of FIG5 , we showed one image occlusion example.

And we showed one adversarial noise example in the right of FIG5 .

We can see the noise is not obvious to the human eyes but can lead a significant influence to the neural network.

We used Fast Gradient Sign Non-target to generate the noise.

The left is the original image and the right one is the image adding the noise.

Table 8 : Noise image classification top-1 accuracy on Imagenet.

In Table 7 , we showed the Imagenet Top1 accuracy results.

Notice that we did not compare our model with VGG-ATT model proposed in BID7 because their model is not reasonable on high resolution image dataset.

Therefore, their model cannot extract effective attention map from the ImageNet images.

In Table 8 , we showed the Imagenet Top1 accuracy results with different level of Gaussian noise.

VGG16 here means the standard VGG16 model.

Notice that we also compared our model with BID11 's model which we name "VGG-LR-2".

@highlight

we proposed a novel contextual recurrent convolutional network with robust property of visual learning 

@highlight

This paper introduces feedback connection to enhance feature learning through incorporating context information.

@highlight

The paper proposes to add "recurrent" connections inside a convolution network with gating mechanism.