Watermarks have been used for various purposes.

Recently, researchers started to look into using them for deep neural networks.

Some works try to hide attack triggers on their adversarial samples when attacking neural networks and others want to watermark neural networks to prove their ownership against plagiarism.

Implanting a backdoor watermark module into a neural network is getting more attention from the community.

In this paper, we present a general purpose encoder-decoder joint training method, inspired by generative adversarial networks (GANs).

Unlike GANs, however, our encoder and decoder neural networks cooperate to find the best watermarking scheme given data samples.

In other words, we do not design any new watermarking strategy but our proposed two neural networks will find the best suited method on their own.

After being trained, the decoder can be implanted into other neural networks to attack or protect them (see Appendix for their use cases and real implementations).

To this end, the decoder should be very tiny in order not to incur any overhead when attached to other neural networks but at the same time provide very high decoding success rates, which is very challenging.

Our joint training method successfully solves the problem and in our experiments maintain almost 100\% encoding-decoding success rates for multiple datasets with very little modifications on data samples to hide watermarks.

We also present several real-world use cases in Appendix.

Security issues of deep learning have been very actively being studied.

It had been already demonstrated that deep learning methods are vulnerable to some carefully devised adversarial attacks BID7 BID4 BID6 BID1 .

At the same time, many researchers are also studying about how to make them more robust against such attacks.

A couple of recent works, for example, proposed to use watermarks BID9 BID0 to protect neural networks.

At the same time, other work wanted to use a similar watermark technique to attack neural networks BID9 .The method of adding watermarks to data samples can be used in various ways to protect deep learning models.

First, the decoder can be implanted into a trained deep learning model and later one can prove the ownership, when other people copied the model, by showing that the copied model reacts to one's watermarked samples.

Second, the implanted decoder may allow only legitimately watermarked samples and reject other non-watermarked samples.

In this case, only people that have the encoder can access the deep learning model.

However, there is one very strict requirement that the decoder should be tiny to minimize the incurred overheads by attaching it as part of the main deep learning model.

Similar techniques can also be used to attack neural networks.

In this paper, we do not propose any specific watermarking techniques.

Instead, we want the encoder and decoder discuss and decide their watermarking method.

Inspired from generative adversarial networks (GANs) BID3 , the encoder and decoder work for the same goal and are jointly trained.

They do not perform the adversarial game of GANs.

Their relationship is rather cooperative than adversarial in our method.

The decoder is a tiny neural network to decode watermarks and the encoder is a high-capacity neural network that can watermark samples in such a way that the tiny neural network can successfully decode.

Therefore, those two neural networks should cooperate to find such a watermarking scheme -in GANs, one neural network (generator) tries to fool the other neural network (discriminator).

Because the decoder has a limited capacity due to its tiny neural network size, the encoder should not decide the watermarking scheme alone.

The encoder should receive feedback from the decoder to revise its watermarking scheme.

After training them, one should keep the encoder in a secure place but can deploy the decoder to as many places as one wants.

We also show that our method can be used for both defences and attacks (refer to Appendix for some of these examples we implemented using our proposed method).We adopt residual blocks BID5 to design the encoder.

Each residual block of the encoder is supposed to learn f (x)+x where x is an input to the block.

One can consider f (x) as a watermark signal discovered by the joint training of the encoder and the decoder.

The signal produced by f (x) should be strong enough to be detected by the decoder but weak enough not to be detected by human eyes.

We design our training loss definition to achieve this goal.

The encoder should modify original samples to implant watermarks.

As more modifications are allowed, stronger watermarks will be implanted but they can be readily detected by human eyes.

Our loss definition has a parameter that can be set by user to limit the modifications by the encoder.

Our experiments show that we can find a well-balanced watermarking scheme that be detected only by the decoder.

We tested many different datasets: face recognition(VGG-Face Data-set), speech recognition BID11 , images with general objects BID7 , and flowers (Flowers Data-set).

Two of them are reported in the main paper with the comparison with other watermarking methods and others are introduced in Appendix.

During experiments, our methods marked 100% decoding success rates for all datasets (in at least one hyper-parameter configuration).

This well outperforms other baseline methods.

In addition, we also found that different watermarking schemes are trained for different datasets.

For instance, the encoder modified the tone of colors for the face recognition images.

For the general object images, however, the encoder explicitly marks some dots rather than modifying their color tones (see FIG1 and Figure 4 ).

This proves our goal that two neural networks cooperate to find the best suited watermarking method for each dataset.

Watermarking data samples, such as images, videos, etc., is a long-standing research problem.

In many cases, watermarking systems merge a specific watermark signal s (set by user) and a data sample x to produce a watermarked sample x , i.e., x = encode(x, s) and s = decode(x ).

In general, the signal s is secret and later used to check where the watermarked sample x is originated from.

There exist many different watermarking techniques for relational databases, images, videos, and so forth.

However, watermarking deep neural networks is still under-explored except for a couple of recent papers BID9 BID0 .

For instance, one can implant a certain signal on neural network weights -technically, this is similar to implanting a signal on a column of table for watermarking a relational database.

However, the signal on the weights will disappear after finetuning the neural network which can preserve its accuracy but reorganize its weight values.

Instead, we need a more robust way for watermarking neural networks.

To this end, a backdoor based watermarking method has been recently proposed BID9 BID0 .

In general, a backdoor means a certain malware piece that can be exploited to avoid authentication processes in computer security.

In their contexts, however, a neural network backdoor means a way to control the final prediction of a target neural network -for instance, retraining a target neural network so that it classifies a certain type of cats as dogs.

The authors want to use the backdoor mechanism to protect the ownership of a neural network.

Because the backdoor reacts to the samples specially watermarked by the owner, the proof of its ownership is available when other people copied the neural network.

Of course, if the backdoor is successfully identified and removed, the proof of the ownership is not possible.

However, this incurs additional costs and greatly decreases the motivation of copying the model.

The same watermarking technique can be used for attacks.

In BID9 , the attacker implants an attack trigger into a data sample using a simple watermarking technique and the target neural network is already compromised by the attack to make it react to their trigger.

Their goal is to induce the compromised neural network outputs a certain label encoded in the attack trigger and preferred by the attacker.

Because this paper uses a very strong watermark signal, their watermarked images are visually impaired.

Due to its strong watermarks, however, their attack shows very high success rates.

GANs are one of the most successful generative models.

They consist of two neural networks, one generator and one discriminator.

They perform the following zero-sum minimax game: DISPLAYFORM0 where p(z) is a prior distribution, G(??) is a generator function, and D(??) is a discriminator function whose output spans DISPLAYFORM1 indicates that the discriminator D classifies a sample x as generated (resp.

real).The generator tries to obfuscate the task of the discriminator by producing realistic fake samples.

We redesign the adversarial game model for our purposes.

In our case, two neural networks, one encoder and one decoder, perform a cooperative game.

A watermarking framework consists of encoder and decoder.

The encoder modifies original samples by adding a watermark signal into them and the decoder is a binary classification to detect the presence of the watermark signal.

Watermarks are used for various purposes in deep learning.

They were used for both of defenses and attacks for deep neural networks.

In our case, we are interested in developing a pair of encoder and decoder and the decoder should be pluggable to other neural networks (as in the malware piece or backdoor in computer security).

Our encoder-decoder pair can be used for both defenses and attacks (refer to Appendix for our case studies).There are several watermarking methods based on CNNs that can be described by x = encode(x, s) and s = decode(x ) BID10 .

However, existing methods do not care about the size of the decoder and we are not interested in implanting a watermark signal s into a data sample x.

We let the encoder modify x in a way that the decoder wants and the decoder performs the binary classification of watermarked or non-watermarked.

Thus, our model can be described as x = encode(x) and decode(x ) ??? {0, 1} without s. In real world applications, this binary classification decoder suffices (refer to our use cases in Appendix) and the decoder should be so tiny that it does not incur any overheads when attached to other main neural networks -this is a strong requirement especially for the backdoor based watermarking method.

Our goal is to develop a watermarking framework that consists of one large encoder (a fatty network) and one tiny decoder (a skinny network) and they should decide their own watermarking scheme without human efforts.

Our overall idea is greatly inspired by generative adversarial networks (GANs).

In GANs, there are two neural networks, generator and discriminator, that are comparable to each other in terms of their neural network capacity.

In our case, however, the encoder and decoder are highly imbalanced in their neural network capacity and they perform a cooperative game (rather than the zero-sum adversarial game of GANs).In our method, the encoder should be capable of generating simple but robust watermarked samples because the decoder has very low capacity and as a result, it may not be able to decode complicated watermarks.

Therefore, the encoder should be large and trained enough to find the watermarking .

+ Figure 1 : The proposed encoder architecture.

Based on the attention map, a series of residual blocks generate a watermark signal specific to the input sample x which will be later merged with the generated watermark signal.

All those convolutions in this encoder use the stride of 1 and the channel of 3 to maintain the identical input and output dimensions.mechanism suitable for the low-capacity decoder.

In other words, we do not teach any watermarking mechanism but let them discover on their own considering the neural network capacity difference.

The encoder (comparable to the generator of GANs) should modify original samples to implant a watermark signal.

We adopt residual blocks to design the encoder as shown in Figure 1 .

Residual blocks are proven to be effective in designing a deep architecture and adopted by many works (e.g., ResNet BID5 ).

Each residual block that can be described as x + f (x) is suitable to perform the watermarking task.

After the multiple stages of residual blocks, the encoder generates a watermark signal 1 that will be merged with the original sample x. We use the multiple residual blocks because it is very unlikely that one residual block is able to generate a robust watermark signal.

The overall watermarked sample generation process can be described as follows: DISPLAYFORM0 where x is the original sample; A is the attention map of x produced after two convolutions, one activation, and a softmax; means the Hadamard product; f i (??) represent an additive term by i-th residual block.

In particular, we use the swish activation BID14 .

Thus, one can consider our generated watermark signal is an ensemble of all those additive terms (Veit et al., 2016) .

Note that our watermark signal is generated after ignoring unimportant parts of x after the element-wise product with the attention map.

After merging the input sample x and the generated watermark, we have one post-processing block to refine the watermarked sample.

This process includes a couple of more convolutions.

In FIG0 , and 4, we show watermarking examples for various datasets.

In FIG0 , watermarks are generated for the parts where the attention map focuses on.

In FIG1 , watermakrs are dispersed over many pixels and in this case, the attention also provides similar weights for those pixels.

The decoder (comparable to the discriminator of GANs) should classify if an input sample has a watermark signal or not.

We adopt the discriminator of DCGAN BID13 (after shrinking its size) as decoder.

Its discriminator follows a standard CNN architecture.

One of its ) and (e) are generated watermarks (before being merged with images).

These are cases where watermarks are generated, aided by attention.

For (e), there is a watermark in the most lower right corner and its attention also focuses on the same area.

However, attention maps sometimes provide similar weights over almost all pixels, in which cases watermarks are scattered over all pixels -examples in FIG1 correspond to this case.advantageous is that it is very hard to identity the decoder after being implanted into a neural network model because it is tiny and uses only very standard neural operators.

We perform experiments by varying the number of convolution layers in order to find the smallest decoder configuration.

We introduce our training method.

The main training loss can be described as follows: DISPLAYFORM0 where E(??) is an encoder, and D(??) is a decoder, and x is a data sample.

This loss definition looks similar to the one in GANs.

However, we do not perform the minimax zero-sum game of GANs.

Both the encoder and decoder cooperate to find the best performing watermarking scheme.

Its equilibrium state analysis is rather meaningless because they do not perform the zero-sum adversarial game of GANs.

It is obvious that the main loss representing equation 3 will be optimized when the decoding success rate of watermarked and non-watermarked cases is 100%.

The main loss can be implemented using the cross-entropy loss as in other GANs.

In addition, we also use one more regularization term to limit the modification by the encoder.

Let L be the main loss in equation 3.

The final loss term is defined as follows: DISPLAYFORM1 where ??(??) s,t means the feature map taken after t-th convolution (after activation) before s-th maxpooling layer in the VGG19 network 2 , ?? is the maximum margin in the hinge-loss based regularization.

We allow the modification up to ??.

Note that the hinge-loss based regularization does not incur any loss up to ??.

L content compares two samples, the original sample x and the watermarked sample E(x), in terms of the feature maps created by the VGG19 network BID8 .

We found that this is better than the pixel-wise mean squared error regularization.

In our case, we add the hinge-loss to control the modification of the input sample x. If ?? is large, more modifications are allowed and as a result, our watermark signals will be more robust.

However, the modified sample can be very different from the original sample in this case, which is not a desired result.

Therefore, ?? should be adjusted very carefully.

Our training algorithm is similar to that of GANs.

The encoder and the decoder are alternately trained to collaboratively minimize L f inal .

We omit the detailed algorithm due to its similarity to the training algorithm of GANs.

We first select several neural networks and their official datasets, considering the diversity in their task and dataset types.

After that, we train the encoder-decoder network using 80% of training samples and check the decoding error rate for the remaining 20% of testing samples.

For this, we test both cases where each testing sample is watermarked or not -i.e., the decoder should successfully distinguish watermarked and non-watermarked cases for the same set of samples.

By varying the number of convolution layers in the decoder and the margin ??, we repeat the experiment.

We also test how much damage those implanted watermarks introduce to data samples.

If the watermark signal is weak, there should not be any differences between them for several popular image comparison metrics.

We introduce detailed experiment results for two neural networks in this paper and some more in Appendix.

To evaluate our method, we compare with the following watermarking techniques.

Note that our baseline selection is so extensive that all different types of watermarking methods are included.1.

In the statistical watermarking method (SWM) introduced in BID15 , authors proposed a method to hide a series of bits (set by user) in a column of table -after flattening an image to an array of pixels, this method can be applied.

It explicitly solves an optimization problem to find the weakest watermark (enough to hide the bits) and performs some statistical tests to decode the watermarked bit pattern.

We test the following two bit patterns to hide: '0101010101', and '0000100001'.

This method cannot be implemented by neural networks but we use this method only for comparison purposes.2.

Trojan in BID9 ) uses a relatively stronger watermark signal, called attack trigger in their paper, than SWM.

This papers proposed a very effective backdoor attack method and our motivation is also influenced by the paper.

We choose the following neural networks and their datasets.

All selected neural networks include their official datasets and we use them.

et al., 2015) as VGG-FACE.

It has 16 layers and its data-set is available at (VGG-Face Data-set).

A CNN model proposed in BID11 is to recognize spoken languages.

It achieves superhuman performance in recognizing spoken numbers.

It uses the dataset of pulse-code modulation (PCM) images of spoken numbers.

We also tested for the ImageNet BID7 and Flowers (Flowers Data-set) datasets.

Experiments for those other neural networks and datasets are in Appendix.

We report i) how many non-watermarked and watermarked samples are correctly recognized and ii) how much damage each watermarking method brings to data samples in each method.

We compare the proposed method with the aforementioned baseline methods.

Our method (decoder size = 3 and ?? = 0.01) marks the best decoding success rate, i.e., 100% in our method vs. 95.5% in the method of BID9 vs. 89.3% in the statistical watermarking method.

Other configurations in our method also outperform all the baseline methods.

Our method Decoder Size = 1 Decoder Size = 2 Decoder Size = 3 ?? = 0.01 ?? = 0.05 ?? = 0.1 ?? = 0.01 ?? = 0.05 ?? = 0.1 ?? = 0.01 ?? = 0.05 ?? = 0.

BID9 .

Others are watermarked by our method.

The decoder has 3 convolution layers in these examples.

Note that there are more modifications on the color tone of images as ?? increases.

For all cases, the trained decoder can successfully decode their watermarks.

Refer to Appendix for examples of watermarking other samples.

Sometimes watermarks incur irreparable damage on images, and as a result, its contents are changed a lot.

We visualize watermarked samples and measure the difference from their original images using the multi-scale structural similarity (MS-SSIM), the peak signal to noise ratio (PSNR), and the Shannon entropy increase after watermarked.

(i) are watermarked by the method of BID9 and their attack trigger signal (in the lower right corner) is very strong.

Compared to them, our methods provide much weaker watermarks.

However, our decoding success rates are much higher than other methods including BID9 .

This proves the efficacy of the joint training mechanism of the encoder and decoder.

Our method is clearly better than Trojan for both PSNR and the entropy change, i.e., 36.580 vs. 21.029 for PSNR.

SWM solves an optimization problem to find the best case to hide watermarks with the smallest changes.

Thus, its PSNR and entropy change are better than our method and Trojan.

However, SWM does not provides reliable decoding success rates.

We also checked the accuracy drop after watermarking images.

With the original images, the FR network's accuracy is 0.795576 and after watermarking them with ?? = 0.01 and the decoder with 3 convolutions, it becomes 0.797864.

After watermarking, the accuracy is slightly improved but we think this is within its error margin and not significant.

They are more or less the same.

Likewise, in almost all cases, their accuracy difference is very trivial.

Other watermarking examples are in Appendix.

For example, Figure 4 in Appendix shows several watermarking examples for the ImageNet dataset.

Watermarks in FIG1 and FIG0 are very different.

In FIG1 , watermarks are implanted in the tone of colors but in FIG0 , several small dots are explicitly marked.

This is because the encoder and decoder networks discover a suitable watermarking method for each dataset.

It is very interesting that they discover how to hide watermarks on their own.

For this SR network netork ans dataset, we repeat the same experiments as the FR case.

In general, these experiment results have the same pattern as the FR results.

In SR, SWM shows very poor decoding success rates.

Both our method and Trojan provides the rate of 100%.

Considering the large damage on samples by Trojan which will be shortly described, however, Trojan's 100% decoding success rate is rather meaningless.

In many configurations in our method, their success rates are more than 99%.

SWM marked the smallest damage but considering it very low success rates, we don't think SWM is suitable for SR.

It cannot be even implemented by neural networks.

Our method introduces less damage to samples than that of Trojan.

Especially, the PSNR of Trojan is much worse than other method.

Because the PSNR is in the log scale, those values mean huge differences.

Its MS-SSIM is also greatly damaged.

Our method shows very stable values for those three metrics.

We present a joint training method of the watermark encoder and decoder.

Our decoder is a very lowcapacity neural network and the encoder is a very high-capacity neural network.

These two skinny and fatty neural networks collaborate to find the best watermarking scheme given data samples.

In particular, we use residual blocks to build the encoder because the definition of the residual block is very appropriate for the task of watermarking samples.

We demonstrated that two different types of watermarks (one to change the color tone and the other to add dots) are found by them without human interventions.

For our experiments with various datasets, our method marked 100% decoding success rates, which means our tiny decoder is able to distinguish watermarked and non-watermarked samples perfectly.

We also listed three use cases in Appendix about how to utilize our proposed encoder and decoder for real-world attacks and defenses.

Our future research will be to implement those use cases.

Figure 4: Examples of watermarking ImageNet images.

Some dots are marked explicitly to hide watermarks when ?? >= 0.05.

Recall that watermarks are hidden in the tone of colors for FR images.

This is a very interesting point because our proposed method can discover two very different watermarking schemes for them.

This is because adding dots does not make the regularization term greatly exceed the margin ??.

When ?? = 0.01, a similar watermarking scheme to the FR exmaples will be used.

This proves that our method is able to fine the best suited watermarking scheme given data samples.

The decoder has 3 convolution layers in these examples.

Note that there are more modifications in general as ?? increases.

For all cases, the trained decoder can successfully decode their watermarks.

Figure 5 : The decoding success rate in the ImageNet dataset.

We report the decoding success rate for non-watermarked/watermarked cases with our method after varying the convolution numbers in the decoder (i.e. decoder size) and ??.

Our method Decoder size = 1 Decoder size = 3 ?? = 0.01 ?? = 0.05 ?? = 0.1 ?? = 0.01 ?? = 0.05 ?? = 0.1 81.2%/100.0% 89.2%/100.0% 92.0%/100.0% 99.0%/100.0% 98.0%/99.4% 99.5%/100.0% A ADDITIONAL EXPERIMENT RESULTS

We introduce additional experiments that were removed from the main paper.

In Table 5 , we report the decoding success rate for the ImageNet dataset.

In all configurations, their success rates are very high.

In particular, the decoder with 3 convolution layers provides the highest decoding success rate.

DISPLAYFORM0 label ??? sof tmax(logit) Figure 4 shows several watermarked and non-watermarked samples.

With ?? = 0.01, modifications are very limited.

In Figure 4 (e), it is very hard to recognize its watermark with human eyes, but the decoder can detect its hidden watermark signal surprisingly.

The smallest decoder with only one convolution works well too.

However, its decoding success rates are smaller than that of the decoder with three convolutions.

We also test the flower images in (Flowers Data-set).

We choose this dataset to test with various types of images.

We tested with face and object images.

Flower images have different characteristics from the previous image datasets.

In the decoder size of 3, the decoding success rates are very high for all ?? configurations.

When there is only one convolution in the decoder, the decoding success rate is proportional to ??.

In this section, we introduce three use cases to both attack and protect neural networks.

The first use case is to utilize the proposed encoder and decoder for backdoor attacks.

The second use case is to allow only legitimately watermarked input samples and comparable to the admission control in operating systems and computer networks, and the last use case is to prove the ownership using the proposed watermarking technique.

The backdoor attack in the context of machine learning means that the attacker modifies a target model and the modified model reacts to samples specially marked by the attacker -the attacker may redistribute it after the modification, and careless users may download and use it.

The special marker is called attack trigger and it contains a target label that is different from its ground-truth label but preferred by the attacker.

The attack trigger is usually implanted using a watermarking method.

We demonstrate that how the attacker utilize our encoder and decoder networks.

We first describe the proposed decode&inject module and how to attach it to the target neural network.

The code snippet in the left column of TAB5 represents a typical multi-class image classification target neural network -we use this image classification neural network as an example but our attack can be applied to any other neural networks.

We attach the module into one of its convolution layers as shown in the right column of the table, i.e., c i ??? relu(conv(c i???1 )) + decode&inject(x) where x is an image and c i is feature maps in i-th convolution layer.

Note that the module reads the input image x and outputs a tensor whose dimensionality is the same as that of the convolution layer if watermarked by the attacker.

Thus, the role of the module is i) decoding the watermark signal and ii) injecting a signal (tensor) to the target neural network to control its final output.

If not watermarked, the module should inject a zero tensor (i.e., keep silent).

The module can be defined as follows: DISPLAYFORM0 It should outputs 0 if no watermark, i.e., x is a non-modified image.

Because of this, the module has zero influences on the target neural network for non-modified images.

If x has a certain watermark, it should output a corresponding tensor w for the label preferred by the attacker.

For instance, all watermarked images with cats can be classified as dogs with the additional feature map w injected to the target neural network.

All the convolution, linear, softmax are initialized and fixed with the weights of the original target neural network and our trained decoder, and we train only w. After being trained, the module can inject a trained feature map w that is able to control the final softmax outputs, i.e., class labels.

The implementation of decode&inject(x) is very straightforward.

On top of the proposed decoder, one trick to implement an if-else statement is enough to make the module fully functioning.

We attacked the neural networks of FR and SR, and the following one more using the proposed backdoor attack mechanism based on our encoder-decoder neural networks.

Inception-v4 Network (IN) Inception-v4 Network is a CNN-based classifier developed by Google BID16 .

It uses inception modules to make training very deep networks very efficient.

We use the Flowers dataset released in (Flowers Data-set).To evaluate the proposed attack, we followed the steps used in BID9 .

A backdoor modification proposed by BID9 makes the target neural network react to their watermarked attack trigger and output the preferred label by the attacker -this is the same as our decode&inject module.

We first prepare the modified target neural network where the decode&inject module is attached 3 .

To perform attacks, we use the original testing set for each target neural network.

Each sample is attacked multiple times for all non-ground-truth labels.

Their attack success rates are summarized in TAB6 .

As you see, our method provides better success rates than the other stateof-the-art method.

The same method can be used for admission control.

For this, we can use the following module that reject or forward input samples to target neural networks.

reject or bypass(x) = 0 if no watermark x if watermark exists on xThe module in equation 6 says that x will be delivered to the target neural network only if x is properly watermarked.

The implementation of the module is similar to that of the backdoor attack.

However, we do not need to train w in this case.

Recall that our watermarks did not decrease the accuracy for both FR and SR netural networks.

This property of no (or very little) accuracy drop is required to use the watermarking method for admission control.

Our method meets the requirement.

The proposed decode&inject module can be used to prove the ownership of neural networks.

One can implant the module in the way we described and later use it to prove the ownership against the plagiarism of neural networks.

If other people copy the neural network protected by our watermarking method, you can show that the copied neural network reacts to your watermarked samples and prove that the copied neural network is originally designed by you.

<|TLDR|>

@highlight

We propose a novel watermark encoder-decoder neural networks. They perform a cooperative game to define their own watermarking scheme. People do not need to design watermarking methods any more.