In this paper, we investigate learning the deep neural networks for automated optical inspection in industrial manufacturing.

Our preliminary result has shown the stunning performance improvement by transfer learning from the completely dissimilar source domain: ImageNet.

Further study for demystifying this improvement shows that the transfer learning produces a highly compressible network, which was not the case for the network learned from scratch.

The experimental result shows that there is a negligible accuracy drop in the network learned by transfer learning until it is compressed to 1/128 reduction of the number of convolution ﬁlters.

This result is contrary to the compression without transfer learning which loses more than 5% accuracy at the same compression rate.

result in [2] , the network trained from scratch can also achieve 99.78% accuracy with the extensively 23 augmented data and a long period of training.

Surprisingly, however, although the performance of 24 both networks is similar, the network trained from scratch learns much denser features of the input 25 data than the network trained by transfer learning.

We experimentally show that using standard 26 teacher-student model compression technique [3], the network trained by transfer learning can be 27 compressed to 1/128 reduction of the number of convolution filters with a negligible accuracy drop.

In contrast, the network trained from scratch loses more than 5% accuracy at the same compression 29 rate.

Previous works on the network compression have focused on the methods of compression

[3, 5, 6, 9], but our work is the first report to the best of our knowledge that the transfer learning is 31 related to the network compression.

The rest of the paper is organized as follows.

In Section 2, we explain the method of experiments and

show the compression result in Section 2.3.

Finally, we conclude with a brief discussion in Section 3.

a given input image, we can see that the TL teacher network learns much sparser features of the 64 input data than the Scratch teacher network (Figure 2 ).

The sparsity of activation in the TL teacher

@highlight

We experimentally show that transfer learning makes sparse features in the network and thereby produces a more compressible network. 