When deep learning is applied to sensitive data sets, many privacy-related implementation issues arise.

These issues are especially evident in the healthcare, finance, law and government industries.

Homomorphic encryption could allow a server to make inferences on inputs encrypted by a client, but to our best knowledge, there has been no complete implementation of common deep learning operations, for arbitrary model depths, using homomorphic encryption.

This paper demonstrates a novel approach, efficiently implementing many deep learning functions with bootstrapped homomorphic encryption.

As part of our implementation, we demonstrate Single and Multi-Layer Neural Networks, for the Wisconsin Breast Cancer dataset, as well as a Convolutional Neural Network for MNIST.

Our results give promising directions for privacy-preserving representation learning, and the return of data control to users.

The healthcare, finance, law and government industries often require complete privacy and confidentiality between various stakeholders and partners.

With the advent of highly effective AI using deep learning, many real-world tasks can be made more effective and efficient in these industries.

However deep learning approaches are seldom performed with privacy preservation in mind, let alone with the encryption of information throughout the entire process.

As a result, current deep learning implementations often cannot be used for these confidential applications.

Homomorphic Encryption (HE) BID28 offers an opportunity to address the privacy preservation gap, for data processing in general and deep learning in particular.

HE can be used to perform computation on encrypted information BID26 , without ever having access to the plaintext information.

Our work combines the paradigms of deep learning and homomorphic encryption, allowing improved privacy for existing server-side models , and thus enabling many novel, intelligent, privacy-guaranteeing services.

Figure 1: General overview of our privacy-preserving method for deep learning.

Encrypted inputs are fed into our hybrid model on the server-side, and this produces encrypted outputs.

Consider the following scenario: Some organization has created a deep learning solution, which can solve or assist in an important problem such as medical diagnosis, legal discovery or financial review.

However they cannot release their model or parameters for client-side use, for fears that their solution could be reverse-engineered.

At the same time, the client is unable or unwilling to send private information to other parties, which seemingly makes a server-sided application impossible.

To preserve the client's privacy, they would have to send their information encrypted, in such a way that only the client could decrypt it.

Under a typical encryption scheme, the organization could not process this input in any meaningful way.

Any complete solution to this problem would require that the client's input is not revealed to the server, the server's model and weights are not revealed to the client, and that arbitrarily deep models with common operations, such as convolution and ReLU, can be supported by the server.

Our approach uses Fully Homomorphic Encryption (FHE), since the client's data can be processed confidentially, in any desired fashion.

We will also learn that there are additional possibilities for making this approach efficient, without sacrificing security or functionality.

Both deep learning and FHE have seen significant advances in the past ten years, making them practical for use in production.

Partially homomorphic encryption (PHE) has enabled multiplication BID11 or addition BID25 between encrypted integers.

However it was not possible to perform both of these operations under the same encryption system.

This resulted in cryptosystems that were not Turingcomplete.

However the cryptosystem from BID13 allowed both multiplication and addition to be performed flawlessly on encrypted integers.

In theory, this system alone could be used to compute any arithmetic circuit.

However in practice the bootstrapping operations were incredibly slow, taking up to 30 minutes in some cases BID12 .

To improve the speed of FHE, several approaches were developed.

The "bootstrapping" procedure is what allows for arbitrarily many sequential FHE operations.

However the vast majority of time in any operation was spent performing bootstrapping, making it the key bottleneck.

BID4 made use of Leveled FHE, which eliminated the bootstrapping procedure, but limited the system to a finite number of sequential operations.

Within these limits however, operations could be performed relatively quickly.

BID3 made use of Ciphertext Packing, allowing cryptosystems to pack many individual messages into a single encrypted message, increasing overall throughput.

FHEW BID10 takes a different approach, reducing the problem down to the NAND operation on binary numbers.

This reduced the bootstrapping time to under 0.5 seconds.

Figure 3: More detailed example of FHE.

After performing FHE operations, ciphertexts may be bootstrapped, so that they can be used for subsequent operations.

However this operation is costly.

TFHE BID5 furthers this approach, reducing the bootstrapping time to under 0.1 seconds, while implementing many additional logic gates such as AND, OR, and XOR.

Both FHEW and TFHE have open-source software implementations, and the FHEW implementation has since added some of the improvements and gates from BID5 .

Mohassel & Zhang (2017) and BID20 do not use homomorphic encryption for online operations, and instead use a combination of garbled circuits and secret sharing.

This approach requires more communication between the client and server(s), and can reveal the structure of a model to the client.

BID20 proposes a form of model obfuscation to alleviate this, but no further details are given.

BID27 allows multiple parties to collaboratively build a model, by securely sharing training information.

BID23 trains a model with secure weights, by obtaining secure training information from a client.

We do not explore these concepts, since our problem scenario assumes that the server has exclusive access to the model structure and weights.

BID24 implemented a number of old deep learning models under PHE, however this system computes activation functions in plaintext, compromising security and making it invalid for our problem scenario.

More recent work has implemented deep learning models under FHE BID30 BID8 .

However these approaches have a variety of limits, such as a Leveled FHE system which limits model depth, and ineffective replacements for common activation functions.

In general, these systems do not have the capacity to implement state-of-the-art models.

BID8 also uses plaintext parameters to reduce processing time, however the techniques used are specific to their cryptosystem.

Our approach uses plaintext weights more directly, since we can exploit the properties of binary logic gates.

We decided to design deep learning functions for a binary FHE system, using bootstrapping.

This is because deep learning operations can require arbitrarily many sequential operations, which would be difficult to implement with leveled FHE.

Furthermore we want to support activation functions like ReLU, which can only be realistically achieved in FHE using a binary representation of the number.

If we used a non-binary system, the number would have to be decomposed into binary for every comparison, which would be extremely inefficient.

Both TFHE and FHEW provide bootstrapped, boolean operations on binary inputs, with open-source software implementations, and thus we decided to support both of these, by abstracting shared concepts such as ciphertexts, logic gates, encryption, decryption and keys.

We also modified FHEW to implement XOR, a critical logic gate for any efficient implementation of our design.

This design should also be able to support any future binary FHE system.

We propose a general, hybridized approach for applying homomorphic encryption to deep learning (see Figure 1) , which we call Hybrid Homomorphic Encryption (HHE).

This approach greatly improves the efficiency and simplicity of our designs.

One important observation with binary bootstrapped FHE, is that a complete bootstrapping operation must be performed whenever all inputs to a logic gate are encrypted (ciphertexts).

When the inputs to a logic gate are all unencrypted inputs (plaintexts), there is clearly no bootstrapping operation required.

However when the input to a logic gate is one ciphertext and one plaintext, we still know enough information to not require a bootstrapping procedure.

Consider the NAND gate in FIG1 .

If input A is a plaintext 1, then the output is always NOT B, while if the input is a plain 0, then the output is always 1.

Even with the XOR gate in FIG1 , we can execute the gate without knowing any information about B, other than its inverse.

Crucially, both FHEW and TFHE can perform the NOT operation almost instantaneously, without needing to perform the bootstrapping procedure.

Given our problem scenario, where the model is being hosted on a server, any parameters supplied by the server can be held in plaintext.

Deep learning inferences largely involve multiplying inputs with static weights, so we can store the model weights in plaintext, and exploit this hybrid approach.

In order to implement a deep learning model from logic gates, we must first build adder and multiplier circuits.

Conveniently, these can largely be constructed out of a Hybrid Full Adder (HFA) circuit, as shown in FIG3 .

When three plaintexts or three ciphertexts are added, a typical plaintext or ciphertext full-adder can be used.

However when two plaintexts and one ciphertext are added, we use a half-adder for the plaintexts, then approach the final sum and carry in a similar fashion to the hybrid XOR from Section 3.1.

As a result, no bootstrapping homomorphic gates are needed, despite the use of one ciphertext.

When two ciphertexts and one plaintext are added, we can first half-add the ciphertexts together.

If the plaintext is 0, we can just output the half-adder's sum and carry results.

If the plaintext is 1, we can apply OR to the two carries.

Here we only used two bootstrapping gates (from the half-adder), with one additional gate if the plaintext is 1.These sub-units of the HFA cover all possible combinations of plaintexts and ciphertexts.

In a number of cases, we use no bootstrapping gates, but in the worst case (all ciphertexts) we use 5 bootstrapping gates (from the full-adder).

An N-bit adder circuit is shown in FIG4 , which minimizes the number of gates required by using a half-adder for the first input, and using just two XORs for the final adder, when no overflow is required.

Since we are creating a software implementation, we can iteratively loop through the middle section, enabling variable bit-depths.

If we make use of HFAs, the first carry-in can be a plaintext 0, while 2 Hybrid XORs can be used for the output.

The implicit optimizations afforded by the HFA can be used to simplify a software implementation of this circuit, and we find that this also applies to other circuits with half-adders, full-adders or implicit plaintext bits.

In the realm of physical circuits, adder circuits can be made more efficient by parallelizing sections of the circuit.

However this is not necessarily beneficial for our implementation, as the key source of improved performance will come from minimizing the number of bootstrapping gates.

We use the matrix from BID0 to structure an efficient multiplier for two's complement numbers.

First we produce partial products by applying AND to each bit in one input, against every bit in the other input.

FIG5 shows a 4-bit matrix where partial products, their complements and binary digits are included in specific locations.

Each row is then added in turn, with the most significant bit of the final output being flipped.

The result of this will be the multiplied number, with the combined bit-depth of the original numbers.

Importantly, the matrix can be scaled to arbitrary bit-depths as explained in BID0 .

Because the HFAs can efficiently handle plaintext inputs, we can populate the matrix with plaintext 0s and 1s.

We want to include the overflow of each addition, so we can simply use an array of HFAs to add each row in turn, minimizing the number of bootstrapping operations, and leading to a surprisingly simple software implementation.

If we are multiplying one plaintext number with one ciphertext number, like in our problem scenario, it is important to consider the computation of partial products.

Since 0 AND anything is 0, any time a 0 appears in the plaintext bit-string, that row in the matrix will be nearly all 0s, resulting in almost no bootstrapping operations.

In fact when multiplying by 0 or 2 n , almost no bootstrapping operations are required.

Since more plaintext 0s in an input bit-string is clearly desirable, we two's-complement both the plaintext and the ciphertext, whenever the number of 1s in the plain bit-string is greater than the number of 0s.

As a result, cases of −2 n also make use of very few bootstrapping gates.

In general, the speed of this multiply operation is affected by not only the bit-depth of the input, but also the degree to which it is all-0s or all-1s.

The worst-case plaintext input for this multiplier then, would be all numbers with bit-strings holding an equal number of 0s and 1s.

It is interesting to note that the properties of this hybrid Baugh-Wooley multiplier are similar to a plaintext Booth Multiplier BID2 , which excels on inputs with long "runs" of 1s or 0s.

In general, these properties could potentially be optimized for in deep learning models, where model weights are adjusted to increase the "purity" of the bit-strings.

Given the implicit optimizations afforded by the HFA inside both the adder and multiplier circuits, it is possible that other digital circuits could be implemented cleanly and efficiently with this approach.

Deep learning models typically apply the ReLU activation to intermediate layers, with a Sigmoid activation on the final layer.

Variations such as the leaky ReLU or the hyperbolic tangent could also be used BID21 .

Any differentiable, nonlinear function could be used as an activation function in principle, and BID8 uses the square (x 2 ) activation in their FHE implementation, however this tends to result in less accurate models, or models which fail to converge.

The square activation also requires that two ciphertexts be multiplied, which leads to a much slower operation.

DISPLAYFORM0 HybridAN D all other bits in H with S not 7: H sign ← plaintext 0 8: return H Because our numbers are represented in two's complement binary, ReLU can be implemented efficiently without approximation, as shown in Algorithm 1.

A HybridText here refers to a bit being possibly plaintext or ciphertext, so a mixed bit-string is allowed.

As observed in BID8 , if only the final layer uses the sigmoid activation function, we could potentially ignore this operation during model inference.

Nevertheless, we can use piecewise approximation to represent it in our system.

Appendix A describes a fast approximation of the sigmoid activation, and more generally a method of using lookup tables to achieve piecewise approximation.

The main operation inside a simple neural network is the weighted sum, or the matrix multiply in the case of batch-processing.

A simple weighted sum can be achieved by multiplying each input with a weight in turn, and adding it to an overall sum, using our adder and multiplier circuits.

Implementing an efficient matrix multiplier could be valuable for future work.

The main operation inside a Convolutional Neural Network (CNN) is the convolution operation.

We implement the naïve approach of performing a weighted sum of each pixel neighborhood, though more sophisticated implementations should be considered for future work, perhaps using fast Fourier transforms and matrix multiplies, as shown in BID22 .

Along with efficient implementations of digital circuits, we can also modify our deep learning models to require fewer bootstrapping operations.

In general, we can use similar techniques to deep learning on embedded systems.

As discussed in Section 3.4, lower precision inputs and weights can be used to directly reduce the number of necessary operations.

Because we support variable bit-depth, it could also be possible to optimize models such that more bits are allocated to weights that need them.

While deep learning frameworks generally use floating point values, the values used at any stage tend to be of similar scale, meaning we can instead use fixed-point arithmetic, and take advantage of our existing two's complement digital circuit designs.

Using separable convolutions can allow for very large kernels, at a comparable cost to traditional convolutions of small kernel size BID7 .

This simply requires convolving the horizontal component, followed by the vertical component, or vice-versa.

While deep learning models can work well on low-precision inputs, most models require normalized inputs.

If one were willing to give means and standard deviations to the client, this is a non-issue.

However if normalization is to be performed server-side, many inputs can have means or standard deviations that cannot fit into low-precision numbers.

The solution is to normalize these inputs at a higher bit-depth, such that the given inputs can be fully represented.

We can normalize encrypted inputs with our existing circuits, by adding the additive-inverse of the means, then multiplying by the multiplicative inverse of the standard deviations.

The result can then be reduced to some desired precision for use in a model, ensuring that 3-7 standard deviations can still be represented.

Our software is written in C++, and implements all the described functionality necessary to run the models in our results.

We also implemented an interpreter, such that models constructed in the Keras framework BID6 can be exported and read in our software.

Appendix B explains our software architecture in more detail.

To demonstrate this framework, we built a single-layer Neural Network (Perceptron), Multi-Layer Neural Network (MLP), and a CNN in Keras.

We built and test our perceptron and MLP on the Wisconsin Breast Cancer dataset BID19 , and the CNN for the MNIST database of handwritten digits BID17 ).The CNN is designed to closely replicate the model in BID8 , but we use ReLU instead of Square activations.

They also apply multiple intermediate, unactivated layers, which are composed into a single intermediate layer for inference.

However we avoided this step as it was unnecessary.

For the breast cancer dataset, 70% of instances (398) are used for training, while 30% of instances (171) are used for testing.

Since the dataset contains more benign samples than malignant, we also weight malignant losses more than benign losses, to help balance the training process.

For the handwritten digits dataset, 60,000 instances are used for training while 10,000 are used for testing.

Appendix C describes the structure and hyperparameters of each model in greater detail.

We measured the performance of various arithmetic and deep learning operations, then compared the speed and accuracy trade-offs for different model variants.

All models and operations are executed on a single thread of an Ivy-Bridge i5-3570 CPU.For comparison, we measure both the hybrid and ciphertext-only variants for the multiplier and adder circuits.

However most other measured operations and models are hybrid, that is, we exploit efficiencies in plaintexts whenever possible.

We measure the average time for logic gate execution across 1,000 runs.

Multiply and add operations are measured by the average from 16 runs, from initially chosen random numbers.

The normalize operation time is an average for 1 input, from the 30 inputs for a Wisconsin breast cancer instance.

The execution times of the models are measured from single runs on a single instance.

Since the normalization timings are shown separately, and since server-side normalization may not always be necessary (as discussed in section 4.1), the model execution times do not include the time spent normalizing inputs.

Some operations such as encryption, decryption and NOT were too fast to meaningfully measure, and had a negligible impact on model execution times, so these are not included in the results.

Appendix D provides additional execution timings, and expands upon these results.

Our measurements indicate that, at the time of this writing, the TFHE library performs operations around 4.1× faster than the FHEW library.

This gap could be wider on systems with FMA and AVX2 support, since TFHE has optional optimizations that utilize these instructions, however our Ivy-Bridge CPU does not support these.

From table 1, we show that for 16-bit inputs, our hybrid multiplier is around 3.2× faster than an equivalent 16-bit ciphertext-only multiplier.

However the 8-bit hybrid multiplier is a further 4.4× faster than that, the 4-bit hybrid multiplier is 7.0× faster than the 8-bit.

Interestingly, despite consisting of both a multiply and add operation, the average normalize operation is faster than an average multiply.

This is because many of the standard deviations for the breast cancer inputs are actually very small, with only a few inputs requiring 16 bits or more.

As a result, many standard deviations contain large numbers of plaintext 0s, which in turn leads to fewer homomorphic operations during the multiply.

On the other hand, our multiply tests use random numbers, which should contain a more even mixture of 0s and 1s, and therefore execute more slowly.

For the breast cancer perceptron in table 2, we observe that the version with 8-bit intermediate values is almost 4.9× faster than the 16-bit variant, while there is a 3.9× speedup for the MLP.

Importantly, these 8-bit speedups allow the CNN to execute in a somewhat feasible amount of time.

Table 1 also shows that the ReLU activation function is so fast, that its execution time becomes negligible relative to other operations.

Unlike BID8 , the vast majority of our performance bottleneck exists in the multiply and add operations of the weighted sums.

This creates an interesting contrast, where our system can handle arbitrarily deep models with little overhead, but is slow when a large number of parameters are used, while BID8 only works on shallow models, but can handle a comparatively large number of parameters efficiently.

When examining model accuracies, there seems to be very little penalty for 8 bit model inputs and parameters, compared to 16-bit.

However these accuracies are examined in Keras, which uses 32-bit floating point intermediate values.

There could potentially be additional losses from our fixed-point arithmetic, which we might not have noticed, given that we only tested individual instances for correctness.

Our work shows that with the proposed Hybrid Homomorphic Encryption system, almost any production deep learning model can be converted, such that it can process encrypted inputs.

Our design also makes it feasible to implement new or bespoke functionality, as the deep learning paradigm evolves.

Depending on the value of the problem and the size of the model, this system is already viable for production use.

New and updated HE libraries appear frequently, and our code should adapt to any library which implements homomorphic logic gates.

Therefore our software could potentially receive "free" performance gains, as the HE paradigm evolves.

Having multi-threaded code would dramatically speed up our models, since individual weighted sum, convolution, multiply or add operations can be performed on separate threads.

However even without multithreading, on a processor with n cores, we could run n duplicate models on n input instances, and expect an 1/n amortized cost.

Implementing an efficient matrix multiplier could also improve batch-processing times, and might allow for faster convolution operations.

Along with multithreading, there have also been efforts to design FHE systems around larger 3-bit circuits like full-adders BID1 , and to accelerate existing HE libraries with GPUs (Lee et al., 2015) .

There is likely value in implementing homomorphic logic gates on GPUs, or even FPGAs, since almost every operation in our system can be made parallel in some form.

Plaintext weights are "encrypted" without noise in BID8 to improve performance.

While both FHEW and TFHE are considerably different from this system, it may be possible to create a cryptosystem like FHEW that efficiently and directly integrates HHE.In practice, the latest deep learning models use lots of additional functionality, such as residual blocks BID14 , or perhaps a self-normalizing activation function BID16 .

It should be feasible to extend our implementation with most if not all of this functionality.

Work has also been done to optimize deep learning models for embedded hardware, and we have not used all of the tricks available in this space -this would be another straightforward opportunity to build upon our approach.

Looking back at our problem scenario, an organization could now use our design, or even our software, and create a deep learning solution that performs powerful analysis, yet guarantees user privacy.

Over time, this could potentially lead to an industry of privacy-centric software development, which can coexist with the expanding role of big data.

A APPENDIX: APPROXIMATING FUNCTIONS Figure 9 : A piecewise linear approximation of Sigmoid, and its error.

DISPLAYFORM0 Cresult ← AND each bit of Cresult with C mask Cthresh ← C − T hreshold DISPLAYFORM1 Lookup tables can be used to approximate functions.

Depending on the function, one could use constants, linear equations, polynomials or even other functions as components for the lookup table, allowing for great versatility in our approach.

As shown in Figure 9 , using a piecewise linear approximation of sigmoid with only factors of 2 n , we can achieve an error of less than 0.02 against the original function.

Algorithm 3 describes the sigmoid approximation, where a number of linear components are computed, and then applied to the lookup table according to a number of thresholds.

Because we do not know what the input number is, we must compute every component, so there is a trade-off between the number of components and the complexity of their calculation.

Algorithm 2 describes a lookup table, which selects one component, corresponding to a threshold for the input number.

By subtracting the threshold from the number, and masking each component in turn with the resulting sign bits, we can isolate the correct component despite not knowing the plaintext value.

At the end, all but one component will be entirely 0s, so each component can be ORed in turn to get the final result.

During our testing, we found that when training the perceptron for 100 epochs with the sigmoid output activation, then swapping in the "fast sigmoid" activation for 100 epochs, we achieve almost identical accuracies and losses, compared to just using sigmoid.

We intend to release the implementation as open-source software, so in order to make it practical for others to use, we designed it to be easily configured and built.

The architecture can largely be divided into four parts: the backend, arithmetic logic, deep learning and testing components.

The backend abstracts shared functionality between FHE libraries, and combines preprocessing macros with a CMake build script to switch between the desired backend and dependencies.

Importantly, these features allow us to support any new backend with relative ease, so our software can take advantage of new or updated HE libraries over time.

We also include a "Fake" backend which uses the same API, but does not actually perform encryption, for testing and debugging purposes.

The arithmetic logic implements our adder and multiplier circuits, which are supported by circuits such as the HFA, and assorted bit-manipulation functions.

The deep learning component implements the weighted sum, convolution, ReLU and normalization operations, as well as the interpreter, as described in the main text.

Finally, the testing component ensures that all digital circuits up to and including the HFA are executing exactly as intended, while the adder and multiplier circuits are tested on a fixed number of initially chosen random inputs.

Deep learning operations and models are tested on single instances of their respective test datasets.

C APPENDIX: DEEP LEARNING MODELS As described in the main text, we created three different models in the Keras framework.

The structure of these models can be observed in FIG7 .Each model is trained using the ADAM optimizer with Nesterov momentum BID15 BID9 , with a cross-entropy loss function and a batch-size of 128.

The perceptron and MLP are trained for 200 epochs with a learning rate of 0.001, while the CNN is trained for 100 epochs with a learning rate of 0.002.Since the perceptron has so few parameters to optimize, it may not always successfully converge on a first attempt.

We repeatedly train the perceptron through the first 100 epochs, until it has found parameters that allow it to converge slightly.

This may require anywhere from 1 to 5 attempts.

Continuing the trend from the main results, the 2-bit hybrid multiplier is 9.6× faster than the 4-bit.

Less dramatic speedups are observed for the add operations, where the 16-bit hybrid adder is 2.1× faster than the cipher adder, and the 8-bit hybrid adder is 2.8× faster than the 16-bit adder.

Table 4 also shows that compared to the hybrid multipliers, the cipher multipliers do not see as great a relative speedup.

The 8-bit cipher multiplier is only 3.5× faster than the 16-bit, and the 4-bit multiplier is only 6.3× faster than the 8-bit.

This can in part be explained by the fact that lower precision numbers are more likely to be of the form 2 n and −2 n , which our hybrid multiplier handles very quickly as discussed in section 3.4.

@highlight

We made a feature-rich system for deep learning with encrypted inputs, producing encrypted outputs, preserving privacy.

@highlight

A framework for private deep learning model inference using FHE schemes that support fast bootstrapping and thus can reduce computation time.

@highlight

The paper presents a means of evaluating a neural network securely using homomorphic encryption.