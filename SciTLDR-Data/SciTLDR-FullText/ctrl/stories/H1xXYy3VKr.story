Multivariate time series with missing values are common in areas such as healthcare and finance, and have grown in number and complexity over the years.

This raises the question whether deep learning methodologies can outperform classical data imputation methods in this domain.

However, naive applications of deep learning fall short in giving reliable confidence estimates and lack interpretability.

We propose a new deep sequential latent variable model for dimensionality reduction and data imputation.

Our modeling assumption is simple and interpretable: the high dimensional time series has a lower-dimensional representation which evolves smoothly in time according to a Gaussian process.

The non-linear dimensionality reduction in the presence of missing data is achieved using a VAE approach with a novel structured variational approximation.

We demonstrate that our approach outperforms several classical and deep learning-based data imputation methods on high-dimensional data from the domains of computer vision and healthcare, while additionally improving the smoothness of the imputations and providing interpretable uncertainty estimates.

Multivariate medical time series, consisting of multiple correlated univariate time series or channels, give rise to two distinct ways of imputing missing information: (1) by exploiting temporal correlations within each channel, and (2) by exploiting correlations across channels, for example by using lower-dimensional representations of the data.

An ideal imputation model for medical time series should take both of these sources of information into account.

Another desirable property of such models is to offer a probabilistic interpretation, allowing for uncertainty estimation.

Unfortunately, current imputation approaches fall short with respect to at least one of these desiderata.

While there are many time-tested statistical methods for multivariate time series analysis (e.g., Gaussian processes (Roberts et al., 2013) ), these methods are generally not applicable when features are missing.

On the other hand, classical methods for time series imputation often do not take the potentially complex interactions between the different channels into account (Little and Rubin, 2002; Pedersen et al., 2017) .

Finally, recent work has explored the use of non-linear dimensionality reduction using variational autoencoders for i.i.d.

data points with missing values (Ainsworth et al., 2018; Ma et al., 2018; Nazabal et al., 2018) , but this work has not considered temporal data and strategies for sharing statistical strength across time.

A more comprehensive analysis of existing approaches and their shortcomings is deferred to the appendix (Sec. A).

In this paper, we propose an architecture that combines deep variational autoencoders (VAEs) with Gaussian process (GP) to efficiently model the latent dynamics at multiple time scales.

Moreover, our inference approach makes use of efficient structured variational approximations, where we fit another multivariate Gaussian process in order to approximate the intractable true posterior.

We make the following contributions:

• A new model.

We propose a VAE architecture for multivariate time series imputation with a GP prior in the latent space to capture temporal dynamics.

• Efficient inference.

We use a structured variational approximation that models posterior correlations in the time domain.

• Benchmarking on real-world data.

We carry out extensive comparisons to classical imputation methods as well as state-of-the-art deep learning approaches, and perform experiments on data from different domains.

We propose a novel architecture for missing value imputation in medical time series.

Our model can be seen as a way to perform amortized approximate inference on a latent Gaussian process model.

Specifically, we combine ideas from VAEs (Kingma and Welling, 2014) , GPs (Rasmussen, 2003) , Cauchy kernels (Jähnichen et al., 2018) , structured variational distributions with efficient inference (Bamler and Mandt, 2017b) , and a special ELBO for missing data (Nazabal et al., 2018) and synthesize these ideas into a general framework for missing data imputation on time series.

In the following, we will outline the assumed generative model and derive our proposed inference scheme.

We use standard notation (similar to (Nazabal et al., 2018) ), which is detailed in the appendix (Sec. B.1).

In this work, we overcome the problem of defining a suitable GP kernel in the data space with missing observations by instead applying the GP in the latent space of a variational autoencoder where the encoded feature representations are complete.

That is, we assign a latent variable z t ∈ R k for every x t , and model temporal correlations in this reduced representation using a GP, z(τ ) ∼ GP(m z (·), k z (·, ·)).

This way, we decouple the step of filling in missing values and capturing instantaneous correlations between the different feature dimensions from modeling dynamical aspects.

The graphical model is depicted in the appendix (Fig. S2) .

In order to model data that varies at multiple time scales, we consider the Cauchy kernel, which has previously been successfully used in the context of robust dynamic topic modeling where similar multi-scale time dynamics occur (Jähnichen et al., 2018) .

It corresponds to an infinite mixture of RBF kernels with different length scales (Rasmussen, 2003) .

Given the latent time series z 1:T , the observations x t are generated time-point-wise by where g θ (·) is a potentially nonlinear function parameterized by the parameter vector θ.

In our experiments, the function g θ is implemented by a deep neural network.

In order to learn the parameters of the deep generative model described above, and in order to efficiently infer its latent state, we are interested in the posterior distribution p(z 1:T | x 1:T ).

Since the exact posterior is intractable, we use variational inference (Blei et al., 2017; Jordan et al., 1999; Zhang et al., 2018) .

Furthermore, to avoid inference over per-datapoint (local) variational parameters, we apply inference amortization (Kingma and Welling, 2014) .

To make our variational distribution more expressive and capture the temporal correlations of the data, we employ a structured variational distribution (Wainwright and Jordan, 2008) with efficient inference that leads to an approximate posterior which is also a GP.

We approximate the true posterior p(z 1:T,j | x 1:T ) with a multivariate Gaussian variational distribution q(z 1:T,j | x

where j indexes the dimensions in the latent space.

Our approximation implies that our variational posterior is able to reflect correlations in time, but breaks dependencies across the different dimensions in z-space (which is typical in VAE training (Kingma and Welling, 2014; Rezende et al., 2014) ).

We choose the variational family to be the family of multivariate Gaussian distributions in the time domain, where the precision matrix Λ j is parameterized as a tridiagonal matrix. (Little and Rubin, 2002) -0.177 ± 0.000 0.935 ± 0.000 0.028 ± 0.000 VAE (Kingma and Welling, 2014) 0.599 ± 0.002 0.232 ± 0.000 0.922 ± 0.000 0.034 ± 0.000 HI-VAE (Nazabal et al., 2018) 0.372 ± 0.008 0.134 ± 0.003 0.962 ± 0.001 0.035 ± 0.000 GP-VAE (proposed) 0.341 ± 0.007 0.117 ± 0.002 0.960 ± 0.002 0.002 ± 0.000

Samples from q can thus be generated in O(T ) time (Bamler and Mandt, 2017b; Huang and McColl, 1997; Mallik, 2001 ) as opposed to the O(T 3 ) time complexity for a full-rank matrix.

Moreover, compared to a fully factorized variational approximation, the number of variational parameters is merely doubled.

Note that while the precision matrix is sparse, the covariance matrix can still be dense, allowing to reflect long-range dependencies in time.

We amortize the inference over m j and Λ j using an inference network q ψ (·).

As in standard VAE training, the parameters of the generative model and of the inference network can be jointly trained by optimizing the evidence lower bound (ELBO),

Following Nazabal et al. (2018) (see Sec. A), we evaluate the ELBO only on the observed features of the data since the remaining features are unknown, and set these missing features to a fixed value (zero) during inference.

We also include an additional tradeoff parameter β into our ELBO, similar to the β-VAE (Higgins et al., 2017) .

This parameter takes care of balancing the influence between the likelihood on the observed data features and the latent prior.

Our training objective is thus the RHS of (3).

We performed experiments on the benchmark data set Healing MNIST (Krishnan et al., 2015) , which combines the classical MNIST data set (LeCun et al., 1998) with properties common to medical time series, the SPRITES data set (Li and Mandt, 2018) , and on a real-world medical data set from the 2012 Physionet Challenge (Silva et al., 2012) .

We compared our model against conventional single imputation methods (Little and Rubin, 2002) , GP-based imputation (Rasmussen, 2003) , VAE-based methods that are not specifically designed to handle temporal data (Kingma and Welling, 2014; Nazabal et al., 2018) , and modern state-of-the-art deep learning methods for temporal data imputation (Cao et al., 2018; Luo et al., 2018) .

We found strong quantitative (Tab.

1, 2) and qualitative ( Fig. 1, 2 ) evidence that our proposed model outperforms most baseline methods in terms of imputation quality on all BRITS (red) and forward imputation (green) yield single imputations, while the GP-VAE (blue) allows to draw samples from the posterior.

The GP-VAE produces smoother curves, reducing noise from the original input, and exhibits an interpretable posterior uncertainty.

Table 2 : Performance of the different models on the Physionet data set in terms of AUROC of a logistic regression trained on the imputed time series.

We observe that the proposed model performs comparably to the state of the art.

Mean imputation (Little and Rubin, 2002) 0.703 ± 0.000 Forward imputation (Little and Rubin, 2002) 0.710 ± 0.000 GP (Rasmussen, 2003) 0.704 ± 0.007 VAE (Kingma and Welling, 2014) 0.677 ± 0.002 HI-VAE (Nazabal et al., 2018) 0.686 ± 0.010 GRUI-GAN (Luo et al., 2018) 0.702 ± 0.009 BRITS (Cao et al., 2018) 0.742 ± 0.008 GP-VAE (proposed) 0.730 ± 0.006 three tasks and performs comparable to the state of the art (BRITS) on the medical data.

This extends even to different missingness mechanisms, as is described in the appendix (Tab.

S1).

For the real medical time series task, no ground-truth data exists, so we cannot report the mean squared error (MSE) or the negative log likelihoood (NLL).

Following (Luo et al., 2018) , we instead use a downstream classifier as a proxy measure.

We use a linear SVM to predict mortality based on the imputed time series, since this was also one of the original tasks in the 2012 Physionet challenge (Silva et al., 2012) .

We find that this proxy measure correlates well with the likelihood in cases where ground-truth data is available (see Healing MNIST AUROC in Tab.

1), lending credence to the metric.

More details about these experiments can be found in the appendix (Sec. C).

Classical statistical approaches.

The problem of missing values has been a long-standing challenge in many time series applications, especially in the field of medicine (Pedersen et al., 2017) .

The earliest approaches to deal with this problem often relied on heuristics, such as mean imputation or forward imputation.

Despite their simplicity, these methods are still widely applied today due to their efficiency and interpretability (Honaker and King, 2010) .

Orthogonal to these ideas, methods along the lines of expectation-maximization (EM) have been proposed, but they often require additional modeling assumptions (Bashir and Wei, 2018) .

Bayesian methods.

When it comes to estimating likelihoods and uncertainties relating to the imputations, Bayesian methods, such as Gaussian processes (GPs) (Rasmussen, 2003) , have a clear advantage over non-Bayesian methods such as single imputation (Little and Rubin, 2002) .

There has been much recent work in making these methods more expressive and incorporating prior knowledge from the domain (e.g., medical time series) (Fortuin and Rätsch, 2019; Wilson et al., 2016) or adapting them to work on discrete domains (Fortuin et al., 2018a) , but their wide-spread adoption is hindered by their limited scalability and the challenges in designing kernels that are robust to missing values.

Our latent GP prior bears certain similarities to the GP latent variable model (GP-LVM) (Lawrence, 2004; Titsias and Lawrence, 2010) , but in contrast to this line of work, we propose an efficient amortized inference scheme.

Deep learning techniques.

Another avenue of research in this area uses deep learning techniques, such as variational autoencoders (VAEs) (Ainsworth et al., 2018; Dalca et al., 2019; Ma et al., 2018; Nazabal et al., 2018) or generative adversarial networks (GANs) (Li et al., 2019; Yoon et al., 2018) .

It should be noted that VAEs allow for tractable likelihoods, while GANs generally do not and have to rely on additional optimization processes to find latent representations of a given input (Lipton and Tripathi, 2017) .

Unfortunately, none of these models explicitly take the temporal dynamics of time series data into account.

Conversely, there are deep probabilistic models for time series (e.g., Fortuin et al., 2018b; Krishnan et al., 2015 Krishnan et al., , 2017 , but those do not explicitly handle missing data.

There are also some VAE-based imputation methods that are designed for a setting where the data is complete at training time and the missingness only occurs at test time (Garnelo et al., 2018a,b; Ivanov et al., 2018) .

We do not regard this setting in our work.

HI-VAE.

Our approach borrows some ideas from the HI-VAE (Nazabal et al., 2018) .

This model deals with missing data by defining an ELBO whose reconstruction error term only sums over the observed part of the data.

For inference, the incomplete data are filled with arbitrary values (e.g., zeros) before they are fed into the inference network, which induces an unavoidable bias.

The main difference to our approach is that the HI-VAE was not formulated for sequential data and therefore does not exploit temporal information in the imputation task.

Deep learning for time series imputation.

While the mentioned deep learning approaches are very promising, most of them do not take the time series nature of the data directly into account, that is, they do not model the temporal dynamics of the data when l G B r r P / w h V r t m F t F g K p o Q F 0 Q J P Q u r + 6 3 z m V 2 e D H 3 F 0 H T F Z c E 4 p n h J 6 j K r 1 q Y 7 Z 9 7 a + 6 R 4 F B T W E / U v R t Y j z D e D n s 9 T 9 z T A 7 m p o t o o + v v S G 5 p + 5 C j / s n p l Z 1 B A + 7 B o i 8 h R p 9 E A X H + k T v 6 R V r J u 3 h 3 q Q G 3 t 9 c N m i / y H o e 1 e X 3 c T i X 2 Q R 1 E 0 M z t T y A X E y c z r P P / d E n O p v p P k h 5 h k 3 m N j F f N t N 5 1 N 0 i 1 z 3 O d Q f 3 8 p l q o j v D a 1 w R d 8 H K Z X P p L P 9 P 1 C e i 6 8 7 P J 6 + 5 j H + Y O f v H u q m c N L Z N 6 B 9 3 a o e t 4 m + 2 S s 9 p g 1 5 y P w + p R c f 4 w h z 6 T j / o J / 0 y j g x p T I w 0 d y 2 t F H u e 0 b X L + P Y X S H 0 f z Q = = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " d M T 0 j I Y N / p J w + Q q v k a S v 9 a z V L x I = " >

A A A F y n i c j Z R L b 9 N A E M e n A U M J j 6 Z w 5 B I 1 Q u L Q V n H S 1 7 H i m U O R C i J t p a a K / N g k V v z S e l M S L N 8 4 8 G m 4 w l f h u 3 D g v 2 N T p S l J Y 2 s 9 s 7 O z v 3 m s b T v 2 v U T V 6 7 9 X S n f u G v f u r z 4 o P 3 z 0 + M l a Z f 3 p S R K N p C P a T u R H 8 s y 2 E u F 7 o W g r T / n i L J b C C m x f n N r D 1 3 r 9 9 F L I x I v C z 2 o S i 4 v A 6 o d e z 3 M s B V O 3 s t F R Y q y Y k 7 q W H G 5 J 4 W Z p J 7 D U w O 6 l 4 6 x r Z t 1 K r b 5 d 5 6 t 6 U z E L p U b F d R y t l 1 r U I Z c i c m h E A Q k K S U H 3 y a I E 9 z m Z V K c Y t g t K Y Z P Q P F 4 X l F E Z e 0 f w E v C w Y B 3 i 2 c f s v L C G m G t m w r s d R P E x J H Z W 6 Q X G O y b a 8 N Z R B f Q E 8 g / G V 7 b 1 5 0 Z I m a w z n E D a T M y Z H 7 C i a A C f 2 / Y G h W e 2 9 E 5 d l 6 I e H X A 9 H j K M 2 a I r d a 4 4 b 7 A i Y R v y S p X e s m c f D J v n l + h B C N l G B r r P / w h V r t m F t F g K p o Q F 0 Q J P Q u r + 6 3 z m V 2 e D H 3 F 0 H T F Z c E 4 p n h J 6 j K r 1 q Y 7 Z 9 7 a + 6 R 4 F B T W E / U v R t Y j z D e D n s 9 T 9 z T A 7 m p o t o o + v v S G 5 p + 5 C j / s n p l Z 1 B A + 7 B o i 8 h R p 9 E A X H + k T v 6 R V r J u 3 h 3 q Q G 3 t 9 c N m i / y H o e 1 e X 3 c T i X 2 Q R 1 E 0 M z t T y A X E y c z r P P / d E n O p v p P k h 5 h k 3 m N j F f N t N 5 1 N 0 i 1 z 3 O d Q f 3 8 p l q o j v D a 1 w R d 8 H K Z X P p L P 9 P 1 C e i 6 8 7 P J 6 + 5 j H + Y O f v H u q m c N L Z N 6 B 9 3 a o e t 4 m + 2 S s 9 p g 1 5 y P w + p R c f 4 w h z 6 T j / o J / 0 y j g x p T I w 0 d y 2 t F H u e 0 b X L + P Y X S H 0 f z Q = = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " d M T 0 j I Y N / p J w + Q q v k a S v 9 a z V L x I = " > A A A F y n i c j Z R L b 9 N A E M e n A U M J j 6 Z w 5 B I 1 Q u L Q V n H S 1 7 H i m U O R C i J t p a a K / N g k V v z S e l M S L N 8 4 8 G m 4 w l f h u 3 D g v 2 N T p S l J Y 2 s 9 s 7 O z v 3 m s b T v 2 v U T V 6 7 9 X S n f u G v f u r z 4 o P 3 z 0 + M l a Z f 3 p S R K N p C P a T u R H 8 s y 2 E u F 7 o W g r T / n i L J b C C m x f n N r D 1 3 r 9 9 F L I x I v C z 2 o S i 4 v A 6 o d e z 3 M s B V O 3 s t F R Y q y Y k 7 q W H G 5 J 4 W Z p J 7 D U w O 6 l 4 6 x r Z t 1 K r b 5 d 5 6 t 6 U z E L p U b F d R y t l 1 r U I Z c i c m h E A Q k K S U H 3 y a I E 9 z m Z V K c Y t g t K Y Z P Q P F 4 X l F E Z e 0 f w E v C w Y B 3 i 2 c f s v L C G m G t m w r s d R P E x J H Z W 6 Q X G O y b a 8 N Z R B f Q E 8 g / G V 7 b 1 5 0 Z I m a w z n E D a T M y Z H 7 C i a A C f 2 / Y G h W e 2 9 E 5 d l 6 I e H X A 9 H j K M 2 a I r d a 4 4 b 7 A i Y R v y S p X e s m c f D J v n l + h B C N l G B r r P / w h V r t m F t F g K p o Q F 0 Q J P Q u r + 6 3 z m V 2 e D H 3 F 0 H T F Z c E 4 p n h J 6 j K r 1 q Y 7 Z 9 7 a + 6 R 4 F B T W E / U v R t Y j z D e D n s 9 T 9 z T A 7 m p o t o o + v v S G 5 p + 5 C j / s n p l Z 1 B A + 7 B o i 8 h R p 9 E A X H + k T v 6 R V r J u 3 h 3 q Q G 3 t 9 c N m i / y H o e 1 e X 3 c T i X 2 Q R 1 E 0 M z t T y A X E y c z r P P / d E n O p v p P k h 5 h k 3 m N j F f N t N 5 1 N 0 i 1 z 3 O d Q f 3 8 p l q o j v D a 1 w R d 8 H K Z X P p L P 9 P 1 C e i 6 8 7 P J 6 + 5 j H + Y O f v H u q m c N L Z N 6 B 9 3 a o e t 4 m + 2 S s 9 p g 1 5 y P w + p R c f 4 w h z 6 T j / o J / 0 y j g x p T I w 0 d y 2 t F H u e 0 b X L + P Y X S H 0 f z Q = = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " E c p c s w N p s h R M C T T R B E 9 C q v 6 r f K Z X Z 4 E f c n Q V M Z 5 x T i m e E n q E q t W p j t j 3 q r 6 p H v m a G s D + U X c t 5 H x 9 + H k s V X 8 z z F 6 P z W b R R x f e k N x T d a H L / R N j q y q C i 1 1 9 R F 5 F j R 6 I g m O 9 o 1 e 0 y 1 q N N n G v U B 3 v b y 7 r t K W z n k Z 1 + H 0 c T G U 2 Q F 3 B U E w l t y F n E 8 f z 7 H F / 1 I l O Z r o F U p 5 h g 7 k N z O f N d B p 1 Q + e 6 y b m u 4 5 4 / U 0 V 0 J n j 1 c + I G W L l s z J 3 l / 4 n q R F T d + f n k N R f x D 6 t N / r E u K w f 1 t R r 0 t + u V n a b + m y 3 S Q 3 p E T 7 i f O 9 S k P X x h N n 2 n H / S T f h n v j c / G F + N r 7 l p Y 0 H u W 6 c J l f P s L u 6 M m G w = = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " y 1 C o V H x t s 1 j Z / P C b F V p 0 + W y s N / g = " > A A A F 1 3 i c j Z R L b 9 N A E M e n A U M J r 5 Q e u U R E S B z a k k d f x 6 q 8 c g C p o K Y t a k r k x y a x 4 p f W T k m x L G 6 I K w c + D V f 4 E H w X D v x 3 v F R p S t L Y W s / s 7 O x v H m v b i j w 3 T q r V 3 w u F a 9 e N G z c X b x V v 3 7 l 7 7 3 5 p 6 c F B H A 6 l L V p 2 6 I X y y D J j 4 b m B a C V u 4 o m j S A r T t z x x a A 2 e q f X D U y F j N w z 2 k 7 N I n P h m L 3 C 7 r m 0 m M H V K T 9 u J G C X M S R 1 T D l a l c L K 0 7 Z t J 3 + q m o + x D y g 6 p F H a W d f a z T q l S X a v y V b 6 s 1 L R S I X 3 t h U u F J r X J o Z B s G p J P g g J K o H t k U o z 7 m G p U p Q i 2 E 0 p h k 9 B c X h e U U R F 7 h / A S 8 D B h H e D Z w + x Y W w P M F T P m 3 T a i e B g S O 8 v 0 G O M l E y 1 4 q 6 g C e g z 5 B + M T 2 3 p T I 6 R M V h m e Q V p M z J l v s J J Q H z 5 X 7 f W 1 Z z b 3 T l V X Q l 3 a 5 n p c Z B i x R V V q n 3 O e Y 0 X C N u C V M r 1 g z x 4 Y F s 9 P 0 Y M A s o U M V J / / E c p c s w N p s h R M C T T R B E 9 C q v 6 r f K Z X Z 4 E f c n Q V M Z 5 x T i m e E n q E q t W p j t j 3 q r 6 p H v m a G s D + U X c t 5 H x 9 + H k s V X 8 z z F 6 P z W b R R x f e k N x T d a H L / R N j q y q C i 1 1 9 R F 5 F j R 6 I g m O 9 o 1 e 0 y 1 q N N n G v U B 3 v b y 7 r t K W z n k Z 1 + H 0 c T G U 2 Q F 3 B U E w l t y F n E 8 f z 7 H F / 1 I l O Z r o F U p 5 h g 7 k N z O f N d B p 1 Q + e 6 y b m u 4 5 4 / U 0 V 0 J n j 1 c + I G W L l s z J 3 l / 4 n q R F T d + f n k N R f x D 6 t N / r E u K w f 1 t R r 0 t + u V n a b + m y 3 S Q 3 p E T 7 i f O 9 S k P X x h N n 2 n H / S T f h n v j c / G F + N r 7 l p Y 0 H u W 6 c J l f P s L u 6 M m G w = = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " y 1 C o V H x t s 1 j Z / P C b F V p 0 + W y s N / g = " > A A A F 1 3 i c j Z R L b 9 N A E M e n A U M J r 5 Q e u U R E S B z a k k d f x 6 q 8 c g C p o K Y t a k r k x y a x 4 p f W T k m x L G 6 I K w c + D V f 4 E H w X D v x 3 v F R p S t L Y W s / s 7 O x v H m v b i j w 3 T q r V 3 w u F a 9 e N G z c X b x V v 3 7 l 7 7 3 5 p 6 c F B H A 6 l L V p 2 6 I X y y D J j 4 b m B a C V u 4 o m j S A r T t z x x a A 2 e q f X D U y F j N w z 2 k 7 N I n P h m L 3 C 7 r m 0 m M H V K T 9 u J G C X M S R 1 T D l a l c L K 0 7 Z t J 3 + q m o + x D y g 6 p F H a W d f a z T q l S X a v y V b 6 s 1 L R S I X 3 t h U u F J r X J o Z B s G p J P g g J K o H t k U o z 7 m G p U p Q i 2 E 0 p h k 9 B c X h e U U R F 7 h / A S 8 D B h H e D Z w + x Y W w P M F T P m 3 T a i e B g S O 8 v 0 G O M l E y 1 4 q 6 g C e g z 5 B + M T 2 3 p T I 6 R M V h m e Q V p M z J l v s J J Q H z 5 X 7 f W 1 Z z b 3 T l V X Q l 3 a 5 n p c Z B i x R V V q n 3 O e Y 0 X C N u C V M r 1 g z x 4 Y F s 9 P 0 Y M A s o U M V J / / E c p c s w N p s h R M C T T R B E 9 C q v 6 r f K Z X Z 4 E f c n Q V M Z 5 x T i m e E n q E q t W p j t j 3 q r 6 p H v m a G s D + U X c t 5 H x 9 + H k s V X 8 z z F 6 P z W b R R x f e k N x T d a H L / R N j q y q C i 1 1 9 R F 5 F j R 6 I g m O 9 o 1 e 0 y 1 q N N n G v U B 3 v b y 7 r t K W z n k Z 1 + H 0 c T G U 2 Q F 3 B U E w l t y F n E 8 f z 7 H F / 1 I l O Z r o F U p 5 h g 7 k N z O f N d B p 1 Q + e 6 y b m u 4 5 4 / U 0 V 0 J n j 1 c + I G W L l s z J 3 l / 4 n q R F T d + f n k N R f x D 6 t N / r E u K w f 1 t R r 0 t + u V n a b + m y 3 S Q 3 p E T 7 i f O 9 S k P X x h N n 2 n H / S T f h n v j c / G F + N r 7 l p Y 0 H u W 6 c J l f P s L u 6 M m G w = = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " y 1 C o V H x t s 1 j Z / P C b F V p 0 + W y s N / g = " > A A A F 1 3 i c j Z R L b 9 N A E M e n A U M J r 5 Q e u U R E S B z a k k d f x 6 q 8 c g C p o K Y t a k r k x y a x 4 p f W T k m x L G 6 I K w c + D V f 4 E H w X D v x 3 v F R p S t L Y W s / s 7 O x v H m v b i j w 3 T q r V 3 w u F a 9 e N G z c X b x V v 3 7 l 7 7 3 5 p 6 c F B H A 6 l L V p 2 6 I X y y D J j 4 b m B a C V u 4 o m j S A r T t z x x a A 2 e q f X D U y F j N w z 2 k 7 N I n P h m L 3 C 7 r m 0 m M H V K T 9 u J G C X M S R 1 T D l a l c L K 0 7 Z t J 3 + q m o + x D y g 6 p F H a W d f a z T q l S X a v y V b 6 s 1 L R S I X 3 t h U u F J r X J o Z B s G p J P g g J K o H t k U o z 7 m G p U p Q i 2 E 0 p h k 9 B c X h e U U R F 7 h / A S 8 D B h H e D Z w + x Y W w P M F T P m 3 T a i e B g S O 8 v 0 G O M l E y 1 4 q 6 g C e g z 5 B + M T 2 3 p T I 6 R M V h m e Q V p M z J l v s J J Q H z 5 X 7 f W 1 Z z b 3 T l V X Q l 3 a 5 n p c Z B i x R V V q n 3 O e Y 0 X C N u C V M r 1 g z x 4 Y F s 9 P 0 Y M A s o U M V J / / E c p c s w N p s h R M C T T R B E 9 C q v 6 r f K Z X Z 4 E f c n Q V M Z 5 x T i m e E n q E q t W p j t j 3 q r 6 p H v m a G s D + U X c t 5 H x 9 + H k s V X 8 z z F 6 P z W b R R x f e k N x T d a H L / R N j q y q C i 1 1 9 R F 5 F j R 6 I g m O 9 o 1 e 0 y 1 q N N n G v U B 3 v b y 7 r t K W z n k Z 1 + H 0 c T G U 2 Q F 3 B U E w l t y F n E 8 f z 7 H F / 1 I l O Z r o F U p 5 h g 7 k N z O f N d B p 1 Q + e 6 y b m u 4 5 4 / U 0 V 0 J n j 1 c + I G W L l s z J 3 l / 4 n q R F T d + f n k N R f x D 6 t N / r E u K w f 1 t R r 0 t + u V n a b + m y 3 S Q 3 p E T 7 i f O 9 S k P X x h N n 2 n H / S T f h n v j c / G F + N r 7 l p Y 0 H u W 6 c J l f P s L u 6 M m G w = = < / l a t e x i t > x rec 1 < l a t e x i t s h a 1 _ b a s e 6 4 = " 1 X I 0 Z T j M f p T 9 7 + P / s n D w E W u J Q 2 w = " > A A A F 1 3 i c j Z R L b 9 N A E M e n A U M J j 6 Z w 5 B I R I X F o S 5 z 0 d a x 4 5 g B S Q a Q t a k r k x y a x 4 p f W 2 5 J i W d w Q V w 5 8 G q 7 w I f g u

9 C q v 6 r f K Z X Z 4 E f c n Q V M Z 5 x T i m e E n q E q t W p j t j 3 q r 6 p H v m a G s D + U X c t 5 H x 9 + H k s V X 8 z z F 6 N z W b R R x f e k N x T d a H L / R N j q y q C i 1 1 9 R F 5 D j R 6 I g m O 9 p Z e 0 x 1 q N t n C v U h 3 v b y 7 r t K 2 z n k Z 1 + H 0 c T G U 2 Q F 3 F U E w l d y B n E 8 f z 7 H F / 1 I l O Z r o N U p 5 h g 7 k N z O f N d B p 1 U + e 6 x b l u 4 J 4 / U 0 V 0 J n j 1 c + I m W L l s z J 3 l / 4 n q R F T d + f n k N R f x D 6 t N / r E u K w f 1 9 R r 0 N x u V 3 a b + m y 3 R A 3 p I j 7 m f u 9 S k f X x h N n 2 n H / S T f h n v j c / G F + N r 7 l p Y 0 H v u 0 4 X L + P Y X D d 8 l + g = = < / l a t e x i t > . . .

t ∈ {1, . . . , T } Figure S2 : Graphical model.

dealing with missing values.

To the best of our knowledge, the only deep generative model for missing value imputation that does account for the time series nature of the data is the GRUI-GAN (Luo et al., 2018), which we describe in Sec. 3.

Another deep learning model for time series imputation is BRITS (Cao et al., 2018) , which uses recurrent neural networks (RNNs).

It is trained in a self-supervised way, predicting the observations in a time series sequentially.

We compare against both of these models in our experiments.

Other related work.

Our proposed model combines several ideas from the domains of Bayesian deep learning and classical probabilistic modeling; thus, removing elements from our model naturally relates to other approaches.

For example, removing the latent GP for modeling dynamics as well as our proposed structured variational distribution results in the HI-VAE (Nazabal et al., 2018) described above.

Furthermore, our idea of using a latent GP in the context of a deep generative model bears similarities to the GPPVAE (Casale et al., 2018) , but note that the GPPVAE was not proposed to model time series data and does not take missing values into account.

Lastly, the GP prior with the Cauchy kernel is reminiscent of Jähnichen et al. (2018) and the structured variational distribution is similar to the one used by Bamler and Mandt (2017b) in the context of modeling word embeddings over time, none of which used amortized inference.

We choose the variational family to be the family of multivariate Gaussian distributions in the time domain, where the precision matrix Λ j is parameterized in terms of a product of bidiagonal matrices, Λ j := B j B j , with {B j } tt = b j tt if t ∈ {t, t + 1} 0 otherwise .

Above, the b j tt 's are local variational parameters and B j is an upper triangular band matrix.

Similar structured distributions were also employed by Bamler and Mandt (2017a) ; Blei and Lafferty (2006) .

This parameterization automatically leads to Λ j being positive definite, symmetric, and tridiagonal.

Samples from q can thus be generated in linear time in T (Bamler and Mandt, 2017b; Huang and McColl, 1997; Mallik, 2001 ) as opposed to the cubic time complexity for a full-rank matrix.

Moreover, compared to a fully factorized variational approximation, the number of variational parameters are merely doubled.

Note that while the precision matrix is sparse, the covariance matrix can still be dense, allowing to reflect long-range dependencies in time.

Instead of optimizing m and B separately for every data point, we amortize the inference through an inference network with parameters ψ that computes the variational parameters based on the inputs as (m, B) = h ψ (x o 1:T ).

In the following, we accordingly denote the variational distribution as q ψ (·).

Following standard VAE training, the parameters of the generative model θ and of the inference network ψ can be jointly trained by optimizing the evidence lower bound (ELBO).

Following Nazabal et al. (2018) (see Sec. A), we evaluate the ELBO only on the observed features of the data since the remaining features are unknown, and set these missing features to a fixed value (zero) during inference.

Our training objective is thus the RHS of (3).

Neural network architectures.

We use a convolutional neural network (CNN) as an inference network and a fully connected multilayer perceptron (MLP) as a generative network.

The inference network convolves over the time dimension of the input data and allows for sequences of variable lengths.

It consists of a number of convolutional layers that integrate information from neighboring time steps into a joint representation using a fixed receptive field (see Figure S1 ).

The CNN outputs a tensor of size R T ×3k , where k is the dimensionality of the latent space.

Every row corresponds to a time step t and contains 3k parameters, which are used to predict the mean vector m t as well as the diagonal and off-diagonal elements {b j t,t , b j t,t+1 } j=1:k that characterize B at the given time step.

More details about the network structure are given in the appendix (Sec. C).

Appendix C. Experimental details C.1.

Baseline methods Forward imputation and mean imputation.

Forward and mean imputation are socalled single imputation methods, which means that they do not attempt to fit a distribution over possible values for the missing features, but only predict one estimate (Little and Rubin, 2002) .

Forward imputation always predicts the last observed value for any given feature, while mean imputation predicts the mean of all the observations of the feature in a given time series.

Gaussian process in data space.

One option to deal with missingness in multivariate time series is to fit independent Gaussian processes to each channel.

As discussed previously (Sec. 2.1), this ignores the correlation between channels.

The missing values are then imputed by taking the mean of the respective posterior of the GP for that feature.

VAE and HI-VAE.

The VAE (Kingma and Welling, 2014) and HI-VAE (Nazabal et al., 2018) are fit to the data using the same training procedure as the proposed GP-VAE model.

The VAE uses a standard ELBO that is defined over all the features, while the HI-VAE uses the ELBO from (3), which is only evaluated on the observed part of the feature space.

During inference, missing features are filled with constant values, such as zero.

The GRUI-GAN (Luo et al., 2018) uses a recurrent neural network (RNN), namely a gated recurrent unit (GRU).

Once the network is trained, a time series is imputed by optimizing the latent vector in the input space of the generator, such that the generator's output on the observed features is closest to the true values.

Time series with missing values play a crucial role in the medical field, but are often hard to obtain.

Krishnan et al. (2015) generated a data set called Healing MNIST, which is designed to reflect many properties that one also finds in real medical data.

We benchmark our method on a variant of this data set.

It was designed to incorporate some properties that one also finds in real medical data, and consists of short sequences of moving MNIST digits (LeCun et al., 1998 ) that rotate randomly between frames.

The analogy to healthcare is that every frame may represent the collection of measurements that describe a patient's health state, which contains many missing measurements at each moment in time.

The temporal evolution represents the non-linear evolution of the patient's health state.

The image frames contain around 60 % missing pixels and the rotations between two consecutive frames are normally distributed.

The benefit of this data set is that we know the ground truth of the imputation task.

We compare our model against a standard VAE (no latent GP and standard ELBO over all features), the HI-VAE (Nazabal et al., 2018) , as well as mean imputation and forward imputation.

The models were trained on time series of digits from the Healing MNIST training set (50,000 time series) and tested on digits from the Healing MNIST test set (10,000 time series).

Negative log likelihoods on the ground truth values of the missing pixels and mean squared errors (MSE) are reported in Table 1 , and qualitative results shown in Figure 1 .

To assess the usefulness of the imputations for downstream tasks, we also trained a linear classifier on the imputed MNIST digits to predict the digit class and measured its performance in terms of area under the receiver-operator-characteristic curve (AUROC) (Tab.

1).

Our approach outperforms the baselines in terms of likelihood and MSE.

The reconstructions (Fig. 1) reveal the benefits of the GP-VAE approach: related approaches yield unstable reconstructions over time, while our approach offers more stable reconstructions, Table S1 : Performance of different models on Healing MNIST data with artificial missingness and different missingness mechanisms.

We report mean squared error (lower is better).

The reported values are means and their respective standard errors over the test set.

Mean imp.

Forward imp.

VAE HI-VAE GP-VAE (proposed) Random 0.069 ± 0.000 0.099 ± 0.000 0.066 ± 0.000 0.042 ± 0.000 0.037 ± 0.000 Spatial 0.069 ± 0.000 0.099 ± 0.000 0.101 ± 0.000 0.060 ± 0.000 0.052 ± 0.000 Temporal + 0.091 ± 0.000 0.116 ± 0.000 0.065 ± 0.000 0.042 ± 0.000 0.037 ± 0.000 Temporal − 0.064 ± 0.000 0.093 ± 0.000 0.066 ± 0.000 0.042 ± 0.000 0.037 ± 0.000 MNAR 0.178 ± 0.000 0.174 ± 0.000 0.152 ± 0.001 0.088 ± 0.000 0.078 ± 0.000 using temporal information from neighboring frames.

Moreover, our model also yields the most useful imputations for downstream classification in terms of AUROC.

The downstream classification performance correlates well with the test likelihood on the ground truth data, supporting the intuition that it is a good proxy measure in cases where the ground truth likelihood is not available.

We also observe that our model outperforms the baselines on different missingness mechanisms (Tab.

S1).

To assess our model's performance on more complex data, we applied it to the SPRITES data set, which has previously been used with sequential autoencoders (Li and Mandt, 2018) .

The dataset consists of 9,000 sequences of animated characters with different clothes, hair styles, and skin colors, performing different actions.

Each frame has a size of 64 × 64 pixels and each time series features 8 frames.

We again introduced about 60 % of missing pixels and compared the same methods as above.

The results are reported in Table 1 and example reconstructions are shown in Figure 1 .

As in the previous experiment, our model outperforms the baselines in terms of likelihood and MSE and also yields the most convincing reconstructions.

The HI-VAE seems to suffer from posterior collapse in this setting, which might be due to the large dimensionality of the input data.

We also applied our model to the data set from the 2012 Physionet Challenge (Silva et al., 2012) .

The data set contains 12,000 patients which were monitored on the intensive care unit (ICU) for 48 hours each.

At each hour, there is a measurement of 36 different variables (heart rate, blood pressure, etc.), any number of which might be missing.

We again compare our model against the standard VAE and HI-VAE, as well as a GP fit feature-wise in the data space and the GRUI-GAN model (Luo et al., 2018) , which reported state-of-the-art imputation performance.

The main challenge is the absence of ground truth data for the missing values.

This cannot easily be circumvented by introducing additional missingness since (1) the mechanism by which measurements were omitted is not random, and (2) the data set is already very sparse with about 90 % of the features missing.

To overcome this issue, Luo et al. (2018) proposed a downstream task as a proxy for the imputation quality.

They chose the task of mortality prediction, which was one of the main tasks of the Physionet Challenge on this data set, and measured the performance in terms of AUROC.

In this paper, we adopt this measure.

For sake of interpretability, we used a linear support vector machine (SVM) as a downstream classification model.

This model tries to optimally separate the whole time series in the input space using a linear hyperplane.

The choice of model follows the intuition that under a perfect imputation similar patients should be located close to each other in the input space, while that is not necessarily the case when features are missing, or when the imputation is poor.

Note that it is unrealistic to ask for high accuracies in this task, as the clean data are unlikely to be perfectly separable.

As seen in Table 1 , this proxy measure correlates well with the ground truth likelihood.

The performances of the different methods under this measure are reported in Table 2 .

Our model outperforms all baselines, including the GRUI-GAN, which provides strong evidence that our model is well suited for real-world medical time series imputations.

<|TLDR|>

@highlight

We perform amortized variational inference on a latent Gaussian process model to achieve superior imputation performance on multivariate time series with missing data.