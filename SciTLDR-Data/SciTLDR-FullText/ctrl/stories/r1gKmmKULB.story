Holistically exploring the perceptual and neural representations underlying animal communication has traditionally been very difficult because of the complexity of the underlying signal.

We present here a novel set of techniques to project entire communicative repertoires into low dimensional spaces that can be systematically sampled from, exploring the relationship between perceptual representations, neural representations, and the latent representational spaces learned by machine learning algorithms.

We showcase this method in one ongoing experiment studying sequential and temporal maintenance of context in songbird neural and perceptual representations of syllables.

We further discuss how studying the neural mechanisms underlying the maintenance of the long-range information content present in birdsong can inform and be informed by machine sequence modeling.

Systems neuroscience has a long history of decomposing the features of complex signals under 13 the assumption that they can be untangled and explored systematically, part-by-part.

For example on a rich understanding of the phonological, semantic, and syntactic features of speech and language.

In contrast, the communicative spaces of many model organisms in auditory neuroscience are more 23 poorly understood, leading to a very small number of model organisms having the necessary tools for 24 study.

In birdsong, for example, biophysical models of song production that have been developed 25 for zebra finches do not capture the dynamics of the dual-syrinx vocal tract of European starlings.

More species general approaches to modeling communication would increase the accessibility of 27 more diverse and more systematic explorations of animal communication systems in neuroscience.

Here, we propose a method based upon recent advances in generative modeling to explore and We show that this method is successful in species as diverse as songbirds, primates, insects, cetaceans,

and amphibians, and in recording conditions both in the lab and in the field.

We demonstrate this basal ganglia and frontal cortex analogous structures actively maintain temporal information, and songbird temporal structure exhibits long-range temporal dependencies that parallel those seen in Figure 2 : Outline of the context-dependent perception task.

Birds are tasked with classifying a smooth morph between syllables generated from a VAE, generating a psychometric function of classification behavior.

Sequential-contextual cues that precede the classified syllables are given to bias the psychometric function.

In the present experiment we explore how sequential context is maintained in the songbird brain.

To this end, we train a songbird to classify regions of a VAE-generated latent space of song, and 69 manipulate the perception of those regions of space based upon sequential-contextual information 70 (Fig 2) .

Specifically, we interpolate between syllables of European starling song projected into latent 71 space.

We train a starling to classify the left and right halves of the interpolation using an operant-72 conditioning apparatus (Fig. 4) .

We then provide a contextual syllable preceding the classified 73 syllable that holds predictive information over the classified syllable (Fig 2 bottom) .

We hypothesize 74 that the perception of the boundary between the classified stimuli shifts as a function of the context 75 cue.

We model this hypothesis using Bayes rule:

prior When a stimulus varies upon a single dimension x (the interpolation), the perceived value of x is 77 a function of the true value of x and contextual information (Fig. 3 left) .

The initial behavioral 78 results of our experiment confirmed our hypotheses (Fig. 3) .

We additionally performed acute

<|TLDR|>

@highlight

We  compare perceptual, neural, and modeled representations of animal communication using machine learning, behavior, and physiology. 