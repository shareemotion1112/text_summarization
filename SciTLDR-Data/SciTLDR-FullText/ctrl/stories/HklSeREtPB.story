Recent work suggests goal-driven training of neural networks can be used to model neural activity in the brain.

While response properties of neurons in artificial neural networks bear similarities to those in the brain, the network architectures are often constrained to be different.

Here we ask if a neural network can recover both neural representations and, if the architecture is unconstrained and optimized, also the anatomical properties of neural circuits.

We demonstrate this in a system where the connectivity and the functional organization have been characterized, namely, the head direction circuit of the rodent and fruit fly.

We trained recurrent neural networks (RNNs) to estimate head direction through integration of angular velocity.

We found that the two distinct classes of neurons observed in the head direction system, the Ring neurons and the Shifter neurons, emerged naturally in artificial neural networks as a result of training.

Furthermore, connectivity analysis and in-silico neurophysiology revealed structural and mechanistic similarities between artificial networks and the head direction system.

Overall, our results show that optimization of RNNs in a goal-driven task can recapitulate the structure and function of biological circuits, suggesting that artificial neural networks can be used to study the brain at the level of both neural activity and anatomical organization.

Artificial neural networks have been increasingly used to study biological neural circuits.

In particular, recent work in vision demonstrated that convolutional neural networks (CNNs) trained to perform visual object classification provide state-of-the-art models that match neural responses along various stages of visual processing Khaligh-Razavi & Kriegeskorte, 2014; Yamins & DiCarlo, 2016; Güçlü & van Gerven, 2015; Kriegeskorte, 2015) .

Recurrent neural networks (RNNs) trained on cognitive tasks have also been used to account for neural response characteristics in various domains (Mante et al., 2013; Sussillo et al., 2015; Song et al., 2016; Cueva & Wei, 2018; Banino et al., 2018; Remington et al., 2018; Wang et al., 2018; Orhan & Ma, 2019; Yang et al., 2019) .

While these results provide important insights on how information is processed in neural circuits, it is unclear whether artificial neural networks have converged upon similar architectures as the brain to perform either visual or cognitive tasks.

Answering this question requires understanding the functional, structural, and mechanistic properties of artificial neural networks and of relevant neural circuits.

We address these challenges using the brain's internal compass -the head direction system, a system that has accumulated substantial amounts of functional and structural data over the past few decades in rodents and fruit flies (Taube et al., 1990a; Turner-Evans et al., 2017; Green et al., 2017; Seelig & Jayaraman, 2015; Stone et al., 2017; Lin et al., 2013; Finkelstein et al., 2015; Wolff et al., 2015; Green & Maimon, 2018) .

We trained RNNs to perform a simple angular velocity (AV) integration task (Etienne & Jeffery, 2004) and asked whether the anatomical and functional features that have emerged as a result of stochastic gradient descent bear similarities to biological networks sculpted by long evolutionary time.

By leveraging existing knowledge of the biological head direction (HD) systems, we demonstrate that RNNs exhibit striking similarities in both structure and function.

Our results suggest that goal-driven training of artificial neural networks provide a framework to study neural systems at the level of both neural activity and anatomical organization. (2017)).

e) The brain structures in the fly central complex that are crucial for maintaining and updating heading direction, including the protocerebral bridge (PB) and the ellipsoid body (EB).

f) The RNN model.

All connections within the RNN are randomly initialized.

g) After training, the output of the RNN accurately tracks the current head direction.

We trained our networks to estimate the agent's current head direction by integrating angular velocity over time (Fig. 1f ).

Our network model consists of a set of recurrently connected units (N = 100), which are initialized to be randomly connected, with no self-connections allowed during training.

The dynamics of each unit in the network r i (t) is governed by the standard continuous-time RNN equation:

for i = 1, . . .

, N .

The firing rate of each unit, r i (t), is related to its total input x i (t) through a rectified tanh nonlinearity, r i (t) = max(0, tanh(x i (t))).

Every unit in the RNN receives input from all other units through the recurrent weight matrix W rec and also receives external input, I(t), through the weight matrix W in .

These weight matrices are randomly initialized so no structure is a priori introduced into the network.

Each unit has an associated bias, b i which is learned and an associated noise term, ξ i (t), sampled at every timestep from a Gaussian with zero mean and constant variance.

The network was simulated using the Euler method for T = 500 timesteps of duration τ /10 (τ is set to be 250ms throughout the paper).

Let θ be the current head direction.

Input to the RNN is composed of three terms: two inputs encode the initial head direction in the form of sin(θ 0 ) and cos(θ 0 ), and a scalar input encodes both clockwise (CW, negative) and counterclockwise, (CCW, positive) angular velocity at every timestep.

The RNN is connected to two linear readout neurons, y 1 (t) and y 2 (t), which are trained to track current head direction in the form of sin(θ) and cos(θ).

The activities of y 1 (t) and y 2 (t) are given by:

Velocity at every timestep (assumed to be 25 ms) is sampled from a zero-inflated Gaussian distribution (see Fig. 5 ).

Momentum is incorporated for smooth movement trajectories, consistent with the observed animal behavior in flies and rodents.

More specifically, we updated the angular velocity as AV(t) = σX + momentum * AV(t−1), where X is a zero mean Gaussian random variable with standard deviation of one.

In the Main condition, we set σ = 0.03 radians/timestep and the momentum to be 0.8, corresponding to a mean absolute AV of ∼100 deg/s.

These parameters are set to roughly match the angular velocity distribution of the rat and fly (Stackman & Taube, 1998; Sharp et al., 2001; Bender & Dickinson, 2006; Raudies & Hasselmo, 2012) .

In Sec. 4, we manipulate the magnitude of AV by changing σ to see how the trained RNN may solve the integration task differently.

We optimized the network parameters W to minimize the mean-squared error in equation (3) between the target head direction and the network outputs generated according to equation (2), plus a metabolic cost for large firing rates (L 2 regularization on r).

Parameters were updated with the Hessian-free algorithm (Martens & Sutskever, 2011) .

Similar results were also obtained using Adam (Kingma & Ba, 2015) .

We found that the trained network could accurately track the angular velocity (Fig. 1g) .

We first examined the functional and structural properties of model units in the trained RNN and compared them to the experimental data from the head direction system in rodents and flies.

We first plotted the neural activity of each unit as a function of HD and AV (Fig. 2a) .

This revealed two distinct classes of units based on the strength of their HD and AV tuning (see Appendix Fig. 6a, b, c) .

Units with essentially zero activity are excluded from further analyses.

The first class of neurons exhibited HD tuning with minimal AV tuning (Fig. 2f) .

The second class of neurons were tuned to both HD and AV and can be further subdivided into two populations -one with high firing rate when animal performs CCW rotation (positive AV), the other favoring CW rotation (negative AV) (CW tuned cell shown in Fig. 2g) .

Moreover, the preferred head direction of each sub-population of neurons tile the complete angular space (Fig. 2b ).

Embedding the model units into 3D space using t-SNE reveals a clear ring-like structure, with the three classes of units being separated (Fig. 2c) .

Neurons with HD tuning but not AV tuning have been widely reported in rodents (Taube et al., 1990a; Blair & Sharp, 1995; Stackman & Taube, 1998) , although the HD*AV tuning profiles of neurons are rarely shown (but see Lozano et al. (2017) ).

By re-analyzing the data from Peyrache et al. (2015) , we find that neurons in the anterodorsal thalamic nucleus (ADN) of the rat brain are selectively tuned to HD but not AV (Fig. 2d , also see Lozano et al. (2017) ), with HD*AV tuning profile similar to what our model predicts.

Preliminary evidence suggests that this might also be true for ellipsoid body (EB) ring neurons in the fruit fly HD system (Green et al., 2017; Turner-Evans et al., 2017) .

Previous studies have shown that the temporal relationship of cell firing to the rat's head direction differs across H D cells recorded from the PoS and ADN (Blair and Sharp, 1995 , Blair et al., 1997 , Taube and Muller, 1998 .

Specifically, H D cells in the ADN appear to encode the rat's f uture directional heading by ϳ25 msec, whereas H D cells in the PoS encode the rat's present or recent past directional heading.

Of the 20 H D cells recorded from the L M N, a second H D cell was recorded simultaneously on the same electrode wire for three cells.

Because the tuning curves of these simultaneously recorded cells partially overlapped one another, the accuracy of the time shift analysis is likely to be compromised for these cells, and they were therefore excluded from the following analyses.

Time shift analyses were conducted on a representative session for each of the remaining 17 L M N H D cells.

Figure 5A -C illustrates an example of the optimal time shifts of an L M N H D cell for each of the three parameters.

For this cell, the optimal time shifts for peak firing rate, range width, and information

Previous studies have shown that the temporal relationship of cell firing to the rat's head direction differs across HD cells recorded from the PoS and ADN (Blair and Sharp, 1995 , Blair et al., 1997 , Taube and Muller, 1998 .

Specifically, HD cells in the ADN appear to encode the rat's future directional heading by ϳ25 msec, whereas HD cells in the PoS encode the rat's present or recent past directional heading.

Of the 20 HD cells recorded from the LMN, a second HD cell was recorded simultaneously on the same electrode wire for three cells.

Because the tuning curves of these simultaneously recorded cells partially overlapped one another, the accuracy of the time shift analysis is likely to be compromised for these cells, and they were therefore excluded from the following analyses.

Time shift analyses were conducted on a representative session for each of the remaining 17 LMN HD cells.

Figure 5A -C illustrates an example of the optimal time shifts of an LMN HD cell for each of the three parameters.

For this cell, the optimal time shifts for peak firing rate, range width, and information

Previous studies have shown that the temporal relationship of cell firing to the rat's head direction differs across H D cells recorded from the PoS and ADN (Blair and Sharp, 1995 , Blair et al., 1997 , Taube and Muller, 1998 .

Specifically, H D cells in the ADN appear to encode the rat's f uture directional heading by ϳ25 msec, whereas H D cells in the PoS encode the rat's present or recent past directional heading.

Of the 20 H D cells recorded from the L M N, a second H D cell was recorded simultaneously on the same electrode wire for three cells.

Because the tuning curves of these simultaneously recorded cells partially overlapped one another, the accuracy of the time shift analysis is likely to be compromised for these cells, and they were therefore excluded from the following analyses.

Time shift analyses were conducted on a representative session for each of the remaining 17 L M N H D cells.

Figure 5A -C illustrates an example of the optimal time shifts of an L M N H D cell for each of the three parameters.

For this cell, the optimal time shifts for peak firing rate, range width, and information et al., 1994) .

The head turn modulation of HD cell activity in these distinct brain areas suggests that LMN HD cells are head turn-modulated HD cells, whereas those in the cortical regions represent a dual code, signaling both head turn and directional heading.

Previous studies have shown that the temporal relationship of cell firing to the rat's head direction differs across HD cells recorded from the PoS and ADN (Blair and Sharp, 1995 , Blair et al., 1997 , Taube and Muller, 1998 .

Specifically, HD cells in the ADN appear to encode the rat's future directional heading by ϳ25 msec, whereas HD cells in the PoS encode the rat's present or recent past directional heading.

Of the 20 HD cells recorded from the LMN, a second HD cell was recorded simultaneously on the same electrode wire for three cells.

Because the tuning curves of these simultaneously recorded cells partially overlapped one another, the accuracy of the time shift analysis is likely to be compromised for these cells, and they were therefore excluded from the following analyses.

Time shift analyses were conducted on a representative session for each of the remaining 17 LMN HD cells.

Figure 5A -C illustrates an example of the optimal time shifts of an LMN HD cell for each of the three parameters.

For this cell, the optimal time shifts for peak firing rate, range width, and information Neurons tuned to both HD and AV tuning have also been reported previously in rodents and fruit flies (Sharp et al., 2001; Stackman & Taube, 1998; Bassett & Taube, 2001) , although the joint HD*AV tuning profiles of neurons have only been documented anecdotally with a few cells (Turner-Evans et al. (2017) ).

In rodents, certain cells are also observed to display HD and AV tuning (Fig. 2e) .

In addition, in the fruit fly heading system, neurons on the two sides of the protocerebral bridge (PB) (Pfeiffer & Homberg, 2014) are also tuned to CW and CCW rotation, respectively, and tile the complete angular space, much like what has been observed in our trained network (Green et al., 2017; Turner-Evans et al., 2017) .

These observations collectively suggest that neurons that are HD but not AV selective in our model can be tentatively mapped to "Ring" units in the EB, and the two sub-populations of neurons tuned to both HD and AV map to "Shifter" neurons on the left PB and right PB, respectively.

We will correspondingly refer to our model neurons as either 'Ring' units or 'CW/CCW Shifters'(Further justification of the terminology will be given in Sec. 3.2 & 3.3).

We next sought to examine the tuning properties of both Ring units and Shifters of our network in greater detail.

First, we observe that for both Ring units and Shifters, the HD tuning curve varies as a function of AV (see example Ring unit in Fig. 2f and example Shifter in Fig. 2g ).

Population summary statistics concerning the amount of tuning shift are shown in Appendix Fig. 7a .

The preferred HD tuning is biased towards a more CW angle at CW angular velocities, and vice versa for CCW angular velocities.

Consistent with this observation, the HD tuning curves in rodents were also dependent upon AV (see example neurons in Fig. 2h,i ) (Blair & Sharp, 1995; Stackman & Taube, 1998; Taube & Muller, 1998; Blair et al., 1997; .

Second, the AV tuning curves for the Shifters exhibit graded response profiles, consistent with the measured AV tuning curves in flies and rodents (see Fig. 1b,d ).

Across neurons, the angular velocity tuning curves show substantial diversity (see Appendix Fig. 6b ), also consistent with experimental reports (Turner-Evans et al., 2017) .

In summary, the majority of units in the trained RNN could be mapped onto the biological head direction system in both general functional architecture and also in detailed tuning properties.

Our model unifies a diverse set of experimental observations, suggesting that these neural response properties are the consequence of a network solving an angular integration task optimally.

vity of the trained network are structured, and exhibit similarity to the connectivity lex.

a) the connectivity of the network trained with mid-speed.

We sort the neuron nctional classes, and further arrange them according to the preferred HD with each ible structural connectivity within and across different cell types.

b) mapping the ture onto fly central complex.

Inserted panels represent the average connectivity one s.d.) as a function of the difference between preferred HD for within and class of neurons.

ly structured connectivity have been proposed to perform integration.

It's shown igure 3: Connectivity of the trained network is structured and exhibits similarities with the connectivity in the fly central complex.

a) Pixels represent connections from the units in each column to the units in each row.

Excitatory connections are in red, and inhibitory connections are in blue.

Units are first sorted by functional classes, and then are further sorted by their preferred HD within each class.

The black box highlights recurrent connections to the ring units from ring units, from CCW Shifters, and from CW Shifters.

b) Ensemble connectivity from each functional cell type to the Ring units as highlighted in a), in relation to the architecture of the PB & EB in the fly central complex.

Plots show the average connectivity (shaded area indicates one s.d.) as a function of the difference between the preferred HD of the cell and the Ring unit it is connecting to.

Ring units connect strongly to units with similar HD tuning and inhibit units with dissimilar HD tuning.

CCW shifters connect strongly to ring units with preferred head directions that are slightly CCW-shifted to its own, and CW shifters connect strongly to Ring units with preferred head directions that are slightly CW-shifted to its own.

Refer to Appendix Fig. 8b for the full set of ensemble connectivity between different classes.

Previous experiments have detailed a subset of connections between EB and PB neurons in the fruit fly.

We next analyzed the connectivity of Ring units and Shifters in the trained RNN to ask whether it recapitulates these connectivity patterns -a test which has never been done to our knowledge in any system between artificial and biological neural networks (see Fig. 3 ).

We ordered Ring units, CCW Shifters, and CW Shifters by their preferred head direction tuning and plotted their connection strengths (Fig. 3a) .

This revealed highly structured connectivity patterns within and between each class of units.

We first focused on the connections between individual Ring units and observed a pattern of local excitation and global inhibition.

Neurons that have similar preferred head directions are connected through positive weights and neurons whose preferred head directions are anti-phase are connected through negative weights (Fig. 3b) .

This pattern is consistent with the connectivity patterns inferred in recent work based on detailed calcium imaging and optogenetic perturbation experiments (Kim et al., 2017) , with one caveat that the connectivity pattern inferred in this study is based on the effective connectivity rather than anatomical connectivity.

We conjecture that Ring units in the trained RNN serve to maintain a stable activity bump in the absence of inputs (see section 3.3), as proposed in previous theoretical models (Turing, 1952; Amari, 1977; Zhang, 1996) .

We then analyzed the connectivity between Ring units and Shifters.

We found that CW shifters excite Ring units with preferred head directions that are clockwise to its own, and inhibit Ring units with preferred head directions counterclockwise to its own (Fig. 3b) .

The opposite pattern is observed for CCW shifters.

Such asymmetric connections from Shifters to the Ring units are consistent with the connectivity pattern observed between the PB and the EB in the fruit fly central complex (Lin et al., 2013; Green et al., 2017; Turner-Evans et al., 2017) , and also in agreement with previously proposed mechanisms of angular integration (Skaggs et al., 1995; Green et al., 2017; Turner-Evans et al., 2017; Zhang, 1996) (Fig. 3b) .

We note that while the connectivity between PB Shifters and EB Ring units are one-to-one (Lin et al., 2013; Wolff et al., 2015; Green et al., 2017) , the connectivity profile in our model is broad, with a single CW Shifter exciting multiple Ring units with preferred HDs that are clockwise to its own, and vice versa for CCW shifters.

In summary, the RNN developed several anatomical features that are consistent with structures reported or hypothesized in previous experimental results.

A few novel predictions are worth mentioning.

First, in our model the connectivity between CW and CCW Shifters exhibit specific recurrent connectivity (Fig. 8) .

Second, the connections from Shifters to Ring units exhibit not only excitation in the direction of heading motion, but also inhibition that is lagging in the opposite direction.

This inhibitory connection has not been observed in experiments yet but may facilitate the rotation of the neural bump in the ring units during turning (Wolff et al., 2015; Franconville et al., 2018; Green et al., 2017; Green & Maimon, 2018) .

In the future, EM reconstructions together with functional imaging and optogenetics should allow direct tests of these predictions.

We have segregated neurons into Ring and Shifter populations according to their HD and AV tuning, and have shown that they exhibit different connectivity patterns that are suggestive of different functions.

Ring units putatively maintain the current heading direction and shifter units putatively rotate activity on the ring according to the direction of angular velocity.

To substantiate these functional properties, we performed a series of perturbation experiments by lesioning specific subsets of connections.

We first lesioned connections when there is zero angular velocity input.

Normally, the network maintains a stable bump of activity within each class of neurons, i.e., Ring units, CW Shifters, and CCW Shifters (see Fig. 4a,b) .

We first lesioned connections from Ring units to all units and found that the activity bumps in all three classes disappeared and were replaced by diffuse activity in a large proportion of units.

As a consequence, the network could not report an accurate estimate of its current heading direction.

Furthermore, when the connections were restored, a bump formed again

We next lesioned connections during either constant CW or CCW angular velocity.

Normally, the network can integrate AV accurately (Fig. 4k-n) .

As expected, during CCW rotation, we observe a corresponding rotation of the activity bump in Ring units and in CCW Shifters, but CW Shifters display low levels of activity.

The converse is true during CW rotation.

We first lesioned connections from CW Shifters to all units, and found that it significantly impaired rotation in the CW direction, and also increased the rotation speed in the CCW direction.

Lesioning of CCW Shifters to all units had the opposite effect, significantly impairing rotation in the CCW direction.

These results are consistent with the hypothesis that CW/CCW Shifters are responsible for shifting the bump in a CW and CCW direction, respectively, and are consistent with the data in Green et al. (2017) , which shows that inhibition of Shifter units in the PB of the fruit fly heading system impairs the integration of HD.

Our lesion experiments further support the segregation of units into modular components that function to separately maintain and update heading during angular motion.

Optimal computation requires the system to adapt to the statistical structure of the inputs (Barlow, 1961; Attneave, 1954) .

In order to understand how the statistical properties of the input trajectories without any external input (Fig. 4d) , suggesting the network can spontaneously generate an activity bump through recurrent connections mediated by Ring units.

We then lesioned connections from CW Shifters to all units and found that all three bumps exhibit a CCW rotation, and the read-out units correspondingly reported a CCW rotation of heading direction (Fig. 4e,f) .

Analogous results were obtained with lesions of CCW Shifters, which resulted in a CW drifting bump of activity (Fig. 4g,h ).

These results are consistent with the hypothesis that CW and CCW Shifters simultaneously activate the ring, with mutually cancelling signals, even when the heading direction is stationary.

When connections are lesioned from both CW and CCW Shifters to all units, we observe that Ring units are still capable of holding a stable HD activity bump (Fig.  4i,j) , consistent with the predictions that while CW/CCW shifters are necessary for updating heading during motion, Ring units are responsible for maintaining heading.

We next lesioned connections during either constant CW or CCW angular velocity.

Normally, the network can integrate AV accurately (Fig. 4k-n) .

As expected, during CCW rotation, we observe a corresponding rotation of the activity bump in Ring units and in CCW Shifters, but CW Shifters display low levels of activity.

The converse is true during CW rotation.

We first lesioned connections from CW Shifters to all units, and found that it significantly impaired rotation in the CW direction, and also increased the rotation speed in the CCW direction.

Lesioning of CCW Shifters to all units had the opposite effect, significantly impairing rotation in the CCW direction.

These results are consistent with the hypothesis that CW/CCW Shifters are responsible for shifting the bump in a CW and CCW direction, respectively, and are consistent with the data in Green et al. (2017) , which shows that inhibition of Shifter units in the PB of the fruit fly heading system impairs the integration of HD.

Our lesion experiments further support the segregation of units into modular components that function to separately maintain and update heading during angular motion.

Optimal computation requires the system to adapt to the statistical structure of the inputs (Barlow, 1961; Attneave, 1954) .

In order to understand how the statistical properties of the input trajectories affect how a network solves the task, we trained RNNs to integrate inputs generated from low and high AV distributions.

When networks are trained with small angular velocities, we observe the presence of more units with strong head direction tuning but minimal angular velocity tuning.

Conversely, when networks are trained with large AV inputs, fewer ring units emerge and more units become Shifter-like and exhibit both HD and AV tuning (Fig. 5c,f,i) .

We sought to quantify the overall AV tuning under each velocity regime by computing the slope of each neuron's AV tuning curve at its preferred HD angle.

We found that by increasing the magnitude of AV inputs, more neurons developed strong AV tuning (Fig.  5b,e,h ).

In summary, with a slowly changing head direction trajectory, it is advantageous to allocate more resources to hold a stable activity bump, and this requires more ring units.

In contrast, with quickly changing inputs, the system must rapidly update the activity bump to integrate head direction, requiring more shifter units.

This prediction may be relevant for understanding the diversity of the HD systems across different animal species, as different species exhibit different overall head turning behavior depending on the ecological demand (Stone et al., 2017; Seelig & Jayaraman, 2015; Heinze, 2017; Finkelstein et al., 2018) .

Previous work in the sensory systems have mainly focused on obtaining an optimal representation (Barlow, 1961; Laughlin, 1981; Linsker, 1988; Olshausen & Field, 1996; Simoncelli & Olshausen, 2001; Khaligh-Razavi & Kriegeskorte, 2014) with feedforward models.

Several recent studies have probed the importance of recurrent connections in understanding neural computation by training RNNs to perform tasks (e.g., Mante et al. (2013); Sussillo et al. (2015) ; Cueva & Wei (2018)), but the relation of these trained networks to the anatomy and function of brain circuits are not mapped.

Using the head direction system, we demonstrate that goal-driven optimization of recurrent neural networks can be used to understand the functional, structural and mechanistic properties of neural circuits.

While we have mainly used perturbation analysis to reveal the dynamics of the trained RNN, other methods could also be applied to analyze the network.

For example, in Appendix Fig. 10 , using fixed point analysis (Sussillo & Barak, 2013; Maheswaranathan et al., 2019) , we found evidence consistent with attractor dynamics.

Due to the limited amount of experimental data available, comparisons regarding tuning properties and connectivity are largely qualitative.

In the future, studies of the relevant brain areas using Neuropixel probes (Jun et al., 2017) and calcium imaging (Denk et al., 1990) will provide a more in-depth characterization of the properties of HD circuits, and will facilitate a more quantitative comparison between model and experiment.

In the current work, we did not impose any additional structural constraint on the RNNs during training.

We have chosen to do so in order to see what structural properties would emerge as a consequence of optimizing the network to solve the task.

It is interesting to consider how additional structural constraints affect the representation and computation in the trained RNNs.

One possibility would to be to have the input or output units only connect to a subset of the RNN units.

Another possibility would be to freeze a subset of connections during training.

Future work should systematically explore these issues.

Recent work suggests it is possible to obtain tuning properties in RNNs with random connections (Sederberg & Nemenman, 2019) .

We found that training was necessary for the joint HD*AV tuning (see Appendix Fig. 9 ) to emerge.

While Sederberg & Nemenman (2019) consider a simple binary classification task, our integration task is computationally more complicated.

Stable HD tuning requires the system to keep track of HD by accurate integration of AV, and to stably store these values over time.

This computation might be difficult for a random network to perform (Cueva et al., 2019) .

Our approach contrasts with previous network models for the HD system, which are based on hand-crafted connectivity (Zhang, 1996; Skaggs et al., 1995; Xie et al., 2002; Green et al., 2017; Kim et al., 2017; Knierim & Zhang, 2012; Song & Wang, 2005; Kakaria & de Bivort, 2017; Stone et al., 2017) .

Our modeling approach optimizes for task performance through stochastic gradient descent.

We found that different input statistics lead to different heading representations in an RNN, suggesting that the optimal architecture of a neural network varies depending on the task demandan insight that would be difficult to obtain using the traditional approach of hand-crafting network solutions.

Although we have focused on a simple integration task, this framework should be of general relevance to other neural systems as well, providing a new approach to understand neural computation at multiple levels.

Our model may be used as a building block for AI systems to perform general navigation (Pei et al., 2019) .

In order to effectively navigate in complex environments, the agent would need to construct a cognitive map of the surrounding environment and update its own position during motion.

A circuit that performs heading integration will likely be combined with another circuit to integrate the magnitude of motion (speed) to perform dead reckoning.

Training RNNs to perform more challenging navigation tasks such as these, along with multiple sources of inputs, i.e., vestibular, visual, auditory, will be useful for building robust navigational systems and for improving our understanding of the computational mechanisms of navigation in the brain (Cueva & Wei, 2018; Banino et al., 2018) .

Figure 9: Joint HD × AV tuning of the initial, randomly connected network and the final trained network.

a) Before training, the 100 units in the network do not have pronounced joint HD × AV tuning.

The color scale is different for each unit (blue = minimum activity, yellow = maximum activity) to maximally highlight any potential variation in the untrained network.

b) After training, the units are tuned to HD × AV, with the exception of 12 units (shown at the bottom) which are not active and do not influence the network.

<|TLDR|>

@highlight

Artificial neural networks trained with gradient descent are capable of recapitulating both realistic neural activity and the anatomical organization of a biological circuit.