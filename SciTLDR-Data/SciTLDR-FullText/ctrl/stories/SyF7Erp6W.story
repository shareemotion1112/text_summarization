Machine learning algorithms for controlling devices will need to learn quickly, with few trials.

Such a goal can be attained with concepts borrowed from continental philosophy and formalized using tools from the mathematical theory of categories.

Illustrations of this approach are presented on a cyberphysical system: the slot car game, and also on Atari 2600 games.

There is a growing need for algorithms that control cyberphysical systems to learn with very little data how to operate quickly in a partially-known environment.

Many reinforcement-learning (RL) solutions using neural networks (NN) have proved to work well with emulators, for instance with the Atari 1 2600 games BID17 , or with real systems such as robots BID11 .

However, these state-of-the-art approaches need a lot of training data, which may not be obtainable within the allowed time frame or budget.

This work thus started as an alternative approach to teach computers to learn quickly to perform as efficiently as the existing solution with approximately one percent of the training data, time, and computing resources.

We first review reinforcement learning methods for Markov Decision Processes (MDP) and Partially Observable MDP (POMDP).

We then explain the motivation behind our continental-philosophyinspired approach.

We describe the two classes of problems on which we focus: the bijective case, which may lead to playing by imitating, and the category-based approach, which should lead to a more innovative behavior of the control algorithms.

Both approaches rely on knowledge accumulated during previous experiences, as in Lifelong Machine Learning BID6 .These two approaches are illustrated by results from both a commercial slot car game controlled by an 8-bit Arduino system, and from Atari 2600 video games running within the Arcade Learning Environment (ALE, see BID1 ).

The development of Artificial Intelligence (AI) owes much to games, which have become one of the classical test-beds for algorithms.

Slot car games, for instance, are used to evaluate the performance of decision-making systems.

The image-processing, RL-based approach presented in BID11 estimates a car's position on the track thanks to a multilayer perceptron with convolutional layers.

It controls the car by applying one out of four possible voltage levels.

Training the perceptron takes twelve hours, and learning the control strategy needs another half an hour.

The faster solution by BID22 to autonomous slot cars relies on added acceleration sensors and an embedded microcontroller to first create a map of curved and straight tracks.

The control algorithm then sets the target velocity and controls it using a Phase-Locked Loop.

TM are trademarks of their respective owners and will be written without the trademark symbol for clarity in the remainder of this document.

Our system learns to rank with human players in less than a minute, without embedded sensors, for both known and unknown circuits, which correspond to the aforementioned bijective and categorybased approaches.

The sensors are a lap counter, and voltage and current from the track.

Video games also become increasingly useful for providing a cyber-physical representation of our environment.

Indeed, realism turns out to be one of the main focuses for game and character designers (depending on the game intentions).

Nevertheless, due to technical limitations and the need for easily described problems, older video-gaming systems, such as the Atari 2600 console, are the go-to systems.

They provide a wide variety of situations ranging from mazes (Pac-Man-style games) and action games (such as Frogger) to ball-and-paddle games (Breakout, Pong).

Although these different problems require varied strategies to be tackled by a standard human player, they all involve decision-making and, therefore, have been modeled as MDPs BID17 .This framework allows the implementation of many different methods.

Some works, such as BID17 use a rescaled picture of the playing area as an input for a deep Q-network (DQN) in order to select the best action available to the agent.

Another possibility is the use of classic searching and planning methods in order to guide the agent, such as the Iterated Width algorithm BID15 or tree search algorithms such as Monte-Carlo Tree Search (MCTS) BID20 to compute the best possible action for the agent.

A less common method is Shallow Reinforcement Learning BID14 : although this relies on a simpler linear representation, it obtains results similar to those of the non-linear approaches.

Finally, Apprenticeship Learning BID2 and Inverse RL BID12 can also be used to train a more humanlike agent which is close to the aforementioned methods in terms of efficiency.

We will show that our system learns how to play unknown games in a few thousand frames with a score on par with or better than humans.

The success of NN is partially due to the very large amount of data that is used, even if the programmer does not know exactly what happens in the NN (like in black box systems).

Two problems are thus the huge quantity of data needed, and the training time required.

Moreover, the attribution of good coefficients in the learning phase is very difficult -or impossible -to be interpreted, making validation very difficult.

NN are able to learn and generalize, but we do not know exactly how.

To improve or complete the NN approach, we propose an approach that tries to explain and use explicitly how an AI can learn, extract features, categorize and generalize.

To do this, we place the theoretical elements necessary for such high level abilities directly in the method.

These abilities may also emerge in NN after many elementary computations (additions, multiplications, comparisons) occur at each artificial neuron.

If this is what effectively happens in each biological neuron, we postulate that intelligence also consists in higher level intellectual operations.

In other words, we do not want to reduce intelligence to basic computation -even if it is biologically the case.

As we do not want higher level abilities to emerge (or not) after long training times, we explicitly place these high level abilities (such as categorizing and generalizing) directly in our theoretical framework.

Thus we can follow some aspects of Dreyfus' critique of AI presented in BID8 .

This author claims that AI researchers should focus more on what human intelligence is in itself and not only refer to the computer model: considering the brain as a computer, and intelligence as the use of software.

More precisely, in the case of RL with NN: considering intelligence as a collection of elementary computations that organize themselves after much training to reach a reward goal.

We postulate that it could have happened in such a way over the course of human development, but human intelligence has much evolved.

It can produce categorization and generalization not merely for a simple reward, but for the goal of understanding.

This is what our AI tries to do.

Dreyfus also often refers to authors such as Heidegger, Husserl or Foucault, whose work later became known as continental philosophy.

This name was given by analytic philosophers who were often Anglo-Saxon in origin, the earliest being Russell, Frege and Wittgenstein.

Analytic philosophy received much influence from mathematical logic that emerged at the end of the 19th century.

It tries to clarify philosophical issues by logical analysis, postulating that only philosophical statements verifiable through empirical observations are meaningful (principle of logical positivism).

This principle, according to analytic philosophers, is not respected by continental philosophers.

Continental philosophy includes a range of French and German doctrines from the 19th and 20th centuries: German idealism, phenomenology, existentialism (influenced by Kierkegaard and Nietzsche), hermeneutics, structuralism, post-structuralism, psychoanalytic theory and object-oriented ontology.

These philosophies are all contrary to the analytic movement.

If we had to project AI in this debate (analytic versus continental), we could say that Dreyfus criticizes early AI for favoring the analytic tradition and for neglecting the continental one.

Of course machines are computers, and computing is closer in nature to logic than phenomenology, metaphysics or psychoanalysis.

Continental philosophy, however, can perhaps help understand and describe what human intelligence is, especially for high level abilities, like learning, categorizing, generalizing and understanding.

It could possibly then improve the quality and efficiency of human-intelligence-based AI.To summarize, we propose to design our AI using an approach based on certain elements of continental philosophy.

This philosophy is described, for lack of a more precise and widely-accepted definition, in terms of its opposition to analytical philosophy.

In the next sections, we will propose some connections between this philosophy and existing mathematical theories.

We express the logic of our AI at the level of entities, and not at a sample or at a pixel level.

In a way, this is similar to working at the morpheme level in structural linguistics, as defined by de Saussure (1916) , which is the smallest meaningful unit of a language.

That implies to start with an analysis of the sampled signals (in one or two dimensions) to detect entities.

These entities are like our everyday life objects: tracks (straights and curves), cars, balls, paddles, walls.

They are geometrically organized in a space and can be described by cartesian coordinates.

We have defined a distance between them that measures how far two entities E and F are one from one another 2 .

The data is collected at each sample time so that we can construct a timeline and provide an elementary cinematic newtonian model of the situation.

This comes from a very old idea of developmental psychology (see for example the works of BID21 ) that the child starts his cognitive development by the skill of experiencing the world through movement and senses (Piaget called it the sensorimotor stage).

But the perceptive world is not a wild set of disordered primitive sensations.

They are organized in objects (we call them entities) that take place in a space and can move during time 3 .

Thus we do not want to take into account all the samples (voltage and current for the slot car, pixels' colors for images) as the fundamental level of knowledge.

We shall try to organize them as soon as possible as entities that occur in space and time (and not wait for them to emerge, or not, after a very long learning process).

These entities, like the objects of cognitive psychologists, have some properties : relative consistency, continuity of movement, permanence of existence and characteristics (sizes, color, shape).

These properties are part of our approach, in the sense that our AI can look for rectangular entities with a particular position, speed and size 4 .

In more complex games, this rectangular form approach could be too simple, but it is adequate for the Atari 2600 games that we study.

One of the main critiques formulated by Dreyfus against the old AI philosophy is the epistemological assumption that claims that all activities can be formalized in terms of predictive rules or laws.

In this context, the learning phase consists of determining these rules (that is, their parameters).

Then, the system has to apply them by looking for objects or general characteristics of the whole organization of samples, that are like those used in the learning phase.

But what about new objects, never seen before, that could appear?

Such a strict and trivial application of the epistemological assumption would lead to ignore them.

It could also be a principle of precaution to ignore new objects.

On the other hand, there could be a principle of curiosity or adventure.

Clever machines could be more efficient were they curious as explained in BID19 .

Referring to the work of Alison Gopnik and Laura Schulz, developmental psychologists at Berkeley and at the Massachusetts Institute of Technology, respectively, it explains that babies naturally gravitate to objects that surprise them rather than to those they are used to, to achieve some extrinsic goal.

An AI that only focuses on application of predictive rules will miss the advantages of curiosity.

We will use this curiosity to further develop our AI in our next work.

If the epistemological assumption of usual AI could be useful for chemistry or physics, because they are context-free, it could be a contradiction in terms with psychology, and behavior understanding.

Dreyfus argued that human problem solving depends rather on our background sense of the context, that is the natural feeling, understanding or intuition of what is important and interesting given a situation.

The world is not just made of objects: it contains subjects.

In particular, in the games we consider, there is a representative of what we call the "Me": the entity that is controlled by actions.

This point of view allows for a more efficient approach than computing all the possible combinations of the available symbols.

This is exactly what we do when we ask our AI to look as soon as possible for some important features (entities and the "Me").

BID7 referred to the Heideggerian concept of Dasein (which means "being there", for a human being confronted with such issues as personhood and mortality), which is a specific way of Being-in-the-world (another Heideggerian concept that considers it as a unity, saying that it is not appropriate to distinguish strictly between the Being and the world that it is in) BID9 .In other words, one of the first things that our AI must do is to identify the "Me" from amongst all the listed entities.

It is not an implicit potential result of a huge number of trainings, like in some RL processes.

Moreover, being the "Me" does not mean only to be lead by actions.

It also implies to struggle for life.

We can say that the "Me" is driven by some life impulses, and that it is attracted to the good objects (that we call friends) and wants to go away from the bad ones (the enemies).

Thus postulate that among all the entities, some are friends (those whose contact implies a reward or avoids loss of lives) and some are enemies (those whose contact implies loss of lives).

The AI has to distinguish as soon as possible the friends versus the enemies of the "Me", without waiting for this to emerge from millions of trials.

After that, the survival strategy is simple: try to meet the friends unless there is an enemy close to the "Me", in which case the first thing to do is to flee.

One of the most efficient tools that humans use to understand new situations is the ability to make analogies between past and present.

For example, if the AI knows how to play the game Breakout, we expect that it will be able to transpose this ability to a (partially) analogous game, Pong.

In particular, we hope to soon use mathematical tools to transpose a policy from one problem (for instance a game) to another.

Such a theory is proposed by BID3 for PONDP (Partially Observable Non Deterministic Problems).The problem is that ideal situations where two problems have exactly the same number of states and isomorphic structures are very rare.

Nevertheless, there are mathematical tools that can be used to identify non isomorphic structures like equivalence of categories in category theory BID16 5 .

The theory of category is a powerful tool in modern mathematics that appeared in the mid20th century in topological and geometrical contexts, after the mathematical logic, based on set theory.

If mathematical logic was a great source of inspiration for analytic philosophy, category theory could inspire and support continental ideas.

The association between category theory and continental philosophy is proposed by Zalamea (2012) and we will follow this path in our work.

In a very simplistic way, we could say that if analytic philosophy analyses situations, by distinguishing states (or objects), continental philosophy provides syntheses, setting higher new levels of being (Beings, concepts, types).

Whereas in set-theory-based logic, identification is reduced to identity and bijective relations, category theory provides richer descriptions of objects by the introduction of arrows between objects, allowing new kind of identifications.

The reader familiar with category theory may find obvious the rest of this paragraph.

However, as most Machine Learning tools rely on set theory and not category theory, we try to illustrate below the added value of this mathematical framework.

The reader is nevertheless referred to Mac Lane (1998) for a thorough and in-depth explanation of category theory.

A category C is a collection of objects with arrows between some of them, so that we can compose them.

It is something like an oriented graph.

In C, an arrow a : A → B is called an isomorphism if it is invertible, that is if there is an arrow b : B → A, such that ba = Id B and ab = Id A .

If it is the case we say that the object A and B are isomorphic.

The relation of isomorphism defines an equivalence relation on the collection of objects of C. We note the quotient C/ .

If F : C → C is an equivalence of categories, it induces a real bijection F : (C/ ) → (C / ) between the classes of isomorphic objects even if F is not bijective.

We do not identify the objects (or the states) of two situations one-to-one, we identify the types (or classes) of these states.

This process can be very useful in the context of observable problems.

Let's consider two nonempty sets (of states) C and C not necessary of the same cardinals.

Let's suppose that we have two functions of observation f : C → O and f : C → O .

Let's assume that they are surjective (if not, we can restrict O and O ).

The sets of observations O and O will define some types of states.

For each o ∈ O, we say that all the states x ∈ C that are observed as o (f (x) = o), have the type T o .

This defines a natural equivalence relation R f on the set C : ∀x, y ∈ C, x R f y if and only if f (x) = f (y).In terms of categories, we put an invertible arrow between two objects x and y of C iff x R f y (iff stands for if and only if).

This makes C a category, where all arrows are invertible and such that C/ is exactly the quotient C/R f : the set of types of states of C. It is well known that the surjection f : C → O induces a bijectionf : (C/ ) → O between the set of type and the set of observations.

This is obvious since the types as been defined by the observations.f is actually the inverse of DISPLAYFORM0 We do exactly the same with C and f .Suppose now, and this is very important, that the sets of observations O ad O have the same cardinality by the means of a bijection G : O → O .

Thus, we can define a bijection F =f DISPLAYFORM1 .

This bijection between the sets of types can be induced by an equivalence of categories F : C → C defined as follows : for every x ∈ C, let's call o = f (x) and chose an arbitrary x ∈ f −1 (G(o)), and define F (x) = x .

If C and C do not have the same cardinality, F has no chance to be bijective, but F is.

F sends every state x to a state x of the "same" type (up to G).

This is the way that we identify (not necessarily bijectively) C and C .

Thus, if we have a strategy to play in C, we can transpose it in C thanks to F .The use of the theory of category results in the ability to formalize a wide variety of games and situations.

An illustration of this would be the ease with which a human player can switch from the Atari 2600 game Breakout to the very similar Pong.

This ease can be transposed into the formalism of categories.

However, even a much more concrete system such as the slot car described in section 2.1 and experimented on in section 4.1 can be transcribed into the formalism of categories 6 .Let us define the following sets:• {C, C } is the set of categories (one per configuration of the track).• {N , N } is the number of sections per configuration of the track.• {s, s ∈ [1, N ]}, {s , s ∈ [1, N ]} are the possible locations of the car on the circuit.

The location is obtained by counting the number of sections the car has passed in its current lap.

We note (u, i) s (Resp. (u , i ) s ) the voltage and current measured when the car crosses section s (Resp.

s ) Let 1 ≤ s 0 ≤ N (Resp.

1 ≤ s 0 ≤ N ) be the current position of the car in configuration C (Resp.

C ).• Let k be a straight section and l a curve section of C .•

The player influences (u, i) s with the controller, which leads to the policy π defined by (1).

DISPLAYFORM2 We want to identify C and C , to transpose the policy π from C to C .

The states of C are the locations s of the car in the circuit.

Similarly, the states of C are the s .

If N = N and s 0 = s 0 , we can define a bijection between C and C and easily transpose π.

But if N = N or s 0 = s 0 it is impossible to define such a bijection.

Nevertheless, if we turn C and C into categories by defining some arrows, we will be able to define an equivalence of categories F : C → C .

To define these arrows, let's use the observable f defined on the states s of C as follows f : C → {1, 2} with f = h • g where g and h are such that g(s) = (u, i) s and h ((u, i) s ) = 1 if s is a curve, and 2 otherwise.

We define f on the s of C the same way, i.e. f : C → {1, 2} with f = h • g and g and h playing the same roles as g and h on the states of C .We can put an invertible arrow between two states of C iff they have the same image by f , and an invertible arrow between two states of C iff they have the same image by f .

We then define F : C → C by equation (2).

DISPLAYFORM3 It is easy to see (if the exact definitions are known) that FORMULA3 is an equivalence of categories that allows to transfer π from C to C .

F induces a bijection F between the sets of classes (or types of position): DISPLAYFORM4 We finally obtain F (C ) = C (types of curves) and F (S ) = S (types of straights).This example of systematic categorization and generalization proves that we do not work at the level of states but that type of states are considered instead.

Results of this approach are presented for a cyberphysical system: a slot car circuit, and for a simulated system: Atari 2600 video games.

The focus on the slot car experimental setup arose from the need to validate the approach on a cyberphysical system.

With its imperfect actuators such as a brushed, direct-current (DC) motor, imperfect contacts such as metallic brushes on strips, it allowed us to evaluate the approach while dealing with a wide range of signals from a real system.

Moreover, its wide availability and low cost allowed to duplicate the test-bed so as to widen the span of the validations.

On the other hand, the configuration is simple, as there is only one entity with dynamic behavior: the "Me' is the slot car.

The enemies are located at unknown curvilinear abscissas where a high velocity is detrimental to the "Me".

The setup is based on a Scalextric MINI Challenge Set C1320T.

We have replaced the mechanical lap counter by a digital omnipolar Hall effect sensor DVR5033 from Texas Instruments.

The current is sensed via a 1 Ω resistor in series with the metal strips carrying the power.

A spectrum analysis of both the voltage and the current showed components in these signals above 350 Hz.

The antialiasing, second-order filter was designed with a cut-off frequency f c = 31 Hz.

The Design-to-Cost approach, classic in high-volume manufacturing, led to the now unusual choice of a real-pole filter G(s) = 1/(sRC +1) 2 , where s is the Laplace variable, approximated by a Cauer Resistor Capacitor (RC) ladder network BID0 .

The values R and C are chosen thanks to f c = 1/(2πRC).

Moreover, the scaled values of the second RC network, R/d and Cd with {d ∈ R : d > 0}, are computed to meet the specifications of the maximum magnitude error e(d) between G(s) and G a (s), the transfer function of the Cauer RC ladder defined by G a (s) = 1/ (sRC) 2 + s(d + 2)RC + 1 .

The value of e(d) is given by equation (3).

We chose d = 0.1 to have less than 0.5 dB error, with no sensible impact on the later computations.

An implementation with two identical RC sections (i.e. d = 1) would lead to e(1) = 3.5 dB, which would degrade the overall performance.

Both the voltage and the current are filtered by such ladder networks before being sampled at f s = 100 Hz:as there are no components in the power spectrum between f s /2 and 350 Hz, there is no aliasing.

DISPLAYFORM0 The algorithms are written in C language and run in real-time on an Arduino Mega 2560 which has 8192 bytes of Random Access Memory (RAM).

The analog signals are sampled and quantized by the integrated analog to digital converter in the microcontroller, with the sampling period defined by t s = 1/f s , and the sampling time being kt s with k ∈ N.The bijective case for the slot car relies on an three-step imitation procedure:1.

A human player first drives the car for n laps, with n = 3 in our experiments.2.

The K sampled voltages v(kt s ) and currents i(kt s ) of the shortest lap (with corresponding t best lap time) are stored in RAM for 0 ≤ k < K, to be replayed by the AI.3.

An optimization method (Newton) minimizes the difference between the AI's lap time and t best by scaling the recorded samples v(kt s ) used to generate the Pulse-Width Modulation (PWM) control signal.

The analogy-based approach relies on two modules: the reward module, and the decision module described below.

As in traditional RL, our approach relies on a reward from the environment.

This reward is based on three variables: the lap time (measured directly with the lap counter), the presence of the car on the track (binary information), and the fact that the car is moving (also binary information).

The algorithm that we designed to provide this reward constantly monitors the car so as to detect that it did not crash (i.e. that it did not leave the track when the velocity was too high) or that it did not stop (when the current was too low to move the car).

Both detectors are based on k-nearest neighbors algorithms (k-NN) applied to the voltage and the current.

They are implemented as boolean tests on the signals after comparison with some thresholds, to speed up the execution of the algorithm on the microcontroller.

As an illustration, a crash can be detected when the voltage is high and the current is near zero: it means that there is no more load (no DC motor) in contact with the strips, even though the voltage is still applied.

Using this reward model, the AI can successfully pilot the car on previously unencountered tracks.

It does not replay scaled samples of any human driving.

The only information reused by the algorithm is the safe speed: it does not trigger the "car crash" reward signal, yet it maintains the car in motion, thus not triggering the "car stop" reward signal.

As the circuit is unknown, the bijective case cannot be used: there is no bijection between circuits.

The algorithm only relies on the analogy-based approach and transposes knowledge previously acquired for a different circuit configuration thanks to equation FORMULA2 .

This knowledge -a safe speed for a given s -is transposed via non-bijective analogies presented in 3.3 with the function h((u, i) s ) evaluated with a classifier.

Any classifier can be used, including unsupervised learning methods, as the two classes are clustered and separated.

For simplicity, we used a k-NN.In practice, the analogy-based approach starts on the unknown circuit with the safe speed.

The algorithm infers in real time, from only current and voltage measurements, whether the car is in a configuration that we humans call either curve or straight.

The algorithm then chooses the best control signal based on its previous experiences (best in order to reach the goal of decreasing lap time while staying on the track).

Even though we use the terms "straight" and "curve" in our explanation, the algorithm simply classifies current and voltage to choose a control signal so as to stay on the track while decreasing the lap time.

The algorithm uses this past knowledge (the control signal for each class) in a previously unencountered situation.

In this way, it generalizes its strategy and adapts to a radically different case: circuit 2 differs from circuit 1, and a replay of a recorded strategy learned on one circuit or scaled recorded samples of the human driving would fail on the second circuit.

The experiments described in this article are conducted on two circuit configurations of different complexity presented in FIG0 .

The results of our experiments for the bijective and the analogy cases are summarized in table 1.

Values are tabulated as the mean and the standard deviation from the mean, except for the best lap which is the shortest lap time among all laps.

We noticed that the first of eight consecutive laps is always the slowest one for the eight human subjects.

The AI, which starts with no previous information, only relies on a safe speed as described in 4.1.1 using a constant PWM of 39% of the full speed.

The analogy-based AI, which does not replay any recorded samples of a human driving, improves lap times in less than ten laps, even on an unknown track.

On the longest and most complex circuit configuration (circuit 2), it almost ranks best, as tabulated on the line "Final lap".

While the final human lap time is lower than the final AI lap time (2.29 s vs 2.52 s for circuit 1, 3.08 s vs 3.13 s for circuit 2), the human unsurprisingly exhibits a higher standard deviation from the mean (140 ms vs 80 ms for circuit 1, 540 ms vs 20 ms for circuit 2).

Future improvements of the AI on the unknown track will include an optimization of the two speeds transposed by the function h((u, i) s ): only a safe speed was used during our experiments, leading to no car crash for the AI, contrary to some laps by the humans and thus not taken into account.

Lastly, the bijective strategy -imitating the best human lap -also leads to the best lap time.

However, contrary to the solution with analogies, it only works for an identical circuit.

This means that while the best bijective (imitation) lap time (2.65 s) for circuit 2 is lower, thus better than the final lap time for the analogy (adaptive speed) AI (3.13 s), this strategy can only be used on circuit 2 and cannot lead to a generalization.

It is only mentioned here as it gives an empirical lower bound for the lap time on a given circuit.

To summarize the slot car case, we implemented the theoretical method exposed in section 3.3 that allows the AI to reuse previously acquired knowledge on a new circuit where a replay of recorded samples of a human driving would lead to an immediate car crash.

Even though there is no bijection between the different circuits, in practice this theory allows to generalize knowledge to any different circuit (within the limits imposed by the size of the available RAM).

While the slot car allowed us to validate the approach on real analog signals in a simple configuration, the ALE allowed us to validate the approach on more complex configurations while dealing with signals already sampled coming from the emulator.

The concepts of entities with "Me" and life impulse introduced in 3.2 are also used to play Atari 2600 games.

Our proof-of-concept is based on the detection of such entities thanks to image processing: Sobel operator (center image on figure 2 ) and bounding-box detection (right image on figure 2).

It relies on the OpenCV (2017) library.

The entity "Me" is found using system identification.

Signals such as impulses and pseudorandom sequences BID13 are sent to ALE to first detect the entities affected by these signals, then to build a dynamic model of the "Me".

One or a few entities are controllable: they are the "Me".

Their shapes can change during the gameplay, such as the paddle in Breakout, thus the possibility to identify different entities as the "Me".

These measurements also update the probability functions p(E, F ) for entities E and F that the contact between these entities changes the score, in a way similar to the reward function in RL.

From these functions p, friends and enemies are inferred, leading to a basic survival strategy outlined in 3.2.The tests are carried out with the settings from BID17 : the AI plays for a maximum of 5 minutes.

We choose to use the DQN as the reference: the reason being that this publication is one of the most cited in relation to Atari 2600 games, and is the de-facto benchmark to which one must refer.

Although we aim to control cyberphysical systems, we needed to validate the versatility of our approach by first testing it on this standard.

We fully replicated the setup using code made publicly available by the authors, and we obtained the same results as the publication.

We were thus able to extract the score for the DQN for a low number of training frames, so as to compare with our approach.

TAB2 for a training time of 10 000 frames (less than 3 minutes), which is 20 000 times less frames than the average training standard reported in BID17 .

While the DQN achieves better results with millions of training frames, our AI reaches decent scores with comparatively much fewer frames, as plotted for Breakout on figure 3.

A preliminary analysis of what really occurs while our algorithm learns is as follows: during the first thousand frames, the "me" is not yet correctly identified, as the digits, the ball and the paddle move randomly when controlled by the pseudorandom sequence.

Once the identification has converged to the only system that the AI directly controls -the paddle, neither the ball nor the digits -, the algorithm looks for friends and enemies.

It also detects that the ball is a friend, as it sometimes increases the score (when it breaks a brick).

The best scores are in the range of 200 points which, after analysis, corresponds to partially destroyed rows of bricks.

It never destroys all the bricks, as it sometimes misses the ball, especially when it looses the "me", for instance when the paddle's size changes or when the paddle disappears according to the basic image processing algorithm.

Moreover, we noticed that the movement of the "me" under control of the algorithm sometimes never reaches a steady-state: it oscillates by a few pixels at a frequency of 5.4 Hz.

Our explanation from a control perspective is as follows: the "me" can be approximated by a second-order system, and the control strategy is almost equivalent to a proportional controller.

This, in the context of Linear-Time-Invariant (LTI) systems, would already explain the oscillations.

Moreover, the strong non-linearities present both in the control input and the non-LTI "me" also explain in part these oscillations.

The input being quantized to only three values (left, nothing, right), the closed-loop system generates a signal similar to limit-cycles.

These oscillations, in turn, are responsible for many of the balls missed by the algorithm.

To summarize the results on the Atari games, we implemented a few of the concepts presented in 3.1: the notion of entities rather than samples or pixels, and the "me" with the behavior introduced in 3.2.

This led to a learning time of a few thousand frames to get a better than human score on Breakout, however it still does not match the best score reached by the DQN after millions of frames on Pong.

Continental philosophy lead us to formalize a mathematical concept to control an agent evolving in a world, whether it is simulated or real.

The power of this framework was illustrated by the theoretical example of the slot car on unknown circuits.

Results from experiments with a real slot car, using real analog signals confirmed our expectations, even though it only used a basic survival approach.

Moreover, the same basic survival strategy was applied to two Atari 2600 games and showed the same trend: even though not as skilled as, for instance, DQN-based agents trained with two hundred million frames, our AI reached in less than ten thousand frames scores that DQN met after learning with a few million frames.

The next steps are to apply the transposition properties to the Atari games, as we did for the slot car, which should further decrease the learning time when playing a new game.

Moreover, going beyond the basic survival strategy will be mandatory to reach higher scores: approaches based on Monte-Carlo Tree Search will be investigated.

<|TLDR|>

@highlight

Continental-philosophy-inspired approach to learn with few data.