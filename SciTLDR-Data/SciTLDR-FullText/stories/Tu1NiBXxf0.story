It is difficult for the beginners of etching latte art to make well-balanced patterns by using two fluids with different viscosities such as foamed milk and syrup.

Even though making etching latte art while watching making videos which show the procedure, it is difficult to keep balance.

Thus well-balanced etching latte art cannot be made easily.

In this paper, we propose a system which supports the beginners to make well-balanced etching latte art by projecting a making procedure of etching latte art directly onto a cappuccino.

The experiment results show the progress by using our system.

We also discuss about the similarity of the etching latte art and the design templates by using background subtraction.

Etching latte art is the practice of literally drawing on a coffee with a thin rod, such as a toothpick, in order to create images in the coffee [2] .

There are several kinds of making method of etching latte art depending on tools and toppings.

A method which is often introduced as easy one for beginners is putting syrup directly onto milk foam and etching to make patterns as shown in Figure 1 .

The color combination automatically makes the drink look impressive by using syrup, so baristas are under less pressure to create a difficult design [8] .

However, it is difficult for beginners to imagine how they ought to put syrup and etch in order to make beautiful patterns since etching latte art offers two fluids with different viscosities.

On top of this, even though they can watch videos which show a making procedure of etching latte art, etching latte art made by imitating hardly looks well-balanced.

It is impossible to make well-balanced etching latte art without repeated practice.

In this paper, we develop a support system which helps even beginners to make well-balanced latte art by directly projecting a making procedure of etching latte art using syrup, which has high needs, onto a cappuccino.

Moreover, projecting a deformation of fluid with viscosity such as syrup which is difficult to imagine as animations in order to support beginners to understand the deformation of fluid with viscosity.

We indicate the usefulness of this system through a questionnaire survey and the similarity to the design templates using background subtraction.

There are two kinds of latte art, etching and free pouring.

The typical making method of the former is putting syrup on the milk foam and making patterns by etching whereas the latter does not use any tools and makes patterns with only flow of milk as poring.

Hu and Chi [4] proposed a simulation method which considered viscosity of milk to express the flow of milk in latte art.

Their research expressed the flow of milk which is really similar to one in actual latte art.

From the viewpoint of practicing latte art, however, users have to estimate paths of pouring milk and acquire the manipulation of the milk jug from the results of the simulation.

Moreover, it will not seem to be able to understand the flow of milk unless the users have advanced skills.

Pikalo [7] developed an automated latte art machine which used a modified inkjet cartridge to infuse tiny droplets of colorant into the upper layer of the beverage.

By this machine, unlimitedly designed latte art can be easily made like a printer.

This machine lets everyone have original latte art without any barista skills.

However, this machine cannot make latte art with milk foam like one in free pour latte art.

Therefore, baristas still have to practice latte art to make other kinds of latte art.

Kawai et al. [5] developed a free pour latte art support system by showing how to pour the milk to make latte art designed by the users as animated lines.

This system targets baristas who have experience of making basic free pour latte art and know how much milk has to be poured.

People who never made free pour latte art have to repeat the practice several times.

Flagg et al. [3] developed a painting support system by projecting a painting procedure on the canvas.

In order to avoid users' shadow hiding a projected painting procedure, they put two projectors behind users.

This system is quite large-scale and involves costs.

Morioka et al. [6] visually supported cooking by projecting how to chop ingredients on the proper place of that.

This system is able to indicate how to chop ingredients, detailed notes, and cooking procedures which is hard to understand as cooking while reading a recipe.

However, to use this system, users have to prepare a dedicated kitchen which has a hole on the ceiling.

This system is quite large-scale since it projects instructions from a hole made on the ceiling.

Xie et al. [9] proposed an interactive system which lets common users build large-scale balloon art in an easy and enjoyable way by using spatial segmented reality solution.

This system provides fabrication guidance to illustrate the differences between the depth maps of the target three-dimensional shape and the current work in progress.

In addition, they design a shaking animation for each number to increase user immersion.

Yoshida et al. [10] proposed an architecture-scale, computerassisted digital fabrication method which used a depth camera and projection mapping.

This system captured the current work using a depth camera, compared the scanned geometry with the target shape, and then projected the guiding information based on an evaluation.

They mentioned that AR devices such as head-mounted displays (HMDs) could serve as interfaces, but would require calibration on each device.

The proposed projector-camera guidance system can be prepared with a simple calibration process, and would also allow for intuitive information-sharing among workers.

As indicated in this paper, projectors are able to project instructions on hand.

Therefore, difference between instructions and users' manipulation would be reduced.

We decided to support making etching latte art with a small projector as well.

We show the system overview in Figure 2 .

The system configuration is written below (Figure 3 ).

• A laptop computer connecting to a projector to show a making procedure on a cappuccino.

• A small projector shows a making procedure on a cappuccino.

• A tripod for a small projector.

Firstly, users select a pattern from Heart Ring, Leaf, or Spider Web as shown in Figure 2 (a).

Next, a making procedure of selected etching latte art is projected on the cappuccino (Figure 2(b) ).

Then, the animation of syrup deformation is projected on the cappuccino (Figure 2(c) ).

Finally, a making procedure is projected on the cappuccino once again.

Users put syrup to trace the projected image.

Then, etching latte art is completed by etching to trace the projected lines ( Figure 2(d) ).

Animations at each step can be played whenever the user wants.

In our system, users select a pattern from three kinds of etching latte art (the first column in Table 1 ).

After the selection, a making procedure written below is projected on the cappuccino.

Firstly, proper places to put syrup corresponding on selected etching latte art are projected on the cappuccino (the second column in Table 1 ).

Next, manipulation of the pick is projected on the cappuccino (the third column in Table 1 ).

It is hard to confirm a making procedure of etching latte art in books since they just line up several frames of a making video as shown in Figure 1 .

Our system displays how to put syrup and how to manipulate the pick separately.

As far as the authors know, there is no system which shows a making procedure of etching latte art like our system.

It is difficult for beginners to imagine the syrup deformation by manipulating the pick.

Our system helps users to understand the syrup deformation as manipulating the pick by directly projecting prepared animations (Table 2 ).

In our system, the syrup is drawn in brown and the manipulation of the pick is drawn in blue.

The animations were created by Adobe After Effects considering two fluids with different viscosities.

It takes approximately 30 minutes to create each design template and two hours to create each animation of syrup deformation.

We have developed a system which has functions mentioned in the previous sub-sections.

As shown in Figure 3 , a cappuccino is put in front of a user and a projector mounted on a tripod is placed on the left side of the cappuccino (if the user is left-handed, it is placed on the right side of the cappuccino).

We project a making procedure of etching latte art and animations of syrup deformation from the top.

In order to evaluate our system, we conducted an experiment.

Twelve etching latte art beginners participated in the experiment.

We divided them into two groups (Group 1 and Group 2).

Participants make two etching latte art in different methods (making by oneself and making with our system).

Group 1 makes etching latte art by themselves firstly.

Then they make etching latte art with our system.

Whereas Group 2 makes etching latte art with our system firstly.

Then they make etching latte art by themselves.

In this experiment, we selected patterns from three etching latte art in order to avoid everyone making the same pattern.

After making etching latte art, participants filled out the questionnaire.

We also took a questionnaire survey to inexperienced people in order to ask which etching latte art (made by oneself or made with our system) looks more well-balanced for each participant's etching latte art.

Moreover, we created foreground images from background subtraction for each design template and etching latte art in order to show which etching latte art is more similar to each design template.

(1) Making by Oneself Participants watch a making video of etching latte art they will make (Table 3) .

Then, they make etching latte art by themselves while watching the making video.

(2) Making with Our System Participants make etching latte art with our system.

Firstly, they watch a making procedure projected on the cappuccino.

Secondly, CG animations of syrup deformation (animation speed is almost the same as actually making etching latte art) are projected.

Finally, a making procedure of etching latte art is projected on the cappuccino once again and participants make etching latte art by tracing the projected making procedure.

(3) Notes as Making Etching Latte Art Generally, baristas use well-steamed silky foamed milk which has fair quality tiny bubbles steamed by a steamer attached to an espresso machine for business use [1] .

However, it is difficult to make such good quality foamed milk with a household milk frother.

Milk foamed by a milk frother has big bubbles and they break easily, so syrup put on such milk foam must spread.

Therefore, in this experiment, participants made etching latte art with yoghurt.

There is no difference between yoghurt and foamed milk as manipulating the pick.

Thus, we consider that using yoghurt in the experiment does not affect the evaluation of this system.

We do not have to care about the difference from using our system with a cappuccino.

The results of the making etching latte art are shown in Table 4 .

We compare and evaluate the etching latte art made by oneself (Table 4 "By oneself" Line) and the etching latte art made with our system (Table 4 "Our system" Line).

Participants A, B, G, and H made "Heart Ring" (Table 4 A, B, G, H).

Participants B and H put too much syrup as making by themselves.

Therefore, the hearts are too big and their shape is not desired one.

Whereas, the participants were able to adjust the amount of syrup as making with our system.

As a result, the shape of each heart is clearer and the etching latte art looks better quality.

Participants C, D, I, and J made "Leaf" (Table 4 C, D, I, J).

Participants C and J were not able to draw a line vertically.

Participants J was not able to give the same distance between the syrup.

Due to these problems, their etching latte art looks distorted.

They were able to make well-balanced etching latte art which has the same distance between the syrup by using our system.

Participants E, F, K, and L made "Spider Web" (Table 4 E, F, K, L).

Participants E, F, and K were not able to draw a spiral with the certain space.

The equally spaced spiral was made with our system and they were able to make better balanced etching latte art.

As we mentioned, the etching latte art supported by our system is good quality.

It is indicated that even beginners are able to make well-balanced etching latte art with our system.

Participants compared two etching latte arts made in different methods.

We conducted a questionnaire survey.

The questionnaire consists of the five questions below.

Question 1.

Can you imagine how to make the etching latte art before watching the making video?

Question 2.

Is it easy to make the etching latte art while watching the making video?

We also took some views, impressions, and improvement points of our system.

The results of the participants' questionnaires are shown in Table 5 .

The participants who answered 1pt or 2pt in Question 1 over 80 percent and the average is 1.92pt which is low.

From this result, patterns of etching latte art are complex for people who see them for the first time.

It is difficult to imagine how to make it for them.

The Average of Question 2 is 3.00pt, however, five participants answered 2pt and only two participants answered 5pt.

There are some comments about this question.

(1) I could not understand where I should put the syrup since it was hard to get sense of the place and the size of syrup from the making video.

(2) It was difficult to manipulate the pick and I could not make desired pattern.

Whereas, all participants answered 4pt or 5pt in Question 5.

We consider that it is possible to make certain quality etching latte art with our system.

Our system is popular with participants since it projects the making procedure directly on the cappuccino, so they do not have to watch another screen displaying making videos while making etching latte art.

In Question 3 and 4, the participants who answered 4pt or 5pt over 90 percent and the average is over 4.50pt.

We can say that the animations of syrup deformation in our system help users to properly understand how syrup deforms as drawing lines by the pick.

Also, the animation speed of syrup deformation is ideal.

The comments about our system are written below.

(3) It was easy to draw a line with a pick since I just needed to trace a line projected on the cappuccino, so it was clear where and how much syrup I should put.

(4) I was delighted that I could make etching latte art even though I had never tried it since the making procedure was easy to understand how to make it.

(5) The animation of syrup deformation indicated how the syrup deforms, so I could imagine it.

From the result of Table 4 and comments (1) to (5), comparing to making latte art as watching making videos on another screen, even beginners are able to make etching latte art easily by using our system since it directly projects the making procedure on the cappuccino, so it is popular with users.

We show improvement points from the participants.

(6) I might have been able to put proper amount of syrup if I also confirm how to put the syrup in animations.

(7) When I trace lines from left to right, the lines were hard to confirm because of my shadow at times.

(8)

I got a bit confused because a lot of syrup stayed at the center and the line projected on the syrup could not be seen.

Users adjust the amount of the syrup by putting it on the pattern projected on the cappuccino, however, this pattern is a static image, so some participants put too much syrup as written in comment (6) .

We consider that preparing animations which show how to put the syrup helps users to clearly understand and imagine the speed and the amount of the syrup.

Also, as written in comments (7) and (8), sometimes the projected making procedure was hard to see due to the position of the user's hands or the color of the background.

We will resolve this problem by using multiple projectors.

We conducted a questionnaire survey about etching latte art had been made by participants.

Sixty inexperienced people who had not participated in the experiment were asked which etching latte art (made by oneself or made with our system) looked more similar to the design template for each participant's etching latte art.

The information that if the etching latte art had been made by oneself or with our system was hidden.

The results of inexperienced people's questionnaires are shown in Figre 4.

Ten participants out of twelve got the result that their etching latte art made with our system is similar to the design template than one made by themselves.

About the etching latte art made by Participant G, they put the proper amount of syrup to the proper place even without our system.

Their two etching latte arts are both look well-balanced.

About the etching latte art made by Participant I, they could not trace the making procedure properly since they put the syrup too quickly.

After making etching latte art with our system, they said that it might have been easier to make well-balanced etching latte art if our system showed how to put the syrup in animations as well.

We will improve the system to resolve this issue by creating new animations which show the proper speed to put the syrup.

We created foreground images from background subtraction for each design template and etching latte art in order to quantitatively evaluate which etching latte art is more similar to the design template.

White pixels in the foreground images indicate the difference between the design template and the etching latte art and black pixels indicate the same parts.

We normalize the black pixels in order to quantify the similarity between each design template and etching latte art.

The larger number indicates the higher similarity.

The results of the background subtraction are shown in Table 6 .

Ten participants out of twelve got the result that their etching latte art made with our system is similar to the design template than one made by themselves.

About the etching latte art made by Participant A, the place of each heart was adjusted by our system, however, they put too much syrup.

As a result, the difference from the design template is big.

About the etching latte art made by Participant D, the syrup is off to right.

As a result, the difference from the design template is big.

However, the difference of similarities between the etching latte art made by themselves and the etching latte art made with our system is only 0.001 which is really little difference.

We need to indicate the proper amount of the syrup more clearly in order to get higher similarity result from background subtraction.

We have developed the system which supports etching latte art beginners to practice and make etching late art and also help them to understand the syrup deformation by directly projecting a making procedure and animations of syrup deformation onto the cappuccino.

The participants' evaluations verified the usefulness of our system.

The results of the inexperienced people's questionnaire and the participants' questionnaire show that more than 80 percent of participants made better-balanced etching latte art with our system.

However, each evaluation says that two participants made betterbalanced etching latte art by themselves and they are all different participants.

From this result, we confirm there are some instances that human beings suppose the etching latte art is similar to the design template even though the result of the background subtraction says it is not similar to the design template, and vice versa.

In our future work, we will improve the system with considering what kind of etching latte art human beings prefer and develop a system which creates animations of syrup deformation automatically.

We also handle the development factors got in the survey.

Table 4 : Experimental result.

Group 1 makes etching latte art by themselves firstly.

Whereas Group 2 makes etching latte art with our system firstly.

Table 5 : Participants' questionnaire result.

Table 6 : Results of background subtraction.

Similarities are represented by a number in the range of 0.000 to 1.000 (1.000 indicates totally the same as the design template).

@highlight

We have developed an etching latte art support system which projects the making procedure directly onto a cappuccino to help the beginners to make well-balanced etching latte art.