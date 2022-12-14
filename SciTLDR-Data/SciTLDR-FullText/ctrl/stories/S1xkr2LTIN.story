Conducting reinforcement-learning experiments can be a complex and timely process.

A full experimental pipeline will typically consist of a simulation of an environment, an implementation of one or many learning algorithms, a variety of additional components designed to facilitate the agent-environment interplay, and any requisite analysis, plotting, and logging thereof.

In light of this complexity, this paper introduces simple rl, a new open source library for carrying out reinforcement learning experiments in Python 2 and 3 with a focus on simplicity.

The goal of simple_rl is to support seamless, reproducible methods for running reinforcement learning experiments.

This paper gives an overview  of the core design philosophy of the package, how it differs from existing libraries, and showcases its central features.

its central features.

The core functionality of simple rl: Create agents and an MDP, then run and plot their resulting interactions.

Running an experiment also creates an experiment log (stored as a JSON file), which can be used to rerun the exact same experiment, thereby facilitating simple reproduction of results.

All practitioners need to do, in theory, is share a copy of the experiment file to someone with the library to ensure result reproduction.

Reinforcement learning (RL) has recently soared in popularity due in large part to recent success 13 in challenging domains, including learning to play Atari games from image input BID10 , beating the 14 world champion in Go [32] , and robotic control from high dimensional sensors BID2 .

In concert with tensorflow BID0 , along with scipy [19] and numpy BID13 .

BID2 To accommodate this growth, there is a need for a simple, lightweight library that supports quick BID3 execution and analysis of RL experiments in Python.

Certainly, many libraries already fulfill this 23 need for many uses cases--as will be discussed in Section 2, many effective RL libraries for Python 24 already exist.

However, the design philosophy and ultimate end user of these packages is distinct 25 from those targeted by simple rl: those users who seek to quickly run simple experiments, look at 26 a plot that summarizes results, and allow for the quick sharing and reproduction of these findings.

The core design principle of simple rl is that of simplicity, per its name.

The library is stripped to an auto-generated JSON file logging the experimental details.

The actual code of the experiment 37 run is shown in FIG13 : in around five lines, we define a Q-Learning instance, a random actor, and 38 a simple grid-world domain, and let these agents interact with the environment for a set number of 39 instances.

As mentioned, running this code produces both a JSON file tracking the experiment that 40 can be used (or shared) to run the same experiment again, and regenerate the plot seen in FIG16 .

We begin by unpacking the example in FIG13 to showcase the main design philosophy of 102 simple rl.

The library primarily consists of agents and environments (called "tasks" in the library).

Agents, by default, are all subclasses of the abstract class, Agent, which is only responsible for tasks currently implemented is presented in Table 1 .

(2) When defining an MDP instance, you must pass in functions of T and R that output a state and re-112 ward, respectively.

In this way, no MDP is ever responsible for enumerating either S or A explicitly, Games, and so on).

For now, let us focus on run agents on mdp function, which is the most ??? instances: The number of times to repeat the entire experiment (will be used to form 128 95% confidence intervals for all experiments conducted).

??? episodes: The number of episodes per instance.

An episode will consist of steps number 130 of steps, after which the agent is reset to the start state (but gets to remember what it has 131 learned so far).

??? steps: The number of steps per episode.

The plotting is set up to plot all of the above appropriately.

For instance, if a user sets episodes=1 134 but steps=50, then the library produces a step-wise plot (that is, the x-axis is steps, not episodes).

Running the function run agents on mdp will create a JSON file detailing all of the components script.

In this way, the JSON file is effectively a certificate that this plot can be reproduced if the 142 same experiment were run again.

We provide more detail on this feature in Section 3.2.

We can also run a similar experiment in the OpenAI Gym FIG11 ).

As can be seen in FIG11 , the structure of the experiment is identical.

Since we define a GymMDP,

we pass as input the name of the environment we'd like to produce: In this case, we're running ex-146 periments in CartPole-v1, but any of the usual Gym environment names will work.

We can also pass 147 in the render boolean flag, indicating whether or not we'd like to visualize the learning process.

Al-

ternatively, we can pass in the render every n episodes flag (along with render=True), which 149 will only render the agent's learning process every N episodes.

On longer experiments, we may want additional feedback about the learning process.

For this pur-pose, the run agents on mdp function also takes as input a Boolean flag verbose, which, if true,

will provide detailed episode-by-episode tracking of the progress of the experiment to the console.

There are a number of other ways to run experiments, but these examples capture the core experi-154 mental cycle.

Other time.

The idea is that these files can be shared across users of the library-if a user gives someone 173 else this file (and the necessary agents and environments), it is a contract that they can rerun exactly 174 the same experiment just run using simple rl.

Using one of these experiment files, the function reproduce from exp file(exp name), will 176 read the experiment file, reconstruct all the necessary components, rerun the entire experiment, and 177 remake the plot.

Thus, providing one of these JSON files is to be interpreted as a certificate that this 178 experiment is guaranteed to produce similar results.

As an example, consider again the code from Which will automatically generate the plot in FIG16 .

is an active area of development for the library.

To recap, the introduced components define the essence of the library:

??? Center everything around agents, MDPs, and interactions thereof.

??? Completely obscure the complexity of plotting and experiment tracking from the program-207 mer, while making it simple to plot and reproduce results if needed.

??? Simplicity above all else.

??? Treat things generatively-namely, MDPs transition models and reward functions are best 210 implemented as functions that return a state or reward, rather than enumerate all state-211 actions pairs.

In addition to the core experimental pipeline described above, the library is well stocked with other 214 utilities useful for RL and planning.

Plotting As is shown by FIG0 , plotting is tightly coupled with running experiments.

Each experiment type is connected with the same plotting script, stored in the library in 217 utils/chart utils.py.

The basic plot shows some measure of time along the x-axis (either in 218 episodes run or steps taken), with cumulative reward shown in the y-axis for each given algorithm.

While this plot is the default learning curve generated, the experimental pipeline gives the end pro-220 grammer control over the type of plot generated.

First, the flag cumulative plot for all of the core Abstraction A core approach to RL involves forming abstractions, either of state BID3

<|TLDR|>

@highlight

This paper introduces and motivates simple_rl, a new open source library for carrying out reinforcement learning experiments in Python 2 and 3 with a focus on simplicity.