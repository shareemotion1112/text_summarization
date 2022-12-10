Minecraft is a videogame that offers many interesting challenges for AI systems.

In this paper, we focus in construction scenarios where an agent must build a complex structure made of individual blocks.

As higher-level objects are formed of lower-level objects, the construction can naturally be modelled as a hierarchical task network.

We model a house-construction scenario in classical and HTN planning and compare the advantages and disadvantages of both kinds of models.

Minecraft is an open-world computer game, which poses interesting challenges for Artificial Intelligence BID0 BID12 , for example for the evaluation of reinforcement learning techniques BID21 .

Previous research on planning in Minecraft focused on models to control an agent in the Minecraft world.

Some examples include learning planning models from a textual description of the actions available to the agent and their preconditions and effects BID4 , or HTN models from observing players' actions BID15 . , on the other hand, focused on online goal-reasoning for an agent that has to navigate in the minecraft environment to collect resources and/or craft objects.

They introduced several propositional, numeric BID7 and hybrid PDDL+ planning models BID8 .In contrast, we are interested in construction scenarios, where we generate instructions for making a given structure (e.g. a house) that is composed of atomic blocks.

Our longterm goal is to design a natural-language system that is able to give instructions to a human user tasked with completing that construction.

As a first step, in the present paper we consider planning methods coming up with what we call a construction plan, specifying the sequence of construction steps without taking into account the natural-language and dialogue parts of the problem.

For the purpose of construction planning, the Minecraft world can be understood as a Blocksworld domain with a 3D environment.

Blocks can be placed at any position having a non-empty adjacent position.

However, while obtaining a sequence of "put-block" actions can be sufficient for an AI agent, communicating the plan to a human user requires more structure in order to formulate higher-level instructions like build-row, or build-wall.

The objects being constructed (e.g. rows, walls, or an entire house) are naturally organized in a hierarchy where high-level objects are composed of lower-level objects.

Therefore, the task of constructing a high-level object naturally translates into a hierarchical planning network (HTN) BID19 BID20 BID22 BID6 .We devise several models in both classical PDDL planning BID5 BID13 ) and hierarchical planning for a simple scenario where a house must be constructed.

Our first baseline is a classical planning model that ignores the high-level objects and simply outputs a sequence of place-blocks actions.

This is insufficient for our purposes since the resulting sequence of actions can hardly be described in natural language.

However, it is a useful baseline to compare the other models.

We also devise a second classical planning model, where the construction of high-level objects is encoded via auxiliary actions.

HTN planning, on the other hand, allows to model the object hierarchy in a straightforward way, where there is a task for building each type of high-level object.

The task of constructing each high-level object can be decomposed into tasks that construct its individual parts.

Unlike in classical planning, where the PDDL language is supported by most/all planners, HTN planners have their own input language.

Therefore, we consider specific models for two individual HTN planners: the PANDA planning system BID3 BID2 and SHOP2 BID14 .

We consider a simple scenario where our agent must construct a house in Minecraft.

We model the Minecraft environment as a 3D grid, where each location is either empty or has a block of a number of types: wood, stone, or dirt.

FIG0 shows the hierarchy of objects of our construction scenario.

For the high-level structure the house consists of four stone walls, a stone roof, and a door.

The walls and the roof are further decomposed into single rows that need to be built out of individual blocks.

The door consists of two gaps, i.e., empty positions inside one of the walls.

As our focus is on the construction elements we abstract low-level details away.

For example, we avoid encoding the position of the agent and assume that all positions are always reachable.

We also assume Minecraft's creative mode, where all block types are always available so we do not need to keep track of which blocks are there in the inventory.

This is a very simplistic model, where planning focuses simply on the construction actions (i.e. placing or removing blocks), of high-level structures.

Nevertheless, it can still pose some challenges to modern planners, specially due to the huge size of the Minecraft environment.

Our first model is a classical planning model in the PDDL language that consists of only two actions: putblock(?location, ?block-type) and remove-block(?location, ?

block-type) where there is a different location for each of the x-y-z coordinates in a 3D grid.

The goal specifies what block-type should be in each location.

As blocks cannot be placed in the air, the precondition of put-block requires one of the adjacent locations of ?

location to be non-empty.

Other than that, blocks of any type can always be added or removed at any location.

The goal is simply a set of block at facts.

A limitation of this simple model is that it completely ignores the high-level structure of the objects being constructed.

As there is no incentive to place blocks in certain order, a high-level explanation of the plan may be impossible.

To address this, we introduce auxiliary actions that represent the construction of high-level objects.

Figure 2 shows the auxiliary actions that represent building a wall.

The attributes of the wall are specified in the initial state via attributes expressed by predicates wall dir, wall length, wall height, wall type, and current wall loc.

In order to avoid the huge amount of combinations of walls that could be constructed of any dimensions and in any direction, the walls that are relevant for the construction at hand are specified in the initial state via these predicates.

These three actions decompose the construction of a wall into several rows.

Action begin wall ensures that no other high-level object is being constructed at the moment and adds the fact constructing wall to forbid the construction of any other wall (or roof) until the current wall has been finished.

Action build row in wall ensures that a row of the given length will be built on the corresponding location and direction by adding predicates (building row) and (rest row ?loc ?

len ?dir ?t).

Simultaneously, it updates the location for the rest of the wall to be built and decreases its height by one. (not (constructing wall)) (not (current wall ?

w))))Figure 2: Auxiliary PDDL actions to build a wall.

When the height is zero, the action end wall becomes applicable, which finishes the construction of the wall.

In the goal we then use the predicates wall at and roof at that force the planner to use these constructions, instead of a set of block at facts as we did in the simple model.

HTN models encode the construction of high-level objects in a straightforward way by defining tasks such as build house, build wall and build row.

These tasks will then be decomposed with methods until only primitive tasks will be left, in our case place-block and remove-block.

We consider specific models for two individual HTN planners: the PANDA planning system BID3 BID2 ) and SHOP2 BID14 .

PANDA uses an HTN formalism BID9 , which allows combining classical and HTN planning.

The predicates describing the world itself, i.e. the relations between different locations remain the same as in the PDDL model, as do the place-block and remove-block primitive actions.

On top of this, high-level objects are described as an HTN where each object corresponds to a task, without requiring to express their attributes with special predicates as we did in the PDDL model.

Specifically, we defined tasks that correspond to building a house, a wall, a roof, a row of blocks, and the door.

FIG1 shows the methods used to decompose the task of building a wall.

These methods work in a recursive fashion over the height of the wall.

For walls with height one, the build wall 1 method is used to build them.

For walls with larger height, the build wall 2 method decomposes the task of building them into building a row in the current location and building the rest of the wall (i.e., a wall of height-1) in the location above the previous one.

These subtasks are ordered, so that walls are always built from bottom to top.

The methods for buildrow and buildroof work in the same fashion, while buildhouse only has one method decomposing the house into four walls, the roof, and the door.

The task builddoor also has just one method stating which two blocks have to be removed to form a door.

Choosing this way of modeling the door by first forcing the planner to place two blocks and later removing them again may seem inefficient, but for communication with a human user this may be preferable over indicating that these positions should remain empty in the first place.

The SHOP2 model follows a similar hierarchical task structure as the PANDA model, having methods for decomposing the house into walls, a wall into rows and rows into single blocks.

Since one of the advantages of SHOP2 is that it can call arbitrary LISP functions, we can represent the locations using integers as coordinates and replace the predicates used in PANDA and PDDL to express their relations by simple arithmetic operations.

This also allows us to compute the end point of rows of any given length in a given direction, which means we can construct the walls by alternating the direction of the rows.

Based on this, we define two different recursive decompositions of walls as shown in FIG2 .

In the first method we simply build the row starting in the current location, while in the second method we change the direction of the row we want to build and identify the position that would previously have been the end of the row by replacing the x-coordinate with x`length´1.

Since this computation is different for each direction, we need separate methods for them.

Apart from this, the decomposition structure is the same as with PANDA, building the walls, roof, and rows incrementally using a recursive structure.

To evaluate the performance of common planners on our models 1 , we scale them with respect to two orthogonal parameters: the size of the construction, and the size of the cubic 3D world we are considering.

We use different planners for each model.

For the classical planning models we use the LAMA planner BID16 .

The PANDA planning system implements several algorithms, including plan space POCL-based search methods BID3 BID2 , SAT-based approaches , and forward heuristic search .

We use a configuration using heuristic search with the FF heuristic, which works well on our models.

For SHOP2, we use the depthfirst search configuration BID14 .

All experiments were run on an Intel i5 4200U processor with a time limit of 30 minutes and a memory limit of 2GB.In our first experiment, we scale the size of the house starting with a 3ˆ3ˆ3 house and increasing one parameter (length, width, and height) at a time (4ˆ3ˆ3, 4ˆ41 3, . . .

, 9ˆ9ˆ9.).

The size of the 3D world is kept as small as possible to fit the house with some slack, so initially is set to 5ˆ5ˆ5 and is increased by one unit in each direction every three steps, once we have scaled the house in all dimensions.

The upper row of FIG3 shows the search and total time of the planners on the different models.

The construction size in the x-axis refers to the number of blocks that need to be placed in the construction.

All planners scale well with respect to search time, solving problems of size up to 9ˆ9ˆ9 in just a few seconds.

The non-hierarchical PDDL planning model (PDDL blocks) that only uses the place-block and remove-block actions without any hierarchical information is the one with worst search performance.

Moreover, it also results in typically longer plans that build many "support" structures to place a block in a wall without one of the adjacent blocks in the wall being there yet.

However, there is a huge gap between search and total time for the PANDA and PDDL models, mostly due to the overhead of the grounding phase.

SHOP2 does not do any preprocessing or grounding so it is not impacted by this.

For the PANDA and PDDL models, total time significantly increases every three problems, whenever the world size is increased.

This suggests that, somewhat counterintuitively, the size of the world environment has a greater impact on these planners' performance than the size of the construction.

In the PDDL based approaches, the number of operators and facts produced in the preprocessing shows a similar trend so the planner's performance seems directly influenced by the size of the grounded task.

For PANDA, on the other hand, we observe a linear increase in the number of facts and only a comparatively small increase in the number of operators.

To test more precisely what is the impact of increasing the world size, we ran a second set of experiments where we kept the size of the house fixed at 5ˆ5ˆ5 and just increased the size of the world.

As shown in the bottom part of FIG3 the performance of SHOP2 is not affected at all, since it does not require enumerating all possible locations.

Search time for PANDA also stays mostly constant, but the overhead in the preprocessing phase dominates the total time.

This contrasts with the number of operators and facts, which is not affected by the world size at all.

The PDDL based models are also affected in terms of preprocessing time, due to a linear increase in the number of facts and operators with respect to world size, but to a lesser degree.

However, search time increases linearly with respect to the world size due to the overhead caused in the heuristic evaluation.

We have introduced several models of a construction scenario in the Minecraft game.

Our experiments have shown that, even in the simplest construction scenario which is not too challenging from the point of view of the search, current planners may struggle when the size of the world increases.

This is a serious limitation in the Minecraft domain, where worlds with millions of blocks are not unrealistic.

Lifted planners like SHOP2 perform well.

However, it must be noted that they follow a very simple search strategy, which is very effective on our models where any method decomposition always leads to a valid solution.

However, it may be less effective when other constraints must be met and/or optimizing quality is required.

For example, if some blocks are removed from the ground by the user, then some additional blocks must be placed as auxiliary structure for the main construction.

Arguably, this could be easily fixed by changing the model so that whenever a block cannot be placed in a target location, an auxiliary tower of blocks is built beneath the location.

However, this increases the burden of writing new scenarios since suitable task decompositions (along with good criteria of when to select each decomposition) have to be designed for all possible situations.

This makes the SHOP2 model less robust to unexpected situations that were not anticipated by the domain modeler.

PANDA, on the other hand, supports insertion of primitive actions BID9 , allowing the planner to consider placing additional blocks, e.g., to build supporting structures that do not correspond to any task in the HTN.

This could help to increase the robustness of the planner in unexpected situations where auxiliary structures that have not been anticipated by the modeler are needed.

However, this is currently only supported by the POCL-plan-based search component and considering all possibilities for task insertion significantly slows down the search and it runs out of memory in our scenarios.

This may point out new avenues of research on more efficient ways to consider task insertion.

In related Minecraft applications, cognitive priming has been suggested as a possible solution to keep the size of the world considered by the planner at bay BID17 .

In construction scenarios, however, large parts of the environment can be relevant so incremental grounding approaches may be needed to consider different parts of the scenario at different points in the construction plan.

Our models are still a simple prototype and they do not yet capture the whole complexity of the domain.

We plan to extend them in different directions in order to capture how hard it is to describe actions or method decompositions in natural language.

For example, while considering the position of the user is not strictly necessary, his visibility may be important because objects in his field of view are easier to describe in natural language.

How to effectively model the field of vision is a challenging topic, which may lead to combinations with external solvers like in the planning modulo theories paradigm BID10 .Another interesting extension is to consider how easy it is to express the given action in natural language and for example by reducing action cost for placing blocks near objects that can be easily referred to.

Such objects could be landmarks e.g. blocks of a different type ("put a stone block next to the blue block") or just the previously placed block (e.g., "Now, put another stone block on top of it").

<|TLDR|>

@highlight

We model a house-construction scenario in Minecraft in classical and HTN planning and compare the advantages and disadvantages of both kinds of models.