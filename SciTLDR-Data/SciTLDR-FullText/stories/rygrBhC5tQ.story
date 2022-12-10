Humans acquire complex skills by exploiting previously learned skills and making transitions between them.

To empower machines with this ability, we propose a method that can learn transition policies which effectively connect primitive skills to perform sequential tasks without handcrafted rewards.

To efficiently train our transition policies, we introduce proximity predictors which induce rewards gauging proximity to suitable initial states for the next skill.

The proposed method is evaluated on a set of complex continuous control tasks in bipedal locomotion and robotic arm manipulation which traditional policy gradient methods struggle at.

We demonstrate that transition policies enable us to effectively compose complex skills with existing primitive skills.

The proposed induced rewards computed using the proximity predictor further improve training efficiency by providing more dense information than the sparse rewards from the environments.

We make our environments, primitive skills, and code public for further research at https://youngwoon.github.io/transition .

@highlight

Transition policies enable agents to compose complex skills by smoothly connecting previously acquired primitive skills.

@highlight

Proposes a scheme for transitioning to favorable strating states for executing given options in continuous domains. This uses two learning processes carried out simultaneously.

@highlight

Presents a method for learning policies for transitioning from one task to another with the goal of completing complex tasks using state proximity estimator to reward for transition policy.

@highlight

Proposes a new training scheme with a learned auxiliary reward function to optimise transition policies that connect the ending state of a previous macro action/option with good initiation states of the following macro action/option