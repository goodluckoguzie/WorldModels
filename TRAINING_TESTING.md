# Enhancing Reinforcement Learning-based Social Navigation using Predictive World Models



## Description
This repository contains the code, models, and results of our research on improving Reinforcement Learning (RL) performance in Social Navigation (SN) tasks by applying predictive world models.



Our research focused on two primary questions: 
1. Can world models enhance RL-based social navigation?
2. Are the predictive capabilities of world models beneficial to RL-based social navigation?



To address these, we proposed and rigorously evaluated three novel models that integrate world models into the RL framework. These models significantly outperform conventional RL models in complex social navigation tasks, demonstrating the potential of predictive world models.



Despite the success of our models, we highlight the need for continued research, particularly in maintaining personal boundaries in complex social environments.



![Research Overview](images/overview.jpg)



## Models



### 2StepAhead Predictive World Model
This model anticipates the future two steps, helping the agent foresee the future states and select an appropriate action accordingly. It showed a significant improvement in performance over conventional RL models.



![2StepAhead Model](images/2StepAhead.jpg)



### Multi Action State Predictive Model (MASPM)
MASPM predicts the future state of the world for all possible actions and chooses the action that leads to the most favorable state, resulting in more informed action selection and improved performance.



![MASPM Model](images/MASPM.jpg)



### Action Dependent Two Step Predictive Model (ADTSPM)
ADTSPM model combines the strategies used by the two previous models. It predicts two steps ahead for all possible actions and chooses the action leading to the most favorable state. This model showed the most considerable performance improvement among the three models.



![ADTSPM Model](images/ADTSPM.jpg)



## Performance Metrics and Evaluation
We deployed diverse metrics to evaluate the performance of our models against the baselines during both training and testing phases.



During the **training phase**, we focused on:
- Cumulative reward: The total reward accumulated over time.
- Training time: The time taken by the model to train.
- Episodes to convergence: The number of episodes needed for the model to converge.



During the **testing phase**, we recorded metrics like:
- Discomfort counts: The number of instances causing discomfort to the humans in the environment.
- Average velocities: The average speed of the robot during navigation.
- Distance traveled: The total distance traversed by the robot.
- Simulation times: The time taken for the simulation.
- Human collisions: The number of times the robot collided with a human.
- Max steps: The maximum number of steps taken by the robot in an episode.
- Successful runs: The number of successful navigation episodes.
- Idle times: The periods where the robot isn't moving.
- Personal space compliance rate: The rate at which the robot respects the personal space of humans.



These metrics encompass the agent’s interactions with humans, its movements, navigation efficiency, task accomplishment, and overall performance in a human-robot interactive environment.



For detailed performance results, refer to our research paper: [Predictive World Models for Social Navigation](PAPER_LINK.md).



## Training and Testing
The training and testing of our world models consist of several steps. Each step includes a command that you can run in your terminal to execute that specific step.



**Note:** For detailed steps with corresponding commands, please refer to the [Training and Testing Guide](TRAINING_TESTING.md).



## License
[License information](LICENSE.md)



We encourage the academic community to use and build upon our research. Feel free to contact us with any questions or feedback.
 
