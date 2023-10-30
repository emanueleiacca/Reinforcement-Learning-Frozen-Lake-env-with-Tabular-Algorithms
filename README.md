## Reinforcement Learning Experimentation on Frozen Lake 8x8

This repository contains the code for a reinforcement learning experimentation on the Frozen Lake 8x8 environment. The following algorithms are implemented:

* Sarsa
* Expected Sarsa
* Q-Learning
* Double Q-Learning

The experiments are conducted using a custom epsilon greedy policy, which decays linearly from 1.0 to 0.01 over the course of 3 million steps. The performance of each algorithm is evaluated on a set of 100 episodes, and the results are logged to TensorBoard.

### Requirements

* Python 3.7+
* NumPy
* Gym
* TensorBoard

### Running the experiments

To run the experiments, run the following command:

```
python activity_1.py
```

This will train and evaluate each algorithm for 3 million steps and log the results to TensorBoard.

### TensorBoard

To view the TensorBoard logs, run the following command:

```
tensorboard --logdir logs
```

This will open a web browser at http://localhost:6006 where you can view the results.

### Analysis

The following observations can be made from the TensorBoard logs:

* All four algorithms learn to solve the Frozen Lake 8x8 environment with sufficiently high epsilon decay.
* Q-Learning outperforms the other algorithms in terms of average reward per episode.
* Sarsa and Expected Sarsa have similar performance.
* Double Q-Learning performs slightly worse than Sarsa and Expected Sarsa.

### Conclusion

These experiments show that all four algorithms are able to learn to solve the Frozen Lake 8x8 environment. However, Q-Learning outperforms the other algorithms in terms of average reward per episode.

## Hyperparameter Optimization

We also used Optuna to find the best set of hyperparameters for the Q-Learning algorithm. The best hyperparameters for the stochastic environment are as follows:
Discount rate: 0.9961459723114788
Learning rate: 0.06698053340104898
Higher discount rates and lower learning rates tend to perform better as indicated by Optuna
### Evaluating the Q-table
Once the Q-table has been trained, we can evaluate its performance by playing the game using the max policy. We do this by running the following code:
```
python Qtable_fromPickle.py
```
This code will evaluate the Q-table by playing 100 episodes of the game and reporting the average reward and number of steps per episode.
### Results
These hyperparameters achieve an average reward of 0.71 on the stochastic environment, which is significantly better than the baseline of 0.54. This shows that hyperparameter optimization can be a very effective way to improve the performance of reinforcement learning agents.

