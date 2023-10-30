import numpy as np
import matplotlib.pyplot as plt

def custom_epsilon(current_step, total_steps=30):
    initial_epsilon = 0.9
    final_epsilon = 0.001
    decay_factor = -np.log(final_epsilon / initial_epsilon) / total_steps
    epsilon = initial_epsilon * np.exp(-decay_factor * current_step)
    return epsilon
#example to see how our function works
'''
total_steps = 30 
for step in range(total_steps):
    epsilon = custom_epsilon(step, total_steps)
    print(f"Step {step + 1} - Epsilon: {epsilon:.4f}")
'''
#output
'''
Step 1 - Epsilon: 0.9000
Step 2 - Epsilon: 0.7174
Step 3 - Epsilon: 0.5719
Step 4 - Epsilon: 0.4558
Step 5 - Epsilon: 0.3634
Step 6 - Epsilon: 0.2896
Step 7 - Epsilon: 0.2309
Step 8 - Epsilon: 0.1840
Step 9 - Epsilon: 0.1467
Step 10 - Epsilon: 0.1169
Step 11 - Epsilon: 0.0932
Step 12 - Epsilon: 0.0743
Step 13 - Epsilon: 0.0592
Step 14 - Epsilon: 0.0472
Step 15 - Epsilon: 0.0376
Step 16 - Epsilon: 0.0300
Step 17 - Epsilon: 0.0239
Step 18 - Epsilon: 0.0191
Step 19 - Epsilon: 0.0152
Step 20 - Epsilon: 0.0121
Step 21 - Epsilon: 0.0097
Step 22 - Epsilon: 0.0077
Step 23 - Epsilon: 0.0061
Step 24 - Epsilon: 0.0049
Step 25 - Epsilon: 0.0039
Step 26 - Epsilon: 0.0031
Step 27 - Epsilon: 0.0025
Step 28 - Epsilon: 0.0020
Step 29 - Epsilon: 0.0016
Step 30 - Epsilon: 0.0013
'''

# Create a graph with Epsilon on the y-axis and step on the x-axis
'''
    epsilon_values = []  # List to store Epsilon values
    step_values = []  # List to store corresponding steps
    #in the in loop:
        epsilon_values.append(epsilon)  # Append Epsilon value
        step_values.append(step)  # Append corresponding step
        
    plt.figure(figsize=(8, 6))
    plt.plot(step_values, epsilon_values, marker='o', linestyle='-')
    plt.title('Epsilon vs. Step')
    plt.xlabel('Step')
    plt.ylabel('Epsilon')
    plt.grid(True)
    plt.show()
'''