""" 
function FirstVisitMonteCarlo
    Initialize Q-values for all state-action pairs
    Initialize an empty list to track returns for each state-action pair
    Initialize an arbitrary policy for each state
    Initialize an empty list for recording an episode
    Initialize an empty set to keep track of visited state-action pairs

    Repeat for each episode:
        Generate an episode by following the current policy
        Record each state, action, and the received reward in the episode

        Initialize a variable G to keep track of the total reward
        For each step in reverse order of the episode:
            Get the current state and action from the episode
            If this is the first visit to the state-action pair:
                Calculate the total discounted reward G from this step onward
                Add G to the list of returns for this state-action pair
                Update the estimated Q-value for this state-action pair
                Improve the policy based on Q-values

    Return the estimated Q-values and the improved policy

    
 """