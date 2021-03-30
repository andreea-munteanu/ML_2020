import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Exercise 2

Suppose we learned a decision tree from a training set with binary output values (either 0 or 1). We find that for a 
leaf node  𝑙 ,there are  𝑀  training examples falling into it; its entropy is  𝐻 .

1. Plot the entropy for all possible ratios of 0 to 1;
2. Create a simple algorithm which takes as input  𝑀  and  𝐻  and that outputs the number of training examples 
misclassified by leaf node  𝑙 .
"""

"""
_________________________________________________________________________________________________
                              EXAMPLE OF BINARY DATASET FOR RANDOM TREE 
_________________________________________________________________________________________________
"""
X = pd.DataFrame({'NotHeavy' : [1, 1, 0, 0, 1, 1, 1, 0],
                  'Smelly' : [0, 0, 1, 0, 1, 0, 0, 1],
                  'Spotted' : [0, 1, 0, 0, 1, 1, 0, 0],
                  'Smooth' : [0, 0, 1, 1, 0, 1, 1, 0]
                  })
Y = pd.Series([1, 1, 1, 0, 0, 0, 0, 0])  # Edible
X['Y'] = Y

print("Dataset is: ", X, sep='\n')

# print(X)

"""
_________________________________________________________________________________________________
                                          COMPUTING ENTROPY 
_________________________________________________________________________________________________
"""


# determine general entropy
def entropy(our_X) :
    entropy_val = 0
    Y_col = our_X.keys()[-1]
    values = our_X[Y_col].unique()
    for value in values :
        term = our_X[Y_col].value_counts()[value] / len(our_X[Y_col])
        entropy_val = entropy_val - np.log2(term) * term
    return abs(entropy_val)


# # for check:
# print(entropy(X))


# determine number of correct entries for leaf node
def count_correct_entries(our_Y) :
    f = our_Y.value_counts(sort=False)
    # f[0] = number of 0s
    # f[1] = number of 1s
    return f


count_correct_entries(Y)


def correctly_and_incorrectly_classified(our_X) :
    number_of_instances = our_X.shape[0]  # number of instances (rows) in dataset
    correctly_classified = []  # list of instance indexes where Y is correctly classified
    for index in range(number_of_instances) :
        if X.loc[index, 'Y'] == 1 :
            correctly_classified.append(index)
    # print("Instances correctly classified: ", correctly_classified)
    incorrectly_classified = []  # instances where T is incorrectly classified
    for index in range(X.shape[0]) :
        if index not in correctly_classified :
            incorrectly_classified.append(index)
    # print("Instances incorrectly classified: ", incorrectly_classified)
    return correctly_classified, incorrectly_classified


# Find all possible entropies by deleting instances from dataset
# All entropies found will be plotted
def compute_all_entropies(our_X, our_Y) :
    entropies = [entropy(our_X)]
    correct_entries, incorrect_entries = count_correct_entries(our_Y)
    correctly_classified, incorrectly_classified = correctly_and_incorrectly_classified(our_X)
    number = [incorrect_entries / correct_entries]
    for i in range(correct_entries) :
        # deleting instance and calculating entropy of dataset - {current instance}
        new_X = our_X.drop(correctly_classified[:i])
        number.append(incorrect_entries / (correct_entries - i))
        entropies.append(entropy(new_X))
        # ent = entropy(our_X.drop(correctly_classified[:i]))
        # entropies.append(ent)
        for j in range(1, incorrect_entries) :
            new_X2 = new_X.drop(incorrectly_classified[:j])
            number.append(len(incorrectly_classified[:j]) / (correct_entries - i))
            entropies.append(entropy(new_X2))

    return entropies, number
    # # At this point, an entropy can appear multiple times
    # # We create a new list of entropies, each entropy appearing only once, and a list of frequencies
    # freq = []
    # occurence = []
    #
    # for i in entropies :
    #     if i in occurence :
    #         freq[occurence.index(i)] += 1
    #     else :
    #         occurence.append(i)
    #         freq.append(1)
    # return occurence, freq


"""
_________________________________________________________________________________________________
                                     PLOTTING ENTROPIES
_________________________________________________________________________________________________
"""


def plot_entropies(our_X, our_Y) :
    entropies, ratios = compute_all_entropies(our_X, our_Y)
    # print(entropies, number_of_occurence, sep='\n')
    # probabilities = []
    # # total number of entropies
    # total = sum(number_of_occurence)
    # for i in range(len(number_of_occurence)) :
    #     probabilities.append((number_of_occurence[i] / total))
    print("Ratios: ", ratios)
    print("Entropies: ", entropies)
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    ax.set_title('Entropy plotting')
    plt.ylabel('Ratio')
    plt.xlabel('Entropies')
    plt.stem(entropies, ratios)
    plt.show()


plot_entropies(X, Y)

"""
_________________________________________________________________________________________________
                                         ALGORITHM (M, H)
_________________________________________________________________________________________________
"""


"""
We know that our dataset has binary outputs ⇒ Y ∈ {0,1}. The entropy of attribute l will thus be:
    H = -P(0) ✕ log2(P(0)) + P(1) ✕ log2P(1),  where:   P(0) = incorrectly classified instance
                                                         P(1) = correctly classified instance
We already know the values of H and P(1) = M. We need to find the value of P(0)
Mathematically:

    H - P(1) ✕ log2P(1) = -P(0) ✕ log2(P(0))  |(-1)
    
    (P(1) ✕ log2P(1)) - H = P(0) ✕ log2(P(0))
    
    P(1) = M   ⇒   M ✕ log2(M) - H = P(0) ✕ log2(P(0))
                  |________________|
                          val       = P(0) ✕ log2(P(0))
                          
We now need to find P(0), by naive parsing. We will use binary search
"""

def misclassified_training_examples(M, H):
    val = M * np.math.log2(M) - H
    total = X.shape[0] # total number of instances
    # P_0 will be an integer (represents the number of misclassified instances)
    st = 0
    dr = total
    m = (st + dr) // 2
    while st <= dr :
        if m == val:
            return m
        elif m > val:
            dr = m - 1
        else:
            st = m + 1
    return st
