"""
Exercise 4/Lab 14
####################################################################################
Recreate the step-by-step example from the lab, but also using the fact that μ1=μ2/2.
Use the same data, same starting hypothesis and perform at least 3 iterations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import norm

""" six-point dataset """

d = pd.DataFrame({
    'X': [10, 11, 12, 19, 20, 22]
})
fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d['X'], np.zeros_like(d))
plt.show()

""" step 1: initial """

c = ListedColormap(['red', 'green'])

means = [11, 22]  # μ1=μ2/2
variances = [1.5, 1.5]


def plot_pdfs(means, variances, ax, colours):
    """ Plot the pdfs for given means and variances """
    for i in range(0, 2):
        mean, var = means[i], variances[i]
        x = np.linspace(norm.ppf(0.01, mean, np.sqrt(var)), norm.ppf(0.99, mean, var), 100)
        pdf = norm.pdf(x, mean, var)
        ax.plot(x, pdf, color=colours(i), label=f"$\mu_{i}$")
        ax.legend()


fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d))
plot_pdfs(means, variances, ax, c)
plt.show()

""" step 2: expectation """

p_1 = norm.pdf(d, means[0], variances[0])
p_2 = norm.pdf(d, means[1], variances[1])

E0 = p_1 / (p_1 + p_2) # E for j=0
E1 = p_2 / (p_1 + p_2) # E for j=1
print("E for j=0:\n", E0)
print("E for j=1:\n", E1)

# colouring each point according to the expected values:
c_grad = LinearSegmentedColormap.from_list('mygrad', [c(0), c(1)])

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d), c=E1, cmap=c_grad)
plot_pdfs(means, variances, ax, c)
plt.show()

""" step 3: maximization """

x = d.to_numpy()
means[0] = (E0 * x).sum() / E0.sum()
means[1] = (E1 * x).sum() / E1.sum()
print("Updated means are: ", means)

# plotting distributions corresponding to new means:
fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d))
plot_pdfs(means, variances, ax, c)
plt.show()

""" second iteration: """

p_1 = norm.pdf(d, means[0], variances[0])
p_2 = norm.pdf(d, means[1], variances[1])

E0 = p_1 / (p_1 + p_2) # E for j=0
E1 = p_2 / (p_1 + p_2) # E for j=1

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d), c=E1, cmap=c_grad)
plot_pdfs(means, variances, ax, c)
plt.title('Expectation')
plt.show()

means[0] = (E0 * x).sum() / E0.sum()
means[1] = (E1 * x).sum() / E1.sum()

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d))
plot_pdfs(means, variances, ax, c)
plt.title("Maximisation")
plt.show()


""" third iteration """

p_1 = norm.pdf(d, means[0], variances[0])
p_2 = norm.pdf(d, means[1], variances[1])

E0 = p_1 / (p_1 + p_2) # E for j=0
E1 = p_2 / (p_1 + p_2) # E for j=1

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d), c=E1, cmap=c_grad)
plot_pdfs(means, variances, ax, c)
plt.title('Expectation')
plt.show()

means[0] = (E0 * x).sum() / E0.sum()
means[1] = (E1 * x).sum() / E1.sum()

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d))
plot_pdfs(means, variances, ax, c)
plt.title("Maximisation")
plt.show()