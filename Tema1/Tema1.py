import numpy as np
import scipy.stats as stats
from scipy.stats import geom
import matplotlib.pyplot as plt
# import pandas as pd

# S     |   1 |   2 |   3 |   4 |   5 |   6
# P(S)  | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |

die = stats.randint(1, 7) # die can take any value in [1,6]
p = die.pmf(3)            # probability for die to be 3 at 1 roll: 0.16666666...
# print("probability = ", p)

r = range(1, 7)
# distribution object for which the probability of happenstance is p = 0.1(6)%:
die_is_3 = stats.geom(p)
probabilities = [die_is_3.pmf(i) for i in r]

die_sample = np.random.randint(1, 7, size=100)

plt.style.use('bmh')
fig, ax = plt.subplots()
plt.xlabel('n')
plt.ylabel('p(n)')
ax.set_title('Geometric distribution')
plt.hist(die_sample, bins=50)
plt.bar(r, probabilities)

mean, var = geom.stats(p, moments='mv')

print("mean(geom) = ", mean, ", mean(formula) = ", 1/p)
print("variance(geom) = ", var, ", by formula = ",  (1-p)/(p*p))
plt.show()
