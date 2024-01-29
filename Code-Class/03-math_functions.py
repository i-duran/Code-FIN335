#### Math module ####
# Reference: https://docs.python.org/3/library/math.html

import math

# Raise to the power
print("---Raise to the power---")
num = math.pow(2, 3)
print(num)

# Exponential
print("\n---Exponential---")
num = math.exp(2)
print(num)

# Log
print("\n---Logs---")
num = math.log(1)  # natural log
print(num)
print(math.log(math.exp(2)))

num = math.log(81, 3)  # log base 3
print(num)

# pi and e constants
print("\n---Pi and e constants---")
print(math.pi)
print(math.e)

# Other functions
print("\n---Other Functions---")
print("min(5, 2, 1):", min(5, 2, 1))
print("max(5, 2, 1):", max(5, 2, 1))
print("math.floor(3.2):", math.floor(3.2))  # = 3
print("math.ceil(3.2):", math.ceil(3.2))  # = 4
