import numpy as np
from sympy import Symbol
from sympy.solvers import solve
import matplotlib.pyplot as plt
import os
import time
import sys
# Import Q-CTRL Python package.
from qctrl import Qctrl
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 9

# Start a session with the API.
qctrl = Qctrl()

print("imports and packages work")
