import matplotlib.pyplot as plt
from bHPSOGWO import BHPSOGWO
from bGWO import BGWO
from bPSO import BPSO
import benchmarks
import numpy as np


function_to_be_used = benchmarks.F1

gwo = BGWO(function_to_be_used)
gwo.opt()
x1 = gwo.return_alpha_cruve()

hybrid = BHPSOGWO(function_to_be_used)
hybrid.opt()
x2 = hybrid.return_gbest_curve()


pso = BPSO(function_to_be_used)
pso.random_init()
pso.opt()
x3 = pso.return_result()

# To find iteration where minimum is reached
# idx1 = (np.where(x1 == min(x1))[0][0])
# idx2 = (np.where(x2 == min(x2))[0][0])
# idx3 = (np.where(x3 == min(x3))[0][0])


# print(idx1+1, min(x1)) #(iteration where minimum is reached, minimum value)
# print(idx2+1, min(x2))
# print(idx3+1, min(x3))

# PLOT
plt.figure()
plt.plot(x1)
plt.plot(x2)
plt.plot(x3)
plt.grid()
plt.legend(["GWO", "HPSOGWO", "PSO"], loc="upper right")

plt.title("Comparision of bHPSOGWO with bPSO and bGWO")
plt.xlabel("Number of Iterations")
plt.ylabel("Best Score")

plt.show()
