# %%
import os
import pandas as pd
import numpy as np

from matpower import start_instance, path_matpower
from matpowercaseframes import CaseFrames
from scipy.stats.distributions import chi2


# %%
# Input parameters
# Example combination of parameters for 14 bus system-> 14,500,13
num_of_bus = int(input("Enter number of bus [5,14,30 etc] : "))  #  Number of buses in the system
num_of_datasets = int(input("Total number of datasets [half will be trained, half will be predicted] [500,800,...] : "))  #  Number of datasets to be generated
k_value = int(input("Value of k [for KNN] [11,13,17,...]: "))  #  Value of K for KNN
print(f"FDI attack detection using KNN algorithm for {num_of_bus} bus System")
print("*" * 50)

# total time taken for script to run
start_time = pd.Timestamp.now()

# %% Import KNN algo
from ml_algo import KNN as the_algo_for_pred

var = the_algo_for_pred(k_value)

# %%
path = os.path.join(path_matpower, f"data/case{num_of_bus}.m")
cf = CaseFrames(path)

# %%
m = start_instance()
mpc = m.eval(f"case{num_of_bus}", verbose=False)
mpc = m.runpf(mpc)

# %%
frombus = np.array(mpc["branch"][:, 0]).astype(int)
tobus = np.array(mpc["branch"][:, 1]).astype(int)
num_of_line = len(frombus)
baseMVA = mpc["baseMVA"]

# %%
r = pd.DataFrame(mpc["branch"][:, 2])  # line resistance
x = pd.DataFrame(mpc["branch"][:, 3])  # line reactance
b = pd.DataFrame(mpc["branch"][:, 4])  # B matrix
s = len(pd.DataFrame(mpc["bus"][:, 8]))  # number of states

# %%
Y = np.zeros((num_of_bus, num_of_bus), dtype=complex)
for k in range(num_of_line):
    Y[frombus[k] - 1][tobus[k] - 1] = -1 / (x[0][k] * 1j)
    Y[tobus[k] - 1][frombus[k] - 1] = -1 / (x[0][k] * 1j)

for i in range(num_of_bus):
    for j in range(num_of_line):
        if frombus[j] == i + 1:
            Y[i][i] += 1 / (x[0][j] * 1j)
        elif tobus[j] == i + 1:
            Y[i][i] += 1 / (x[0][j] * 1j)

# %% Generate datasets

multipliers = np.random.random((num_of_datasets)) * np.random.randint(1, 100)
multipliers = np.append(multipliers, 1)

Y_org = Y.copy()

final_H, final_Z, final_W = None, None, None
retryFlag, idx = 0, 0
expected_result, predicted_result = None, None
correct_prediction,incorrect_prediction = 0,0

while idx < len(multipliers):
    multiplier = multipliers[idx]

    # Sometime the G matrix is not invertible, so we retry with a different multiplier
    # if the G matrix is not invertible after 10 retries, we stop
    if retryFlag > 10:
        print("MAX RETRY REACHED")
        break

    # %%
    # multiply each element of Y with multiplier
    Y = Y_org * multiplier
    B = np.imag(Y)

    # %%
    pinj = np.zeros((num_of_bus, 1))

    for i in range(num_of_bus):
        for j in range(num_of_line):
            if frombus[j] == i + 1:
                pinj[i] += mpc["branch"][j][14] / baseMVA
            if tobus[j] == i + 1:
                pinj[i] += mpc["branch"][j][16] / baseMVA

    # %%
    pf = np.zeros((num_of_line, 1))
    for i in range(num_of_line):
        pf[i] = mpc["branch"][i][14] / baseMVA

    # %%
    Z = np.concatenate((pinj, pf))
    pd.DataFrame(Z)
    z = len(Z)  # total number of states

    # %%
    H1 = np.zeros((num_of_bus, num_of_bus))
    for i in range(num_of_bus):
        for j in range(num_of_bus):
            H1[i][j] = -B[i][j]

    H2 = np.zeros((num_of_line, num_of_bus))
    for i in range(num_of_line):
        H2[i][frombus[i] - 1] = B[frombus[i] - 1][tobus[i] - 1]
        H2[i][tobus[i] - 1] = -B[tobus[i] - 1][frombus[i] - 1]

    H = np.concatenate((H1, H2))

    # %%
    # FDI Attack
    # Stealthy Complete information Attack
    col = np.shape(H)[1]
    c = np.random.rand(col, 1)
    if np.random.rand() > 0.5:
        a = H @ c  # a = HC
    else:
        a = np.random.rand(z, 1)
    Z = np.add(Z, a)  # Z = Z + a

    # %%
    W = np.zeros((z, z))
    np.fill_diagonal(W, np.random.random())

    # %%
    G = H.transpose() @ W @ H

    # check if G is Singular
    if np.linalg.det(G) == 0:
        multipliers[idx] = np.random.random() * np.random.randint(1, 100)
        idx -= 1
        retryFlag += 1
        print(f"Singular G matrix, retrying... {retryFlag}")
        continue
    else:
        retryFlag = 0

    X = np.linalg.inv(G) @ H.transpose() @ W @ Z

    # %%
    Zest = H @ X  # estimated Z
    EstError = Z - Zest  # estimated error

    # %%
    fCap = 0  # objective function
    for i in range(num_of_bus):
        fCap += W[i][i] * (EstError[i] ** 2)
    fCap = fCap[0]

    # %%
    k = Z.shape[0] - X.shape[0]  # degree of freedom
    gamma = 0.99
    chiValue = chi2.ppf(gamma, k)
    result = "Invalid gamma value"
    if fCap > chiValue:
        result = "Bad data detected"
    else:
        result = "All data is good"

    # print(f"Iteration {idx+1} : {result}")
    print(f"Iteration {idx+1} is running...")

    # %%
    # Training the model

    # First half of dataset will be trained,
    # Second half of dataset will be predicted
    # Also using actual detection technique to check the accuracy of the model
    if idx >= len(multipliers)//2 :
        predicted_result = var.predict([(H,W,Z)])[0]
        if predicted_result == result:
            correct_prediction+=1
        else:
            incorrect_prediction+=1
    else:
        x_dataset = [(H, W, Z)]
        y_dataset = [result]
        var.fit(x_dataset, y_dataset)

    idx += 1  # increase the index : end of while loop

if retryFlag <= 10:
    print("*" * 50)
    print("Training completed")
    print("*" * 50)
    # %%
    var.get()
    # accuracy
    print(f"FDI attack detection using KNN algorithm for {num_of_bus} bus System")
    print(f"Correct prediction : {correct_prediction}")
    print(f"Incorrect prediction : {incorrect_prediction}")
    print(f"Accuracy : {correct_prediction/(correct_prediction+incorrect_prediction)*100}%")

end_time = pd.Timestamp.now()
# total time in seconds
print(f"Total time taken : {(end_time-start_time).total_seconds()} seconds")
