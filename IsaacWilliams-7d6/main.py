#Libraries
import numpy as np


#%%
#Functions
def Pauli(i):
    if i=='x' :
        return np.array([[0, 1], [1, 0]])
    elif i=='y':
        return np.array([[0, -1j], [1j, 0]])
    elif i=='z':
        return np.array([[1, 0], [0, -1]])

def Tensor(MatA, MatB):
    return np.kron(MatA,MatB)

def Kronecker (n, j, i): 
#n is the number of qubits, j is the position of the Pauli matrix, and i is the type of Pauli matrix (x, y, z)
    if n == 1:
        if j == 1:
            return Pauli(i)
        else:
            return I_2
    elif n > 1:
        if j == n:
            return np.kron(Pauli(i), Kronecker(n-1, j, i))
        else:
            return np.kron(I_2, Kronecker(n-1, j, i))    
    else:
        raise ValueError("n must be a positive integer")

def H_Ising(J):
    epsilon = 0.2
    kappa = np.array([[epsilon],[epsilon],[epsilon]])
    #kappa is to remove bad solutions
    h = -np.sum(J + np.transpose(J), axis=0) + kappa

    field = np.zeros((2**len(J), 2**len(J)))
    coupling = np.zeros((2**len(J), 2**len(J)))

    for i in range(0, len(J)):
        
        field += h[0,i] * Kronecker(len(J), i+1, 'z')

    for k in range(0, len(J)):
        for l in range(k, len(J)):
            coupling += J[k][l] * np.dot(Kronecker(len(J), k+1, 'z'), Kronecker(len(J), l+1, 'z'))

    return coupling + field

def GroundState(H):

    eigenvalues, eigenvectors = np.linalg.eig(H)
    min_index = np.argmin(eigenvalues)
    ground_state = eigenvectors[:, min_index]
    return ground_state

def FuncA(t):
    tmax = 1
    return (1-t)/tmax

def FuncB(t):
    tmax = 1
    return t/tmax

def TransFieldH (H):
    t=1
    Bitflip = np.zeros((2**len(H), 2**len(H)))
    for i in range(0, len(H)):
        Bitflip += -FuncA(t)*Kronecker(len(H), i+1, 'x')

    return Bitflip + FuncB(t) * H

#time evolution operator
#



#%% Tests

M = np.array([[0,1,1,0,0],[0,0,1,0,1],[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0]]) #NP hard problem
test = np.array([[0,1,0],[0,0,1],[0,0,0]])

Up = np.array([[1], [0]]) #Column vector for spin up
Down = np.array([[0], [1]]) #Column vector for spin down
I_2 = np.array([[1,0],[0,1]]) #Identity matrix 2x2

#print(H_Ising(test))   

#print(GroundState(H_Ising(test)))
print(GroundState(H_Ising(M)))
# %%
