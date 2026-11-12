#Libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt
#%%
#Functions
def Pauli(i):
    #Defining Pauli Spin Matricies
    if i=='x' :
        return np.array([[0, 1], [1, 0]])
    elif i=='y':
        return np.array([[0, -1j], [1j, 0]])
    elif i=='z':
        return np.array([[1, 0], [0, -1]])
    
def InnerProd(bra, ket):
    bra = bra.conjugate()
    return np.inner(bra.transpose(), ket.transpose())

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
    #Defining the Ising Hamiltonian
    epsilon = 0.5
    kappa = epsilon * np.array([[1],[1],[1],[1],[1]])
    #kappa = epsilon * np.ones((len(J)))
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
    #Extracting the ground state from the hamiltonian
    eigenvalues, eigenvectors = np.linalg.eig(H)
    min_index = np.argmin(eigenvalues)
    ground_state = eigenvectors[:, min_index]
    return ground_state

def FuncA(t, tmax):
    return 1-t/tmax

def FuncB(t, tmax):
    return t/tmax

def TransFieldH(M, t, tmax):
    #Defining the Transverse-Field Hamiltonian
    H = H_Ising(M)
    Bitflip = np.zeros((len(H), len(H)))
    for i in range(0, len(H)):
        Bitflip += -FuncA(t, tmax)*Kronecker(len(M), i+1, 'x')

    return Bitflip + FuncB(t, tmax) * H

def TimeEvolution(M, q, tmax):
    #Time Evolution operator
    U = np.identity(2**len(M), complex) 
    psi_start = GroundState(TransFieldH(M, 0, tmax))
    psi_ground = GroundState(H_Ising(M))
    Prob = np.array([])
    TStep = np.array([])
    for i in range(1, q+1):
        ExpH = scipy.linalg.expm(-1j*(tmax/q)*TransFieldH(M, (i*tmax)/q, tmax))
        U = np.dot(ExpH, U)

        psi_current = np.dot(U, psi_start)
        P_current = abs(InnerProd(psi_current, psi_ground))**2
        Prob = np.append(Prob, P_current)
        TStep = np.append(TStep, (i*tmax)/q)

    return Prob, TStep


#time evolution operator



#%% Tests

M = np.array([[0,1,1,0,0],[0,0,1,0,1],[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0]]) #NP hard problem
test = np.array([[0,1,0],[0,0,1],[0,0,0]])

Up = np.array([[1], [0]]) #Column vector for spin up
Down = np.array([[0], [1]]) #Column vector for spin down
I_2 = np.array([[1,0],[0,1]]) #Identity matrix 2x2


#fig = plt.figure(figsize=(12,8))
for i in range (0,5):
    Runtime = [1, 2, 5, 10, 100]
    step = 1000
    colour = ['blue', 'orange', 'green', 'red', 'purple']
    PSuccess, Time = TimeEvolution(M, step, Runtime[i])
    plt.plot(Time/Runtime[i], PSuccess, color = colour[i])
plt.show()



# %%



