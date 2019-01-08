
import random
import pylab
def matrixPrinter(A):
    for i in range(len(A)):
        line = ""
        for j in range(len(A[0])):
            line += str(A[i][j])
        print(line)
q = 0.5
def RELU(x):
    if x > 0:
        return x
    else:
        return q*x
def createQTests(numTrials, numTests, numInputs, numOutputs, Q, c):
    testSamples = []
    for trial in range(numTrials):
        tests = []
        for i in range(numTests):
            data = [random.uniform(0, 1) * c for b in range(numInputs)]
            labels = MVM(Q, data)
            test = (data, labels)
            tests.append(test)
        testSamples.append(tests)
    return testSamples
def R(X):
    result = []
    for x in X:
        result.append(RELU(x))
    return result
def VVA(v1, v2):
    result = [0 for i in range(len(v1))]
    for i in range(len(v1)):
        result[i] = v1[i] + v2[i]
    return result
def MVM(A, v):
    """
    >>> A = [[1,2,3],[4,5,6],[7,8,9]]
    >>> matrixPrinter(A)
    123
    456
    789
    >>> v = [1,2,3]
    >>> MVM(A, v)
    [14, 32, 50]
    >>> W = [[0.5]]
    >>> v = [0.1]
    >>> MVM(W,v)
    [0.05]
    """
    result = []
    for i in range(len(A)):
        component = 0
        for j in range(len(A[0])):
            component += A[i][j] * v[j]
        result.append(component)
    return result
def MMA(A1, A2):
    # for 3 dim matrices only
    result = [[[0 for i in range(len(A1[0][0]))] for j in range(len(A1[0]))] for k in range(len(A1))]
    for v1 in range(len(A1)):
        for v2 in range(len(A1[v1])):
            for v3 in range(len(A1[v1][v2])):
                result[v1][v2][v3] = A1[v1][v2][v3] + A2[v1][v2][v3]
    return result
def SMM(s, A1):
    # for 3 dim matrices only
    result = [[[0 for i in range(len(A1[0][0]))] for j in range(len(A1[0]))] for k in range(len(A1))]
    for v1 in range(len(A1)):
        for v2 in range(len(A1[v1])):
            for v3 in range(len(A1[v1][v2])):
                result[v1][v2][v3] = s * A1[v1][v2][v3]
    return result
def copy3x3Matrix(A):
    """
    >>> A = [[[1]],[[2]],[[3]]]
    >>> B = copy3x3Matrix(A)
    >>> print(B)
    [[[1]], [[2]], [[3]]]
    >>> A[0][0][0] = -10
    >>> print(A)
    [[[-10]], [[2]], [[3]]]
    >>> print(B)
    [[[1]], [[2]], [[3]]]
    """
    B = []
    for matrix in A:
        newMatrix = []
        for row in matrix:
            newRow = []
            for item in row:
                newRow.append(item)
        newMatrix.append(newRow)
        B.append(newMatrix)
    return B
def SVM(s, v):
    # returns the scalar s times the vector v
    result = [0 for i in range(len(v))]
    for i in range(len(v)):
        result[i] = s * v[i]
    return result
