'''
Building neural network skills.
Going to have 3 layers, one node in each layer.

'''

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
        # print("q is {} x is {}".format(q, x))
        return q*x

def R(X):
    result = []
    for x in X:
        # print("test" + str(x))
        result.append(RELU(x))
    return result
# test based design: always create tests first
def createQTests(numTrials, numTests, Q):
    testSamples = []
    for trial in range(numTrials):
        tests = []
        for i in range(numTests):
            data = random.uniform(0, 1) * 1
            test = (data, Q * data)
            tests.append(test)
        testSamples.append(tests)
    return testSamples
def VVA(v1, v2):
    result = [0 for i in range(len(v1))]
    # print("v1 {} v2 {} ".format(v1, v2))
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
    # print("A {} v {}".format(A, v))
    for i in range(len(A)):
        component = 0
        for j in range(len(A[0])):
            # print(A[i][j])
            component += A[i][j] * v[j]
        result.append(component)
    return result
def MMA(A1, A2):
    result = [[[0 for i in range(len(A1[0][0]))] for j in range(len(A1[0]))] for k in range(len(A1))]
    for v1 in range(len(A1)):
        for v2 in range(len(A1[v1])):
            for v3 in range(len(A1[v1][v2])):
                result[v1][v2][v3] = A1[v1][v2][v3] + A2[v1][v2][v3]
    return result
def SMM(s, A1):
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

# returns the scalar s times the vector v
def SVM(s, v):
    result = [0 for i in range(len(v))]
    for i in range(len(v)):
        result[i] = s * v[i]
    return result

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # constants
    random.seed(10)
    # numTrials = 1000
    # numTests = 10
    numTrials = 1000
    numTests = 10

    # make tests
    Q = 3.14
    testSamples = createQTests(numTrials, numTests, Q)
    print(testSamples)

    # hold data
    xVals = []
    yVals1 = []
    yVals2 = []
    yVals3 = []


    # Ws = [[[2]], [[4]], [[6]]]
    # Ws = [[[3]], [[3]]]
    Ws = [[[5]]]
    # Ws = [[[-0.2914820574596308]], [[-6.861485806127167]]]
    # Ws = [[[-6.861485806127167]], [[-0.2914820574596308]]]
    # Ws = [[[0.9]], [[10]], [[3]]]
    numLayers = len(Ws) + 1

    # total_grad_c = [[[0]] for s in range(len(Ws))]
    for k in range(numTrials):
        # get a new Sample
        tests = testSamples[k]
        total_c = 0
        # total_grad_1_c = 0
        # total_grad_2_c = 0
        a = [[0] for i in range(numLayers)]
        z = [[0] for i in range(numLayers)]

        total_grad_c = [[[0]] for i in range(1)]

        # need to fix this for multiple layers first

        for test in tests:
            test_input = test[0]
            test_label = test[1]
            # number indicates what layer, later will be a vector, and component indicates what node

            # z[0] = MVM(Ws[0], [test_input])
            # a[0] = R(z[0])
            a[0] = [test_input]
            for t in range(1, numLayers):
                W = Ws[t-1]
                z[t] = MVM(W, a[t-1])
                a[t] = R(z[t])
                # print("t is {}".format(t))

            # later will be a sum
            difference = a[numLayers-1][0] - test_label
            c = pow(difference, 2)
            total_c += c

            dCdaL = 2 * difference
            daLdzL = 1 if z[0][0] >= 0 else q
            dzLdwL = test_input
            grad_c = dCdaL * daLdzL * dzLdwL
            total_grad_c[0][0][0] += grad_c


        avg_c = total_c / numTests
        avg_c = round(avg_c, 4)

        xVals.append(k)
        yVals1.append(avg_c)
        yVals2.append(Ws[0][0][0])
        # yVals3.append(Ws[0][0][0])

        temp_Ws = MMA(SMM(-1 / numTests, total_grad_c), Ws)

        temp_total_c = 0
        for test in tests:
            test_input = test[0]
            test_label = test[1]
            # number indicates what layer, later will be a vector, and component indicates what node
            a = [[0] for i in range(numLayers)]
            z = [[0] for i in range(numLayers)]
            z[0] = MVM(temp_Ws[0], [test_input])
            a[0] = R(z[0])
            for t in range(1, len(temp_Ws)):
                W = temp_Ws[t]
                z[t] = MVM(W, a[t-1])
                a[t] = R(z[t])
            # later will be a sum
            difference = a[numLayers - 1][0] - test_label
            c = pow(difference, 2)
            temp_total_c += c
        temp_avg_c = temp_total_c / numTests
        temp_avg_c = round(temp_avg_c, 4)

        if avg_c == 0 and temp_avg_c == 0:
            # found local min
            break

        k1 = 1 - avg_c / (avg_c + temp_avg_c)
        k2 = 1 - temp_avg_c / (avg_c + temp_avg_c)
        Ws = MMA(SMM(k1, Ws), SMM(k2, temp_Ws))

    # print(yVals2)
    # print(yVals3)
    print(Ws)
    # print("test {} * {} = {}".format(Ws[0][0][0], Ws[1][0][0], Ws[0][0][0] * Ws[1][0][0]))
    # print("test {} * {} * {} = {}".format(Ws[0][0][0], Ws[1][0][0], Ws[2][0][0], Ws[0][0][0] * Ws[1][0][0] * Ws[2][0][0]))

    print(yVals1)

    pylab.subplot(311)
    pylab.plot(xVals, yVals1, 'r--')

    pylab.subplot(312)
    pylab.plot(xVals, yVals2, 'b--')

    # pylab.subplot(313)
    # pylab.plot(xVals, yVals3, 'b--')

    pylab.show()


