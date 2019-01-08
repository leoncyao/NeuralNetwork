'''

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
        return q*x

# test based design: always create tests first
def createQTests(numTrials, numTests, Q):
    testSamples = []
    for trial in range(numTrials):
        tests = []
        for i in range(numTests):
            # could have used SVM here
            data = random.uniform(0, 1) * 100
            labels = []
            for j in range(len(Q)):
                labels.append(data*Q[j])
            test = (data, labels)
            tests.append(test)
        testSamples.append(tests)
    return testSamples
def R(X):
    result = []
    for x in X:
        # print("test" + str(x))
        result.append(RELU(x))
    return result
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
    # print(A1)
    # print(A2)
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

    # constants
    random.seed(51)
    numTrials = 10000
    numTests = 10
    numOutputs = 2
    # make tests
    # Q = [13 , 3.14]
    # Q = [3.14, 13]
    Q = [random.uniform(0, 1) * 10 for v in range(numOutputs)]
    print("Q is {}".format(Q))
    testSamples = createQTests(numTrials, numTests, Q)
    # print(testSamples)

    # hold data
    xVals = []
    yVals1 = []
    yVals2 = []
    yVals3 = []

    # Ws = [[[50], [10]]]
    # Ws = [[[random.uniform(0, 10) * 10]] for m in range(numOutputs)]
    # Ws = [[[random.uniform(0, 10)] for s in range(numOutputs)]]
    # print("initial Ws is {}".format(Ws))
    W = 10
    Ws = [[[W] for s in range(numOutputs)]]

    # matrixPrinter(Ws)
    # print(Ws)

    for k in range(numTrials):
        # get a new Sample
        tests = testSamples[k]
        total_c = 0
        # total_grad_c = [[[k] for i in range(numOutputs)]]
        # total_grad_c = [[[0], [0]]]
        total_grad_c = [[[0] for s in range(numOutputs)]]
        # total_grad_c = 0
        a = [[0] for i in range(numOutputs)]
        z = [[0] for i in range(numOutputs)]
        for test in tests:
            test_input = test[0]
            test_label = test[1]
            # number indicates what layer, later will be a vector, and component indicates what node

            # zL = W * test_input
            # zL = Ws[0][0][0] * test_input
            # zL = MVM(Ws[0], [test_input])[0]
            z[0] = MVM(Ws[0], [test_input])
            # z[0] = MVM(Ws[0], [test_input])
            # a = RELU(zL)
            a[0] = R(z[0])
            # print("len a[0] is {}".format(len(a[0])))
            for i in range(len(a[0])):
                difference = a[0][i] - test_label[i]
                c = pow(difference, 2)
                total_c += c
                dCdaL = 2 * difference
                daLdzL = 1 if z[0][i] >= 0 else q
                dzLdwL = test_input
                grad_c = dCdaL * daLdzL * dzLdwL
                total_grad_c[0][i][0] += grad_c
        avg_c_1 = total_c / numTests
        # avg_c_1 = round(avg_c_1, 4)
        avg_grad_c = SMM(-1/numTests, total_grad_c)
        temp_Ws = MMA(Ws,  avg_grad_c)
        # avg_grad_c = total_grad_c[0][0][0] / numTests
        # temp_W = Ws[0][0][0] + -1 * avg_grad_c
        # temp_W = temp_Ws[0][0][0]
        total_c = 0
        for test in tests:
            test_input = test[0]
            test_label = test[1]
            # number indicates what layer, later will be a vector, and component indicates what node
            for i in range(len(a[0])):
                z = MVM(temp_Ws[0], [test_input])
                # z[0] = MVM(Ws[0], [test_input])
                # a = RELU(zL)
                a[0] = R(z)
                difference = a[0][0] - test_label[0]
                c = pow(difference, 2)
                total_c += c
                # zL = temp_W * test_input
            # a = RELU(zL)
            # difference = a - test_label[0]
            # c = pow(difference, 2)
            # total_c += c

        avg_c_2 = total_c / numTests
        # avg_c_1 = round(avg_c_1, 4)
        if avg_c_2 + avg_c_1 == 0:
            print("there were {} trials".format(k))
            # at local min
            break

        k1 = 1 - avg_c_1/(avg_c_1 + avg_c_2)
        # k2 = 1 - avg_c_2/(avg_c_1 + avg_c_2)
        k2 = 1 - k1

        xVals.append(k)
        yVals1.append(avg_c_1)
        yVals2.append(Ws[0][0][0])
        # yVals2.append(W)
        # yVals3.append(Ws[0][1][0])



        # W = k1 * W + k2 * temp_W
        # Ws[0][0][0] = k1 * Ws[0][0][0] + k2 * temp_W
        Ws = MMA(SMM(k1, Ws), SMM(k2, temp_Ws))

    # for k in range(numTrials):
    #     # we will compute the cost C1 given the current w, and the cost C2 with w + delta_w
    #     # where delta_w is the derivative of C1 with respect to w
    #     # then w will become the weighted sum
    #     # b1 * w + b2 * (w + delta_w)
    #     # This is to ensure that the function doesnt overshoot while trying to optimize
    #
    #     # get a new Sample
    #     tests = testSamples[k]
    #     total_c = 0
    #     # total_grad_c = [[[k] for i in range(numOutputs)]]
    #     # total_grad_c = [[[0], [0]]]
    #     total_grad_c = [[[0] for s in range(numOutputs)]]
    #     a = [[0]]
    #     z = [[0]]
    #     for test in tests:
    #         test_input = test[0]
    #         test_label = test[1]
    #         # number indicates what layer, later will be a vector, and component indicates what node
    #
    #         z[0] = MVM(Ws[0], [test_input])
    #         a[0] = R(z[0])
    #
    #         for i in range(len(a[0])):
    #             # print("how many times did this run {}".format(i))
    #             difference = a[0][i] - test_label[i]
    #             c = pow(difference, 2)
    #             total_c += c
    #             dCdaL = 2 * difference
    #             daLdzL = 1 if z[0][0 ] >= 0 else q
    #             dzLdwL = test_input
    #             grad_c = dCdaL * daLdzL * dzLdwL
    #             total_grad_c[0][i][0] += grad_c
    #     avg_c_1 = total_c / numTests
    #     # avg_c_1 = round(avg_c_1, 4)
    #     avg_grad_c = SMM(-1/numTests, total_grad_c)
    #     temp_Ws = MMA(Ws,  avg_grad_c)
    #
    #     total_c = 0
    #     for test in tests:
    #         test_input = test[0]
    #         test_label = test[1]
    #         # number indicates what layer, later will be a vector, and component indicates what node
    #
    #         z[0] = MVM(Ws[0], [test_input])
    #         a[0] = R(z[0])
    #
    #         for i in range(len(a[0])):
    #             difference = a[0][i] - test_label[i]
    #             c = pow(difference, 2)
    #             total_c += c
    #
    #     avg_c_2 = total_c / numTests
    #     # avg_c_1 = round(avg_c_1, 4)
    #     if avg_c_2 + avg_c_1 == 0:
    #         print("there were {} trials".format(k))
    #         # at local min
    #         break
    #
    #     k1 = 1 - avg_c_1/(avg_c_1 + avg_c_2)
    #     # k2 = 1 - avg_c_2/(avg_c_1 + avg_c_2)
    #     k2 = 1 - k1
    #
    #     xVals.append(k)
    #     yVals1.append(avg_c_1)
    #     yVals2.append(Ws[0][0][0])
    #     # yVals3.append(Ws[0][1][0])
    #
    #     Ws = MMA(SMM(k1, Ws), SMM(k2, temp_Ws))

    print(yVals2)
    print(Ws)
    # print(W)

    pylab.subplot(311)
    pylab.plot(xVals, yVals1, 'b--')

    pylab.subplot(312)
    pylab.plot(xVals, yVals2, 'r--')

    # pylab.subplot(313)
    # pylab.plot(xVals, yVals3, 'r--')

    pylab.show()



