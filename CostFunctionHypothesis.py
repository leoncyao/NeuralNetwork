
import random
import pylab


random.seed(120)
#
# # A neural network takes in data and spits out a guess
#
# # Basic one for learning how to make one
#
# Let's try to design a one node network that tells if a light is on or not.
# The input x is a decimal that indicates the brightness of a light bulb. The label is either 1 or 0, on or off.
# If the input is less than 0.5 assume that the bulb is off, else it is on
# There is one weight or parameter to the network w, and the product of x and w is the guess that the network makes
# Hypothesis: Given enough training, after optimization, the weight w will increase to a very high number or 1.

"""
Function to perform matrix vector multiplcation
"""


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
            # print(A[i][j])
            component += A[i][j] * v[j]
        result.append(component)
    return result
def VVA(v1, v2):
    """
    """
    result = [0 for i in range(len(v1))]
    print("v1 {} v2 {} ".format(v1, v2))
    for i in range(len(v1)):
        result[i] = v1[i] + v2[i]
    return result

def sVM(s, v):
    result = [0 for i in range(len(v))]
    for i in range(len(v)):
        result[i] = s * v[i]
    return result

def sigmoid(x):
    return 1 / (1 + pow(2.71, -x))

def s(v):
    return [sigmoid(v[i]) for i in range(len(v))]
def matrixPrinter(A):
    for i in range(len(A)):
        line = ""
        for j in range(len(A[0])):
            line += str(A[i][j])
        print(line)

def cost (a, y):
    sum = 0
    for i in range(len(a)):
        sum += pow(a[i]-y[i], 2)
    return sum

if __name__ == "__main__":
    numTrials = 6
    numTests = 10
    proportion = 1
    numOptions = 3
    randomNums = [[random.uniform(0, 1) for i in range(numOptions)] for j in range(numTests)]
    tests = []


    for setOfNums in randomNums:
        total = 0
        for k in range(numOptions):
            if str(k) in "2357":
                total += setOfNums[k]
                # print("{} is in".format(k))
            else:
                total -= setOfNums[k]
        tests.append((setOfNums, total))
    print("tests are " + str(tests))
    wL = [[random.uniform(0, 1) for i in range(numOptions)]]
    xVals = []
    CVals = []
    for m in range(numTrials):
        gradC = [0 for i in range(numOptions)]
        C = [0 for i in range(numTests)]

        for k in range(numTests):
            test_input = tests[k][0]
            test_label = tests[k][1]
            aL1 = test_input
            zL = MVM(wL, aL1) # add bias vector later
            aL = s(zL)
            y = [test_label]

            # calculate gradient vector
            print("zL is {}".format(zL))
            for j in range(len(aL)):
                dCdaL = 2 * (aL[j] - y[j])
                daLdzL = 0
                numerator = 1 * pow(2.71, -1 * zL[j])
                denominator = 1 / pow((1 + pow(2.71, -1 * zL[j])), 2)
                daLdzL = numerator / denominator
                dzLdwL = aL1[j]
                dCdwL = dCdaL * daLdzL * dzLdwL
                gradC[j] += dCdwL
            C[k] = pow(aL[0] - y[0], 2)
        xVals.append(m)
        CVaL = sum(C)/len(C)
        CVals.append(CVaL)
        avg_gradC = [-1 * gradC[i]/numTests for i in range(len(gradC))]
        # wL[0] = VVA(wL[0], sVM(proportion, avg_gradC))
        wL[0] = VVA(wL[0], avg_gradC)
    print(CVals)
    pylab.plot(xVals, CVals, 'bo', label='wL[0][0]')
    pylab.legend()
    pylab.show()
    # pylab.plot(xVals, CVals, 'r-', label='C')

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
#
#     numTrials = 10
#     numTests = 100
#     proportion = 1
#     randomNums = [random.uniform(0, 0.95) for i in range(numTests)]
#     tests = []
#     count0 = 0
#     count1 = 0
#     for num in randomNums:
#         # print("how many times did this run")
#         if num < 0.5:
#             tests.append((num, 0))
#             count0 += 1
#         else:
#             tests.append((num, 1))
#             count1 += 1
#     print("count0 is count1 is {} {}".format(count0, count1))
#     # second last layers activation
#     aL1 = []
#
#     # last layers activation
#     aL = []
#
#     # weight matrix for last layer
#     wL = [[0.5]]
#
#     xVals = []
#     yVals = []
#     CVals = []
#
#     for trialNum in range(numTrials):
#
#         C = [0 for i in range(numTests)]
#         gradC = [0]
#         for k in range(numTests):
#             test_input = tests[k][0]
#             test_label = tests[k][1]
#             # print("test is {} {}".format(test_input, test_label))
#             aL1 = [test_input]
#             zL = MVM(wL, aL1) # add bias vector later
#             aL = s(zL)
#             y = [test_label]
#
#             # later the test_label will be a vector
#             C[k] = cost(aL, y)
#
#             # for i in range(len(aL)):
#             print("zL[0] {} ".format(zL[0]))
#             # FOR THE FIRST WEIGHT
#             # Partial derivative of the cost function with respect to a weight is made up of 3 parts (by chain rule):
#
#             # the derivative of cost function with respect to the first input aL
#             dCdaL = 2 * (aL[0]-y[0])
#
#             # the derivative of aL with respect to the first input ZL
#             numerator = 1 * pow(2.71, -1 * zL[0])
#             denominator = 1 / pow((1 + pow(2.71, -1 * zL[0])), 2)
#             # print("numerator {} denominator {}".format(numerator, denominator))
#             daLdzL = numerator / denominator
#
#             # the derivative ZL with respect to the weight wL
#             dzLdwL = aL1[0]
#             print("dCdaL is {} daLdzL is {} dzLdwL is {}".format(dCdaL, daLdzL, dzLdwL))
#             dCdwL = dCdaL * daLdzL * dzLdwL
#             gradC[0] += dCdwL
#         # totalChange = sum(C) / numTests
#
#         # afterwards, average the gradient
#         # avg_gradC = [gradC[i]/numTests for i in range(len(gradC))]
#         # avgChange = totalChange / numTests
#         # wL[0] = VVA(wL[0], sVM(proportion, avg_gradC))
#         # print("C is {}".format(str(C)))
#         # need negative since its trying to find a local min of the cost function
#         toChange = -1 * proportion * gradC[0] / numTests
#         print("toChange is {} ".format(toChange))
#         wL[0][0] += toChange
#
#         xVals.append(trialNum)
#         yVals.append(wL[0][0])
#
#         # avgCost = sum(C) / numTests
#         # CVals.append(avgCost)
#
#     # print("xVals is {} yVals is {}".format(xVals,yVals))
#     pylab.plot(xVals, yVals, 'bo', label='wL[0][0]')
#     pylab.legend()
#     pylab.show()
#     # pylab.plot(xVals, CVals, 'r-', label='C')
#








