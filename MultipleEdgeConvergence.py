'''
Building neural network skills.
I have one input layer with 5 input node, one output layer with one output node, and one edge connecting every input
node to the output node.
The data will be a tuple random values x1...x5  between 0 and 1, and the labels will be the sum x2 + x3 + x5.
Hypothesis: the weights will eventually converge to the tuple (0, 1, 1, 0, 1) given enough trials.
'''


import random
import pylab
import matplotlib

a = 0.5
def RELU(x):
    if x > 0:
        return x
    else:
        return a*x
    # return max(x, 0)

# test based design: always create tests first
def createTests(numTrials, numTests, numInputs):
    testSamples = []
    for trial in range(numTrials):
        tests = []
        for i in range(numTests):
            test_input = [0 for m in range(numInputs)]
            for j in range(numInputs):
                # test_input[j] = int(random.uniform(0, 1)*10)
                test_input[j] = random.uniform(0, 1)
            test = (test_input, test_input[1] + test_input[2] + test_input[4])
            tests.append(test)
        testSamples.append(tests)
    return testSamples

def createQTests(numTrials, numTests, numInputs, Q):
    testSamples = []
    for trial in range(numTrials):
        tests = []
        for i in range(numTests):
            test_input = [0 for m in range(numInputs)]
            for j in range(numInputs):
                # test_input[j] = int(random.uniform(0, 1) * 10)
                test_input[j] = random.uniform(0, 1) * 2
            test = (test_input, Q*sum(test_input))
            tests.append(test)
        testSamples.append(tests)
    return testSamples

# assuming v and w are vectors with the same # of components
def dot(v, w):
    sum = 0
    for i in range(len(v)):
        sum += v[i] * w[i]
    return sum

# assuming v and w are vectors with the same # of components
# returns the sum of the 2 vectors
def VVA(v1, v2):
    result = [0 for i in range(len(v1))]
    # print("v1 {} v2 {} ".format(v1, v2))
    for i in range(len(v1)):
        result[i] = v1[i] + v2[i]
    return result

# returns the scalar s times the vector v
def SVM(s, v):
    result = [0 for i in range(len(v))]
    for i in range(len(v)):
        result[i] = s * v[i]
    return result


def computeChanges(tests, W):
    total_c = 0
    total_grad_c = [0 for q in range(len(W))]
    for test in tests:
        # compute preliminaries
        test_input = test[0]
        test_label = test[1]
        zL = dot(test_input, W)
        aL = RELU(zL)

        # compute cost
        difference = aL - test_label
        c = pow(difference, 2)
        total_c += c

        # compute grad
        grad_c = [0 for q in range(numInputs)]
        for j in range(numInputs):
            dCdaL = 2 * (aL - test_label)
            daLdzL = 1 if zL >= 0 else a
            dzLdwL = test_input[j]
            grad_c[j] = dCdaL * daLdzL * dzLdwL
        total_grad_c = VVA(total_grad_c, grad_c)
    return total_c, total_grad_c


if __name__ == "__main__":

    # constants
    random.seed(3)
    numTrials = 100
    numTests = 10
    numInputs = 3

    # make tests
    # testSamples = createTests(numTrials, numTests, 5)
    Q = 3.14
    testSamples = createQTests(numTrials, numTests, numInputs, Q)


    xVals = []
    yVals1 = []

    # W = [0.1, 0, 0, 0, 0]
    W1 = [10 for i in range(numInputs)]
    # weightsOverTrials = [[0 for a in range(numTrials)] for i in range(numInputs)]
    weightsOverTrials = [[] for i in range(numInputs)]
    for k in range(numTrials):

        # get a new Sample
        tests = testSamples[k]

        # iterate over every test case in tests

        results = computeChanges(tests, W1)
        avg_c1 = results[0] / numTests
        avg_grad_c = SVM(1 / numTests, results[1])

        W2 = VVA(W1,  SVM(-1, avg_grad_c))

        # print("cost is " + str(avg_c));
        # print("avg_grad_c {}".format(avg_grad_c))

        results = computeChanges(tests, W2)
        # i only need the cost and not the gradient here
        avg_c2 = results[0] / numTests

        if avg_c2 + avg_c1 == 0:
            # I am already at a local min
            print("found local min at " + str(W1))
            break

        b1 = 1 - avg_c1/(avg_c1 + avg_c2)
        b2 = 1 - avg_c2/(avg_c1 + avg_c2)

        # recording data
        for q in range(numInputs):
            # weightsOverTrials[q][k] = W1[q]
            weightsOverTrials[q].append(W1[q])
        xVals.append(k)
        yVals1.append(avg_c1)

        W1 = VVA(SVM(b1, W1), SVM(b2, W2))

        # without averaging
        # adding the negative of the gradient to the weight should make it cost lower on the next trial
        # proportion = 0.1
        # w += -1 * avg_grad_c #* proportion
        # W = VVA(W, SVM(-1, avg_grad_c))

        # # variable to hold the sum of costs across test cases
        # total_c = 0
        #
        # # later shall be a vector
        # total_grad_c = [0 for q in range(numInputs)]
        #
        # # iterate over every test case in tests
        # for test in tests:
        #     test_input = test[0]
        #     test_label = test[1]
        #
        #     zL = dot(test_input, W)
        #     aL = RELU(zL)
        #     difference = aL - test_label
        #     # print("W {} test_input {} test_label {} zl {} aL {} difference {}".format(W, test_input, test_label, zL,  aL, difference))
        #     c = pow(difference, 2)
        #
        #     total_c += c
        #
        #     # calculate gradient for the given test case
        #     grad_c = [0 for q in range(numInputs)]
        #     for j in range(numInputs):
        #         dCdaL = 2 * (aL - test_label)
        #         daLdzL = 1 if zL >= 0 else a
        #         dzLdwL = test_input[j]
        #         grad_c[j] = dCdaL * daLdzL * dzLdwL
        #     total_grad_c = VVA(total_grad_c, grad_c)
        #
        # avg_c = total_c / numTests
        # avg_grad_c = SVM(1/numTests, total_grad_c)
        #
        # # print("cost is " + str(avg_c));
        # # print("avg_grad_c {}".format(avg_grad_c))
        #
        # # recording data
        #
        # for q in range(numInputs):
        #     weightsOverTrials[q][k] = W[q]
        #
        # xVals.append(k)
        # yVals1.append(avg_c)
        #
        # # adding the negative of the gradient to the weight should make it cost lower on the next trial
        # # proportion = 0.1
        # # w += -1 * avg_grad_c #* proportion
        # W = VVA(W, SVM(-1, avg_grad_c))

    pylab.subplot(numInputs+1, 1, 1)
    pylab.plot(xVals, yVals1, 'r--')
    for w in range(numInputs):
        # pass
        pylab.subplot(numInputs+1, 1, w+2)
        pylab.plot(xVals, weightsOverTrials[w], 'b--')
    pylab.show()
    print("W is {}".format(W1))



