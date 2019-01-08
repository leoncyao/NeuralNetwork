'''
Building neural network skills.
I have one input layer with one input node, one output layer with one output node, and one edge connecting them.
The data will have random values x between 0 and 1, and the labels will all be a constant Q * x.
Hypothesis: the weight will eventually converge to Q given enough trials.
'''

''' 
Part 1: zero convergence
1. make test cases
2. process single case computation
    2.1. take in input from test case
    2.2. multiply it by weight
    2.3. take difference with test_label and square it
3. make for loop for processing all test cases
4. average cost over the number of test cases
5. repeat process over trials
6. show graph of cost function vs trial
    6.1. as a sanity check, i can try setting the weight to be very close to 0 and very far from 0,
         which should induce a very low cost and very high cost respectively
7. compute gradient of cost function, which in this case is the derivative of the cost function
with respect to the single edge weight
8. since I am trying to minimize the cost, add the negative of the gradient to the weight
9. ideally, the cost should become 0, as the weight will become 0, and every test case will have a cost of 0.

Part 2: one convergence
10. ok it seems to be working when all the labels are 0, lets try a data set 
where the test_input and test_label are the same. In this case the weight should approach 1.

Part 3: Q convergence
11. Make the test cases (x, Q * x) where x is taken uniformly from (0, 1)
12. w should converge to Q

'''

import random
import pylab

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

def computeChanges(tests, w):
    total_c = 0
    total_grad_c = 0
    for test in tests:
        # cozse
        #
        # \\d'dsmpute preliminaries
        teszst_input = test[0]
        test_label = test[1]
        zL = test_input * w
        aL = RELU(zL)

        # compute cost
        difference = aL - test_label
        c = pow(difference, 2)
        total_c += c

        # compute grad
        dCdaL = 2 * (aL - test_label)
        daLdzL = 1 if zL >= 0 else a
        dzLdwL = test_input
        grad_c = dCdaL * daLdzL * dzLdwL
        total_grad_c += grad_c
    return total_c, total_grad_c
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

def computeLayer(inputVector, Matrix):
    MVM(Matrix, inputVector)


if __name__ == "__main__":

    # constants
    random.seed(3)
    numTrials = 100
    numTests = 10

    # make tests
    Q = 1
    testSamples = createQTests(numTrials, numTests, Q)

    # hold data
    xVals = []
    yVals1 = []
    yVals2 = []
    yVals3 = []

    # plugging in 2 1 gets golden ratio lol in first component
    Ws = [[[2]], [[1]]]

    # plugging in 1 2 gets golden ratio in second component
    # plugging in n 1 gets nth ratio in 1 first copmonent, that is so cool
    # (for Q = 1)

    numLayers = len(Ws)

    # W0 = [[-2]]
    # W1 = [[3]]

    for k in range(numTrials):
        # get a new Sample
        tests = testSamples[k]
        total_c = 0
        total_grad_1_c = 0
        total_grad_2_c = 0
        a = [[0] for i in range(numLayers)]
        z = [[0] for i in range(numLayers)]
        for test in tests:
            test_input = test[0]
            test_label = test[1]
            # number indicates what layer, later will be a vector, and component indicates what node

            z[0] = MVM(Ws[0], [test_input])
            a[0] = R(z[0])
            for t in range(1, len(Ws)):
                W = Ws[t]
                z[t] = MVM(W, a[t-1])
                # print(z)
                a[t] = R(z[t])
                # print("a[t] became {}".format(a[t]))
            # later will be a sum
            print("check " + str(a))
            difference = a[numLayers-1][0] - test_label

            print("difference is {}".format(difference))
            # print("test_input {} test_label {} aL {} difference {}".format(test_input, test_label, aL, difference))
            c = pow(difference, 2)
            total_c += c
            # number indicates what layer, later will be a vector, and component indicates what node
            difference = a[numLayers-1][0] - test_label
            # print("test_input {} test_label {} aL {} difference {}".format(test_input, test_label, aL, difference))

            dCdaL = 2 * difference
            daLdzL = 1 if z[numLayers-1][0] >= 0 else q
            dzLdwL = test_input

            # print("dCdaL {} daLdzL {} dzLdwL".format(dCdaL, daLdzL, dzLdwL))
            grad_c = dCdaL * daLdzL * dzLdwL
            total_grad_1_c += grad_c

            dzLdaLminusone = Ws[numLayers-1][0][0]
            dCdaLminusone = dCdaL * daLdzL * dzLdaLminusone

            daLminusonedzLminusone = 1 if z[0][0] >= 0 else q
            dzLminusonedwLminusone = Ws[0][0][0]

            grad_c = dCdaLminusone * daLminusonedzLminusone*dzLminusonedwLminusone

            total_grad_2_c += grad_c

        avg_c = total_c / numTests
        print("avg_c was {}".format(avg_c))
        avg_grad_1_c = -1 * total_grad_1_c / numTests

        xVals.append(k)
        yVals1.append(avg_c)
        yVals2.append(Ws[1][0][0])
        Ws[1][0] = VVA(Ws[1][0], [avg_grad_1_c])

        avg_grad_2_c = -1 * total_grad_1_c / numTests
        yVals3.append(Ws[0][0][0])

        # lol this was wrong, but it led to the godlen ratio
        Ws[0][0] = VVA(Ws[0][0], [avg_grad_1_c])

        # for test in tests:
        #     test_input = test[0]
        #     test_label = test[1]
        #     # number indicates what layer, later will be a vector, and component indicates what node
        #     difference = a[numLayers-1][0] - test_label
        #     # print("test_input {} test_label {} aL {} difference {}".format(test_input, test_label, aL, difference))
        #
        #     dCdaL = 2 * difference
        #     daLdzL = 1 if z[numLayers-1][0] >= 0 else q
        #     dzLdaLminusone = Ws[numLayers-1][0][0]
        #
        #     # print("dCdaL {} daLdzL {} dzLdwL".format(dCdaL, daLdzL, dzLdwL))
        #     dCdaLminusone = dCdaL * daLdzL * dzLdaLminusone
        #
        #     daLminusonedzLminusone = 1 if z[0][0] >= 0 else q
        #     dzLminusonedwLminusone = Ws[0][0][0]
        #
        #     grad_c = dCdaLminusone * daLminusonedzLminusone*dzLminusonedwLminusone
        #
        #     total_grad_c += grad_c
        # avg_grad_2_c = -1 * total_grad_1_c / numTests
        #
        # yVals3.append(Ws[0][0][0])
        # Ws[0][0] = VVA(Ws[0][0], [avg_grad_c])




    # for k in range(numTrials):
    #     # get a new Sample
    #     tests = testSamples[k]
    #     total_c = 0
    #     for test in tests:
    #         test_input = test[0]
    #         test_label = test[1]
    #         # number indicates what layer, later will be a vector, and component indicates what node
    #         a = [test[0]]
    #         for W in Ws:
    #             a = MVM(W, a)
    #         difference = a[0] - test_label
    #         # print("test_input {} test_label {} aL {} difference {}".format(test_input, test_label, aL, difference))
    #         c = pow(difference, 2)
    #         total_c += c
    #     avg_c = total_c / numTests
    #
    #     xVals.append(k)
    #     yVals1.append(avg_c)

        # for k in range(numTrials):
        #     # get a new Sample
        #     tests = testSamples[k]
        #     total_c = 0
        #     for test in tests:
        #         test_input = test[0]
        #         test_label = test[1]
        #         # number indicates what layer, later will be a vector, and component indicates what node
        #         a0 = MVM(W0, [test[0]])
        #         a1 = MVM(W1, a0)
        #         difference = a1[0] - test_label
        #         # print("test_input {} test_label {} aL {} difference {}".format(test_input, test_label, aL, difference))
        #         c = pow(difference, 2)
        #         total_c += c
        #     avg_c = total_c / numTests
        #
        #     xVals.append(k)
        #     yVals1.append(avg_c)

        # iterate over every test case in tests
        # results = computeChanges(tests, A, B)
        # avg_c1 = results[0] / numTests
        # avg_grad_c = results[1] / numTests
        #
        # w2 = w1 + -1 * avg_grad_c
        #
        # results = computeChanges(tests, w2)
        # # i only need the cost and not the gradient here
        # avg_c2 = results[0] / numTests
        #
        # # both have to positive, so its not possible for one to be the negative of the other
        # if avg_c2 + avg_c1 == 0:
        #     # I am already at a local min
        #     print("found local min at " + str(w1))
        #     break
        #
        # k1 = 1 - avg_c1/(avg_c1 + avg_c2)
        # k2 = 1 - avg_c2/(avg_c1 + avg_c2)
        #
        # # record data
        # xVals.append(k)
        # yVals1.append(avg_c1)
        # yVals2.append(w1)
        #
        # # print("prop1 {} w {} prop2 {} tempw {}".format(prop1, w, prop2, tempw))
        # # print("avg_c1 {} avg_c2 {}".format(avg_c1, avg_c2))
        #
        # w1 = k1 * w1 + k2 * w2
    print(yVals2)
    print(yVals3)
    print(Ws)

    pylab.subplot(311)
    pylab.plot(xVals, yVals1, 'r--')

    pylab.subplot(312)
    pylab.plot(xVals, yVals2, 'b--')

    pylab.subplot(313)
    pylab.plot(xVals, yVals3, 'b--')

    pylab.show()
    # print("w1 is {}".format(w1))

