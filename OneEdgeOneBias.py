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

a = 0.5
def RELU(x):
    if x > 0:
        return x
    else:
        return a*x

# test based design: always create tests first
def createQTests(numTrials, numTests, Q):
    testSamples = []
    for trial in range(numTrials):
        tests = []
        for i in range(numTests):
            data = random.uniform(0, 1) * 10
            test = (data, Q * data)
            tests.append(test)
        testSamples.append(tests)
    return testSamples

def computeChanges(tests, w, b):
    total_c = 0
    total_grad_1_c = 0
    total_grad_2_c = 0
    for test in tests:
        # compute preliminaries
        test_input = test[0]
        test_label = test[1]
        zL = test_input * w + b
        aL = RELU(zL)

        # compute cost
        difference = aL - test_label
        c = pow(difference, 2)
        total_c += c

        # compute grad
        dCdaL = 2 * (aL - test_label)
        daLdzL = 1 if zL >= 0 else a
        dzLdbL = 1
        grad_c = dCdaL * daLdzL * dzLdbL
        total_grad_1_c += grad_c
    return total_c, total_grad_1_c, total_grad_2_c

if __name__ == "__main__":

    # constants
    random.seed(3)
    numTrials = 1000
    numTests = 100

    # make tests
    Q = 0
    testSamples = createQTests(numTrials, numTests, Q)

    # hold data
    xVals = []
    yVals1 = []
    yVals2 = []
    yVals3 = []

    w1 = 100
    b1 = 1
    for k in range(numTrials):
        # get a new Sample
        tests = testSamples[k]

        # iterate over every test case in tests
        results = computeChanges(tests, w1, b1)
        avg_c1 = results[0] / numTests
        avg_grad_1_c = results[1] / numTests
        avg_grad_2_c = results[2] / numTests

        w2 = w1 + -1 * avg_grad_1_c
        b2 = b1 + -1 * avg_grad_2_c

        results = computeChanges(tests, w2, b2)
        # i only need the cost and not the gradient here
        avg_c2 = results[0] / numTests

        if avg_c2 + avg_c1 == 0:
            # I am already at a local min
            print("found local min at {} {}".format(w1, b1))
            break

        k1 = 1 - avg_c1/(avg_c1 + avg_c2)
        k2 = 1 - avg_c2/(avg_c1 + avg_c2)

        # record data
        xVals.append(k)
        yVals1.append(avg_c1)
        yVals2.append(w1)
        yVals3.append(b1)

        # print("prop1 {} w {} prop2 {} tempw {}".format(prop1, w, prop2, tempw))
        # print("avg_c1 {} avg_c2 {}".format(avg_c1, avg_c2))

        w1 = k1 * w1 + k2 * w2
        b1 = k1 * b1 + k2 * b2

    print(yVals1)

    pylab.subplot(311)
    pylab.plot(xVals, yVals1, 'b--')

    pylab.subplot(312)
    pylab.plot(xVals, yVals2, 'r--')

    pylab.subplot(313)
    pylab.plot(xVals, yVals3, 'r--')

    print("w1 is {}".format(w1))
    print("b1 is {}".format(b1))

    pylab.show()



