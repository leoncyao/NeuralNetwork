from Useful import *
if __name__ == "__main__":

    # constants
    random.seed(51)
    numTrials = 1000
    numTests = 100
    # numInputs = 2
    # numOutputs = 2
    # make tests
    Q = [[1, 3], [4, 2]]
    # Q = [[1, 1]]
    numInputs = len(Q[0])
    numOutputs = len(Q)
    print("Q is {}".format(Q))
    c = 2
    testSamples = createQTests(numTrials, numTests, numInputs, numOutputs, Q, c)
    # print(testSamples[0][0])
    # hold data
    xVals = []
    yVals1 = []
    yVals2 = []
    yVals3 = []

    Ws = [[[1,2],[100,4]]]
    # Ws = [[[5, 6]]]
    # Ws = [[[] for s in range(numOutputs)]]

    # matrixPrinter(Ws)
    # print(Ws)

    for k in range(numTrials):
        tests = testSamples[k]
        total_c = 0
        total_grad_c = [[[0 for r in range(numInputs)] for s in range(numOutputs)]]
        a = [[0] for t in range(numOutputs)]
        z = [[0] for o in range(numOutputs)]

        # BTW I AM JUST HARDCODING FOR LAYER NEED A LOOOP for multiple layers later
        for test in tests:
            test_input = test[0]
            test_label = test[1]
            z[0] = MVM(Ws[0], test_input)
            a[0] = R(z[0])
            for i in range(numOutputs):
                difference = a[0][i] - test_label[i]
                c = pow(difference, 2)
                total_c += c
                for j in range(numInputs):
                    dCdaL = 2 * difference
                    daLdzL = 1 if z[0][i] >= 0 else q
                    dzLdwL = test_input[j]
                    grad_c = dCdaL * daLdzL * dzLdwL
                    total_grad_c[0][i][j] += grad_c
        avg_c_1 = total_c / numTests
        avg_c_1 = round(avg_c_1, 4)
        avg_grad_c = SMM(-1/numTests, total_grad_c)
        temp_Ws = MMA(Ws,  avg_grad_c)
        total_c = 0
        for test in tests:
            test_input = test[0]
            test_label = test[1]
            for i in range(len(a[0])):
                z = MVM(temp_Ws[0], test_input)
                a[0] = R(z)
                difference = a[0][0] - test_label[0]
                c = pow(difference, 2)
                total_c += c
        avg_c_2 = total_c / numTests
        avg_c_2 = round(avg_c_2, 4)
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
        Ws = MMA(SMM(k1, Ws), SMM(k2, temp_Ws))

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



