#grapher
import matplotlib.pyplot as plt
import os
#import sys
#import os.path

#we shouldn't display losses and accs in the same graph. Their scales are
#too different.

def graphTraining(data, iteration, dir):
    #generates a graph charting training progress - losses wrt steps
    #dir is the directory to which we save the graph
    plt.close()
    steps = [] 
    trainLosses = []

    for k in range(len(data)):
        steps.append(data[k][0])
        trainLosses.append(data[k][1])

    plt.plot(steps, trainLosses, 'b-')
    lossGraphName = 'training_loss_at_' + str(iteration) + '.png'
    lossGraphPath = os.path.join(dir, lossGraphName)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.savefig(lossGraphPath)

def graphTesting(data, iteration, dir):
    #generates a graph charting testing accuracy wrt training iterations
    #dir is the directory to which we save the graph
    plt.close()
    steps = []
    testLosses = []

    for k in range(len(data)):
        steps.append(data[k][0])
        testLosses.append(data[k][1])

    plt.plot(steps, testLosses, 'c-')
    lossGraphName = 'testing_loss_at_' + str(iteration) + '.png'
    lossGraphPath = os.path.join(dir, lossGraphName)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.savefig(lossGraphPath)
