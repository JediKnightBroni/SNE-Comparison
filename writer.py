#for displaying the pickle files in readable form, or writing them to a text
#file
import pickle

#we'll take a file name as input.
#It will be a picle file, so we'll have to use picke.load. This will be used to
#put the file's contents into a list L.
#we'll then write L to a text file T.
fileName = input('Please enter the pickle file you want to load: ')
with open(fileName, 'rb') as f:
    L = pickle.load(f)

outFileName = input('Please enter the name of the output file, without extension: ')
outFileName = outFileName + '.txt'
with open(outFileName, 'w') as fout:
    for line in L:
        fout.write('steps: ' + str(line[0]) + ', loss: ' + str(line[1]) + '\n')

print('all done!')
