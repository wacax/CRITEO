__author__ = 'wacax'

#libraries
from os import getcwd, chdir, system
from numpy import exp

#Add user defined functions
from csv2vw import csv_to_vw

def sigmoid(x):
    return 1 / (1 + exp(-x))

#directories; change directories accordingly
wd = '/home/wacax/Wacax/Kaggle/criteoLabs/CRITEO Display Advertising Challenge/'
dataDir = '/home/wacax/Wacax/Kaggle/criteoLabs/Data/'

hipersearchScriptLoc = '/home/wacax/vowpal_wabbit-7.7/utl/'
vw77Dir = '/home/wacax/vowpal_wabbit-7.7/vowpalwabbit/'

print getcwd()
if getcwd() + '/' != wd:
    chdir(wd)

#Transform the .csv files to vw format
csv_to_vw(dataDir + 'train.csv', dataDir + 'train2.vw', train=True)
csv_to_vw(dataDir + 'test.csv', dataDir + 'test2.vw', train=False)

#csv_to_vw(dataDir + 'train.csv', dataDir + 'train2.vw', invalidFeatures=['Label', 'Id'],
# Label='Label', ID='Id', weights=NULL, train=True)
#csv_to_vw(dataDir + 'test.csv', dataDir + 'test2.vw', invalidFeatures=['Id'],
# Label='Label', ID='Id', weights=NULL, train=False)

# Find the l1-l2 error resulting in the lowest average loss
#vw hypersearch
# for a logistic loss train-set:
l1Value = system(hipersearchScriptLoc + 'vw-hypersearch -L 0.0000000000000001 1 vw --loss_function logistic --l1 % ' + dataDir + 'train.vw -q :: --cubic ::: -b 28')
l2Value = system(hipersearchScriptLoc + 'vw-hypersearch -L 0.0000000000000001 1 vw --loss_function logistic --l2 % ' + dataDir + 'train.vw -q :: --cubic :::-b 28')

#MODELING
#Logistic Regression with quadratic numerical features
#Training VW:
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelLogQl1.vw --loss_function logistic -q :: --cubic ::: -b 28 --l1 1.00628e-12')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelLogQl2.vw --loss_function logistic -q :: --cubic ::: -b 28 --l2 1.63753e-09')

#Testing VW:
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelLogQCl1.vw -p ' + dataDir + 'logQl1.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelLogQCl2.vw -p ' + dataDir + 'logQl2.txt')

#Neural Networks
#Training VW:
system('vw ' + dataDir + 'train.vw -f ' + dataDir + 'NN.vw --nn 100 --loss_function logistic -q ii -b 28')
#Testing VW:
system('vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'NN.vw -p ' + dataDir + 'NN100.txt')

with open(dataDir + 'PredictionIX.csv', 'wb') as outfile:
    outfile.write('Id,Predicted\n')
    for line in open(dataDir + 'logQl1.txt'):
        row = line.strip().split(" ")
        outfile.write("%s,%f\n"%(row[1], sigmoid(float(row[0]))))