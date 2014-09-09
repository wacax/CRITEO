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

print getcwd()
if getcwd() + '/' != wd:
    chdir(wd)

#Transform the .csv files to vw format
csv_to_vw(dataDir + 'train.csv', dataDir + 'train.vw', train=True)
csv_to_vw(dataDir + 'test.csv', dataDir + 'test.vw', train=False)

#Use vowpal wabbit directly from command line for training
#vw hypersearch
# for a logistic loss train-set:
#system('vw-hypersearch 1e-10 1 vw --l1 % ' + dataDir + 'train.vw')

# Find the learning-rate resulting in the lowest average loss
# for a logistic loss train-set:
#system('vw-hypersearch 0.1 100 vw --loss_function logistic --learning_rate % train.dat')

#MODELING
#Logistic Regression with l1 regularization
#Training VW:
system('vw ' + dataDir + 'train.vw -f ' + dataDir + 'modell1.vw --loss_function logistic --passes 20 -q ii -b 28')
#Testing VW:
system('vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modell1.vw -p ' + dataDir + 'logRegl1.txt')

#Neural Networks
#Training VW:
system('vw ' + dataDir + 'train.vw -f ' + dataDir + 'NN.vw --nn 100 --passes 20 -q ii -b 28')
#Testing VW:
system('vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'NN.vw -p ' + dataDir + 'NN100.txt')

with open(dataDir + 'PredictionIII.csv', 'wb') as outfile:
    outfile.write('Id,Predicted\n')
#    for line in open(dataDir + 'logRegl1.txt'):
    for line in open(dataDir + 'NN100.txt'):
        row = line.strip().split(" ")
        outfile.write("%s,%f\n"%(row[1], sigmoid(float(row[0]))))