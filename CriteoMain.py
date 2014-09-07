__author__ = 'wacax'

#libraries
from os import getcwd, chdir, system
from wabbit_wappa import *
#Add user defined functions
from csv2vw import csv_to_vw

#directories; change directories accordingly
wd = '/home/wacax/Wacax/Kaggle/criteoLabs/CRITEO Display Advertising Challenge/'
dataDir = '/home/wacax/Wacax/Kaggle/criteoLabs/Data/'

print getcwd()
if getcwd() + '/' != wd:
    chdir(wd)

#Transform the .csv files to vw format
csv_to_vw(dataDir + 'train.csv', dataDir + 'train.vw', train=True)
csv_to_vw(dataDir + 'test.csv', dataDir + 'test.vw', train=False)

#Use wabbit_wappa for training
#vw = VW(loss_function='logistic')
#print vw.command

#Use vowpal wabbit directly from command line for training
#changew working directory to data directory to make the operations there
chdir(dataDir)

#Training VW:
system('vw train.vw -f model.vw --loss_function logistic')

#Testing VW:
system('vw test.vw -t -i model.vw -p preds.txt')
