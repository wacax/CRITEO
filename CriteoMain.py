__author__ = 'wacax'

#libraries
from os import getcwd, chdir
from wabbit_wappa import *

#directories; change directories accordingly
wd = '/home/wacax/Wacax/Kaggle/criteoLabs/CRITEO Display Advertising Challenge/'
dataDir = '/home/wacax/Wacax/Kaggle/criteoLabs/Data/'

print getcwd()
if getcwd() + '/' != wd:
    chdir(wd)
#Add user defined functions
from csv2vw import csv_to_vw

#Transform the .csv files to vw format
csv_to_vw(dataDir + 'train.csv', dataDir + 'train.vw', train=True)
csv_to_vw(dataDir + 'test.csv', dataDir + 'test.vw', train=False)

#Use vowpal Wabbit for training
vw = VW(loss_function='logistic')