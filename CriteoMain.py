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

#DNS and Keys
KeypairFile = '/home/wacax/Wacax/AWSCredentials/wacax-key-pair-uswest2.pem '
DNS = 'ec2-54-69-30-200.us-west-2.compute.amazonaws.com'

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
l1Value = system(hipersearchScriptLoc + 'vw-hypersearch -L 0.0000000000000000001 1 vw -q :: --loss_function logistic'
                                        ' --l1 % ' + dataDir + 'train.vw -b 28')
l2Value = system(hipersearchScriptLoc + 'vw-hypersearch -L 0.0000000000000000001 1 vw -q :: --loss_function logistic'
                                        ' --l2 % ' + dataDir + 'train.vw -b 28')

#NN hyperparameter search:
#raw input
lValueNN = system(hipersearchScriptLoc + 'vw-hypersearch 1e-10 1 vw --loss_function logistic --oaa 10 -l % '
                  + dataDir + 'train.vw --nn 10 --inpass --adaptive --invariant --holdout_off -b 28 --passes 15 -k --compressed')

#Send Train.vw file
#zip it first to reduce sending time
system('gzip -c -9 ' + dataDir + 'train.vw > ' + dataDir + 'train.gz')
system('gzip -c -9 ' + dataDir + 'test.vw > ' + dataDir + 'test.gz')

#send train.gz to the remote instance
system('scp -i ' + KeypairFile + dataDir + 'train.gz' + ' ubuntu@' + DNS + ':')
#send vowpal wabbit source to the remote instance
system('scp -i ' + KeypairFile + dataDir + 'vowpal_wabbit-7.7.tar.gz' + ' ubuntu@' + DNS + ':')

#Connect with the remote instance
#system('ssh -X -i ' + KeypairFile + ' ubuntu@' + DNS) #you need a ssh connection so it has to be done from command line

#MODELING
#Logistic Regression with quadratic numerical features
#Training VW:
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelLogQallL1.model --loss_function logistic '
                                                              '-q :: -b 28 --l1 2.19068e-16')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelLogQallL2.model --loss_function logistic '
                                                              '-q :: -b 28 --l2 4.22829e-16')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelLogQallL1L2.model --loss_function logistic '
                                                              '-q :: -b 28 --l1 2.19068e-16 --l2 4.22829e-16')

#Testing VW:
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelLogQallCallL1.model -p ' + dataDir + 'logQl1.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelLogQallCallL2.model -p ' + dataDir + 'logQl2.txt')

#Neural Networks
#Training VW:
system(vw77Dir + 'vw --oaa 10 ' + dataDir + 'train.vw -f ' + dataDir + 'NN.vw --loss_function logistic'
                                                                       ' --adaptive --invariant --holdout_off --nn 20 -b 28 -l 0.02 --passes 15 -c')
#Testing VW:
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'NN.vw -p ' + dataDir + 'NN100.txt')

with open(dataDir + 'PredictionIX.csv', 'wb') as outfile:
    outfile.write('Id,Predicted\n')
    for line in open(dataDir + 'logQl1.txt'):
        row = line.strip().split(" ")
        outfile.write("%s,%f\n"%(row[1], sigmoid(float(row[0]))))