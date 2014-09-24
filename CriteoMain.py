__author__ = 'wacax'

#libraries
from os import getcwd, chdir, system
from numpy import exp
from pandas import read_csv
from numpy import vstack

#Add user defined functions
from csv2vw import csv_to_vw
from rankEnsemble import kaggle_rank_avg

def sigmoid(x):
    return 1 / (1 + exp(-x))

#directories; change directories accordingly
wd = '/home/wacax/Wacax/Kaggle/criteoLabs/CRITEO Display Advertising Challenge/'
dataDir = '/home/wacax/Wacax/Kaggle/criteoLabs/Data/'
ensembleDir = '/home/wacax/Wacax/Kaggle/criteoLabs/Data/EnsembleData/'

hipersearchScriptLoc = '/home/wacax/vowpal_wabbit-7.7/utl/'
vw77Dir = '/home/wacax/vowpal_wabbit-7.7/vowpalwabbit/'

print getcwd()
if getcwd() + '/' != wd:
    chdir(wd)

#DNS and Keys
KeypairFile = '/home/wacax/Wacax/AWSCredentials/wacax-key-pair-uswest2.pem '
DNS = 'ec2-54-69-15-255.us-west-2.compute.amazonaws.com'

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
l1ValueLog = system(hipersearchScriptLoc + 'vw-hypersearch -L 1e-20 1 vw -q ii --ngram 2 --loss_function logistic'
                                        ' --l1 % ' + dataDir + 'train.vw -b 28')
l2ValueLog = system(hipersearchScriptLoc + 'vw-hypersearch -L 1e-20 1 vw -q ii --ngram 2 --loss_function logistic'
                                        ' --l2 % ' + dataDir + 'train.vw -b 28')

l1ValueHin = system(hipersearchScriptLoc + 'vw-hypersearch -L 1e-20 1 vw -q ii --ngram 2 --loss_function hinge'
                                        ' --l1 % ' + dataDir + 'train.vw -b 28')
l2ValueHin = system(hipersearchScriptLoc + 'vw-hypersearch -L 1e-20 1 vw -q ii --ngram 2 --loss_function hinge'
                                        ' --l2 % ' + dataDir + 'train.vw -b 28')

l1ValueSq = system(hipersearchScriptLoc + 'vw-hypersearch 1e-20 1 vw -q ii --ngram 2 --loss_function squared'
                                        ' --l1 % ' + dataDir + 'train.vw -b 28')
l2ValueSq = system(hipersearchScriptLoc + 'vw-hypersearch 1e-20 1 vw -q ii --ngram 2 --loss_function squared'
                                        ' --l2 % ' + dataDir + 'train.vw -b 28')

l1ValueNN = system(hipersearchScriptLoc + 'vw-hypersearch -L 1e-20 1 vw -q ii --ngram 2 --loss_function logistic'
                                        ' --l1 % ' + dataDir + 'train.vw -b 28 --adaptive --invariant')
l2ValueNN = system(hipersearchScriptLoc + 'vw-hypersearch -L 1e-20 1 vw -q ii --ngram 2 --loss_function logistic'
                                        ' --l2 % ' + dataDir + 'train.vw -b 28 --adaptive --invariant')
l2ValueNN = system(hipersearchScriptLoc + 'vw-hypersearch -L 1e-20 1 vw -q ii --ngram 2 --loss_function logistic'
                                        ' -l % ' + dataDir + 'train.vw -b 28 --adaptive --invariant')

#NN hyperparameter search:
#raw input
lValueNN = system(hipersearchScriptLoc + 'vw-hypersearch 1e-20 1 vw --loss_function logistic -l % '
                  + dataDir + 'train.vw --nn 10 --inpass -b 28')

#Send Train.vw file
#zip it first to reduce sending time
system('gzip -c -9 ' + dataDir + 'train.vw > ' + dataDir + 'train.gz')
system('gzip -c -9 ' + dataDir + 'test.vw > ' + dataDir + 'test.gz')

#send train.gz to the remote instance
system('scp -i ' + KeypairFile + dataDir + 'train.gz' + ' ubuntu@' + DNS + ':')
#send vowpal wabbit source to the remote instance
system('scp -i ' + KeypairFile + dataDir + 'vowpal_wabbit-7.7.tar.gz' + ' ubuntu@' + DNS + ':')

#MODELING
#Training VW:
#Logistic Regression with quadratic numerical features
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelLogQallNgram2.model --loss_function logistic '
                                                              '-q ii --ngram 2 -b 28')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelLogQallNgram2L1.model --loss_function logistic '
                                                              '-q ii --ngram 2 -b 28 --l1 7.97443e-20')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelLogQallNgram2L2.model --loss_function logistic '
                                                              '-q ii --ngram 2 -b 28 --l2 1.51475e-14')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelLogQallNgram2L1L2.model --loss_function logistic '
                                                              '-q ii --ngram 2 -b 28 --l1 7.97443e-20 --l2 1.51475e-14')

#Hinge Regression with quadratic numerical features
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelHinQallNgram2.model --loss_function hinge '
                                                              '-q ii --ngram 2 -b 28')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelHinQallNgram2L1.model --loss_function hinge '
                                                              '-q ii --ngram 2 -b 28 --l1 4.69724e-19')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelHinQallNgram2L2.model --loss_function hinge '
                                                              '-q ii --ngram 2 -b 28 --l2 1.51475e-14')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelHinQallNgram2L1L2.model --loss_function hinge '
                                                              '-q ii --ngram 2 -b 28 --l1 2.14195e-15 --l2 1.51475e-14')

#Squared Regression with quadratic numerical features
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelSqQallNgram2.model --loss_function squared '
                                                              '-q ii --ngram 2 -b 28')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelSqQallNgram2L1.model --loss_function squared '
                                                              '-q ii --ngram 2 -b 28 --l1 2.14195e-15')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelSqQallNgram2L2.model --loss_function squared '
                                                              '-q ii --ngram 2 -b 28 --l2 6.75593e-09')
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'modelSqQallNgram2L1L2.model --loss_function squared '
                                                              '-q ii --ngram 2 -b 28 --l1 2.14195e-15 --l2 6.75593e-09')

#Neural Networks
#Training VW:
system(vw77Dir + 'vw ' + dataDir + 'train.vw -f ' + dataDir + 'NN.model --loss_function logistic'
                                                              ' --nn 8 --inpass -b 28 -q ii --ngram 2 --adaptive --invariant')

#Testing VW:
#LogLoss
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelLogQallNgram2.model -p ' + dataDir + 'LogQallNgram2.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelLogQallNgram2L1.model -p ' + dataDir + 'LogQallNgram2L1.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelLogQallNgram2L2.model -p ' + dataDir + 'LogQallNgram2L2.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelLogQallNgram2L1L2.model -p ' + dataDir + 'LogQallNgram2L1L2.txt')

#Hinge
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelHinQallNgram2.model -p ' + dataDir + 'HinQallNgram2.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelHinQallNgram2L1.model -p ' + dataDir + 'HinQallNgram2L1.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelHinQallNgram2L2.model -p ' + dataDir + 'HinQallNgram2L2.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelHinQallNgram2L1L2.model -p ' + dataDir + 'HinQallNgram2L1L2.txt')

#Squared
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelSqQallNgram2.model -p ' + dataDir + 'SqQallNgram2.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelSqQallNgram2L1.model -p ' + dataDir + 'SqQQallNgram2L1.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelSqQallNgram2L2.model -p ' + dataDir + 'SqQallNgram2L2.txt')
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'modelSqQallNgram2L1L2.model -p ' + dataDir + 'SqQallNgram2L1L2.txt')

#Neural Networks
system(vw77Dir + 'vw ' + dataDir + 'test.vw -t -i ' + dataDir + 'NN.vw -p ' + dataDir + 'NN100.txt')

#Make Kaggle .csv
submissionTemplate = read_csv(dataDir + 'random_submission.csv', index_col=False)

#Logistic
vwTextOutputLog = read_csv(dataDir + 'LogQallNgram2.txt', sep=' ', header=None)
submissionTemplate['Predicted'] = sigmoid(vwTextOutputLog.ix[:, 0])
submissionTemplate.to_csv(ensembleDir + 'PredictionX.csv', index=False)
#Hinge
vwTextOutputHin = read_csv(dataDir + 'HinQallNgram2.txt', sep=' ', header=None)
HingeOutput = (vwTextOutputHin.ix[:, 0]).as_matrix()
HingeOutputSTD = (HingeOutput - HingeOutput.min(axis=0)) / (HingeOutput.max(axis=0) - HingeOutput.min(axis=0))
Hingescaled = HingeOutputSTD / (1. - 0.) + 0.
submissionTemplate['Predicted'] = Hingescaled
submissionTemplate.to_csv(ensembleDir + 'PredictionXV.csv', index=False)
#Squared
vwTextOutputSq = read_csv(dataDir + 'SqQallNgram2.txt', sep=' ', header=None)
SqOutput = (vwTextOutputSq.ix[:, 0]).as_matrix()
SqOutputSTD = (SqOutput - SqOutput.min(axis=0)) / (SqOutput.max(axis=0) - SqOutput.min(axis=0))
Sqscaled = SqOutputSTD / (1. - 0.) + 0.
submissionTemplate['Predicted'] = Sqscaled
submissionTemplate.to_csv(ensembleDir + 'PredictionXX.csv', index=False)

#Simple Ensemble (Predictions average)
ensembleAvg = vstack((sigmoid(vwTextOutputLog.ix[:, 0]).as_matrix(), Hingescaled, Sqscaled)).T
submissionTemplate['Predicted'] = ensembleAvg.mean(axis=1)
submissionTemplate.to_csv(ensembleDir + 'PredictionXXV.csv', index=False)

#Simple Ensemble
vwTextOutputLog1 = read_csv(dataDir + 'LogQallNgram2.txt', sep=' ', header=None)
vwTextOutputLog2 = read_csv(dataDir + 'LogQallNgram2L1.txt', sep=' ', header=None)
vwTextOutputLog3 = read_csv(dataDir + 'LogQallNgram2L2.txt', sep=' ', header=None)
vwTextOutputLog4 = read_csv(dataDir + 'LogQallNgram2L1L2.txt', sep=' ', header=None)

ensembleAvgLog = vstack((sigmoid(vwTextOutputLog1.ix[:, 0]).as_matrix(),
                         sigmoid(vwTextOutputLog2.ix[:, 0]).as_matrix(),
                         sigmoid(vwTextOutputLog3.ix[:, 0]).as_matrix(),
                         sigmoid(vwTextOutputLog4.ix[:, 0]).as_matrix(),
                         Hingescaled, Sqscaled)).T
submissionTemplate['Predicted'] = ensembleAvgLog.mean(axis=1)
submissionTemplate.to_csv(dataDir + 'SimpleEnsembleLogHinSq.csv', index=False)

#Ranked Ensemble (Ranked Average)
kaggle_rank_avg(ensembleDir + '*.csv', dataDir + 'RankEnsembleFull.csv')
submissionRankTemplate = read_csv(dataDir + 'RankEnsemble.csv', index_col=False)
submissionTemplate['Predicted'] = ensembleAvg.mean(axis=1)

