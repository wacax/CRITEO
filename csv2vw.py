# -*- coding: UTF-8 -*-
__author__ = 'wacax'

#code based on https://github.com/MLWave/kaggle-criteo/blob/master/csv_to_vw.py

#libraries
import os
import pandas as pd

#directories; change directories accordingly
wd = '/home/wacax/Wacax/Kaggle/criteoLabs/CRITEO Display Advertising Challenge/'
dataDir = '/home/wacax/Wacax/Kaggle/criteoLabs/Data/'

print os.getcwd()
if os.getcwd() + '/' != wd:
    os.chdir(wd)

#open 100 lines from text
#--------------------------------------------------------------------
#Load Data as Pandas Data Frame
#Read .csv Data
#2Train
train = pd.read_csv(dataDir + 'train.csv', nrows=100)
#2Test
test = pd.read_csv(dataDir + 'test.csv', nrows=100)
#check initial rows
print train.head()
print test.head()
#Data Dimensions
print train.shape
print test.shape


from datetime import datetime
from csv import DictReader

def csv_to_vw(loc_csv, loc_output, train=True):
  """
  Munges a CSV file (loc_csv) to a VW file (loc_output). Set "train"
  to False when munging a test set.
  TODO: Too slow for a daily cron job. Try optimize, Pandas or Go.
  """
  start = datetime.now()
  print("\nTurning %s into %s. Is_train_set? %s"%(loc_csv,loc_output,train))

  with open(loc_output,"wb") as outfile:
    for e, row in enumerate( DictReader(open(loc_csv)) ):

	  #Creating the features
      numerical_features = ""
      categorical_features = ""
      for k,v in row.items():
        if k not in ["Label","Id"]:
          if "I" in k: # numerical feature, example: I5
            if len(str(v)) > 0: #check for empty values
              numerical_features += " %s:%s" % (k,v)
          if "C" in k: # categorical feature, example: C2
            if len(str(v)) > 0:
              categorical_features += " %s" % v

	  #Creating the labels
      if train: #we care about labels
        if row['Label'] == "1":
          label = 1
        else:
          label = -1 #we set negative label to -1
        outfile.write( "%s '%s |i%s |c%s\n" % (label,row['Id'],numerical_features,categorical_features) )

      else: #we dont care about labels
        outfile.write( "1 '%s |i%s |c%s\n" % (row['Id'],numerical_features,categorical_features) )

	  #Reporting progress
      if e % 1000000 == 0:
        print("%s\t%s"%(e, str(datetime.now() - start)))

  print("\n %s Task execution time:\n\t%s"%(e, str(datetime.now() - start)))

#csv_to_vw("d:\\Downloads\\train\\train.csv", "c:\\click.train.vw",train=True)
#csv_to_vw("d:\\Downloads\\test\\test.csv", "d:\\click.test.vw",train=False)