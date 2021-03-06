# -*- coding: UTF-8 -*-
__author__ = 'wacax'

#code based on https://github.com/MLWave/kaggle-criteo/blob/master/csv_to_vw.py

#libraries
from datetime import datetime
from csv import DictReader
#from pandas import read_csv

def csv_to_vw(loc_csv, loc_output, train=True):
#def csv_to_vw(loc_csv, loc_output, invalidFeatures=NULL, Label=NULL, ID=NULL, weights=NULL, train=True):

    #open 1000 lines from text
    #Load Data as Pandas Data Frame
    #Read .csv Data
    #minifile = read_csv(loc_csv, nrows=1000)

    #typeFeatures = minifile.dtypes
    #fileObjectIXs = (typeFeatures.index not in invalidFeatures) and (typeFeatures == 'object')
    #fileNumericIXs = (typeFeatures.index not in invalidFeatures) and (typeFeatures == 'float64' or type == 'int64')

    #Features have to be in the form:
    #[Label] [Importance [Tag]]|Namespace Features |Namespace Features ... |Namespace Features
  """
  Munges a CSV file (loc_csv) to a VW file (loc_output). Set "train"
  to False when munging a test set.
  TODO: Too slow for a daily cron job. Try optimize, Pandas or Go.
  """
  start = datetime.now()
  print("\nTurning %s into %s. Is_train_set? %s"  %(loc_csv, loc_output, train))

  with open(loc_output, "wb") as outfile:
    for e, row in enumerate(DictReader(open(loc_csv))):

	  #Creating the features
      numerical_features = ""
      categorical_features = ""
      for k,v in row.items():

#     k,v = row.items()
#     factorIx = fileObjectIXs and (len(str(v)) > 0)
#     categorical_features = v[factorIx]

#     numIx = fileNumericIXs and (len(str(v)) > 0)
#     categorical_features = v[numIx]

        if k not in ["Label", "Id"]:
          if "I" in k: # numerical feature, example: I5
            if len(str(v)) > 0: #check for empty values
              numerical_features += " %s:%s" % (k,v)
          if "C" in k: # categorical feature, example: C2
            if len(str(v)) > 0:
              categorical_features += " %s" % v

	  #Creating the labels
      if train: #we care about labels
#        if row[Label] == '1':
        if row['Label'] == "1":
          label = 1
        else:
          label = -1 #we set negative label to -1
        outfile.write("%s '%s |i%s |c%s\n" % (label, row['Id'], numerical_features, categorical_features))
#        outfile.write("%s '%s |i%s |c%s\n" % (label, row[ID], numerical_features, categorical_features))

      else: #we dont care about labels
        outfile.write("1 '%s |i%s |c%s\n" % (row['Id'], numerical_features, categorical_features))
#        outfile.write("1 '%s |i%s |c%s\n" % (row[ID], numerical_features, categorical_features))

	  #Reporting progress
      if e % 1000000 == 0:
        print("%s\t%s" % (e, str(datetime.now() - start)))

  print("\n %s Task execution time:\n\t%s" % (e, str(datetime.now() - start)))

def test(plot=True):
    from pandas import read_csv
    import matplotlib.pyplot as plt
    #open 100 lines from text
    #--------------------------------------------------------------------
    #Load Data as Pandas Data Frame
    #Read .csv Data
    #sample Train
    #train = read_csv(dataDir + 'train.csv', nrows=10000)
    #sample Test
    #test = read_csv(dataDir + 'test.csv', nrows=10000)

    #sample Submission
    #submission = read_csv(dataDir + 'random_submission.csv')

    if plot==True:
        #EDA
    #    train['Label'].hist()
        plt.ylabel('Amount of positive labels')
        plt.show()


