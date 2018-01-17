
# coding: utf-8

# In[1]:


import sys
import scipy
import numpy as np
import matplotlib
import pandas as pd;
import sklearn
pd.options.display.precision = 3;
pd.set_option('display.height', 2000)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 2000)
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy:{}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sklearn.__version__))


# In[3]:


from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
pd.options.display.width = 180


# In[4]:


import requests
from io import BytesIO
import scipy.io as sio

urlData = 'https://github.com/dyt811/QCMetrics/raw/master/Results/BDP/Metrics/LabeledMatrix.csv'
urlLabel = 'https://raw.githubusercontent.com/CNBP/DICOMetrics/master/Results/BDPLabel1.csv'

Data = requests.get(urlData)
Label = requests.get(urlLabel)

data = pd.read_csv(BytesIO(Data.content))
label = pd.read_csv(BytesIO(Label.content), header=None, names=['Class'])

label['Class']=label['Class'].astype('category')
label['Class'].value_counts()

#1 is bad image (target)
#0 is good image (non-target)
pattern = r'^.SS'
#Epic tips from the internet: https://stackoverflow.com/questions/31551412/how-to-select-dataframe-columns-based-on-partial-matching
dataFocus = data[data.columns[data.columns.to_series().str.contains('Focus')]]
dataSNR = data[data.columns[data.columns.to_series().str.contains('SNR')]]
dataTexture = data[data.columns[data.columns.to_series().str.contains('Texture')]]
dataNSS = data[data.columns[data.columns.to_series().str.contains(pattern)]]
dataDICOM = data[data.columns[data.columns.to_series().str.contains('Dicom')]]

dataArray = [dataFocus, dataSNR, dataTexture, dataNSS, dataDICOM]

if label.size == dataFocus.shape[0]:
             dataFocusLabelled = pd.concat([dataFocus,label], axis=1)
if label.size == dataSNR.shape[0]:
             dataSNRLabelled = pd.concat([dataSNR,label], axis=1)
if label.size == dataTexture.shape[0]:
             dataTextureLabelled = pd.concat([dataTexture,label], axis=1)
if label.size == dataNSS.shape[0]:
             dataNSSLabelled = pd.concat([dataNSS,label], axis=1)
if label.size == dataDICOM.shape[0]:
             dataDICOMLabelled = pd.concat([dataDICOM,label], axis=1)

# In this step we are going to take a look at the data a few different ways:
#
# Dimensions of the dataset.
# Peek at the data itself.
# Statistical summary of all attributes.
# Breakdown of the data by the class variable.

# ### Basic Data Description Section
dataFocusLabelled
dataSNRLabelled
dataTextureLabelled
dataNSSLabelled
dataDICOMLabelled

dataLabelledArray = [dataFocusLabelled, dataSNRLabelled,dataTextureLabelled, dataNSSLabelled, dataDICOMLabelled]
dataLabelledArray
dataFocusLabelled.columns

plt.clf()

#Generate today' date:

for currentDataType in dataLabelledArray:
             currentDataType

             #ncolumns, nrows = (6, 6)
             #fig = plt.figure(ncolumns, nrows)
             #gs = gridspec.gridspec()

             plt.clf()
             plt.subplot(10,10,10)
             # Loop through ALl columns within the currenet
             for currentMetric in range(0,(len(currentDataType.columns)-1)):


                          #plt.subplot(ncolumns,nrows,currentMetric+1)

                          #Get data.
                          C1=currentDataType[currentDataType.columns[currentMetric]]
                          #Append the CLASS information.
                          C2=pd.concat([C1,label], axis=1)

                          C3 = np.vstack([C2.loc[C2.Class==0],C2.loc[C2.Class==1]]).T

                          #C2.boxplot(by='Class',figsize=(6,6))
                          C3.hist(figsize=(6,6), label=['QC Fail', 'QC Pass'], histtype='barstacked',stacked=True)
                          plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

                          #plt.xticks([1,2],['Pass','Fail'])
                          #C2.plot(kind='box',by='Class',figsize=(6,6))
                          plt.savefig(str('Histogram_'+currentDataType.columns[currentMetric])+'.png')
                          plt.show()

#DataType Loop
for currentDataType in dataLabelledArray:
             currentDataType

             #ncolumns, nrows = (6, 6)
             #fig = plt.figure(ncolumns, nrows)
             #gs = gridspec.gridspec()

             plt.clf()
             plt.subplot(10,10,10)
             # Loop through ALl columns within the currenet
             for currentMetric in range(0,(len(currentDataType.columns)-1)):


                          #plt.subplot(ncolumns,nrows,currentMetric+1)

                          #Get data.
                          C1=currentDataType[currentDataType.columns[currentMetric]]
                          #Append the CLASS information.
                          C2=pd.concat([C1,label], axis=1)

                          C2.boxplot(by='Class',figsize=(6,6))
                          plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                          plt.xticks([1,2],['Pass','Fail'])

                          #C2.plot(kind='box',by='Class',figsize=(6,6))
                          plt.savefig(str('Histogram_'+currentDataType.columns[currentMetric])+'.png')
                          plt.show()




dataTextureLabelled
grouped = dataTextureLabelled.groupby('Class')

data = [grouped.get_group(0).columns[0],grouped.get_group(1).columns[0]]
plt.figure()
plt.boxplot(data)

grouped = dataTextureLabelled.groupby('Class')
rowlength = grouped.ngroups//2
fig, axs = plt.subplots(figsize=(9,4),
                        nrows=2, ncols=rowlength,     # fix as above
                        gridspec_kw=dict(hspace=0.4))
targets = zip(grouped.groups.keys(), axs.flatten())
for i, (key, ax) in enumerate(targets):
    grouped.get_group(key).plot(kind='box')


ax.legend()
plt.show()
