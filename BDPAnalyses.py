import sys
import scipy
import numpy
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

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy:{}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sklearn.__version__))

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
