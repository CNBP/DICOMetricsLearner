

# Plot all metric types broken down by the anlayses pipe.
for dataFrame in dataArray:

             print(type(dataFrame))
             print(dataFrame.describe())
             print(dataFrame.shape)



             plt.rcParams["figure.figsize"]=[60,60]
             # Univariate Plots
             dataFrame.plot(kind='box',
                          subplots=True,
                          layout = (6,6),
                          sharex = False,
                          sharey = False)

             plt.show();

             # Histograms
             dataFrame.hist(layout =(6,6))
             plt.show()





# Plot all metric types broken down by the anlayses pipe.
for dataFrameLabelled in dataLabelledArray:

             #Encode Position
             print(type(dataFrameLabelled))
             print(dataFrameLabelled.describe())
             print(dataFrameLabelled.shape)
             fig, ax = plt.subplots(figsize=(30,30))
             plt.rcParams["figure.figsize"]=[10,10]
             dataFrameLabelled.boxplot([''])
             dataFrameLabelled.boxplot(by='Class',ax=ax)
             plt.show();

# Plot all metric types broken down by the anlayses pipe.
for dataFrameLabelled in dataLabelledArray:
             print(type(dataFrameLabelled))
             print(dataFrameLabelled.describe())
             print(dataFrameLabelled.shape)


             plt.rcParams["figure.figsize"]=[50,50]
             bp = dataFrameLabelled.boxplot(by='Class',figsize=(50,50))
             plt.show();
# In[5]:




pd.options.display.mpl_style = 'default'


plt.show();


dataTextureLabelled.loc[dataTextureLabelled['Class']==1]



# Histograms
dataTextureLabelled.hist(layout =(6,6))
plt.show()

# In[6]:




# In[7]:



# In[8]:


print(dataset.groupby('class').size())


# ### Univariate Plot Section

# In[ ]:

numpy.random.randn(1000,4).shape


# In[ ]:





# ### Multivariatn Plot Section

# In[ ]:


# Scatter Plot Matrix
scatter_matrix(dataset);
plt.show();


# ### Algorithms

# In[ ]:


# 80% 20 % Split
array = dataset.values;
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)



# In[ ]:


seed = 7;
scoring = 'accuracy';


# In[ ]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[ ]:


fig = plt.figure()
fig.suptitle('Algorithms Comparion')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[ ]:


knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))
