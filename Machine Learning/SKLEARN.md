# SKLEARN

## 1. Install

```bash
conda create -n sklearn python=3.5
source activate sklearn
sudo apt-get install python3-pip
pip3 install numpy
pip3 install scipy
pip3 install matplotlib
pip3 install pandas
pip3 install -U scikit-learn # or conda: conda install scikit-learn
```



optional:

```bash
conda install ipython
conda install jupyter

jupyter notebook
```



## 2. Introduction

### 2.1 Algorithm Selection

[scikit-learn algorithm cheat-sheet](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

### 2.2 Load Datasets

#### 2.2.1 Load Packaged Dataset

[Dataset](http://scikit-learn.org/stable/datasets/index.html)

#### 2.2.2 Load CSV

If the csv only contains numerical value: 

```python
import numpy as np
dataset = np.loadtxt('./example.csv', delimiter=',')
```

Otherwise, when the csv contains the mixture of string and numerical value:

```python
import pandas as pd
data = pd.read_csv(filename, header=None)
y = data.ix[:,np.size(data,1)-1]
X = data.ix[:,np.size(data,1)-2]

# Convert string into numerical value
X = X.replace('string', 0)
```

#### 2.2.3 Normalization

[Reference](http://scikit-learn.org/stable/modules/preprocessing.html)

```python
from sklearn import preprocessing

X_scaled = preprocessing.scale(X_train)
```



### 2.3 Visualization

#### 2.3.1 Confusion Matrix

[Reference](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)

Function:



```python
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
```

Demo

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cnf = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
```



#### 2.3.2 Learning Curves

[Reference](http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)

Function:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
```

Demo

```python
digits = load_digits()
X, y = digits.data, digits.target


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()
```

#### 2.3.3 Validation Curve

Parameters:

> ==estimator==: object type that implements the "fit" and "predict" methods.
>
> ==X==: array-like, shape(n_samples, n_features)
>
> ==y==: array-like, shape(n_samples, n_features)
>
> ==param_name==: string, the name of parameter that will be varied.
>
> ==param_range==: array-like, shape(n_values)
>
> ==cv==: int, cross-validation generator.
>
> ==n_jobs==: int, ==-1== means using all processor.

Demo:

```python

```



## 3. Practice

### 3.1 Iris Dataset

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)
```

### 3.2 Boston Dataset

```python
from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)
print(model.fit(data_X[:4,:]))
print(data_y[:4])
```

### 3.3 Create Samples

```python
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=1)

plt.scatter(X,y)
plt.show()
```

 ## 4. Common Algorithm

### 4.1 Decision Tree

```python
from sklearn.datasets import load_iris
from sklearn import tree

iris = datasets.load_iris()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

print(clf.predict(iris.data))
print(iris.target)
```

### 4.2 SVM

[Reference](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

```python
from sklearn.svm import SVC

clf = SVC(gamma='auto')
clf.fit(X, y)
print(clf.predict(X))
```

### 4.3 K-NN

[Reference](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
print(knn.predict(X))
```

### 4.4 Logistic Regression

[Reference](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

```python
form sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0,
                         solver='lbfgs'
                         multi_class='multinomial').fit(X,y)
clf.predict(X[:2,:])
clf.predict_proba(X[:2,:])
clf.score(X,y)
```

### 4.5 Bayesian Ridge Regression

[Reference](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge)

```python
from sklearn.linear_model import BayesianRidge()

clf = BayesianRidge()
clf.fit(X,y)
clf.predict(X[:2,:])
```



## 5 Load Dataset from CSV

```python

```





