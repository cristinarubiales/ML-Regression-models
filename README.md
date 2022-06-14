# ML-Regression-models
# MACHINE LEARNING: SUPERVISED LEARNING


## AIRBNB Price prediction (June 2022)

The aim of this project is to find the best supervised model that fits our data. We want to predict the variable Price with our different variables.It is a regression model. We are using three models: Random Forest, Support Vector Machine and Neural Networks. 

Here we are using Python 3.6. This language is one of the most common ones to use supervised learning in order to find the regression model. Mainly we use libraries such: pandas, numpy, sklearn, tensorflow, keras, seaborn and stats. 

This study is for CUNEF University. It includes a file that takes into consideration airbnb rents with the night price of each on them. The data is collected by our teacher. The data is geolocated maily in Madrid, however, there is also from London, France or Barcelona.

Keywords: Machine Learning, supervised data, Python, PCA, variables, house, price, weekly price, regressor, Random forest, SVR, Neuronal Network, prediction, Accomodation, OnehotEncoding, Keras, MSE, R2, metrics, loss, feature_importance, GridsearchCV, cross validation, train, test. 

#### METHODOGICAL INFORMATION

Firstly in our this project, we have looked at our variables by making an exploratory analisis. By making the correlation of the variables, we saw which are the ones that correlate more with the main variable, Price. With this correlations, we made some drop of the variables that did not correlate with it.As we saw that there are outliers, we made a PCA (Principal Component Analisis) so that we can reduce the dimensionality and increases the variability.The next step is train-test split in order to split our dataset into two.

The main point in this analysis is the regression models. We look at Random Forest, Support Vector Machine and Neural Network.


>Random forest: is one of the bagging embsambladors that are able to predict prices and adjust to the ribnb datasets. However, this models can suffer from bias and variance, so we are going to use cross validation so that we can use different strategies to reduce it. Also, we study the variables that help the model be with a better score and a low error. Finally, in this model, we use prunning that also help us to reduce de variance of the model and make a better prediction.


>Support Vector machine: we use Supooort vector regression that generates an algorithm that finds an hyperplane that optimizes the separation between a data set with two linearly separable classes. We look at linearSVR and also SVR. Also, we look at which hiperparameters are the best for this model. And finally, we calculate RMSE and score for train and also for test.


>Neural network: is the last model we are going to study. Deep learning is wsupported by many libraries, among other we find, Keras and tensorFlow. We follow the steps that are important so that we can evaluate the model : compile, fit, predict and evaluate. Moreover, we study the MSE and R2, so that we can look if the model fits our data. 

Finally, we compare the three models, and we found that neural networks are the ones that best fits our model. With high r2 and low MSE.


#### DATA SPECIFIC INFORMATION

Headings

1. ANÁLISIS EXPLORATORIO ("Exploratory analysis")
2. CONJUNTO DATOS: TRAIN & TEST ("Train and Test")
3. MODELOS DE REGRESION ("Regression models")
4. CONCLUSION


Units of measurement: are all numeric and they are standarized.

Missing data: we fill some variables with "0" and others with mean().


This project is written in SPANISH, however, if you have any doubts, do not hesitate to write me. 
