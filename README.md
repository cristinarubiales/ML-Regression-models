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


>Neural network: is the last model we are going to study. Deep learning is supported by many libraries, among other we find, Keras and tensorFlow. We follow the steps that are important so that we can evaluate the model : compile, fit, predict and evaluate. Moreover, we study the MSE and R2, so that we can look if the model fits our data. 

Finally, we compare the three models, and we found that SVR is the best model.


#### DATA SPECIFIC INFORMATION

Headings

1. ANÁLISIS EXPLORATORIO ("Exploratory analysis")
2. CONJUNTO DATOS: TRAIN & TEST ("Train and Test")
3. MODELOS DE REGRESION ("Regression models")
4. CONCLUSION


Units of measurement: are all numeric and they are standarized.

Missing data: we fill some variables with "0" and others with mean().        

# APRENDIZAJE AUTOMÁTICO: APRENDIZAJE SUPERVISADO


## Predicción de precios de AIRBNB (junio de 2022)

El objetivo de este proyecto es encontrar el mejor modelo supervisado que se ajuste a nuestros datos. Queremos predecir la variable Precio con nuestras diferentes variables.Es un modelo de regresión. Estamos utilizando tres modelos: Random Forest, Support Vector Machine y Neural Networks. 

Aquí estamos utilizando Python 3.6. Este lenguaje es uno de los más comunes para utilizar el aprendizaje supervisado con el fin de encontrar el modelo de regresión. Principalmente utilizamos librerías como: pandas, numpy, sklearn, tensorflow, keras, seaborn y stats. 

Este estudio es para la Universidad de CUNEF. Incluye un archivo que tiene en cuenta los alquileres de airbnb con el precio de la noche de cada uno en ellos. Los datos son recogidos por nuestro profesor. Los datos están geolocalizados principalmente en Madrid, sin embargo, también hay de Londres, Francia o Barcelona.

Palabras clave: Machine Learning, datos supervisados, Python, PCA, variables, casa, precio, precio semanal, regresor, Random forest, SVR, Neuronal Network, predicción, Accomodation, OnehotEncoding, Keras, MSE, R2, métrica, pérdida, feature_importance, GridsearchCV, validación cruzada, train, test. 

#### INFORMACIÓN METODOLÓGICA

En primer lugar, en este proyecto hemos analizado nuestras variables haciendo un análisis exploratorio. Al hacer la correlación de las variables, vimos cuáles son las que más se correlacionan con la variable principal, el Precio. Como hemos visto que hay valores atípicos, hemos hecho un PCA (Análisis de Componentes Principales) para reducir la dimensionalidad y aumentar la variabilidad.

El punto principal de este análisis son los modelos de regresión. Analizamos el Random forest, la máquina de vectores de apoyo y la red neuronal.


>Random forest: es uno de los embsambladores de bolsas que son capaces de predecir los precios y ajustarse a los conjuntos de datos de Ribnb. Sin embargo, este modelo puede sufrir de sesgo y varianza, por lo que vamos a utilizar la validación cruzada para poder utilizar diferentes estrategias para reducirlo. Además, estudiamos las variables que ayudan a que el modelo sea con una mejor puntuación y un bajo error. Por último, en este modelo, utilizamos prunning que también nos ayuda a reducir la varianza del modelo y hacer una mejor predicción.


>Máquina de vectores de apoyo: utilizamos la regresión de vectores de apoyo que genera un algoritmo que encuentra un hiperplano que optimiza la separación entre un conjunto de datos con dos clases linealmente separables. Miramos la linearSVR y también la SVR. Además, miramos qué hiperparámetros son los mejores para este modelo. Y finalmente, calculamos el RMSE y la puntuación para el entrenamiento y también para la prueba.


>Red neuronal: es el último modelo que vamos a estudiar. El aprendizaje profundo está soportado por muchas librerías, entre otras encontramos, Keras y tensorFlow. Seguimos los pasos importantes para poder evaluar el modelo: compilar, ajustar, predecir y evaluar. Además, estudiamos el MSE y el R2, para poder mirar si el modelo se ajusta a nuestros datos. 

Finalmente, comparamos los tres modelos, y encontramos que SVR es el mejor modelo.



