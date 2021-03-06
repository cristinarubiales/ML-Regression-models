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

1. AN??LISIS EXPLORATORIO ("Exploratory analysis")
2. CONJUNTO DATOS: TRAIN & TEST ("Train and Test")
3. MODELOS DE REGRESION ("Regression models")
4. CONCLUSION


Units of measurement: are all numeric and they are standarized.

Missing data: we fill some variables with "0" and others with mean().        

# APRENDIZAJE AUTOM??TICO: APRENDIZAJE SUPERVISADO


## Predicci??n de precios de AIRBNB (junio de 2022)

El objetivo de este proyecto es encontrar el mejor modelo supervisado que se ajuste a nuestros datos. Queremos predecir la variable Precio con nuestras diferentes variables.Es un modelo de regresi??n. Estamos utilizando tres modelos: Random Forest, Support Vector Machine y Neural Networks. 

Aqu?? estamos utilizando Python 3.6. Este lenguaje es uno de los m??s comunes para utilizar el aprendizaje supervisado con el fin de encontrar el modelo de regresi??n. Principalmente utilizamos librer??as como: pandas, numpy, sklearn, tensorflow, keras, seaborn y stats. 

Este estudio es para la Universidad de CUNEF. Incluye un archivo que tiene en cuenta los alquileres de airbnb con el precio de la noche de cada uno en ellos. Los datos son recogidos por nuestro profesor. Los datos est??n geolocalizados principalmente en Madrid, sin embargo, tambi??n hay de Londres, Francia o Barcelona.

Palabras clave: Machine Learning, datos supervisados, Python, PCA, variables, casa, precio, precio semanal, regresor, Random forest, SVR, Neuronal Network, predicci??n, Accomodation, OnehotEncoding, Keras, MSE, R2, m??trica, p??rdida, feature_importance, GridsearchCV, validaci??n cruzada, train, test. 

#### INFORMACI??N METODOL??GICA

En primer lugar, en este proyecto hemos analizado nuestras variables haciendo un an??lisis exploratorio. Al hacer la correlaci??n de las variables, vimos cu??les son las que m??s se correlacionan con la variable principal, el Precio. Como hemos visto que hay valores at??picos, hemos hecho un PCA (An??lisis de Componentes Principales) para reducir la dimensionalidad y aumentar la variabilidad.

El punto principal de este an??lisis son los modelos de regresi??n. Analizamos el Random forest, la m??quina de vectores de apoyo y la red neuronal.


>Random forest: es uno de los embsambladores de bolsas que son capaces de predecir los precios y ajustarse a los conjuntos de datos de Ribnb. Sin embargo, este modelo puede sufrir de sesgo y varianza, por lo que vamos a utilizar la validaci??n cruzada para poder utilizar diferentes estrategias para reducirlo. Adem??s, estudiamos las variables que ayudan a que el modelo sea con una mejor puntuaci??n y un bajo error. Por ??ltimo, en este modelo, utilizamos prunning que tambi??n nos ayuda a reducir la varianza del modelo y hacer una mejor predicci??n.


>M??quina de vectores de apoyo: utilizamos la regresi??n de vectores de apoyo que genera un algoritmo que encuentra un hiperplano que optimiza la separaci??n entre un conjunto de datos con dos clases linealmente separables. Miramos la linearSVR y tambi??n la SVR. Adem??s, miramos qu?? hiperpar??metros son los mejores para este modelo. Y finalmente, calculamos el RMSE y la puntuaci??n para el entrenamiento y tambi??n para la prueba.


>Red neuronal: es el ??ltimo modelo que vamos a estudiar. El aprendizaje profundo est?? soportado por muchas librer??as, entre otras encontramos, Keras y tensorFlow. Seguimos los pasos importantes para poder evaluar el modelo: compilar, ajustar, predecir y evaluar. Adem??s, estudiamos el MSE y el R2, para poder mirar si el modelo se ajusta a nuestros datos. 

Finalmente, comparamos los tres modelos, y encontramos que SVR es el mejor modelo.



