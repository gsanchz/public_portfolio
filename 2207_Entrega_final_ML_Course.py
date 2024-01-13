# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:54:36 2022

@author: guill
@certification_obtained: https://drive.google.com/file/d/1Id1TQ3hUSk0FQyjbLCoQjaIgi2obHDWt/view
"""

#Q1: El dataset que emplearemos en este caso, al tratarse de un ejercicio de
#naturaleza formativa, procede de kaggle. ¿Podrías indicar de qué tipo de
#plataforma estamos hablando?

#A1: Kaggle es una plataforma de acceso libre que ofrece una gran cantidad
#de recursos vinculados al Data Science y, en particular, al Machine Learning
#como, por ejemplo, datasets de utilidad para la aplicación de técnicas de
#Machine Learning, pequeños cursos de formación y foros de discusión sobre
#las materias. Cuenta con una comunidad muy activa.

#Q2: El dataset al que hacemos referencia se denomina datos_moviles.csv y
#puedes descargarlo al final de la descripción de este proyecto. ¿Qué tipo
#de archivo es este? ¿Cómo se puede cargar esta tipología de ficheros
#mediante la biblioteca estándar de Python?

#A2: Se trata de un archivo .csv, siglas de 'comma separated values', donde
#los datos se distinguen por la utilización de comas y puntos y comas. Esto
#simplifica mucho la complejidad del archivo y disminuye su peso dando como
#resultado que puedan ser leídos fácilmente por cualquier aplicación de
#hojas de cálculo. La forma de cargar los datos csv a través de la biblioteca
#estándar de Python lo explicaré con comentarios a continuación para una mejor
#comprensión:
    
    #Tenemos que tener el archivo correctamente localizado en nuestra carpeta
    #de trabajo habitual para poder invocarlo únicamente con su nombre.
    #En caso contrario, tendríamos que invocarlo a través del nombre
    #de la ruta en la cuál se encuentre. Supondremos lo primero.
    
import csv #Método de la biblioteca estándar de Python

nomb_fich='datos_moviles (2).csv'
with open(nomb_fich) as f:
    lectura=csv.reader(f,delimiter=',') #Aunque es el valor por defecto,
    #indicamos que los valores están separados por comas con la variable
    #delimiter.
    encabezado=next(lectura) #next es una función que desplaza la primera
    #fila (la del encabezado, en este caso) fuera del conjunto de filas con
    #las que sí contaremos para realizar bucles (las que contienen los datos
    #de nuestro estudio). El encabezado queda guardado en la variable 'encabezado'
    
    #De esta forma, podemos empezar a hacer bucles for sobre las filas de
    #datos del csv (guardadas en la variable lectura).

#Q3: Carga el fichero datos_moviles.csv en Python, de modo que los datos se
#guarden en un DataFrame, y escribe el código necesario para visualizar las
#primeras filas del dataset y para saber cuántos registros, así como el
#número de características que tiene.

#A3:

import pandas as pd

chrs=pd.read_csv(nomb_fich) #Método lectura csv de pandas que lo interpreta
#como DataFrame.
print(chrs.head()) #5 primeras filas del DataFrame
print(chrs.shape) #Aunque mirando en el explorador de variables podemos
#visualizarlo fácilmente.

#Q4:En primer lugar, nos centraremos en la variable etiqueta price_range o
#rango de precios. Esta variable toma el valor 0 para un coste bajo del
#móvil, 1 para coste medio, 2 para coste alto y 3 para coste muy alto
#(por ejemplo, un móvil de lujo). Determina las correlaciones existentes
#entre todas las variables, pero, en particular, céntrate en las relaciones
#con price_range. ¿Cuáles son las 5 variables que tienen mayor correlación
#con price_range?

#A4:
    
print('Las correlaciones entre variables son:\n',
      chrs.corr()) #Matriz de correlaciones, que es simétrica
#Para visualizar fácilmente las mayores correlaciones con price_range
#imprimimos la fila correspondiente a price_range
print('Las correlaciones con price_range son:\n',
      chrs.corr()['price_range'])
#Como se trata de muchos valores vamos a hacer un bucle para saber cuáles
#son los máximos en valor absoluto.
price_range_corr=chrs.corr()['price_range']
price_range_corr_abs=abs(price_range_corr)
price_range_corr_abs['price_range']=-1 #Fuerzo a que sea -1 porque una
#variable consigo misma tiene una correlación de 1, un dato que no es de
#interés.
max_ord_corr=[[],[]] #En esta lista ordenaré las correlaciones de price_range
#con el resto de features en orden de mayor a menor. En la primera lista
#guardaré los features y en la segunda los valores de sus respectivas
#correlaciones con price_range.
i=0
while i<len(price_range_corr_abs):
    if price_range_corr_abs.max()==-1: #Salimos del bucle si hemos ordenado
        i=len(price_range_corr_abs) #todos los elementos
    elif price_range_corr_abs[i]==price_range_corr_abs.max():
        max_ord_corr[0].append(encabezado[i]) #Hago uso de la variable encabezado
        #para saber a qué feature_name corresponde el valor máximo.
        max_ord_corr[1].append(price_range_corr[i]) #Lo añado con signo (- ó +)
        price_range_corr_abs[i]=-1 #Cuando ordeno un elemento cambio su valor
        i=0 #a -1 para que no estorbe en la búsqueda del próximo máximo
            #y vuelvo a recorrer el bucle desde la primera posición (i=0).
    else:
        i+=1 #Si no coincide con el máximo, continuamos el bucle
max5_corr=max_ord_corr
while len(max5_corr[0])>5: #Obtengo una lista con las 5 mayores correlaciones
    max_ord_corr[0].pop() #eliminando elementos de la cola de la lista ordenada
    max_ord_corr[1].pop() #de mayor a menor hasta que esta tenga 5 elementos
                          #en sus dos componentes (feature_names y correlaciones)
print('Las 5 variables con mayor correlación con price_range son:',
      max5_corr[0])

#Q5: Dado que price (el precio en euros de cada móvil) es una variable
#continua, más interesante para nuestra investigación que range_price,
#procede a representar gráficamente la matriz de correlaciones
#considerando las dos variables más correlacionadas con price
#(excluyendo a price_range, que sirve para etiquetar los móviles en
#función de dicho precio): ram y battery_power. Recuerda incluir en la
#matriz a la propia variable price.

#A5: Hacemos uso del módulo heatmap de la librería seaborn y corrcoef de
#la librería NumPy.

import numpy as np
import seaborn as sns

var_corr=['price','ram','battery_power']
matriz_price_corr=np.corrcoef(chrs[var_corr].values.T)
sns.set(font_scale=1.5)
price_corr_heatmap=sns.heatmap(matriz_price_corr,cbar=True,annot=True,
                               square=True,fmt='.2f',annot_kws={'size':15},
                               yticklabels=var_corr,xticklabels=var_corr)

#Q6: Procede a obtener la regresión lineal de la variable price frente a
#la variable ram. Genera la representación gráfica, determina los
#coeficientes de regresión y los de determinación. ¿Se alcanza un buen
#ajuste?

#A6: Los coeficientes de determinación nos revelan un bajo riesgo de
#sobreajuste, pero no un ajuste muy satisfactorio (alrededor del 65%)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

ram=np.array([chrs['ram']])
ram=np.transpose(ram) #Array columna para variable independiente
price=np.array(chrs['price']) #Array fila para variable dependiente
#Hacemos las particiones de los conjuntos de entrenamiento y prueba
X_train,X_test,y_train,y_test=train_test_split(ram,price,random_state=27)
lr=LinearRegression().fit(X_train,y_train)
y_test_pred=lr.predict(X_test)
#Obtenemos los coeficientes de regresión
print('Coeficiente a:',lr.coef_)
print('Coeficiente b:',lr.intercept_)
#Salida gráfica
plt.subplots()
plt.scatter(X_test,y_test,color='black')
plt.xlabel('ram')
plt.ylabel('price')
plt.plot(X_test,y_test_pred,color='blue',linewidth=3)
#Coeficientes de determinación
print('Valor del coeficiente de determinación del conjunto de entrenamiento:',round(lr.score(X_train,y_train),3))
print('Valor del coeficiente de determinación del conjunto de prueba:',round(lr.score(X_test,y_test),3))

#Q7: Si quisieras fijar el precio de un móvil con 3100 GB de memoria RAM,
#considerando el anterior ajuste lineal, ¿qué valor establecerías?

#A7: La solución a la pregunta es la solución a la ecuación: price= a*ram+b.

print('Según la anterior regresión lineal el precio ideal para un',
      'móvil de 3100 GB de RAM es:',
      round(float(lr.coef_*3100+lr.intercept_),2),'€')

#Q8: Representa gráficamente los residuos obtenidos frente a los valores
#predichos según el modelo de regresión lineal generado (ten en cuenta que
#los precios de los móviles oscilan aproximadamente entre 20 y 2000 €).

#A8:

y_train_pred=lr.predict(X_train)
plt.subplots()
plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolor='white',label='Datos de entrenamiento')
plt.scatter(y_test_pred,y_test_pred-y_test,c='limegreen',marker='s',edgecolor='white',label='Datos de prueba')
plt.xlabel('Predicción precios')
plt.ylabel('Residuos')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=chrs['price'].min(),xmax=chrs['price'].max(),color='black',lw=2)
plt.xlim([chrs['price'].min(),chrs['price'].max()])

#Q9: Céntrate a continuación en las variables ram y battery_power,
#considerando price_range como una etiqueta de clasificación. Genera una
#clasificación del conjunto mediante un kernel lineal, incorporando, si
#puedes, la función plot_decisions_regions para mejorar la salida gráfica.
#Determina también la exactitud del test. Nota: carga los datos de
#price_range mediante la instrucción
#price_range=np.array(data['price_range']), a fin de que no tengas
#problemas con la dimensión de los arrays (recuerda transponerlo
#seguidamente).

#A9:

from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#Datos de trabajo
battery_power=chrs['battery_power']
ram=chrs['ram']
X=np.vstack((ram,battery_power))
X=np.transpose(X) #Puntos en el plano
y=np.array(chrs['price_range'])
y=np.transpose(y) #Etiquetados según price_range
#Partición del conjunto
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=27)
#Una característica de los modelos SVM es que resulta muy sensible a las
#escalas pues cambios no parejos en cada una de las características
#afectarán a la distribución espacial de las instancias y, por tanto, a
#la línea SVM.
#El problema del escalado lo resolvemos a través de dos técnicas:
    #-Prescalado de características: para que los datos queden recogidos
    #  dentro de un intervalo. Utilizamos la función StandardScaler.
    #-Normalización: transforma los datos para que los nuevos datos
    #  compartan un mismo valor medio y otra característica similar.
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
#Construimos el modelo SVM con kernel lineal
lin_clf=SVC(kernel='linear',random_state=96,C=1) #Lo mismo que invocar LinearSVC
lin_clf.fit(X_train_std,y_train)
#Fabricamos la salida gráfica a partir del método plot_decision_regions
#por lo que necesitaremos arrays bidimensionales
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
#Atributos de mejora de la salida gráfica
plt.subplots()
scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
contourf_kwargs = {'alpha': 0.2}
scatter_highlight_kwargs = {'s': 120, 'label': 'Datos para el test', 'alpha': 0.7}
plot_decision_regions(X_combined_std,y_combined,lin_clf,legend=2,
                      X_highlight=X_test_std,scatter_kwargs=scatter_kwargs,
                      contourf_kwargs=contourf_kwargs,
                      scatter_highlight_kwargs=scatter_highlight_kwargs)
plt.xlabel('RAM')
plt.ylabel('Capacidad de la batería')
plt.title('SVM sobre price_range de un móvil')
predicciones = lin_clf.predict(X_test_std)
accuracy = accuracy_score(y_true = y_test, y_pred = predicciones, normalize = True)
accuracy=round(accuracy,2)
print(f"La exactitud del test SVR kernel lineal es: {100*accuracy}%")

#Q10: ¿Qué resultado obtendrías si aplicas una clasificación, en el caso
#anterior, de base radial gaussiana con gamma = 20?

#A10:

#Construimos el modelo SVC gaussiano
gauss_clf=SVC(kernel='rbf',random_state=96,gamma=20,C=1)
gauss_clf.fit(X_train_std,y_train)
#Fabricamos la salida gráfica a partir del método plot_decision_regions
#por lo que necesitaremos arrays bidimensionales
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
#Atributos de mejora de la salida gráfica
plt.subplots()
scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
contourf_kwargs = {'alpha': 0.2}
scatter_highlight_kwargs = {'s': 120, 'label': 'Datos para el test', 'alpha': 0.7}
plot_decision_regions(X_combined_std,y_combined,gauss_clf,legend=2,
                      X_highlight=X_test_std,scatter_kwargs=scatter_kwargs,
                      contourf_kwargs=contourf_kwargs,
                      scatter_highlight_kwargs=scatter_highlight_kwargs)
plt.xlabel('RAM')
plt.ylabel('Capacidad de la batería')
plt.title('SVM sobre price_range de un móvil')
predicciones = gauss_clf.predict(X_test_std)
accuracy = accuracy_score(y_true = y_test, y_pred = predicciones, normalize = True)
accuracy=round(accuracy,2)
print(f"La exactitud del test SVR kernel gaussiano es: {100*accuracy}%")

#Q11: Aplica una estrategia OvR para realizar la clasificación de los
#datos (el Foro de trabajo 2 te proporcionará pautas para ello) y
#determina de nuevo la exactitud del algoritmo.

#A11:

from sklearn.multiclass import OneVsRestClassifier

#Con constructor SVR lineal utilizado
ovr_clf=OneVsRestClassifier(lin_clf)
ovr_clf.fit(X_train_std,y_train)
predicciones = ovr_clf.predict(X_test_std)
accuracy = accuracy_score(y_true = y_test, y_pred = predicciones)
accuracy=round(accuracy,2)
print(f"La exactitud del test OVR con constructor SVR lineal es: {100*accuracy}%")

#Con constructor SVR gaussiano utilizado
ovr_clf=OneVsRestClassifier(gauss_clf)
ovr_clf.fit(X_train_std,y_train)
predicciones = ovr_clf.predict(X_test_std)
accuracy = accuracy_score(y_true = y_test, y_pred = predicciones)
accuracy=round(accuracy,2)
print(f"La exactitud del test OVR con constructor SVR gaussiano (gamma=20) es: {100*accuracy}%")

#Q12: Supón ahora que no dispones del etiquetado de datos (es decir, de la
#variable price_range). Considerando las variables ram y price trata de
#obtener los posibles agrupamientos del conjunto de todos los datos
#mediante el algoritmo de k-medias. ¿Qué número de clústeres deberías
#plantear? Obtén la solución para un número de clústeres superior en una
#unidad. Compara los dos resultados observando las correspondientes
#gráficas.

#A12: Observando la gráfica de regresión RAM-price ya hecha podemos observar
#la presencia clara de, al menos, dos clúster bien diferenciados. Podrían
#ser tres o incluso cuatro, con una eficiencia más discutible en estos dos
#casos. Para un estudio 'a ciegas' del problema deberíamos haber procedido
#construyendo el diagrama de silueta, tal y como se explicó en el ejercicio
#del examen final.

#Por tanto, tal y como pide el problema, aplicaremos el método de K-medias
#para k=2 y para k=3.

import mglearn
from sklearn.cluster import KMeans

#Preparamos los datos
ram=chrs['ram']
price=chrs['price']
X=np.vstack((ram,price))
X=np.transpose(X)
#Para k=2
kmeans2=KMeans(n_clusters=2)
kmeans2.fit(X)
#Salida gráfica mediante la librería mglearn
plt.subplots()
mglearn.discrete_scatter(X[:,0],X[:,1],kmeans2.labels_,markers='o')
mglearn.discrete_scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],[0,1],markers='^',markeredgewidth=4)
plt.xlabel('RAM')
plt.ylabel('Price')
#Para k=3
kmeans3=KMeans(n_clusters=3)
kmeans3.fit(X)
#Salida gráfica mediante la librería mglearn
plt.subplots()
mglearn.discrete_scatter(X[:,0],X[:,1],kmeans3.labels_,markers='o')
mglearn.discrete_scatter(kmeans3.cluster_centers_[:,0],kmeans3.cluster_centers_[:,1],[0,1,2],markers='^',markeredgewidth=4)
plt.xlabel('RAM')
plt.ylabel('Price')

#Observamos que para k=2 los conjuntos no están bien diferenciados y que
#para k=3 el conjunto de arriba está prácticamente bien diferenciado, pero
#el de abajo no presenta una clasificación satisfactoria debida a la
#distribución horizontal de la nube de puntos. Recordemos que uno de los
#puntos flacos del algoritmo kmedias es cuando este trabaja con conjuntos
#de puntos reunidos en formas no esféricas.

#Q13: Obtén ahora los agrupamientos mediante el método DBSCAN y, si
#resulta posible, con el método HDBSCAN. Recuerda que en el Foro de
#trabajo 3 ya has tratado sobre ambos métodos. Para el método DBSCAN
#investiga un posible valor de épsilon que proporcione un agrupamiento que
#te resulte razonable y para HDBSCAN emplea el recomendado por las
#personas que lo han desarrollado.

#A13:

from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=180,min_samples=2)
dbscan.fit(X)
plt.subplots()
mglearn.discrete_scatter(X[:,0],X[:,1],dbscan.labels_,markers="o")

from hdbscan import HDBSCAN

hscan=HDBSCAN()
hscan.fit(X)
plt.subplots()
mglearn.discrete_scatter(X[:,0],X[:,1],hscan.labels_,markers="o")

#Q14: Aplica el algoritmo de agrupamiento por aglomeración al conjunto de
#datos, considerando el número que consideres más adecuado de clústeres.

#A14: La agrupación más lógica para mí es la que se consigue con linkage
#single y dos clústeres.

    
from sklearn.cluster import AgglomerativeClustering

agg=AgglomerativeClustering(n_clusters=2,linkage='single')
agg.fit(X)
mglearn.discrete_scatter(X[:,0],X[:,1],agg.labels_)

#Q15: Considera ahora el dataset completo. Aplica el algoritmo PCA y obtén
#y representa la varianza explicada en función del número de dimensiones.
#¿Cuántas dimensiones requerirás para salvaguardar una varianza en torno
#al 95 %?

#A15:

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

pca_pipe=make_pipeline(StandardScaler(),PCA()) #El escalado es necesario pues
#a través de la instrucción print(chrs.var(axis=0)) podemos observar la
#disparidad de varianzas existente.
pca_pipe.fit(chrs)
#Guardamos el método PCA en una variable
modelo_pca=pca_pipe.named_steps['pca']
#Representación gráfica
cumsum=np.cumsum(modelo_pca.explained_variance_ratio_) #Suma acumulativa
#del porcentaje de varianza que queda por explicar según utilizamos
#componentes principales.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
ax.plot(np.arange(len(chrs.columns)) + 1,cumsum, marker = 'o') #Puntos
#con componentes x en array de 0 hasta 22, componentes y la suma acumulada
#de la varianza explicada.
plt.xlabel("Dimensiones principales")
plt.ylabel("Varianza explicada")
for x, y in zip(np.arange(len(chrs.columns)) + 1, cumsum):
    label = round(y,2) #Etiqueta de altura de cada punto
    ax.annotate(label,(x,y),textcoords="offset points",xytext=(0,10),ha='center')

#A partir de 18 componentes la varianza queda explicada en más de un 95%