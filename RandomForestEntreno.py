# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:34:20 2020

@author: Yeray
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import re

# Procedemos a leer nuestro csv, que carece de una primera fila indicando el nombre de cada columna. Para ello utilizaremos "header names"
# El csv contiene 1600000 filas. Al ser de un tamaño tan grande un ordenador de uso cotidiano no podría procesar toda esta información.
# Iniciaremos el análisis desde la fila 750000 y leeremos las 100000 siguientes   
header_names=['target', 'id', 'date', 'flag', 'user', 'text']
df = pd.read_csv('training.1600000.processed.noemoticon.csv',sep=',',engine='python', skiprows=750000, nrows=100000, names=header_names)

# comprobamos que seguimos teniendo 1600000 filas y se han borrado las columnas, quedándonos solo 2
print("The shape of our data:",df.shape,"\n")

# Pintamos el nombre de las cokumnas
print("Our column names are:",df.columns.values)
   
# Inicializamos el objeto BeautifulSoup en el primer tuit del dataframe  
example1 = BeautifulSoup(df["text"][0])

# Print the raw review and then the output of get_text(), for comparison
print (df["text"][0])
print (example1.get_text())

# Usamos expresiones regulares para encontrar y reemplazar
letters_only = re.sub("[^a-zA-Z]",           # El patrón para buscar
                      " ",                   # El patrón para reemplazarlo
                      example1.get_text() )  # El texto a buscar
# print (letters_only)

lower_case = letters_only.lower()        # Convertir a minúsculas
words = lower_case.split()               # Dividido en palabras

# Eliminar palabras vacías de "palabras"
words = [w for w in words if not w in stopwords.words("english")]
print (words)

def text_to_words( raw_text ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    # Función para convertir un tweet sin procesar en una cadena de palabras
    # La entrada es una sola cadena (un tweet sin procesar), y
    # la salida es una sola cadena (un tweet preprocesado)
    #
    # 1. Eliminamos HTML
    text_text = BeautifulSoup(raw_text).get_text() 
    #
    # 2. Eliminar caracteres que no sean letras     
    letters_only = re.sub("[^a-zA-Z]", " ", text_text) 
    #
    # 3. Convertir a minúsculas, dividir en palabras individuales
    words = letters_only.lower().split()                             
    #
    # 4. En Python, buscar un conjunto es mucho más rápido que buscar
    # una lista, así que convertimos las palabras vacías en un conjunto
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Eliminamos palabras vacías
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Vuelvemos a unir las palabras en una cadena separada por espacio,
    # y devuelve el resultado.
    return( " ".join( meaningful_words ))

clean_text = text_to_words( df["text"][0] )
# print (clean_text)   

# Devuelve el numero de tuits basado en la columna text del dataframe
num_texts = df["text"].size

# Inicializa una lista vacia de tuits
clean_df_texts = []

# Recorre cada tuit; crea un índice i que va de 0 a la longitud
# de la lista
for i in range( 0, num_texts ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    # Llama a nuestra función para cada uno y agrega el resultado a la lista de
    # tuits vacio
    clean_df_texts.append( text_to_words( df["text"][i] ) )
    
print ("Cleaning and parsing the training set tuits...\n")
clean_df_texts = []
for i in range( 0, num_texts ):
    # Escribira un mensaje por cada tuit leido
    if( (i+1)%1 == 0 ):
        #print ("Text %d of %d\n" % ( i+1, num_texts ))                                                         
        clean_df_texts.append( text_to_words( df["text"][i] ))
    
print ("Creating the bag of words...\n")

# Inicializa el objeto "CountVectorizer" ,que es una herramienta  de
# scikit-learn's bag of words.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform () realiza dos funciones: Primero, se ajusta al modelo
# y aprende el vocabulario; segundo, transforma nuestros datos de entrenamiento
# en vectores de características. La entrada para fit_transform debe ser una e
# lista de cadenas.
df_data_features = vectorizer.fit_transform(clean_df_texts)

# Los Numpy arrays son faciles de trabajar con ellos, convertimos el resultado 
# en array
df_data_features = df_data_features.toarray()

# Vemos los datos limpios
# print (train_data_features.shape)

# Echamos un vistazo a nuestro vocabulario
vocab = vectorizer.get_feature_names()
# print (vocab)

dist = np.sum(df_data_features, axis=0)

# Para cada uno, escribe la palabra de vocabulario y el número de veces que
# aparece en el conjunto de entrenamiento
for tag, count in zip(vocab, dist):
    print (count, tag)

print( "Training the random forest...")

# Inicializamos el Random Forest classifier con 10 arboles
X_train, X_test, y_train, y_test = train_test_split(df_data_features, df["target"], test_size=0.3)
forest = RandomForestClassifier(n_estimators = 10) 

# Podemos adaptar el bosque al conjunto de entrenamiento, usando bag of words 
# como características y las etiquetas de sentimiento como la variable de respuesta
#
# Esto puede tardar unos minutos en ejecutarse
forest = forest.fit( df_data_features, df["target"] )
y_pred=forest.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average=None))
print("Recall:",metrics.recall_score(y_test, y_pred, average=None))
print("F1:",metrics.f1_score(y_test, y_pred, average=None))

import matplotlib.pyplot as plt
plt.scatter(X_train[:,0], X_train[:,1])