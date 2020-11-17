# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:33:55 2020

@author: Yeray
"""

import nltk
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def load_dataset(filename, cols):
    header_names=['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(filename, encoding='latin-1', sep=',',engine='python', skiprows=750000, nrows=100000, names=header_names)
    df.columns = cols
    return df

def remove_unwanted_cols(df, cols):
    for col in cols:
        del df[col]
    return df

df = load_dataset('training.1600000.processed.noemoticon.csv', ['target', 'id', 'date', 'flag', 'user', 'text'])
df = remove_unwanted_cols(df, ['id', 'date', 'flag', 'user'])

stemming = PorterStemmer()
stops = set(stopwords.words("english"))

def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X


def clean_text(raw_text):
    """Esta función funciona en una cadena de texto sin formato y:
        1) cambios en minúsculas
        2) tokeniza (se descompone en palabras)
        3) elimina la puntuación y el texto que no es de la palabra 
        4) encuentra tallos de palabras
        5) elimina las palabras de detención
        6) se une a las palabras significativas del tallo"""

    
    # Transforma a minusculas
    text = raw_text.lower()
    
    # Tokenizamos
    tokens = nltk.word_tokenize(text)
    
    # Mantiene solo palabras (elimina sigonos de puntuacion y numeros)
    # Usar .isalnum para manetener los numeros
    token_words = [w for w in tokens if w.isalpha()]
    
    # Stemming
    stemmed_words = [stemming.stem(w) for w in token_words]
    
    # Eliminamos palabras vacías
    meaningful_words = [w for w in stemmed_words if not w in stops]
    
    # Volvemos a unir las palabras en una cadena separada por espacio
    joined_words = ( " ".join(meaningful_words))
    
    # Devuelve el resultado
    return joined_words

# Devuelve el texto limpio
text_to_clean = list(df['text'])

# Clean text
cleaned_text = apply_cleaning_function_to_list(text_to_clean)

# Añade una columna al DF con el texto limpio
df['cleaned_text'] = cleaned_text

del cleaned_text

X = list(df['cleaned_text'])
y = list(df['target'])
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.3)

def create_bag_of_words(X):
    
    print ('Creating bag of words...')
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    
    # In this example features may be single words or two consecutive words
    # (as shown by ngram_range = 1,2)
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 ngram_range = (1,2), \
                                 max_features = 10000
                                ) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings. The output is a sparse array
    train_data_features = vectorizer.fit_transform(X)
    
    # Convert to a NumPy array for easy of handling
    train_data_features = train_data_features.toarray()
    
    # tfidf transform
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(train_data_features).toarray()

    # Get words in the vocabulary
    vocab = vectorizer.get_feature_names()
   
    return vectorizer, vocab, train_data_features, tfidf_features, tfidf

vectorizer, vocab, train_data_features, tfidf_features, tfidf  = \
    create_bag_of_words(X_train)

bag_dictionary = pd.DataFrame()
bag_dictionary['ngram'] = vocab
bag_dictionary['count'] = train_data_features[0]
bag_dictionary['tfidf_features'] = tfidf_features[0]

# Ordenar por count
bag_dictionary.sort_values(by=['count'], ascending=False, inplace=True)
# Mostrar top 10 
print(bag_dictionary.head(10))

def train_logistic_regression(features, label):
    print ("Training the logistic regression model...")
    from sklearn.linear_model import LogisticRegression
    ml_model = LogisticRegression(C = 100,random_state = 0)
    ml_model.fit(features, label)
    print ('Finished the logistic regression model')
    return ml_model

ml_model = train_logistic_regression(tfidf_features, y_train)

test_data_features = vectorizer.transform(X_test)
# Convertimos a numpy array
test_data_features = test_data_features.toarray()

test_data_tfidf_features = tfidf.fit_transform(test_data_features)
# Converttimos a numpy array
test_data_tfidf_features = test_data_tfidf_features.toarray()

predicted_y = ml_model.predict(test_data_tfidf_features)
correctly_identified_y = predicted_y == y_test
accuracy = np.mean(correctly_identified_y) * 100
#Para visualizar la exactitud redondeada con dos decimales
print ('Accuracy = %.0f%%' %accuracy)

#Obtenemos los valores exactos de Accuracy, Precision, Recall y F1
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, predicted_y))
print("Precision:",metrics.precision_score(y_test, predicted_y, average=None))
print("Recall:",metrics.recall_score(y_test, predicted_y, average=None))
print("F1:",metrics.f1_score(y_test, predicted_y, average=None))


