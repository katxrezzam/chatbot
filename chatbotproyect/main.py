import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import json
import random
import pickle
stemmer = LancasterStemmer()

with open('contenido.json', encoding='utf-8') as archivo:
    datos = json.load(archivo)

try:
    with open('variables.pickle','rb') as archivoPickle:
        palabras,tags, entrenamiento, salida = pickle.load(archivoPickle)
except:
    palabras = []
    tags = []
    auxX = []
    auxY = []

    for contenido in datos['contenido']:
        for patron in contenido['patrones']:
            palabra = nltk.word_tokenize(patron)
            palabras.extend(palabra)
            auxX.append(palabra)
            auxY.append(contenido["tag"])

            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])

    palabras = [stemmer.stem(w.lower()) for w in palabras if w!='?']
    palabras = sorted(list(set(palabras)))

    tags = sorted(tags)

    entrenamiento = []
    salida = []
    salida_vacio = [0 for _ in range(len(tags))]


    for x, documento in enumerate(auxX):
        cubeta = []
        palabra = [stemmer.stem(w.lower()) for w in documento]
        for w in palabras:
            if w in palabra:
                cubeta.append(1)
            else:
                cubeta.append(0)

        fila_salida = salida_vacio[:]
        fila_salida[tags.index(auxY[x])] = 1
        entrenamiento.append(cubeta)
        salida.append(fila_salida)

    entrenamiento = numpy.array(entrenamiento)
    salida = numpy.array(salida)
    with open('variables.pickle','wb') as archivoPickle:
        pickle.dump((palabras, tags, entrenamiento, salida),archivoPickle)


red = tflearn.input_data( shape=[None, len(entrenamiento[0])])
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, len(salida[0]),activation="softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red)

try:
    modelo.load("moodel.tflearn")
except:
    modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=31, show_metric=True)
    modelo.save("moodel.tflearn")

##RUTINA DEL BOT
def mainBot():
    while True:
        entrada = input("Tu: ")
        cubeta = [0 for _ in range(len(palabras))]
        entrada_procesada = nltk.word_tokenize(entrada)

        entrada_procesada = [stemmer.stem(palabra.lower()) for palabra in entrada_procesada]

        for palabraI in entrada_procesada:
            for i,palabra in enumerate(palabras):
                if palabra == palabraI:
                    cubeta[i] = 1
        resultado = modelo.predict([numpy.array(cubeta)])
        resultadoIndices = numpy.argmax(resultado)
        tag = tags[resultadoIndices]

        for tagAux in datos["contenido"]:
            if tagAux["tag"] == tag:
                respuesta = tagAux["respuestas"]

        print("BOT: ",random.choice(respuesta))

mainBot()