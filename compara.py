# Trabalho de COS738 - 2022/1 - Busca e Mineração de Texto
# Eduardo do Valle e Herbert Salazar

# Bibliotecas necessárias para execução do código
from string import punctuation
from scipy.spatial import distance
from nltk.stem import RSLPStemmer
import nltk 
import numpy as np
import logging
import time
import csv
import math
import sys

# Arquivos de Leitura das diretrizes dos grupos de cursos e descrição das ocupações
arquivoCursos = "dados\lista_grupos_cursos.csv"
arquivoOcupacoes = "dados\dadosOcupacoes.csv"

# Parte para aumentar o limite da leitura do csv
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# Inicializa logging
logging.basicConfig(filename="log/compara.log",format='%(asctime)s %(message)s',filemode='w', encoding="utf-8")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Função para stemming de palavras em português 
def Stemming(sentence):
    stemmer = RSLPStemmer()
    phrase = []
    for word in sentence:
        phrase.append(stemmer.stem(word.lower()))
    return phrase

# Função para remover pontuação, dígitos e stopwords do português 
punctuation = list(punctuation)
def RemoveRuido(sentence):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    phrase = []
    for word in sentence:
        contains_digit = any(map(str.isdigit, word))
        if word not in stopwords and word not in punctuation:
            if len(word) > 2 and not contains_digit:
                phrase.append(word)
    return phrase

# Função para transformar o texto em letras minusculas e separar as palavras contidas nele 
def Tokenize(sentence):
    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence)
    return sentence

logger.info("Geração da lista invertida de palavras das diretrizes dos grupos de cursos")
start = time.time()

# Inicializa a lista invertida de palavras das diretrizes dos grupos de cursos
dicionarioListaInvertida = {}
palavrasDocumento = []
i=0

# Gera a lista invertida a partir da leitura das diretrizes dos grupos de cursos
with open(arquivoCursos, encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader)
    for row in csv_reader:
        texto = row[1]
        palavras = Tokenize(texto)
        palavrasSemRuido = RemoveRuido(palavras)
        palavrasStem = Stemming(palavrasSemRuido)
        palavrasDocumento.append(len(palavrasStem))
        for word in palavrasStem: 
            if word in dicionarioListaInvertida:
                dicionarioListaInvertida[word].append(i)
            else:
                dicionarioListaInvertida[word] = [i]
        i+=1

end = time.time()
logger.info("Fim da geração da lista invertida")
logger.info("Executou o procedimento anterior em: %f segundos", end-start)
logger.info("Leu %d grupos de cursos", i)

logger.info("Criação da matriz dos vetores tf-idf das palavras contidas nas descrições dos grupos de cursos")
start = time.time()

# Inicializa os valores iniciais para a criação da matriz dos vetores tf-idf
totalDocumentos = i
palavrasLista = []
matrixModelo = []
k = 0

# Criação da matriz dos vetores tf-idf das palavras contidas nas descrições dos grupos de cursos a partir da lista invertida
for key in dicionarioListaInvertida:
    palavrasLista.append(key)
    numeroDocumentosDistintos = 0
    aux = [0] * totalDocumentos
    elementosUnicos = []
    matrixModelo.append(aux) 
    for presenca in dicionarioListaInvertida[key]:
        matrixModelo[k][int(presenca)] += 1
        if presenca not in elementosUnicos:
            numeroDocumentosDistintos +=1
            elementosUnicos.append(presenca)

    for peso in elementosUnicos:
        matrixModelo[k][int(peso)] = (matrixModelo[k][int(peso)]/palavrasDocumento[int(peso)])*math.log(totalDocumentos/len(elementosUnicos))
    k+=1

# transforma a matriz em um np.array para facilitar as contas
matrizVetoresDoc = np.array(matrixModelo)
matrizVetoresDoc = matrizVetoresDoc.T

end = time.time()
logger.info("Fim da geração da matriz dos vetores tf-idf")
logger.info("Executou o procedimento anterior em: %f segundos", end-start)

logger.info("Compara os vetores tf-idf das diretrizes dos grupos de cursos com as descrições das ocupações")
start = time.time()

# Compara os vetores tf-idf das diretrizes dos grupos de cursos com as descrições das ocupações
data = []
with open(arquivoOcupacoes, encoding="utf-8") as csv_file2:
    csv_reader = csv.reader(csv_file2, delimiter=';')
    next(csv_reader)
    k = 0
    for row in csv_reader:
        # Seleciona o texto contido no arquivo "arquivoOcupacoes" para criar o vetor tf-idf
        texto = row[2] + " " + row[3] + " " + row[4] + " " + row[5] + " " + row[6] + " " + row[7]
        palavras = Tokenize(texto)
        palavrasSemRuido = RemoveRuido(palavras)
        palavrasStem = Stemming(palavrasSemRuido)
        auxVetor = np.zeros(len(matrizVetoresDoc[0]))
        for word in palavrasStem:
            if word in palavrasLista:
                auxVetor[palavrasLista.index(word)] = 1
        matrizResultado = []
        tuplasConsulta = []
        # Calcula os cossenos do vetor de ocupação aos vetores dos grupos de cursos
        for x in range(len(matrizVetoresDoc)):
            distancia = 1 - distance.cosine(auxVetor, matrizVetoresDoc[x])
            matrizResultado.append([distancia,x])
        matrizResultado.sort()
        # Geração das tuplas para melhor representação dos resultados
        for z in range(len(matrizResultado)):
            auxTuple = (len(matrizResultado)-z,matrizResultado[z][1],matrizResultado[z][0])
            tuplasConsulta.append(auxTuple)
        daux = [k, tuplasConsulta]
        data.append(daux)
        k+=1

# Escrita dos resultados
arquivoResultado = "Resultados\distanciaVetores.csv"
with open(arquivoResultado, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerows(data)

end = time.time()
logger.info("Fim da comparação")
logger.info("Executou o procedimento anterior em: %f segundos", end-start)