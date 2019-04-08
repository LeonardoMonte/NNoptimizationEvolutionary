import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.decomposition import PCA
import glob
from sklearn.metrics import pairwise_distances_argmin_min
import imutils
from sklearn.metrics import accuracy_score
from operator import itemgetter


def WCSSgenetic(x, population):

    arrayint = []
    for a in x:
        arrayint.append(int(a))

    print(arrayint)
    soma = 0
    for b in arrayint:
        labels, distances = pairwise_distances_argmin_min(population[b],population, metric='minkowski')
        for x in distances:
            soma += x**2
    return soma


def generatepopulation(X,numberpopu, K, rng):

    population = []

    for x in range(numberpopu):
        first = rng.permutation(X.shape[0])[:K]
        print(first)
        population.append(np.concatenate(X[first]))

    return population


def find_clusters(X, n_clusters, rng, max_it):

    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    max_iterator = 0
    distances = []
    while True:

        labels,distance = pairwise_distances_argmin_min(X,centers,metric='minkowski')
        distances.append(distance)

        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        if np.all(centers == new_centers) or max_iterator > max_it:
            break
        centers = new_centers
        max_iterator += 1

    print("centers")
    print(centers[0])
    print("a")
    print(centers[1])
    print("distances")
    print(distances)

    return centers, labels, distances


def find_clustersgenetic(X, n_clusters, max_it, array):

    centers = array

    max_iterator = 0
    distances = []
    while True:

        labels,distance = pairwise_distances_argmin_min(X,centers,metric='minkowski')
        distances.append(distance)

        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        if np.all(centers == new_centers) or max_iterator > max_it:
            break
        centers = new_centers
        max_iterator += 1

    return centers, labels, distances


#FUNÇÃO DE NORMALIZAÇÃO


def normalize(df1):
    x = df1.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(scaled)
    return df_normalized

def normalizearray(array):
    scaled = preprocessing.MinMaxScaler().fit_transform(array)
    return scaled

def normalizeArrayofArrays(arrayphotos):

    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = preprocessing.MinMaxScaler().fit_transform(arrayphotos[x])

    return arrayphotos

def PCAarray(pcanumber, arrayphotos):

    pca = PCA(n_components=pcanumber)
    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = pca.fit_transform(arrayphotos[x])

    return arrayphotos

def PCAarrayONLY(pcanumber, arrayphotos):

    pca = PCA(n_components=pcanumber)
    arrayphotos = pca.fit_transform(arrayphotos)

    return arrayphotos

def PCAdataframe(pcanumber,dataframe):

    pca = PCA(n_components=pcanumber)
    dataframe = pca.fit_transform(dataframe)

    return dataframe

def pca1darray(pcanumber,array):
    pca = PCA(n_components=pcanumber)
    pca.fit(array)
    array = pca.transform(array)

    return array

def Turntogray(arrayphotos):

    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = cv2.cvtColor(arrayphotos[x], cv2.COLOR_BGR2GRAY)

    return arrayphotos


def reshape2dto1d(arrayphotos):

    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = arrayphotos[x].ravel()

    return arrayphotos

def resizephotos(arrayphotos, size1, size2):

    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = cv2.resize(arrayphotos[x], (size1,size2))

    return arrayphotos

def gaussianblurArray(arrayphotos,val1,val2):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.GaussianBlur(arrayphotos[x],(val1,val1), val2)

    return arrayphotos

def binaryadaptive(arrayphotos,threshold,val1):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.adaptiveThreshold(arrayphotos[x],threshold,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,val1,10)

    return arrayphotos

def invertbinaryphotos(arrayphotos):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.bitwise_not(arrayphotos[x])

    return arrayphotos

def resizeimutils(arrayphotos, width):

    for x in range(len(arrayphotos)):
        arrayphotos[x] = imutils.resize(arrayphotos[x],width=width)

    return arrayphotos

def loadfolderimgs(path):

    arrayphotos = []

    for img in glob.glob(path):
        n = cv2.imread(img)
        arrayphotos.append(n)

    return arrayphotos


def reshape3dto2d(arrayphotos):

    size = len(arrayphotos)
    for x in range (0 , size):
        arrayphotos[x] = np.reshape(arrayphotos[x], (arrayphotos[x].shape[0], (arrayphotos[x].shape[1]*arrayphotos[x].shape[2])))

    return arrayphotos

def imgtoarray(arrayphotos):

    size = len(arrayphotos)
    for x in range (0,size):
        arrayphotos[x] = np.array(arrayphotos[x] , dtype=float)

    return arrayphotos

def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def graphicPCA(pca):

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Dataset Explained Variance')
    plt.show()




# FUNÇÃO QUE FAZ A AMOSTRA ESTRATIFICADA

def amostra_estrat(tam , df , classe):
    classes = df[classe].unique()
    qtde_por_classe = round(tam / len(classes))
    amostras_por_classe = []
    for c in classes:
        indices_c = df[classe] == c
        obs_c = df[indices_c]
        amostra_c = obs_c.sample(qtde_por_classe)
        amostras_por_classe.append(amostra_c)
    amostra_estratificada = pd.concat(amostras_por_classe)
    return amostra_estratificada

def gethumoments(arrayphotos):

    for x in range(len(arrayphotos)):

        arrayphotos[x] = cv2.HuMoments(cv2.moments(arrayphotos[x]), True).flatten()

    return arrayphotos

def getHOG(arrayphotos):

    for x in range(len(arrayphotos)):
        hog = cv2.HOGDescriptor()
        arrayphotos[x] = hog.compute(arrayphotos[x]).flatten()

    return arrayphotos

def getHOGplusHU(arrayphotos):
    hog = cv2.HOGDescriptor()
    for x in range(len(arrayphotos)):
        aux = []


        h = hog.compute(arrayphotos[x]).flatten()

        for ho in h:
            aux.append(ho)

        hu = cv2.HuMoments(cv2.moments(arrayphotos[x]), True).flatten()

        for huu in hu:
            aux.append(huu)

        arrayphotos[x] = aux

    return arrayphotos


def getHOGplusHU2(arrayphotos):

    hogarray = []
    huarray = []
    hog = cv2.HOGDescriptor()
    for x in range(len(arrayphotos)):
        hogarray.append(hog.compute(arrayphotos[x]).flatten())

    hogarray = pd.DataFrame(hogarray)
    hogarray = PCAdataframe(50,hogarray)

    for y in range(len(arrayphotos)):
        huarray.append(cv2.HuMoments(cv2.moments(arrayphotos[y]),True).flatten())

    for h in range(len(arrayphotos)):
        arrayphotos[h] = np.concatenate((hogarray[h],huarray[h]))

    return arrayphotos

def extratorcaracteristicafiggeometrica(arrayimgs):

    squarescarac = []

    for x in arrayimgs:

        aux = []

        im2, countours, hierachy = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        peri = cv2.arcLength(countours[0], True)  # perimetro
        aux.append(peri)

        aproxx = cv2.approxPolyDP(countours[0], 0.04 * peri, True)  # vertices
        vertc = len(aproxx)
        aux.append(vertc)

        area = cv2.contourArea(countours[0])  # area
        aux.append(area)

        momentum = cv2.moments(x)

        #cX = int(momentum["m10"] / momentum["m00"])
        #cY = int(momentum["m01"] / momentum["m00"])

        #aux.append(cX)
        #aux.append(cY)

        moments = cv2.HuMoments(momentum, True).flatten()

        for i in moments:
            aux.append(i)

        squarescarac.append(aux)


    return squarescarac

def initialize_dic_majority(classes):
    majority = {}
    for c in classes:
        majority[c] = 0

    return majority

def accuracy_majority_vote(base, predict_labels, real_labels, n_clusters):

    classes = real_labels.unique()

    majority = []
    groups = []
    k = 0
    for i in range(n_clusters):
        group = []
        for a in range(len(base)):
            if predict_labels[a] == i:
                group.append(real_labels[a])
        groups.append(group)
        majority.append(initialize_dic_majority(classes))
        for real_label in group:
            majority[k][real_label] += 1
        k += 1

    label_groups = []
    for m in majority:
        label_groups.append(max(m.items(), key=itemgetter(1))[0])

    pred_labels = []
    true_labels = []
    for g in range(len(groups)):
        pred_labels = pred_labels + ([label_groups[g]]*len(groups[g]))
        true_labels = true_labels + [a for a in groups[g]]

    return accuracy_score(pred_labels, true_labels)

