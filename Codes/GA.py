import numpy as np
import random
from Auxiliar import Functions as fc
import pandas as pd

def creatpopu(sizepopu, sizecromo):

    chromossomos = []
    for x in range(sizepopu):
        aux = []
        for y in range(sizecromo):
            element = random.randint(-20, 20)
            aux.append(element)

        chromossomos.append(aux)

    return chromossomos


#START Pre processing Images
positivas = fc.loadfolderimgs('C:/Users/leopk/PycharmProjects/CompEvolutiva/binarizadasnegativa/*.jpg')
negativas = fc.loadfolderimgs('C:/Users/leopk/PycharmProjects/CompEvolutiva/binarizadasnegativa/*.jpg')
positivas = fc.Turntogray(positivas)
negativas = fc.Turntogray(negativas)
positivas = fc.resizephotos(positivas, 64, 128)
negativas = fc.resizephotos(negativas, 64, 128)
positivas = fc.getHOGplusHU(positivas)
negativas = fc.getHOGplusHU(negativas)

df1 = pd.DataFrame(positivas)
df2 = pd.DataFrame(negativas)

df1 = fc.PCAdataframe(50,df1)
df2 = fc.PCAdataframe(50,df2)

labelsize = 219

trainparampos = df1[:800,]
testparampos = df1[800:,]
trainparamneg = df2[:800,]
testparamneg  = df2[800:,]

trainparampos = pd.DataFrame(trainparampos)
testparampos = pd.DataFrame(testparampos)
trainparamneg = pd.DataFrame(trainparamneg)
testparamneg = pd.DataFrame(testparamneg)

trainlabelspos = pd.DataFrame(np.ones(800))
trainlabelsneg = pd.DataFrame(np.zeros(800))
testlabelspos = pd.DataFrame(np.ones(labelsize))
testlabelsneg = pd.DataFrame(np.zeros(labelsize))

framestrain = [trainparampos,trainparamneg]
framestrainlabels = [trainlabelspos,trainlabelsneg]
framestest = [testparampos,testparamneg]
framestestlabels = [testlabelspos,testlabelsneg]

df_train_nolabels = pd.concat(framestrain, ignore_index=True)
df_train_labels = pd.concat(framestrainlabels, ignore_index=True)
df_test_nolabels = pd.concat(framestest, ignore_index=True)
df_test_labels = pd.concat(framestestlabels, ignore_index=True)

df_train_nolabels = fc.normalize(df_train_nolabels)
df_test_nolabels = fc.normalize(df_test_nolabels)

print(df_train_nolabels)
print(df_train_labels)
print(df_test_labels)
print(df_test_nolabels)

#END preprocessing



