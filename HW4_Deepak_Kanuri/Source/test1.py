from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import warnings

def reachDist(df, MinPts):
    clf = NearestNeighbors(n_neighbors=MinPts)
    clf.fit(df)
    distancesMinPts, indicesMinPts = clf.kneighbors(df)
    distancesMinPts[:,0] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,1] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,2] = np.amax(distancesMinPts,axis=1)
    return distancesMinPts, indicesMinPts

#lrd calculates the Local Reachability Density
def lrd(MinPts,knnDistMinPts):
    return (MinPts/np.sum(knnDistMinPts,axis=1))

#Finally lof calculates lot outlier scores
def lof(Ird,MinPts,dsts):
    lof=[]
    for item in dsts:
       tempIrd = np.divide(Ird[item[1:]],Ird[item[0]])
       lof.append(tempIrd.sum()/MinPts)
    return lof




totalX = []
totalY = []

for i in range(200):
    f = open("base/ModeA/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    if len(data)>1:
        totalX.append(data[:-1])
        totalY.append(0)
print(len(totalX))

for i in range(200):
    f = open("base/ModeB/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    if len(data)>1:
        totalX.append(data[:-1])
        totalY.append(0)
print(len(totalX))

for i in range(200):
    f = open("base/ModeC/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    if len(data)>1:
        totalX.append(data[:-1])
        totalY.append(0)

print(len(totalX))

for i in range(200):
    f = open("base/ModeD/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    if len(data)>1:
        totalX.append(data[:-1])
        totalY.append(0)
print(len(totalX))
##totalX=np.array(totalX, dtype=float)
##print(totalX)

fftX = np.absolute(np.fft.fft(totalX))
print(fftX)
pca = PCA(n_components=99, svd_solver='full')
warnings.filterwarnings("ignore")
newX = pca.fit_transform(fftX)

#warnings.filterwarnings("ignore")
#newX = np.fft.fft(totalX)
#newX = totalX
##
##clf = NearestNeighbors(n_neighbors=3)
##distances, indices = clf.fit(newX).kneighbors(newX)

m = 30
reachdist, reachindices = reachDist(newX, m)
irdMatrix = lrd(m, reachdist)
lofScores = lof(irdMatrix, m, reachindices)
alpha = lofScores


testX = []
testY = []
for i in range(200):
    f = open("base/ModeM/File" + str(i) + ".txt", "r")
    data = f.read().split("\t")
    if len(data)>1:
        testX.append(data[:-1])
        testY.append(0)





fftTestX = np.absolute(np.fft.fft(testX))
print(fftTestX)
pca = PCA(n_components=99, svd_solver='full')
newTestX = pca.fit_transform(fftTestX)

#newTestX = np.fft.fft(testX)
#newTestX = testX

##clf = NearestNeighbors(n_neighbors=3)
##distances, indices = clf.fit(newX).kneighbors(newTestX)


m = 30
reachdist, reachindices = reachDist(newTestX, m)
irdMatrix = lrd(m, reachdist)
lofScores = lof(irdMatrix, m, reachindices)


conf = 0.005
for k in range(20):
    num = 0.0
    for i in range(len(newTestX)):
        b = 0.0
        strangeness_i = lofScores[i]
        for j in range(len(alpha)):
            if strangeness_i <= alpha[j]:
                b += 1.0
        pvalue = (b+1.0)/(float(len(alpha))+1.0)
        if pvalue < conf:
            num += 1.0

    print ("Accuracy for confidence "+ str(1.0-conf) +" : " + str(num/float(len(testY))*100) + "%")
    conf += 0.005
