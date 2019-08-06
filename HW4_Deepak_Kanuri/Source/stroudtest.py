from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import warnings

base = []

#reachDist() calculates the reach distance of each point to MinPts around it

#lrd calculates the Local Reachability Density
def lrd(MinPts,DistMinPts):
    return (MinPts/np.sum(DistMinPts,axis=1))

#Finally lof calculates LOF outlier scores
def lof(Ird,MinPts,dsts):
    lof=[]
    for item in dsts:
       temp = np.divide(Ird[item[1:]],Ird[item[0]])
       lof.append(temp.sum()/MinPts)
    return lof

def reachDistance(df, MinPts):
    clf = NearestNeighbors(n_neighbors=MinPts)
    clf.fit(df)
    distancesMinPts, indicesMinPts = clf.kneighbors(df)
    distancesMinPts[:,0] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,1] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,2] = np.amax(distancesMinPts,axis=1)
    return distancesMinPts, indicesMinPts

for i in range(200):                               
    f = open("base/ModeA/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    if len(data)>1:
        data[20000] = 0
        base.append(data[:-1])

for i in range(200):                                
    f = open("base/ModeB/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    if len(data)>1:
        data[20000] = 0
        base.append(data[:-1])

for i in range(200):                                
    f = open("base/ModeC/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    if len(data)>1:
        data[20000] = 0
        base.append(data[:-1])

for i in range(200):                                
    f = open("base/ModeD/File"+str(i)+".txt", "r")
    data = f.read().split("\t")
    if len(data)>1:
        data[20000] = 0
        base.append(data[:-1])


fft = np.absolute(np.fft.fft(base))
pca = PCA(n_components=199,svd_solver='full')
warnings.filterwarnings("ignore")
newX = pca.fit_transform(fft)



#calculating LOF scores for each point
m = 10
reachdist, reachindices = reachDistance(newX, m)
lrdMat = lrd(m, reachdist)
lofScores = lof(lrdMat, m, reachindices)

alpha = lofScores
alpha.sort()

test = []
for i in range(499):                                
    f = open("Test/TestWT/Data"+str(i+1)+".txt", "r")
    data = f.read().split("\t")
    test.append(data[:-1])


fftTestX = np.absolute(np.fft.fft(test))
pca = PCA(n_components=499, svd_solver='full')
warnings.filterwarnings("ignore")
newTestX = pca.fit_transform(fftTestX)


f = open("resultstest.txt", "w")                        


m = 10
reachdist, reachindices = reachDistance(newTestX, m)
lrdMat = lrd(m, reachdist)
lofScores = lof(lrdMat, m, reachindices)

#StrOUD Algorithm with LOF as strangeness function
for i in range(len(newTestX)):
    b = 0.0
    strangeness_i = lofScores[i]
    for j in range(len(alpha)):
        if strangeness_i <= alpha[j]:
            b += 1.0
    pvalue = (b+1.0)/(float(len(alpha))+1.0)
    pvalues=1-pvalue
    print(pvalues)
    f.write(str(pvalues)+"\n")

f.close()
