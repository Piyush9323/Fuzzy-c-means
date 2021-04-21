
#  Author: Piyush Sharma

# Implementation of Fuzzy-c-means
"""

# Importing libraries
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
import operator

#Loading iris dataset
iris = pd.read_csv('https://raw.githubusercontent.com/Piyush9323/NaiveBayes_in_Python/main/iris.csv')
iris.head()
#iris.tail()

#Describing dataset
iris.describe()
iris.shape

#Dropping ID column from dataset
iris = iris.drop(['Id'], axis=1)

# Extracting features and class label
columns = list(iris.columns)
features = columns[:len(columns)-1]
class_labels = list(iris[columns[-1]])
df = iris[features]
columns

features

class_labels

df.head

"""# Defining parameters..."""

#Number of Clusters
k = 3
#Maximum number of iterations
MAX_ITER = 100 
#Number of data points
n = len(df) 
#Fuzzy parameter
m = 1.7
n

"""# Scatter Plot of iris data"""

#Scatter plot (Petal width vs Petal length)
plt.figure(figsize=(10,10))                                           
plt.scatter(list(df.iloc[:,2]), list(df.iloc[:,3]), marker='o')       
plt.axis('equal')                                                                 
plt.xlabel('Petal Length', fontsize=16)                                                 
plt.ylabel('Petal Width', fontsize=16)                                                 
plt.title('Petal Plot', fontsize=22)                                            
plt.grid()                                                                         
plt.show()

"""# -> Implementation of fuzzy-c-means from scratch.

# Initializing the membership matrix...
"""

#Function for initializing membership matrix
import random as rd
def initializeMembershipMatrix():
    membership_mat = []
    for i in range(n):
        random_num_list = [rd.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        
        flag = temp_list.index(max(temp_list))
        for j in range(0,len(temp_list)):
            if(j == flag):
                temp_list[j] = 1
            else:
                temp_list[j] = 0
        
        membership_mat.append(temp_list)
    return membership_mat

#Initializing the membership matrix
membership_mat = initializeMembershipMatrix()

"""#Calculating Cluster Center..."""

#Function for Calculating the Cluster Center
def calculateClusterCenter(membership_mat): 
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = []
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [p ** m for p in x]
        denominator = sum(xraised)
        temp_num = []
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, list(zip(*temp_num)))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers

calculateClusterCenter(membership_mat)

"""# Updating Membership Value..."""

#Function forUpdating the membership value
def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2/(m-1))
    for i in range(n):
        x = list(df.iloc[i])
        distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)       
    return membership_mat

"""## Getting the clusters..."""

#Getting the clusters
def getClusters(membership_mat): 
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels

"""# "Fuzzy C-Means""""

def fuzzyCMeansClustering():
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    acc=[]
    while curr < MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        
        acc.append(cluster_labels)
        
        if(curr == 0):
            print("Cluster Centers:")
            print(np.array(cluster_centers))
        curr += 1
    print("---------------------------")
    print("Partition matrix:")
    print(np.array(membership_mat))
    return cluster_labels, cluster_centers, acc

"""## Running Fuzzy-Means"""

labels, centers, acc = fuzzyCMeansClustering()

#Final cluster centers
print("Cluster center vectors:")
print(np.array(centers))

"""#Plotting the data"""

#Finding mode
seto = max(set(labels[0:50]), key=labels[0:50].count)
vers = max(set(labels[50:100]), key=labels[50:100].count)
virg = max(set(labels[100:]), key=labels[100:].count)

#print(seto,vers,virg)
#Clusters centers
p_mean_clus1 = np.array([centers[seto][2],centers[seto][3]])
p_mean_clus2 = np.array([centers[vers][2],centers[vers][3]])
p_mean_clus3 = np.array([centers[virg][2],centers[virg][3]])

#Extracting Petal Width and Petal Length
petal_df = iris.iloc[:,2:4]

#Labels
values = np.array(labels) 

#Search all 3 species
searchval_seto = seto
searchval_vers = vers
searchval_virg = virg

#Index of all 3 species
ii_seto = np.where(values == searchval_seto)[0]
ii_vers = np.where(values == searchval_vers)[0]
ii_virg = np.where(values == searchval_virg)[0]
ind_seto = list(ii_seto)
ind_vers = list(ii_vers)
ind_virg = list(ii_virg)

seto_df = petal_df[petal_df.index.isin(ind_seto)]
vers_df = petal_df[petal_df.index.isin(ind_vers)]
virg_df = petal_df[petal_df.index.isin(ind_virg)]

cov_seto = np.cov(np.transpose(np.array(seto_df)))
cov_vers = np.cov(np.transpose(np.array(vers_df)))
cov_virg = np.cov(np.transpose(np.array(virg_df)))

petal_df = np.array(petal_df)

x1 = np.linspace(0.5,7,150)
x2 = np.linspace(-1,4,150)
X, Y = np.meshgrid(x1,x2)

Z1 = multivariate_normal(p_mean_clus1, cov_seto)
Z2 = multivariate_normal(p_mean_clus2, cov_vers)
Z3 = multivariate_normal(p_mean_clus3, cov_virg)

pos = np.empty(X.shape + (2,)) 
pos[:, :, 0] = X; pos[:, :, 1] = Y

plt.figure(figsize=(10,10)) 
plt.scatter(petal_df[:,0], petal_df[:,1], marker='o')
plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5)
plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5)
plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5)
plt.axis('equal') 
plt.xlabel('Petal Length', fontsize=16)
plt.ylabel('Petal Width', fontsize=16)
plt.title('Petal Cluster', fontsize=22)
plt.grid()
plt.show()

# predicted labels
print(labels[:50])
print(labels[50:100])
print(labels[100:150])

"""# Calculating accuracy..."""

def accuracy(cluster_labels, class_labels):
    correct_pred = 0
    #print(cluster_labels)
    seto = max(set(labels[0:50]), key=labels[0:50].count)
    vers = max(set(labels[50:100]), key=labels[50:100].count)
    virg = max(set(labels[100:]), key=labels[100:].count)
    
    for i in range(len(df)):
        if cluster_labels[i] == seto and class_labels[i] == 'Iris-setosa':
            correct_pred = correct_pred + 1
        if cluster_labels[i] == vers and class_labels[i] == 'Iris-versicolor' and vers!=seto:
            correct_pred = correct_pred + 1
        if cluster_labels[i] == virg and class_labels[i] == 'Iris-virginica' and virg!=seto and virg!=vers:
            correct_pred = correct_pred + 1
            
    accuracy = (correct_pred/len(df))*100
    return accuracy

acc = accuracy(labels, class_labels)
print("The accuracy observed in the fuzzy-means inmplementation is :",acc)
#acc

"""# Fuzzy c-means clustering using fcmeans library"""

!pip install fuzzy-c-means

from fcmeans import FCM
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
#from seaborn import scatterplot as scatter

df1 = iris.iloc[:,[2,3]].values
n_bins = 3
x1 = np.linspace(0.5,7,150)
x2 = np.linspace(-1,4,150)
X, Y = np.meshgrid(x1,x2)

#Fit the fuzzy-c-means
fcm = FCM(n_clusters=3)
fcm.fit(df1)

#Getting cluster centers
fcm_centers = fcm.centers
fcm_centers

#Plotting library fcmeans output clusters
fcm_labels = fcm.u.argmax(axis=1)
fcm_centers = np.array(fcm_centers)
fcm_centers[[1,2]] = fcm_centers[[2,1]]

Z1 = multivariate_normal(fcm_centers[seto], cov_seto)
Z2 = multivariate_normal(fcm_centers[vers], cov_vers)
Z3 = multivariate_normal(fcm_centers[virg], cov_virg)

pos = np.empty(X.shape + (2,)) 
pos[:, :, 0] = X; pos[:, :, 1] = Y

plt.figure(figsize=(10,10))
plt.scatter(df1[:,0], df1[:,1], marker='o')
plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5)
plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5)
plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5)
plt.xlabel('Petal Length', fontsize=16)
plt.ylabel('Petal Width', fontsize=16)
plt.title('Iris Clusters', fontsize=22)
plt.axis('equal')
plt.grid()
plt.show()

# library predicted labels
l = fcm.predict(df1)
l

#Calculating fcmeans library function ACCURACY
j = 0
match = 0
for i in range(3):
    pred = {}
    for item in l[j:j+50]:
        if (item in pred):
            pred[item] += 1
        else:
            pred[item] = 1
    
    keymax = max(pred, key=pred.get)
    c = pred[keymax]
    #print(keymax,": ",c)
    match += c
    j += 50

accu = (match / 150) * 100
print("The library output accuracy is : ",accu," %")

"""# Comparision of fuzzy-c-means implementation with in-built library result

**Implemented algorithm accuracy    :  88.66 % as shown above**

**In-buit library function accuracy :  94.66 % as calculated above**

*Both the plots are almost similar.*

