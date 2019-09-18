# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:39:39 2019

@author: Kirti Sodhi
"""
############ IMPORTING Libraries 
import nltk, re, pprint
from nltk import word_tokenize
from urllib import request
from nltk import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
from nltk.corpus import gutenberg
####### Importing First Book 
emma = gutenberg.words('whitman-leaves.txt')
emma1=[x.lower() for x in emma]
stop_words=set(stopwords.words("english"))
filtered_sent=[]
for w in emma1:
    if w not in stop_words:
        filtered_sent.append(w)
words = [word for word in filtered_sent if word.isalpha()]
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
#words_L = [lmtzr.lemmatize(word) for word in words]
x1=[]
for i in range (0,200*150,150):
     x1.append(words[i:i+150])
a=[0]*200


####################### Importing second book 
book2 = gutenberg.words('melville-moby_dick.txt')
book21=[x.lower() for x in book2]
stop_words1=set(stopwords.words("english"))

filtered_sent1=[]
for w in book21:
    if w not in stop_words1:
        filtered_sent1.append(w)
books = [word for word in filtered_sent1 if word.isalpha()]
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
#words_L1 = [lmtzr.lemmatize(word) for word in books]
x2=[]
for i in range (0,200*150,150):
     x2.append(books[i:i+150])
b=[1]*200
#
#
############### Importing third book 
book3 = gutenberg.words('bible-kjv.txt')
book31=[x.lower() for x in book3]
stop_words2=set(stopwords.words("english"))
filtered_sent2=[]
for w in book31:
    if w not in stop_words2:
        filtered_sent2.append(w)
words2 = [word for word in filtered_sent2 if word.isalpha()]
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
#words_L2 = [lmtzr.lemmatize(word) for word in words2]

x3=[]
for i in range (0,200*150,150):
     x3.append(words2[i:i+150])
c=[2]*200
#
#
#
############Merge three books and three labels
Merge=x1+x2+x3
import pandas as pd
Documents=np.asarray(Merge)
Label=a+b+c
df=[" ".join(c) for c in Merge]

#df=pd.DataFrame(df)



#importing various modules of scikit-learn's
from sklearn.metrics import confusion_matrix 
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#############TFIDF
tfidf_vectorizer = TfidfVectorizer(max_features=3000,min_df=3,max_df=0.6)
tfidf_matrix = tfidf_vectorizer.fit_transform(df).toarray()
#K-means algo start
km = KMeans(n_clusters=3, verbose=0, random_state=20,init='k-means++')
km.fit(tfidf_matrix)
labels = km.labels_
####top 10 common words from three clusters
print("top 10 clusters:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(3):
     print("Cluster %d:" % i),
     for ind in order_centroids[i, :10]:
         print(' %s' % terms[ind])
centers = km.cluster_centers_
score = silhouette_score (tfidf_matrix, labels, metric='euclidean')
print("Silhoutte_score of k-means:",score)
from sklearn.metrics import cohen_kappa_score
kappa1 = cohen_kappa_score(Label,labels,weights='linear')
print("Kappa of k-means:",kappa1)
from sklearn.metrics.cluster import adjusted_rand_score
rand1=adjusted_rand_score(Label, labels)
confusion1=confusion_matrix(Label, labels)

#Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(affinity='euclidean', 
             linkage='average', memory=None, n_clusters=3)
clustering.fit(tfidf_matrix)
labels_AGG=clustering.labels_
score_AGG = silhouette_score (tfidf_matrix, labels_AGG, metric='euclidean')
print("Silhoutte_score of AgglomerativeClustering:",score_AGG)
kappa2=cohen_kappa_score(Label,labels_AGG,weights='linear')
print("Kappa of AgglomerativeClustering:",kappa2)
from sklearn.metrics.cluster import adjusted_rand_score
rand2=adjusted_rand_score(Label, labels_AGG)
confusion2=confusion_matrix(Label, labels_AGG)

##GaussianMixture
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3,random_state=10,n_init=1, max_iter=50).fit(tfidf_matrix)
labels_gmm=gmm.predict(tfidf_matrix)
score_gmm = silhouette_score (tfidf_matrix, labels_gmm, metric='euclidean')
print("Silhoutte_score of GaussianMixture:",score_gmm)
kappa3=cohen_kappa_score(Label,labels_gmm,weights='linear')
print("Kappa of GaussianMixture:",kappa3)
from sklearn.metrics.cluster import adjusted_rand_score
rand3=adjusted_rand_score(Label, labels_gmm)
confusion3=confusion_matrix(Label, labels_gmm)

################Bag of Words
count_vect = CountVectorizer(max_features=1200)
X_train_counts = (count_vect.fit_transform(df)).toarray()
####k-means
km = KMeans(n_clusters=3, verbose=0, random_state=20,n_init=1, max_iter=50,init='k-means++')
km.fit(X_train_counts)
y_cluster_kmeans = km.predict(X_train_counts)
score1 = silhouette_score (X_train_counts, y_cluster_kmeans, metric='euclidean')
print("Silhoutte_score of k-means BOW:",score1)
kappa4 = cohen_kappa_score(Label,y_cluster_kmeans,weights='linear')
print("Kappa of k-means BOW:",kappa4)
from sklearn.metrics.cluster import adjusted_rand_score
rand4=adjusted_rand_score(Label, y_cluster_kmeans)

##Agglomerative clusterinng
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering( linkage='single', n_clusters=3)
clustering.fit(X_train_counts)
labels_AGG1=clustering.labels_
score_AGG1 = silhouette_score (X_train_counts, labels_AGG1, metric='euclidean')
print("Silhoutte_score of AgglomerativeClustering BOW:",score_AGG1)
kappa5=cohen_kappa_score(Label,labels_AGG1,weights='linear')
print("Kappa of AgglomerativeClustering BOW:",kappa5)
from sklearn.metrics.cluster import adjusted_rand_score
rand5=adjusted_rand_score(Label, labels_AGG1)
print(rand5)

#Gaussian Mixture
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, random_state=10).fit(X_train_counts)
labels_gmm1=gmm.predict(X_train_counts)
score_gmm1 = silhouette_score (X_train_counts, labels_gmm1, metric='euclidean')
print("Silhoutte_score of GaussianMixture BOW:",score_gmm1)
kappa6=cohen_kappa_score(Label,labels_gmm1,weights='linear')
print("Kappa of GaussianMixture BOW:",kappa6)
from sklearn.metrics.cluster import adjusted_rand_score
rand6=adjusted_rand_score(Label, labels_gmm1)

################consistency measure between various clusters
consistency1=cohen_kappa_score(labels,labels_gmm,weights='linear')
print('Consistency of K-means and GMM for TFIDF',consistency1 )

consistency2=cohen_kappa_score(y_cluster_kmeans,labels,weights='linear')
print('Consistency of K-means for BOW and K-means for TFIDF',consistency2 )

consistency3=cohen_kappa_score(labels_gmm,labels_AGG,weights='linear')
print('Consistency of GMM and Agglomerative clustering',consistency3 )

################Visualising comparisons of Kappa scores and random index
################kappa and adjusted random index TFIDF
import matplotlib.pyplot as plt
N = 3
kappa_score = [kappa1,kappa2,kappa3]
rand_score = [rand1,rand2,rand3]

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, kappa_score, width, label='Kappa Score')
plt.bar(ind + width, rand_score, width,
    label='Adjusted Random_Score')

plt.ylabel('Scores')
plt.title('Scores by different clustering algorithms using TF-IDF')

plt.xticks(ind + width / 2, ('K-Means','Agglomerative Clustering','Gausian Mixture'))
plt.legend(loc='best')
plt.show()


#################kappa and adjusted random index BOW
N = 3
kappa_score1 = [kappa4,kappa5,kappa6]
rand_score1 = [rand4,rand5,rand6]

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, kappa_score1, width, label='Kappa Score')
plt.bar(ind + width, rand_score1, width,
    label='Adjusted Random_Score')

plt.ylabel('Scores')
plt.title('Scores by different clustering algorithms using TF-IDF')

plt.xticks(ind + width / 2, ('K-Means','Agglomerative Clustering','Gausian Mixture'))
plt.legend(loc='best')
plt.show()


####################confusion matrix of k-means(ERROR ANALYSIS)
import seaborn as sn
import matplotlib.pyplot as plt
df_cm = pd.DataFrame(confusion1, range(3),
                  range(3))
sn.set(font_scale=1)#for label size
print('Confusion matrix of k-means for Tf-IDF')
sn.heatmap(df_cm,annot_kws={"size": 15},cmap="Blues",annot=True,cbar=False,fmt='g')

####################visualisng Clusters

###########Dendogram for TF-IDF features
from scipy.cluster.hierarchy import dendrogram, linkage
np.set_printoptions(precision=6, suppress=True)
H_cluster = linkage(tfidf_matrix,'ward')
plt.title('Dendogram')
plt.xlabel('Data')
plt.ylabel('Distance bewteen data points')
dendrogram(
H_cluster,
truncate_mode='lastp',  # show only the last p merged clusters
p=13,  # show only the last p merged clusters
leaf_rotation=90.,
leaf_font_size=12.,
show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

#########Scatter plot to visualise k-means clusters
from yellowbrick.text import TSNEVisualizer
tsne = TSNEVisualizer()
tsne.fit(tfidf_matrix, ["c{}".format(c) for c in labels])
tsne.poof()




