from sklearn import svm
X = [[0,0], [1,1]]
Y = [0,1]
clf = svm.SVC()
clf.fit(X,Y)
clf.predict([[2.,2.]])
svc = svm.SVC(kernel = 'linear')
svc = svm.SVC(kernel = 'polynomial', degree = 3)
svc = svm.SVC(kernel = 'rbf') #radial basis function

#K-Means clustering
from sklearn import cluster, datasets
iris = datasets.load_iris()
k_means = cluster.KMeans(k=3)
k_means.fit(iris.data)
print k_means.labels_[::10]
print iris.target[::10]