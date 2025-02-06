
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# genera dati artificiali in due classi
X, y = make_blobs(n_samples=500,
                  centers=[[-0.3, 0.6],
                           [0.6, -0.3]],
                  cluster_std=0.4,
                  random_state=6830)

# grafica i dati
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('2 Classes data')
plt.show()

# definire DUE SVC
# 1) con kernel LINEARE
clf_lin = SVC(kernel='linear')

# 2) con kernel RBF o GAUSSIANO, gamma = 30. e C '1000.0'
clf_rbf = SVC(kernel='rbf',
              gamma=30.,
              C=1000.0)


# addestrare le SVC sui dati,etichette: Xtr,ytr
Xtr, ytr = X, y
Xts, yts = X, y

# Scommentare per splittare set in train 66% e test 33%
# creo un subset di train su cui addestrare
# from sklearn.model_selection import train_test_split
# Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.33)

# 1) Linear
clf_lin.fit(Xtr, ytr)

# 2) RBF
clf_rbf.fit(Xtr, ytr)


# predire le etichette y1 e y2 dei vettori in X con la funzione predict(X) per i due classificatori
# 1) Linear
y_lin = clf_lin.predict(X)
print('\nLinear - CS(norm):\n', confusion_matrix(y, y_lin), '\nWhere:\n[[y1-ok, y1-not]\n[y2-not, y2-ok]')

# 2) RBF
y_rbf = clf_rbf.predict(X)
print('\nRBF - CS(over):\n', confusion_matrix(y, y_rbf), '\nWhere:\n[[y1-ok, y1-not]\n[y2-not, y2-ok]')


# grafico la sup. di separazione
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# 1) grafico Linear
Z_lin = clf_lin.predict(np.c_[np.ravel(xx), np.ravel(yy)])
Z_lin = Z_lin.reshape(xx.shape)

plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.contourf(xx, yy, Z_lin, alpha=0.3, cmap=cm_bright)
plt.title('Linear kernel')
plt.show()

# 2) grafico RBF
Z_rbf = clf_rbf.predict(np.c_[np.ravel(xx), np.ravel(yy)])
Z_rbf = Z_rbf.reshape(xx.shape)

plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.contourf(xx, yy, Z_rbf, alpha=0.3, cmap=cm_bright)
plt.title('RBF kernel')
plt.show()
