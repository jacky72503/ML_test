import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
colors = ['red', 'orange', 'yellow', 'green', 'cyan',
          'blue', 'purple', 'brown', 'grey', 'black']
# 產生的資料組數 (10)
clusters = 10
# K 值的範圍 (2~10)
k_range = range(2, clusters + 1)
dx, dy = make_blobs(n_samples=500, n_features=2, centers=clusters, random_state=42)
distortions = []
scores = []
# 記錄每種 K 值建出的 KMeans 模型的成效
for i in k_range:
    kmeans = KMeans(n_clusters=i,n_init='auto').fit(dx)
    distortions.append(kmeans.inertia_) # 誤差平方和 (SSE)
    scores.append(silhouette_score(dx, kmeans.predict(dx))) # 側影係數
# 找出最大的側影係數來決定 K 值
selected_K = scores.index(max(scores)) + 2
# 重新建立 KMeans 模型並預測目標值
kmeans = KMeans(n_clusters=selected_K,n_init='auto').fit(dx)
new_dy = kmeans.predict(dx)
# 新分組的資料中心點
centers = kmeans.cluster_centers_
plt.rcParams['font.size'] = 12
plt.figure(figsize=(12, 12))
# 原始資料分組
plt.subplot(221)
plt.title(f'Original data ({clusters} groups)')
plt.scatter(dx.T[0], dx.T[1], c=dy, cmap=plt.cm.Set1)
# 新資料分組
plt.subplot(222)
plt.title(f'KMeans={selected_K} groups')
plt.scatter(dx.T[0], dx.T[1], c=new_dy, cmap=plt.cm.Set3)
plt.scatter(centers.T[0], centers.T[1], marker='^', color='orange')
for i in range(centers.shape[0]): # 標上各分組中心點
    plt.text(centers.T[0][i], centers.T[1][i], str(i + 1),
             fontdict={'color': 'red', 'weight': 'bold', 'size': 24})
# 繪製誤差平方和圖 (手肘法)
plt.subplot(223)
plt.title('SSE (elbow method)')
plt.plot(k_range, distortions)
plt.plot(selected_K, distortions[selected_K - 2], 'go') # 最佳解
# 繪製係數圖
plt.subplot(224)
plt.title('Silhouette score')
plt.plot(k_range, scores)
plt.plot(selected_K, scores[selected_K - 2], 'go') # 最佳解
plt.tight_layout()
plt.show()