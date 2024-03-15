import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list
data1 = pd.DataFrame({
    'x': random_int_list(1,100,100),
    'y': random_int_list(1,100,100),
})

data = pd.DataFrame({
    'x': [12,20,28,18,29,33,24,45,45,52,51,52,55,53,55,61,64,69,72,90,85],
    'y': [39,36,30,52,54,46,55,59,63,70,66,63,58,23,14,8,19,7,24,95,88],
})

# np.random.seed(200)
k = 4
centroids = {
    i + 1: [np.random.randint(0, 100), np.random.randint(0, 100)]
    for i in range(k)
}
fig = plt.figure(figsize=(5, 5))
plt.scatter(data['x'], data['y'], color='k')
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm', 6: 'y'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()
print(centroids)


def assignment(data, centroids):
    for i in centroids.keys():
        data['distance_from_{}'.format(i)] = (
            np.sqrt((data['x'] - centroids[i][0]) ** 2 + (data['y'] - centroids[i][1]) ** 2)
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    data['closest'] = data.loc[:, centroid_distance_cols].idxmin(axis=1)
    data['closest'] = data['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    data['color'] = data['closest'].map(lambda x: colmap[x])
    return data


# data = assignment(data, centroids)
#
# fig = plt.figure(figsize=(5, 5))
# plt.scatter(data['x'], data['y'], color=data['color'], alpha=0.5, edgecolors='k')
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()

import copy

# old_centroids = copy.deepcopy(centroids)


def update_mean(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(data[data['closest'] == i]['x'])
        centroids[i][1] = np.mean(data[data['closest'] == i]['y'])
    return k


# def update_distance(k):
#     return k
# centroids = update_mean(centroids)
# print(old_centroids)
# print(centroids)
# fig = plt.figure(figsize=(5, 5))
# ax = plt.axes()
# plt.scatter(data['x'], data['y'], color=colmap[i], alpha=0.5, edgecolors='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# for i in old_centroids.keys():
#     old_x = old_centroids[i][0]
#     old_y = old_centroids[i][1]
#     dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
#     dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
#     ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
# plt.show()
data = assignment(data, centroids)
fig = plt.figure(figsize=(5, 5))
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.scatter(data['x'], data['y'], color=data['color'], alpha=0.5, edgecolors='k')
plt.show()
while True:
    closest_centroids = data['closest'].copy(deep=True)
    centroids = update_mean(centroids)
    data = assignment(data, centroids)
    plt.scatter(data['x'], data['y'], color=data['color'], alpha=0.5, edgecolors='k')
    plt.show()
    if closest_centroids.equals(data['closest']):
        break
