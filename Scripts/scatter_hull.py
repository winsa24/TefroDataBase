import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

class ScatterHull:

    def __init__(self):
        data = pd.read_excel("../data/TephraDataBase_normalizado.xlsx")
        data = data.replace('-',np.nan)
        data = data.replace('not determined',np.nan)
        data = data.replace('Not analyzed',np.nan)
        data = data.replace('n.a.',np.nan)
        data = data.replace('n.d.',np.nan)
        data = data.replace('Over range',np.nan)
        data = data.replace('bdl',np.nan)
        data = data.replace('<1',np.nan)
        data = data.replace('<4',np.nan)
        data = data.replace('<6',np.nan)
        data = data.replace('<5',np.nan)
        data = data.replace('<10',np.nan)
        data = data.replace('<0.01',np.nan)
        data = data.replace('<0.1',np.nan)
        data.Flag = data.Flag.astype(str)
        data = data[(data.Flag.str.contains('|'.join(['Outlier','VolcanicSource_Issue'])) == False)]
        data = data[data.Volcano!='Unknown']

        self.data = data

    def get_points_per_volcano(self, name, x_dim, y_dim):

        v_samples = self.data[self.data.Volcano==name]
        points = []
        for i, row in v_samples.iterrows():
            if np.isnan(row[x_dim]) or np.isnan(row[y_dim]): continue
            p = [row[x_dim], row[y_dim]]
            points.append(p)
        points = np.array(points)
        return points

    def get_clusters_kmeans(self, n_clusters, points):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
        clusters = {}
        for i in range(0, n_clusters):
            clusters[i] = []
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(points[i])
        for i in range(0, n_clusters):
            clusters[i] = np.array(clusters[i])
        return clusters

    def get_clusters_dbscan(self, points, eps, algorithm):
        cluster_instance = DBSCAN(eps=eps, min_samples=5, algorithm=algorithm).fit(points)
        labels = cluster_instance.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        clusters_names = list(set(labels))
        clusters_names = [str(x) for x in clusters_names]
        clusters = {}
        for i in clusters_names:
            clusters[i] = []
        for i, label in enumerate(cluster_instance.labels_):
            clusters[str(label)].append(points[i])
        for i in clusters_names:
            clusters[i] = np.array(clusters[i])
        return clusters

    def get_all_volcanoes(self):
        all_volcanoes = sorted(list(self.data['Volcano'].unique()))
        return all_volcanoes