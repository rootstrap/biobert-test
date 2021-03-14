from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from scipy.spatial import distance


def elbow_test(data, n_init=10, max_clusters=30, max_iter=300):
    distortions = []
    for i in range(1, max_clusters):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=n_init, max_iter=max_iter,
            tol=1e-04, random_state=0
        )
        km.fit(data)
        distortions.append(km.inertia_)

    plt.plot(range(1, max_clusters), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig('kmeans_clusters.png')

    kn = KneeLocator(
        range(1, max_clusters),
        distortions,
        curve='convex',
        direction='decreasing',
        interp_method='interp1d',
    )
    return kn.knee

    predictors = data[data.columns[1:]]
    n_clusters = elbow_test(predictors, 10, 20, 300)
    print('Number of clusters: ', n_clusters)
    return n_clusters


def clustering_kmeans(sentences, n_clusters):
    representation = list(map(lambda sentence: sentence.representation, sentences))
    kmeans = KMeans(n_clusters, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(representation)
    clusters = []
    centers = kmeans.cluster_centers_

    for j in range(len(y_kmeans)):
        cluster_num = y_kmeans[j]
        sentences[j].avg_distance = distance.euclidean(centers[cluster_num], sentences[j].representation)
        sentences[j].cluster_index = cluster_num

    return sentences
