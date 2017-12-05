
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from darpy.plot import get_figsize
import cartopy.crs as ccrs
import cmocean

from . util import _isin


def silhouette_cluster_labels(samples, labels_da, n_clusters=None,
                              skip_clusters=[]):
    """ Return a 3-panel plot showing the cluster-based silhouette sample scores produced
    for a given set of labels.

    This plot is intended to help indentify poorly-converged or over-fitted cluster cases. To use,
    first generate the average silhouette score and samples for your analysis data and predicted
    labels::

        from sklearn.metrics import silhouette_score, silhouette_samples

        labels = labels_da['label'].ravel()
        samples = silhouette_samples(X, labels)
        score = np.mean(samples)

    Then you can pass these arguments directly to this method.

    """

    nrows, ncols = 1, 2
    size, aspect = 4., 2.
    figsize = get_figsize(nrows, ncols, size, aspect)

    cmap = cmocean.cm.thermal

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection=ccrs.PlateCarree(),
                          aspect='auto')

    labels = labels_da.data.ravel()
    if n_clusters is None:
        n_clusters = labels.max()-1
    score = np.nanmean(samples)

    y_lower = 10
    silh_avgs = []
    for i in range(n_clusters):
        if i in skip_clusters: continue
        # ith_cluster_silhouette_values = sv[pred == i]
        # ith_cluster_silhouette_values.sort()
        # ith_cluster_silhouette_values = np.sort(np.abs(samples[labels == i]))
        ith_cluster_silhouette_values = np.sort(samples[labels == i])
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)


        # Label the silhoutte plots with hteir cluster numbers at hte middle
        y_mid =  y_lower+0.5*size_cluster_i
        ax1.text(-0.05, y_mid, str(i))

        # Compute the average silhouette score for this cluster
        silh_avg = np.mean(ith_cluster_silhouette_values)
        if np.isnan(silh_avg):
            silh_avg = 0
        silh_avgs.append(silh_avg)
        # red if < average score, lack if >
        color = 'r' if silh_avg < score else 'k'
        ax1.hlines(y_mid, 0, silh_avg, color=color, lw=1)

        # compute hte new y_lower for the next plot
        y_lower = y_upper + 10

    # Average score of all the values
    ax1.axvline(x=score, color='red', linestyle='--')
    ax1.set_xlim(-0.2, 1.)
    ax1.set_title("n_clusters = {}".format(n_clusters), loc='left')

    # Plot distribution of average silhouette scores
    sns.distplot(silh_avgs, ax=ax2, color='0.7')
    ax2.axvline(x=score, color='red', linestyle='--')
    ax2.set_title("Avg. Silhouette Score = {:3.3f}".format(score))

    # Map of clusters
    print("a")
    if skip_clusters:
        labels_da = labels_da.where(~_isin(labels_da, skip_clusters)).copy()

    labels_da.plot.pcolormesh(ax=ax3, infer_intervals=True, cmap=cmap,
                              transform=ccrs.PlateCarree(), add_colorbar=False)
    ax3.set_title("")
    ax3.set_global()
    ax3.coastlines()

    return fig, [ax1, ax2, ax3]
