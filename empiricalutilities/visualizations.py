import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools

from matplotlib import cm

def plot_color_table(df,
                     axis=1,
                     title=None,
                     rev_index=None,
                     color='RdYlGn',
                     prec=1,
                     ):
    """
    Creates color coded comparison table from dataframe values (green high, red low)

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of values
    axis: int, default=1
        axis to normalize data upon
    title: str
        title for table
    rev_index: list(str)
        list of column names to reverse color coding
    color: str, default='RdYlGn'
        color palette for table
    prec: int, default=1
        decimals places to show

    Returns
    -------
        plot of color coded table
    """

    labels = df.values
    cdf = df.copy()
    if axis == 1:
        cdf -= cdf.mean(axis=0)
        cdf /= cdf.std(axis=0)
    else:
        cdf = cdf.transpose()
        cdf -= cdf.mean(axis=0)
        cdf /= cdf.std(axis=0)
        cdf = cdf.transpose()

    if rev_index:
        for i in rev_index:
            cdf.iloc[:, i] = 1 - cdf.iloc[:, i]

    plt.figure()
    if title:
        plt.title(title)
    sns.heatmap(cdf, cmap='RdYlGn', linewidths=0.5, annot=labels,
                fmt=f'0.{prec}f', cbar=False)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

def plot_explained_variance(ev,
                            type='bar',
                            toff=0.02,
                            boff=0.08,
                            yoff=None,
                            **kwargs,
                            ):
    """
    Plot explained variance of PCA

    Parameters
    ----------
    ev: nd.array
        explained variance values
    type: str, default='bar'
        - 'bar': bar plots for cumulative variance
        - 'line': line plot for cumulative variance
    toff: float
        top x offset for cumulative variance values on plot
    boff: float
        bottom x offset for individual variance values on plot
    yoff: float
        tottom y offset for individual variance values on plot
    **kwargs:
        kwargs for plt.plot() if type==line for cumulative variance

    Returns
    -------
    plot of explained variance
    """

    ax = pd.Series(100*ev).plot(kind='bar', figsize=(10,5),
                   color='steelblue', fontsize=13)

    ax.set_ylabel('Explained Variance')
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('Principal Component')

    # Create a list to collect the plt.patches data.
    totals = []

    # Find the values and append to list.
    for i in ax.patches:
        totals.append(i.get_height())

    # Set individual bar lables using patch list.
    total = sum(totals)

    yoff_d = {'bar': 0, 'line': 0.2}
    yoff_d[type] = yoff_d[type] if yoff is None else yoff

    # Set individual bar lables using patch list.
    for j, i in enumerate(ax.patches):
        # Get_x pulls left or right; get_height pushes up or down.
        if j > 0:
            ax.text(i.get_x()+boff, i.get_height()+1,
                    str(round(100*ev[j], 1))+'%', fontsize=11,
                    color='dimgrey')
        ax.text(i.get_x()+toff, 100*np.cumsum(ev)[j]+1+yoff_d[type],
                str(round(100*np.cumsum(ev)[j], 1))+'%', fontsize=12,
                color='dimgrey')

    xlab = [f'PC-{i+1}' for i, _ in enumerate(ax.patches)]
    if type == 'bar':
        ax2 = pd.Series(np.cumsum(100*ev)).plot(kind='bar', figsize=(10,5),
                   color='steelblue', fontsize=8, alpha=0.25)
        for i in ax2.patches:
            totals.append(i.get_height())
        ax2.set_xticklabels(xlab, rotation='horizontal')
        ax2.xaxis.grid(False)
    elif type == 'line':
        plt.plot(np.cumsum(100*ev), ':o', ms=8, c='steelblue', alpha=0.75, **kwargs)
        ax.xaxis.grid(False)
        ax.set_xticklabels(xlab, rotation='horizontal')

def correlation(df, sort_col=None, plot=False):
    """
    Find correlation of DataFrame, plot results and/or
    return sorted correlations for a single column of
    the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of input values
    sort_col: str, default=None
        column to sort correlations on. If provided,
        function returns DataFrame of results
    plot: bool, default=False
        if True plot correlation matrix

    Returns
    -------
    ret_col: pd.Series
        results for one column if sort_col is given
    plot of correlation matrix if plot=True

    """

    corr = df.corr().abs()

    if plot:
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.matshow(corr)
        cmap = cm.get_cmap('coolwarm', 300)
        cax = ax.imshow(corr, interpolation="nearest", cmap=cmap)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        cax.set_clim([0, 1])
        fig.colorbar(cax, ticks=[0, .25, .5, .75, 1])


    if ret_col:
        return corr.sort_values([col], ascending=False)[col][1:]

def plot_ROC_curve(roc,
                   fpr,
                   tpr,
                   color_palette='muted',
                   colors=None,
                   figsize=(10, 10),
                   title='ROC Curves for Studied Models',
                   ):
    """
    Plot ROC curve with AUC for multliple models.

    Parameters
    ----------
    roc: dict
        dictionary of ROC AUC values
        with model names as keys
    fpr: dict
        dictionary of false positive rate
        values with model names as keys
    roc: dict
        dictionary of true positive rate
        values with model names as keys
    color_palette: str, defualt='muted'
        name of seaborn color palette to use
    colors: list(str), default=None
        if given, use provided colors instead
        of a color palette
    figsize: tuple(int), default=(10,10)
        figure size
    title: str
        figure title

    Returns
    -------
    plot of ROC curve
    """

    plt.figure(figsize=figsize)
    sns.set_palette(color_palette, len(roc))
    for m in roc.keys():
        plt.plot(fpr[m], tpr[m], alpha=0.7, label=f'{m} (area = {roc[m]:0.3f})')
    plt.plot([0, 1], [0, 1], color='dimgrey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if title:
        plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

def plot_confusion_matrix(cm,
                          classes,
                          prec=0,
                          title='Confusion Matrix',
                          cbar=True,
                          cmap=plt.cm.Blues,
                          ):
    """
    Plots the confusion matrix with count and normalzied values.

    Parameters
    ----------
    cm: nd.array
        confusiton matrix count values
    classes: list(str)
        class labels for x-axis
    prec: int, default=0
        precision of normalized values
    title: str, default='Confusion Matrix'
        figure title
    cbar: bool, default=True
        if True include color bar in figure
    cmap: matplotlib colormap, default=plt.cm.Blues
        colormap for confusion matrix

    Returns
    -------
    plot of confusion matrix
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    # Normalize counts to %
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.  # threshold for text color so it is visible
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]:.0f}\n({100*cm_norm[i, j]:.{prec}f}%)',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.grid(False)
