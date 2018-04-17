import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm
from tabulate import tabulate


def prettyPrint(obj):
    """
    Prints object to console with nice asethetic
    :param obj: object to print
    :return:
    """
    if isinstance(obj, pd.DataFrame):
        print(tabulate(obj, headers='keys', tablefmt='psql'))
    elif isinstance(obj, dict):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(obj)
    else:
        raise ValueError('Error: {} not yet supported'.format(type(obj)))


def correlation(data, col='return', exclude=None, values=False, show=False, save=False, ret_col=False):
    if exclude is not None:
        data.drop(exclude, axis=1, inplace=True)

    corr = data.corr().abs()

    if values:
        print(corr.sort_values([col], ascending=False)[col][1:])

    if show:
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
        if save:
            plt.savefig('correlation_plot.png', bbox_inches='tight', format='png')
        plt.show()

    if ret_col:
        return corr.sort_values([col], ascending=False)[col][1:]


def color_table(df, title=None, rev_index=None):
    """
    Creates color coded comparison table from dataframe values (green high, red low)
    :param df: dataframe of table
    :param title: Title for table
    :param rev_index: list of column valuse to reverse color coding (red high, green low)
    :return: Plot of color coded table
    """
    labels = df.values
    norm_df = df.divide(np.max(df), axis=1)

    if rev_index:
        for i in rev_index:
            norm_df.iloc[:, i] = 1 - norm_df.iloc[:, i]

    plt.figure()
    if title:
        plt.title(title)
    sns.heatmap(norm_df, cmap='RdYlGn', linewidths=0.5, annot=labels, fmt='0.1f', cbar=False)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.show()


def latex_print(obj, prec=3, adjust=False, matrix_type='b'):
    """
    Returns object with syntax formatted for LaTeX
    :param obj: object (e.g., array, matrix, list)
    :param prec: precision for printing decimals
    :param adjust: adjust formatting to fit in LaTeX page width for large tables & arrays
    :param matrix_type: style for matrix in latex {'b': bracket, 'p': paranthesis}
    :return: LaTeX formatted object
    """

    if isinstance(obj, pd.Series):
        y = ['{:.{}f}'.format(elem, prec) for elem in list(obj)]
        y = ',\ '.join(y)
        y = '[' + y + ']'
        print(y)
    elif isinstance(obj, pd.DataFrame):
        print()
        print(r'﻿\begin{table}[H]')
        print(r'\centering')
        if adjust:
            print(r'\begin{adjustbox}{width =\textwidth}')
        print(obj.round(prec).to_latex())
        if adjust:
            print(r'\end{adjustbox}')
        print(r'\end{table}')
        print()
    elif isinstance(obj, np.matrix):

        if len(obj.shape) > 2:
            raise ValueError('matrix can at most display two dimensions')
        robj = obj.round(prec)
        lines = str(robj).replace('[', '').replace(']', '').splitlines()
        print()
        if matrix_type == 'b':
            rv = [r'\begin{bmatrix}']
            rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
            rv += [r'\end{bmatrix}']
        elif matrix_type == 'p':
            rv = [r'\begin{pmatrix}']
            rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
            rv += [r'\end{pmatrix}']
        else:
            raise ValueError("Error: matrix_type must be 'b' or 'p'")
        print('\n'.join(rv), '\n')
        print()
    elif isinstance(obj, np.ndarray):
        if len(obj.shape) == 1:
            y = ['{:.{}f}'.format(elem, prec) for elem in list(obj)]
            y = ',\ '.join(y)
            y = '[' + y + ']'
            print('\n', y, '\n')
        else:
            latex_print(np.matrix(obj))
    else:
        print('Error: datatype: {} is not defined in function latex_print')
