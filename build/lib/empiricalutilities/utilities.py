import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm
from tabulate import tabulate


class Memoize:
    def __init__(self, fn):
        self.fn = fn
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.fn(*args)
        return self.cache[args]


def prettyPrint(obj):
    """
    Prints object to console with nice asethetic
    :param obj: dataframe, ndarray; object to print
    :return: None
    """
    if isinstance(obj, pd.DataFrame):
        print(tabulate(obj, headers='keys', tablefmt='psql'))
    elif isinstance(obj, dict):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(obj)
    else:
        raise ValueError('Error: {} not yet supported'.format(type(obj)))

def create_df(columns, header, index=None):
    """
    Create Dataframe from given index and columns.
    :param columns: list[list]; lists containing col values
    :param header: list[str]; column names
    :param index: list; index values
    :return: DataFrame
    """

    # Ensure columns is list of list and header is list of str
    if not isinstance(columns, list):
        cols = [columns]
    else:
        cols = columns

    if not isinstance(header, list):
        head = [header]
    else:
        head = header

    # Create dict.
    d = {}
    if index:
        d['index'] = index

    for h, c in zip(head, cols):
        d[h] = c

    # Convert dict to Dataframe.
    df = pd.DataFrame.from_dict(d)
    if index:
        df.set_index('index', inplace=True)

    return df

def add_start(df, start_date):
    """
    Adds row of NaN values to dataframe with specified time index
    :param df: dataframe; old dataframe to be updated
    :param start_date: str; date to enter in dataframe
    :return: dataframe; updated dataframe with new index
    """
    start = pd.to_datetime(start_date)

    # Create time 0 dataframe row to insert
    t0 = {'time': start}
    for col in list(df):
        t0[col] = None
    t0 = pd.DataFrame([t0], columns=t0.keys())
    t0.set_index('time', inplace=True)

    # Add t0 row to dataframe
    new_df = df.append(t0)
    new_df.sort_index(inplace=True)
    return new_df

def combine_standard_errors(vals, errs, latex_format=True):
    """
        Combines standard errors and values into dataframe to print in LaTeX
        :param vals: dataframe; values for table
        :param errs: dataframe; respective errors for values
        :param latex_format: boolean; what str to us to join vals and errs, {True: ' pm ', False: ' +/- '}
        :return: dataframe; values +/- se
        """

    def err_table(val, err, join):
        """
            Given a column of values and a column of respective errors,
            return single column of joined val +/- err
            """
        v_e = []
        for i, (v, e) in enumerate(zip(val, err)):
            v_e.append('{} {} {}'.format(v, join, e))
        return np.asarray(v_e)

    if vals.size != errs.size:
        raise ValueError("Error: vals is size: {}, while errs is is size: {}.".format(vals.size, errs.size))
    df = pd.DataFrame(index=vals.index)

    pm = 'pm' if latex_format else '+/-'
    for col in list(vals):
        df[col] = err_table(vals[col], errs[col], join=pm)
    return df

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
    :param df: dataframe; table of values
    :param title: str; title for table
    :param rev_index: str list; list of column valuse to reverse color coding (red high, green low)
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


def latex_print(obj, prec=3, col_fmt=None, adjust=False, hide_index=False, matrix_type='b'):
    """
    Returns object with syntax formatted for LaTeX
    :param obj: ndarray, matrix, list, dataframe; object to convert for printing in LaTeX
    :param prec: int; precision for printing decimals
    :param col_fmt: str, column format for latex, (e.g., 'lrc')
    :param adjust: boolean; adjust formatting to fit in LaTeX page width for large tables & arrays
    :param hide_index: boolean, if True remove index from printing in LaTeX
    :param matrix_type: str; style for matrix in latex {'b': bracket, 'p': paranthesis}
    :return: LaTeX formatted object
    """
    # Series
    if isinstance(obj, pd.Series):
        # Format to LaTeX list
        y = ['{:.{}f}'.format(elem, prec) for elem in list(obj)]
        y = ',\ '.join(y)
        y = '[' + y + ']'

        # Print
        print(y)

    # DataFrame
    elif isinstance(obj, pd.DataFrame):
        # Format columns
        if col_fmt:
            fmt = col_fmt
        else:
            fmt = 'l' + 'c'*len(list(obj))

        # Round values
        obj = obj.round(prec)

        # Hide index if required
        if hide_index:
            obj.index = ['{}' for i in obj.index]

        # Remove NaN's
        obj = obj.replace(np.nan, '-', regex=True)

        # Print
        print()
        print(r'﻿\begin{table}[H]')
        print(r'\centering')
        if adjust:
            print(r'\begin{adjustbox}{width =\textwidth}')
        print(obj.to_latex(column_format=fmt))
        if adjust:
            print(r'\end{adjustbox}')
        print(r'\end{table}')
        print()

    # Matrix
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
