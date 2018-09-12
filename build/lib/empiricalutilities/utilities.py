import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import re

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
    """Print object to screen in nice formatting"""
    if isinstance(obj, pd.DataFrame):
        print(tabulate(obj, headers='keys', tablefmt='psql'))
    elif isinstance(obj, dict):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(obj)
    else:
        raise ValueError('Error: {} not yet supported'.format(type(obj)))


def custom_sort(str_list, alphabet):
    """Custom sort of list of strings for given alphabet"""
    # Add all unique characters in list to end of provided alphabet.
    alphabet = alphabet + ''.join(list(set(''.join(str_list))))

    # Sort list with alphabet.
    return sorted(str_list, key=lambda x: [alphabet.index(c) for c in x])

def add_start(df, start_date):
    """
    Adds row of NaN values to dataframe with specified time index

    Parameters
    ----------
    df: pd.DataFrame
        input DataFrame
    start_date: str
        date to enter in dataframe

    Returns
    -------
    new_df: pd.DataFrame
        updated dataframe with empty row of added index
    """

    sd = pd.to_datetime(start_date)

    # Create time 0 dataframe row to insert
    t0 = {'time': sd}
    for col in list(df):
        t0[col] = None
    t0 = pd.DataFrame([t0], columns=t0.keys())
    t0.set_index('time', inplace=True)

    # Add t0 row to dataframe
    new_df = df.append(t0)
    new_df.sort_index(inplace=True)
    return new_df

def find_max_locs(df):
    """Find iloc location of maximum values in pd.DataFrame"""
    num_df = df.apply(pd.to_numeric, errors='coerce').values
    row_max = num_df.max(axis=1)
    ij_max = []

    for i, i_max in enumerate(row_max):
        if np.isnan(i_max):  # ignore nan values
            continue
        for j in range(num_df.shape[1]):
            if num_df[i, j] == i_max:
                ij_max.append((i, j))
    return(ij_max)

def combine_errors_table(vals, errs, latex_format=False):
    """
    Combines standard errors and values into DataFrame.

    Parameters
    ----------
    vals: pd.DataFrame
        values for table
    errs: pd.DataFrame
        errors for values
    latex_format: bool
        characters to use to separate values and errors
        {True: ' pm ', False: ' +/- '}

    Returns
    -------
    df: pd.DataFrame
        DataFrame of values +/- errors
    """

    def build_error_column(val, err, sep):
        """Combine single column of values and errors with specifed separator"""
        v_e = []
        for i, (v, e) in enumerate(zip(val, err)):
            v_e.append('{} {} {}'.format(v, sep, e))
        return np.asarray(v_e)

    if vals.size != errs.size:
        raise ValueError(f'vals size {vals.size} does not match errs {err.size}')
    df = pd.DataFrame(index=vals.index)

    sep = 'pm' if latex_format else '+/-'
    for col in vals.columns:
        df[col] = build_error_column(vals[col], errs[col], sep=sep)
    return df

def correlation(df, sort_col=None, plot=False):
    """
    Find correlation of DataFrame, plot results and/or
    return sorted correlations for a sincle column of
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
    plot of correlation matrix if plot=True
    correlation results for one column if sort_col is True
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


def color_table(df, title=None, rev_index=None, color='RdYlGn'):
    """
    Creates color coded comparison table from dataframe values (green high, red low)

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of values
    title: str
        title for table
    rev_index: list(str)
        list of column names to reverse color coding
    color: str, default='RdYlGn'
        color palette for table

    Returns
    -------
        plot of color coded table
    """
    labels = df.values
    norm_df = df.divide(np.max(df), axis=1)

    if rev_index:
        for i in rev_index:
            norm_df.iloc[:, i] = 1 - norm_df.iloc[:, i]

    plt.figure()
    if title:
        plt.title(title)
    sns.heatmap(norm_df, cmap='RdYlGn', linewidths=0.5, annot=labels,
                fmt='0.1f', cbar=False)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)


def replace_multiple(text, dic):
    """replace multiple text fragments in a string"""
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def latex_print(obj,
                prec=3,
                matrix_type='b',
                col_fmt=None,
                caption=None,
                adjust=False,
                hide_index=False,
                int_vals=False,
                font_size=None,
                multi_col_header=False,
                bold_locs=None,
                ):

    r"""
    Returns object with syntax formatted for LaTeX

    Parameters
    ----------

    obj: {ndarray, matrix, list, pd.Series, pd.DataFrame}
        object to convert for printing in LaTeX
    prec: int
        precision for printing decimals

    Matrix Parameters
    -----------------
    matrix_type: str
        style for matrix in latex
        {'b': bracket, 'p': paranthesis}

    DataFrame Parameters
    --------------------
    col_fmt: str
        column format for latex, (e.g., 'lrc')
    caption: str
        add caption to table
    adjust: bool
        if True, adjust formatting to fit in LaTeX page
        width for large tables & arrays
         - requires \usepackage{adjustbox} in LaTeX preamble
    hide_index: bool
        if True remove index from printing in LaTeXint_vals: bool
        if True, remove decimal places if all are 0
        (e.g., 3.00 --> 3)
    font_size: str
        font size for tables
        {'tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize',
        'large', 'Large', 'LARGE', 'huge', 'Huge'}
    multi_col_header: bool
        if True, use \thead to make header column multiple lines.
            - expects '*' as separator between lines in header column
            - requires \usepackage{makecell} in LaTeX preamble
    bold_locs: list(tuples)
        ilocs of table values to bold

    Returns
    -------
    prints LaTeX formatted object to screen
    """

    # Series
    if isinstance(obj, pd.Series):
        # Format to LaTeX list
        y = ['{:.{}f}'.format(elem, prec) for elem in list(obj)]
        y = ',\ '.join(y)
        y = '[' + y + ']'
        print(y)

    # DataFrame
    elif isinstance(obj, pd.DataFrame):
        # Format columns.
        if col_fmt:
            fmt = col_fmt
        else:
            fmt = 'l' + 'c'*len(list(obj))

        # Round values.
        obj = obj.round(prec)

        # Hide index if required
        if hide_index:
            obj.index = ['{}' for i in obj.index]

        # Remove NaN's.
        obj = obj.replace(np.nan, '-', regex=True)

        # Print table settings.
        print()
        print(r'﻿\begin{table}[H]')
        if font_size:
            print(r'\{}'.format(font_size))
        if caption:
            print(r'\caption{{{}}}'.format(caption))
        else:
            print(r'%\caption{}')
        print(r'\centering')
        if adjust:
            print(r'\begin{adjustbox}{width =\textwidth}')

        # Bold values if required.
        if bold_locs:
            for loc in bold_locs:
                obj.iloc[loc] = '\textbf{{{}}}'.format(obj.iloc[loc])

        # Replace improperly formatted values in table.
        reps = {}

        if int_vals:
            reps = {'.{} '.format('0'*i): '  '+' '*i for i in range(1, 10)}

        if multi_col_header:
            header = obj.to_latex().split('toprule\n')[1].split('\\\\\n\midrule')[0]
            new_header = ''
            for i, h in enumerate(header.split('&')):
                if i == 0:
                    new_header += '{}\n'
                elif i == len(header.split('&'))-1:
                    new_header += '& \\thead{{{}}}'.format(h).replace('*', '\\\\')
                else:
                    new_header += '& \\thead{{{}}}\n'.format(h).replace('*', '\\\\')
            reps[header] = new_header

        reps['\$'] = '$'
        reps[r'\textbackslash '] = '\\'
        reps['\{'] = '{'
        reps['\}'] = '}'
        reps['\_'] = '_'

        # Print table.
        print(replace_multiple(obj.to_latex(column_format=fmt), reps), end='')

        # Print end of table settings.
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

def date_subset(df, start=None, end=None):
    """
    Subset DataFrame to given period

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of values
    start: {str, datetime obj, Timestamp}, default None
        start date, str fmt = '%m/%d/%Y'
    end: {str, datetime obj, Timestamp}, default None
        end date, str fmt = '%m/%d/%Y'

    Returns
    -------
    df: pd.DataFrame
        DataFrame subset to given dates
    """

    if start:
        df = df[df.index >= pd.to_datetime(start, infer_datetime_format=True)]
    if end:
        df = df[df.index <= pd.to_datetime(end, infer_datetime_format=True)]
    return df

def datetime_range(start, end, minute_delta):
    """
    Create list of dates with constant timedelta.

    Parameters
    ----------
    start: datetime obj
        starting date
    end datetime obj
        ending date
    minute_delta: int
        number of minutes for time delta

    Returns
    -------
    list of datetime objects
    """

    def dt_gen(start, end, delta):
        """Create generator of datetimes with given timedelta"""
        current = start
        while current < end:
            yield current
            current += delta

    return [dt for dt in dt_gen(start, end, timedelta(minutes=minute_delta))]

def remove_outer_periods(df, cols, starts, ends):
    """
    Replace all vales outside of specifed ranges
    in DataFrame columns with NaN's

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of values
    cols: list(str)
        column names
    starts: list(datetime obj)
        list of starting dates in col order
    ends: list[datetime obj]
        list of ending dates in col order

    Returns
    -------
    new_df: pd.DataFrame
        DataFrame with NaN's before and after specified dates
    """
    d = {}
    for c, s, e in zip(cols, starts, ends):
        df_c = df[c]
        df_c = pd.DataFrame(df_c[(df_c.index >= s) & (df_c.index <= e)])
        d[c] = df_c[~df_c.index.duplicated(keep='first')]  # drop dupes
    new_df = pd.DataFrame(index=df.index)
    new_df = new_df.join(d[c] for c in cols) # add each col

    return new_df

def plot_explained_variance(ev, type='bar', toff=0.02, boff=0.08, yoff=None, **kwargs):
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
