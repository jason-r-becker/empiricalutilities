import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import re

from datetime import timedelta
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

def find_min_locs(df):
    """Find iloc location of minimum values in pd.DataFrame"""
    num_df = df.apply(pd.to_numeric, errors='coerce').values
    row_min = num_df.min(axis=1)
    ij_min = []

    for i, i_min in enumerate(row_min):
        if np.isnan(i_min):  # ignore nan values
            continue
        for j in range(num_df.shape[1]):
            if num_df[i, j] == i_min:
                ij_min.append((i, j))
    return(ij_min)

def combine_errors_table(vals, errs, prec=3, latex_format=False):
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
        {True: ' \pm ', False: ' +/- '}

    Returns
    -------
    df: pd.DataFrame
        DataFrame of values +/- errors
    """

    def build_error_column(val, err, latex_format):
        """Combine single column of values and errors with specifed separator"""
        v_e = []
        for i, (v, e) in enumerate(zip(val, err)):
            if latex_format:
                v_e.append(f'{v:.{prec}f} $\pm$ {abs(e):.{prec}f}')
            else:
                v_e.append(f'{v:.{prec}f} ± {abs(e):.{prec}f}')
        return np.asarray(v_e)

    if vals.size != errs.size:
        raise ValueError(f'vals size {vals.size} does not match errs {err.size}')
    df = pd.DataFrame(index=vals.index)

    for col in vals.columns:
        df[col] = build_error_column(vals[col], errs[col], latex_format)
    return df

def replace_multiple(text, dic):
    """replace multiple text fragments in a string"""
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def greeks_to_latex(text):
    """
    Transform text containing greeks letters with/without
    subscripts to proper LaTeX representation.

    Parameters
    ----------
    text: str
        text containing greek letter to be formatted to LaTeX

    Returns
    -------
    text: str
        text with all greeks formatted into LaTeX syntax

    Examples
    --------
    greeks_to_latex('alpha=0.5, Beta=2, \tgamma_=3, \nDelta^4=16')
    >>> $\alpha$= 0.5, $\Beta$= 2, 	gamma_=3,
    >>> $\Delta^{4}$=16

    greeks_to_latex('partial u / partial t - alpha_gold nabla^2 u = 0')
    >>> $\partial$ u / $\partial$ t - $\alpha_{gold}$ $\nabla^{2}$ u = 0
    """

    greeks = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta',
              'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron',
              'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi',
              'omega','varepsilon', 'vartheta', 'varpi', 'varrho', 'varsigma',
              'varphi', 'digamma', 'partial', 'eth', 'hbar', 'nabla', 'infty',
              'aleph', 'beth', 'gimel',
              ]

    # Capitalize each greek letter and append to greeks list.
    greeks = greeks + [greek.title() for greek in greeks]
    text = ' ' + text  # add leading space

    for greek in greeks:
        # Case 1: 'greek_123'
        p = re.compile('\s'+greek+'[_]\w+')
        findings = p.findall(text)
        for finding in findings:
            lead_char = re.search('\s', finding).group(0)
            split = finding.split('_')
            new = r'$\{}_{{{}}}$'.format(*split).replace(lead_char, '')
            text = text.replace(finding, lead_char+new)

        # Case 2: 'greek123'
        p = re.compile('\s'+greek+'\d+')
        findings = p.findall(text)
        for finding in findings:
            lead_char = re.search('\s', finding).group(0)
            letter = finding[:len(lead_char+greek)]
            num = finding[len(lead_char+greek):]
            new = r'$\{}_{{{}}}$'.format(letter, num).replace(lead_char, '')
            text = text.replace(finding, lead_char+new)

        # Case 3: 'greek^123'
        p = re.compile('\s'+greek+'[\^]\d+')
        findings = p.findall(text)
        for finding in findings:
            lead_char = re.search('\s', finding).group(0)
            split = finding.split('^')
            new = r'$\{}^{{{}}}$'.format(*split).replace(lead_char, '')
            text = text.replace(finding, lead_char+new)

        # Case 4: 'greek'
        p = re.compile('\s'+greek+'[^_]')
        findings = p.findall(text)
        for finding in findings:
            lead_char = re.search('\s', finding).group(0)
            letter = re.sub(r'\W+', '', finding) #remove non-alphanum chars
            new = r'$\{}${}'.format(letter, finding[-1]).replace(lead_char, '')
            text = text.replace(finding, lead_char+new+' ')

    return text[1:]  # remove leading space

def latex_print(obj,
                prec=3,
                matrix_type='b',
                col_fmt=None,
                caption=None,
                adjust=False,
                hide_index=False,
                int_vals=False,
                font_size=None,
                greeks=True,
                multi_row_header=False,
                bold_locs=None,
                ):

    r"""
    Returns object with syntax formatted for LaTeX

    Parameters
    ----------

    obj: {ndarray, matrix, list, pd.Series, pd.DataFrame}
        object to convert for printing in LaTeX
    prec: int, default=3
        precision for printing decimals

    Matrix Parameters
    -----------------
    matrix_type: str, defualt='b'
        style for matrix in latex
        {'b': bracket, 'p': paranthesis}

    DataFrame Parameters
    --------------------
    col_fmt: str, default=None
        column format for latex, (e.g., 'lrc')
    caption: str, default=None
        add caption to table
    adjust: bool, default=False
        if True, adjust formatting to fit in LaTeX page
        width for large tables & arrays
         - requires \usepackage{adjustbox} in LaTeX preamble
    hide_index: bool, default=False
        if True remove index from printing in LaTeXint_vals: bool
        if True, remove decimal places if all are 0
        (e.g., 3.00 --> 3)
    font_size: str, default=None
        font size for tables
        {'tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize',
        'large', 'Large', 'LARGE', 'huge', 'Huge'}
    greeks: bool, default=True
        convert all instances of greek letters to LaTeX syntax
    multi_row_header: bool, default=False
        if True, use \\thead to make header column multiple lines.
            - expects '*' as separator between lines in header column
            - requires \usepackage{makecell} in LaTeX preamble
    bold_locs: list(tuples), default=None
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

        if multi_row_header:
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
        reps[r'\textasciicircum '] = '^'
        
        # Print table.
        table = replace_multiple(obj.to_latex(column_format=fmt), reps)
        if greeks:
            print(greeks_to_latex(table), end='')
        else:
            print(table, end='')

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
        current = pd.to_datetime(start, infer_datetime_format=True)
        while current < pd.to_datetime(end, infer_datetime_format=True):
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
