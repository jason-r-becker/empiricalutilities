import numpy as np
import pandas as pd
import empiricalutilities as eu

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
            offset = ' ' if v > 0 else ''
            if latex_format:
                v_e.append(offset+f'{v:.{prec}f} $\pm$ {abs(e):.{prec}f}')
            else:
                v_e.append(offset+f'{v:.{prec}f} ± {abs(e):.{prec}f}')
        return np.asarray(v_e)

    if vals.size != errs.size:
        raise ValueError(f'vals size {vals.size} does not match errs {err.size}')
    df = pd.DataFrame(index=vals.index)


    for col in vals.columns:
        df[col] = build_error_column(vals[col], errs[col], latex_format)
    return df


np.random.seed(8675309)
cols = ['QB #1001', 'RB #9458', 'WR #7694', 'QB #5463', 'WR #7584', 'QB #7428']
table = pd.DataFrame(np.random.randn(5, 6), columns=cols)
# eu.latex_print(table)



errors = pd.DataFrame(np.random.randn(5, 6), columns=cols) / 10
error_table = combine_errors_table(table, errors, prec=3)
eu.prettyPrint(error_table)
error_table = combine_errors_table(table, errors, prec=3, latex_format=True)
error_table

# %%
max_locs = eu.find_max_locs(table)
multi_cols = [col.replace(' ', ' * ') for col in cols]
error_table.columns = multi_cols

sort_alphabet = 'QWR9876543210'
sorted_cols = eu.custom_sort(multi_cols, sort_alphabet)

eu.latex_print(error_table[sorted_cols],
               caption='Advanced example of printing to LaTeX.',
               adjust=True,
               multi_col_header=True,
               hide_index=True,
               bold_locs=max_locs,
               )
