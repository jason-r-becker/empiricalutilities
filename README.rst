Empirical Utitlities
====================

|PyPI-Status| |PyPI-Versions| |LICENSE|

``empiricalutilities`` is an empirical package for manipulating DataFrames,
 especially those with datetime indexes, and other common data science functions
 with an emphasis on improving visualization of results.

.. contents:: Table of contents
   :backlinks: top
   :local:

Installation
------------

Latest PyPI stable release
~~~~~~~~~~~~~~~~~~~~~~~~~~

|PyPI-Status|

.. code:: sh

    pip install empiricalutilities

Latest development release on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|GitHub-Status| |GitHub-Stars| |GitHub-Commits| |GitHub-Forks|

Pull and install in the current directory:

.. code:: sh

    pip install -e git+https://github.com/jason-r-becker/empiricalutilities.git@master#egg=empiricalutilities


.. code:: python

    import empiricalutilities as eu
    eu.latex_print(np.arange(1, 10))
        ...


Usage
-----

``empiricalutilities`` is very versatile and can be used in a number of ways.
Some examples of visualizing data in DataFrames and exporting to LaTeX are
provided below.

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import empiricalutilities as eu

Data Visulization
~~~~~~~~~~~~~~~~~

After generating a random DataFrame, ``color_table()`` can be used
to observe relative values compared across rows or columns. Take a dataset
that may be football players' scores on different, unnamed drills:

.. code:: python

    np.random.seed(8675309)
    cols = ['QB #1001', 'RB #9458', 'WR #7694', 'QB #5463', 'WR #7584', 'QB #7428']
    table = pd.DataFrame(np.random.randn(5, 6), columns=cols)

    color_table(table, axis=0)
    plt.show()

|Screenshot-color_table|

Simple Export to LaTeX
~~~~~~~~~~~~~~~~~~~~~~

The table of values can be easily exported to LaTeX using ``latex_print()``

.. code:: python

    eu.latex_print(table)

|Screenshot-latex_print_simple_code|

Which can be copied and pasted into LaTeX:

|Screenshot-latex_print_simple|

Table with Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, lets assume the players have run the drills multiple times so we have
average scores and standard errors. We can combine the average values with
their respective errors with just one line using ``combine_errors_table()``.
Further, we can print the results to the screen such that they are easy
to interpret using ``prettyPrint()``:

.. code:: python

    errors = pd.DataFrame(np.random.randn(5, 6), columns=cols) / 10
    error_table = combine_errors_table(table, errors, prec=3)
    eu.prettyPrint(error_table)

|Screenshot-prettyprint|

Advanced Export to LaTeX
~~~~~~~~~~~~~~~~~~~~~~~~
To export this table, we must first create the table with an additional
argument ``latex_format=True`` which lets ``combine_errors_table()`` know it
needs to print with LaTeX formatting.

.. code:: python

    error_table = combine_errors_table(table, errors, prec=3, latex_format=True)

We can also explore some of the advanced options available in ``latex_print()``.
First, the table header can be split into two rows, which is accomplished with
the ``multi_row_header=True`` argument. When True,  ``latex_print()`` expects
a DataFrame with column headers containing a ``'*'`` to mark the start of each
new row. We will use list comprehension to create a new column header list where
spaces are replaced with ``' * '``, resulting in the top header row being
player position and bottom being player number.

.. code:: python

    multi_cols = [col.replace(' ', ' * ') for col in cols]
    error_table.columns = multi_cols

Next, we can sort the header. Let's assume we want to group by position, and
are most interested in quarterbacks (QB), especially those with high numbers.
``custom_sort()`` can be used to create our own sorting rules. By setting the
sorting alphabet to ``'QWR9876543210'``, we empasize position first, QB->WR->RB,
and number second in decreasing order from 9.

.. code:: python

    sort_alphabet = 'QWR9876543210'
    sorted_cols = eu.custom_sort(multi_cols, sort_alphabet)

Additionally, we can add some expressive ability to the table by bolding the score
of the top performer for each drill. ``find_max_locs()`` identifies the
location of each row-wise maximum in the DataFrame. We must be careful to sort
the original table identically to the table with standard errors when the order
of header columns is altered.

.. code:: python

    table.columns = multi_cols
    max_locs = eu.find_max_locs(table[sorted_cols])

Finally, adding a caption can be accomplished with the ``caption`` argument, and
the uninformative index can be removed with ``hide_index=True``. For wide tables,
adding ``adjust=True`` automatically sizes the table to the proper width of
your LaTeX environment, adjusting the text size as needed.

.. code:: python

    eu.latex_print(error_table[sorted_cols],
                   caption='Advanced example of printing to LaTeX.',
                   adjust=True,
                   multi_row_header=True,
                   hide_index=True,
                   bold_locs=max_locs,
                   )

|Screenshot-latex_print_advanced|

Contributions
-------------

|GitHub-Commits| |GitHub-Issues| |GitHub-PRs|

All source code is hosted on `GitHub <https://github.com/tjason-r-becker/empiricalutilities>`__.
Contributions are welcome.


LICENSE
-------

Open Source (OSI approved): |LICENSE|


Authors
-------

The main developer(s):

- Jason R Becker (`jrbecker <https://github.com/jason-r-becker>`__)


.. |GitHub-Status| image:: https://img.shields.io/github/tag/jason-r-becker/empiricalutilities.svg?maxAge=86400
   :target: https://github.com/jason-r-becker/empiricalutilities/releases
.. |GitHub-Forks| image:: https://img.shields.io/github/forks/jason-r-becker/empiricalutilities.svg
   :target: https://github.com/jason-r-becker/empiricalutilities/network
.. |GitHub-Stars| image:: https://img.shields.io/github/stars/jason-r-becker/empiricalutilities.svg
   :target: https://github.com/jason-r-becker/empiricalutilities/stargazers
.. |GitHub-Commits| image:: https://img.shields.io/github/commit-activity/y/jason-r-becker/empiricalutilities.svg
   :target: https://github.com/jason-r-becker/empiricalutilities/graphs/commit-activity
.. |GitHub-Issues| image:: https://img.shields.io/github/issues-closed/jason-r-becker/empiricalutilities.svg
   :target: https://github.com/jason-r-becker/empiricalutilities/issues
.. |GitHub-PRs| image:: https://img.shields.io/github/issues-pr-closed/jason-r-becker/empiricalutilities.svg
   :target: https://github.com/jason-r-becker/empiricalutilities/pulls
.. |GitHub-Contributions| image:: https://img.shields.io/github/contributors/jason-r-becker/empiricalutilities.svg
   :target: https://github.com/jason-r-becker/empiricalutilities/graphs/contributors
.. |PyPI-Status| image:: https://img.shields.io/pypi/v/empiricalutilities.svg
   :target: https://pypi.org/project/empiricalutilities
.. |PyPI-Downloads| image:: https://img.shields.io/pypi/dm/v.svg
   :target: https://pypi.org/project/empiricalutilities
.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/empiricalutilities.svg
   :target: https://pypi.org/project/empiricalutilities
.. |LICENSE| image:: https://img.shields.io/pypi/l/empiricalutilities.svg
   :target: https://raw.githubusercontent.com/jason-r-becker/empiricalutilities/master/License.txt
.. |Screenshot-prettyprint| image:: https://raw.githubusercontent.com/jason-r-becker/empiricalutilities/master/images/pretty_print.png
.. |Screenshot-color_table| image:: https://raw.githubusercontent.com/jason-r-becker/empiricalutilities/master/images/color_table.png
.. |Screenshot-latex_print_simple_code| image:: https://raw.githubusercontent.com/jason-r-becker/empiricalutilities/master/images/latex_print_simple_output.png
.. |Screenshot-latex_print_simple| image:: https://raw.githubusercontent.com/jason-r-becker/empiricalutilities/master/images/latex_print_simple.png
.. |Screenshot-latex_print_advanced| image:: https://raw.githubusercontent.com/jason-r-becker/empiricalutilities/master/images/latex_print_advanced.png
