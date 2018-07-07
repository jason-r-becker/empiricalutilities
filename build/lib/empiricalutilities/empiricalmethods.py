import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from sympy import Matrix, solve_linear_system, var
from IPython.display import display

from .utilities import prettyPrint

class OLS:
    """
    Example Use:
    df = pd.DataFrame({'y': [13, 15, 17, 18, 19, 22, 25, 27, 29, 30, 32, 35],
                   'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
    y, x = df['y'], df['x']
    mod = OLS(y, x)
    res = mod.fit(const=True, cov_type='HC1')  # add constant and use heteroskedastic White variance estimate
    mod.summary(res)
    mod.plot_single_variable(res, xlab='x', ylab='y')
    """
    def __init__(self, Y, X, dropna=False):
        """
        Oridnary Least Squares Regression class
        :param Y: Dataframe column of dependent variable
        :param X: Series of explanatory variable
        """

        # Make X and Y Dataframes
        self.x = pd.DataFrame(X)

        if isinstance(Y, pd.DataFrame):
            Y.columns = ['y']
            self.y = Y
        elif isinstance(Y, pd.Series):
            Y = pd.DataFrame(Y)
            Y.columns = ['y']
            self.y = Y
        else:
            self.y = pd.DataFrame(Y, columns=['y'])

        if dropna:
            combined = self.x.join(self.y, how='inner')
            combined.dropna(inplace=True)
            self.y = combined['y']
            self.x = combined.drop('y', axis=1)

        self.preds = None  # predictions

    def fit(self, const=True, cov_type='HC1', ci=95):
        """
        Fit OLS by minimizing sum of squares
        :param const: Add constant to regression {Default: True}
        :param cov_type: Covariance type {Homoskedastic: 'hom', Heteroskedastic White: 'het',
                                          White biased: 'HC0', White unbiased: 'HC1'}
        :param ci: Confidence Interval expressed as integer percent {Default: 95 (%)}
        :return: Dictionary of results including test statistics
        """

        results = {}

        # Add 1's column if required
        if const:
            labels = ['const'] + list(self.x)
            self.x['const'] = 1
            self.x = self.x[labels]

        # Solve OLS equation B = (X'X)^-1 (X'Y)
        X = self.x.values
        Y = np.squeeze(self.y.values)

        n = X.shape[0]  # n observations
        k = X.shape[1]  # k predictors
        p = k - 1
        coeffs = np.squeeze(np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y)))
        results['coeffs'] = coeffs
        results['n'] = n
        results['df_resids'] = n - k
        results['df_model'] = p

        # Use model to evaluate residuals and MSE
        self.preds = np.dot(X, coeffs)
        results['preds'] = self.preds
        e = Y - self.preds
        mse = np.dot(e, e) / (n - k)
        results['resids'] = e
        results['mse'] = mse

        # Calculate R^2 and adj R^2
        et = Y - np.mean(Y)
        r2 = 1 - np.dot(e, e) / np.dot(et, et)
        adj_r2 = r2 - (1 - r2) * p / (n - p - 1)
        results['R^2'] = r2
        results['adj. R^2'] = adj_r2

        # F-test
        em = self.preds - np.mean(Y)
        msm = np.dot(em, em) / p
        F = msm / mse
        p_F = 1 - stats.f.cdf(F, p, n - p - 1)
        results['F'] = F
        results['p_F'] = p_F

        # Log-likelihood
        ssr = np.dot(e, e)  # sum square residuals
        s2 = ssr / n
        L = np.log((2 * np.pi * s2) ** (-n / 2) * np.exp(-ssr / (2 * s2)))
        results['L'] = L

        # AIC and BIC
        AIC = 2 * k - 2 * L
        BIC = k * np.log(n) - 2 * L
        results['AIC'] = AIC
        results['BIC'] = BIC

        # Variance-covariance matrix
        if cov_type in ['het', 'hetero', 'heteroskedastic', 'HC1']:  # Unbiased White estimator
            vcm = np.diag(
                np.linalg.inv(X.T.dot(X)).dot((X.T.dot(np.diag(e ** 2)).dot(X))).dot(np.linalg.inv(X.T.dot(X))))
            vcm = vcm * n / (n - k)  # make unbiased White estimator
            results['cov_type'] = 'HC1'
        elif cov_type == 'HC0':  # Biased White estimator
            vcm = np.diag(
                np.linalg.inv(X.T.dot(X)).dot((X.T.dot(np.diag(e ** 2)).dot(X))).dot(np.linalg.inv(X.T.dot(X))))
            results['cov_type'] = 'HC0'
        elif cov_type in ['hom', 'homo', 'homoskedastic']:
            vcm = np.diag(mse * np.linalg.inv(X.T.dot(X)))
            results['cov_type'] = 'Homoskedastic'
        else:
            raise ValueError("cov_type unknown, select either 'het' or 'hom'")
        results['VCM'] = vcm

        # Standard error
        se = np.sqrt(vcm)
        results['se'] = se

        # t-statistic & z-statistic
        tz = coeffs / se
        if cov_type in ['het', 'hetero', 'heteroskedastic', 'HC1', 'HC0']:  # White estimator
            # z-statistic
            test = 'z'
            p_z = 2 * stats.norm.sf(abs(tz))
            p_test = p_z
            results['z'] = tz
            results['p_z'] = p_z

        elif cov_type in ['hom', 'homo', 'homoskedastic']:
            # t-statistic
            test = 't'
            p_t = 2 * stats.t.sf(np.abs(tz), n - k)
            p_test = p_t
            results['t'] = tz
            results['p_t'] = p_t

        # Confidence Intervals
        ci_ = se * stats.t.ppf(1 - (1 - ci / 100) / 2, n - k)
        ci_left = (1 - ci / 100) / 2  # left tail value
        ci_right = 1 - (1 - ci / 100) / 2  # right tail value
        results['ci'] = ci_

        # Skewness and Kurtosis
        S = np.mean(e ** 3) / np.mean(e ** 2) ** (3 / 2)
        K = np.mean(e ** 4) / np.mean(e ** 2) ** 2
        results['skew'] = S
        results['kurtosis'] = K

        # Omnibus
        def Z1(S, N):
            Y = S * np.sqrt(((N + 1) * (N + 3)) / (6.0 * (N - 2.0)))
            b = 3.0 * (N ** 2.0 + 27.0 * N - 70) * (N + 1.0) * (N + 3.0)
            b /= (N - 2.0) * (N + 5.0) * (N + 7.0) * (N + 9.0)
            W2 = - 1.0 + np.sqrt(2.0 * (b - 1.0))
            alpha = np.sqrt(2.0 / (W2 - 1.0))
            z = 1.0 / np.sqrt(np.log(np.sqrt(W2)))
            z *= np.log(Y / alpha + np.sqrt((Y / alpha) ** 2.0 + 1.0))
            return z

        def Z2(K, N):
            E = 3.0 * (N - 1.0) / (N + 1.0)
            v = 24.0 * N * (N - 2.0) * (N - 3.0)
            v /= (N + 1.0) ** 2.0 * (N + 3.0) * (N + 5.0)
            X = (K - E) / np.sqrt(v)
            b = (6.0 * (N ** 2.0 - 5.0 * N + 2.0)) / ((N + 7.0) * (N + 9.0))
            b *= np.sqrt((6.0 * (N + 3.0) * (N + 5.0)) / (N * (N - 2.0) * (N - 3.0)))
            A = 6.0 + (8.0 / b) * (2.0 / b + np.sqrt(1.0 + 4.0 / b ** 2.0))
            z = (1.0 - 2.0 / A) / (1.0 + X * np.sqrt(2.0 / (A - 4.0)))
            z = (1.0 - 2.0 / (9.0 * A)) - z ** (1.0 / 3.0)
            z /= np.sqrt(2.0 / (9.0 * A))
            return z

        omni = Z1(S, n) ** 2 + Z2(K, n) ** 2
        p_omni = 1 - stats.chi2(2).cdf(omni)
        results['omnibus'] = omni
        results['p_omnibus'] = p_omni

        # Durbin-Watson Test
        DW = np.sum(np.diff(e) ** 2) / ssr
        results['DW'] = DW

        # Jarque-Bera Test
        JB = (n / 6) * (S ** 2 + (1 / 4) * (K - 3) ** 2)
        p_JB = 1 - stats.chi2(2).cdf(JB)
        results['JB'] = JB
        results['p_JB'] = p_JB

        # Condition Number
        X_mat = np.asmatrix(X)
        EV, _ = np.linalg.eig(X_mat.T * X_mat)
        CN = np.sqrt(np.max(EV) / np.min(EV))
        results['CN'] = CN

        # Summary table
        summary = pd.DataFrame({'coef': coeffs, 'std err': se, test: tz, 'P>|{}|'.format(test): p_test,
                                '[{:.3f}'.format(ci_left): coeffs - ci_,
                                '{:.3f}]'.format(ci_right): coeffs + ci_},
                               index=list(self.x))
        summary = summary[['coef', 'std err', test, 'P>|{}|'.format(test),
                           '[{:.3f}'.format(ci_left), '{:.3f}]'.format(ci_right)]]
        results['summary'] = summary.round(3)

        # Test table
        r = results
        table = pd.DataFrame(
            {'Statistic': ['Cov Type:', 'No. Obsv:', 'DF Resids:', 'DF Model:', 'R-squared:', 'Adj. R-squared:',
                           'MSE:', 'F-statistic:', 'P (F-statistic):', 'Log-Likelihood:'],
             'Values': [0, r['n'], r['df_resids'], r['df_model'], r['R^2'], r['adj. R^2'],
                        r['mse'], r['F'], r['p_F'], r['L']],
             'Statistic ': ['AIC:', 'BIC:', 'Omnibus:', 'P (Omnibus):', 'Skew:', 'Kurtosis:', 'Durbin-Watson:',
                            'Jarque-Bera:', 'P (JB):', 'Cond. No.:'],
             'Values ': [r['AIC'], r['BIC'], r['omnibus'], r['p_omnibus'], r['skew'], r['kurtosis'], r['DW'],
                         r['JB'], r['p_JB'], r['CN']]})
        table.set_index('Statistic', inplace=True)
        table = table.round(3)
        pd.options.mode.chained_assignment = None  # turn off warning for next line
        table['Values']['Cov Type:'] = r['cov_type']  # must be float for rounding, change 0 val to str
        table = table[['Values', 'Statistic ', 'Values ']]
        results['table'] = table

        return results

    def summary(self, results):
        """
        Print summary of results including all test statistics
        :param results: Dictionary of OLS.fit() results
        :return: print table of results to screen
        """

        prettyPrint(results['summary'])
        prettyPrint(results['table'])
    
    def jsummary(self, results):
        """
        Print summary of results including all test statistics in Jupyter Notebook
        :param results: dict; OLS.fit() results
        :return: display table of results in Jupyter Notebook
        """
    
        display(results['summary'])
        display(results['table'])

    def plot_single_variable(self, results, xlab=None, ylab=None, save=None):
        """
        Plot regression for single explanatory variable
        :param results: Dictionary of OLS.fit() results
        :param xlab: str x-axis label
        :param ylab: str y-axis label
        :param save: str of filename to save plot
        :return: plot of regression
        """

        n, m = self.x.shape
        if m != 2:
            raise ValueError('Error: can only plot single explanatory variable. Current X shape: ({},{})'.format(n, m))
        X = self.x.drop('const', axis=1)
        X = np.squeeze(X.values)
        Y = np.squeeze(self.y.values)
        plt.plot(X, Y, 'o', c='steelblue')
        Xmin = np.argmin(X)
        Xmax = np.argmax(X)
        plt.plot([X[Xmin], X[Xmax]], [self.preds[Xmin], self.preds[Xmax]], c='darkorange')
        if xlab:
            plt.xlabel(xlab)
        if ylab:
            plt.ylabel(ylab)
        if save:
            plt.savefig('{}.png'.format(save), dpi=200)
        plt.show()

class AR:
    
    def __init__(self, X, p=1):
        self.x = np.asarray(X)
        # create dataframe of lags
        self.lags = pd.DataFrame()
        for i in range(p):
            self.lags['phi_{}'.format(i)] = self.x[i:-p+i]
        self.lags['phi_{}'.format(p)] = self.x[p:]
        self.lags.dropna(inplace=True)
        
        self.y = np.array(self.lags['phi_0'])
        self.lags.drop(['phi_0'], axis=1, inplace=True)
    
    def fit(self, kendall_correction=False):
        """
            AR(p) OLS results of AR(1) model
            :param self.t0: ndarray; x_(t) values to run AR model on.
            :param self.lags: dataframe; columns of x_(t-1), x_(t-2), ..., x_(t-p) lags for AR model.
            :param kendall_correction: boolean; if True, perform Kendall correction on phi for phi ~ 1
            :return: Dictionary of OLS results
            """
        
        # Run OLS
        mod = eu.OLS(self.y, self.lags)
        res = mod.fit()
        
        # Kendall correction
        if kendall_correction:
            for i, phi in enumerate(res['coeffs'][1:]):  # remove constant coefficient
                if phi > 0.98:
                    phi_cor = phi + (1+3*phi) / len(self.y)
                    print('phi_{} after Kendall Correction: {:.5f}'.format(i + 1, phi_cor))
        else:
            for i, phi in enumerate(res['coeffs'][1:]):  # remove constant coefficient
                if phi > 0.98:
                    print('Warning: phi_{} = {:.5f}, may be persistent'.format(i + 1, phi))
        return res
    
    def summary(self, results, prec=3):
        """
        Print summary of results including all test statistics
        :param results: Dictionary of OLS.fit() results
        :return: print table of results to screen
        """
        
        prettyPrint(results['summary'])
        prettyPrint(results['table'])

    def jsummary(self, results):
        """
        Print summary of results including all test statistics in Jupyter Notebook
        :param results: dict; OLS.fit() results
        :return: display table of results in Jupyter Notebook
        """
            
        display(results['summary'])
        display(results['table'])

class ARCH_1:
    """
        ARCH(1) model with attributes fit() and predict()
        """
    def __init__(self, r):
        self.raw_ret = r
        self.ret = np.array(r.dropna())
        self.params = None
        self.err = None
    
    def fit(self, n):
        """
        Run ARCH(1) model n times, saving returning result with lowest standard error
        :param n: int; number of ARCH(1) models to run
        :return: (ndarray, ndarray); (parameter estimates, standard errors)
        for ARCH(1) model with minimum sum of standard errors
        """
        
        def log_likelihood_ARCH(theta, X):
            """
            Returns log likelihood of ARCH(1) model
            :param X: array: values to run ARCH model on
            :return: float: log likelihood
            """
            
            # Initialize variables
            x_t = X[1:-1]  # x(t)
            x_lag = X[:-2]  # x(t-1)
            
            # Compute residuals
            err = x_t - theta[0] - theta[1]*x_lag
            err_t = err[1:]
            err_lag = err[:-1]
            
            # Log likelihood function
            h_t = theta[2] + theta[3]*err_lag**2
            llf = 0.5 * np.sum(np.log(2*np.pi) + np.log(h_t) + err_t**2/h_t)
            return llf
        
        def single_run(ret):
            """
            Single ARCH Model run
            """
            theta0 = 0.001 + np.random.rand(4)
            bnds = ((0.001, 1), (0.001, 1), (0.001, 1), (0.001, 1))
            res = optimize.minimize(log_likelihood_ARCH, theta0, args=ret, bounds=bnds, method='L-BFGS-B')
            hess_inv = res.hess_inv.todense()
            se = np.array(np.sqrt(np.diag(hess_inv)))
            return res.x, se
        
        params = []
        err = []
        sum_err = np.zeros(n)
        for i in range(n):
            param_i, err_i = single_run(self.ret)
            params.append(param_i)
            err.append(err_i)
            sum_err[i] = np.sum(err_i)
        opt = int(np.argmin(sum_err))
        self.params = params[opt]
        self.err = err[opt]
        return params[opt], err[opt]
    
    def predict(self):
        # Initialize variables
        x_t = self.ret[1:-1]  # x(t)
        x_lag = self.ret[:-2]  # x(t-1)
        
        # Compute residuals
        err = x_t - self.params[0] - self.params[1]*x_lag
        err_lag = err[:-1]
        
        h_t = self.params[2] + self.params[3]*err_lag**2
        
        # Add results to original index
        preds = pd.Series(index=self.raw_ret.index)
        preds[-len(h_t) - 1:-1] = np.sqrt(h_t)
        return preds

class GARCH_11:
    """
    CH(1,1) model with attributes fit() and predict()
    """
    
    def __init__(self, r):
        self.raw_ret = r
        self.ret = np.array(r.dropna())
        self.params = None
        self.err = None
    
    def fit(self, n):
        """
        Run GARCH(1,1) model n times, saving returning result with lowest standard error
        :param n: int; number of GARCH(1,1) models to run
        :return: (ndarray, ndarray); (parameter estimates, standard errors)
        for GARCH(1,1) model with minimum sum of standard errors
        """
        
        def log_likelihood_GARCH(theta, X):
            """
            Returns log likelihood of ARCH(1) model
            :param X: array: values to run ARCH model on
            :return: float: log likelihood
            """
            
            # Initialize variables
            x_t = X[1:-1]  # x(t)
            x_lag = X[:-2]  # x(t-1)
            
            # Compute residuals
            err = x_t - theta[0] - theta[1]*x_lag
            err_t = err[1:]
            err_lag = err[:-1]
            
            h_t = np.zeros(len(err_t))
            h_t[0] = theta[2] + theta[3]*err_lag[0]**2
            for i in range(1, len(h_t)):
                h_t[i] = theta[2] + theta[3]*err_lag[i]**2 + theta[4]*h_t[i-1]
            
            # Log likelihood function
            llf = 0.5 * np.sum(np.log(2*np.pi) + np.log(h_t) + err_t**2/h_t)
            return llf
        
        def single_run(ret):
            """
            Single ARCH Model run
            """
            theta0 = 0.001 + np.random.rand(5)
            theta0[4] = 0.9
            bnds = ((0.001, 1), (0.001, 1), (0.0001, 1), (0.001, 1), (0.001, 1))
            res = optimize.minimize(log_likelihood_GARCH, theta0, args=ret, bounds=bnds, method='L-BFGS-B')
            hess_inv = res.hess_inv.todense()
            se = np.array(np.sqrt(np.diag(hess_inv)))
            return res.x, se
        
        params = []
        err = []
        sum_err = np.zeros(n)
        for i in range(n):
            param_i, err_i = single_run(self.ret)
            params.append(param_i)
            err.append(err_i)
            sum_err[i] = np.sum(err_i)
        opt = int(np.argmin(sum_err))
        self.params = params[opt]
        self.err = err[opt]
        return params[opt], err[opt]
    
    def predict(self):
        # Initialize variables
        x_t = self.ret[1:-1]  # x(t)
        x_lag = self.ret[:-2]  # x(t-1)
        
        # Compute residuals
        err = x_t - self.params[0] - self.params[1]*x_lag
        err_lag = err[:-1]
        
        h_t = self.params[2] + self.params[3]*err_lag**2
        h_t[0] = self.params[2] + self.params[3]*err_lag[0]**2
        for i in range(1, len(h_t)):
            h_t[i] = self.params[2] + self.params[3]*err_lag[i]**2 + self.params[4]*h_t[i-1]
        
        # Add results to original index
        preds = pd.Series(index=self.raw_ret.index)
        preds[-len(h_t) - 1:-1] = np.sqrt(h_t)
        return preds

def sym_solve_matrix(A, b):
    """
    Symbolically solve matrix in form Ax = b
    :param A: numpy matrix [N x M]
    :param b: numpy matrix [N x 1]
    :return: Dictonary of x_i values
    """

    N, M = A.shape
    v = var('x:{}'.format(M))
    system = Matrix(np.concatenate((A, b), axis=1))
    syms = 'v[0]'
    for i in range(M-1):
        syms = ''.join([syms, ', v[{}]'.format(i+1)])
    return eval('solve_linear_system(system, {})'.format(syms))


def t_test(a, b):
    s1 = np.std(a, ddof=1)
    s2 = np.std(b, ddof=1)
    n1 = len(a)
    n2 = len(b)
    t = (np.mean(a) - np.mean(b)) / np.sqrt(s1**2/n1 + s2**2/n2)
    dof = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    p = 2 * stats.t.sf(np.abs(t), dof)
    return t, p

