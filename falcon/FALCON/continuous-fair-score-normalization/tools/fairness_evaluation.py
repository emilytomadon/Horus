import itertools
from tools.fnmr_fmr import *
from tools.enums import *


def fdr_at_threshold(fmrs, fnmrs, alpha: float = 0.5) -> float:
    """
    Calculates the Fairness Discrepancy Rate (FDR) from the given False Match Rates (FMR) and False None Match Rates (FNMR)
    These should be calculated from the comparison scores of all classes of a specific attribute of a specific dataset at a specific threshold

    Parameters
    ----------
    fmrs : np.array
        List of all False Match Rates (FMR) of all classes of a specific attribute of a specific dataset at a specific threshold
        Does not have to be in same order like FNMRs
    fnmrs : np.array
        List of all False Non Match Rates (FNMR) of all classes of a specific attribute of a specific dataset at a specific threshold
        Does not have to be in same order like FMRs
    alpha : float
        This parameter is used for the calculation of the FDR and determines the weighting of the fmrs compared to the fnmrs
        See https://arxiv.org/pdf/2203.05051.pdf for more information

    Returns
    -------
    float
        The Fairness Discrepancy Rate (FDR) of the classes from the given attribute
    """

    a = abs(max(fmrs) - min(fmrs))
    b = abs(max(fnmrs) - min(fnmrs))
    return 1 - (alpha*a + (1-alpha)*b)


def ir_at_threshold(fmrs, fnmrs, alpha: float = 0.5) -> float:
    """
    Calculates the Inequity Rate (IR) from the given False Match Rates (FMR) and False None Match Rates (FNMR)
    These should be calculated from the comparison scores of all classes of a specific attribute of a specific dataset at a specific threshold

    Parameters
    ----------
    fmrs : np.array
        List of all False Match Rates (FMR) of all classes of a specific attribute of a specific dataset at a specific threshold
        Does not have to be in same order like FNMRs
    fnmrs : np.array
        List of all False Non Match Rates (FNMR) of all classes of a specific attribute of a specific dataset at a specific threshold
        Does not have to be in same order like FMRs
    alpha : float
        This parameter is used for the calculation of the IR and determines the weighting of the fmrs compared to the fnmrs
        See https://arxiv.org/pdf/2203.05051.pdf for more information

    Returns
    -------
    float
        The Inequity Rate (IR) of the classes from the given attribute
    """
    if 0 in fmrs or 0 in fnmrs:
        return None
    a = max(fmrs) / min(fmrs)
    b = max(fnmrs) / min(fnmrs)
    return a**alpha * b**(1-alpha)

def garbe_a_b(x : list) -> float:
    """
    Calculates A and B for the Gini Aggregation Rate for Biometric Equitability (GARBE)

    Parameters
    ----------
    x : list[float]
        The fnmrs or fmrs values to calculate A or B with
    
    Returns
    -------
    float
        A or B for GARBE
    """
    n = len(x)
    sum_of_differences = 0
    for x_a, x_b in itertools.permutations(x, r=2):
        sum_of_differences += abs(x_a - x_b)
    average_x = np.average(np.array(x))
    if sum_of_differences == 0:
        return None
    return n/(n-1) * (sum_of_differences/(2 * n**2 * average_x))

def garbe_at_threshold(fmrs, fnmrs, alpha: float = 0.5) -> float:
    """
    Calculates the Gini Aggregation Rate for Biometric Equitability (GARBE) from the given False Match Rates (FMR) and False None Match Rates (FNMR)
    These should be calculated from the comparison scores of all classes of a specific attribute of a specific dataset at a specific threshold

    Parameters
    ----------
    fmrs : np.array
        List of all False Match Rates (FMR) of all classes of a specific attribute of a specific dataset at a specific threshold
        Does not have to be in same order like FNMRs
    fnmrs : np.array
        List of all False Non Match Rates (FNMR) of all classes of a specific attribute of a specific dataset at a specific threshold
        Does not have to be in same order like FMRs
    alpha : float
        This parameter is used for the calculation of the GARBE and determines the weighting of the fmrs compared to the fnmrs
        See https://arxiv.org/pdf/2203.05051.pdf for more information

    Returns
    -------
    float
        The Inequity Rate (IR) of the classes from the given attribute
    """    
    a = garbe_a_b(fmrs)
    if a is None: return None
    b = garbe_a_b(fnmrs)
    if b is None: return None
    return alpha*a + (1-alpha)*b

def fairness_scores(fnmrs, fmrs, alpha_fdr=0.995, alpha_ir=0.5, alpha_garbe=0.5):
    fdr = fdr_at_threshold(fmrs, fnmrs, alpha_fdr)
    ir = ir_at_threshold(fmrs, fnmrs, alpha_ir)
    garbe = garbe_at_threshold(fmrs, fnmrs, alpha_garbe)

    return fdr, ir, garbe

if __name__ == "__main__":
    pass