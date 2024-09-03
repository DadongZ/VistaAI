def get_ppv_stats(ppa, npa, prev, n1, n0):
    """
    Calculate PPV estimate and its variance based on the delta method.
    
    Parameters:
    ppa (float): pr(CDx+|CTA+)
    npa (float): pr(CDx-|CTA-)
    prev (float): prevalence
    n1 (int): number of CTA+
    n0 (int): number of CTA-
    
    Returns:
    dict: PPV estimate and variance
    """
    
    if ppa <= 0 or ppa > 1:
        raise ValueError("ppa must be greater than zero and less than or equal to one")

    if npa < 0 or npa > 1:
        raise ValueError("npa must be between 0 and 1")

    if prev <= 0 or prev > 1:
        raise ValueError("prev (prevalence) must be greater than zero and less than or equal to one")

    ppv = (prev * ppa) / (prev * ppa + (1 - prev) * (1 - npa))
    m = n1 / prev
    var_prev = (prev * (1 - prev)) / m
    var_ppa = (ppa * (1 - ppa)) / n1
    phi10 = 1 - npa
    var_phi10 = (phi10 * (1 - phi10)) / n0
    v1 = (ppv * (1 - ppv)) ** 2
    v2 = (1 / (prev * (1 - prev)) ** 2) * var_prev
    v3 = (1 / ppa ** 2) * var_ppa
    v4 = (1 / phi10 ** 2) * var_phi10
    var_ppv = v1 * (v2 + v3 + v4)

    return {"ppv_est": ppv, "var_ppv": var_ppv}

import math
from scipy.stats import norm

def get_bridging(ppa, npa, prev, n1, n0, c, delta1, var_delta1, conf_level=0.95):
    """
    Calculate the clinical efficacy in CDx+ group and its associated confidence interval.
    
    Parameters:
    ppa (float): pr(CDx+|CTA+)
    npa (float): pr(CDx-|CTA-)
    prev (float): prevalence
    n1 (int): number of CTA+
    n0 (int): number of CTA-
    c (float): assumed scale parameter for efficacy
    delta1 (float): calculated clinical efficacy in CTA+ and CDx+ group (e.g. log(HR))
    var_delta1 (float): the variance of delta1
    conf_level (float): confidence interval level, default is 0.95
    
    Returns:
    dict: Clinical efficacy estimate and confidence interval
    """
    
    z = norm.ppf(1 - (1 - conf_level) / 2)

    if npa == 1:
        cdx = delta1
        var_cdx = var_delta1
    else:
        ppv_info = get_ppv_stats(ppa, npa, prev, n1, n0)
        ppv = ppv_info["ppv_est"]
        var_ppv = ppv_info["var_ppv"]
        cdx = ((1 - c) * ppv + c) * delta1

        v1 = (2 * ppv ** 2 - 2 * ppv + 1) * var_delta1
        v2 = (1 - c) ** 2 * delta1 ** 2 + 2 * var_delta1
        var_cdx = v1 + v2 * var_ppv

    lower = cdx - z * math.sqrt(var_cdx)
    upper = cdx + z * math.sqrt(var_cdx)

    return {"delta_cdx": cdx, "delta_cdx_lower": lower, "delta_cdx_upper": upper}
