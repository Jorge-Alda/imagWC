from math import sqrt
from scipy.special import erf, erfinv
from scipy.stats import chi2

def chi2sigma(Dchi2, dof):
	'Significance of a Delta chi^2 value'
	return sqrt(2) * erfinv(chi2.cdf(Dchi2, dof))

def sigma2chi(chi, dof):
	'chi^2 value corresponding to a significance'
	return chi2.isf(1-erf(sigma/sqrt(2)), dof)
