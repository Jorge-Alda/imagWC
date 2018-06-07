from math import sqrt, log, floor
from scipy.special import erf, erfinv
from scipy.stats import chi2

def chi2sigma(Dchi2, dof):
	'Significance of a Delta chi^2 value'
	return sqrt(2) * erfinv(chi2.cdf(Dchi2, dof))

def sigma2chi(chi, dof):
	'chi^2 value corresponding to a significance'
	return chi2.isf(1-erf(sigma/sqrt(2)), dof)

def roundC(n, places=2):
	expR = floor(log(abs(n.real), 10))
	nR = 10**(expR) * round(n.real*10**(-expR) ,places)
	if n.imag == 0:
		nI = 0
		return nR
	else:
		expI = floor(log(abs(n.imag), 10))
		nI = 10**(expI) * round(n.imag*10**(-expI) ,places)
		return nR + nI*1j


