#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Fit of LFUV and Bs mixing observables to Wilson Coefficients in the Z' and LQ models
'''

from math import pi, sqrt, sin
from cmath import phase
import flavio
import flavio.plots
import flavio.statistics.fits
import matplotlib.pyplot as plt
from flavio.classes import Observable, Prediction, Parameter
from flavio.physics.running.running import get_alpha
import flavio.measurements


Parameter('eta_LL')
flavio.default_parameters.set_constraint('eta_LL', u'0.77 ± 0.02')
Parameter('DM_s')
flavio.default_parameters.set_constraint('DM_s', '1.1688 ± 0.111103 e-11')
R = 1.3397e-3
Parameter('beta_s')
flavio.default_parameters.set_constraint('beta_s', u'0.01852 ± 0.00032')
par = flavio.default_parameters.get_central_all()

GF = par['GF']
Vtb = sqrt(1-par['Vub']**2-par['Vcb']**2)
Vts = sqrt(1-Vtb**2)
sq2 = sqrt(2)

def Delta_Ms(wc_obj, par):
	CVLL = wc_obj.get_wc('bsbs', scale=4.8, par=par)['CVLL_bsbs']
	return par['DM_s']*abs(1.0+CVLL/R)

Observable('Delta_Ms')
Prediction('Delta_Ms', lambda wc_obj, par: Delta_Ms(wc_obj,par))

def ACP_mix(wc_obj, par):
	CVLL = wc_obj.get_wc('bsbs', scale=4.8, par=par)['CVLL_bsbs']
	phi = phase(1.0+CVLL/R)
	return sin(phi-2*par['beta_s'])

Observable('ACP_mix')
Prediction('ACP_mix', ACP_mix)
	

def wc_Z(lambdaQ, MZp):
	"Wilson coefficients as functions of Z' couplings"
	alpha = get_alpha(par, MZp+0.2)['alpha_e']
	return {
            'C9_bsmumu': -pi/(sq2* GF * (MZp+0.2)**2 *alpha) * lambdaQ/(Vtb*Vts) *1j,
            'C10_bsmumu': pi/(sq2* GF * (MZp+0.2)**2 *alpha) * lambdaQ/(Vtb*Vts) *1j,
            'CVLL_bsbs': par['eta_LL']/(4 * sq2 *GF * (MZp+0.2)**2) * (lambdaQ/(Vtb*Vts))**2 *(-1)
	}

def wc_LQ(yy, MS):
	"Wilson coefficients as functions of Leptoquark couplings"
	alpha = get_alpha(par, MS+0.2)['alpha_e']
	return {
		'C9_bsmumu': pi/(sq2* GF * (MS+0.2)**2 *alpha) * yy/(Vtb*Vts) * 1j,
		'C10_bsmumu': -pi/(sq2* GF * (MS+0.2)**2 *alpha) * yy/(Vtb*Vts) * 1j,
		'CVLL_bsbs': par['eta_LL']/(4 * sq2 *GF * (MS+0.2)**2) * (yy/(Vtb*Vts))**2 * 5 /(64*pi**2) * (-1)
	}

flavio.measurements.read_file('measCVLL.yml')
wcObj = flavio.WilsonCoefficients()


'''
exp = []
uncExp = []
expD = {}
uncExpD = {}
for m in measurements:
	expD.update(flavio.Measurement[m].get_central_all())
	uncExpD.update(flavio.Measurement[m].get_1d_errors_random())

unc = []
uncTot = []
observablesSM = [ ('BR(Bs->mumu)',), ('BR(B0->mumu)',), ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0), ('<Rmue>(B+->Kll)', 1.0, 6.0), ('<P4p>(B0->K*mumu)', 1.1, 6.0) , ('<P5p>(B0->K*mumu)', 1.1, 6.0), ('Delta_Ms',), ('ACP_mix',) ]
measurements = ['LHCb Bs->mumu 2017', 'CMS Bs->mumu 2013', 'LHCb RK* 2017', 'LHCb B->Kee 2014', 'LHCb B->K*mumu 2015 P 1.1-6', 'ATLAS B->K*mumu 2017 P4p', 'ATLAS B->K*mumu 2017 P5p', 'Rule them all']
for obs in observablesSM:
	unc += [flavio.sm_uncertainty(*obs)]
	if len(obs)==1:
		obs = obs[0]
	exp += [expD[obs]]
	uncExp += [uncExpD[obs]]
	uncTot += [sqrt(unc[-1]**2 + uncExpD[obs]**2) ]
'''

exp = [3.03448275862069e-09, 3.620689655172414e-10, 0.652542372881356, 0.6813559322033899, 0.745, 0.07, 0.01, 1.1688e-11, -0.021]
uncTot = [1.0224162247111247e-09, 2.1633572944255006e-10, 0.09783105808975297, 0.1049727973615827, 0.09067364365960803, 0.32969017997333544, 0.23984462133464168, 1.1110302680969927e-12, 0.03128684613867722]


observablesNP = [ ('BR(Bs->mumu)', wcObj), ('BR(B0->mumu)', wcObj), ('<Rmue>(B0->K*ll)', wcObj, 0.045, 1.1), ('<Rmue>(B0->K*ll)', wcObj, 1.1, 6.0), ('<Rmue>(B+->Kll)', wcObj, 1.0, 6.0), ('<P4p>(B0->K*mumu)', wcObj, 1.1, 6.0) , ('<P5p>(B0->K*mumu)', wcObj, 1.1, 6.0), ('Delta_Ms', wcObj), ('ACP_mix', wcObj) ]

	
def chicalc(l , M, wc):
	"Chi-squared statistic for the fit"
	wcObj.set_initial(wc(l, M), scale=4.8)
	chi2 = 0
	for o in range(0, len(observablesNP)):
		chi2 += ((flavio.np_prediction(*observablesNP[o]) - exp[o] ))**2/(uncTot[o]**2 )
	return chi2

def chicalcRK(l , M, wc):
	"Chi-squared statistic for the fit to RK(*)-related observables"
	wcObj.set_initial(wc(l, M), scale=4.8)
	chi2 = 0
	for o in range(0, len(observablesNP)-2):
		chi2 += ((flavio.np_prediction(*observablesNP[o]) - exp[o] ))**2/(uncTot[o]**2 )
	return chi2

def chicalcBs(l , M, wc):
	"Chi-squared statistic for the fit to Bs-mixing-related observables"
	wcObj.set_initial(wc(l, M), scale=4.8)
	chi2 = 0
	for o in (-2,-1):
		chi2 += ((flavio.np_prediction(*observablesNP[o]) - exp[o] ))**2/(uncTot[o]**2 )
	return chi2


def makefit(wc, stepM, stepL, maxM, maxL, filename, minM=0, minL=0):
	'''
	Fit and print results to file
	wc: Function that computes Wilson Coefficients from the model parameters (e.g. wc_Z, wc_LQ)
	stepM, maxM in TeV
	'''
	print('There we go!')
	chiRK0 = chicalcRK(0, 5000, wc)
	chiBs0 = chicalcBs(0, 5000, wc)
	chitot0 = chiRK0 + chiBs0
	numM = int(maxM/stepM)
	numL = int(maxL/stepL)
	for M in range(int(minM/stepM), numM+1):
		chi = []
		for l in range(int(minL/stepL), numL+1):
			chiRK = chicalcRK(l*stepL, M*stepM*1000, wc)
			chiBs = chicalcBs(l*stepL, M*stepM*1000, wc)
			chitot = chiRK + chiBs
			chi += [(chiRK, chiBs, chitot)]
		f = open(filename, 'at')
		f.write(str(M*stepM) + '\t0\t0\t0\t' + str(chiRK0) + '\t' + str(chiBs0) + '\t' + str(chitot0) + '\n' )
		i = 0		
		for l in range(int(minL/stepL), numL+1):
			WCs = wc(l*stepL, M*stepM*1000)
			C9 = WCs['C9_bsmumu']
			CVLL = WCs['CVLL_bsbs']
			f.write(str(M*stepM) + '\t' + str(l*stepL) + '\t'  + str(C9.imag) + '\t' + str(CVLL) + '\t' + str(chi[i][0]) + '\t' + str(chi[i][1]) + '\t' + str(chi[i][2]) + '\n' )
			i = i+1
		f.close() 
