#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Fit of LFUV and Bs mixing observables to Wilson Coefficients in the Z' and LQ models
'''

from math import pi, sqrt
from cmath import phase
import flavio
from flavio.physics.running.running import get_alpha
from flavio.physics import ckm
import flavio.measurements
from wilson import Wilson
from utils import roundC

par = flavio.default_parameters.get_central_all()
GF = par['GF']
sq2 = sqrt(2)
	

def wc_Z(lambdaQ, MZp):
	"Wilson coefficients as functions of Z' couplings"
	alpha = get_alpha(par, MZp)['alpha_e']
	return {
            'C9_bsmumu': -pi/(sq2* GF * MZp**2 *alpha) * lambdaQ/ckm.xi('t', 'bs')(par),
            'C10_bsmumu': pi/(sq2* GF * MZp**2 *alpha) * lambdaQ/ckm.xi('t', 'bs')(par),
            'CVLL_bsbs': - (lambdaQ/MZp )**2 #flavio defines the effective Lagrangian as L = CVLL (bbar gamma d)^2, with NO prefactors
	}

def wc_LQ(yy, MS):
	"Wilson coefficients as functions of Leptoquark couplings"
	alpha = get_alpha(par, MS)['alpha_e']
	return {
		'C9_bsmumu': pi/(sq2* GF * MS**2 *alpha) * yy/ckm.xi('t', 'bs')(par),
		'C10_bsmumu': -pi/(sq2* GF * MS**2 *alpha) * yy/ckm.xi('t', 'bs')(par),
		'CVLL_bsbs': - (yy/MS)**2 * 5 /(64*pi**2) #flavio defines the effective Lagrangian as L = CVLL (bbar gamma d)^2, with NO prefactors
	}


observables = [ ('BR(Bs->mumu)',), ('BR(B0->mumu)',), ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0), ('<Rmue>(B+->Kll)', 1.0, 6.0), ('<P4p>(B0->K*mumu)', 1.1, 6.0) , ('<P5p>(B0->K*mumu)', 1.1, 6.0), ('DeltaM_s',), ('S_psiphi',) ]
exp = [3.03448275862069e-09, 3.620689655172414e-10, 0.652542372881356, 0.6813559322033899, 0.745, 0.07, 0.01, 1.1688e-11, -0.021]
uncTot = [1.03797015913348e-09, 2.0431630348780073e-10, 0.09887068577171323, 0.10385498758186626, 0.08884998657827543, 0.3412395073412979, 0.23268144369851448, 1.0645216693544418e-12, 0.030144582822607187]
def calc_obs():
	"Re-obtain the experimental values and their uncertainties within the SM"
	exp = []
	uncExp = []
	expD = {}
	uncExpD = {}
	unc = []
	uncTot = []
	measurements = ['LHCb Bs->mumu 2017', 'CMS Bs->mumu 2013', 'LHCb RK* 2017', 'LHCb B->Kee 2014', 'LHCb B->K*mumu 2015 P 1.1-6', 'ATLAS B->K*mumu 2017 P4p', 'ATLAS B->K*mumu 2017 P5p', 'HFAG osc summer 2015', 'SLAC HFLAV 2018']
	
	for m in measurements:
		expD.update(flavio.Measurement[m].get_central_all())
		uncExpD.update(flavio.Measurement[m].get_1d_errors_random())
	for obs in observables:
		unc += [flavio.sm_uncertainty(*obs)]
		if len(obs)==1:
			obs = obs[0]
		exp += [expD[obs]]
		uncExp += [uncExpD[obs]]
		uncTot += [sqrt(unc[-1]**2 + uncExpD[obs]**2) ]

	
def chicalc(l , M, wc):
	"Chi-squared statistic for the fit"
	wcObj = Wilson(wc(l, M), scale=M, eft='WET', basis='flavio')
	chi2 = 0
	for o in range(0, len(observables)):
		chi2 += ((flavio.np_prediction(observables[o][0], wcObj, *observables[o][1:]) - exp[o] ))**2/(uncTot[o]**2 )
	return chi2

def chicalcRK(l , M, wc):
	"Chi-squared statistic for the fit to RK(*)-related observables"
	wcObj = Wilson(wc(l, M), scale=M, eft='WET', basis='flavio')
	chi2 = 0
	for o in range(0, len(observables)-2):
		chi2 += ((flavio.np_prediction(observables[o][0], wcObj, *observables[o][1:]) - exp[o] ))**2/(uncTot[o]**2 )
	return chi2

def chicalcBs(l , M, wc):
	"Chi-squared statistic for the fit to Bs-mixing-related observables"
	wcObj = Wilson(wc(l, M), scale=M, eft='WET', basis='flavio')
	chi2 = 0
	for o in (-2,-1):
		chi2 += ((flavio.np_prediction(observables[o][0], wcObj, *observables[o][1:]) - exp[o] ))**2/(uncTot[o]**2 )
	return chi2

def values(l, M, wc, ps=True):
	"Values of the observables in a NP scenario. Now in Markdown style"
	wcObj = Wilson(wc(l, M), scale=M, eft='WET', basis='flavio')
	print('|C9_bsmumu\t|' + str(wc(l, M)['C9_bsmumu']) + '|')
	print('|CVLL_bsbs\t|' + str(wc(l, M)['CVLL_bsbs']) + '|')
	for o in range(0, len(observables)):
		if o == len(observables) - 2 and ps:
			print('|' + observables[o][0] + '\t|' + str(roundC(flavio.np_prediction(observables[o][0], wcObj, *observables[o][1:])/flavio.sm_prediction('DeltaM_s')*20.01 ) ) + '|')
		else:
			print('|' + observables[o][0] + '\t|' + str(roundC(flavio.np_prediction(observables[o][0], wcObj, *observables[o][1:]))  ) + '|')


def makefit_imag(wc, stepM, stepL, maxM, maxL, filename, minM=0.1, minL=0):
	'''
	Fit and print results to file
	wc: Function that computes Wilson Coefficients from the model parameters (e.g. wc_Z, wc_LQ)
	Imaginary couplings
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
			if l == 0:
				chi += [(chiRK0, chiBs0, chitot0)]
			else:
				chiRK = chicalcRK(l*stepL*1j, M*stepM*1000, wc)
				chiBs = chicalcBs(l*stepL*1j, M*stepM*1000, wc)
				chitot = chiRK + chiBs
				chi += [(chiRK, chiBs, chitot)]
		f = open(filename, 'at')
		i = 0		
		for l in range(int(minL/stepL), numL+1):
			WCs = wc(l*stepL*1j, M*stepM*1000)
			C9 = WCs['C9_bsmumu']
			CVLL = WCs['CVLL_bsbs']
			f.write(str(M*stepM) + '\t' + str(l*stepL) + '\t'  + str(C9.imag) + '\t' + str(CVLL) + '\t' + str(chi[i][0]) + '\t' + str(chi[i][1]) + '\t' + str(chi[i][2]) + '\n' )
			i = i+1
		f.close() 

def makefit_real(wc, stepM, stepL, maxM, maxL, filename, minM=0.1, minL=0):
	'''
	Fit and print results to file
	wc: Function that computes Wilson Coefficients from the model parameters (e.g. wc_Z, wc_LQ)
	Real couplings
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
			if l == 0:
				chi += [(chiRK0, chiBs0, chitot0)]
			else:
				chiRK = chicalcRK(l*stepL, M*stepM*1000, wc)
				chiBs = chicalcBs(l*stepL, M*stepM*1000, wc)
				chitot = chiRK + chiBs
				chi += [(chiRK, chiBs, chitot)]
		f = open(filename, 'at')
		i = 0		
		for l in range(int(minL/stepL), numL+1):
			WCs = wc(l*stepL, M*stepM*1000)
			C9 = WCs['C9_bsmumu']
			CVLL = WCs['CVLL_bsbs']
			f.write(str(M*stepM) + '\t' + str(l*stepL) + '\t'  + str(C9.real) + '\t' + str(CVLL) + '\t' + str(chi[i][0]) + '\t' + str(chi[i][1]) + '\t' + str(chi[i][2]) + '\n' )
			i = i+1
		f.close() 

def makefit_complex(wc, stepM, stepL, maxM, maxL, filename, minM=0.1, minL=0):
	'''
	Fit and print results to file
	wc: Function that computes Wilson Coefficients from the model parameters (e.g. wc_Z, wc_LQ)
	Complex couplings
	stepM, maxM in TeV
	'''
	print('There we go!')
	chiRK0 = chicalcRK(0, 5000, wc)
	chiBs0 = chicalcBs(0, 5000, wc)
	chitot0 = chiRK0 + chiBs0
	numM = int(maxM/stepM)
	numL = int(maxL/stepL)
	for M in range(int(minM/stepM), numM+1):
		for lR in range(int(minL/stepL), numL+1):
			chi = []
			for lI in range(int(minL/stepL), numL+1):
				if (lR == 0) and (lI==0):
					chi += [(chiRK0, chiBs0, chitot0)]
				else:
					chiRK = chicalcRK( (lR + 1j*lI)*stepL, M*stepM*1000, wc)
					chiBs = chicalcBs( (lR + 1j*lI)*stepL, M*stepM*1000, wc)
					chitot = chiRK + chiBs
					chi += [(chiRK, chiBs, chitot)]
			f = open(filename, 'at')
			i = 0		
			for lI in range(int(minL/stepL), numL+1):
				WCs = wc( (lR + 1j*lI)*stepL, M*stepM*1000)
				C9 = WCs['C9_bsmumu']
				CVLL = WCs['CVLL_bsbs']
				f.write(str(M*stepM) + '\t' + str(lR*stepL) + '\t' + str(lI*stepL) + '\t' + str(C9) + '\t' + str(CVLL) + '\t' + str(chi[i][0]) + '\t' + str(chi[i][1]) + '\t' + str(chi[i][2]) + '\n' )
				i = i+1
			f.close() 

def makefit_mass(wc, stepM, stepL, maxM, numL, filename, minM=0.1, minL=0):
	'''
	Fit and print results to file
	wc: Function that computes Wilson Coefficients from the model parameters (e.g. wc_Z, wc_LQ)
	Only finds best fits for a fixed mass, and discards any other datapoint
	stepM, maxM in TeV
	'''
	print('There we go!')
	numM = int(maxM/stepM)
	prev = max(minL, stepL)
	for M in range(int(minM/stepM), numM+1):
		chi = []
		for lambd in range(0, numL):
			l = prev + stepL*lambd
			chiRK = chicalcRK(l*1j, M*stepM*1000, wc)
			chiBs = chicalcBs(l*1j, M*stepM*1000, wc)
			chitot = chiRK + chiBs
			chi += [chitot]
		f = open(filename, 'at')
		posmin = chi.index(min(chi))
		prev += stepL*posmin
		WCs = wc(prev*1j, M*stepM*1000)
		C9 = WCs['C9_bsmumu']
		CVLL = WCs['CVLL_bsbs']
		f.write(str(M*stepM) + '\t' + str(prev) + '\t'  + str(C9.imag) + '\t' + str(CVLL.real) + '\t' + str(chi[posmin]) + '\n' )
		f.close() 
