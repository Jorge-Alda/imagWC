from math import pi, sqrt
from cmath import phase
from wilson import Wilson
import flavio
import flavio.statistics.fits
from flavio.physics.running.running import get_alpha
from flavio.physics import ckm
from flavio.classes import Observable, Prediction, Measurement, Parameter
from flavio.statistics.probability import NormalDistribution
from flavio.statistics.functions import pull
import yaml
from iminuit import Minuit
import numpy as np

par = flavio.default_parameters.get_central_all()
GF = par['GF']
sq2 = sqrt(2)

DMs_SM = Parameter('Delta M_S')
flavio.default_parameters.set_constraint('Delta M_S', '20.01 ± 1.25')

def myDeltaMS(wc_obj, par):
	DMs_SM = par['Delta M_S']
	if wc_obj.wc is None:
		return DMs_SM
	else:
		Cbs = - sq2/(4*GF*ckm.xi('t', 'bs')(par)**2)*wc_obj.wc['CVLL_bsbs']
		return DMs_SM*abs(1+Cbs/(1.3397e-3))

Observable('DMs')
Prediction('DMs', myDeltaMS)
m = Measurement('DMs exp')
m.add_constraint(['DMs'], NormalDistribution(17.757, 0.021) )
m2 = Measurement('SLAC HFLAV 2018')
m2.add_constraint(['S_psiphi'], NormalDistribution(0.021, 0.030144582822607187))

def wc_Z(lambdaQ, MZp):
	"Wilson coefficients as functions of Z' couplings"
	if MZp < 100:
		return {}
	alpha = get_alpha(par, MZp)['alpha_e']
	return {
            'C9_bsmumu': -pi/(sq2* GF * MZp**2 *alpha) * lambdaQ/ckm.xi('t', 'bs')(par),
            'C10_bsmumu': pi/(sq2* GF * MZp**2 *alpha) * lambdaQ/ckm.xi('t', 'bs')(par),
            'CVLL_bsbs': - 0.5* (lambdaQ/MZp )**2 #flavio defines the effective Lagrangian as L = CVLL (bbar gamma d)^2, with NO prefactors
	}

def wc_LQ(yy, MS):
	"Wilson coefficients as functions of Leptoquark couplings"
	if MS < 100:
		return {}
	alpha = get_alpha(par, MS)['alpha_e']
	return {
		'C9_bsmumu': pi/(sq2* GF * MS**2 *alpha) * yy/ckm.xi('t', 'bs')(par),
		'C10_bsmumu': -pi/(sq2* GF * MS**2 *alpha) * yy/ckm.xi('t', 'bs')(par),
		'CVLL_bsbs': - 0.5* (yy/MS)**2 * 5 /(64*pi**2) #flavio defines the effective Lagrangian as L = CVLL (bbar gamma d)^2, with NO prefactors
	}

def C9mu(cr, ci):
	return Wilson({'C9_bsmumu': cr + 1j*ci}, scale=4.2, eft='WET', basis='flavio')

def C10mu(cr, ci):
	return Wilson({'C10_bsmumu': cr + 1j*ci}, scale=4.2, eft='WET', basis='flavio')

def C910mu(cr, ci):
	return Wilson({'C9_bsmumu': cr + 1j*ci, 'C10_bsmumu': -cr-1j*ci}, scale=4.2, eft='WET', basis='flavio')

def C9pmu(cr, ci):
	return Wilson({'C9p_bsmumu': cr + 1j*ci}, scale=4.2, eft='WET', basis='flavio')

def C10pmu(cr, ci):
	return Wilson({'C10p_bsmumu': cr + 1j*ci}, scale=4.2, eft='WET', basis='flavio')

def C9e(cr, ci):
	return Wilson({'C9_bsee': cr + 1j*ci}, scale=4.2, eft='WET', basis='flavio')

def C10e(cr, ci):
	return Wilson({'C10_bsee': cr + 1j*ci}, scale=4.2, eft='WET', basis='flavio')

def C910e(cr, ci):
	return Wilson({'C9_bsee': cr + 1j*ci, 'C10_bsee': -cr-1j*ci}, scale=4.2, eft='WET', basis='flavio')

def C9pe(cr, ci):
	return Wilson({'C9p_bsee': cr + 1j*ci}, scale=4.2, eft='WET', basis='flavio')

def C10pe(cr, ci):
	return Wilson({'C10p_bsee': cr + 1j*ci}, scale=4.2, eft='WET', basis='flavio')

def CRe910(c9, c10):
	return {'C9_bsmumu': c9, 'C10_bsmumu':c10}

def CIm910(c9, c10):
	return {'C9_bsmumu': c9*1j, 'C10_bsmumu':c10*1j}

def C910Im(c9):
	return {'C9_bsmumu': c9*1j, 'C10_bsmumu':c9*1j}


wcs = [C9mu, C10mu, C910mu, C9pmu, C10pmu, C9e, C10e, C910e, C9pe, C10pe]


all_measurements=['LHCb Bs->mumu 2017', 'CMS Bs->mumu 2013', 'LHCb RK* 2017', 'LHCb B->Kee 2014', 'LHCb B->K*mumu 2015 P 1.1-6', 'ATLAS B->K*mumu 2017 P4p', 'ATLAS B->K*mumu 2017 P5p', 'HFAG osc summer 2015', 'CMS B->K*mumu 2017 P5p', 'LHCb B->K*mumu 2015 P 0.1-0.98', 'LHCb B->K*mumu 2015 P 1.1-2.5', 'LHCb B->K*mumu 2015 P 2.5-4', 'LHCb B->K*mumu 2015 P 4-6', 'LHCb B->K*mumu 2015 P 6-8', 'LHCb B->K*mumu 2015 P 11-12.5', 'LHCb B->K*mumu 2015 P 15-17', 'LHCb B->K*mumu 2015 P 17-19', 'LHCb B->K*mumu 2015 P 15-19', 'DMs exp', 'HFAG UT summer 2015', 'SLAC HFLAV 2018']
bins_P4ps_P5ps_LHCb=[( 0.1 , 0.98 ), ( 1.1 , 2.5 ), ( 2.5 , 4. ), ( 4. , 6. ), ( 6. , 8. ), ( 11. , 12.5 ), ( 15. , 17. ), ( 17. , 19. ), ( 1.1 , 6. ), ( 15. , 19. )]
bins_P5ps_CMS=[( 1 , 2),  ( 2 , 4.3), ( 4.3 , 6), ( 6 , 8.68), ( 10.09 , 12.86 ), ( 14.18 , 16 ), ( 16, 19 )]
bins_P4ps_P5ps_ATLAS=[( 0.04 , 2), ( 2 , 4), ( 4 , 6), ( 0.04 , 4), ( 1.1 , 6 ), ( 0.04 , 6 )]
bins_P4ps_P5ps_LHCb=[x for x in bins_P4ps_P5ps_LHCb if x[1]<=8.7 or x[0]>=14]
bins_P5ps_CMS=[x for x in bins_P5ps_CMS if x[1]<=8.7 or x[0]>=14]
bins_P4ps_P5ps_ATLAS=[x for x in bins_P4ps_P5ps_ATLAS if x[1]<=8.7 or x[0]>=14]


observables = []
for x in bins_P4ps_P5ps_LHCb + bins_P5ps_CMS + bins_P4ps_P5ps_ATLAS:
	observables += [( '<P5p>(B0->K*mumu)', ) + x]

for x in bins_P4ps_P5ps_LHCb  + bins_P4ps_P5ps_ATLAS:
	observables += [( '<P4p>(B0->K*mumu)', ) + x]

#observables += ['BR(B0->mumu', 'DMs', 'S_psiphi', ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0), ('<Rmue>(B+->Kll)', 1.0, 6.0) , 'BR(Bs->mumu)' ]
observables += [('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0), ('<Rmue>(B+->Kll)', 1.0, 6.0)]
observables = list(set(observables))


def save_observables(filename):
	obslist = []
	for j in all_measurements:
		meas1=flavio.Measurement[j].get_central_all()
		for k in meas1.keys():
			if k in observables:
				exp = meas1[k]
				err_exp = flavio.Measurement[j].get_1d_errors_random()[k]
				if isinstance(k, tuple):
					err_th = flavio.sm_uncertainty(k[0], q2min=k[1], q2max=k[2])
				else:
					err_th = flavio.sm_uncertainty(k)
				err = sqrt(err_th**2 + err_exp**2)
				obslist.append({'obs': k, 'central': exp, 'error': err})
	f = open(filename, 'w')
	yaml.dump(obslist, f)
	f.close()

def read_observables(filename):
	global obslist
	global nobs
	f = open(filename, 'r')
	obslist = yaml.load(f)
	f.close()
	nobs = len(obslist)

def chi2(wc=None):
	chi = 0
	chiM = 0
	chiACP = 0
	for i in range(0, nobs):
		if isinstance(obslist[i]['obs'], tuple):
			if wc is None:
				th = flavio.sm_prediction(obslist[i]['obs'][0], q2min=obslist[i]['obs'][1], q2max=obslist[i]['obs'][2])
			else:
				th = flavio.np_prediction(obslist[i]['obs'][0], wc_obj=wc, q2min=obslist[i]['obs'][1], q2max=obslist[i]['obs'][2])
		else:
			if wc is None:
				th = flavio.sm_prediction(obslist[i]['obs'])
			else:
				th = flavio.np_prediction(obslist[i]['obs'], wc_obj=wc)

		chiterm = (th - obslist[i]['central'])**2/obslist[i]['error']**2
		chi += chiterm
		if obslist[i]['obs'] == 'DMs':
			chiM += chiterm
		if obslist[i]['obs'] ==  'S_psiphi':
			chiACP += chiterm
	return (chi-chiM-chiACP, chiM, chiACP, chi)

def chi2_budget(wc=None):
	chi = []
	for i in range(0, nobs):
		if isinstance(obslist[i]['obs'], tuple):
			if wc is None:
				th = flavio.sm_prediction(obslist[i]['obs'][0], q2min=obslist[i]['obs'][1], q2max=obslist[i]['obs'][2])
			else:
				th = flavio.np_prediction(obslist[i]['obs'][0], wc_obj=wc, q2min=obslist[i]['obs'][1], q2max=obslist[i]['obs'][2])
		else:
			if wc is None:
				th = flavio.sm_prediction(obslist[i]['obs'])
			else:
				th = flavio.np_prediction(obslist[i]['obs'], wc_obj=wc)

		chi.append( (th - obslist[i]['central'])**2/obslist[i]['error']**2)
	return chi

def define_fit(wc, M=4.2):
	fast_fit = flavio.statistics.fits.FastFit(
                name = 'Global fit',
		observables = observables,
                fit_wc_function = wc,
                input_scale = M,
		nuisance_parameters = 'all'
        )
	fast_fit.make_measurement(threads=4)
	return fast_fit

def predictions(filename):
	f = open(filename, 'at', buffering=1)
	chiSM = chi2()
	obscalc = [('<Rmue>(B+->Kll)', 1.0, 6.0), ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0)]
	for w in wcs:
		f.write(w.__name__ + '\n=================\n')
		chi = lambda cr, ci: chi2(w(cr, ci))
		m = Minuit(chi, cr=0, ci=0, error_cr=0.01, error_ci=0.01, errordef=1, print_level=0)
		m.migrad()
		f.write('\tBest fit: ' + str(m.values[0]) + ' + ' + str(m.values[1]) + 'i\n') 
		chibf = m.fval
		f.write('\tPull (sqrt): ' + str(sqrt(chiSM-chibf)) + '\n')
		f.write('\tPull (sigma): ' + str(pull(chiSM-chibf, 2)) + r' \sigma' + '\n')
		f.write('\tChi2/dof: ' + str(chibf/(nobs-2)) + '\n')
		#m.minos()
		xr_centr = m.values[0]
		xi_centr = m.values[1]
		wcObj = w(xr_centr, xi_centr)
		cont = m.mncontour('cr', 'ci', numpoints=40)[2]
		for o in range(0, len(obscalc)):
			obs_centr = flavio.np_prediction(obscalc[o][0], wcObj, *obscalc[o][1:])
			obs_max = obs_min = obs_centr			
			for i in range(0, len(cont) ):
				wcObj = w(*cont[i])
				obs_max = max(obs_max, flavio.np_prediction(obscalc[o][0], wcObj, *obscalc[o][1:]))
				obs_min = min(obs_min, flavio.np_prediction(obscalc[o][0], wcObj, *obscalc[o][1:]))
			f.write('\t' + str(obscalc[o]) + ': ' + str(obs_centr) + ' + '+ str(obs_max - obs_centr) + ' - ' + str(obs_centr - obs_min) + '\n')
		f.write('\n\n')
	f.close() 

def makefit_complex(wc, rangeR, rangeI, rangeM, filename):
	'''
	Fit and print results to file
	wc: Function that computes Wilson Coefficients from the model parameters (e.g. wc_Z, wc_LQ)
	Complex couplings
	stepM, maxM in TeV
	'''
	chi0 = chi2()
	for M in rangeM:
		f = open(filename, 'at')
		#f.write(str(M) + '\t' + '0' + '\t' + '0' +  '\t' + str(chi0[0]) +  '\t' + str(chi0[1])  +  '\t' + str(chi0[2]) +  '\t' + str(chi0[3]) +  '\n' )
		for lR in rangeR:
			chi = []
			for lI in rangeI:
				if (lR==0) and (lI==0):
					chitot = chi0
				else:
					wcobj = Wilson(wc(lR+lI*1j, M*1000), scale=M*1000, eft='WET', basis='flavio')
					chitot = chi2(wcobj)
				f.write(str(M) + '\t' + str(lR) + '\t' + str(lI) +  '\t' + str(chitot[0]) +  '\t' + str(chitot[1])  +  '\t' + str(chitot[2]) + '\t' + str(chitot[3]) +  '\n' )
		f.close()

def predmodel(wc, l0, M):
	chi = lambda lr, li: chi2(Wilson(wc(lr+li*1j, M*1000), scale=M*1000, eft='WET', basis='flavio'))[-1]
	m = Minuit(chi, lr=l0[0], li=l0[1], error_lr=0.001, error_li=0.001, errordef=1, print_level=0)
	m.migrad()
	cont = [(l0[0]+m.errors[0], l0[1]), (l0[0]-m.errors[0], l0[1]), (l0[0], l0[1]+m.errors[1]), (l0[0], l0[1]-m.errors[1])]
	#obs0 = [('<Rmue>(B+->Kll)', 1.0, 6.0), ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0), ('DMs',), ('S_psiphi',)]
	obs0 = [('S_psiphi',)]
	for ob in obs0:
		lim = []
		if isinstance(ob, tuple):
			centr = flavio.np_prediction(ob[0], Wilson(wc(l0[0]+l0[1]*1j, M*1000), scale=M*1000, eft='WET', basis='flavio'), *ob[1:])
			for p in cont:
				lim.append(flavio.np_prediction(ob[0], Wilson(wc(p[0]+p[1]*1j, M*1000), scale=M*1000, eft='WET', basis='flavio'), *ob[1:]))
		else:
			centr = flavio.np_prediction(ob, Wilson(wc(l0[0]+l0[1]*1j, M*1000), scale=M*1000, eft='WET', basis='flavio'))
			for p in cont:
				lim.append(flavio.np_prediction(ob, Wilson(wc(p[0]+p[1]*1j, M*1000), scale=M*1000, eft='WET', basis='flavio')))

		errorsup = max(lim) - centr
		errorinf = centr - min(lim)
		print(ob, ':\t', centr, ' + ', errorsup, ' - ', errorinf)

def run():
	import numpy as np
	read_observables('observables_ZLQ.yaml')
	makefit_complex(wc_LQ, np.linspace(-0.5, 0.5, 20), np.linspace(-0.5, 0.5, 30), np.linspace(4, 6, 20), 'bf_ZComp.dat')

def plot(fast_fit, x):
	'''
	Plots the allowed regions in the C9-C10 plane for imaginary Wilson coefficients
	'''
	import texfig
	import flavio.plots
	import matplotlib.pyplot as plt
	fig = texfig.figure()
	opt = dict(x_min=-2, x_max=2, y_min=-2, y_max=2, n_sigma=(1,2), interpolation_factor=5)
	flavio.plots.likelihood_contour(fast_fit.log_likelihood, col=0, **opt, threads=2)
	#flavio.plots.flavio_branding(y=0.07, x=0.05) #crashes LaTeX
	plt.gca().set_aspect(1)
	plt.axhline(0, c='k', lw=0.2)
	plt.axvline(0, c='k', lw=0.2)
	plt.plot(x[0], x[1], marker='x') #compute best fit first!
	plt.xlabel(r'$\mathrm{Im}\ C_9$')
	plt.ylabel(r'$\mathrm{Im}\ C_{10}$')
	texfig.savefig('fitIm_C9C10')

def chiM(wc, l, M):
	wcobj = Wilson(wc(l, M*1000), scale=M*1000, eft='WET', basis='flavio')
	chiM = 0
	for i in range(0, nobs):
		if obslist[i]['obs'] in ['DMs', 'S_psiphi']:
			th = flavio.np_prediction(obslist[i]['obs'], wcobj)
			chiM += (th - obslist[i]['central'])**2/obslist[i]['error']**2
	return chiM

def massdep(wc, rangeM, maxl, fout):
	f = open(fout, 'wt', buffering=1)
	for M in rangeM:
		chi = lambda l: chiM(wc, l*1j, M)
		m = Minuit(chi, l=0, error_l=0.001, limit_l=(0, maxl), errordef=1)
		m.migrad()
		f.write(str(M) + '\t' + str(m.values[0]) + '\n')
	f.close()
