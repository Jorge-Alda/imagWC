import texfig # download from https://github.com/nilsleiffischer/texfig
import flavio
import flavio.plots
import flavio.statistics.fits
import matplotlib.pyplot as plt

def wc(C9, C10):
	'''
	Wilson coeffients settings
	'''
	return {
		'C9_bsmumu': C9*1j,
		'C10_bsmumu': C10*1j,
	}

fast_fit = flavio.statistics.fits.FastFit(
                name = "C9-C10 SMEFT fast fit",
                observables = [ 'BR(Bs->mumu)', 'BR(B0->mumu)', ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0), ('<Rmue>(B+->Kll)', 1.0, 6.0), ('<P4p>(B0->K*mumu)', 1.1, 6.0) , ('<P5p>(B0->K*mumu)', 1.1, 6.0) ],
                fit_wc_function = wc,
                input_scale = 4.8,
                include_measurements = ['LHCb Bs->mumu 2017', 'CMS Bs->mumu 2013', 'LHCb RK* 2017', 'LHCb B->Kee 2014', 'LHCb B->K*mumu 2015 P 1.1-6', 'ATLAS B->K*mumu 2017 P4p', 'ATLAS B->K*mumu 2017 P5p'],
            )

fast_fit.make_measurement(threads=2)


def plot():
	'''
	Plots the allowed regions in the C9-C10 plane for imaginary Wilson coefficients
	'''
	fig = texfig.figure()
	opt = dict(x_min=-2, x_max=2, y_min=-2, y_max=2, n_sigma=(1,2), interpolation_factor=5)
	flavio.plots.likelihood_contour(fast_fit.log_likelihood, col=0, **opt, threads=2)
	#flavio.plots.flavio_branding(y=0.07, x=0.05) #crashes LaTeX
	plt.gca().set_aspect(1)
	plt.axhline(0, c='k', lw=0.2)
	plt.axvline(0, c='k', lw=0.2)
	plt.plot(0.31, 0.29, marker='x') #compute best fit first!
	plt.xlabel(r'$\mathrm{Im} C_9$')
	plt.ylabel(r'$\mathrm{Im} C_{10}$')
	texfig.savefig('fitim')


def best_fit(x0=[0.3,0.3]):
	'''
	Computes the best fit starting at point x0 = [Im C9, Im C10]
	'''
	bf_global = fast_fit.best_fit(x0)
	print('Global: C9='+str(bf_global['x'][0])+ 'i\tC10='+str(bf_global['x'][1])+'i\nchi2 = ' + str(bf_global['log_likelihood']))

