import texfig # https://github.com/knly/texfig
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from flavio.statistics.functions import delta_chi2
import flavio.plots
from matplotlib.patches import Rectangle

def readfile(filename, col, threshold, offset=0):
	'''
	Reads the data in file and saves it in three lists (first column, second column, and column number col) if the z-value is under the threshold
	'''
	f = open(filename, 'rt')
	xtot = []
	ytot = []
	ztot = []
	for l in f.readlines():
		znew = float(l.split('\t')[col]) - offset
		if znew < threshold:
			xtot += [float(l.split('\t')[0])]
			ytot += [float(l.split('\t')[1])]
			ztot += [znew]
	f.close()
	return [xtot, ytot, ztot]

def find_min(filename, col):
	'''
	Finds the minimum of a column in the file
	'''
	f = open(filename, 'r')
	minf = 500
	for l in f.readlines():
		if float(l.split('\t')[col]) < minf:
			minf = float(l.split('\t')[col])
			print(l.split('\t')[0] + '\t' + l.split('\t')[1] + '\t' + str(minf))
	f.close()

def triang(X):
	'''
	Interpolation by triangulation used to plot contour levels
	'''
	trg = tri.Triangulation(X[0], X[1])
	refiner = tri.UniformTriRefiner(trg)
	trg_ref, z_ref = refiner.refine_field(X[2], subdiv=3)
	return [trg_ref, z_ref]

def drawplot(filein, fileout, model, offtot=0, offBs=0, offRK=0):
	'''
	Draw the plot using shaded areas for the global fit and contour lines for Bs-only and RK-only fits
	'''
	fig = texfig.figure()

	trgtot, ztot = triang(readfile(filein, -1, 9, offtot))
	plt.tricontourf(trgtot, ztot, levels = [0.0, 1.0, 4.0, 9.0], colors = ('#00B400', '#00FF00', '#BFFF80'))
	trgBs, zBs = triang(readfile(filein, -2, 6, offBs))
	plt.tricontour(trgBs, zBs, levels = [1.0, 4.0], colors = 'b', linestyles = ('-', '--'))
	trgRK, zRK = triang(readfile(filein, -3, 6, offRK))
	plt.tricontour(trgRK, zRK, levels = [1.0, 4.0], colors = '#800000', linestyles = ('-.', ':'))

	if model == 'LQ':
		plt.xlabel(r"$M_{S_3} [\mathrm{TeV}]$")
		plt.ylabel(r'$\mathrm{Im}\ y^{QL}_{32}y^{QL*}_{32}$')
	if model == 'Z':
		plt.xlabel(r"$M_{Z'} [\mathrm{TeV}]$")
		plt.ylabel(r'$\mathrm{Im}\ \lambda^Q_{23}$')
	texfig.savefig(fileout)


def plot(fin, fout, x0=None):
	'''
	Read data from file and plot it in flavio-style
	'''
	f = open(fin, 'rt')
	_x = []
	_y = []
	for l in f.readlines():
		ls = l.split('\t')
		_x.append(float(ls[1]))
		_y.append(float(ls[2]))
	f.close()
	stepx = float('Inf')
	stepy = float('Inf')
	minx = min(_x)
	miny = min(_y)
	maxx = max(_x)
	maxy = max(_y)
	for i in range(0, len(_x)):
		if _x[i] != minx:
			stepx = min(stepx, _x[i]-minx)
	for i in range(0, len(_y)):
		if _y[i] != miny:
			stepy = min(stepy, _y[i]-miny)
	x, y = np.meshgrid(np.arange(minx, maxx, stepx), np.arange(miny, maxy, stepy))
	shape1, shape2 = x.shape
	f = open(fin, 'rt')
	i = 0
	zbs = np.zeros(x.shape)
	zDMs = np.zeros(x.shape)
	zACP = np.zeros(x.shape)	
	zglob = np.zeros(x.shape)
	for l in f.readlines():
		i1 = i%shape1
		i2 = i//shape1
		i += 1
		ls = l.split('\t')
		zbs[i1, i2] = float(ls[-4])
		zACP[i1, i2] = float(ls[-2])
		zDMs[i1, i2] = float(ls[-3])
		zglob[i1, i2] = float(ls[-1])
	f.close()
	zbs = zbs - np.min(zbs)
	zDMs = zDMs - np.min(zDMs)
	zACP = zACP - np.min(zACP)
	zglob = zglob - np.min(zglob)
	levels = [delta_chi2(n, dof=2) for n in (1,2)]
	plotbs = {'x': x, 'y':y, 'z':zbs, 'levels': levels, 'interpolation_factor':5, 'col':0, 'label':r'$b \to s \mu^+ \mu^-$'}
	plotDMs = {'x': x, 'y':y, 'z':zDMs, 'levels': levels, 'interpolation_factor':5, 'col':1, 'label':r'$\Delta B_s$'}
	plotACP = {'x': x, 'y':y, 'z':zACP, 'levels': levels, 'interpolation_factor':5, 'col':2, 'label':r'$A_{CP}^{\mathrm{mix}}$'}
	plotglob = {'x': x, 'y':y, 'z':zglob, 'levels': levels, 'interpolation_factor':5, 'col':3, 'label':'Global'}
	fig = texfig.figure()
	#fig = plt.figure()
	plt.xlim([-0.15, 0.15])
	plt.ylim([-0.15, 0.15])
	flavio.plots.contour(**plotbs)
	flavio.plots.contour(**plotDMs)
	flavio.plots.contour(**plotACP)
	flavio.plots.contour(**plotglob)
	plt.axhline(0, c='k', lw=0.2)
	plt.axvline(0, c='k', lw=0.2)
	if x0 is not None:
		plt.plot(x0[0], x0[1], marker='x', c='k')
	plt.xlabel(r'$\mathrm{Re}\ y^{QL}_{32} y^{QL*}_{22}$')
	plt.ylabel(r'$\mathrm{Im}\ y^{QL}_{32} y^{QL*}_{22}$')
	#plt.xlabel(r'$\mathrm{Re}\ \lambda^Q_{23}$')
	#plt.ylabel(r'$\mathrm{Im}\ \lambda^Q_{23}$')
	plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
	texfig.savefig(fout)

def errorplot(data, smdata, expdata, obslabels, leglabels, fout):
	'''
	Plots the model predictions as dots+errorbars, and SM predictions and experimental values as shaded rectangles
	'''
	fig = texfig.figure()
	nobs = len(obslabels)
	nhyp = len(leglabels)
	ax=plt.gca()
	plt.xlim([0, nobs+0.7])
	#plt.ylim([-0.055, 0.015])
	markers = ['o', '^', 's', 'o', '^', 's']
	colors = ['b', 'b', 'b', 'r', 'r', 'r']
	for o in range(0, nobs):
		for i in range(0, nhyp):
			if o==0:
				plt.plot(o+(i+1)/(nhyp+1), data[o][i][0], marker=markers[i], color=colors[i], label=leglabels[i])
			else:
				plt.plot(o+(i+1)/(nhyp+1), data[o][i][0], marker=markers[i], color=colors[i])
			plt.errorbar(o+(i+1)/(nhyp+1), data[o][i][0], yerr=data[o][i][1], color=colors[i])
			
		if o==0:
			ax.add_patch(Rectangle( (o, smdata[o][0]-smdata[o][1]), 1, 2*smdata[o][1], color='orange', alpha=0.7, label='SM'))		
			ax.add_patch(Rectangle( (o, expdata[o][0]-expdata[o][1]), 1, 2*expdata[o][1], color='green', alpha=0.7, label='Experimental'))
		else:
			ax.add_patch(Rectangle( (o, expdata[o][0]-expdata[o][1]), 1, 2*expdata[o][1], color='green', alpha=0.7))
			ax.add_patch(Rectangle( (o, smdata[o][0]-smdata[o][1]), 1, 2*smdata[o][1], color='orange', alpha=0.7))		
			
		
	ax.set_xticks(np.linspace(0.5, nobs+0.5, nobs+1) )
	ax.set_xticklabels(obslabels + [''])
	plt.legend()
	texfig.savefig(fout)

#Including the plot in LaTex doc:
#% in the preamble
#\usepackage{pgf}
#% somewhere in your document
#\input{WCLQ.pgf}

