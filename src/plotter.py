import texfig # https://github.com/knly/texfig
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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




#Including the plot in LaTex doc:
#% in the preamble
#\usepackage{pgf}
#% somewhere in your document
#\input{WCLQ.pgf}

