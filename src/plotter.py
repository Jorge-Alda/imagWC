import texfig
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

def triang(X):
	'''
	Interpolation by triangulation used to plot contour levels
	'''
	trg = tri.Triangulation(X[0], X[1])
	refiner = tri.UniformTriRefiner(trg)
	trg_ref, z_ref = refiner.refine_field(X[2], subdiv=3)
	return [trg_ref, z_ref]

def drawplot(filein, fileout):
	'''
	Draw the plot using shaded areas for the global fit and contour lines for Bs-only and RK-only fits
	'''
	fig = texfig.figure()

	trgtot_ref, ztot_ref = triang(readfile(filein, -1, 12))
	plt.tricontourf(trgtot_ref, ztot_ref, levels = [0.0, 1.0, 4.0, 9.0], colors = ('#008000', '#00FF00', '#BFFF80'))
	trgtot_Bs, ztot_Bs = triang(readfile(filein, -2, 12))
	plt.tricontour(trgBs_ref, zBs_ref, levels = [1.0, 4.0], colors = 'b', linestyles = ('solid', 'dashed'))
	trgRK_ref, zRK_ref = triang(readfile(filein, -3, 12))
	plt.tricontour(trgRK_ref, zRK_ref, levels = [1.0, 4.0], colors = 'r', linestyles = ('solid', 'dashed'))

	plt.xlabel(r"$M_{S_3} [\mathrm{TeV}]$")
	plt.ylabel(r'$\mathrm{Im}\ y_{32}^{QL} y_{32}^{QL*}$')
	#axes = plt.gca()
	#axes.set_ylim([0, 1.5])
	texfig.savefig(fileout)




#Including the plot in LaTex doc:
#% in the preamble
#\usepackage{pgf}
#% somewhere in your document
#\input{WCLQ.pgf}

