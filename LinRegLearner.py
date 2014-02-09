import numpy as np

class LinRegLearner:
	'''
	Linear regression learner class
	'''

	def __init__(self):
		'''
		constructor
		@data is a numpy array trained
		'''

		self.data = None

	def addEvidence(self, Xdata, Ydata):
		'''
		add training be trained
		@Xdata: numpy arrays set as x columns
		@Ydata: numpy arrays set as y columns
		'''
		self.data = None

		Xrow_n = Xdata.shape[0]
		Xcol_n = Xdata.shape[1]
		Yrow_n = Ydata.shape[0]	
		if Yrow_n != Xrow_n:
			print " numbers of X data and Y data are not matching"

		data = np.zeros([Xrow_n, Xcol_n+1])
		data[:, 0:Xcol_n] = Xdata
		data[:, Xcol_n] = np.ones(Xrow_n).T # a column filled with 1 for the intercept transform frow a row

		cof = np.linalg.lstsq(data, Ydata)[0]
		print "the coefficient of linear regression model"
		print cof

		return cof

	def query(self, Xdata, cof):
		''' 
		get the value of test data on the linear regression line
		@Xdata: numpy data in x column for linear regresiion test
		@cof: the coefficient vector of the regression 
		'''

		Xrow_n = Xdata.shape[0]
		Xcol_n = Xdata.shape[1]

		#add the constant term to the X data
		Xdata_r = np.zeros([Xrow_n, Xcol_n+1])
		Xdata_r[:, 0:Xcol_n] = Xdata
		Xdata_r[:, Xcol_n] = np.ones(Xrow_n).T
		
		# Y data on the regression line
		Ydata = np.dot(Xdata_r, cof)

		return Ydata
