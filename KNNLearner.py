import numpy as np

class KNNLearner:
	'''
	machine learning KNN learner class
	'''

	def __init__(self, k):
		'''
		constructor
		@k: the numbers of nearest neighbors
		'''
		self.k = k;
		self.data = None

	def addEvidence(self, Xdata, Ydata):
		'''
		add data be trained
		@Xdata: numpy arrays set as x columns to be classfied
		@Ydata: numpy arrays set as y columns which is the function value of Xdata
		'''
		self.data = None
		Xrow_n = Xdata.shape[0]
		Xcol_n = Xdata.shape[1]
		Yrow_n = Ydata.shape[0]
		if Yrow_n != Xrow_n:
			print " numbers of X data and Y data are not matching"

		#print Xdata

		data = np.zeros([Xrow_n, Xcol_n+1])
		data[:, 0:Xcol_n] = Xdata
		data[:, Xcol_n] = Ydata[:, 0]
		
		self.data = data
		#if self.data == None:
		#	self.data = data
		#else:
		#	self.data = np.append(self.data, data, axis = 0) # add the data after the end of rows

	def query(self, Xdata):
		'''
		classify test data
		@Xdata: numpy data in x column for KNN test
		'''
		k = self.k
		#the number of rows
		row_n = Xdata.shape[0]
		#the number of columns
		col_n = Xdata.shape[1]
		#change test data type from string to float
		Xlearn = np.zeros([row_n, col_n])
		Xlearn[:, 0:col_n] = Xdata
		#print Xlearn
		
		row_train = self.data.shape[0]

		#store the Y which from the learner
		Ylearn = np.zeros([row_n, 1])

		ii = 0
		slf_Xdata = self.data[:, 0:col_n] # train X data
		for row in Xlearn:
			#print row
			#print slf_Xdata
			dis_temp = slf_Xdata - row
			dis = np.zeros([row_train, 1])
			for i in range(0, col_n):
				dis = dis + np.reshape(np.power(dis_temp[:,i], 2.0), (-1, 1))
			#print dis
			temp_data = np.zeros([row_train, 2]) # temp_data include the value of train data and the distances to test data
			temp_data[:, 0] = self.data[:, -1]
			temp_data[:, 1] = dis[:,0]

			temp_data = temp_data[np.argsort(temp_data[:,1])] # sort tha array by the dis column in ascending order

			for j in range(0, k):
				Ylearn[ii] = Ylearn[ii] + temp_data[j][0]
			Ylearn[ii] = Ylearn[ii] / k
			ii = ii + 1

		return Ylearn

		#XYdata = np.zeros([row_n, 2])
		#XYdata[:, 0] = Xdata
		#XYdata[:, 1] = Ylearn

		#return XYdata
