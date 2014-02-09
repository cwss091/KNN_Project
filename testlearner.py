import numpy as np
import matplotlib.pyplot as plt
import csv
import time

from KNNLearner import *
from LinRegLearner import *

def _csv_read(filename):
	'''
	import csv files which contains the data to train and test
	@filename: imported file's name
	'''

	reader = csv.reader(open(filename, 'rU'), delimiter = ',')
	Xdata = []
	Ydata = []
	for row in reader:
		Ydata.append(row[-1])
		Xdata.append(row[0:-1])
	
	Xdata = np.array(Xdata)
	Ydata = np.array(Ydata)

	return Xdata, Ydata

def RMSE(DTest, DLearn):
	'''
	Root-Mean-Square Error
	@DTest: tested correct data
	@Dlearn: data form learning
	'''
	row_queryr = float(DLearn.shape[0])
	RMSE = np.sum(np.power((DTest - DLearn), 2.0)) / row_queryr
	RMSE = np.sqrt(RMSE)
	return RMSE
	

def testlearner():
	'''
	test KNN and Linear regression learner
	'''

	Xdcp, Ydcp = _csv_read("data-classification-prob.csv")
	Xdrp, Ydrp = _csv_read("data-ripple-prob.csv") # the data in numpy array now is string instead of float
	
	#divide data for train and test
	dcp_row_N = Xdcp.shape[0]
	drp_row_N = Xdrp.shape[0]
	trainperct = 0.6 # data for training is 60% of total data
	dcp_trp = int(dcp_row_N * trainperct)
	drp_trp = int(drp_row_N * trainperct)
	#testperct = 1.0 - trainperct # data for test's percent 
	#data for training
	Xdcp_train = Xdcp[0:dcp_trp, :]
	Ydcp_train = np.zeros([dcp_trp, 1])
	Ydcp_train[:, 0] = Ydcp[0:dcp_trp]
	Xdrp_train = Xdrp[0:drp_trp, :]
	Ydrp_train = np.zeros([drp_trp, 1])
	Ydrp_train[:, 0] = Ydrp[0:drp_trp]
	#data for test (query)
	Xdcp_test = Xdcp[dcp_trp:dcp_row_N, :]
	Ydcp_test = np.zeros([dcp_row_N - dcp_trp, 1])
	Ydcp_test[:, 0] = Ydcp[dcp_trp:dcp_row_N]
	#Ydcp_test = [:, 0:col_n] = Xdata
	Xdrp_test = Xdrp[drp_trp:drp_row_N, :]
	Ydrp_test = np.zeros([drp_row_N - drp_trp, 1])
	Ydrp_test[:, 0] = Ydrp[drp_trp:drp_row_N]

	

	#KNN learner

	# result of KNN learn, rows records k, training time cost, query time cost, total time cost, RMSError and Correlation coeffient
	KNN_dcp_result = np.zeros([7, 50]) # result of data-classification-prob.csv
	KNN_drp_result = np.zeros([7, 50]) # result of data-ripple-prob.csv

	for k in range(1, 51):
		KNN_lner = KNNLearner(k)
		KNN_dcp_result[0][k-1] = k
		KNN_drp_result[0][k-1] = k
		
		# results of data-classification-prob.csv
		stime = time.time()
		KNN_lner.addEvidence(Xdcp_train, Ydcp_train)
		etime = time.time()
		KNN_dcp_result[1][k-1] = (etime - stime) / dcp_trp # training time cost

		stime = time.time()
		Ydcp_learn = KNN_lner.query(Xdcp_test)
		etime = time.time()
		KNN_dcp_result[2][k-1] = (etime - stime) / (dcp_row_N - dcp_trp) # query time cost

		KNN_dcp_result[3][k-1] = KNN_dcp_result[1][k-1] + KNN_dcp_result[2][k-1] # total time cost
		
		#print Ydcp_test
		#print Ydcp_learn
		KNN_dcp_result[4][k-1] = RMSE(Ydcp_test, Ydcp_learn) # Root-Mean-square error

		KNN_dcp_result[5][k-1] = np.corrcoef(Ydcp_learn.T, Ydcp_test.T)[0][1] # correlation coefficient

		Ydcp_osp = KNN_lner.query(Xdcp_train)
		KNN_dcp_result[6][k-1] = RMSE(Ydcp_train, Ydcp_osp) # the RMS error between in-sample and out-sample data
		
		# results of data-ripple-prob.csv
		stime = time.time()
		KNN_lner.addEvidence(Xdrp_train, Ydrp_train)
		etime = time.time()
		KNN_drp_result[1][k-1] = (etime - stime) / drp_trp # training time cost

		stime = time.time()
		Ydrp_learn = KNN_lner.query(Xdrp_test)
		etime = time.time()
		KNN_drp_result[2][k-1] = (etime - stime) / (drp_row_N - drp_trp) # query time cost

		KNN_drp_result[3][k-1] = KNN_drp_result[1][k-1] + KNN_drp_result[2][k-1] # total time cost

		KNN_drp_result[4][k-1] = RMSE(Ydrp_test, Ydrp_learn) # Root-Mean-Square error

		KNN_drp_result[5][k-1] = np.corrcoef(Ydrp_learn.T, Ydrp_test.T)[0][1] # correlation coefficient

		# insample and outsample error of ripple
		Ydrp_osp = KNN_lner.query(Xdrp_train)
		KNN_drp_result[6][k-1] = RMSE(Ydrp_train, Ydrp_osp) # the RMS error between in-sample and out-sample data

		#plot the predicted Y vesus actual Y when K = 3
		if k == 27:
			# plot the Y data of classification data
			plt.clf()
			fig = plt.figure()
			fig.suptitle('Y of classification data')
			#f1 = fig.add_subplot(2, 1, 1)
			plt.plot(Ydcp_test, Ydcp_learn, 'o', markersize = 5)
			plt.xlabel('Actual Y')
			plt.ylabel('Predicted Y')
			#f1.set_title('data-classcification-prob.csv')
			fig.savefig('classification_Y.pdf', format = 'pdf')

		if k == 3:
			# plot the Y data of ripple data
			#f2 = fig.add_subplot(2, 1, 2)
			plt.clf()
			fig = plt.figure()
			fig.suptitle('Y of ripple data')
			plt.plot(Ydrp_test, Ydrp_learn, 'o', markersize = 5)
			plt.xlabel('Actual Y')
			plt.ylabel('Predicted Y')
			#f2.set_title('data-ripple-prob.csv')
			fig.savefig('ripple_Y.pdf', format = 'pdf')

	print KNN_dcp_result[:, 2] #the result of k=3 for dcp.csv
	Kdcp_best_pos = np.argmax(KNN_dcp_result[5, :])	#the indices of the maximum correlation coeffiecient
	print KNN_dcp_result[:, Kdcp_best_pos]

	print KNN_drp_result[:, 2] #the result of k=3 for drp.csv
	Kdrp_best_pos = np.argmax(KNN_drp_result[5, :]) #the indices of the maximum correlation
	print KNN_drp_result[:, Kdrp_best_pos]

	#plot the correlation
	plt.clf()
	fig = plt.figure()
	plt.plot(KNN_dcp_result[0, :], KNN_dcp_result[5, :], 'r', label = 'Classification')
	plt.plot(KNN_drp_result[0, :], KNN_drp_result[5, :], 'b', label = 'Ripple')
	plt.legend()
	plt.xlabel('K')
	plt.ylabel('Correlation Coefficient')
	fig.savefig('Correlation_KNN.pdf', format = 'pdf')

	#plot the error between in sample and out-of-sample data
	plt.clf()
	fig = plt.figure()
	#f1 = fig.add_subplot(2, 1, 1)
	fig.suptitle('RMS error of classification data')
	plt.plot(KNN_dcp_result[0, :], KNN_dcp_result[4, :], 'or', label = 'out of sample')
	plt.plot(KNN_dcp_result[0, :], KNN_dcp_result[6, :], 'ob', label = 'in sample')
	#f1.axis([0:0.1:1.0]
	plt.legend(loc = 4)
	plt.xlabel('K')
	plt.ylabel('RMS Error')

	fig.savefig('classification-RMSE.pdf', format = 'pdf')
	#f1.set_title('data-classification-prob.csv')
	
	#f2 = fig.add_subplot(2, 1, 2)
	plt.clf()
	fig = plt.figure()
	fig.suptitle('RMS error of ripple data')
	plt.plot(KNN_drp_result[0, :], KNN_drp_result[4, :], 'or', label = 'out of sample')
	plt.plot(KNN_drp_result[0, :], KNN_drp_result[6, :], 'ob', label = 'in sample')
	#f2.axis([0:0.1:1.0]
	plt.legend(loc = 4)
	plt.xlabel('K')
	plt.ylabel('RMS Error')
	#f2.set_title('data-ripple-prob.csv')
	plt.savefig('ripple-RMSE.pdf', format = 'pdf')

	# plot the train time
	plt.clf()
	fig = plt.figure()
	plt.plot(KNN_dcp_result[0, :], KNN_dcp_result[1, :], 'r', label = 'Classification')
	plt.plot(KNN_drp_result[0, :], KNN_drp_result[1, :], 'b', label = 'Ripple')
	plt.legend(loc=1)
	plt.xlabel('K')
	plt.ylabel('train time / s')
	fig.savefig('traintime.pdf', format = 'pdf')

	# plot the query time
	plt.clf()
	fig = plt.figure()
	plt.plot(KNN_dcp_result[0, :], KNN_dcp_result[2, :], 'r', label = 'Classification')
	plt.plot(KNN_drp_result[0, :], KNN_drp_result[2, :], 'b', label = 'Ripple')
	plt.legend(loc=4)
	plt.xlabel('K')
	plt.ylabel('query time / s')
	fig.savefig('querytime.pdf', format = 'pdf')
	


	# Linear regression
	LR_lner = LinRegLearner()
	LR_dcp_result = np.zeros(5)	#Linear regression results of data-classification-prob.csv
	LR_drp_result = np.zeros(5) #Linear regression results of data-ripple-prob.csv
	
	# results of data-classification-prob.csv
	stime = time.time()
	dcp_cof = LR_lner.addEvidence(Xdcp_train, Ydcp_train)
	etime = time.time()
	LR_dcp_result[0] = (etime - stime) / dcp_trp# train time cost

	stime = time.time()
	Ydcp_LRL = LR_lner.query(Xdcp_test, dcp_cof)
	etime = time.time()
	LR_dcp_result[1] = (etime - stime) / (dcp_row_N - dcp_trp) # query time cost

	LR_dcp_result[2] = LR_dcp_result[0] + LR_dcp_result[1] # total time cost

	LR_dcp_result[3] = RMSE(Ydcp_test, Ydcp_LRL) # root-mean-square error

	LR_dcp_result[4] = np.corrcoef(Ydcp_test.T, Ydcp_LRL.T)[0][1] # correlation efficient

	print LR_dcp_result

	# results of data-ripple-prob.csv
	stime = time.time()
	drp_cof = LR_lner.addEvidence(Xdrp_train, Ydrp_train)
	etime = time.time()
	LR_drp_result[0] = (etime - stime) / drp_trp # train time cost

	stime = time.time()
	Ydrp_LRL = LR_lner.query(Xdrp_test, drp_cof)
	etime = time.time()
	LR_drp_result[1] = (etime - stime) / (drp_row_N - drp_trp) # query time cost

	LR_drp_result[2] = LR_drp_result[0] + LR_drp_result[1] # total time cost

	LR_drp_result[3] = RMSE(Ydrp_test, Ydrp_LRL) # root-mean-square error

	LR_drp_result[4] = np.corrcoef(Ydrp_test.T, Ydrp_LRL.T)[0][1] # correlation efficient
	
	print LR_drp_result
	

if __name__ == "__main__":
	testlearner()
