import numpy as np
import os
import pandas as pd
import random
import time
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import json
import os.path

random.seed(time.time())


class StockDataSet(object):
	def __init__(self,
				 stock_sym,
				 input_size=1,
				 num_steps=30,
				 test_ratio=0.1,
				 normalized=True):
		self.stock_sym = stock_sym
		self.input_size = input_size
		self.num_steps = num_steps
		self.test_ratio = test_ratio	
		self.normalized = normalized
		
		data=None
		
		filename = 'data/'+self.stock_sym+'_dailyadj.csv'
		if os.path.isfile(filename):
			data = pd.read_csv(filename)
		else:
			ts = TimeSeries(key='XNL4', output_format='pandas')
			data, meta_data = ts.get_daily_adjusted(symbol=self.stock_sym, outputsize='full')
			data.to_csv(filename)

		#data['close'].plot()
		#plt.title('Intraday Times Series for the MSFT stock (1 day)')
		#plt.show()
		#raw_df = pd.read_csv(os.path.join("data", "%s.csv" % 'SP500'))
		#self.raw_seq = raw_df['Close'].tolist()
		
		# Merge into one sequence
		self.raw_seq = data['close'].tolist()
		#print(self.raw_seq)
		self.raw_seq = np.array(self.raw_seq)
		self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

	def info(self):
		return "StockDataSet [%s] train: %d test: %d" % (
			self.stock_sym, len(self.train_X), len(self.test_y))

	def _prepare_data(self, seq):
		# split into items of input_size
		seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
			   for i in range(len(seq) // self.input_size)]
			   
		if self.normalized:
			seq = [seq[0] / seq[0][0] - 1.0] + [
				curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

		# split into groups of num_steps
		# ex: if num_steps = 30, each element in X at time t represents a vector of 30 elements from t-31 to t-1 et Y is the element t (the prediction of the next day)  
		# where elements is a 1D vector (it only contains the close price for now but it could be more) 
		# X = [ [ [p(0)], [p(1)], ..., [p(30)] ], [ [p(1)], [p(2)], ..., [p(31)] ], ...] s.t. p(t) is thr close price a time t
		# Y = [ [p(31)]], [p(32)], ...] s.t. p(t) is thr close price a time t
		X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
		y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

		train_size = int(len(X) * (1.0 - self.test_ratio))
		train_X, test_X = X[:train_size], X[train_size:]
		train_y, test_y = y[:train_size], y[train_size:]
		return train_X, train_y, test_X, test_y

	# Create packet of size batch_size and shuffle the order of paccket randomly
	def generate_one_epoch(self, batch_size):
		num_batches = int(len(self.train_X)) // batch_size
		if batch_size * num_batches < len(self.train_X):
			num_batches += 1

		batch_indices = list(range(num_batches))
		random.shuffle(batch_indices)
		for j in batch_indices:
			batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
			batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
			assert set(map(len, batch_X)) == {self.num_steps}
			yield batch_X, batch_y
