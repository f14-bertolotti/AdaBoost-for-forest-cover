import os
import wget
import math
import numpy
import random

random.seed(0)

def normalize(x):
	try: return int(x)
	except ValueError as e: return x

def centile(l,percentile):
	return l[int(len(l)*percentile)]

def centiles(l, percentiles):
	return {centile(l,percentile) for percentile in percentiles}

def discriminant(sel,threshold):
	def f(l): return True if l[sel] <= threshold else False
	f.__name__ = "x["+str(sel)+"] <= " + str(threshold)
	return f

def onevsall(dataset, clss):
	return [sample[1:-1]+[1] if clss==sample[-1] else sample[1:-1]+[-1] for sample in dataset]


class stump:
	def __init__(self, test, training_set): 
		self.test     = test
		self.__name__ = test.__name__
		self.left     = None
		self.right    = None
		self.classify(training_set)
	def classify(self, training_set):
		self.left        = [x for x in training_set if     self.test(x)]
		self.right       = [x for x in training_set if not self.test(x)]
		self.prb_right   = sum([1 for x in training_set if self.predict(x) == x[-1]])/len(training_set)
		self.prb_wrong   = 1 - self.prb_right
		self.tails       = (self.prb_wrong - 0.5)**2
	def predict(self,sample):
		return 1 if self.test(sample) else -1

def learner(T, training_set, ranges):

	m 		   = len(training_set)
	weights    = []
	predictors = []
	prbs       = [1/m  for i in range(m)]
	

	for i in range(T):
		# print(i)
		# choose a test ###
		attribute     = random.choice(range(len(ranges)))
		# threshold     = max([(stump(discriminant(attribute, thr),training_set).tails,thr) for thr in list(ranges[attribute])])[1]

		threshold = random.choice(list(ranges[attribute]))
		test_func = discriminant(attribute, threshold)
		predictor = stump(test_func, training_set)
		while predictor.tails < 0.12:
			# print(predictor.tails)
			threshold = random.choice(list(ranges[attribute]))
			test_func = discriminant(attribute, threshold)
			predictor = stump(test_func, training_set)
		predictors.append(predictor)
		print(i)
		
		# next weight ###
		prb_right  = sum([p for x,p in zip(training_set, prbs) if predictors[i].predict(x) == x[-1]])
		prb_wrong  = 1 - prb_right
		weights.append(0.5*math.log(prb_right/prb_wrong))
		
		# next prbs ###
		mean = sum([prbs[t]*math.exp(-weights[i]*predictors[i].predict(x)*x[-1])    for t,x in enumerate(training_set)])
		prbs = [(prbs[t]*math.exp(-weights[i]*predictors[i].predict(x)*x[-1]))/mean for t,x in enumerate(training_set)]

		if (i+1) % 10 == 0:
			# test ###
			results = []
			for x in training_set:
				predictions = numpy.array([predictor.predict(x) for predictor in predictors])
				result      = 1 if sum([w*p for w,p in zip(weights, predictions)]) >= 0  else -1
				results.append((result, x[-1]))

			print("acc:",sum([1 for tpl in results if tpl[0] == tpl[1]])/m,", i:",i)



if __name__ == "__main__":
	# dataset download ###
	dataset_path = "./data/forest_dataset"
	if not os.path.isfile("./data/forest_dataset"):
		url = "https://www.dropbox.com/s/sr856cj0s2re4qp/forest-cover-type.csv?dl=1"
		wget.download(url)
		os.rename("./forest-cover-type.csv",dataset_path)


	# parameter settings ###
	step     = 0.001

	# data preprocessing ###
	classes  = list(numpy.arange(1,8,1))
	steps 	 = list(numpy.arange(0,1,step))
	header   = [[normalize(x) for x in line.split(',')] for line in open(dataset_path).read().split()][0 ] 
	data     = [[normalize(x) for x in line.split(',')] for line in open(dataset_path).read().split()][1:]
	ranges   = [sorted(list(set(column))) for column in zip(*data)]
	cents     = [centiles(rng, steps) for rng in ranges]
	
	# for t in range(1000):
	learner(10000, onevsall(data,3), cents[1:-1])
