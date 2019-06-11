import os
import wget
import math
import numpy
import random

random.seed(0)

def normalize(x):
	try: return int(x)
	except ValueError as e: return x

def centiles(l, percentiles):
	return {l[int(len(l)*percentile)] for percentile in percentiles}

def discriminant(sel,threshold):
	def f(l): return 1 if l[sel] <= threshold else -1
	f.__name__ = "x["+str(sel)+"] <= " + str(threshold)
	return f

def onevsall(dataset, clss):
	return [sample[1:-1]+[1] if clss==sample[-1] else sample[1:-1]+[-1] for sample in dataset]

def learner(T, training_set, ranges):

	m 		   = len(training_set)
	weights    = []
	predictors = []
	prbs       = [1/m  for i in range(m)]
	flat_idxs = [(att,thr) for att in range(len(ranges)) for thr in ranges[att] if thr not in [max(ranges[att])]]

	for i in range(T):

		predictors.append(discriminant(*random.choice(flat_idxs)))
		prb_right  = sum([p for x,p in zip(training_set, prbs) if predictors[-1](x) == x[-1]])
		prb_wrong  = 1 - prb_right

		weights.append(0.5*math.log(prb_right/prb_wrong))
		
		mean = sum([prbs[t]*math.exp(-weights[i]*predictors[i](x)*x[-1])    for t,x in enumerate(training_set)])
		prbs = [(prbs[t]*math.exp(-weights[i]*predictors[i](x)*x[-1]))/mean for t,x in enumerate(training_set)]

		if i in [1,2,5,10,50,100,200,300,500,1000,1500,2000,3000,5000,7000,10000-1]:
			# test ###
			results = []
			for x in training_set:
				predictions = numpy.array([predictor(x) for predictor in predictors])
				result      = 1 if sum([w*p for w,p in zip(weights, predictions)]) >= 0  else -1
				results.append((result, x[-1]))

			print("acc:",sum([1 for tpl in results if tpl[0] == tpl[1]])/m,", i:",i)


# parameter settings ###
step         = 0.01
dataset_path = "./data/forest_dataset"

# data preprocessing ###
classes  = list(numpy.arange(1,8,1))
steps 	 = list(numpy.arange(0,1,step))
header   = [[normalize(x) for x in line.split(',')] for line in open(dataset_path).read().split()][0 ] 
data     = [[normalize(x) for x in line.split(',')] for line in open(dataset_path).read().split()][1:]
ranges   = [sorted(list(set(column))) for column in zip(*data)]
cents    = [centiles(rng, steps) for rng in ranges][1:-1]

training1 = onevsall(data,1)

print(cents)
learner(10000, training1, cents)