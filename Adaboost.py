import os
import wget
import math
import numpy
import random


def getPartitions(lst, npartitions):
	''' Given a list of elements partitionate it in npartitions.
		Args: lst           list of samples.
		      npartitions   no. of partition to obtain.
	'''
	plength = int(len(lst)/npartitions)
	return [lst[plength*i:plength*(i+1)] for i in range(npartitions)]


def flat(lst):
	''' Given a list of list, flats the first level.
		Args: lst    a list to be flatten.
	'''
	return [e for l in lst for e in l]


def toInt(x):
	''' try to get the int from a value 
	    Args: x    a string representing and int.
	'''
	try: return int(x)
	except ValueError as e: return x


def centiles(l, percentiles):
	''' return the set of centiles for each percentile 
	    Args: l            list of values. 
	          percentiles  list of percentile.
	'''
	return {l[int(len(l)*percentile)] for percentile in percentiles}


def predict(predictor, sample):
	''' A stump predictor is defined by a selector and a threshold.
		Returns:	The response of the respective test.
		Args: 		predictor a tuple composed of selector and a threshold.
	'''
	selector  = predictor[0]
	threshold = predictor[1]
	return +1 if sample[selector] <= threshold else -1


def onevsall(dataset, clss):
	''' from a dataset get its version of one vs all, also trims the index.
		Args: dataset    dataset to change.
		      clss       class that is one.
	'''

	return [sample[1:-1]+[1] if clss==sample[-1] else sample[1:-1]+[-1] for sample in dataset]


def sgn(x):
	''' given x return its sign -1 or +1'''
	return -1 if x < 0 else +1


def predict_AdaBoost(weights, predictors, sample):
	''' given weights, predictors and a sample returns the prediction.
		Args:   	weights			AdaBoost weights.
					predictors 		Predictors genereted from AdaBoost.
					sample 			A sample to be predicted.
		Returns: 	+1 positive prediction, -1 otherwise.'''
	return sgn(sum([w*predict(p,sample) for w,p in zip(weights,predictors)]))


def is_correct(prediction, sample):
	''' Returns True if the prediction is correct False otherwise. '''
	return True if prediction*sample[-1] == +1 else False


# def simple_random_choicer(tests):


def accuracy(weights, predictors, samples):
	''' Returns the accuracy for the givens samples.
		Args: 		weights      Adaboost weights.
		      		predictors   Adaboost predictors.
		      		samples      sample to test.
		Returns: 	A value in [0,1] representing the accuracy.
	'''
	predictions = [1 if is_correct(predict_AdaBoost(weights,predictors,sample),sample) else 0 for sample in samples]
	accuracy 	= sum(predictions)/len(predictions)
	return accuracy


def accuracy_wrapper(training_set, test_set, step, iterations):
	''' Generate an insider that test the predictor on a given test_set.
		Args: test_set    test set.
		      step        no. iteration from each test.
	'''
	def insider(iteration, weights, predictors):

		if iteration % step == 0 or iteration == iterations-1:
			# get no. of correctly classified samples ###
			print("acc {: <3.5f}, valacc: {: <3.5f}, iteration: {}".format(accuracy(weights, predictors, training_set),
																		   accuracy(weights, predictors, test_set    ), iteration))

	return insider


def Adaboost(T, training_set, ranges, mesurer = None):
	''' trains a list of weights and predictors.
		Args: T              number of max predictor to train.
		      training_set   training set.
		      ranges         list of thresholds, used to generate tests.
		      mesurer        a function that gets called at each iteration.
		      test_choicer   choice a test given the current predictor
	'''
	nsamples   = len(training_set)
	prbs       = [1/nsamples for i in range(nsamples)]  # starting prbs
	tests      = [(att,thr) for att in range(len(ranges)) for thr in ranges[att] 
				 if thr not in [max(ranges[att])]]      # remove max value (gets always the whole dataset)
	ntests     = len(tests)
	weights    = []
	predictors = []

	for i in range(T):

		choices = {i for i in range(ntests)}
		while True:

			choice    = random.choice(list(choices))    # random int from 0, 1,..., no.tests.
			predictor = tests[choice]				    # get a random predictor.
	
			# find the prb to be right of the predictor. ###
			prb_right = sum([p for x,p in zip(training_set, prbs) if predict(predictor,x) == x[-1]])
			prb_wrong = 1 - prb_right

			# if the predictor is not significant then drop it. ###
			if prb_right != 0.5: 
				predictors.append(predictor)
				break

			# if all possible predictors have been generated then return. ###
			choices.remove(choice)
			if len(choices) == 0: 
				print("terminated at iteration {}, lacking {}.".format(i, T-i))
				return weights, predictors


		# get the new weight ###
		weights.append(0.5 * math.log(prb_right / prb_wrong))
		
		# calc the new probabilities ###
		prbs_tmp = [(prbs[t]*math.exp(-weights[i]*predict(predictors[i],sample)*sample[-1])) for t,sample in enumerate(training_set)]
		prbs_sum = sum(prbs_tmp)                  # normalization factor.
		prbs     = [p/prbs_sum for p in prbs_tmp] # normalize, so that it sum to 1.

		if mesurer != None: mesurer(i, weights, predictors)

	return weights, predictors


if __name__ == '__main__':

	random.seed(0)

	# dataset download ###
	dataset_path = "./data/forest_dataset"
	if not os.path.isfile("./data/forest_dataset"):
		url = "https://www.dropbox.com/s/sr856cj0s2re4qp/forest-cover-type.csv?dl=1"
		wget.download(url)
		os.rename("./forest-cover-type.csv",dataset_path)


	# parameter settings ###
	step         = 0.001 					# step for percentiles generation
	T            = 10000                    # no. Adaboost step
	dataset_path = "./data/forest_dataset"  # dataset path
	npartitions  = 4						# np. partiotion for cross validation


	# data preprocessing ###
	classes  = list(numpy.arange(1,8,1))
	steps 	 = list(numpy.arange(0,1,step))
	header   = [[toInt(x) for x in line.split(',')] for line in open(dataset_path).read().split()][0 ] 
	data     = [[toInt(x) for x in line.split(',')] for line in open(dataset_path).read().split()][1:]
	ranges   = [sorted(list(set(column))) for column in zip(*data)]
	cents    = [centiles(rng, steps) for rng in ranges][1:-1]


	# training ###
	partitions = getPartitions(data, npartitions)
	for clss in classes:
		print("training for class {}.".format(clss))
		valacc = 0
		for i in range(npartitions):
			print("testing for partition {}.".format(i))
			training_set         = onevsall(flat(partitions[:i]+partitions[i+1:]), clss)
			test_set             = onevsall(partitions[i], clss)
			weights, predictors  = Adaboost(T, training_set, cents, mesurer = accuracy_wrapper(test_set, training_set, 10, T))
			current_valacc       = accuracy(weights, predictors, test_set)
			valacc 				+= current_valacc
			print("current validation accuracy for class {} is {: <3.5f}".format(clss, current_valacc))

		print("cross-validation accuracy for class {} is {: <3.5f}".format(clss, valacc/npartitions))
