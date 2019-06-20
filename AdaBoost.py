import os
import wget
import math
import numpy
import random
from Utils import Utils

class AdaBoost:
	def predict(predictor, sample):
		''' A stump predictor is defined by a selector and a threshold.
			Returns:	The response of the respective test.
			Args: 		predictor a tuple composed of selector and a threshold.
		'''
		selector  = predictor[0]
		threshold = predictor[1]
		return +1 if sample[selector] <= threshold else -1


	def predict_AdaBoost(weights, predictors, sample):
		''' given weights, predictors and a sample returns the prediction.
			Args:   	weights			AdaBoost weights.
						predictors 		Predictors genereted from AdaBoost.
						sample 			A sample to be predicted.
			Returns: 	+1 positive prediction, -1 otherwise.'''
		return Utils.sgn(sum([w*AdaBoost.predict(p,sample) for w,p in zip(weights,predictors)]))


	def is_correct(prediction, sample):
		''' Returns True if the prediction is correct False otherwise. '''
		return True if prediction*sample[-1] == +1 else False


	def get_error(predictor, probabilities, training_set):
		''' Returns:	the error as the prb of getting the wrong answer on the tr.set.
			Args: 		predictor 		a tuple of a selector and a threshold.
						probabilities 	adaboost prbs for the samples.
						training_set	just samples.'''
		prb_right = sum([p for sample,p in zip(training_set, probabilities) if AdaBoost.is_correct(AdaBoost.predict(predictor,sample),sample)])
		prb_wrong = 1 - prb_right
		return prb_wrong



	def train(T, training_set, choicer, mesurer = None):
		''' trains a list of weights and predictors.
			Args: T              number of max predictor to train.
			      training_set   training set.
			      mesurer        a function that gets called at each iteration.
			      test_choicer   choice a test given the current predictor
		'''
		nsamples   = len(training_set)
		prbs       = [1/nsamples for i in range(nsamples)]  # starting prbs
		weights    = []
		predictors = []

		for i in range(T):

			predictors.append(choicer(prbs, training_set))
			if predictors[-1] == None: return weights, predictors

			error = AdaBoost.get_error(predictors[-1], prbs, training_set)

			# get the new weight ###
			weights.append(0.5 * math.log((1-error) / error))
			
			# calc the new probabilities ###
			prbs_tmp = [(prbs[t]*math.exp(-weights[i]*AdaBoost.predict(predictors[i],sample)*sample[-1])) for t,sample in enumerate(training_set)]
			prbs_sum = sum(prbs_tmp)                  # normalization factor.
			prbs     = [p/prbs_sum for p in prbs_tmp] # normalize, so that it sum to 1.

			if mesurer != None: mesurer(i, weights, predictors)

		return weights, predictors
