from AdaBoost import AdaBoost

class Measurers:

	def tp_tn_fp_fn(weights, predictors, samples):
		''' Returns (tp, tn, fp, fn) all in one. '''
		tp, tn, fp, fn = 0, 0, 0, 0
		for sample in samples:
			y_pred = AdaBoost.predict_AdaBoost(weights, predictors, sample)
			y_true = sample[-1]
			if   y_pred == +1 and y_true == +1: tp += 1
			elif y_pred == -1 and y_true == -1: tn += 1
			elif y_pred == -1 and y_true == +1: fn += 1
			elif y_pred == +1 and y_true == -1: fp += 1
		assert(tp+tn+fp+fn == len(samples))
		return tp, tn, fp, fn

	def recall(tp, tn, fp, fn):
		''' Returns recall value given tp, tn, fp, fn. '''
		return tp / max((tp + fn),0.0001)

	def precision(tp, tn, fp, fn):
		''' Returns precision value given tp, tn, fp, fn. '''
		return tp / max((tp + fp),0.0001)

	def accuracy(tp, tn, fp, fn):
		''' Returns accuracy value given tp, tn, fp, fn. '''
		return (tp + tn) / max((tp + fn + fp + tn),0.0001)
