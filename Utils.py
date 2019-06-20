

class Utils:
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


	def onevsall(dataset, clss):
		''' from a dataset get its version of one vs all, also trims the index.
			Args: dataset    dataset to change.
			      clss       class that is one.
		'''

		return [sample[1:-1]+[1] if clss==sample[-1] else sample[1:-1]+[-1] for sample in dataset]


	def sgn(x):
		''' given x return its sign -1 or +1'''
		return -1 if x < 0 else +1

