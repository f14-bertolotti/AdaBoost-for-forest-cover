import os
import wget
import math
import numpy
import random
from Utils     import Utils
from Choicers  import Choicers
from Adaboost  import Adaboost
from Measurers import Measurers
from Insider   import Insider

if __name__ == '__main__':

	# fix the seed ###
	random.seed(42)

	# images directory ###
	if not os.path.isdir('./images'):
		os.mkdir('./images')

	# dataset download ###
	dataset_path = "./data/forest_dataset"
	if not os.path.isfile("./data/forest_dataset"):
		url = "https://www.dropbox.com/s/sr856cj0s2re4qp/forest-cover-type.csv?dl=1"
		wget.download(url)
		os.rename("./forest-cover-type.csv",dataset_path)


	# parameter settings ###
	dataset_path = "./data/forest_dataset"  # dataset path
	T            = 1001	         	        # no. Adaboost step
	step         = 0.001 					# step for percentiles generation
	prt_list     = [4]						# list of possible partition in cross validation


	# data preprocessing ###
	classes  = list(numpy.arange(1,8,1))
	steps 	 = list(numpy.arange(0,1,step))
	header   = [[Utils.toInt(x) for x in line.split(',')] for line in open(dataset_path).read().split()][0 ] 
	data     = [[Utils.toInt(x) for x in line.split(',')] for line in open(dataset_path).read().split()][1:]
	ranges   = [sorted(list(set(column))) for column in zip(*data)]
	cents    = [Utils.centiles(rng, steps) for rng in ranges][1:-1]
	tests    = [ [(sel,thr) for thr in cents[sel] ] for sel in range(len(cents)) ]     
	
	random.shuffle(data)

	test_data     = data[:len(data)//3 ]
	training_data = data[ len(data)//3:]
	best_algo     = None 
	best_val_acc  = 0

	# training ###
	for npartitions in prt_list:
		partitions = Utils.getPartitions(training_data, npartitions)

		for choicer, name in [(Choicers.best_choicer                (tests, 0.01), "best_with_threshold_"     ),
					   		  (Choicers.random_choicer              (tests, 0   ), "random_choice_"           ), 
					   		  (Choicers.random_choicer              (tests, 0.01), "random_with_threshold_"   ),
					   		  (Choicers.random_selector_best_choicer(tests, 0.5 ), "random_selctor_best_test_")]:

			print("training for {}".format(name))


			avg_best_on_classes = 0
			for j, clss in enumerate(classes):
				print("training for class {}.".format(clss))

				insider = Insider(clss,npartitions,name,10,T)

				for i in range(npartitions):
					print("testing for partition {}.".format(i))
					training_set         = Utils.onevsall(Utils.flat(partitions[:i]+partitions[i+1:]), clss)
					test_set             = Utils.onevsall(partitions[i]                              , clss)
					insider.training_set = training_set
					insider.test_set     = test_set
					weights, predictors  = AdaBoost.train(T, training_set, mesurer = insider, choicer = choicer)

				insider.save()
				insider.close()

				avg_best_on_classes += (insider.best_val_acc() - avg_best_on_classes)/(j+1)

			if best_val_acc  < avg_best_on_classes:
				best_val_acc = avg_best_on_classes
				best_algo    = (choicer,name)


	# best training ###
	test_acc = 0
	for j,clss in enumerate(classes):
		print("best predictor {} for class {}".format(best_algo[1], clss))
		insider              = Insider(clss,1,"best_predictor_",10,T)
		training_set         = Utils.onevsall(training_data, clss)
		test_set             = Utils.onevsall(test_data    , clss)
		insider.training_set = training_set
		insider.test_set     = test_set	
		weights, predictors  = AdaBoost.train(T, training_set, mesurer = insider, choicer = best_algo[0])
		test_acc += (insider.best_val_acc()-test_acc)/(j+1)
		insider.save()
		insider.close()


	print("best test acc: {}".format(test_acc))
# best test acc: 0.9220521541950113
