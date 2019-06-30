
import random 
from Utils import Utils
from Adaboost import Adaboost

class Choicers:

	def random_choicer(tests, epsilon):
		''' Returns:	a random choicer with not .5 - epsilon >= err >= .5 + epsilon.
			Args: 		epsilon 	if not .5 - epsilon >= error >= .5 + epsilon the choice is acceptable.
									if equal 0 its like a random choice.
						tests 		tests to choose from.'''
		
		tests = Utils.flat(tests)
		if epsilon > 0:
			def choicer(probabilities, training_set):
				''' Returns a random choice that has error != 0.5 if there is one, the best encountered otherwise. '''
				choices = list(range(len(tests)))
				random.shuffle(choices)
				best_choice = 0
				best_error  = 0.5
				# print(choices[0])
				for choice in choices:
					error = Adaboost.get_error(tests[choice], probabilities, training_set)
					if error >= 0.5 + epsilon or error <= 0.5 - epsilon: return tests[choice]
					elif error >= best_error:
						best_error = error
						best_test  = tests[choice]
					elif 1-error >= best_error:
						best_error = 1-error
						best_test  = tests[choice]
				return best_test if best_error != 0.5 else None

			return choicer

		else:
			def choicer(probabilities, training_set):
				''' Returns a random choice '''
				return random.choice(tests)
			return choicer


	def random_selector_best_choicer(tests, epsilon):
		''' Returns:	a choicer which choose a random selector and then 
						searches for the best predictor it stops if not 0.5 - epsilon >= error >= 0.5 + epsilon.
			Args: 		tests 		tests to choose from.
						epsilon 	stop threshold.
									if 0 is just a random choicer.
									if >=0.5 is just find the best.'''
		if epsilon > 0:
			def choicer(probabilities, training_set):
				sub_tests  = random.choice(tests)
				while sub_tests == []:
					sub_tests = random.choice(tests)
				
				best_error = 0.5
				best_test  = None 
				for test in sub_tests:
					error = Adaboost.get_error(test, probabilities, training_set)
					if error <= 0.5 - epsilon or error >= 0.5 + epsilon: return test
					elif error >= best_error:
						best_error = error
						best_test  = test
					elif 1-error >= best_error:
						best_error = 1-error
						best_test  = test
					else: continue
				return best_test
			return choicer
		else:
			def choicer(probabilities, training_set):
				''' Returns a random choice '''
				return random.choice(tests)
			return choicer


	def best_choicer(tests, epsilon):
		''' Returns:	a choicer which searches for the best predictor, 
						it stops if not 0.5 - epsilon >= error >= 0.5 + epsilon.
			Args: 		tests 		tests to choose from.
						epsilon 	stop threshold.
									if 0 is just a random choicer.
									if >=0.5 is just find the best.'''
		tests = Utils.flat(tests)
		if epsilon > 0:
			def choicer(probabilities, training_set):
				choices = list(range(len(tests)))
				random.shuffle(choices)
				best_error = 0.5
				best_test  = None
				for choice in choices:
					test  = tests[choice]
					error = Adaboost.get_error(test, probabilities, training_set)
					if error <= 0.5 - epsilon or error >= 0.5 + epsilon: return test
					elif error >= best_error:
						best_error = error
						best_test  = test
					elif 1-error >= best_error:
						best_error = 1-error
						best_test  = test
					else: continue
				return best_test
			return choicer
		else:
			def choicer(probabilities, training_set):
				''' Returns a random choice '''
				return random.choice(tests)
			return choicer
