import matplotlib.pyplot as plt
from Measurers import Measurers
plt.style.use('fivethirtyeight')


class Insider:
	def __init__(self, clss, npartitions, name, step, max_iteration):
		self.data 			  = dict()
		self.clss 			  = clss
		self.step 			  = step
		self.max_iteration 	  = max_iteration
		self.npartitions 	  = npartitions
		self.name 			  = name
		plt.ion()
		self.figure, self.axs = plt.subplots(3,1,figsize=(10, 10),tight_layout=True)

	def __call__(self, iteration, weights, predictors):
		''' each calls collect meaned datas for each iterations. '''
		if iteration % self.step == 0 or iteration == self.max_iteration-1:

			trtp, trtn, trfp, trfn = Measurers.tp_tn_fp_fn(weights, predictors, self.training_set)
			tstp, tstn, tsfp, tsfn = Measurers.tp_tn_fp_fn(weights, predictors, self.test_set    )

			trn_acc = Measurers.accuracy (trtp, trtn, trfp, trfn)
			val_acc = Measurers.accuracy (tstp, tstn, tsfp, tsfn)
			trn_prc = Measurers.precision(trtp, trtn, trfp, trfn)
			val_prc = Measurers.precision(tstp, tstn, tsfp, tsfn)
			trn_rec = Measurers.recall   (trtp, trtn, trfp, trfn)
			val_rec = Measurers.recall   (tstp, tstn, tsfp, tsfn)

			if iteration not in self.data:
				self.data[iteration] = [1, trn_acc, val_acc, trn_prc, val_prc, trn_rec, val_rec]
			else: # calc all the means ###
				self.data[iteration][0] += 1
				self.data[iteration][1] += (trn_acc-self.data[iteration][1])/self.data[iteration][0]
				self.data[iteration][2] += (val_acc-self.data[iteration][2])/self.data[iteration][0]
				self.data[iteration][3] += (trn_prc-self.data[iteration][3])/self.data[iteration][0]
				self.data[iteration][4] += (val_prc-self.data[iteration][4])/self.data[iteration][0]
				self.data[iteration][5] += (trn_rec-self.data[iteration][5])/self.data[iteration][0]
				self.data[iteration][6] += (val_rec-self.data[iteration][6])/self.data[iteration][0]

			print(( "acc: {: <3.5f}, valacc: {: <3.5f}, "+
			 		"prc: {: <3.5f}, valprc: {: <3.5f}, "+
			 		"rec: {: <3.5f}, valrec: {: <3.5f}, "+
				    "iteration: {}").format(trn_acc, val_acc, trn_prc, 
										    val_prc, trn_rec, val_rec, iteration))

			self.plot()


	def plot(self):
		
		self.axs[0].cla()
		self.axs[1].cla()
		self.axs[2].cla()

		self.axs[0].set_title('accuracy')
		self.axs[0].plot(list(self.data.keys()),list(zip(*self.data.values()))[1], label='accuracy' )
		self.axs[0].plot(list(self.data.keys()),list(zip(*self.data.values()))[2], label='val acc'  )
		# self.axs[0].set_xlabel('iteration')
		self.axs[0].set_ylabel('output'   )
		self.axs[0].set_yticks([0.2*i for i in range(6)])
		self.axs[0].legend(fancybox=True, framealpha=0.2)
		

		self.axs[1].set_title('precision')
		self.axs[1].plot(list(self.data.keys()),list(zip(*self.data.values()))[3], label='precision')
		self.axs[1].plot(list(self.data.keys()),list(zip(*self.data.values()))[4], label='val_pre'  )
		# self.axs[1].set_xlabel('iteration')
		self.axs[1].set_ylabel('output'   )
		self.axs[1].set_yticks([0.2*i for i in range(6)])
		self.axs[1].legend(fancybox=True, framealpha=0.2)
		

		self.axs[2].set_title('recall')
		self.axs[2].plot(list(self.data.keys()),list(zip(*self.data.values()))[5], label='recall'   )
		self.axs[2].plot(list(self.data.keys()),list(zip(*self.data.values()))[6], label='val_rec'  )
		self.axs[2].set_xlabel('iteration')
		self.axs[2].set_ylabel('output'   )
		self.axs[2].set_yticks([0.2*i for i in range(6)])
		self.axs[2].legend(fancybox=True, framealpha=0.2)

		self.figure.canvas.draw()
		self.figure.canvas.flush_events()

	def save(self):
		title = self.name+'class_{}_cross_val_{}'.format(self.clss, self.npartitions)
		self.figure.savefig("./images/" + title)