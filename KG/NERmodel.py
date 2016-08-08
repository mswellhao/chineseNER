# encoding=utf-8
from util import *
from NNmodel import NNmodel
import pickle
import numpy as np
import os
import shutil
import random

class NERmodel(NNmodel): # sequence model (one token to one label) and B-I lable encoding scheme  
	def __init__(self, args, embDic, tagDic):
		NNmodel.__init__(self, args.wdecay, args.opt, args.wbound, args.gradbound, args.mweight, args.lrate)

		self.args = args
		self.embDic = embDic
		self.tagDic = tagDic
		self.fake_label = len(tagDic) # label id for padding 
		for tag_id in tagDic.values():
			assert self.fake_label != tag_id

	def train_ready(self): raise  NotImplementedError(); # implement updatefunc for training
	def evaluate_ready(self): raise NotImplementedError; #implement evafunc for evaluating
	def get_best_seq(self, x): raise NotImplementedError(); # employ evafunc to compute the most likely label sequence (label seq, logp) 

	def train(self, traindata, devdata, testdata):

		args = self.args
		batch_size = args.batch_size
		score_dir = args.score_dir
		lrate = args.lrate
		epoch = args.epoch

  		if os.path.exists(score_dir):
			shutil.rmtree(score_dir)
		os.mkdir(score_dir)
		self.set_lrate(lrate)

		print "current learning rate .............."+str(lrate)

		for epoch in xrange(args.epoch):


			print "new epoch start....................."+str(epoch)
			batches = self.create_batch(traindata, batch_size, self.embDic["<padding>"])

			sumloss = 0
			for batch in batches:

				loss = self.updatefunc(batch[0], batch[1])
				print "batch loss  :  "+str(loss)
				sumloss += loss
				
			print "epoch done ...............   sum loss  :  "+str(sumloss)
			

			# test on train data
			self.evaAndPrint(traindata, score_dir+"/trainscore",epoch)
	
			# test on development data
			self.evaAndPrint(devdata, score_dir+"/devscore",epoch)

			# test on development data
			self.evaAndPrint(testdata, score_dir+"/testscore",epoch)

			self.dumpmodel(score_dir+"/model_"+str(epoch))


	def evaluate(self, data):
		#input : data (list) each item is a instance including tokens and labels
		#output : overall p,r,f1 and p,r,f1 for each entity type
		

		gold = []
		pre = []
		for ins in data:
			if len(ins[0]) == 1:
				pre.append(ins[1])
				gold.append(ins[1])
				continue
			tags, _ = self.get_best_seq(ins[0])
			pre.append(tags)
			gold.append(ins[1])


		overall, pr_types = computePR(self.tagDic, gold, pre)

		return overall, pr_types

	def evaAndPrint(self, data, score_file, epoch):
		overall, _ = self.evaluate(data)
		precision, recall, f1 = overall
		s = score_file[score_file.index("/")+1:]+" ........  epoch  :  "+str(epoch)+" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f1)+"\n"
		print s
		trainscore_file = open(score_file,'ab')
		trainscore_file.write(s)
		trainscore_file.close()


	def extract(self,data):
		#input : data (list) each item is a instance including tokens 
		#output : ens (list) each item is a list of extracted entities, each entity is like (start index, end index, type)

		taglists = []
		for ins in data:
			tags, _ = self.get_best_seq(ins[0])
			taglists.append(tags)


		ens = scanEns(self.tagDic, taglists)

		return ens


	def create_batch(self, data, batch_size, padding_id, len_bound = 100):
		randomdata(data)

		batches = []
		for start in xrange(0,len(data),batch_size):
			batch  = data[start:start+batch_size]
			max_len = max(len(item[0]) for item in batch)
			if max_len > len_bound:
				max_len = len_bound 

			x, y = [], []

			for ins in batch:
				if len(ins[0]) <= max_len:
					x.append(ins[0]+(max_len-len(ins[0]))*[padding_id])
					y.append(ins[1]+(max_len-len(ins[1]))*[self.fake_label])
				else:
					start = random.randint(0,len(ins[0]) - max_len)
					x.append(ins[0][start:start+max_len])
					y.append(ins[1][start:start+max_len])

			batches.append((np.asarray(x, dtype = np.int32), np.asarray(y, dtype = np.int32)))


		return batches




	def dumpmodel(self, fname):
		model = {}
		model["args"] = self.args
		if not self.fix_emb:
			model["embMatrix"] = self.embMatrix.get_value()
			model["embDic"] = self.embDic

		model["model_weight"] = self.w
		model["tagDic"] = self.tagDic

		pickle.dump(model, open(fname, 'w+'))


