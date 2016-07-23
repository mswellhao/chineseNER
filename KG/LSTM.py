# encoding=utf-8
from NNmodel import NNmodel
from util import *
import theano.tensor as T
import theano
import numpy as np
import pickle
import random
import os
import shutil

class LSTMbaseline(NNmodel):

	def __init__(self, args, embMatrix, embDic, tagDic, pre_weight = {}):
		NNmodel.__init__(self, args.wdecay, args.opt, args.wbound, args.gradbound, args.mweight, args.lrate)
		def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
		def f_rectlin(x): return x*(x>0)
		def f_rectlin2(x): return x*(x>0) + 0.01 * x

		self.args = args
		self.nonlinear = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus, 'rectlin': f_rectlin, 'rectlin2': f_rectlin2}
		self.fake_label = -1

		self.model_type = args.model_type
		self.net_size = args.net_size
		self.secondLayer= args.second_layer	
		self.istoken = args.istoken
		self.fix_emb = args.fix_emb
		self.emb_lrate = args.emb_lrate
	
		
		
		print "whether fixing word embedding  :  "+str(self.fix_emb)

		self.w = pre_weight
		self.embMatrix = shared32(embMatrix)
		self.embDic = embDic
		self.tagDic = tagDic

		self.dropout_prob = shared32(args.drop_pro)
		self.srng = T.shared_randomstreams.RandomStreams(random.randint(0,9999))


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
		drop = self.get_droppro()
		self.set_droppro(0.)

		gold = []
		pre = []
		for ins in data:
			scores = self.evafunc(ins[0])
			pre.append(np.argmax(scores, 1))
			gold.append(ins[1])

		self.set_droppro(drop)

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


		drop = self.get_droppro()
		self.set_droppro(0.)

		taglists = []
		for ins in data:
			scores = self.evafunc(ins)
			tags = np.argmax(scores, 1)
			taglists.append(tags)


		self.set_droppro(drop)
		ens = scanEns(self.tagDic, taglists)

		return ens




	def train_ready(self):
		var_x = T.imatrix()
		var_y = T.imatrix()

		loss = self.l2reg(self.w, self.wdecay)+self.logp_loss(var_x,var_y)
		witems = self.w.values()
		#ave_w = sum(T.sum(item**2) for item in witems)/len(witems)

		wg = T.grad(loss, witems)
		#ave_g = sum(T.sum(item**2) for item in wg) /len(wg)

		weight_up = self.upda(wg, witems, self.lrate, self.mweight, self.opt, self.gradbound)

		if not self.fix_emb:
			dicitems = self.embMatrix.values()
			dg = T.grad(loss, dicitems)
			dic_up = self.upda(dg, dicitems, self.emb_lrate, self.mweight, self.opt)
			weight_up.update(dic_up)

		up  = weight_up
		self.updatefunc = theano.function([var_x, var_y], loss, updates = up)



	def evaluate_ready(self):
		x = T.ivector()

		em = self.embedLayer(x, self.embMatrix)
		hid1 = self.applyDropout(self.BLSTMLayer(em, self.w, "layer_1", (self.net_size[0], self.net_size[1])))
		if self.secondLayer == "forward":
			hid2 = self.applyDropout(self.forwardLayer(hid1,self.w, "layer_2", self.nonlinear["tanh"], (self.net_size[1], self.net_size[2])))
		elif self.secondLayer == "LSTM":
			hid2 = self.applyDropout(self.BLSTMLayer(hid1, self.w, "layer_2", (self.net_size[1], self.net_size[2])))
		else:
			raise Exception


		out = self.softmaxLayer(hid2, self.w, "layer_3", (self.net_size[2], self.net_size[3]))

		self.evafunc = theano.function([x], out)


		


	def logp_loss(self,x,y):
		#y 中 -1 为 不计算loss的实例的label， fake_label = -1 用来产生mask掩盖掉 label为 -1 的实例
		# 返回平均 -logp 损失
		y = y.dimshuffle((1,0))
		x = x.dimshuffle((1,0))

		em = self.embedLayer(x, self.embMatrix)
		hid1 = self.applyDropout(self.BLSTMLayer(em, self.w, "layer_1", (self.net_size[0], self.net_size[1])))
		if self.secondLayer== "forward":
			hid2 = self.applyDropout(self.forwardLayer(hid1,self.w, "layer_2", self.nonlinear["tanh"], (self.net_size[1], self.net_size[2])))
		elif self.secondLayer== "LSTM":
			hid2 = self.applyDropout(self.BLSTMLayer(hid1, self.w, "layer_2", (self.net_size[1], self.net_size[2])))
		else:
			raise Exception

		pro = self.softmaxLayer(hid2, self.w, "layer_3", (self.net_size[2], self.net_size[3]))


		mask = T.neq(y,self.fake_label)
		y = y*mask
		pro = pro.reshape((pro.shape[0]*pro.shape[1], pro.shape[2]))
		y = y.flatten()

		losslist = T.nnet.categorical_crossentropy(pro, y)
		losslist = losslist.reshape(mask.shape)
		losslist = losslist*mask



		# if istoken is True: adopt token based training
		# else :  adopt sequence based training 
		if self.istoken:
			loss =  T.sum(losslist)/T.sum(mask)

			print "adopt token based  training ..............."

		else:
			loss = T.sum(losslist)/losslist.shape[1]

			print "adopt sequence based training .............. "

		return loss




	def create_batch(self, data, batch_size, padding_id):
		randomdata(data)

		batches = []
		for start in xrange(0,len(data),batch_size):
			batch  = data[start:start+batch_size]
			max_len = max(len(item[0]) for item in batch)

			x, y = [], []

			for ins in batch:
				x.append(ins[0]+(max_len-len(ins[0]))*[padding_id])
				y.append(ins[1]+(max_len-len(ins[1]))*[self.fake_label])

			batches.append((np.asarray(x, dtype = np.int32), np.asarray(y, dtype = np.int32)))


		return batches













	


		
