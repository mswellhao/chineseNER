# encoding=utf-8
from NERmodel import NERmodel
from util import *
import theano.tensor as T
import theano
import numpy as np
import pickle
import random
import os
import shutil

class LSTMbaseline(NERmodel):

	def __init__(self, args, embMatrix, embDic, tagDic, pre_weight = {}):
		NERmodel.__init__(self, args, embDic, tagDic)

		self.args = args
		
		self.model_type = args.model_type
		self.net_size = args.net_size
		self.second_layer= args.second_layer	
		self.istoken = args.istoken
		self.CRF = args.CRF

		self.fix_emb = args.fix_emb
		self.emb_lrate = args.emb_lrate
		print "whether fixing word embedding  :  "+str(self.fix_emb)

		self.w = pre_weight
		self.embMatrix = shared32(embMatrix)
	
		self.dropout_prob = shared32(args.drop_pro)
		self.srng = T.shared_randomstreams.RandomStreams(random.randint(0,9999))


	def train_ready(self):
		var_x = T.imatrix()
		var_y = T.imatrix()

		loss = self.l2reg(self.w, self.wdecay)+self.logp_loss(var_x,var_y)
		witems = self.w.values()
		#ave_w = sum(T.sum(item**2) for item in witems)/len(witems)

		wg = T.grad(loss, witems, disconnected_inputs = 'ignore')
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
		# create evaluation function that computes the most likely tag sequence and its log probability 
		x = T.ivector()

		em = self.embedLayer(x, self.embMatrix)
		hid1 = self.applyDropout(self.BLSTMLayer(em, self.w, "layer_1", (self.net_size[0], self.net_size[1])))
		if self.second_layer == "forward":
			hid2 = self.applyDropout(self.forwardLayer(hid1,self.w, "layer_2", self.nonlinear["tanh"], (self.net_size[1], self.net_size[2])))
		elif self.second_layer == "LSTM":
			hid2 = self.applyDropout(self.BLSTMLayer(hid1, self.w, "layer_2", (self.net_size[1], self.net_size[2])))
		else:
			raise Exception



		if self.CRF:
			print "evalute adopting model  CRF .............. "
			scores = self.forwardLayer(hid2, self.w, "layer_3", self.nonlinear["identity"], (self.net_size[2], self.net_size[3]))
			small = -10000
			scores = T.concatenate([scores, T.ones((scores.shape[0], 1))*small], axis = 1)
			scores = T.concatenate([scores, np.asarray([ [small]*self.net_size[-1]+[0] ], dtype = 'float32')], axis = 0)
			tags, max_score = self.CRFLayer(scores, self.w["trans"], viterbi = True)
			sum_score = self.CRFLayer(scores, self.w["trans"], viterbi = False)
			logp = max_score - sum_score

		else:
			pros = self.softmaxLayer(hid2, self.w, "layer_3", (self.net_size[2], self.net_size[3]))
			tags = T.argmax(pros,1)
			logp = T.sum(T.log(T.max(pros,1)))

		self.evafunc = theano.function([x],[tags, logp])


	def get_best_seq(self, x):
		#return the most likely label sequence and its score (log probability)

		drop = self.get_droppro()
		self.set_droppro(0.)
		tags, logp = self.evafunc(x)
		self.set_droppro(drop)
		
		return tags, logp

	def logp_loss(self,x,y):
		#y 中 -1 为 不计算loss的实例的label， fake_label = -1 用来产生mask掩盖掉 label为 -1 的实例
		# 返回平均 -logp 损失
		y = y.dimshuffle((1,0))
		x = x.dimshuffle((1,0))

		em = self.embedLayer(x, self.embMatrix)
		hid1 = self.applyDropout(self.BLSTMLayer(em, self.w, "layer_1", (self.net_size[0], self.net_size[1])))
		if self.second_layer== "forward":
			hid2 = self.applyDropout(self.forwardLayer(hid1,self.w, "layer_2", self.nonlinear["tanh"], (self.net_size[1], self.net_size[2])))
		elif self.second_layer== "LSTM":
			hid2 = self.applyDropout(self.BLSTMLayer(hid1, self.w, "layer_2", (self.net_size[1], self.net_size[2])))
		else:
			raise Exception



		if self.CRF:
			print "train adopting model  CRF .............. "
			# set trans matrix length to tag num plus 1 for fake label id 
			self.set_para(self.w, "trans", shared32(1./np.sqrt(self.net_size[-1]+1)*np.random.randn(self.net_size[-1]+1,self.net_size[-1]+1)))

			small = -10000
			scores = self.forwardLayer(hid2, self.w, "layer_3", self.nonlinear["identity"], (self.net_size[2], self.net_size[3]))
			mask = T.neq(y,self.fake_label)
			t_mask = mask.dimshuffle((0,1,'x'))
			scores = scores*t_mask#mask scores of  fake label 
			scores += T.eq(y, self.fake_label).dimshuffle((0,1,'x'))*small# set scores of fake label to small value 
			scores = T.concatenate([scores, T.ones((scores.shape[0], scores.shape[1], 1))*small*t_mask], axis = 2)

			new_scores = scores.reshape((scores.shape[0]*scores.shape[1], scores.shape[2]))
			new_y = y.flatten()
			tag_score = new_scores[T.arange(new_y.shape[0]), new_y].sum()
			
			seq_len = y.shape[0]
			left_index = y[T.arange(seq_len-1)].flatten()
			right_index = y[T.arange(seq_len - 1)+1].flatten()
			tag_score += self.w["trans"][right_index, left_index].sum()
			sum_score = self.CRFLayer(scores,self.w["trans"],viterbi = False, batch = True).sum()
			logp = tag_score - sum_score
			loss = (0 - logp)/y.shape[1]
			#debug
			#loss = tag_score
			
		else:

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




	













	


		
