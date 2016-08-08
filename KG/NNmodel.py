 # encoding=utf-8
import theano.tensor as T
import theano
import numpy as np
import pickle
import collections
from util import *


class NNmodel:

    def __init__(self, wdecay, opt, wbound, gradbound, mweight = 0.9, lrate = 0.01):
        def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
        def f_rectlin(x): return x*(x>0)
        def f_rectlin2(x): return x*(x>0) + 0.01 * x
        def f_identity(x):return x
        self.nonlinear = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus, 'rectlin': f_rectlin, 'rectlin2': f_rectlin2,'identity':f_identity}
        
        self.wdecay = wdecay
        self.mweight = shared32(mweight)
        self.lrate = shared32(lrate)
        self.opt = opt
        self.gradbound = gradbound
        self.wbound = wbound
        # these variables are defined by subclass
    
    def applyDropout(self,x):

        d = 1-self.dropout_prob
        mask = self.srng.binomial(
                n = 1,
                p = 1-self.dropout_prob,
                size = x.shape
            )
        mask = T.cast(mask, theano.config.floatX) / d
        return x*mask

    def applyDropout2list(self,listx):

        d = 1-self.dropout_prob
        mask = self.srng.binomial(
                n = 1,
                p = 1-self.dropout_prob,
                size = listx[0].shape
            )
        mask = T.cast(mask, theano.config.floatX) / d
        return [x*mask for x in listx]


    def embedLayer(self,x,dic ):
        return dic[x]

    def forwardLayer(self, x,w, layerid, non_linear, size):
        
        fw = self.set_para(w, "w"+layerid, shared32(1./np.sqrt(size[0])*np.random.randn(size[0],size[1])) )
        fb = self.set_para(w, "b"+layerid, shared32(np.random.randn(size[1])) )

        out = non_linear(T.dot(x,fw)+fb)

        return out
    

    def BLSTMLayer(self,x,w,layerid,size):
        wxi = self.set_para(w,"wxi"+layerid,shared32(1./np.sqrt(size[0])*np.random.randn(size[0],size[1]/2)))
        whi = self.set_para(w,"whi"+layerid,shared32(1./np.sqrt(size[1]/2)*np.random.randn(size[1]/2,size[1]/2)))
        wci = self.set_para(w,"wci"+layerid,shared32(np.random.randn(size[1]/2)))
        wxf = self.set_para(w,"wxf"+layerid,shared32(1./np.sqrt(size[0])*np.random.randn(size[0],size[1]/2)) )
        whf = self.set_para(w,"whf"+layerid,shared32(1./np.sqrt(size[1]/2)*np.random.randn(size[1]/2,size[1]/2)))
        wcf = self.set_para(w,"wcf"+layerid,shared32(np.random.randn(size[1]/2)))
        wx = self.set_para(w,"wx"+layerid,shared32(1./np.sqrt(size[0])*np.random.randn(size[0],size[1]/2)))
        wh = self.set_para(w,"wh"+layerid,shared32(1./np.sqrt(size[1]/2)*np.random.randn(size[1]/2,size[1]/2)))
        #输入
        wxo = self.set_para(w,"wxo"+layerid,shared32(1./np.sqrt(size[0])*np.random.randn(size[0],size[1]/2)))
        who = self.set_para(w,"who"+layerid,shared32(1./np.sqrt(size[1]/2)*np.random.randn(size[1]/2,size[1]/2)))
        wco = self.set_para(w,"wco"+layerid,shared32(np.random.randn(size[1]/2)))

        big = self.set_para(w,"big"+layerid,shared32(np.random.randn(size[1]/2)))
        bfg = self.set_para(w,"bfg"+layerid,shared32(np.random.randn(size[1]/2)))
        bog = self.set_para(w,"bog"+layerid,shared32(np.random.randn(size[1]/2)))
        bx = self.set_para(w,"bx"+layerid,shared32(np.random.randn(size[1]/2)))##输入

        
        #反向权重
        wxi_r = self.set_para(w,"wxi_r"+layerid,shared32(1./np.sqrt(size[0])*np.random.randn(size[0],size[1]/2)))
        whi_r = self.set_para(w,"whi_r"+layerid,shared32(1./np.sqrt(size[1]/2)*np.random.randn(size[1]/2,size[1]/2)))
        wci_r = self.set_para(w,"wci_r"+layerid,shared32(np.random.randn(size[1]/2)))
        wxf_r = self.set_para(w,"wxf_r"+layerid,shared32(1./np.sqrt(size[0])*np.random.randn(size[0],size[1]/2))) 
        whf_r = self.set_para(w,"whf_r"+layerid,shared32(1./np.sqrt(size[1]/2)*np.random.randn(size[1]/2,size[1]/2)))
        wcf_r = self.set_para(w,"wcf_r"+layerid,shared32(np.random.randn(size[1]/2)))
        wx_r = self.set_para(w,"wx_r"+layerid,shared32(1./np.sqrt(size[0])*np.random.randn(size[0],size[1]/2)))
        wh_r = self.set_para(w,"wh_r"+layerid,shared32(1./np.sqrt(size[1]/2)*np.random.randn(size[1]/2,size[1]/2)))
        #输入

        wxo_r = self.set_para(w,"wxo_r"+layerid,shared32(1./np.sqrt(size[0])*np.random.randn(size[0],size[1]/2)))
        who_r = self.set_para(w,"who_r"+layerid,shared32(1./np.sqrt(size[1]/2)*np.random.randn(size[1]/2,size[1]/2)))
        wco_r = self.set_para(w,"wco_r"+layerid,shared32(np.random.randn(size[1]/2)))

        big_r = self.set_para(w,"big_r"+layerid,shared32(np.random.randn(size[1]/2)))
        bfg_r = self.set_para(w,"bfg_r"+layerid,shared32(np.random.randn(size[1]/2)))
        bog_r = self.set_para(w,"bog_r"+layerid,shared32(np.random.randn(size[1]/2)))
        bx_r = self.set_para(w,"bx_r"+layerid,shared32(np.random.randn(size[1]/2)))##输入

        sig= self.nonlinear["sigmoid"]
        tan =self.nonlinear["tanh"]
        def forward_pass(lx, h, c, wxi, whi, wci, wxf, whf, wcf, wx, wh,  wxo, who, wco, big, bfg, bog,bx):
            igate = T.dot(lx,wxi)+T.dot(h,whi)+c*wci+big
            fgate = T.dot(lx,wxf)+T.dot(h,whf)+c*wcf+bfg
            inpu_t = T.dot(lx,wx)+T.dot(h,wh)+bx
            new_c = sig(fgate)*c + sig(igate)*tan(inpu_t)
            ogate = T.dot(lx,wxo)+T.dot(h,who)+new_c*wco+bog
            new_h = sig(ogate)*tan(new_c)
            return [new_h, new_c ]

        def backward_pass(lx, for_h, h, c, wxi, whi, wci, wxf, whf, wcf, wx, wh,  wxo, who, wco, big, bfg, bog,bx):
            igate = T.dot(lx,wxi)+T.dot(h,whi)+c*wci+big
            fgate = T.dot(lx,wxf)+T.dot(h,whf)+c*wcf+bfg
            inpu_t = T.dot(lx,wx)+T.dot(h,wh)+bx
            new_c = sig(fgate)*c + sig(igate)*tan(inpu_t)
            ogate = T.dot(lx,wxo)+T.dot(h,who)+new_c*wco+bog
            new_h = sig(ogate)*tan(new_c)
            all_h = T.concatenate([for_h,new_h], axis = for_h.ndim - 1)

            return [new_h, new_c, all_h]


        initial = T.zeros_like(T.dot(x[0], wxi))
        result,up = theano.scan(fn = forward_pass, sequences = x, outputs_info = [initial,initial],
                                non_sequences = [wxi, whi, wci,wxf, whf, wcf, wx, wh,  wxo, who, wco, big, bfg, bog,bx])

        self.debug_result1 = result[0]
        result_r,up_r = theano.scan(fn = backward_pass, sequences = [x, result[0]],
                                    outputs_info = [initial, initial, None],
                                non_sequences = [wxi_r, whi_r, wci_r, wxf_r, whf_r, wcf_r, wx_r, wh_r,  wxo_r, who_r, wco_r,big_r, bfg_r, bog_r,bx_r], go_backwards  = True)


        #print type(result_r)
        final = result_r[2][::-1]
        #print type(final)

        return final


    def softmaxLayer(self, x, w, layerid, size):
        sw = self.set_para(w,"w"+layerid,shared32(1./np.sqrt(size[0])*np.random.randn(size[0],size[1])))
        sb = self.set_para(w,"b"+layerid,shared32(np.random.randn(size[1])))

        out = T.dot(x, sw) + sb

        if x.ndim == 2:
            sflayer = T.nnet.softmax(out)
        elif x.ndim == 3:

            tshape = out.shape
            sflayer = T.nnet.softmax(out.reshape((tshape[0]*tshape[1], tshape[2]))).reshape(tshape)
        
        return sflayer



    def set_para(self, w, key, random_init):
        if key in w:
            return w[key]
        else:
            w[key] = random_init
            return random_init


    def upda(self,gw,w,lrate, mweight= 0.9 , grad = "momentum",gradbound= -1,wbound= -1, eps = 1e-8):
        updates = {}


        print "adopt learning method .............  "+grad
        assert len(gw) == len(w)
        scale = 1
        if gradbound > 0:
            print "adopt  gradient clopping ................."+str(gradbound)
            norm = T.sqrt(sum(T.sum(item**2) for item in gw))
            def clopping(x):return (x < 1)*(x-1)+1
            scale  = clopping(gradbound/norm)
        if wbound > 0:
            print "adopt weight cliopping .................."+str(wbound)
            assert len(w) == 1
            def bound(iw):
                norm = T.sum(iw**2, 0)
                x = T.sqrt(wbound/norm)
                wscale = (x < 1)*(x-1)+1
                return iw*wscale
        else:
            def bound(iw):return iw

        
        if grad == "adaGrad":
            for i in range(len(gw)):
                acc = shared32(w[i].get_value()*0.)
                new_acc = acc + gw[i]**2
                #print "learning rate ........ "+str(lrate.get_value())
                new_w = w[i] - lrate*gw[i]*scale/T.sqrt(new_acc + eps)

                updates[acc] = new_acc
                updates[w[i]] = bound(new_w)


        elif grad == "momentum":

            for i in range(len(gw)):
                mom = shared32(w[i].get_value()*0.)
                new_mom = mom*mweight - lrate*gw[i]

                new_w = w[i] + new_mom

                updates[mom] = new_mom
                updates[w[i]] = new_w

        elif grad == "sgd":
            for i in xrange(len(gw)):
                new_w = w[i] - lrate*gw[i]*scale
                updates[w[i]] = new_w
        

        return collections.OrderedDict(updates)


    def log_sum_exp(self, x):
        xmax = x.max(axis = -1, keepdims = True)
        xmax_ = x.max(axis = -1)

        return xmax_ + T.log(T.sum(T.exp(x-xmax), axis = -1))

    def CRFLayer(self, scores, trans, viterbi = True, batch = False):

        if batch:
             batch_num = scores.shape[1]
             trans = T.tile(trans, [batch_num,1,1])
        def recurrence(current_score, prev, trans):
            if batch:
                current_score = current_score.dimshuffle(0,1,'x')
                prev = prev.dimshuffle(0,'x',1)
                temp = trans + current_score 
                cans = temp + prev

            else:
                current_score = current_score.dimshuffle(0,'x')
                prev = prev.dimshuffle('x',0)
                cans = trans + current_score  + prev
            if viterbi:
                new_prev = T.max(cans, axis = -1)
                max_path = T.cast(T.argmax(cans, axis = -1), 'int32')
                return new_prev, max_path
            else:
                # compute sum of all path score
                new_prev = self.log_sum_exp(cans)
                return new_prev

        result, _ = theano.scan(fn = recurrence, outputs_info = (scores[0], None) if viterbi else scores[0], sequences = scores[1:], non_sequences = trans)

        if viterbi:
            max_path_index, _ = theano.scan(fn = lambda current_mark, prev_index: current_mark[prev_index],
             outputs_info = T.cast(T.argmax(result[0][-1]), 'int32'), sequences = result[1][::-1])
            
            max_path_index = T.concatenate([max_path_index[::-1], [T.cast(T.argmax(result[0][-1]), 'int32')]])

            return max_path_index, T.max(result[0][-1])

        else:
            sumscore = self.log_sum_exp(result[-1])
            return sumscore
            # score sum of all possible path in a sequence or a batch of sequences


    def decode(self,scores, trans,top_n = 1):
        #input : scores (n_seq, n_tags),
        #         trans(n_tags, n_tags),
        #         top_n( return top n decode sequences)
        #output: a list of tuples consisted of tage sequence (sequence) and score (exp)s
       
        road = []
        i = 0
        while i < len(scores):
            score  = scores[i]

            if i == 0:
                node = []
                logp = np.log(np.exp(score + trans[-1])/np.sum(np.exp(score)))

                for k in range(score.shape[0]):
                    node.append([(-1, -1, logp[k])])
                road.append(node)

            else :
                node = []
                logp = []
                for l1 in range(self.tag_num):
                    logp.append(np.log(np.exp(score + trans[l1])/np.sum(np.exp(score + trans[l1]))))

                for l in range(self.tag_num):
                    candidates = []
                    for j in range(len(road[-1])):
                        for k in range(len(road[-1][j])):
                            candidates.append((j,k,road[-1][j][k][2] + logp[j][l]))

                    candidates.sort(lambda x,y:cmp(y[2],x[2]))
                    node.append(candidates[0:top_n])

                road.append(node)
            i += 1
        can = []
        for i in range(len(road[-1])):
            for j in range(len(road[-1][i])):
                can.append((i,j,road[-1][i][j][2]))

        can.sort(lambda x,y:cmp(y[2],x[2]))

        result = []
        for i in range(top_n):
            sequence = []
            tag = can[i][0]
            offset = can[i][1]

            index = len(road) - 1
            try:
                while index >= 0:
                    sequence.append(tag)
                    tag, offset = road[index][tag][offset][0], road[index][tag][offset][1]
                    index  = index - 1
            except Exception:
                print "index  "+str(index)
                print "offset  "+str(offset)
            sequence.reverse()
            result.append((sequence, can[i][2]))


        return  result  

   

    def l2reg(self,w,wdecay): # only deal with laywise weights
        reg = 0
        for key in w:
            reg += T.sum(w[key]**2)

        return reg*wdecay


    def get_droppro(self):
        return self.dropout_prob.get_value()

    def set_droppro(self, value):

        self.dropout_prob.set_value(value)

    def get_lrate(self):
        return self.lrate.get_value()
    def set_lrate(self,value):
        self.lrate.set_value(value)


