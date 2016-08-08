import pickle
import argparse
import sys
import myio
from LSTM import LSTMbaseline
from regLSTM import regLSTM


import json

def main(args):

	print "choose model type .................. "+args.model_type
	embeddings = pickle.load(open(args.embeddings))
	embMatrix = embeddings["matrix"]
	embDic = embeddings["dic"]
	#print embDic.items()[:10]
	print "embedding size :  "+str(len(embMatrix[0]))+"  embeddings num  :  "+str(len(embDic))
	if "<padding>" not in embDic:
		embDic["<padding>"] = len(embDic) + 1

	if args.action == "train":

		tagDic = myio.create_tagdic(args.train)
		pre_weight = {}
		
		if args.model:
			print "loaing model............"
			loaded_model = pickle.load(open(args.model))
			if "embMatrix" in loaded_model:
				embMatrix = loaded_model["embMatrix"]
				embDic = loaded_model["embDic"]
			loaded_args = loaded_model["args"]
			args.net_size = loaded_args.net_size
			args.model_type = loaded_args.model_type
			args.second_layer= loaded_args.second_layer	
			
			print "loaded model training  args : "+str(loaded_args)

			tagDic = loaded_model["tagDic"]
			pre_weight = loaded_model["model_weight"]

		print "model training  args : "+str(args)
		traindata = myio.create_labled_data(args.train, embDic, tagDic)
		devdata = myio.create_labled_data(args.dev, embDic, tagDic)
		testdata = myio.create_labled_data(args.test, embDic, tagDic)
		
		if args.model_type == "LSTMbaseline":
			model = LSTMbaseline(args, embMatrix, embDic, tagDic, pre_weight)
		elif args.model_type == "regLSTM":
			model = regLSTM(args, embMatrix, embDic, tagDic, pre_weight)
		print "model  size :  "+str(args.net_size) + " ;  model  type  :  "+ args.model_type + " ;  model second layer type  :  "+ args.second_layer

		model.train_ready()
		model.evaluate_ready()
		model.train(traindata, devdata, testdata)


	elif args.action == "evaluate":

		
		print "loaing model............"
		loaded_model = pickle.load(open(args.model))
		if "embMatrix" in loaded_model:
			embMatrix = loaded_model["embMatrix"]
			embDic = loaded_model["embDic"]
		loaded_args = loaded_model["args"]
		print "loaded  model  arg  ............"+str(loaded_args)
		
		tagDic = loaded_model["tagDic"]
		print "model tag  dic ................. "+str(tagDic)
		pre_weight = loaded_model["model_weight"]

		if args.model_type == "LSTMbaseline":
			model = LSTMbaseline(args, embMatrix, embDic, tagDic, pre_weight)
		elif args.model_type == "regLSTM":
			model = regLSTM(args, embMatrix, embDic, tagDic, pre_weight)

		model.evaluate_ready()

		evadata = myio.create_labled_data(args.eva, embDic, tagDic)
		overall, pr_types = model.evaluate(evadata)
		precision, recall, f1 = overall
		s = "ovrall result ......   P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f1)+"\n"
		print s
		for key in pr_types:
			precision, recall, f1 = pr_types[key]
			s = "entity type ..........." + str(key) + "  P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f1)+"\n"
			print s


	elif args.action == "extract":

		print "loaing model............"
		loaded_model = pickle.load(open(args.model))
		if "embMatrix" in loaded_model:
			embMatrix = loaded_model["embMatrix"]
			embDic = loaded_model["embDic"]
		loaded_args = loaded_model["args"]
		print "loaded  model  arg  ............"+str(loaded_args)
		tagDic = loaded_model["tagDic"]
		print "model tag  dic ................. "+str(tagDic)
		pre_weight = loaded_model["model_weight"]


		if args.model_type == "LSTMbaseline":
			model = LSTMbaseline(args, embMatrix, embDic, tagDic, pre_weight)
		elif args.model_type == "regLSTM":
			model = regLSTM(args, embMatrix, embDic, tagDic, pre_weight)

		model.evaluate_ready()


		data = myio.create_unlabled_data(args.corpus, embDic)
		ens = model.extract(data)

		outputfile = args.output
		json.dump(ens, open(outputfile,"w+"))




if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])

	argparser.add_argument("action",
		type = str,
		default = "train"
		)

	argparser.add_argument("--model",
		type = str,
		default = ""
		)

	# data for training 
	argparser.add_argument("--train",
		type = str,
		default = "data/traindata"
		)

	# data for test
	argparser.add_argument("--test",
		type = str,
		default = "data/testdata"
		)

	# data for development
	argparser.add_argument("--dev",
		type = str,
		default = "data/devdata"
		)

	# data for evaluation
	argparser.add_argument("--eva",
		type = str,
		default = "data/testdata"
		)

	# data for extracting entities 
	argparser.add_argument("--corpus",
		type = str,
		default = "data/rawdata"
		)

	argparser.add_argument("--output",
		type = str,
		default = "result"
		)

	argparser.add_argument("--embeddings",
		type = str,
		default = "vector/baidu_charvectors_150"
		)

	argparser.add_argument("--model_type", 
		type = str, 
		default = "LSTMbaseline"
		)

	argparser.add_argument("--net_size",
		nargs = 4,
		type = int,
		default = [150, 450, 300, 11]
		)

	argparser.add_argument("--second_layer",
		type = str,
		default = "forward"
		)

	argparser.add_argument("--CRF",
		type = bool,
		default = False
		)

	# adopt sequence based learning or token based learning
	argparser.add_argument("--istoken",
		type = bool,
		default = False
		)

	argparser.add_argument("--fix_emb",
		type = bool,
		default = True
		)


	#embedding learning rate if fine-tning embeddings 
	argparser.add_argument("--emb_lrate", 
		type = float,
		default = 0.01
		)

	argparser.add_argument("--drop_pro",
		type = float,
		default = 0.35
		)

	argparser.add_argument("--wdecay",
		type = float,
		default = 0.
		)

	argparser.add_argument("--opt",
		type = str,
		default = "adaGrad"
		)

	argparser.add_argument("--lrate",
		type = float,
		default = 0.02
		)

	# weight norm bound (if set < 0, no bound)
	argparser.add_argument("--wbound",
		type = float,
		default = -0.1
		)

	# gradient norm bound (if set < 0, no bound)

	argparser.add_argument("--gradbound",
		type = float,
		default = -0.1
		)

	argparser.add_argument("--mweight",
		type = float,
		default = 0.9
		)

	argparser.add_argument("--batch_size",
		type = int,
		default = 64
		)

	argparser.add_argument("--score_dir",
		type = str,
		default = "trainResult"
		)

	argparser.add_argument("--epoch",
		type = int,
		default = 10
		)
	
	argparser.add_argument("--lowreg_weight",
		type = float,
		default = 0.1
		)

	argparser.add_argument("--highreg_weight",
		type = float,
		default = 0.1,
		)

	argparser.add_argument("--variance",
		type = float,
		default = 0.01
		)

	argparser.add_argument("--nc",
		type = int,
		default = 3
		)



	args = argparser.parse_args()
	main(args)

