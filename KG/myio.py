# encoding=utf-8
import sys
import io
import re
import json
import pickle


# data format in file: each line is a pair of char and tag; different sentences are seperated by a empty line
def create_labled_data(filename, embDic, tagDic):
	f = io.open(filename, encoding = "utf-8")

	data = []
	chars, tags = [], []
	for line in f:
		if line.strip():
			items = line.strip().split()
			if len(items) != 2:
				continue
			#print items
			chars.append(embDic.get(items[0], 0))
			tags.append(tagDic.get(items[1], -1))
		else:
			if len(chars) > 0:
				assert len(chars) == len(tags)
				data.append((chars,tags))
			chars, tags = [], []


	return data



# data format in file: each line is a sentence 
def create_unlabled_data(filename, embDic):
	data = readsen(filename, False)
	data = sen2index(data, embDic)

	return data


# data format in file: each line is a pair of char and tag; different sentences are seperated by a empty line
def create_tagdic(filename):

	tagDic = {}

	f = open(filename)
	for line in f:
		if line.strip():
			items = line.strip().split()
			if len(items) != 2:
				continue
			tag = items[1]
			if tag not in tagDic:
				tagDic[tag] = len(tagDic)

	print tagDic
	return tagDic



def readsen(filename, is_segment):
	f = io.open(filename, encoding = "utf-8")
	sens = []
	for line in f:
		if is_segment:
			words = [word.strp() for word in jieba.cut(line) if word.strip()]
			sens.append(words)
		else:
			line = re.sub("\d+\.\d+|\d+", "NUM", line)
			chars = list(line)
			i = 0                                                                            
			while i < len(chars):
				if chars[i].islower() or chars[i].isupper():
					start = i
					while i < len(chars) and (chars[i].islower() or chars[i].isupper()):
						i +=1
					end = i
					chars = chars[:start] + ["".join(chars[start:end])]+chars[end:]                          
					i = start

				i += 1
			sens.append(chars)

	return sens


def sen2index(tokendata, dic, default = 0):#[token, token, token] -> [1,3,5]

	indexdata = []
	for item in tokendata:
		senindex = [dic.get(re.sub("\d+\.\d+|\d+", "NUM", token), default) for token in item]
	 	indexdata.append(senindex)

		
	return indexdata 





