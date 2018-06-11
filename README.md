## Named Entity Recognition (NER) on chinese weibo domain
---

This repo contains Theano implementation of LSTM neural model for Chinese NER.

The directory KG contains sourece code and working directory.
Sina Weibo datasets and pre-trained character vectors are respectively available in [KG/workDir/data](KG/workDir/data) and [KG/workDir/vector](KG/workDir/vector)

##usage
---
### Extract entities from sentences

Extract entities from sentence using the pretrained model (assuming that the current directory is KG/workDir):

```
python ../main.py extract --model model.NER --corpus data/rawdata --output result
```

The input file should contain one sentence by line. (refer to file [KG/workDir/data/rawdata](KG/workDir/data/rawdata))
The output file is json formatted, which includes a list of entity extraction results. Each item corresponds to a sentence and records the start index, the end index and entity type of extracted entities.

Eg.
```
乔布斯是苹果公司的创始人。 --- > [(0,3,PER),(4,8,COM)] 
```
There are five types of entities which can be recognized by the extractor.(Person, Loction, Organization, Company, Product) 

### Train a model

Train your own model (assuming that the current directory is KG/workDir):

```
python ../main.py train --train data/traindata --dev data/devdata --test data/testdata --score_dir trainResult
```
The script will automatically store the models in the directory trainResult after every epoch and record the performance of the current model on training, developing and testing dataset.

Many hyper-parameters can be tuned (network size, dropout rate, learning rate).
Input files for the training script have to follow the same format as the file [KG/workDir/data/traindata](KG/workDir/data/traindata).



##Key classes and Methods
---
class **NNmodel** (NNmodel.py)
This class is composed of general methods used in neural network based models , including some essential building block like softmaxLayer, Bidirectional LSTMLayer and different learning methods like SGD and AdaGrad

class **LSTMbaseline** (LSTM.py)
This class is the LSTM model used to do Named Entity Recognition. 

- The method **train_ready** builds the computation graph and generates callable function that would be used in training.
- The method **evaluate_ready** builds the computation graph and generates callable function that would be used in evaluating.
- The method **train** takes train, dev and test data as inputs and trains the model. During training, it also stores the intermediate model after each epoch and records  its performance on train, dev, test data.
- The method **evaluate** takes labeld data (the same format as training data) and outputs precision, recall and f1 score that the model achieves on the data for every entity type.
- The method **extract** takes unlabeld data and outputs extracted entities.

