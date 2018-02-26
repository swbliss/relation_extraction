# Relation Extraction with Pre-training
This project is about relation extraction adopting two novel pre-training methods for better performance in data-scarce setting.
Before supervised learning with relatively a small amount of labeled data, unsupervised pre-training is performed with a large amount of text data.
Inspired by skip-gram model, the model attempts to predict key words, defined in two different ways, from sentence embedding during pre-training phase.

## Execution
python nyt_ds.py -p skipgram -r 3435 -L 0.05 -m 0 -S 2

parameters: 
-p: pre-training method {none | skipgram | seq2seq | depsp}
-r: random seed 
-L: learning rate
-m: pre-training mode (for skipgram)
% for others: check parse_argv function[nyt_ds.py:27] (some parameters dealt in this function are meaningless)

## Data
format(train/test): 
          {index of e1} {index of e2}
          {relation index(id)} {the number of example sentences}
          {e1's position in a sentence} {e2's position in the sentence} { id of word sequence in the sentence ...}

* data_ace: Hachey et al., 2012
* data_figer: Desmet and Hoste, 2014
* data_riedel: Riedel et al., 2010
* data_sem_eval: Hendrickx et al., 2010

## Built With
* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [Theano 0.9.0](http://deeplearning.net/software/theano/)


## References
* Zeng, Daojian, et al. "Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks." EMNLP. 2015.
* Bart Desmet and Ve ́ronique Hoste. 2014. Fine-grained dutch named entity recognition. Language Re- sources and Evaluation 48(2):307–343.
* Iris Hendrickx, Su Nam Kim, Zornitsa Kozareva, Preslav Nakov, Diarmuid O ́ Se ́aghdha, Sebastian Pado ́, Marco Pennacchiotti, Lorenza Romano, and Stan Szpakowicz. 2010. Semeval-2010 task 8: Multi-way classification of semantic relations be- tween pairs of nominals. In Proceedings of the 5th International Workshop on Semantic Evaluation, Se- mEval@ACL 2010, Uppsala University, Uppsala, Sweden, July 15-16, 2010. pages 33–38.
* Ben Hachey, Claire Grover, and Richard Tobin. 2012. Datasets for generic relation extraction. Natural Language Engineering 18(1):21–59.
* Sebastian Riedel, Limin Yao, and Andrew McCallum. 2010. Modeling relations and their mentions with- out labeled text. Machine learning and knowledge discovery in databases pages 148–163.

