from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.model_selection import cross_val_score
from scipy.sparse import lil_matrix
import numpy as np
import json
from time import time
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

colorset = dict()
target = list()

def colorinit():
	colorset[0] = '#108831'
	colorset[1] = '#880c7f'
	colorset[2] = '#4e8ab5'

def datainit(orlabel, topaim):
	global target
	target = [1, 4, 7]
	print(type(target))
	
	
def format_data_for_display(emb_file, i2l_file):
	i2l = dict()
	with open(i2l_file, 'r') as r:
		r.readline()
		for line in r:
			parts = line.strip().split()
			n_id, l_id = int(parts[0]), int(parts[1])
			i2l[n_id] = l_id
	
	i2e = dict()
	with open(emb_file, 'r') as r:
		r.readline()
		for line in r:
			embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
			node_id = embeds[0]
			if node_id in i2l:
				i2e[node_id] = embeds[1:]
	
	i2l_list = sorted(i2l.items(), key=lambda x:x[0])
	
	X = []
	Y = []
	for (id, label) in i2l_list:
		X.append(i2e[id])
		Y.append(label)
		
	return X,Y
	
def getdata(oremb, orlabel):
	print(target)
	emb = []
	label = []
	nodechoice = dict()
	index = 0
	for i in orlabel:
		if i in target:
			if i not in nodechoice:
				nodechoice[i]=set()
			nodechoice[i].add(index)
		index=index+1
	for i in target:
		print(str(i)+' '+str(len(nodechoice[i])))
		temp = random.sample(nodechoice[i], 500)
		for index in temp:
			emb.append(oremb[index])
			label.append(orlabel[index])
	
	return emb, label



def plot_emb(emb, label, title):
	global target
	x_min, x_max = np.min(emb, 0), np.max(emb, 0)
	data = (emb - x_min) / (x_max - x_min)
	datasize = data.shape[0]
	plt.figure()
	for i in range(datasize):
		plt.text(data[i, 0], data[i, 1], str('.'),
				 color = colorset[target.index(label[i])],
		     	 fontdict={'weight': 'bold', 'size': 15})
	plt.axis('off')
	#plt.title(title)
	plt.show()

def run(oremb, orlabel):
	colorinit()
	datainit(orlabel, 3)
	emb, label = getdata(oremb, orlabel)
	#print(label)
	tsne = TSNE(n_components=2, init='pca', random_state=0)
	tstart = time()
	result = tsne.fit_transform(emb)
	plot_emb(result, label, 't-SNE embedding (time %.2fs)'% (time()-tstart))

if __name__ == '__main__':
	oremb, orlabel = format_data_for_display('../../emb/dblp/dblp_MNCI_10.emb', '../../../data/dblp/node2label.txt')
	run(oremb, orlabel)
