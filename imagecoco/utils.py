import torch
import glob
import os
import numpy as numpy
import matplotlib.pyplot as plt
import pickle


def save(net, file_name, num_to_keep=4):
	"""Saves the net to file, creating folder paths if necessary.

	Parameters:
		net(torch.nn.module): The network to save
		file_name(str): the path to save the file.
		num_to_keep(int): Specifies how many previous saved states to keep once this one has been saved.
			Defaults to 4. Specifying < 0 will not remove any previous saves.
	"""

	folder = os.path.dirname(file_name)
	if not os.path.exists(folder):
		os.makedirs(folder)
	torch.save(net.state_dict(), file_name)
	checkpoints = sorted(glob.glob(folder + '/*.pt'), key=os.path.getmtime)
	print('Saved %s\n' % file_name)
	if num_to_keep > 0:
		for ff in checkpoints[:-num_to_keep]:
			os.remove(ff)

def restore(net, save_file):
	"""
	Restores the weights from a saved file

	Parameters:
		net(torch.nn.Module): The net to restore
		save_file(str): The file path
	"""
	checkpoint = torch.load(save_file)

	net.load_state_dict(checkpoint['model_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	print('Restored %s' % save_file)
	return epoch, loss


def restore_latest(net, folder, model_type):
	"""Restores the most recent weights in a folder

	Parameters:
		net(torch.nn.module): The net to restore
		folder(str): The folder path
		type(str): "g" or "r" to indicate which model type to restore
	"""

	checkpoints = sorted(glob.glob(folder + '/' + model_type + '*.pt'), key=os.path.getmtime)
	start_it = 0
	if len(checkpoints) > 0:
		return restore(net, checkpoints[-1])