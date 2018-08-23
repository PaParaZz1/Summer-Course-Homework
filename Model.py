import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

USE_CUDA = torch.cuda.is_available()
class lSTM(nn.Module):
	'''
		LSTM+FC
		single FC is better than 2 layers FC
		no dropout will be better
		softmax makes the result worse
		Xavier initialization is nearly the same as default initialization
	'''
	def __init__(self, input_size, hidden_size, num_layers, batch = 100, isFC = True, FC_size = 50):
		super(lSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.isFC = isFC
		self.FC_size = FC_size
		self.batch = batch
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True) # dropout = 0.5
		self.hidden = self.init_hidden()
		self.fc1 = nn.Linear(hidden_size, FC_size)
		self.fc2 = nn.Linear(FC_size, 20)
		self.fc_single = nn.Linear(hidden_size, 20)
		#self.softmax = nn.Softmax(dim=1)


	def init_hidden(self):
		tensor1 = torch.zeros(self.num_layers, self.batch, self.hidden_size)
		tensor2 = torch.zeros(self.num_layers, self.batch, self.hidden_size)
		if USE_CUDA:
			tensor1=tensor1.cuda()
			tensor2=tensor2.cuda()
		return (Variable(tensor1), Variable(tensor2))


	def forward(self, input):
		'''
			if swap fc and output slicing, the CUDA memory will overhead after training 25 epochs,
			I don't know the reason, maybe more connection in fc layer. 
		'''
		output, _ = self.lstm(input, self.hidden)
		output=output[:,-1,:]
		if(self.isFC):
			output = self.fc_single(output)
		return output









