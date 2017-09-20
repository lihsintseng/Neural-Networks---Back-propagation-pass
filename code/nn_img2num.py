# Credit: Li-Hsin Tseng
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
'''
Update your code in order to create your network, perform forward and back-prop using Pytorch’s nn package.
In order to update parameters use optim package.
Compare speed and training error vs epochs charts.
'''

class NnImg2Num(nn.Module):

	def __init__(self):
		super(NnImg2Num, self).__init__()
		self.error = []
		self.fc1 = nn.Linear(28*28, 256)
		self.fc2 = nn.Linear(256, 64)
		self.fc3 = nn.Linear(64, 10)
		self.loss_fn = nn.MSELoss()
		self.m = nn.Sigmoid()
		root, download, self.batch_size = './data', False, 64
		# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
		
		self.train_loader = torch.utils.data.DataLoader(
		    datasets.MNIST(root, train=True, download=download,
		                   transform=transforms.Compose([
		                       transforms.ToTensor(),
		                       transforms.Normalize((0.1307,), (0.3081,))
		                   ])),
		    batch_size=self.batch_size, shuffle=True)

	# [nil] train()

	def train(self):
		eta = 1
		optimizer = optim.SGD(self.parameters(), lr = eta)
		for batch_idx, (in_data, in_target) in enumerate(self.train_loader):
			size = len(in_data)
			optimizer.zero_grad()
			data, target = torch.zeros(size, 28*28), torch.zeros(size, 10)
			for i in range(size):
				target[i][in_target[i]] = 1
				tmp = in_data[i].view(28*28)
				for j in range(28*28):
					data[i][j] = tmp[j]
			data, target = Variable(data), Variable(target)
			# forward
			tmp = self.m(self.fc1(data))
			tmp = self.m(self.fc2(tmp))
			tmp = self.m(self.fc3(tmp))
			# backword
			output = self.loss_fn(self.m(tmp), target)

			if batch_idx % 10 == 0: self.error.append(output)
			
			output.backward()
			# update params
			optimizer.step()

	# [int] forward([28x28 ByteTensor] img)
	def forward(self, img):
		img = Variable(img.view(28*28))
		tmp = self.m(self.fc1(img))
		tmp = self.m(self.fc2(tmp))
		tmp = self.m(self.fc3(tmp))
		_, idx = torch.max(tmp, 0)
		return idx