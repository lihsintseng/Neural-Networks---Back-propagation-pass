# Credit: Li-Hsin Tseng
from torchvision import datasets, transforms
import torch
import NeuralNetwork
'''
Train the img2num network using the MNIST dataset using the NeuralNetwork API. 
To speed up training, you may want to use the NeuralNetwork API in batch mode 
(where you forward chunks of the design and target matrices at a time). 
This means that if size(x) = n and you have m examples, 
then size(X) = m × n (I know it is transpose of what you were providing 
in last homework) and NeuralNetwork’s forward() can take x or 
X and NeuralNetwork’s backward() can take y or Y. 
Output of NeuralNetwork’s forward() will be m × c 
(where m is number of examples and c is number of classes/categories).

Your img2num network will need view the 2D input into a 1D Tensor so that it can be fed to the network. The labels will need a oneHot encoding.
You will probably need to download the MNIST data set. Instruction about how to use the data set are provided here -> https://github.com/andresy/mnist.

no other helper functions (which must be local, if needed, like the oneHot() conversion function).
'''


class MyImg2Num(object):

	def __init__(self, ):
		self.error = []
		self.model = NeuralNetwork.NeuralNetwork([28*28, 256, 64, 10])
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
		loss, eta = 'MSE', 1
		for batch_idx, (in_data, in_target) in enumerate(self.train_loader):
			# data: 64*1*28*28, target: 64
			# reshape data, one-hot target
			#if batch_idx > 900:
			#print('idx = ' + str(batch_idx))
			size = len(in_data)
			#print('size = ' + str(size))
			data, target = torch.zeros(28*28, size), torch.zeros(10, size)
			for i in range(size):
				target[in_target[i]][i] = 1
				tmp = in_data[i].view(28*28)
				for j in range(28*28):
					data[j][i] = tmp[j]
			self.model.forward(data)
			self.model.backward(target, loss)

			if batch_idx % 10 == 0: self.error.append(self.model.error)

			self.model.updateParams(eta)

	# [int] forward([28x28 ByteTensor] img)
	def forward(self, img):
		# https://discuss.pytorch.org/t/any-alternatives-to-flat-for-tensor/3106/3
		input = img.view(28*28)
		tmp = self.model.forward(input)
		_, idx = torch.max(tmp, 0)
		return idx




