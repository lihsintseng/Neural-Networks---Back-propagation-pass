# Credit: Li-Hsin Tseng
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import torch
import nn_img2num
import my_img2num
import numpy as np
import matplotlib.pyplot as plt

path = './error/'

root, download, batch_size, test_batch_size = './data', False, 64, 1

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


def test_nn(epoch_num):
	### epoch size = epoch_num
	# NnImg2Num
	nn = nn_img2num.NnImg2Num()
	nn_start = time.time()
	for i in range(epoch_num):
		nn.train()
	nn_end = time.time()
	nn_time = nn_end - nn_start
	nn_cnt = 0

	np.save(path + 'nn_' +str(epoch_num) + '.npy', nn.error)

	for batch_idx, (in_data, target) in enumerate(test_loader):
		nn_res = nn.forward(in_data)
		if (torch.eq(nn_res.data, target) == 1).numpy(): nn_cnt += 1

	print(str(epoch_num) + ' epoch')
	print('NnImg2Num time (sec):', end = '')
	print(nn_time)
	print('NnImg2Num accuracy:', end = '')
	print(nn_cnt/len(test_loader))

def test_mine(epoch_num):
	### epoch size = epoch_num
	# MyImg2Num
	mine = my_img2num.MyImg2Num()
	mine_start = time.time()
	for i in range(epoch_num):
		mine.train()
	mine_end = time.time()
	mine_time = mine_end - mine_start
	mine_cnt = 0

	np.save(path + 'mine_' +str(epoch_num) + '.npy', mine.error)

	for batch_idx, (in_data, target) in enumerate(test_loader):
		mine_res = mine.forward(in_data)
		if (torch.eq(mine_res, target) == 1).numpy():
			mine_cnt += 1

	print(str(epoch_num) + ' epoch')
	print('MyImg2Num time (sec):', end = '')
	print(mine_time)
	print('MyImg2Num accuracy:', end = '')
	print(mine_cnt/len(test_loader))


### epoch size
for i in [1, 10, 30]:
	test_nn(i)
	test_mine(i)

### errors during training
nn_error_30 = np.load(path + 'nn_30' + '.npy')
mine_error_30 = np.load(path + 'mine_30' + '.npy')

tmp_nn, tmp_mine = [], []
for i in range(len(nn_error_30.flatten())):
	if (i+1) % 94 == 0:
		tmp_nn.append(nn_error_30.flatten()[i].data.numpy())
		tmp_mine.append(abs(mine_error_30[i]))

nn_error_30 = np.reshape(tmp_nn, (len(tmp_nn)))
mine_error_30 = np.reshape(tmp_mine, (len(tmp_mine)))

x = [i for i in range(len(nn_error_30))]

plt.figure(1)
plt.plot(x, nn_error_30, 'r')
plt.legend()
plt.xlabel('epoch number')
plt.ylabel('output error')

plt.title('nn img2num error vs. epoch number')
plt.tight_layout()
plt.show()


plt.figure(2)
plt.plot(x, mine_error_30, 'r')
plt.legend()
plt.xlabel('epoch number')
plt.ylabel('output error')

plt.title('mine img2num error vs. epoch number')
plt.tight_layout()
plt.show()


'''
1 epoch
NnImg2Num time (sec):153.33218598365784
NnImg2Num accuracy:0.1135
1 epoch
MyImg2Num time (sec):565.1161580085754
MyImg2Num accuracy:0.1035
10 epoch
NnImg2Num time (sec):1526.6758909225464
NnImg2Num accuracy:0.1254
10 epoch
MyImg2Num time (sec):5623.603133201599
MyImg2Num accuracy:0.1452
30 epoch
NnImg2Num time (sec):4562.95757484436
NnImg2Num accuracy:0.1135
30 epoch
MyImg2Num time (sec):25038.49795818329
MyImg2Num accuracy:0.1035
'''


### errors v.s. epochs
x = [1,10, 30]
nn_acc = [0.1135, 0.1254, 0.1135]
nn_time = [153.33218598365784, 1526.6758909225464, 4562.95757484436]
mine_acc = [0.1035, 0.1452, 0.1035]
mine_time = [565.1161580085754, 5623.603133201599, 25038.49795818329]

plt.figure(3)
plt.plot(x, nn_acc, 'r', label = 'nn')
plt.plot(x, mine_acc, 'b', label = 'mine')
plt.legend()
plt.xlabel('epoch number')
plt.ylabel('accuracy')

plt.title('epoch number v.s. accuracy')
plt.tight_layout()
plt.show()

plt.figure(4)
plt.plot(x, nn_time, 'r', label = 'nn')
plt.plot(x, mine_time, 'b', label = 'mine')
plt.legend()
plt.xlabel('epoch number')
plt.ylabel('time')

plt.title('epoch number v.s. time')
plt.tight_layout()
plt.show()



