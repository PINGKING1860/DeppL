import numpy as np 
# 将一维数据创建为时间序列数据集
def new_dataset(dataset, step_size=3):
	data_X, data_Y = [], []
	for i in range(0,len(dataset)-step_size,4):
		a = dataset[i:(i+3)]
		data_X.append(a)
		data_Y.append(dataset[i + 3])
	data_Y = np.array(data_Y)
	data_Y = np.reshape(data_Y, (data_Y.shape[0], 1))
	return np.array(data_X), np.array(data_Y)
