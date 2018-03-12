import numpy as np
from sklearn import datasets


#initialize parameters(w,b)
def initialize_parameters(layer_dims):
	"""
	:param layer_dims: list,每一层单元的个数（维度）
	:return:dictionary,存储参数w1,w2,...,wL,b1,...,bL
	"""
	L = len(layer_dims)#the number of layers in the network
	parameters = {}
	for l in range(1,L):
		parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
	return parameters
#Implement the linear part of a layer's forward propagation: z = w[l] * a[l-1] + b[l]
def linear_forward(A_pre,W,b):
	"""
	:param A_pre:上一层的激活值,shape:(size of previous layer,m)
	:param W: weight matrix,shape:(size of current layer,size of previous layer)
	:param b: bias vector,shape:(size of current layer,1)
	:return:
	Z：激活函数的输入值（线性相加和）
	cache：因为bp的时候要用到w，b，a所以要把每一层的都存起来，方便后面用
	"""
	Z = np.dot(W,A_pre) + b
	cache = (A_pre,W,b)
	return Z,cache
#implement the activation function(ReLU and sigmoid)
def relu(Z):
	"""
	:param Z: Output of the linear layer
	:return:
	A: output of activation
	activation_cache: 要把Z保存起来，因为后面bp，对relu求导，求dz的时候要用到
	"""
	A = np.maximum(0,Z)
	activation_cache = Z #要把Z保存起来，因为后面bp，对relu求导，求dz的时候要用到
	return A, activation_cache
#implement the activation function(ReLU and sigmoid)
def sigmoid(Z):
	"""
	:param Z: Output of the linear layer
	:return:
	"""
	A = 1 / (1 + np.exp(-Z))
	activation_cache = Z
	return A,activation_cache
#calculate the output of the activation
def linear_activation_forward(A_pre,W,b,activation):
	"""
	:param A_pre: activations from previous layer,shape(size of previous layer, number of examples)
	:param W:weights matrix,shape(size of current layer, size of previous layer)
	:param b:bias vector, shape(size of the current layer, 1)
	:param activation:the activation to be used in this layer(ReLu or sigmoid)
	:return:
	A: the output of the activation function
	cache: tuple,形式为:((A_pre,W,b),Z),后面bp要用到的((A_pre,W,b),Z)
	"""
	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_pre,W,b)#linear_cache:(A_pre,W,b)
		A, activation_cache = sigmoid(Z)# activation_cache: Z
	elif activation == "relu":
		Z, linear_cache = linear_forward(A_pre, W, b)#linear_cache:(A_pre,W,b)
		A, activation_cache = relu(Z)# activation_cache: Z
	cache = (linear_cache, activation_cache)
	return A, cache
# Implement the forward propagation of the L-layer model
def L_model_forward(X,parameters):
	"""
	:param X: data set,input matrix,shape(feature dimensions,number of example)
	:param parameters: W,b
	:return:
	AL: activation of Lth layer i.e. y_hat(y_predict)
	caches: list,存储每一层的linear_cache(A_pre,W,b),activation_cache(Z)
	"""
	caches = []#用于存储每一层的，A_pre,W,b,Z
	A = X
	L = len(parameters) // 2 # number of layer
	#calculate from 1 to L-1 layer activation
	for l in range(1,L):
		A_pre = A
		W = parameters["W" + str(l)]
		b = parameters["b" + str(l)]
		A, cache = linear_activation_forward(A_pre, W, b, "relu")
		caches.append(cache)
	#calculate Lth layer activation
	AL, cache = linear_activation_forward(A,parameters["W" + str(L)],parameters["b" + str(L)],"sigmoid")
	caches.append(cache)
	return AL,caches
#calculate cost function
def compute_cost(AL,Y):
	"""
	:param AL: 最后一层的激活值，即预测值，shape:(1,number of examples)
	:param Y:真实值,shape:(1, number of examples)
	:return:
	"""
	m = Y.shape[0]
	cost = -1/m * np.sum(Y*np.log(AL)+(1-Y)*np.log(1- AL))#py中*是点乘
	#从数组的形状中删除单维条目，即把shape中为1的维度去掉，比如把[[[2]]]变成2
	cost = np.squeeze(cost)
	return cost
#calculate dA_pre,dW,db
def linear_backward(dZ, cache):
	"""
	:param dZ:
	:param cache: 前面fp保存的linear_cache(A_pre,W,b)
	:return:
	"""
	A_prev, W, b = cache
	m = A_prev.shape[1]
	dW = 1/m * np.dot(dZ,A_prev.T)#有时候不敢确定是线代乘还是点乘，有个小trick就是dW维度一定和W保持一致，这样就好确定是np.dot()还是*了
	db = 1/m * np.sum(dZ,axis=1,keepdims=True)
	dA_pre = np.dot(W.T,dZ)
	return dA_pre,dW,db

def sigmoid_backward(dA, Z):
	"""
	:param dA:
	:param Z:
	:return:
	"""
	a = 1/(1 + np.exp(-Z))
	dZ = dA * a*(1-a)
	return dZ

def relu_backward(dA, Z):
	"""
	:param dA:
	:param z:
	:return:
	"""
	dZ = np.copy(dA)
	dZ[Z <= 0] = 0 #先复制dA，然后知道Z<=0对应的位置，把该位置元素置为0即可
	return dZ

def linear_activation_backward(dA, cache, activation):
	"""
	:param dA:
	:param cache:
	:param activation:
	:return:
	"""
	linear_cache, activation_cache = cache#((A_pre,W,b),Z)
	if activation == "relu":
		dZ = relu_backward(dA,activation_cache)
		dA_pre, dW, db = linear_backward(dZ,linear_cache)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_pre, dW, db = linear_backward(dZ, linear_cache)
	return dA_pre, dW, db

# Implement the backward propagation of the L-layer model
def L_model_backward(AL, Y, caches):
	"""
	:param AL: 最后一层激活值（i.e y_hat）
	:param Y: 实际类别(0,1)
	:param caches: fp时各层的((A_pre,W,b),Z)
	:return:
	"""
	grads = {}#存放各层的dW，db
	L = len(caches)
	# 这个地方之所以没有1/m，是因为对Z,A等中间变量求导时，直接使用的是交叉熵函数对Z,A求导，
	# 而不是cost function，只有对W，b求导时使用cost function
	#第L层单独算,因为激活函数是sigmoid
	dAL = -(np.divide(Y,AL) - np.divide((1-Y),(1-AL)))
	current_cache = caches[L - 1]
	grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] \
		= linear_activation_backward(dAL,current_cache,"sigmoid")
	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dA_pre_temp, dW_temp, db_temp \
			= linear_activation_backward(grads["dA" + str(l + 1)],current_cache,"relu")
		grads["dA" + str(l)] = dA_pre_temp
		grads["dW" + str(l+1)] = dW_temp
		grads["db" + str(l+1)] = db_temp
	return grads
# update w,b
def update_parameters(parameters, grads, learning_rate):
	"""
	:param parameters: dictionary,  W,b
	:param grads: dW,db
	:param learning_rate: alpha
	:return:
	"""
	L = len(parameters) // 2
	for l in range(L):
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
	return parameters

def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations):
	"""
	:param X:
	:param Y:
	:param layer_dims:list containing the input size and each layer size
	:param learning_rate:
	:param num_iterations:
	:return:
	parameters：final parameters:(W,b)
	"""
	# initialize parameters
	parameters = initialize_parameters(layer_dims)
	for i in range(0, num_iterations):
		#foward propagation
		AL,caches = L_model_forward(X, parameters)
		# calculate the cost
		cost = compute_cost(AL, Y)
		#backward propagation
		grads = L_model_backward(AL, Y, caches)
		#update parameters
		parameters = update_parameters(parameters, grads, learning_rate)
	return parameters

#predict function
def predict(X,y,parameters):
	"""
	:param X:
	:param y:
	:param parameters:
	:return:
	"""
	m = y.shape[1]
	Y_prediction = np.zeros((1, m))
	prob, caches = L_model_forward(X,parameters)
	for i in range(prob.shape[1]):
		# Convert probabilities A[0,i] to actual predictions p[0,i]
		### START CODE HERE ### (≈ 4 lines of code)
		if prob[0, i] > 0.5:
			Y_prediction[0, i] = 1
		else:
			Y_prediction[0, i] = 0
	accuracy = 1- np.mean(np.abs(Y_prediction - y))
	return accuracy
#DNN model
def DNN(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000):
	parameters = L_layer_model(X, Y, layer_dims, learning_rate, num_iterations)
	accuracy = predict(X,Y,parameters)
	return accuracy
if __name__ == "__main__":
	X_train,y_train = datasets.load_breast_cancer(return_X_y=True)
	features = X_train.shape[1]
	# print(X_train.shape)
	# print(y_train.shape)
	y_train = y_train.reshape(y_train.shape[0], -1).T
	#print(y_train.shape)
	accuracy = DNN(X_train.T,y_train,[features,10,7,5,1])
	print(accuracy)