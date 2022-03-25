import numpy as np
import time
import matplotlib.pyplot as plt
X=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
X=X/np.amax(X,axis=0)
y=y/100
lr=0.1
inputlayer_neurons=2
hiddenlayer_neurons=3
output_neurons=1


def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)

A=[]
B=[]
for i in range(5):
    epoch=700*i
    A.append(epoch) 
    hidden_weight=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
    hidden_bias=np.random.uniform(size=(1,hiddenlayer_neurons))
    output_weight=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
    output_bias=np.random.uniform(size=(1,output_neurons))

    start_time=time.time()

    for i in range(epoch):
        hidden_input=np.dot(X,hidden_weight)+hidden_bias
        hidden_output=sigmoid(hidden_input)
        
        output_input=np.dot(hidden_output,output_weight)+output_bias
        output_output=sigmoid(output_input)
        
        Output_Error=y-output_output
        Output_Difference=Output_Error*derivatives_sigmoid(output_output)
        
        Hidden_Error=Output_Difference.dot(output_weight.T)
        Hidden_Difference=Hidden_Error*derivatives_sigmoid(hidden_output)
        
        output_weight=output_weight+hidden_output.T.dot(Output_Difference)*lr
        output_bias=output_bias+np.sum(Output_Difference,axis=0,keepdims=True)*lr
        
        hidden_weight=hidden_weight+X.T.dot(Hidden_Difference)*lr
        hidden_bias=hidden_bias+np.sum(Hidden_Difference,axis=0,keepdims=True)*lr
        
    B.append((time.time() - start_time))

plt.plot(A, B)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('My first graph!')
plt.show()
print("input:\n" + str(X))
print("Actual output:\n" + str(y))
print("Predicted output \n"+str(output_output))
