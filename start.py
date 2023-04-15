import numpy as np
import pandas as pd
import networkx as nx
import pickle
from scipy import optimize
from scipy.sparse import csr_array

def none_check(x):
    if x==None:
        return 0
    else:
        return x

def Weisfeiler_Lehman_kernel(data1,data2,h=4):
    n,m = len(data1),len(data2)
    K = np.zeros((n,m))
    l_i = []
    l_j = []
    for i in range(n):
        dict_i = {}
        dict_wli = nx.weisfeiler_lehman_subgraph_hashes(data1[i],edge_attr="labels",node_attr="labels",iterations=h)
        for x in dict_wli:
            for y in dict_wli[x]:
                dict_i[y] = none_check(dict_i.get(y)) + 1
        l_i.append(dict_i)
    for j in range(m):
        dict_j = {}
        dict_wlj = nx.weisfeiler_lehman_subgraph_hashes(data2[j],edge_attr="labels",node_attr="labels",iterations=h)
        for x in dict_wlj:
            for y in dict_wlj[x]:
                dict_j[y] = none_check(dict_j.get(y)) + 1
        l_j.append(dict_j)
    for i in range(n):
        for j in range(m):
            dict_i, dict_j = l_i[i], l_j[j]
            common_keys = dict_i.keys() & dict_j.keys()
            for x in common_keys:
                K[i,j] += dict_i[x] * dict_j[x]
    return K
    
    def sigmoid (x):
    return 1/(1+np.exp(-x))

class KernelLogistique:
    def __init__(self, lamda, kernel,precomputed = False, b=555/5445): # b = initial odds
        self.lamda = lamda
        self.kernel = kernel
        self.alpha = None
        self.b = np.log(b)
        self.precomputed = precomputed
        self.support = None
    
    def fit(self, X, y):
        N = len(y)
        if self.precomputed:
            K = pd.read_csv(f"kernel{N}.csv").iloc[:,1:].to_numpy()
        else:
            K = self.kernel(X,X)
            pd.DataFrame(K).to_csv(f"kernel{N}.csv")
        # Lagrange dual problem
        def loss(alpha):
            K_a = K@alpha
            aux = np.logaddexp(0,-y*(K_a + self.b))
            aux = np.mean(aux)
            return  (self.lamda/2) * alpha @ K_a + aux
        #'''--------------dual loss ------------------ '''

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            K_a = K@alpha
            aux = -y*sigmoid(-y*(K_a+ self.b))
            aux = K@aux/N
            return self.lamda * K_a + aux
        # '''----------------partial derivative of the dual loss wrt alpha -----------------'''
        
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.zeros(N), 
                                   method='BFGS', 
                                   jac=lambda alpha: grad_loss(alpha))
        print(optRes)
        self.alpha = optRes.x
        self.support = X
    
    
    def predict_proba(self, X):
        """ Predict proba classe 1"""
        K_x = self.kernel(X,self.support)
        f_x = K_x @self.alpha
        return sigmoid(f_x + self.b)
    
    def predict_loggit(self, X):
        K_x = self.kernel(X,self.support)
        f_x = K_x @self.alpha
        return f_x + self.b

with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
with open('training_labels.pkl', 'rb') as f:
    label = pickle.load(f)
    
############# Entraînement ############
lamda = 2
Kernel = lambda x,y: Weisfeiler_Lehman_kernel(x,y,h=4)
model = KernelLogistique(lamda=lamda, kernel=Kernel,precomputed = False)
model.fit(data[0:6000], 2*label[0:6000]-1)
#######################################

with open('test_data.pkl', 'rb') as f:
    test = pickle.load(f)

########### Prédiction ###############
y_test = model.predict_loggit(test)
DF = pd.DataFrame(y_test-np.quantile(y_test,q=.9),columns=["Predicted"])
DF.index += 1
DF.to_csv('test_pred.csv',index_label='Id')
