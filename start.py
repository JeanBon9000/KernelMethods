import numpy as np
import pandas as pd
import networkx as nx
import pickle
from scipy import optimize
from scipy.sparse import csr_array

########################## Walk kernel ########################################################
def graphe_produit(G_1,G_2):
    G_p = nx.Graph()
    labels_1 = nx.get_node_attributes(G_1,"labels")
    labels_2 = nx.get_node_attributes(G_2,"labels")
    for k in range(len(G_1)):
        for l in range(len(G_2)):
            if labels_1[k] == labels_2[l]:
                G_p.add_node((k,l))
    labels_1 = nx.get_edge_attributes(G_1,"labels")
    labels_2 = nx.get_edge_attributes(G_2,"labels")
    for x in G_p.nodes():
        for y in G_p.nodes():
            if ((x[0],y[0]) in labels_1) and ((x[1],y[1]) in labels_2):
                if labels_1[(x[0],y[0])] == labels_2[(x[1],y[1])]:
                    G_p.add_edge(x,y)
    return G_p

def Kernel_walk (data1,data2,n):
    K = np.zeros((len(data1),len(data2)))
    for k in range(len(data1)):
        for l in range(len(data2)):
            G1,G2 = data1[k],data2[l]
            G_p = graphe_produit(G1,G2)
            if len(G_p) >0:
                M = nx.adjacency_matrix(G_p)
                M_n = M.copy()
                for k in range(n-1):
                    M_n = M_n @ M
                K[k,l] = np.ones(M.shape[0]).T @ M_n @ np.ones(M.shape[0])
            else:
                K[k,l] = 0
    return K
#############################################################################################

############################### Shortest path kernel #########################################
def feature_vector_sp(data,maxlength = 105, maxlabel = 50,transpose = False):
    n = len(data)
    if transpose:
        features_array = np.zeros((maxlabel*maxlabel*maxlength,n),dtype = int)
    else:
        features_array = np.zeros((n,maxlabel*maxlabel*maxlength),dtype = int)
    for k in range(n):    
        dictlen1 = dict(nx.shortest_path_length(data[k]))
        dictnode1 = nx.get_node_attributes(data[k],"labels")
        features = np.zeros((maxlabel,maxlabel,maxlength),dtype = int)
        for i in range(len(data[k])):
            for j in dictlen1[i]:
                x,y,z = dictnode1[i],dictnode1[j],dictlen1[i][j]
                features[x,y,z] += 1
        if transpose:
            features_array[:,k] = np.ravel(features)
        else:
            features_array[k] = np.ravel(features)
    features_array = csr_array(features_array)
    return features_array

def shortest_path_kernel(data1,data2,maxlength = 105, maxlabel = 50): #data = liste de graphes
    features1 = feature_vector_sp(data1,maxlength = maxlength, maxlabel = maxlabel)
    features2 = feature_vector_sp(data2,maxlength = maxlength, maxlabel = maxlabel,transpose = True)
    return (features1@features2).todense()
#####################################################################################################


############################ Weisfeiler Lehman kernel ###############################################
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
######################################################################################################

####################################### SVC ###########################################################
class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-6,precomputed = False):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.b = None
        self.norm_f = None
        self.precomputed = precomputed
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        Y = np.diag(y)
        
        if self.precomputed:
            K = pd.read_csv(f"kernel{N}.csv").iloc[:,1:].to_numpy()
        else:
            K = self.kernel(X,X)
            pd.DataFrame(K).to_csv(f"kernel{N}.csv")
        # Lagrange dual problem
        def loss(alpha):
            return  alpha @ K @ alpha/2 - alpha@y 
        #'''--------------dual loss ------------------ '''

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return K @ alpha - y
        # '''----------------partial derivative of the dual loss wrt alpha -----------------'''


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha:  np.sum(alpha) # '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha:   np.ones(N)#'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha:  np.hstack((self.C*np.ones(N), np.zeros(N))) -  np.hstack((y*alpha, - y*alpha)) # '''---------------function defining the inequality constraint-------------------'''     
        jac_ineq = lambda alpha:   np.vstack((-Y, Y)) # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},{'type': 'ineq','fun': fun_ineq,'jac': jac_ineq})
        
        #bounds = optimize.Bounds(lb = 0,ub = self.C)
        
        

        
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.zeros(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x
        print(optRes)
        ## Assign the required attributes

        booleen = np.logical_and(y*self.alpha>self.epsilon,y*self.alpha<self.C-self.epsilon)
        self.margin_points =  [X[k] for k in range(len(X)) if booleen[k]]
        #'''------------------- A matrix with each row corresponding to a point that falls on the margin ------------------'''
        
        #''' -----------------Verification------------------ '''
        f_vect = K@self.alpha
        offset_vect = y[booleen] - f_vect[booleen]
        print(np.std(offset_vect),offset_vect[0],np.mean(offset_vect))
        self.b = np.median(offset_vect)
        #''' -----------------Verification------------------ '''
        
        #self.b =  np.median(offset_vect)#''' -----------------offset of the classifier------------------ '''
        self.norm_f = np.sqrt(self.alpha @ K @ self.alpha)
        # '''------------------------RKHS norm of the function f ------------------------------'''
        #Support + mise à jour alpha pour le calcul de separting function
        booleen = y*self.alpha >self.epsilon
        self.support = [X[k] for k in range(len(X)) if booleen[k]]
        self.alpha =self.alpha[booleen]

    ### Implementation of the separting function $f$ 
    def separating_function(self,x): 
        K_x = self.kernel(x,self.support)
        return K_x @self.alpha
    
    
    def predict(self, X):
        """ Predict y values in {0, 1} """
        d = self.separating_function(X)
        return 2*(d> 0)-1
#################################################################################


################################## Kernel Logistic RIDGE #########################
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
#############################################################################

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
