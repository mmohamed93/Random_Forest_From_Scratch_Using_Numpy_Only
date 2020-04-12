#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math


# In[3]:


class TreeEnsemble():
    def __init__(self,x,y,n_trees,sample_sz,min_leaf=5, bootstrap = False, random_state = None, max_features = 1):
        #the seed will be set for a constant for now to be able to compare with sklearn tree
        if random_state != None:
            np.random.seed(random_state)
        self.x,self.y,self.sample_sz,self.min_leaf,self.bootstrap, = x,y,sample_sz,min_leaf,bootstrap
        self.max_features = max_features
        #here we create trees by calling a function create tree for the required number of trees
        self.trees = [self.create_tree() for i in range(n_trees)]
    def create_tree(self):
        if self.bootstrap == True:
            idxs = []
            for i in range(self.sample_sz):
                idxs.append(np.random.randint(0,len(self.y)))
            idxs = np.array(idxs)
        else:
            idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        #iloc[row,column] or iloc[[list of rows],[list of columns]] .. columns are optional 
        return DecisionTree(self.x.iloc[idxs],self.y[idxs],min_leaf = self.min_leaf, max_features = self.max_features)
#         return idxs
    def predict(self,x):
        #the below is not a recursive function the predict below will be done for a decision tree
        #we will calculate the prediction of each tree for given x (features) and average using np.mean
        #axis = 0 will return the average of matrix rows
        return np.mean([tree.predict(x) for tree in self.trees],axis = 0)
    @property
    def _feature_importance(self):
        fi = []
        for i in range(self.x.shape[1]):
            shuf = self.x.iloc[:,i].copy()
            self.x.iloc[:,i] = np.random.permutation(self.x.iloc[:,i])
            fi.append(rmse(self.predict(self.x.values),self.y))
            self.x.iloc[:,i] = shuf
        return np.array(fi)


# In[4]:


class DecisionTree():
    #the indices are there to keep track of which of the row indixes went to the left and the right side of the tree
    def __init__ (self,x,y,idxs = None,min_leaf = 5, max_features = 1):
        # y passed to the decision tree will be the random selection of the treeensemble
        if idxs is None: idxs = np.arange(len(y))
        self.x, self.y, self.idxs, self.min_leaf = x,y,idxs,min_leaf
        # we need to keep track of the number of rows and number of columns
        self.n,self.c = len(idxs), x.shape[1]
        self.max_features = max_features
        #every node of the tree will have a value(prediction) which is equal to the mean of the node indexes
        self.val = np.mean(y[idxs])
        #some nodes of the tree will have a score which is how effective is the split only if it is not a LEAF NODE
        #at the begining we have not created any splits yet so the score will be infinity
        self.score = float('inf')
        #the next step is to determine which variable we will split on and what level we will split on
        self.find_varsplit()
        
    def find_varsplit(self):
        # max features will be a number from 0 to 1 randomly select a ratio of the columns to look for the next split
        #we will create a random list of all columns indexes and return only the ratio determined by max_features
        for i in np.random.permutation(range(self.c))[:int(self.max_features*self.c)]:
            self.find_better_split(i)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        self.lhs = DecisionTree(self.x,self.y,self.idxs[lhs])
        self.rhs = DecisionTree(self.x,self.y,self.idxs[rhs])

    #another method to calculate the standard deviation is square root of (the mean of squares minues square of the mean)
    #the function takes the count to calculate the means, sum of values and sum of squared values.
#     def std_agg(cnt,s1,s2): return math.sqrt((s2/cnt)  - (s1/cnt)**2)
    
    def find_better_split(self,var_idx):
        #x will be all the value in a specific column at certain indexes (when the tree splits the indexes will change so
        #we need to mark the idexes for the next split
        #y is the value of the indexes
        
        x,y = self.x.values[self.idxs,var_idx],self.y[self.idxs]
        
        #we will sort all the values based on X
        sort_idx = np.argsort(x)
        #create a sorted x and y based on the sorting indexes from argsort
        x_sorted,y_sorted = x[sort_idx],y[sort_idx]
        #the RHS will be initialzed with all value (count, sums and sums squares)
        rhs_cnt,rhs_sum,rhs_sum2 = self.n, y_sorted.sum(),(y_sorted**2).sum()
        #the LHS will be empty and we will start moving a value one by one and caluclate the standard deviation
        lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.
        
        #now we will loop through each index in x up to the min leaf (to avoid having values less than min leaf in a side)
        
        for i in range(0,self.n-self.min_leaf-1):
            xi , yi = x_sorted[i],y_sorted[i]
            lhs_cnt +=1 ; rhs_cnt -=1
            lhs_sum += yi; rhs_sum -=yi
            lhs_sum2 += yi**2 ; rhs_sum2 -= yi**2
            
            # we have to check that i is more than min sample leaf (to avoid having values less than min leaf in a side)
            #we also have to check that the next index is not equal to include every similar number in each trial
            if i<self.min_leaf or xi == x_sorted[i+1]: continue
            
            #now we will calculate the std for each split trial
            
            lhs_std = std_agg(lhs_cnt,lhs_sum,lhs_sum2); rhs_std = std_agg(rhs_cnt,rhs_sum,rhs_sum2)
            
            #now we will calculate the weighted average std of RHS and LHS and compare to the best available split score
            
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
            
            if curr_score<self.score:
                self.var_idx,self.score,self.split = var_idx,curr_score, xi
            
    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x.values[self.idxs,self.var_idx]
    
    @property
    def is_leaf(self): return self.score == float('inf')
    
    def __repr__(self):
        s = f'n: {self.n}; val: {self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s
    
    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return t.predict_row(xi)


# In[5]:


def std_agg(cnt,s1,s2): return math.sqrt((s2/cnt)  - (s1/cnt)**2)
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

