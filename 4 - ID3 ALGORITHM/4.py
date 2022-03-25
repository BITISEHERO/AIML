import numpy as np
import pandas as pd
import math

class Node:
    def __init__(self,l):
        self.label=l
        self.branch={}

#Calculayes Entropy of a dataset
def entropy(data):
    total_ex=len(data)
    p_ex=len(data.loc[data['play']=='yes'])
    n_ex=len(data.loc[data['play']=='no'])
    en=0
    if(p_ex>0):
        en = -(p_ex/float(total_ex))*(math.log(p_ex,2)- math.log(total_ex,2))
    if(n_ex>0):
        en += -(n_ex/float(total_ex))*(math.log(n_ex,2)- math.log(total_ex,2))
    return en


#Calculates Gain of an attribute
def gain(data_s,attrib):
    values=set(data_s[attrib])
    gain=entropy(data_s)
    for val in values:
        gain= gain - len(data_s.loc[data_s[attrib]==val])/float(len(data_s))*entropy(data_s[data_s[attrib]==val])
    return gain


#Gets attribute with highest gain
def get_attr(data):
    attribute=" "
    max_gain=0
    for attr in data.columns[:-1]:
        g=gain(data,attr)
        if g>max_gain:
            max_gain=g
            attribute =attr
    return attribute



#Constructs Decision Tree
def decision_tree(data):
    root=Node("NULL")

    #If Entropy is 0, All data is Yes/No.
    if(entropy(data)==0):
        if(len(data.loc[data['play']=='yes']) == len(data)):
            root.label='yes'
        else:
            root.label='no'
        return root
    #If only one attribute is left,Tree is complete
    if(len(data.columns)==1):
        return
    else:
        #Get the attribute with highest gain.
        attr=get_attr(data)
        root.label=attr
        values=set(data[attr])
        for value in values:
            root.branch[value]=decision_tree(data.loc[data[attr]==value].drop(attr,axis=1))
        return root


def test(tree,test_str):
    if not tree.branch:
        return tree.label
    return test(tree.branch[str(test_str[tree.label])],test_str)



data=pd.read_csv('playtennis.csv')
print("Number of Records : ",len(data))
tree=decision_tree(data)
test_str={ 
"outlook" : "Sunny",
"temperature" : "Hot",
"humidity" : "high",
"wind" : "Weak" , 
}
print(test(tree,test_str))
