import math
import numpy as np
import pandas as pd

#Creates a dictionary where the keys are the target values
#and classifies the records based on the target value.
#Appending the whole record except the last column(The target value)

def separateByClass(dataset):
    separate={}
    for i in range(len(dataset)):
        if(dataset[i][-1] not in separate):
            separate[dataset[i][-1]]=[]
        separate[dataset[i][-1]].append(dataset[i][0:-1])
    return separate


def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg=mean(numbers)
    varience=sum([math.pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(varience)




#Takes in a set of class records,Here,Its "1.0" & "0.0". 
#Summarizes the mean and stddev of each attribute for those set of class records.

def getclassdetails(class_records):
    mean_std=[(mean(attribute),stdev(attribute)) for attribute in zip(*class_records)]
    return mean_std




#Here,We are creating another dictionary after segregating the records
#The values for each key is the set of mean and stdev instead of set of records.

def summarizeByClass(dataset):
    separated=separateByClass(dataset)
    summaries={}
    #items returns key,value pair in the dictionary.
    for classval, class_records in separated.items():
        summaries[classval]=getclassdetails(class_records)
    return summaries




#Calculates the probabilty given an attribute x

def cal_attr_probability(mean,stdev,x):
    expo=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    result=(1/(math.sqrt(2*math.pi)*stdev))*expo
    return result


#Calculates the probabilty of all classes(Target value) for
#one record based on summaries

def calprobability(summaries,record):
    probability={}
    #items returns key,value pair in the dictionary
    for classval, class_det in summaries.items():
        probability[classval]=1
        for i in range(len(class_det)):
            mean,stdev=class_det[i]
            attr=record[i]
            probability[classval]*=cal_attr_probability(mean,stdev,attr)
    return probability



#Takes in one record and classifies it

def predict_this(summaries,record):
    probability=calprobability(summaries,record)
    bestlabel,bestprob=None, -1
    #Finding the classval with the most probabilty
    for classval, prob in probability.items():
        if bestlabel is None or prob>bestprob:
            bestprob=prob
            bestlabel=classval
    return bestlabel





#The real main function. Takes in testset,Returns Classfied class values array

def prediction(summaries,testset):
    predict=[]
    for i in range(len(testset)):
        result=predict_this(summaries,testset[i])
        predict.append(result)
    return predict
        

    
# Finds accuracy of prediction

def getaccuracy(testset,prediction):
    correct=0
    for i in range(len(testset)):
        if testset[i][-1]==prediction[i]:
            correct+=1
    return (float(correct)/float(len(testset)))*100.0
    


def main():
    trainingset=np.array(pd.DataFrame(pd.read_csv('NaiveBayesDiabetes.csv')),float)
    testset=np.array(pd.DataFrame(pd.read_csv('NaiveBayesDiabetes1.csv')),float)
    print("records in training dataset={0} and test dataset={1} rows".format(len(trainingset),len(testset)))
    summaries=summarizeByClass(trainingset)
    predict=prediction(summaries,testset)
    print(predict)
    accuracy=getaccuracy(testset,predict)
    print(accuracy)

main()
