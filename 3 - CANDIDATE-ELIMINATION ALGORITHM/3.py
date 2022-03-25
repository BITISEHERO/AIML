import numpy as np
import pandas as pd


# Loading Data from a CSV File
data = pd.DataFrame(pd.read_csv('finds.csv'))



# Separating concept features from Target
concepts = np.array(data.iloc[:,0:-1])


# Isolating target into a separate DataFrame
target = np.array(data.iloc[:,-1])
def learn(concepts, target):
    specific_h=[0,0,0,0,0,0]
    print ('s0',specific_h)
    specific_h = concepts[0].copy()
    print('s1',specific_h)
    general_h = [["?" for i in range(len(concepts[0]))] for j in range(len(concepts[0]))]
    print('g0',general_h)
    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(h)): # Change values in S & G only if values change
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
                    
        if target[i] == "No":
            for x in range(len(h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        print(f"s{i}",specific_h)
        print(f"g{i}",general_h)
        print()
                                
    # find indices where we have empty rows, meaning those that are unchanged
    indices = [i for i,val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        # remove those rows from general_h
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    
    # Return final values
    return specific_h,general_h
s_final, g_final = learn(concepts, target)
print('Final specific Hypothesis : ',s_final)
print('Final general Hypothesis :',g_final)
