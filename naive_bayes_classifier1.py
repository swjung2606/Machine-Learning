import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *




# make a table for likelihood by considering all documents from one category as one large document
def likelihood_table_generator(raw_data):
    
    likelihood_table_1 = pd.DataFrame(columns=["E", "P", "TT"])
    
    
    # Add all rows by keywords
    # agency
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].agency), 
                     "P" : sum(raw_data[raw_data.Category == "P"].agency), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].agency)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[1])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    
    
    # airplane
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].airplane), 
                     "P" : sum(raw_data[raw_data.Category == "P"].airplane), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].airplane)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[2])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)


    # beach
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].beach), 
                     "P" : sum(raw_data[raw_data.Category == "P"].beach), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].beach)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[3])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    

    # boat
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].boat), 
                     "P" : sum(raw_data[raw_data.Category == "P"].boat), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].boat)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[4])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)



    # city
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].city), 
                     "P" : sum(raw_data[raw_data.Category == "P"].city), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].city)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[5])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    
    
    # company
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].company), 
                     "P" : sum(raw_data[raw_data.Category == "P"].company), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].company)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[6])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    
    
    # government
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].government), 
                     "P" : sum(raw_data[raw_data.Category == "P"].government), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].government)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[7])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    
    
    # island
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].island), 
                     "P" : sum(raw_data[raw_data.Category == "P"].island), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].island)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[8])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    
    
    # journey
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].journey), 
                     "P" : sum(raw_data[raw_data.Category == "P"].journey), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].journey)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[9])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    
    
    # law
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].law), 
                     "P" : sum(raw_data[raw_data.Category == "P"].law), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].law)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[10])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    
    
    # president
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].president), 
                     "P" : sum(raw_data[raw_data.Category == "P"].president), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].president)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[11])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    
    
    # tax
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].tax), 
                     "P" : sum(raw_data[raw_data.Category == "P"].tax), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].tax)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[12])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    
    
    # turnover
    values_to_add = {"E" : sum(raw_data[raw_data.Category == "E"].turnover), 
                     "P" : sum(raw_data[raw_data.Category == "P"].turnover), 
                     "TT" : sum(raw_data[raw_data.Category == "TT"].turnover)
                     }
    row_to_add = pd.Series(values_to_add, name=raw_data.columns[13])
    
    likelihood_table_1 = likelihood_table_1.append(row_to_add)
    
    a = likelihood_table_1

    
    
    
    # From here, we will change the table as likelihood values
    denominator_E = sum(a.E) + len(a.E)
    denominator_P = sum(a.P) + len(a.P)
    denominator_TT = sum(a.TT) + len(a.TT)
    
    for i in range(len(a.E)):
        a.E[i] = (a.E[i]+1) / denominator_E
        a.P[i] = (a.P[i]+1) / denominator_P
        a.TT[i] = (a.TT[i]+1) / denominator_TT
    
    return a




def classifier(sample:dict ):
    
    a = likelihood_table_generator(raw_data)
    P_TT = len(raw_data[raw_data.Category == "TT"]) / len(raw_data)
    P_P = len(raw_data[raw_data.Category == "P"]) / len(raw_data)
    P_E = len(raw_data[raw_data.Category == "E"]) / len(raw_data)
    
    
    sum_TT = log(P_TT, 10)
    sum_P = log(P_P, 10)
    sum_E = log(P_E, 10)
    
    
    for i in sample:
        for j in range(len(a)):
            if i == a.index[j]:
                sum_TT += log( (a.TT[j])**sample[i] , 10)
                sum_P += log( (a.P[j])**sample[i] , 10)
                sum_E += log( (a.E[j])**sample[i] , 10)
    
    print("P(E|d) = ", sum_E)
    print("P(P|d) = ", sum_P)
    print("P(TT|d) = ", sum_TT)
    print("")
    
    if max(sum_E, sum_P, sum_TT) == sum_E:
        print("The sample belongs to Economics Documents.")
    elif max(sum_E, sum_P, sum_TT) == sum_P:
        print("The sample belongs to Politics Documents.")
    elif max(sum_E, sum_P, sum_TT) == sum_TT:
        print("The sample belongs to Tourism Documents.")

        

# Generate raw_data from excel file
raw_data = pd.read_csv("naive_bayes_classifier1.csv")

# Given sample 
sample1 = {"agency" : 1, "island" : 2, "journey" : 2, "beach" : 4}

# Claasify the sample
classifier(sample1)
     
                 
    



