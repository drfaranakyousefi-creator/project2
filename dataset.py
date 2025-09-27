import pandas as pd
import torch
import random
from torch.utils.data import  TensorDataset , DataLoader
import math
# مسیر فایل خود را اینجا وارد کنید


def filter_noisy_data(x , dataset_name): 
    item_id = {
        'metavision' : [  
                220045,  # Heart Rate
                220210,  # Respiratory Rate
                220179, 220180,  # Non-invasive BP Mean
                220052,  # Arterial BP Mean
                220277   # SpO2 (هدف پیش‌بینی) 
        ],
        'carevue' : [
                211,   # ضربان قلب
                618,   # نرخ تنفس
                52,    # متوسط فشار خون شریانی
                456,          # متوسط فشار خون NBP
                676, 678,     # دما
                646   # SpO2 (هدف پیش‌بینی)
        ]
    }

    filtered_df = x[x['itemid'].isin(item_id[dataset_name])].copy()
    return filtered_df

def extract_data_from_person(dataframe , window_lengh , dataset_name , target) : 
    sparse_data =[] 
    dense_data  =[] 
    label =[]
    if dataset_name == 'metavision'  : 
        N = 4 
    else : 
        N = 5
    W_sparse = [torch.zeros(N) for i in range(window_lengh)]
    W_dense = [[0 for i in range(window_lengh)] for i in range(N)]
    for index, row in dataframe.iterrows():
        item_id = row['itemid']
        value = row['value']


        try:
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                continue   # رد کردن مقادیر نامعتبر
        except (ValueError, TypeError):
            continue

    #data _order for metavision : (heart rate , respiratory rate , Non-invasive BP Mean , Arterial BP Mean)
    #data order for 'carevue' is (heart rate, respiratory rate, arterial BP mean, NBP mean, temperature)

        if (item_id == 646) |  (item_id==220277) :  # spo2
            if target == 'spO2' :   
                sparse_data.append(torch.stack(W_sparse , dim=0))
                dense_data.append(torch.tensor(W_dense).T)
                label.append(torch.tensor(value))
            elif target == 'BP'  : 
                a = torch.zeros(N )
                a[ 0] = torch.tensor(value)
                W_sparse.append(a)
                W_sparse.pop(0)
                W_dense[0].append(value)
                W_dense[0].pop(0)
            elif target == 'RR' : 
                a = torch.zeros( N )
                a[ 1] = torch.tensor(value)
                W_sparse.append(a)
                W_sparse.pop(0) 
                W_dense[1].append(value)
                W_dense[1].pop(0)   

        elif (item_id==52) | (item_id == 220052)  :
            if target == 'BP' : 
                sparse_data.append(torch.stack(W_sparse , dim=0))
                dense_data.append(torch.tensor(W_dense).T)
                label.append(torch.tensor(value))
            else : 
                a = torch.zeros(N )
                a[ 0] = torch.tensor(value)
                W_sparse.append(a)
                W_sparse.pop(0)
                W_dense[0].append(value)
                W_dense[0].pop(0)

        elif(item_id==618) | (item_id == 220210) :#RR
            if target == 'RR' : 
                sparse_data.append(torch.stack(W_sparse , dim=0))
                dense_data.append(torch.tensor(W_dense).T)
                label.append(torch.tensor(value))
            else : 
                a = torch.zeros( N )
                a[ 1] = torch.tensor(value)
                W_sparse.append(a)
                W_sparse.pop(0) 
                W_dense[1].append(value)
                W_dense[1].pop(0)      

        elif (item_id == 211) |  (item_id==220045) :   
            a = torch.zeros(N )
            a[2] = torch.tensor(value)
            W_sparse.append(a)
            W_sparse.pop(0)
            W_dense[2].append(value)
            W_dense[2].pop(0)

        elif ((item_id == 220179 ) |  (item_id==220180)):#metavision
            a = torch.zeros( N )
            a[3] = torch.tensor(value)
            W_sparse.append(a)
            W_sparse.pop(0)
            W_dense[3].append(value)
            W_dense[3].pop(0)

        elif(item_id==456)  :  
            a = torch.zeros( N )
            a[ 3] = torch.tensor(value)
            W_sparse.append(a)
            W_sparse.pop(0)
            W_dense[3].append(value)
            W_dense[3].pop(0)  

        elif ((item_id==678 ) | (item_id == 676))  : 
            a = torch.zeros(N )
            a[ 4] = torch.tensor(value)
            W_sparse.append(a)
            W_sparse.pop(0)
            W_dense[4].append(value)
            W_dense[4].pop(0)
    if len(sparse_data ) > 0  : 
        sparse_data = torch.stack(sparse_data , dim = 0)
        dense_data = torch.stack(dense_data , dim=0)
        data = torch.stack([sparse_data , dense_data] , dim = 1 )
        label =  torch.tensor(label)
        idx = list(range(label.shape[0]))
        random.shuffle(idx)
        data = data[idx , : , : , :]  
        label = label[idx]
    else : 
        data , label = None  , None

    return data , label 


  

def extract_data(dataset_name , df_chartevents , w ,  target ) : 
    totol_subject_ids  = df_chartevents['subject_id'].unique()
    all_user_data  = [] 
    all_labels = [] 
    for subject_id in totol_subject_ids : 
        subject_data = df_chartevents[df_chartevents['subject_id'] == subject_id]
        filtered_df = filter_noisy_data( subject_data, dataset_name)
        data , label = extract_data_from_person(filtered_df  , w , dataset_name , target )
        if label != None : 
            all_labels.append(label)
            all_user_data.append(data)
        #the function will return data of torch type tensor and shape is : (sample of this user , 2 , w , N)
        # N is 5 for 'carevue' and 4 for 'metavision'
    return torch.concat(all_user_data , dim=0) , torch.concat(all_labels , dim=0)


        


class data_preparing : 
    def __init__(self ,data_frame , dataset_name , w , test_size , target ='spO2' ) :  #target can be spo2 or BP or RR:raspiratory rate  
        self.data , self.label = extract_data(dataset_name , data_frame , w , target)
        self.test_size = test_size
    def load_test(self , batch_size ) : 
        start = int((1-self.test_size)*self.label.shape[0])
        dataset = TensorDataset(self.data[start: ,: ,  : , : ] , self.label[start:])
        test_loader = DataLoader(dataset , batch_size=batch_size , shuffle= True)
        return test_loader
    def load_train(self , batch_size ) : 
        end = int((1-self.test_size)*self.label.shape[0])
        dataset = TensorDataset(self.data[:end ,: ,  : , : ] , self.label[:end])
        train_loader = DataLoader(dataset , batch_size=batch_size , shuffle= True)
        return train_loader    


