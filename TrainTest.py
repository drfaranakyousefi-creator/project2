from client_net import client_network
from dataset import data_preparing
from client_transmitter import Transmitter
import torch  
import torch.nn as nn 
import pandas as pd 
import numpy as np 

class HTTPS(nn.Module) : 
    def __init__(self , w , dataset_name,batch_size , server_url ) -> None:
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if dataset_name == 'metavision' : 
            N = 4
        else:
            N =5
        self.network = client_network(w , N ,lr=0.01).to(device)
        chartevents_path = "/content/drive/MyDrive/split_learning/CHARTEVENTS.csv"
        df_chartevents = pd.read_csv(chartevents_path)
        self.data = data_preparing(df_chartevents , dataset_name , w , test_size = 0.2 )
        self.transmittion = Transmitter(server_url , device)
        self.batch_size = batch_size 
        self.loss_fn = nn.MSELoss()

    def fit(self , epochs ): 
        history = {
            'loss_train' : [] , 
            'loss_test'  : []
        }
        for epoch in range(epochs) : 
            self.train_one_epoch()
            loss_train , loss_test = self.evaluate_one_epoch()
            print(f'''
            [epoch {epoch} / {epochs}    train_loss = {loss_train}    test_loss = {loss_test}]
            ''')
            loss_test =loss_test.item()
            loss_train =loss_train.item()
            history['loss_test'].append(loss_test)
            history['loss_train'].append(loss_train)

        return history

    def train_one_epoch(self) :
        for x , l in self.data.load_train(batch_size = self.batch_size) :  
            prediction_inp , dense_decoder_out  , dense_inp  = self.network(x.to(self.device))
            grad = self.transmittion.send_data(prediction_inp , l , status='train')
            self.network.train_one_batch(prediction_inp , dense_decoder_out  , dense_inp , grad.clone())
        return True
    def evaluate_one_epoch(self)  :
        loss_train = 0 
        number = 0 
        for x , l in self.data.load_train(batch_size = self.batch_size) :  
            l = l.to(self.device)
            prediction_inp , _  , _   = self.network(x.to(self.device))
            prediction = self.transmittion.send_data(prediction_inp , l , status='test')
            loss_train +=x.shape[0] * self.loss_fn(prediction.to(self.device) , l )
            number += x.shape[0]
        loss_train = loss_train/number
        loss_test = 0 
        number = 0 
        for x , l in self.data.load_test(batch_size = self.batch_size) :  
            l = l.to(self.device)
            prediction_inp , _  , _    = self.network(x.to(self.device))
            prediction = self.transmittion.send_data(prediction_inp, l , status='test')
            loss_test +=x.shape[0] * self.loss_fn(prediction.to(self.device) , l )
            number += x.shape[0]
        loss_test = loss_test/number 
        return loss_train , loss_test       




            



        



        






