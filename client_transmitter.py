import requests
import json
import time
import torch 

class Transmitter : 
    def __init__(self,server_url , device) : 
        self.server_url = server_url #" https://d5e33cbc658b.ngrok-free.app/is_even"
        self.device = device
    def send_data(self , x , label ,  status  ): #status if is train or test 
        x_copy = x.detach().cpu().tolist()
        l= label.detach().cpu().tolist()
        if status == 'train' : 
            data = {
                'prediction_iput':x_copy,
                'label': l ,
                'status' : status
            }
        elif status == 'test' : 
            data = {
                'prediction_iput':x_copy,
                'label': [] , 
                'status' : status
            }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.server_url, data=json.dumps(data), headers=headers , timeout=120)
        response.raise_for_status()
        result = response.json()
        if status == 'train' : # result = {'grad' :  }
            grad = result['grad']
            return torch.tensor(grad).to(self.device)
        elif status == 'test' :
            prediction = torch.tensor(result['prediction']).to(self.device)
            return  prediction



