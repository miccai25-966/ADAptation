from PIL import Image
import json
import numpy as np
import os
import cv2
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    #def __init__(self, json_file_path="/home/data/duanyaofei/ControlNet-main-1022/output_train1107.json"):
    def __init__(self, resolution = 512,):    
        
        self.data = []
        self.resolution = resolution
        with open(r"/home/data/duanyaofei/ControlNet-main-1022/output_Anotherchoice.json", 'r', encoding='utf-8') as f:
            content = f.read()
            file = json.loads(content)[:600]
        for i in file:
            self.data.append(i) #将每个元素添加到self.data列表中，self.data列表将包含所有从JSON文件中加载的数据

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        item = self.data[idx]
            
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        
        image = cv2.imread(source_filename)
        image = (image.astype(np.float32) / 127.5) - 1.0 #/127.5-1==从[0,255]映射为[-1,1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        
        canny_hint = cv2.imread(target_filename)
        canny_hint = cv2.cvtColor(canny_hint, cv2.COLOR_BGR2RGB)
        canny_hint = cv2.resize(canny_hint, (512, 512))
        canny_hint = canny_hint.astype(np.float32) / 255.0      
        
        
        #-----------------drop control mode------------------------- 
        if random.random() < 0.2:
            prompt = ""
        
        if random.random() < 0.2:
            canny_hint = np.zeros(canny_hint.shape)
        
            
        return dict(jpg=image, txt=prompt, hint=canny_hint)


dataset = MyDataset()
print(len(dataset))   #697
# # # dataset = Dataset("/home/data/duanyaofei/ControlNet-main-1022/output_beni+mali.json")

# print(len(dataset)) #1000
# item = dataset[1]
# jpg = item['jpg']    #(512, 512, 3)
#                         #(512, 512, 3)
# txt = item['txt']
# hint = item['hint']
# print(txt)   #Ultrasound of benign breast tumor
# print(jpg.shape)  #(512, 512, 3) ('b h w c -> b c h w')
# print(hint.shape)  #(4, 64, 64) (b, c, h, w)
# einops.EinopsError:  Error while processing rearrange-reduction pattern "b h w c -> b c h w".


def main():
    aset = MyDataset()
    dataloader1 = DataLoader(aset, shuffle=True)
    print(len(aset))    ##697
    # print(dataloader1)
    for sample in dataloader1:
            c = sample['txt']
            B = sample['jpg'] 
            a = sample['hint']  
            print("START!") 
            print("BB:", B.dtype)
            print("BBshape:", B.shape)
            print("aa:", a.dtype)
            print("aashape:", a.shape)
            print(c)
            print("----------------------!") 
        
# '''
# START!
# BB: torch.float32
# BBshape: torch.Size([1, 512, 512, 3])
# aa: torch.float32
# aashape: torch.Size([1, 512, 512, 3])
# ['Ultrasound of breast tumor']
# '''
#     # print("OVERs")  
    
if __name__ == '__main__' :
    main()