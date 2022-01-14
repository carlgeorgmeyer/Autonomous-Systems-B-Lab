from fastai.vision.all import *
import datetime
import fastai
import json
import threading
import time
import collections
import pandas as pd
#torch.cuda.set_device(3)
from tqdm import tqdm


GROUPX = str(1)
GROUPY = str(1)
GROUP = 1
EXT = ".PNG"

MODEL_TO_LOAD = "stage-10a"
DRONES = 5
PATH = "/home/ele_group_1/ml/dataset_group"+str(GROUP)
LABELS_PATH = "/home/ele_group_1/ml/labels_group"+str(GROUP)

OUTPUT_PATH = "/home/ele_group_1/ml/COMPETITION_RESULTS"
def get_x(r): return PATH+"/"+r['id']+EXT
def get_y(r): return r['label']



class Application():
    drone_list = []
    threads = []
    environment = []
    def __init__(self):
        
        a = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=get_x, 
            get_y=get_y,
            item_tfms = Resize(224))
        
        
        
        
        
        
        df = pd.read_csv(LABELS_PATH+".csv", sep=';', converters={'id': lambda x: str(x)})
        dl = a.dataloaders(df)
        self.model  = cnn_learner(dl, fastai.vision.models.resnet152, metrics=[accuracy])
        self.model.load(MODEL_TO_LOAD)
        self.model.export("competition_group1.pkl")
                   
        
        fnames = get_image_files(PATH)
        print("Initiating the drones")
        for i in range(0,DRONES):
            self.drone_list.append(Drone(i,180*i,fnames, dl))
        for drone in self.drone_list:
            thread = threading.Thread(target=drone.work)
            self.threads.append(thread)


    def start(self):
        print("Simulation starts")
        self.report = {}
        for thread in self.threads:
            thread.start()
        for thread in self.threads:
            thread.join()
        
        for drone in self.drone_list:
            df = pd.DataFrame.from_dict(drone.report[drone.name], orient="index", columns=['id']).iloc[: , 0:0]
            df.to_csv(OUTPUT_PATH+"/drone"+drone.name+".txt",sep=' ', index=True, header = False)
            self.report.update(drone.report[drone.name])
            
        orderedDictList = collections.OrderedDict(sorted(self.report.items()))
       
        temp_id = {}
        i=0
        for item in orderedDictList.items():
            temp_id[i] = item
            i+=1
            
        df = pd.DataFrame.from_dict(temp_id, orient="index", columns=['id','label'])
        df.to_csv(OUTPUT_PATH+"/results_group"+GROUPX+"_vs_group"+GROUPY+".txt",sep=';', index=False)
class Drone():
    report = {}
    def __init__(self,name, start, fnames, dl):
        self.start_point = start
        self.fnames = fnames
        self.name = str(name+1)
        self.model  = cnn_learner(dl, fastai.vision.models.resnet152, metrics=[accuracy])
        self.model.load(MODEL_TO_LOAD)
        self.report[self.name] = {}
        
    def work(self):
        minus = False
        point = self.start_point
        self.predict()
        count = 0
        for j in range(6):
            for i in range(0,30):
                if(not minus):
                    self.report[self.name][str(point).zfill(3)] = self.preds[str(point).zfill(3)] 
                    point +=1
                else: 
                    point-=1
                    self.report[self.name][str(point).zfill(3)] = self.preds[str(point).zfill(3)] 
                    
            count+=1
            if(point > self.start_point+180):
                break
            point+=30
            
            minus = not minus
        print("Drone "+self.name+ " stops working")
        
    def predict(self):
        self.prediction = []
        self.preds = {}
        count = self.start_point
        print("Drone "+self.name+ " starts working")
        with self.model.no_bar():
            for fname in self.fnames:
                nam = str(fname).split("/")[-1]
                if( nam == str(self.start_point+180).zfill(3)+EXT):
                    break
                if(nam == str(count).zfill(3) + EXT):
                    self.preds[nam.replace(EXT, "")] = self.model.predict(fname)[0]
                    count+=1

if __name__ == "__main__":
    ap = Application()
    ap.start()