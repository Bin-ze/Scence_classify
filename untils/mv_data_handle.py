import os
import shutil
path="/home/data/gzb/南京外景拍摄/"
for i in os.listdir(path):
    for j in os.listdir(os.path.join(path,i)):
        if not os.path.isfile(os.path.join(path,i,j)):
            print("remove{}".format(j))
            shutil.rmtree(os.path.join(path,i,j))
