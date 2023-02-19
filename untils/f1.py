import os
from shutil import rmtree,copy
from PIL import Image
import tqdm
path="/home/data/gzb/南京外景拍摄/"
savepath="/home/data/gzb/Scene_classification"
class_list=os.listdir(path)
def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)
mk_file("Scene_classification")
for cla in class_list:

    mk_file(os.path.join(savepath,cla))

for i, classes in enumerate(class_list):
        print("\n")
        print("The scene {} is being processed!".format(classes))
        img_list=os.listdir(os.path.join(path,classes))
        for i,img in enumerate(img_list):
            print("\r>>> 正在处理第{}张照片<<<".format(i),end="")
            image=Image.open(os.path.join(path,classes,img))
            image = image.convert("RGB")
            w1=848
            h1=480
            img1 = image.resize((w1, h1), Image.ANTIALIAS)
            try:
                img1.save(os.path.join(savepath,classes)+"/"+img.split(".")[0]+"(4)"+"."+img.split(".")[1])
            except:
                continue
            top = 0
            left = 0
            w =image.size[0]/2
            h =image.size[1]/2
            size=(w,h)

            for i in range(4):
                if i == 2:
                    # 当循环到第三个值时，需要切第二行的图片
                    top += 1
                    left = 0
                a = int(size[0] * left) # 图片距离左边的大小
                b = int(size[1] * top) # 图片距离上边的大小
                c = int(size[0] * (left + 1)) # 图片距离左边的大小 + 图片自身宽度
                d = int(size[1] * (top + 1)) # 图片距离上边的大小 + 图片自身高度
                # print('a= {},b= {},c= {}, d= {}'.format(a,b,c,d))
                left+=1
                croping = image.crop((a,b,c,d)).resize((w1,h1),Image.ANTIALIAS)
                croping.save(os.path.join(savepath,classes)+"/"+img.split(".")[0]+"({})".format(str(i))+"."+img.split(".")[1])

