import os
from shutil import rmtree,copy
from PIL import Image
import tqdm
path="/home/data/gzb/南京外景拍摄/"
scene='医院大门'
savepath="/home/data/gzb/Scene_classification"
img_list=os.listdir(os.path.join(path,scene))
for j,img in enumerate(img_list):
    image=Image.open(os.path.join(path,scene,img))
    image=image.convert("RGB")
    w1=848
    h1=480
    img1 = image.resize((w1, h1), Image.ANTIALIAS)
    img1.save(os.path.join(savepath,scene)+"/"+img.split(".")[0]+"(4)"+"."+img.split(".")[1])
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
        croping.save(os.path.join(savepath,scene)+"/"+img.split(".")[0]+"({})".format(str(i))+"."+img.split(".")[1])
    print("\rprocessing [{}/{}]".format(j+1,len(img_list)+1), end="")

