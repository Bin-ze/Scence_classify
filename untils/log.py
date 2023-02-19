import os
path="./visdom_text (2).txt"
with open(path) as fid:
    fid=fid.read()
    fid=fid.split("<br>",-1)
# print(fid)
if not os.path.exists("./swin_train_log.txt"):
    os.mknod("./swin_train_log.txt")
with open("./swin_train_log.txt","w") as w:
    for index,k in enumerate(fid):
        if index + 1 == len(fid):
            w.write(k)
        else:
            w.write(k + "\n")
