import os
#import visdom
#from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import time
from model import resnet34#,resnet101
#import  matplotlib.pyplot  as plt
#import numpy as np
from torch.optim import *
#def tarin_val_acc_plot(x,y1,y2,label1,label2):
#    """
#    simple advanced plot
#    """
#    plt.figure(figsize=(16, 12), dpi=100)
#    ax_1 = plt.subplot()
#    plt.title("figure")
#    plt.grid(True)
#    ax_1.plot(x, y1, color="blue", linewidth=2.0, linestyle="--", label=label1)
#    ax_1.legend(loc="upper left", shadow=True)
#    ax_1.set_ylabel("train_acc")
#    ax_2 = ax_1.twinx()
#    ax_2.plot(x, y2, color="green", linewidth=2.0, linestyle="-", label=label2)
#    ax_2.legend(loc="upper right", shadow=True)
#    ax_2.set_ylabel("val_acc")
#    ax_1.set_xlabel("time")
#    plt.show()
#    return
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    #data_root ="/home/data/gzb/" # get data root path
    #image_path = os.path.join(data_root, "Scence_train_val_data")  # Scence data set path
    image_path = args.scence_data_path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    Scense_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in Scense_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=51)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size =args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # logging.info('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images fot validation.".format(train_num,
                                                                           val_num))
    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    #model_weight_path = "./resNet34.pth"
    #assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    #missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 51)
    #net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    #lr_list=[]
   # train_acc=[]
    #val_acc1=[]
    LR = args.LR
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    epochs = args.epoch
    best_acc = 0.0
    save_path =args.save_path
    train_steps = len(train_loader)
    #train_loss=[]
    #train_time=[]
   # vis=visdom.Visdom(env="restrain_Loss")
    for epoch in range(epochs):
        # train
        net.train()
        running_loss, start_time= 0.0 ,time.time()
        train_bar = train_loader
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            #  train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
            #                                                          epochs,
            #                                                          loss)

        scheduler.step()
        #lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        net.eval()
        #acc = 0.0
        val_acc=0
        with torch.no_grad():
            val_bar=validate_loader
           # train_bar =train_loader
           # for train_data in train_bar:
           #     train_images, train_labels = train_data
           #     outputs = net(train_images.to(device))
           #     # loss = loss_function(outputs, test_labels)
           #     predict_y = torch.max(outputs, dim=1)[1]
           #     acc += torch.eq(predict_y, train_labels.to(device)).sum().item()
           # train_accurate = acc / train_num
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs1 = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs1, dim=1)[1]
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            val_accurate = val_acc / val_num
            # vis.line(X=np.array([epoch]),Y=np.array([running_loss / train_steps]),win="loss_train",update="append",opts={"title":"train_loss"},name="train_loss")
            # vis.line(X=np.array([epoch]), Y=np.array([train_accurate]), win="loss_train", update="append",name="train_acc")
            # vis.line(X=np.array([epoch]), Y=np.array([val_accurate]), win="loss_train", update="append", name="val_acc")
            #train_loss.append(running_loss / train_steps)
            #train_time.append(epoch)
            #train_acc.append(train_accurate)
            #val_acc1.append(val_accurate)
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net, save_path)
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  run_time %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate,(time.time()-start_time)))
            with open('result.txt','a') as f:
                f.write('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  run_time %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate,(time.time()-start_time)))
                f.write('\n')


    # logging.info('Finished Training')
    print('Finished Training')
    #train_loss = np.array(train_loss)
    #train_time = np.array(train_time)
    #train_time = np.cumsum(train_time)
    #plt.figure(figsize=(16, 12), dpi=100)
    #plt.title("train_loss")
    #plt.xlabel("time")
    #plt.ylabel("train_loss")
    #plt.plot(train_time, train_loss, "r-")
    #plt.show()
    #plt.savefig("./resnet_loss.png")
    #plt.plot(range(100),lr_list,color = 'r')
    #plt.show()
    #tarin_val_acc_plot(train_time,train_acc,val_acc1,"train_acc","val_acc")






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--scence_data_path', default='/home/data/gzb/Scence_train_val_data', help='train/val path')
    parser.add_argument('--save_path', default='./model.pth', help='weight save path')
    parser.add_argument('--epoch', default=100, help='train epoch')
    parser.add_argument('--LR', default=0.001, help='train learn rate')
    parser.add_argument('--batch_size', default=128, help='train learn rate')
    args = parser.parse_args()
    main(args)
