import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import time
from tqdm import tqdm
#from model import resnet34
def main(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data_transform = {
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = args.scence_data_path
    image_path = '/home/data/gzb/Scence_datasets/images'
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)



    validate_dataset = datasets.ImageFolder(root=image_path,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)

    Scense_list = validate_dataset.class_to_idx
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=args.batch_size, shuffle=False,
                                                  num_workers=4)
   # net = resnet34()
    #in_channel = net.fc.in_features
   # net.fc = nn.Linear(in_channel, 51)
    #net.to(device)
    # load model weights
    weights_path = "/home/guozebin/Scence_project/model.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net=torch.load(weights_path, map_location=device)
    val_acc = 0
    start_time=time.time()
    with torch.no_grad():
        val_bar=tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs1 = net(val_images.to(device))
            predict_y = torch.max(outputs1, dim=1)[1]
            val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = (val_acc / val_num)*100
    print("#Totoal Images {}".format(val_num))
    with open('val_result.txt','a') as f:
        f.write("#Totoal Images {}".format(val_num))
        f.write('\n')
    print('#Class {}'.format(len(Scense_list)))
    with open('val_result.txt','a') as f:
        f.write('#Class {}'.format(len(Scense_list)))
        f.write('\n')
    print('Recognition accuracy ={:.1f}%({}/{}) run_time {:.2f}'.format(val_accurate,val_acc,val_num,(time.time()-start_time)))
    with open('val_result.txt','a') as f:
        f.write('Recognition accuracy ={:.1f}%({}/{}) run_time {:.2f}s'.format(val_accurate,val_acc,val_num,(time.time()-start_time)))
        f.write('\n')
        f.write('Finished validation')
    print('Finished validation')






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--scence_data_path', default='/home/data/gzb/Scence_train_val_data', help='val path')
    parser.add_argument('--batch_size', default=64, help='val batch size')
    args = parser.parse_args()
    main(args)
