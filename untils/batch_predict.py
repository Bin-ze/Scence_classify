import os
import json

import torch
from PIL import Image
from torchvision import transforms
import time
from model import resnet34


def main():
    start_time=time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path_list=os.listdir("/home/liuguangcan/guozebin/deep-learning-for-image-processing/data_set/flower_data/flower_photos/daisy")
    # img_path_list = ["../tulip.jpg", "../rose.jpg"]
    img_list = []

    for img_path in img_path_list:
        img_path=os.path.join("/home/liuguangcan/guozebin/deep-learning-for-image-processing/data_set/flower_data/flower_photos/daisy",img_path)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = data_transform(img)
        img_list.append(img)

    # batch img
    batch_img = torch.stack(img_list, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = model(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)
        i=0

        for idx, (pro, cla) in enumerate(zip(probs, classes)):
            if class_indict[str(cla.numpy())]=="daisy":
                i+=1
            print("image: {}  class: {}  prob: {:.3}".format(img_path_list[idx],
                                                             class_indict[str(cla.numpy())],
                                                             pro.numpy()))
    print("acc:  %.3f"% (i/len(img_path_list)))
    print("predict number: %d  all run time : %.3f" %(len(img_path_list),(time.time()-start_time)))


if __name__ == '__main__':
    main()
