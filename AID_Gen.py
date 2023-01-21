import argparse
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2
import torch.nn.functional as F
import time
# from osgeo import gdal
from math import ceil
from tifffile import *
import json
from PIL import Image
import cv2
from image_preprocess import load_img
from PIL import Image
# import scipy
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imgsize = 448
class LULC:

    def __init__(self, model_path):

        if torch.cuda.is_available():
            self.model = torch.load(model_path).to(device)
        else:
            self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((imgsize, imgsize)),  # 将图像转化为128 * 128
            transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
            # transforms.Normalize([0.485, 0.66, 0.406], [0.229, 0.224, 0.225]) # 归一化
        ])


    def detect(self, image):

        image = self.transform(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        outputs = self.model(image)
        # print('outputs:', outputs)
        prob = F.softmax(outputs, dim=1)
        # print('prob:', prob[0])
        pred = torch.argmax(prob, dim=1)
        # pred = pred.numpy()
        return prob[0]


def predict_sliding(model, image, tile_size, classes):
    # interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    print(image_size)
    overlap = 0 #每次滑动的重合率为1/3
    stride = int(ceil(tile_size[0] * (1 - overlap)))
    tile_rows = int(ceil((image_size[1] - tile_size[0]) / stride) + 1)  #行滑动步数:
    tile_cols = int(ceil((image_size[2] - tile_size[1]) / stride) + 1)  #列滑动步数:
    print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[1], image_size[2], classes))  #初始化全概率矩阵
    count_predictions = np.zeros((image_size[1], image_size[2], classes))   #初始化计数矩阵
    tile_counter = 0    #滑动计数0
    for row in range(tile_rows):    # row = 0,1
        if row % 100 == 0:
            print('in row:', row)
        for col in range(tile_cols):    # col = 0,1,2,3
            x1 = int(col * stride)  #起始位置x1 = 0 * 513 = 0
            y1 = int(row * stride)  #        y1 = 0 * 513 = 0
            x2 = min(x1 + tile_size[1], image_size[2])  #末位置x2 = min(0+769, 2048)
            y2 = min(y1 + tile_size[0], image_size[1])  #      y2 = min(0+769, 1024)
            x1 = max(int(x2 - tile_size[1]), 0)  #重新校准起始位置x1 = max(769-769, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  #                y1 = max(769-769, 0)
            # print(x1, x2, y1, y2)
            img = image[:, y1:y2, x1:x2] #滑动窗口对应的图像 imge[:, :, 0:769, 0:769]
            imsave('./temp.jpg', img)
            img = cv2.imread('./temp.jpg', 1)
            predict_proprely = model.detect(img)
            predict_proprely = list(predict_proprely.cpu().data.numpy())
            count_predictions[y1:y2, x1:x2] += 1    #窗口区域内的计数矩阵加1
            full_probs[y1:y2, x1:x2] += predict_proprely  #窗口区域内的全概率矩阵叠加预测结果
            tile_counter += 1   #计数加1
    full_probs /= count_predictions
    return full_probs

def color_predicts(img):
    # img = cv2.imread(label_path,cv2.CAP_MODE_GRAY)
    color = np.zeros([img.shape[0], img.shape[1], 3])  # BGR
    color[img==1] = [0, 255, 255] # 黄色 裸地
    color[img==0] = [255, 255, 0] #青色 住宅区
    color[img==2] = [0, 255, 0] # 绿色 植被覆盖区
    color[img==3] = [255, 0, 0] # 蓝色 水体
    color[img==4] = [200, 0, 0]
    color[img==5] = [250,    0, 150]# 黄色 裸地
    color[img==6] = [255, 255, 0] #青色 住宅区
    color[img==7] = [200, 150, 150] # 绿色 植被覆盖区
    color[img==8] = [250, 150, 150] # 蓝色 水体
    color[img==9] = [0, 200, 0]
    color[img==10] = [150, 250,  0] # 黄色 裸地
    color[img==11] = [150, 200, 150] #青色 住宅区
    color[img==12] = [200, 0, 200] # 绿色 植被覆盖区
    color[img==13] = [150, 0, 250] # 蓝色 水体
    color[img==14] = [150, 150, 250]
    return color

def addImage(img1_path, img2_path, name, im_proj, im_geotrans):
    img1 = gdal.Open(img1_path)
    im_width = img1.RasterXSize #栅格矩阵的列数
    im_height = img1.RasterYSize #栅格矩阵的行数
    img1 = img1.ReadAsArray(0,0,im_width,im_height)#获取数据
    imsave('..\\Result\\img_ori.png', img1)
    img1 = cv2.imread('..\\Result\\img_ori.png')
    img2 = cv2.imread(img2_path)
    alpha = 0.5
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img1[:, :, :], alpha, img2, beta, gamma)
    img_add_path = '..\\Result\\' + str(name).replace('.tif', '') + '#' + 'img_add.png'
    imsave(img_add_path, img_add)
    # write_img(img_add_path, im_proj, im_geotrans, img_add)


def test_image_predict_top_k(lulc1, lulc2, lulc3, test_image_path):

    #model.load_weights(WEIGHTS_PATH)
    image_list = generator_list_of_imagepath(test_image_path)
    print(image_list)
    predict_label = []
    # class_indecs = label_of_directory(directory)
    len_img = len(image_list)
    for i in range(len(image_list)):
        try:
            print(image_list[i])
            # img = cv2.imread(image_list[i])
            img = Image.open(image_list[i]).convert('RGB')
            # predict_proprely = model.detect(img)
            predict_proprely1 = lulc1.detect(img)
            predict_proprely2 = lulc2.detect(img)
            predict_proprely3 = lulc3.detect(img)

            predict_proprely = 0.33*predict_proprely1 + 0.34*predict_proprely2 + 0.33*predict_proprely3
            predict_proprely = list(predict_proprely.cpu().data.numpy())
            # predict_proprely = list(predict_proprely.cpu().data.numpy())
            # return a list of label max->min
            # label_index = get_label_predict_top_k(img, model)

            # label_value_dict = []
            # for label in label_index:
            #     # label_value = get_key_from_value(class_indecs, label)
            #     label_value_dict.append(label)

            predict_label.append(predict_proprely)
        except:
            if i < len_img:
                image_list.pop(i)
                len_img = len_img -1
        # print(label_index)

    return image_list, predict_label


def get_label_predict_top_k(predict_proprely, top_k):
    """
    image = load_image(image), input image is a ndarray
    return top-5 of label
    """
    # array 2 list
    # predict_proprely = model.predict(image)
    # print(np.argmax(predict_proprely)/100)
    predict_list = list(predict_proprely)
    min_label = min(predict_list)
    label_k = []
    for i in range(top_k):
        label = np.argmax(predict_list)
        #print(label)
        predict_list.remove(predict_list[label])
        predict_list.insert(label, min_label)
        label_k.append(label)
        # print(label_k)
    return label_k

def write_img(filename,im_proj,im_geotrans,im_data):
    #gdal数据类型包括
    #gdal.GDT_Byte,
    #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    #gdal.GDT_Float32, gdal.GDT_Float64

    #判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    #判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
    dataset.SetProjection(im_proj)                    #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset


def select_top_k(acc=.9):
    sampled_image_dict = {}
    sampled_image_dict["all"] = []
    k = 1000000
    with open("./sampling_dict_AID.json", "r", encoding="utf-8", errors="ignore") as f:
        load_data = json.load(f)

        for key in load_data.keys():
            # print("label: ", key)
            all_items = load_data[key]
            all_items.sort(key=lambda x: x[1], reverse=True)

            # all_items_array = all_items\
            len_item_list = len(all_items)
            # print(all_items)
            # print("each label item count: ", len(all_items))
            # if len(all_items) < k:
            #     k = int(len(all_items))
            for items_index in range(len(all_items)):
                # print(all_items[items_index])
                if float(all_items[items_index][1]) < acc:
                    # print(all_items[items_index], int(key))
                    # all_items_array.pop(items_index)
                    len_item_list -= 1
            if len_item_list < k:
                k = len_item_list

        for key in load_data.keys():
            print("label: ", key)
            all_items = load_data[key]
            all_items.sort(key=lambda x: x[1], reverse=True)
            all_items = np.array(all_items)
            print("each label item count: ", len(all_items))
            # if len(all_items) < k:
            #     k = int(len(all_items))
            print(k)
            for index in range(0, k):
                sampled_image_dict["all"].append([all_items[index][0], int(key)])

    print("Saving.. selected image json")
    j = json.dumps(sampled_image_dict)
    with open("./selected_image_AID.json", "w") as f:
        f.write(j)


def select_top_k2(k=10):
    sampled_image_dict = {}
    sampled_image_dict["all"] = []
    with open("./sampling_dict_AID.json", "r", encoding="utf-8", errors="ignore") as f:
        load_data = json.load(f)

        for key in load_data.keys():
            print("label: ", key)
            all_items = load_data[key]
            all_items.sort(key=lambda x: x[1], reverse=True)
            all_items = np.array(all_items)
            print("each label item count: ", len(all_items))
            if len(all_items) < k:
                k = int(len(all_items))
            for index in range(0, k):
                sampled_image_dict["all"].append([all_items[index][0], int(key)])

    print("Saving.. selected image json")
    j = json.dumps(sampled_image_dict)
    with open("./selected_image_AID.json", "w") as f:
        f.write(j)

test_image_path = '.\\AID\\unlabel_data\\'
# print(len(generator_list_of_imagepath(test_image_path)))
#

def generator_list_of_imagepath(test_image_path):
    image_list = []
    dict = label_of_directory(test_image_path)
    # print(dict)

    for folder in dict:
        img_list = [f for f in os.listdir(os.path.join(test_image_path, folder)) if not f.startswith('.')]
        for img in img_list:
            str0 = os.path.join(test_image_path, os.path.join(folder, img))
            image_list.append(str0)
    return image_list

# print(generator_list_of_imagepath(test_image_path))

def main():
    test_image_path = '.\\AID\\unlabel_data\\'
    label_dict = label_of_directory(test_image_path)

    lulc1 = LULC('.\\model-AID\\AID-30-teacher-ResNet50.pth')
    lulc2 = LULC('.\\model-AID\\AID-30-teacher-resnext50_32x4d.pth')
    lulc3 = LULC('.\\model-AID\\AID-30-teacher-shufflenetv2_x1.pth')
    # lulc = LULC('./model/lulc_6-50.pth')
    # root = '../image'
    # img_list = [f for f in os.listdir(root) if f.endswith('.jpg')]
    # for img in img_list:
    #     image = cv2.imread(os.path.join(root, img))
    #     pred = lulc.detect(image)
    #     print(list(label_dict.keys())[list(label_dict.values()).index(pred)])
    #     cv2.imshow('test', image)
    #     cv2.waitKey(0)
    # test_image_path = '..\\unlabel\\'
    test_image_path = '.\\AID\\unlabel_data\\'

    print("=====test label=====")
    # model.summary()
    image_list, predict_label = test_image_predict_top_k(lulc1, lulc2, lulc3, test_image_path)
    print(len(image_list))

    sampling_dictionary = {}
    maxk = max((1, ))
    batch_image_path = image_list
    output = torch.tensor(predict_label)

    _, top_p = output.topk(maxk, 1, True, True) # maxk = 10
    # [1,2,3] =[[2,3],[]]
    # print(top_p.t())

    # make sampling dictionary
    for top in top_p.t():
        for idx, i in enumerate(top):
            num = i.data.cpu().numpy()
            value = float(output[idx][i].data.cpu().numpy())
            if str(num) in sampling_dictionary:
                sampling_dictionary[str(num)].append([batch_image_path[idx], value])
            else:
                sampling_dictionary[str(num)] = [[batch_image_path[idx], value]]
    print("Saving.. sampling_dict")
    j = json.dumps(sampling_dictionary)
    with open("./sampling_dict_AID.json", "w") as f:
        f.write(j)

# test_image_path = '.\\AID\\unlabel_data\\'
# print(len(generator_list_of_imagepath(test_image_path)))
def label_of_directory(directory):
    """
    sorted for label indices
    return a dict for {'classes', 'range(len(classes))'}
    """
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)

    num_classes = len(classes)
    class_indices = dict(zip(classes, range(len(classes))))
    #print(num_classes)#,class_indices(classes))
    return class_indices
def gen_data():
    test_image_path = '.\\AID\\unlabel_data\\'
    label_dict = label_of_directory(test_image_path)
    print('gen data')
    select_top_k(0.750)
    with open("./selected_image_AID.json", "r", encoding="utf-8", errors="ignore") as f:
        json_data = json.load(f)
    image_list = json_data["all"]
    # random.shuffle(image_list)
    # print(image_list)
    print(image_list)
    # F:\Shenzhen
    # im = Image.open('G:\\LULC\\unlabeltest\\11828.jpg')
    # im = image.load_img(img[0])  # , target_size=(224, 224))
    # print(im)
    # 图像预处理
    # im = image.img_to_array(im)
    for img in image_list:
        print(img)
        label_image = list(label_dict.keys())[list(label_dict.values()).index(img[1])]
        # save_img_path = os.path.join('.\\Shenzhen7-5-5\\train_data', label_image)
        # print(img[0], save_img_path)
        # im = gdal.Open(image[0])
        # im_width = im.RasterXSize #栅格矩阵的列数
        # im_height = im.RasterYSize #栅格矩阵的行数
        # im = im.ReadAsArray(0,0,im_width,im_height)#获取数据
        try:
            # im = cv2.imread('G:\\LULC\\unlabeltest\\11828.jpg', 1)
            # print(im.shape)
            im = load_img(img[0])  # , target_size=(224, 224))
            # print(im.shape)
            # im.save('./image.jpg')
            # 图像预处理
            # im = image.img_to_array(im)
            # im = cv2.imread(img[0])
            # im.save(os.path.join(save_img_path, img[0].split('\\')[-1].replace('tif', 'png')))
            if not os.path.exists('.\\GenDatasetAID'):
                os.mkdir('.\\GenDatasetAID')
            if not os.path.exists(os.path.join('.\\GenDatasetAID', label_image)):
                os.mkdir(os.path.join('.\\GenDatasetAID', label_image))
            print(os.path.join(os.path.join('.\\GenDatasetAID', label_image), img[0].split('\\')[-1]))
            im.save(os.path.join(os.path.join('.\\GenDatasetAID', label_image), img[0].split('\\')[-1]))
            # os.remove(img[0])
        except:
            print('image deleted!')
        # im = load_img(img[0])  # , target_size=(224, 224))
        # # print(im.shape)
        # # im.save('./image.jpg')
        # # 图像预处理
        # # im = image.img_to_array(im)
        # # im = cv2.imread(img[0])
        # # im.save(os.path.join(save_img_path, img[0].split('\\')[-1].replace('tif', 'png')))
        # if not os.path.exists('.\\GenData_Shenzhen'):
        #     os.mkdir('.\\GenData_Shenzhen')
        # if not os.path.exists(os.path.join('.\\GenData_Shenzhen', label_image)):
        #     os.mkdir(os.path.join('.\\GenData_Shenzhen', label_image))
        # print(os.path.join(os.path.join('.\\GenData_Shenzhen', label_image), img[0].split('\\')[-1].replace('tif', 'png')))
        # im.save(os.path.join(os.path.join('.\\GenData_Shenzhen', label_image), img[0].split('\\')[-1].replace('tif', 'png')))
        # # os.remove(img[0])

if __name__ == '__main__':
    main()
    gen_data()
