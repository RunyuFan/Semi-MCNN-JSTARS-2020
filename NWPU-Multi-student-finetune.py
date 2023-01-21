import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from dataset_NWPU import CarDateSet
# from resnet_lulc_NWPU import ResNet18, ResNet34, ResNet50, ResNet101
import argparse
# from ResNext_NWPU import resnext50_32x4d
# from MSDnet_NWPU import msdnet
from torchvision.models import resnet50, resnext50_32x4d, shufflenet_v2_x1_0
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # train_datasets = CarDateSet('.\\GenDataset300NWPU\\', './data/trainNWPU-gen.txt', transforms=None)
    train_datasets = CarDateSet('.\\NWPU\\train_data\\', './data/trainNWPU-teacher.txt', transforms=None)
    test_datasets = CarDateSet('.\\NWPU\\test_data\\', './data/testNWPU-teacher.txt', transforms=None)
    # test_datasets = CarDateSet('G:\\LULC\\PytorchLULC\\NWPU-RESISC45-unlabel-test\\test_data\\', './data/testNWPU.txt', transforms=None)

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    print("Train numbers:{:d}".format(len(train_datasets)))
    print("Test numbers:{:d}".format(len(test_datasets)))

    if args.pretrained:
        model = ResNet50(num_classes=1000)
        model.load_state_dict(torch.load(args.pretrained_model))
        channel_in = model.fc.in_features  # 获取fc层的输入通道数
        # 然后把resnet的fc层替换成自己分类类别的fc层
        model.fc = nn.Linear(channel_in, args.num_class)
    else:
        # model1 = ResNet50(args.num_class)

        # model1 = resnet50(pretrained=False)
        # # model.load_state_dict(torch.load(args.pretrained_model))
        # channel_in = model1.fc.in_features  # 获取fc层的输入通道数
        # # 然后把resnet的fc层替换成自己分类类别的fc层
        # model1.fc = nn.Linear(channel_in, args.num_class)
        #
        # # model2 = ResNet34()
        # model2 = resnext50_32x4d(pretrained=False)
        # channel_in_2 = model2.fc.in_features  # 获取fc层的输入通道数
        # # 然后把resnet的fc层替换成自己分类类别的fc层
        # model2.fc = nn.Linear(channel_in_2, args.num_class)
        #
        # model3 = shufflenet_v2_x1_0(pretrained=False)
        # channel_in_3 = model3.fc.in_features  # 获取fc层的输入通道数
        # # 然后把resnet的fc层替换成自己分类类别的fc层
        # model3.fc = nn.Linear(channel_in_3, args.num_class)
            # lulc1 = LULC('.\\model-NWPU\\NWPU-45-teacher-ResNet50.pth')
            # lulc2 = LULC('.\\model-NWPU\\NWPU-45-teacher-resnext50_32x4d.pth')
            # lulc3 = LULC('.\\model-NWPU\\NWPU-45-teacher-shufflenetv2_x1.pth')
        model1 = torch.load('.\\model-NWPU\\NWPU-45-student-ResNet50-1.pth')
        # model2 = ResNet34()
        model2 = torch.load('.\\model-NWPU\\NWPU-45-student-resnext50_32x4d-1.pth')
        model3 = torch.load('.\\model-NWPU\\NWPU-45-student-shufflenetv2_x1-1.pth')
        # print(model3)
        # channel_in_3 = model3.classifier[0]  # 获取fc层的输入通道数
        # # 然后把resnet的fc层替换成自己分类类别的fc层
        # model3.fc = nn.Linear(channel_in_3, args.num_class)
        print('model1 parameters:', sum(p.numel() for p in model1.parameters() if p.requires_grad))
        print('model2 parameters:', sum(p.numel() for p in model2.parameters() if p.requires_grad))
        print('model3 parameters:', sum(p.numel() for p in model3.parameters() if p.requires_grad))

    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    cost1 = nn.CrossEntropyLoss().to(device)
    cost2 = nn.CrossEntropyLoss().to(device)
    cost3 = nn.CrossEntropyLoss().to(device)

    ignored_params1 = list(map(id, model1.fc.parameters()))
    base_params1 = filter(lambda p: id(p) not in ignored_params1, model1.parameters())

    # 对不同参数设置不同的学习率
    params_list1 = [{'params': base_params1, 'lr': 1e-5}]
    params_list1.append({'params': model1.fc.parameters(), 'lr': 1e-3})
    optimizer1 = optim.Adam(params_list1, lr=args.lr, weight_decay=1e-6)

    ignored_params2 = list(map(id, model2.fc.parameters()))
    base_params2 = filter(lambda p: id(p) not in ignored_params2, model2.parameters())

    # 对不同参数设置不同的学习率
    params_list2 = [{'params': base_params2, 'lr': 1e-5}]
    params_list2.append({'params': model2.fc.parameters(), 'lr': 1e-3})
    optimizer2 = optim.Adam(params_list2, lr=args.lr, weight_decay=1e-6)

    ignored_params3 = list(map(id, model3.fc.parameters()))
    base_params3 = filter(lambda p: id(p) not in ignored_params3, model1.parameters())

    # 对不同参数设置不同的学习率
    params_list3 = [{'params': base_params3, 'lr': 1e-5}]
    params_list3.append({'params': model3.fc.parameters(), 'lr': 1e-3})
    optimizer3 = optim.Adam(params_list3, lr=args.lr, weight_decay=1e-6)

    # Optimization
    # optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=1e-6)
    # optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-6)
    # optimizer3 = optim.Adam(model3.parameters(), lr=args.lr, weight_decay=1e-6)

    best_acc_1 = 0.
    best_acc_2 = 0.
    best_acc_3 = 0.

    for epoch in range(1, args.epochs + 1):
        model1.train()
        model2.train()
        model3.train()
        # start time
        start = time.time()
        index = 0
        for images, labels in train_loader:
            images = images.to(device)
            # print(images.shape)
            labels = labels.to(device)

            # Forward pass
            outputs1 = model1(images)
            outputs2 = model2(images)
            outputs3 = model3(images)
            loss1 = cost1(outputs1, labels)
            loss2 = cost2(outputs2, labels)
            loss3 = cost3(outputs3, labels)

            # if index % 10 == 0:
                # print (loss)
            # Backward and optimize
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            loss1.backward()
            loss2.backward()
            loss3.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            index += 1


        if epoch % 1 == 0:
            end = time.time()
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss1.item(), (end-start) * 2))
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss2.item(), (end-start) * 2))
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss3.item(), (end-start) * 2))

            model1.eval()
            model2.eval()
            model3.eval()

            # classes = ('bareland', 'cropland', 'forest', 'impervious', 'shrub', 'water')
            classes = ('airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland')
            # classes = ('1 industrial land', '10 shrub land', '11 natural grassland', '12 artificial grassland', '13 river', '14 lake', '15 pond', '2 urban residential', '3 rural residential', '4 traffic land', '5 paddy field', '6 irrigated land', '7 dry cropland', '8 garden plot', '9 arbor woodland')
            class_correct1 = list(0. for i in range(args.num_class))
            class_total1 = list(0. for i in range(args.num_class))
            class_correct2 = list(0. for i in range(args.num_class))
            class_total2 = list(0. for i in range(args.num_class))
            class_correct3 = list(0. for i in range(args.num_class))
            class_total3 = list(0. for i in range(args.num_class))
            class_correct_all = list(0. for i in range(args.num_class))
            class_total_all = list(0. for i in range(args.num_class))
            correct_prediction_1 = 0.
            total_1 = 0
            correct_prediction_2 = 0.
            total_2 = 0
            correct_prediction_3 = 0.
            total_3 = 0
            correct_prediction_all = 0.
            total_all = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    # to GPU
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs1 = model1(images)
                    _1, predicted1 = torch.max(outputs1, 1)
                    c1 = (predicted1 == labels).squeeze()
                    for label_idx in range(len(labels)):
                        label = labels[label_idx]
                        class_correct1[label] += c1[label_idx].item()
                        class_total1[label] += 1
                    total_1 += labels.size(0)
                    # add correct
                    correct_prediction_1 += (predicted1 == labels).sum().item()


                    outputs2 = model2(images)
                    _2, predicted2 = torch.max(outputs2, 1)
                    c2 = (predicted2 == labels).squeeze()
                    for label_idx in range(len(labels)):
                        label = labels[label_idx]
                        class_correct2[label] += c2[label_idx].item()
                        class_total2[label] += 1
                    total_2 += labels.size(0)
                    # add correct
                    correct_prediction_2 += (predicted2 == labels).sum().item()

                    outputs3 = model3(images)
                    _3, predicted3 = torch.max(outputs3, 1)
                    c3 = (predicted3 == labels).squeeze()
                    for label_idx in range(len(labels)):
                        label = labels[label_idx]
                        class_correct3[label] += c3[label_idx].item()
                        class_total3[label] += 1

                    total_3 += labels.size(0)
                    # add correct
                    correct_prediction_3 += (predicted3 == labels).sum().item()

                    blending_y_pred = outputs1 * 0.33 + outputs2 * 0.34 + outputs3 * 0.33
                    _, predicted_blending = torch.max(blending_y_pred, 1)
                    c_all = (predicted_blending == labels).squeeze()
                    for label_idx in range(len(labels)):
                        label = labels[label_idx]
                        class_correct_all[label] += c_all[label_idx].item()
                        class_total_all[label] += 1

                    total_all += labels.size(0)
                    # add correct
                    correct_prediction_all += (predicted_blending == labels).sum().item()

            # for i in range(args.num_class):
            #     print('Model ResNet50 - Accuracy of %5s : %2d %%: Correct Num: %d in Total Num: %d' % (
            #         classes[i], 100 * class_correct1[i] / class_total1[i], class_correct1[i], class_total1[i]))
            acc_1 = correct_prediction_1 / total_1
            print("Total Acc Model ResNet50: %.4f" % (correct_prediction_1 / total_1))
            print('----------------------------------------------------')
            # for i in range(args.num_class):
            #     print('Model ResNeXt - Accuracy of %5s : %2d %%: Correct Num: %d in Total Num: %d' % (
            #         classes[i], 100 * class_correct2[i] / class_total2[i], class_correct2[i], class_total2[i]))
            acc_2 = correct_prediction_2 / total_2
            print("Total Acc Model ResNeXt: %.4f" % (correct_prediction_2 / total_2))
            print('----------------------------------------------------')
            # for i in range(args.num_class):
            #     print('Model shufflenet - Accuracy of %5s : %2d %%: Correct Num: %d in Total Num: %d' % (
            #         classes[i], 100 * class_correct3[i] / class_total3[i], class_correct3[i], class_total3[i]))
            print("Total Acc Model shufflenet: %.4f" % (correct_prediction_3 / total_3))
            acc_3 = correct_prediction_3 / total_3
            print('----------------------------------------------------')
            # for i in range(args.num_class):
            #     print('Model blending - Accuracy of %5s : %2d %%: Correct Num: %d in Total Num: %d' % (
            #         classes[i], 100 * class_correct_all[i] / class_total_all[i], class_correct_all[i], class_total_all[i]))
            print("Total Acc Model blending: %.4f" % (correct_prediction_all / total_all))
            print('####################################################')
            # correct_prediction = 0.
            # total = 0
            # for images, labels in test_loader:
            #     # to GPU
            #     images = images.to(device)
            #     labels = labels.to(device)
            #     # print prediction
            #     outputs = model(images)
            #     # equal prediction and acc
            #
            #     _, predicted = torch.max(outputs.data, 1)
            #     # val_loader total
            #     total += labels.size(0)
            #     # add correct
            #     correct_prediction += (predicted == labels).sum().item()

            # print("Acc: %.4f" % (correct_prediction / total))

        # Save the model checkpoint
        if acc_1 > best_acc_1:
            print('save new best acc_1', acc_1)
            torch.save(model1, os.path.join(args.model_path, 'NWPU-45-student-finetune-ResNet50-2.pth'))
            best_acc_1 = acc_1
        if acc_2 > best_acc_2:
            print('save new best acc_2', acc_2)
            torch.save(model2, os.path.join(args.model_path, 'NWPU-45-student-finetune-resnext50_32x4d-2.pth'))
            best_acc_2 = acc_2
        if acc_3 > best_acc_3:
            print('save new best acc_3', acc_3)
            torch.save(model3, os.path.join(args.model_path, 'NWPU-45-student-finetune-shufflenetv2_x1-2.pth'))
            best_acc_3 = acc_3
    print("Model save to %s."%(os.path.join(args.model_path, 'NWPU-45-student-finetune-model.pth')))
    print('save new best acc_1', best_acc_1)
    print('save new best acc_2', best_acc_2)
    print('save new best acc_3', best_acc_3)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=45, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    # parser.add_argument("--net", default='ResNet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--model_name", default='NWPU-45-student-finetune', type=str)
    parser.add_argument("--model_path", default='./model-NWPU', type=str)
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--pretrained_model", default='./model/ResNet50.pth', type=str)
    args = parser.parse_args()

    main(args)
