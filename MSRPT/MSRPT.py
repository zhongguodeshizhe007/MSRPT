import os
import sys
import torch
import pandas as pd
from sklearn.metrics.ranking import roc_auc_score
from models.modeling_MSRPT import MSRPT, CONFIGS
from tqdm import tqdm
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import argparse
from apex import amp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
import argparse
import warnings
from __future__ import print_function, division 
if not sys.warnoptions:
    warnings.simplefilter("ignore")

tk_lim = 20 # token大小

risk_levels = ['ELPR', 'LPR', 'MPR', 'HPR', 'EHPR'] # 风险等级分类

def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    print("Loading MSRPT...")
    return model

def computeAUROC (dataGT, dataPRED, classCount=5):
    outAUROC = []
        
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
        
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUROC

class Data(Dataset):
    def __init__(self, set_type, img_dir, transform=None, target_transform=None):
        dict_path = set_type+'.pkl'
        f = open(dict_path, 'rb') 
        self.mm_data = pickle.load(f)
        f.close()
        self.idx_list = list(self.mm_data.keys())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        k = self.idx_list[idx]
        img_path = os.path.join(self.img_dir, k) + '.png'
        img = Image.open(img_path).convert('RGB')

        label = self.mm_data[k]['label'].astype('float32')
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        sd = torch.from_numpy(self.mm_data[k]['pdesc']).float()
        demo = torch.from_numpy(np.array(self.mm_data[k]['bics'])).float()
        tee = torch.from_numpy(self.mm_data[k]['bts']).float()
        return img, label, sd, demo, tee

def train(args):
    
    # 配置模型
    num_classes = args.CLS
    config = CONFIGS["MSRPT"]
    model = MSRPT(config, 224, zero_head=True, num_classes=num_classes)
    
    # 数据增强和转换
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 创建数据集和数据加载器
    train_data = Data('train', args.DATA_DIR, transform=data_transforms['train'])
    val_data = Data('val', args.DATA_DIR, transform=data_transforms['val'])
    
    train_loader = DataLoader(train_data, batch_size=args.BSZ, shuffle=True, 
                             num_workers=16, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.BSZ, shuffle=False, 
                           num_workers=16, pin_memory=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # 混合精度训练
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    # 数据并行
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # 固定训练30个epoch
    max_epochs = 30
    best_val_auc = 0.0
    
    # 训练循环
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        
        # 在第20个epoch降低学习率
        if epoch == 19:
            print(f"\nReducing learning rate at epoch 20 (10x reduction)")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        # 训练阶段
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        for images, labels, sd, demo, tee in train_bar:
            # 准备数据
            images = images.to(device)
            labels = labels.to(device)
            sd = sd.view(-1, tk_lim, sd.shape[3]).to(device).float()
            demo = demo.view(-1, 1, demo.shape[1]).to(device).float()
            tee = tee.view(-1, tee.shape[1], 1).to(device).float()
            
            # 拆分人口统计学特征
            sex = demo[:, :, 1].view(-1, 1, 1)
            age = demo[:, :, 0].view(-1, 1, 1)
            TypeOfPlace = demo[:, :, 2].view(-1, 1, 1)
            Mode = demo[:, :, 3].view(-1, 1, 1)
            Frequency = demo[:, :, 4].view(-1, 1, 1)
            Distance = demo[:, :, 5].view(-1, 1, 1)
            NumOfAccidents = demo[:, :, 6].view(-1, 1, 1)
            NumOfAnger = demo[:, :, 7].view(-1, 1, 1)
            DSLHW = demo[:, :, 8].view(-1, 1, 1)
            DSLRR = demo[:, :, 9].view(-1, 1, 1)
            DrivingClose = demo[:, :, 10].view(-1, 1, 1)
            BeatDriver = demo[:, :, 11].view(-1, 1, 1)
            SoundingHorn = demo[:, :, 12].view(-1, 1, 1)
            UseEarpiece = demo[:, :, 13].view(-1, 1, 1)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images, sd, tee, sex, age, TypeOfPlace, Mode, Frequency,
                           Distance, NumOfAccidents, NumOfAnger, DSLHW, DSLRR, 
                           DrivingClose, BeatDriver, SoundingHorn, UseEarpiece)[0]
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            
            # 更新进度条
            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss/(train_bar.n+1))
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]")
            for images, labels, sd, demo, tee in val_bar:
                # 准备数据（与训练相同）
                images = images.to(device)
                labels = labels.to(device)
                sd = sd.view(-1, tk_lim, sd.shape[3]).to(device).float()
                demo = demo.view(-1, 1, demo.shape[1]).to(device).float()
                tee = tee.view(-1, tee.shape[1], 1).to(device).float()
                
                sex = demo[:, :, 1].view(-1, 1, 1)
                age = demo[:, :, 0].view(-1, 1, 1)
                TypeOfPlace = demo[:, :, 2].view(-1, 1, 1)
                Mode = demo[:, :, 3].view(-1, 1, 1)
                Frequency = demo[:, :, 4].view(-1, 1, 1)
                Distance = demo[:, :, 5].view(-1, 1, 1)
                NumOfAccidents = demo[:, :, 6].view(-1, 1, 1)
                NumOfAnger = demo[:, :, 7].view(-1, 1, 1)
                DSLHW = demo[:, :, 8].view(-1, 1, 1)
                DSLRR = demo[:, :, 9].view(-1, 1, 1)
                DrivingClose = demo[:, :, 10].view(-1, 1, 1)
                BeatDriver = demo[:, :, 11].view(-1, 1, 1)
                SoundingHorn = demo[:, :, 12].view(-1, 1, 1)
                UseEarpiece = demo[:, :, 13].view(-1, 1, 1)
                
                # 前向传播
                outputs = model(images, sd, tee, sex, age, TypeOfPlace, Mode, Frequency,
                              Distance, NumOfAccidents, NumOfAnger, DSLHW, DSLRR, 
                              DrivingClose, BeatDriver, SoundingHorn, UseEarpiece)[0]
                
                # 计算损失
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 收集预测结果
                probs = torch.sigmoid(outputs)
                outGT = torch.cat((outGT, labels), 0)
                outPRED = torch.cat((outPRED, probs.data), 0)
                
                # 更新进度条
                val_bar.set_postfix(loss=val_loss/(val_bar.n+1))
        
        # 计算验证指标
        aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
        aurocMean = np.array(aurocIndividual).mean()
        
        # 打印验证结果
        print(f"\nEpoch {epoch+1}/{max_epochs} Validation Results:")
        print(f"Loss: {val_loss/len(val_loader):.4f} | Mean AUROC: {aurocMean:.4f}")
        for i in range(len(aurocIndividual)):
            print(f"{risk_levels[i]}: {aurocIndividual[i]:.4f}")
        
        # 保存最佳模型
        if aurocMean > best_val_auc:
            best_val_auc = aurocMean
            # 保存完整模型（包括DataParallel封装）
            torch.save(model.state_dict(), f"best_model_epoch{epoch+1}_auc{aurocMean:.4f}.pth")
            print(f"Saved new best model with AUROC: {aurocMean:.4f}")
    
    # 训练结束后保存最终模型
    torch.save(model.state_dict(), "final_model_epoch30.pth")
    print(f"Saved final model after 30 epochs")
    print("Training completed!")


def test(args):
    torch.manual_seed(0)
    num_classes = args.CLS
    config = CONFIGS["MSRPT"]
    model = MSRPT(config, 224, zero_head=True, num_classes=num_classes)
    msrpt = load_weights(model, 'model.pth')
    img_dir = args.DATA_DIR

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    test_data = Data(args.SET_TYPE, img_dir, transform=data_transforms['test'])

    testloader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=16, pin_memory=True)

    optimizer_MSRPT = torch.optim.AdamW(msrpt.parameters(), lr=3e-5, weight_decay=0.01)
    msrpt, optimizer_MSRPT = amp.initialize(msrpt.cuda(), optimizer_MSRPT, opt_level="O1")

    msrpt = torch.nn.DataParallel(msrpt)


    msrpt.eval()
    with torch.no_grad():
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)
        for data in tqdm(testloader):
            imgs, labels, sd, demo, tee = data
            sd = sd.view(-1, tk_lim, sd.shape[3]).cuda(non_blocking=True).float()
            demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
            tee = lab.view(-1, tee.shape[1], 1).cuda(non_blocking=True).float()
            sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
            age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
            TypeOfPlace = demo[:, :, 2].view(-1, 1, 1).cuda(non_blocking=True).float()
            Mode = demo[:, :, 3].view(-1, 1, 1).cuda(non_blocking=True).float()
            Frequency = demo[:, :, 4].view(-1, 1, 1).cuda(non_blocking=True).float()
            Distance = demo[:, :, 5].view(-1, 1, 1).cuda(non_blocking=True).float()
            NumOfAccidents = demo[:, :, 6].view(-1, 1, 1).cuda(non_blocking=True).float()
            NumOfAnger = demo[:, :, 7].view(-1, 1, 1).cuda(non_blocking=True).float()
            DSLHW = demo[:, :, 8].view(-1, 1, 1).cuda(non_blocking=True).float()
            DSLRR = demo[:, :, 9].view(-1, 1, 1).cuda(non_blocking=True).float()
            DrivingClose = demo[:, :, 10].view(-1, 1, 1).cuda(non_blocking=True).float()
            BeatDriver = demo[:, :, 11].view(-1, 1, 1).cuda(non_blocking=True).float()
            SoundingHorn = demo[:, :, 12.view(-1, 1, 1).cuda(non_blocking=True).float()
            UseEarpiece = demo[:, :, 13].view(-1, 1, 1).cuda(non_blocking=True).float()
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            preds = msrpt(imgs, sd, tee, sex, age, TypeOfPlace, Mode, Frequency,Distance,NumOfAccidents,NumOfAnger,DSLHW,DSLRR,DrivingClose,BeatDriver,SoundingHorn,UseEarpiece)[0]

            probs = torch.sigmoid(preds)
            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)

        aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
        aurocMean = np.array(aurocIndividual).mean()
        
        print('mean AUROC:' + str(aurocMean))
         
        for i in range (0, len(aurocIndividual)):
            print(risk_levels[i] + ': '+str(aurocIndividual[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int)
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int)
    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str)
    parser.add_argument('--SET_TYPE', action='store', dest='SET_TYPE', required=True, type=str)
    args = parser.parse_args()
    train(args)
    # test(args)
