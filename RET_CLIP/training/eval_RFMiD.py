import os
from PIL import Image
import torch
from torch import optim
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
import json
from tqdm import tqdm
import pandas as pd
import sklearn.metrics as metrics
import logging


def _convert_to_rgb(image):
    return image.convert('RGB')


class RFMiDDataset(data.Dataset):
    def __init__(self, data_dir, split='train', imsize=224):

        self.split = split
        self.data_dir = data_dir
        if self.split is not None:
            assert os.path.isdir(data_dir), "The data directory {} of {} split does not exist!".format(data_dir, split)
            self.image_dir = '{}/{}'.format(data_dir, self.split)
        else:
            self.image_dir = data_dir

        self.imsize = imsize
        # self.transforms_valid = transforms.Compose([transforms.Resize([self.imsize, self.imsize])])
        # self.norm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        # ])

        if self.split == 'train':
            self.transform = self.transforms_valid(resolution=imsize)
        elif self.split == 'valid' or 'test':
            self.transform = self.transforms_valid(resolution=imsize)
        self.len = 0

        self.imgfiles = []
        self.imglabels = []
        if self.split is not None:
            for _, _, files in os.walk(self.image_dir):
                for filename in files:
                    if filename[-4::] == '.jpg' or filename[-4::] == '.png':
                        self.len += 1
                        self.imgfiles.append("{}/{}".format(self.image_dir, filename))

    def transforms_train(self, resolution):
        # print("using augment...")
        # transform = create_transform(
        #     input_size=resolution,
        #     scale=(0.8, 1.0),
        #     is_training=True,
        #     hflip=0.5,
        #     vflip=0.5,
        #     color_jitter=0.1,
        #     # auto_augment='original',
        #     interpolation='bicubic',
        #     mean=(0.48145466, 0.4578275, 0.40821073),
        #     std=(0.26862954, 0.26130258, 0.27577711),
        #     re_prob=0.1,
        #     crop_pct=0.875,
        # )
        # transform = Compose(transform.transforms[:-4] + [_convert_to_rgb] + transform.transforms[-4:])
        #
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop((resolution, resolution), scale=(0.9, 1.), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0., hue=0.),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
            # transforms.RandomErasing(p=0.1)
        ])
        return transform

    def transforms_valid(self, resolution):
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        return transform

    def get_imgs(self, img_path, transform=None):
        if self.split == 'train':
            img = Image.open(img_path)
            if transform is not None:
                img = transform(img)
            ret = []
            ret.append(img)
        elif self.split == 'valid' or 'test':
            img = Image.open(img_path).convert('RGB')
            if transform is not None:
                img = transform(img)
            ret = []
            ret.append(img)

        return ret

    def get_label(self, filepath):

        if self.split == 'train':
            labelfile = self.data_dir + '/RFMiD_Training_Labels.csv'
        elif self.split == 'valid':
            labelfile = self.data_dir + '/RFMiD_Validation_Labels.csv'
        elif self.split == 'test':
            labelfile = self.data_dir + '/RFMiD_Testing_Labels.csv'

        df = pd.read_csv(labelfile)
        id = int(filepath.split('/')[-1].split('.')[0])
        ret = torch.tensor(df.loc[df['ID'] == id].values[0][2:]).float()
        return ret

    def __getitem__(self, index):

        filepath = self.imgfiles[index]
        image = self.get_imgs(filepath, transform=self.transform)[0]
        label = self.get_label(filepath)

        return filepath, image, label

    def __len__(self):
        return self.len


def eval_multiLabelCls_ViT(model, epoch_main, cuda_rank):
    device = 'cuda:{}'.format(cuda_rank)

    TRAIN = True
    DATA_DIR = '/home/ubuntu/nfs/8T1/yangsz/6disease/AD_text_to_image/text2image-main/data/RFMiD_processed512'

    BATCH_SIZE = 128
    NUM_WORKERS = 16

    for param in model.parameters():
        param.data = param.data.float()
        param.requires_grad = True
    model = model.float().to(device=device)

    num_classes = 28

    # for ViT
    classifier = torch.nn.Sequential(
        torch.nn.Linear(512, num_classes),
        torch.nn.Sigmoid()
    ).to(device)

    if TRAIN:
        # 2023.10.24 fully fine-tune
        exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n: not exclude(n)
        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                # {"params": gain_or_bias_params, "lr": 1e-5, "weight_decay": 0., "betas": (0.9, 0.999)},
                # {"params": rest_params, "lr": 1e-5, "weight_decay": 0.01, "betas": (0.9, 0.999)},
                {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 0., "betas": (0.9, 0.999)},
            ],
            eps=1e-6,
        )
        print('optimizer init finished...')
        criterion = torch.nn.BCELoss()

        train_dataset = RFMiDDataset(data_dir=DATA_DIR, split='train', imsize=224)
        val_dataset = RFMiDDataset(data_dir=DATA_DIR, split='valid', imsize=224)
        test_dataset = RFMiDDataset(data_dir=DATA_DIR, split='test', imsize=224)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                   num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
                                                 num_workers=NUM_WORKERS)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
                                                  num_workers=NUM_WORKERS)

        best_auc = 0
        best_map = 0
        best_both = 0
        best_epoch = 0

        best_auc_test = 0
        best_map_test = 0
        best_both_test = 0
        best_epoch_test = 0

        for epoch in range(75):
            model.train()
            classifier.train()
            data_iter = iter(train_loader)
            for step in tqdm(range(len(data_iter))):
                _, image, label = next(data_iter)

                optimizer.zero_grad()
                imgs = image.to(device)
                label = label.to(device)

                image_features = model(imgs, None, None)
                # patch_features, image_features = model(imgs, None)
                # patch_features = patch_features.view(patch_features.size(0), -1)
                # # patch_features = torch.mean(patch_features, dim=1)
                # image_features = torch.cat((image_features, patch_features), dim=-1)

                probs = classifier(image_features)
                loss = criterion(probs, label)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                classifier.eval()
                data_iter = iter(val_loader)
                preds = []
                labels = []
                for step in tqdm(range(len(data_iter))):
                    _, image, label = next(data_iter)

                    imgs = image.to(device)

                    image_features = model(imgs, None, None)

                    pred = classifier(image_features)
                    preds.append(pred)
                    labels.append(label)

            preds = torch.cat(preds, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()
            auc = metrics.roc_auc_score(labels, preds)
            map = metrics.average_precision_score(labels, preds)

            if auc > best_auc:
                best_auc = auc
            if map > best_map:
                best_map = map
            if auc + map > best_both:
                best_both = auc + map
                best_epoch = epoch + 1

            print(f"Epoch {epoch + 1}: AUC = {auc}, mAP = {map}, Loss = {loss}")
            logging.info(
                f"MultiLabelCls Validation Training(epoch {epoch + 1}) | "
                f"Valid Loss: {loss:.6f} | "
                f"Valid AUC: {auc:.6f} | "
                f"Valid_MAP: {map:.6f} | "
            )
            with torch.no_grad():
                model.eval()
                classifier.eval()
                data_iter = iter(test_loader)
                preds_test = []
                labels_test = []
                for step in tqdm(range(len(data_iter))):
                    _, image, label = next(data_iter)

                    imgs = image.to(device)

                    image_features = model(imgs, None, None)

                    pred = classifier(image_features)
                    preds_test.append(pred)
                    labels_test.append(label)

            preds_test = torch.cat(preds_test, dim=0).cpu().numpy()
            labels_test = torch.cat(labels_test, dim=0).cpu().numpy()
            auc_test = metrics.roc_auc_score(labels_test, preds_test)
            map_test = metrics.average_precision_score(labels_test, preds_test)

            if auc_test > best_auc_test:
                best_auc_test = auc_test
            if map_test > best_map_test:
                best_map_test = map_test
            if auc_test + map_test > best_both_test:
                best_both_test = auc_test + map_test
                best_epoch_test = epoch + 1
            print(f"Epoch {epoch + 1}: AUC-TEST = {auc_test}, mAP-TEST = {map_test}")
            logging.info(
                f"MultiLabelCls Validation Training(epoch {epoch + 1}) | "
                f"Test AUC: {auc_test:.6f} | "
                f"Test MAP: {map_test:.6f} | "
            )

        return epoch_main, best_auc, best_map, best_epoch, best_auc_test, best_map_test, best_epoch_test


def eval_multiLabelCls_RN50(model, epoch_main, cuda_rank):
    device = 'cuda:{}'.format(cuda_rank)

    TRAIN = True
    DATA_DIR = '/home/ubuntu/nfs/8T1/yangsz/6disease/AD_text_to_image/text2image-main/data/RFMiD_processed512'

    BATCH_SIZE = 128
    NUM_WORKERS = 16

    for param in model.parameters():
        param.requires_grad = True

    num_classes = 28

    classifier = torch.nn.Sequential(
        torch.nn.Linear(768, num_classes),
        torch.nn.Sigmoid()
    ).to(device)

    model.to(device)
    if TRAIN:
        optimizer = torch.optim.AdamW(
            [
                {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 0., "betas": (0.9, 0.999)},
            ],
            eps=1e-6,
        )

        criterion = torch.nn.BCELoss()

        # 2023.10.13 注意改image_size!!!
        train_dataset = RFMiDDataset(data_dir=DATA_DIR, split='train', imsize=224)
        val_dataset = RFMiDDataset(data_dir=DATA_DIR, split='valid', imsize=224)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                   num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                 num_workers=NUM_WORKERS)

        best_auc = 0
        best_map = 0
        best_both = 0
        best_epoch = 0

        for epoch in range(75):
            model.train()
            classifier.train()
            data_iter = iter(train_loader)
            for step in tqdm(range(len(data_iter))):
                _, image, label = next(data_iter)

                optimizer.zero_grad()
                imgs = image.to(device)
                label = label.to(device)

                image_features = model(imgs, None, None)
                # patch_features, image_features = model(imgs, None)
                # patch_features = patch_features.view(patch_features.size(0), -1)
                # # patch_features = torch.mean(patch_features, dim=1)
                # image_features = torch.cat((image_features, patch_features), dim=-1)

                probs = classifier(image_features)
                loss = criterion(probs, label)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                classifier.eval()
                data_iter = iter(val_loader)
                preds = []
                labels = []
                for step in tqdm(range(len(data_iter))):
                    _, image, label = next(data_iter)

                    imgs = image.to(device)

                    image_features = model(imgs, None, None)

                    pred = classifier(image_features)
                    preds.append(pred)
                    labels.append(label)

            preds = torch.cat(preds, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()
            auc = metrics.roc_auc_score(labels, preds)
            map = metrics.average_precision_score(labels, preds)

            if auc > best_auc:
                best_auc = auc
            if map > best_map:
                best_map = map
            if auc + map > best_both:
                best_both = auc + map
                best_epoch = epoch + 1

            print(f"Epoch {epoch + 1}: AUC = {auc}, mAP = {map}, Loss = {loss}")
            logging.info(
                f"MultiLabelCls Validation Training(epoch {epoch + 1}) | "
                f"Valid Loss: {loss:.6f} | "
                f"Valid AUC: {auc:.6f} | "
                f"Valid_MAP: {map:.6f} | "
            )

        return epoch_main, best_auc, best_map, best_epoch
