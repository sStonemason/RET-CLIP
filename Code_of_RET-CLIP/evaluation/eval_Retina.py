import os
from PIL import Image
import torch
from torch import optim
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
import json
from tqdm import tqdm
from time import gmtime, strftime
import pandas as pd
# import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, \
    multilabel_confusion_matrix
import logging
from cn_clip_v3.training.logger import setup_primary_logging, setup_worker_logging
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from RET_CLIP.clip.model import CLIP, resize_pos_embed
import math
from timm.data.mixup import Mixup

TRAIN = True
DATA_DIR = ''
LABEL_DIR = ''
RANK = 0
BATCH_SIZE = 16
NUM_WORKERS = 16


def _convert_to_rgb(image):
    return image.convert('RGB')


class RetinaDataset(data.Dataset):
    def __init__(self, data_dir, label_dir, split='train', imsize=224):

        self.split = split
        self.data_dir = data_dir
        self.label_dir = label_dir
        if self.split is not None:
            assert os.path.isdir(data_dir), "The data directory {} of {} split does not exist!".format(data_dir, split)
            self.image_dir = '{}/{}'.format(data_dir, self.split)
        else:
            self.image_dir = data_dir

        self.imsize = imsize

        if self.split == 'train':
            self.transform = self.transforms_train(resolution=imsize)
        elif self.split == 'valid' or 'test':
            self.transform = self.transforms_valid(resolution=imsize)
        self.len = 0

        self.imgfiles = []
        self.imglabels = []
        if self.split is not None:
            for _, _, files in os.walk(self.image_dir):
                for filename in files:
                    self.len += 1
                    self.imgfiles.append("{}/{}".format(self.image_dir, filename))

    def transforms_train(self, resolution):
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop((resolution, resolution), scale=(0.6, 1.2), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0., hue=0.),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
            transforms.RandomErasing(p=0.2)
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
            labelfile = self.label_dir + '/labels.csv'
        elif self.split == 'valid':
            labelfile = self.label_dir + '/labels.csv'
        elif self.split == 'test':
            labelfile = self.label_dir + '/labels.csv'

        df = pd.read_csv(labelfile)
        id = str(filepath.split('/')[-1])
        ret = torch.tensor(df.loc[df['ID'] == id].values[0][1])
        return ret

    def __getitem__(self, index):

        filepath = self.imgfiles[index]
        image = self.get_imgs(filepath, transform=self.transform)[0]
        label = self.get_label(filepath)

        return filepath, image, label

    def __len__(self):
        return self.len


def adjust_learning_rate(optimizer, epoch, epochs, warmup_epochs, lr, min_lr):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    optimizer.param_groups[0]["lr"] = lr
    optimizer.param_groups[1]["lr"] = lr
    return lr


def eval_multiLabelCls_ViT(model, device):
    num_classes = 4

    # for ViT
    classifier = torch.nn.Sequential(
        torch.nn.Linear(512, num_classes),
        torch.nn.Softmax()
    ).to(device)

    for param in model.parameters():
        param.requires_grad = True

    if TRAIN:
        # 2023.10.24 fully fine-tune
        exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n: not exclude(n)
        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "lr": 2e-6, "weight_decay": 0., "betas": (0.9, 0.999)},
                {"params": rest_params, "lr": 2e-6, "weight_decay": 0.02, "betas": (0.9, 0.98)},
                {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 0., "betas": (0.9, 0.999)},
            ],
            eps=1e-6,
        )
        print('optimizer init finished...')
        mixup_fn = Mixup(
            mixup_alpha=0.2, cutmix_alpha=0.2, cutmix_minmax=(0.1, 0.5),
            prob=0.5, switch_prob=0.5, mode="batch",
            label_smoothing=0.1, num_classes=3)

        criterion = LabelSmoothingCrossEntropy()

        train_dataset = RetinaDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='train', imsize=224)
        val_dataset = RetinaDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='valid', imsize=224)
        test_dataset = RetinaDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='test', imsize=224)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
                                                  num_workers=NUM_WORKERS)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                   num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
                                                 num_workers=NUM_WORKERS)

        best_auc = 0
        best_map = 0
        best_both = 0
        best_epoch = 0

        best_auc_test = 0
        best_map_test = 0
        best_both_test = 0
        best_epoch_test = 0

        for epoch in range(50):
            model.train()
            classifier.train()
            data_iter = iter(train_loader)
            for step in tqdm(range(len(data_iter))):
                _, image, label = next(data_iter)
                # adjust_learning_rate(optimizer, epoch, 50, 20, 2e-6, 1e-6)

                optimizer.zero_grad()
                imgs = image.to(device)
                label = label.to(device)

                image_features = model(imgs, None, None)

                probs = classifier(image_features)
                loss = criterion(probs, label)

                loss.backward()
                optimizer.step()
            logging.info(f"LR: {optimizer.param_groups[0]['lr']:7f} | ")

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
            auc = roc_auc_score(labels, preds, multi_class='ovr')
            map = average_precision_score(labels, preds)

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
                    pred_test = classifier(image_features)
                    preds_test.append(pred_test)
                    labels_test.append(label)

            preds_test = torch.cat(preds_test, dim=0).cpu().numpy()
            labels_test = torch.cat(labels_test, dim=0).cpu().numpy()
            auc_test = roc_auc_score(labels_test, preds_test, multi_class='ovr')
            map_test = average_precision_score(labels_test, preds_test)

            logging.info(
                f"---TEST--- |"
                f"Test AUC: {auc_test:.6f} | "
                f"Test MAP: {map_test:.6f} | "
            )
            if auc_test > best_auc_test:
                best_auc_test = auc_test
            if map_test > best_map_test:
                best_map_test = map_test
            if auc_test + map_test > best_both_test:
                best_both_test = auc_test + map_test
                best_epoch_test = epoch + 1
        return best_auc, best_map, best_epoch, best_auc_test, best_map_test, best_epoch_test


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    for i in range(5):
        clip_resume = ''

        vision_model_config_file = "./RET_CLIP/clip/model_configs/ViT-B-16.json"
        print('Loading vision model config from', vision_model_config_file)
        assert os.path.exists(vision_model_config_file), "The vision_model_config_file does not exist!"

        text_model_config_file = "./RET_CLIP/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json"
        print('Loading text model config from', text_model_config_file)
        assert os.path.exists(text_model_config_file), "The text_model_config_file does not exist!"

        with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
            model_info = json.load(fv)
            if isinstance(model_info['vision_layers'], str):
                model_info['vision_layers'] = eval(model_info['vision_layers'])
            for k, v in json.load(ft).items():
                model_info[k] = v

        model = CLIP(**model_info)

        model.set_grad_checkpointing()
        checkpoint = torch.load(clip_resume, map_location="cpu")
        sd = {k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
        # Load the state dict
        model.load_state_dict(sd, strict=False)

        for param in model.parameters():
            param.data = param.data.float()
        model = model.float().to(device=device)

        time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        log_path = ''
        log_path = os.path.join(log_path, "".format(i))
        log_level = logging.INFO
        log_queue = setup_primary_logging(log_path, log_level, RANK)
        setup_worker_logging(RANK, log_queue, log_level)

        best_auc, best_map, best_epoch, best_auc_test, best_map_test, best_epoch_test = eval_multiLabelCls_ViT(model,
                                                                                                               device=device)
        logging.info(
            f"Best epoch_valid: {best_epoch:.6f} | "
            f"Best Valid AUC: {best_auc:.6f} | "
            f"Best Valid MAP: {best_map:.6f} | "
        )
        logging.info(
            f"Best epoch_test: {best_epoch_test:.6f} | "
            f"Best Test AUC: {best_auc_test:.6f} | "
            f"Best Test MAP: {best_map_test:.6f} | "
        )


if __name__ == "__main__":
    main()
