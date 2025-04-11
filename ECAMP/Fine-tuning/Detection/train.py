from __future__ import absolute_import, division, print_function

import logging
import os
from argparse import ArgumentParser
import random
import numpy as np
from collections import OrderedDict as odict

from datetime import timedelta

# import segmentation_models_pytorch as smp
import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.dist_util import get_world_size

import torch.optim as optim
from sklearn import metrics

from timm.models.layers import trunc_normal_
from torchmetrics.detection import MeanAveragePrecision

from utils.data_utils import get_loader
from detector_backbone_resnet import ResNetDetector
from detector_backbone_vit import ViTDetector
from detector_model import ModelMain
from utils.yolo_loss import YOLOLoss
from utils.detection_utils import non_max_suppression

import ipdb

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model_map(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_bestmap_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def load_weights(model, weight_path, args):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    if args.stage=='train':
        pretrained_weights = pretrained_weights['model']
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    # print("load weights")
    # for k, _ in load_weights.items():
    #     print(k)

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def setup(args):
    if args.model == "resnet50":
        if args.name == "prior":
            args.img_encoder = ResNetDetector("resnet_50", in_channels=1)
        else:
            args.img_encoder = ResNetDetector("resnet_50", in_channels=3)
        if args.stage == "train":
            if args.name == "prior":
                model = ModelMain(args.img_encoder, is_training=True)
            state_dict = model.state_dict()
            ckpt = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
            matched_dict = odict()

            if args.name == "prior":
                for k, v in ckpt.items():
                    for key, value in state_dict.items():
                        if k.replace("encoder.", "") == key.replace('backbone.model.', ''):
                            matched_dict[key] = v
                            break
            else:
                for k, v in ckpt.items():
                    for key, value in state_dict.items():
                        if k.replace("module.img_encoder.", "") == key.replace('backbone.model.', ''):
                            matched_dict[key] = v
                            break
            msg = model.load_state_dict(matched_dict, strict=False)
            # print(msg)
        
            for param in args.img_encoder.parameters():
                param.requires_grad = False
        
        else:
            ckpt = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
            model = ModelMain(args.img_encoder, is_training=False)
            msg = model.load_state_dict(ckpt, strict=False)
            # print(msg)
    
    else:
        expansion = 8 if args.data_volume in ['1', '10'] else 4
        args.img_encoder = ViTDetector(
            patch_size=16,
            out_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            expansion=expansion,
        )
        if args.stage == "train":
            # print(args.pretrained_path)
            checkpoint = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
            if args.name == 'ecamp':
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint['model_state_dict']
            state_dict = args.img_encoder.encoder.state_dict()
            matched_dict = odict()
            for key1 in checkpoint_model:
                for key2 in state_dict:
                    if key1 == key2:
                        key2 = "backbone.encoder." + key2
                        matched_dict[key2] = checkpoint_model[key1]
                        break
            
            model = ModelMain(args.img_encoder, is_training=True)

            # load pre-trained model
            msg = model.load_state_dict(matched_dict, strict=False) # type: ignore
            # print(msg)
        
            for name, param in model.backbone.encoder.named_parameters():
                if not name.startswith("det_head"):
                    param.requires_grad = False
            for name, param in model.named_parameters():
                if not name.startswith("backbone.encoder"):
                    param.requires_grad = True
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
            if args.resume > 0:
                model_checkpoint = os.path.join(args.output_dir, "%s_bestmap_checkpoint.bin" % args.name)
                ckpt = torch.load(model_checkpoint, map_location=torch.device('cpu'))
                ckpt_dict = dict()
                model = ModelMain(args.img_encoder, is_training=False)
                msg = model.load_state_dict(ckpt, strict=False)
                # print(msg)


        else:
            ckpt = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
            ckpt_dict = dict()
            model = ModelMain(args.img_encoder, is_training=False)
            msg = model.load_state_dict(ckpt, strict=False)
            # print(msg)

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model




def valid(args, model, writer, val_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    # logger.info("***** Running Validation *****")
    # logger.info("  Num steps = %d", len(val_loader))
    # logger.info("  Batch size = %d", args.eval_batch_size)

    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(model.anchors[i], model.classes, (args.img_size, args.img_size)))
    
    val_map = MeanAveragePrecision(iou_thresholds=[0.4, 0.45, 0.5,
                                    0.55, 0.6, 0.65, 0.7, 0.75])

    model.eval()
    epoch_iterator = tqdm(val_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(epoch_iterator):
        x = batch["imgs"].to(args.device)
        y = batch["labels"].to(args.device)
        with torch.no_grad():
            logits = model(x)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](logits[i], y)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            eval_losses.update(loss)

            out_list = []
            for i in range(3):
                out_list.append(yolo_losses[i](logits[i]))
            output = torch.cat(out_list, 1)
            output = non_max_suppression(
                output,
                model.classes,
                conf_thres = 0.5,
                nms_thres=0.5
            )
            targets = y.clone()
            # cxcywh -> xyxy
            h, w = x.shape[2:]
            targets[:, :, 1] = (y[..., 1] -
                                y[..., 3] / 2) * w
            targets[:, :, 2] = (y[..., 2] -
                                y[..., 4] / 2) * h
            targets[:, :, 3] = (y[..., 1] +
                                y[..., 3] / 2) * w
            targets[:, :, 4] = (y[..., 2] +
                                y[..., 4] / 2) * h
            sample_preds, sample_targets = [], []
            for i in range(targets.shape[0]):
                target = targets[i]
                out = output[i]
                if out is None:
                    continue
                filtered_target = target[target[:, 3] > 0]
                if filtered_target.shape[0] > 0:
                    sample_target = dict(
                        boxes=filtered_target[:, 1:],
                        labels=filtered_target[:, 0]
                    )
                    sample_targets.append(sample_target)

                    out = output[i]
                    sample_pred = dict(
                        boxes=out[:, :4],
                        scores=out[:, 4],
                        labels=out[:, 6]
                    )

                    sample_preds.append(sample_pred)
        val_map.update(sample_preds, sample_targets)

    map = val_map.compute()["map"]

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid mAP: %2.5f" % map)

    return map
    

def test(args, model, test_loader):
    eval_losses = AverageMeter()
    
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(model.anchors[i], model.classes, (args.img_size, args.img_size)))
    
    test_map = MeanAveragePrecision(iou_thresholds=[0.4, 0.45, 0.5,
                                    0.55, 0.6, 0.65, 0.7, 0.75])

    model.eval()
    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    
    for step, batch in enumerate(epoch_iterator):
        x = batch["imgs"].to(args.device)
        y = batch["labels"].to(args.device)
        with torch.no_grad():
            logits = model(x)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](logits[i], y)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            eval_losses.update(loss)

            out_list = []
            for i in range(3):
                out_list.append(yolo_losses[i](logits[i]))
            output = torch.cat(out_list, 1)
            output = non_max_suppression(
                output,
                model.classes,
                conf_thres = 0.5,
                nms_thres=0.5
            )
            targets = y.clone()
            # cxcywh -> xyxy
            h, w = x.shape[2:]
            targets[:, :, 1] = (y[..., 1] -
                                y[..., 3] / 2) * w
            targets[:, :, 2] = (y[..., 2] -
                                y[..., 4] / 2) * h
            targets[:, :, 3] = (y[..., 1] +
                                y[..., 3] / 2) * w
            targets[:, :, 4] = (y[..., 2] +
                                y[..., 4] / 2) * h
            sample_preds, sample_targets = [], []
            for i in range(targets.shape[0]):
                target = targets[i]
                out = output[i]
                if out is None:
                    continue
                filtered_target = target[target[:, 3] > 0]
                if filtered_target.shape[0] > 0:
                    sample_target = dict(
                        boxes=filtered_target[:, 1:],
                        labels=filtered_target[:, 0]
                    )
                    sample_targets.append(sample_target)

                    out = output[i]
                    sample_pred = dict(
                        boxes=out[:, :4],
                        scores=out[:, 4],
                        labels=out[:, 6]
                    )

                    sample_preds.append(sample_pred)
        test_map.update(sample_preds, sample_targets)
    map = test_map.compute()["map"]

    logger.info("\n")
    logger.info("Test Results")
    logger.info("Test Loss: %2.5f" % eval_losses.avg)
    logger.info("Test mAP: %2.5f" % map)



def train(args, model):
    writer = None
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))  #  tensorboard Supporting documents, in logs/name/

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, val_loader = get_loader(args)

    if "vit_base" in args.model:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
        # optimizer_decoder = optim.AdamW(
        #     model.decoder.parameters(),
        #     lr=args.learning_rate,
        #     betas=(0.9, 0.999),
        #     weight_decay=args.weight_decay
        # )
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # optimizer_decoder = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    t_total = args.num_steps
    if args.resume > 0:
        optimizer.param_groups[0]['initial_lr'] = args.learning_rate
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total, last_epoch=args.resume-1)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total, last_epoch=args.resume-1)

    if args.fp16:
        model, optimizers = amp.initialize(models=model,
                                          optimizers=[optimizer],
                                          opt_level=args.fp16_opt_level)
        optimizer = optimizers[0]
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
    
    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Training
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_map = args.resume, 0
    min_loss = 1000000
    down = 0
    
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(model.anchors[i], model.classes, (args.img_size, args.img_size)))

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):
            x = batch["imgs"].to(args.device)
            y = batch["labels"].to(args.device)
            # ipdb.set_trace()
            logits = model(x)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            train_losses = []
            for _ in range(len(losses_name)):
                train_losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](logits[i], y)
                for j, l in enumerate(_loss_item):
                    train_losses[j].append(l)
            train_losses = [sum(l) for l in train_losses]
            loss = train_losses[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f) {lr=%f}" % (global_step, t_total, losses.val, scheduler.get_lr()[0])
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                
                len_train = len(train_loader)
                if global_step > args.start_eval and global_step % len_train == 0 and args.local_rank in [-1, 0]:
                    map = valid(args, model, writer, val_loader, global_step)
                    writer.add_scalar("mAP", scalar_value=map, global_step=global_step)

                    if map >= best_map:
                        save_model_map(args, model)
                        best_map = map
                        down = 0
                    else:
                        down = down + 1
                        print(down)
        losses.reset()
        if global_step % t_total == 0 or down >= 20:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    
    logger.info("End Training!")




def main():
    parser = ArgumentParser()
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

    parser.add_argument("--stage", type=str, default="train", help="train or test?")
    
    parser.add_argument("--task", choices=["RSNA", "ObjectCXR"],
                        default="RSNA",
                        help="Which finetune task to take.")
    parser.add_argument("--pretrained_path", type=str, default="../dataset/MRM.pth",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Image size for segmentation")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=2e-4, type=float,
                        help="The initial learning rate for SGD.")               
    parser.add_argument("--weight_decay", default=0.05, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=3000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--data_volume", type=str)

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=50, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--resume', type=int, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--start_eval', type=int, default=60,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--dataset_path", type=str)

    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"

    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        print('##############################')   
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                            timeout=timedelta(minutes=60)
                                            )
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    log_handler = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    if args.stage == "train":
        args, model = setup(args)
        train(args, model)

    args.pretrained_path = os.path.join(args.output_dir, "%s_bestmap_checkpoint.bin" % args.name)
    args.stage = "test"
    args, model = setup(args)
    test_loader = get_loader(args)
    test(args, model, test_loader)

    log_handler.close()

if __name__ == "__main__":
    main()
