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
import segmentation_models_pytorch as smp

from timm.models.layers import trunc_normal_

from utils.data_utils import get_loader
from utils.segmentation_loss import MixedLoss
from models_vit_RIGA import SegViT

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


def save_model_dice(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_bestdice_checkpoint.bin" % args.name)
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
    if args.model not in ["vit_base_patch16", "resnet50"]:
        raise ValueError("Invalid model name!")

    if args.model == "vit_base_patch16":
        model = SegViT(
            patch_size = 16,
            out_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            decode_features=[512, 256, 128, 64]
        )

        if args.stage=="train":
            if args.name == "random":
                model.to(args.device)
                num_params = count_parameters(model)

                logger.info("Training parameters %s", args)
                logger.info("Total Parameter: \t%2.1fM" % num_params)
                return args, model


            checkpoint = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
            if args.name == "gloria":
                checkpoint_model = checkpoint['state_dict']
            else:
                checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            # ipdb.set_trace()
            matched_dict = odict()
            if args.name == "gloria":
                for key1 in checkpoint_model:
                    for key2 in state_dict:
                        if key1.replace("gloria.img_encoder.model.", "") == key2.replace("encoder.", ""):
                            matched_dict[key2] = checkpoint_model[key1]
                            break
            else:
                for key1 in checkpoint_model:
                    for key2 in state_dict:
                        if key1 == key2.replace("encoder.", ""):
                            matched_dict[key2] = checkpoint_model[key1]
                            break

            # load pre-trained model
            msg = model.load_state_dict(matched_dict, strict=False) # type: ignore
            print(msg)

            for name, param in model.encoder.named_parameters():
                if not name.startswith("seg_head"):
                    param.requires_grad = False
            for name, param in model.named_parameters():
                if not name.startswith("encoder"):
                    param.requires_grad = True
            # ipdb.set_trace()
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        else:
            model = load_weights(model, args.pretrained_path, args)
    else:
        if args.name == "prior":
            model = smp.Unet(
                encoder_name=args.model,
                encoder_weights=None,
                activation=None,
                in_channels=1,
            )
        else:
            model = smp.Unet(
                encoder_name=args.model,
                encoder_weights=None,
                activation=None,
            )
        if args.stage == "train":
            checkpoint = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
            if args.name == "prior" or args.name == "sat":
                checkpoint_model = checkpoint
            else:
                checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            matched_dict = odict()
            
            if args.name == "prior":
                for key1 in checkpoint_model:
                    for key2 in state_dict:
                        if key1 == key2.replace("encoder.", ""):
                            matched_dict[key2] = checkpoint_model[key1]
                            break
            elif args.name == "sat":
                for key1 in checkpoint_model:
                    for key2 in state_dict:
                        if key1.replace("module.img_encoder.", "") == key2.replace("encoder.", ""):
                            matched_dict[key2] = checkpoint_model[key1]
                            break


            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
        
        else:
            model = load_weights(model, args.pretrained_path, args)


    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def get_dice(probability, truth, threshold=0.5):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.reshape(batch_size, -1)
        truth = truth.reshape(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])
        
    return dice.cpu().detach().numpy()


def valid(args, model, writer, val_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(val_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    loss_fct = MixedLoss(alpha=10)

    model.eval()
    dice_scores_disc = []
    dice_scores_cup = []
    epoch_iterator = tqdm(val_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        y_disc = y[:, 0, :, :]
        y_cup = y[:, 1, :, :]

        with torch.no_grad():
            logits_disc, logits_cup = model(x)
            logits_disc = logits_disc.squeeze(dim=1)
            logits_cup = logits_cup.squeeze(dim=1)
            loss = (loss_fct(logits_disc, y_disc) + loss_fct(logits_cup, y_cup)) / 2
            eval_losses.update(loss.item())

            prob_disc = torch.sigmoid(logits_disc)
            prob_cup = torch.sigmoid(logits_cup)
            dice_disc = get_dice(prob_disc, y_disc)
            dice_cup = get_dice(prob_cup, y_cup)
            
            dice_scores_disc.append(dice_disc)
            dice_scores_cup.append(dice_cup)

    dice_scores_disc = np.concatenate(dice_scores_disc)
    dice_disc = dice_scores_disc.mean()
    dice_scores_cup = np.concatenate(dice_scores_cup)
    dice_cup = dice_scores_cup.mean()
    dice = (dice_disc + dice_cup) / 2
    epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)


    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Dice: %2.5f" % dice)
    writer.add_scalar("valid/loss", scalar_value=eval_losses.avg, global_step=global_step)
    return dice, eval_losses.avg


def test(args, model, test_loader):
    eval_losses = AverageMeter()
    loss_fct = MixedLoss(alpha=10)

    model.eval()
    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    

    dice_scores_disc = []
    dice_scores_cup = []
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        y_disc = y[:, 0, :, :]
        y_cup = y[:, 1, :, :]
        with torch.no_grad():
            logits_disc, logits_cup = model(x)
            logits_disc = logits_disc.squeeze(dim=1)
            logits_cup = logits_cup.squeeze(dim=1)
            loss = (loss_fct(logits_disc, y_disc) + loss_fct(logits_cup, y_cup)) / 2
            eval_losses.update(loss.item())

            prob_disc = torch.sigmoid(logits_disc)
            prob_cup = torch.sigmoid(logits_cup)
            dice_disc = get_dice(prob_disc, y_disc)
            dice_cup = get_dice(prob_cup, y_cup)
            dice_scores_disc.append(dice_disc)
            dice_scores_cup.append(dice_cup)

    dice_scores_disc = np.concatenate(dice_scores_disc)
    dice_disc = dice_scores_disc.mean()
    dice_scores_cup = np.concatenate(dice_scores_cup)
    dice_cup = dice_scores_cup.mean()
    dice = (dice_disc + dice_cup) / 2
    epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    logger.info("\n")
    logger.info("Test Results")
    logger.info("Test Loss: %2.5f" % eval_losses.avg)
    logger.info("Test Dice: %2.5f" % dice)


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
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

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
    global_step, best_dice =0, 0
    min_loss = 1000000
    down = 0
    loss_fct = MixedLoss(alpha=10)
    if args.name == "ecamp":
        patience = 40
    else:
        patience = 20

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            y_disc = y[:, 0, :, :]
            y_cup = y[:, 1, :, :]

            logits_disc, logits_cup = model(x)
            logits_disc = logits_disc.squeeze(dim=1)
            logits_cup = logits_cup.squeeze(dim=1)
            loss_disc = loss_fct(logits_disc, y_disc)
            loss_cup = loss_fct(logits_cup, y_cup)
            loss = (loss_disc + loss_cup) / 2
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
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                
                len_train = len(train_loader)
                if global_step % len_train == 0 and args.local_rank in [-1, 0]:
                    dice, val_loss = valid(args, model, writer, val_loader, global_step)
                    writer.add_scalar("dice", scalar_value=dice, global_step=global_step)

                    if best_dice <= dice:
                        save_model_dice(args, model)
                        best_dice = dice
                        down = 0
                    else:
                        down = down + 1
                        print(down)
        losses.reset()
        if global_step % t_total == 0 or down >= patience:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    
    logger.info("min_Loss: \t%f" % min_loss)
    logger.info("End Training!")




def main():
    parser = ArgumentParser()
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

    parser.add_argument("--stage", type=str, default="train", help="train or test?")
    
    parser.add_argument("--task", choices=["RSNA", "SIIM", "RIGA"],
                        default="SIIM",
                        help="Which finetune task to take.")
    parser.add_argument("--pretrained_path", type=str, default="../dataset/MRM.pth",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Image size for segmentation")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=2e-4, type=float,
                        help="The initial learning rate for SGD.")               
    parser.add_argument("--weight_decay", default=0.05, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--data_volume", type=str)

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
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
    parser.add_argument('--start_eval', type=int, default=100,
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


    args, model = setup(args)
    train(args, model)

    args.pretrained_path = os.path.join(args.output_dir, "%s_bestdice_checkpoint.bin" % args.name)
    args.stage = "test"
    args, model = setup(args)
    test_loader = get_loader(args)

    test(args, model, test_loader)

    log_handler.close()

if __name__ == "__main__":
    main()
