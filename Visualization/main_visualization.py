# Some codes are borrowed from MAE and MRM-pytorch
import argparse
import numpy as np
import os
import yaml
import tokenizers
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image as ToPILImage

import module.model_ecamp as model_ecamp

import ipdb


# --------------------------------------------------------
# References:
# MRM-pytorch: https://github.com/RL4M/MRM-pytorch
# --------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('ECAMP visualization', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='ecamp', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=448, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)


    # Dataset parameters
    parser.add_argument('--data_path', default='./dataset_dir', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--job_dir', default='code_repo',
                        help='path where to save the codes')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint or pretrained visual encoder')

    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def load_weight(model, weight_path):
    r"""Load weights from the specified path.

    Parameters
    ----------
    model: torch.nn.Module
        Model to load weights to.
    weight_path: str
        Path to the weights to be loaded.
    """
    state_dict = torch.load(weight_path, map_location='cpu')
    pretrained_weight = state_dict['model']
    model_state_dict = model.state_dict()
    matched_weights = {}
    for k, v in pretrained_weight.items():
        if k in model_state_dict:
            matched_weights[k] = v
        elif k.replace("bert_encoder.model.bert.cross_attn_layer", "bert_encoder.model.bert.context_fusion_layer") in model_state_dict:
            matched_weights[k.replace("bert_encoder.model.bert.cross_attn_layer", "bert_encoder.model.bert.context_fusion_layer")] = v
    model.load_state_dict(matched_weights, strict=True)

    return model


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        print(img.mode)
        return img.convert('RGB')


def main(args):
    # text input
    text = "TEXT INPUT"
    tokenizer = tokenizers.Tokenizer.from_file("./dataset/mimic_wordpiece.json")

    tokenizer.enable_padding(max_length=256)
    tokenizer.enable_truncation(max_length=256)
    text = '[CLS] ' + text

    encoded = tokenizer.encode(text)
    text_ids = torch.tensor(encoded.ids).unsqueeze(0)
    attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0)
    type_ids = torch.tensor(encoded.type_ids).unsqueeze(0)

    # image input
    image = pil_loader("IMAGE PATH")


    transform_test_origin = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.4722], std=[0.3028])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4722], std=[0.3028])
    ])
    
    img = transform_test(image).unsqueeze(0)
    img_origin = np.array(transform_test_origin(image))
    Image.fromarray(img_origin[:,:,0], "L").save("vis_original.png")
    device = torch.device('cuda')

    # define the model
    with torch.no_grad():
        model = model_ecamp.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

        model = load_weight(model, args.resume)
        model.to(device)
        model.eval()

        attention = model(img, text_ids, attention_mask, type_ids)
        attention = attention[0,:,4].cpu().detach().numpy()
    
    attention = attention.reshape(attention.shape[0], 14, 14)
    # ipdb.set_trace()


    attention_map = attention.max(axis=0)
    attention_map = attention_map ** 0.25
    
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    attention_map = torch.tensor(attention_map).unsqueeze(0).unsqueeze(0)  # (1, 1, 14, 14)
    attention_map = torch.nn.functional.interpolate(attention_map, size=(224, 224), mode="bilinear", align_corners=False)
    attention_map = attention_map.squeeze().numpy()

    # attention_map = gaussian_filter(attention_map, sigma=5)

    cmap = plt.get_cmap('jet')
    attention_color = cmap(attention_map)[:, :, :3]
    ipdb.set_trace()
    attention_color = (attention_color * 255).astype(np.uint8)

    blended = (0.5 * img_origin + 0.5 * attention_color).astype(np.uint8)

    Image.fromarray(blended).save("vis_heatmap.png")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
