# ChestX-ray14
CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task ChestX-ray14 --num_classes 14 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO ChestX-ray14/' \
    --output_dir "output/ChestX-ray14/1/" --data_volume '1' --num_steps 3000  --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 96

CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task ChestX-ray14 --num_classes 14 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO ChestX-ray14/' \
    --output_dir "output/ChestX-ray14/10/" --data_volume '10' --num_steps 3000  --eval_batch_size 1024 --img_size 224 \
    --learning_rate 2.4e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 768

CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task ChestX-ray14 --num_classes 14 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO ChestX-ray14/' \
    --output_dir "output/ChestX-ray14/100/" --data_volume '100' --num_steps 30000  --eval_batch_size 1024 --img_size 224 \
    --learning_rate 1e-2 --warmup_steps 500 --fp16 --fp16_opt_level O2 --train_batch_size 768


# CheXpert
CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO CheXpert' \
    --output_dir "output/CheXpert/1/" --data_volume '1' --num_steps 30000  --eval_batch_size 1024 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 768

CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO CheXpert' \
    --output_dir "output/CheXpert/10/" --data_volume '10' --num_steps 90000  --eval_batch_size 1024 --img_size 224 \
    --learning_rate 5e-3 --warmup_steps 1500 --fp16 --fp16_opt_level O2 --train_batch_size 768

CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO CheXpert' \
    --output_dir "output/CheXpert/100/" --data_volume '100' --num_steps 90000  --eval_batch_size 1024 --img_size 224 \
    --learning_rate 4e-3 --warmup_steps 1500 --fp16 --fp16_opt_level O2 --train_batch_size 768

# RSNA
CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task RSNA --num_classes 1 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO RSNA' \
    --output_dir "output/RSNA/1/" --data_volume '1' --num_steps 2000  --eval_batch_size 1024 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 256

CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task RSNA --num_classes 1 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO RSNA' \
    --output_dir "output/RSNA/10/" --data_volume '10' --num_steps 9000  --eval_batch_size 1024 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 768

CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task RSNA --num_classes 1 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO RSNA' \
    --output_dir "output/RSNA/100/" --data_volume '100' --num_steps 90000  --eval_batch_size 1024 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 150 --fp16 --fp16_opt_level O2 --train_batch_size 768


# COVID-x
CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task COVIDx --num_classes 3 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO COVIDx' \
    --output_dir "output/COVIDx/1/" --data_volume '1' --num_steps 30000  --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 256

CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task COVIDx --num_classes 3 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO COVIDx' \
    --output_dir "output/COVIDx/10/" --data_volume '10' --num_steps 30000  --eval_batch_size 512 --img_size 224 \
    --learning_rate 1e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 768

CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task COVIDx --num_classes 3 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO COVIDx' \
    --output_dir "output/COVIDx/100/" --data_volume '100' --num_steps 30000  --eval_batch_size 512 --img_size 224 \
    --learning_rate 1e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 768
