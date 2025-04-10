# Finetune Seg on SIIM
# ECAMP
CUDA_VISIBLE_DEVICES=5 python train.py --name ecamp --stage train --model vit_base_patch16 --task SIIM --img_size 224 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO SIIM/' \
    --output_dir "output/SIIM/1/" --data_volume '1' --num_steps 3000  --eval_batch_size 512 \
    --learning_rate 5e-4 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 512 --weight_decay 0.05

CUDA_VISIBLE_DEVICES=4 python train.py --name ecamp --stage train --model vit_base_patch16 --task SIIM --img_size 224 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO SIIM/' \
    --output_dir "output/SIIM/10/" --data_volume '10' --num_steps 3000  --eval_batch_size 512 \
    --learning_rate 5e-4 --warmup_steps 150 --fp16 --fp16_opt_level O2 --train_batch_size 1024 --weight_decay 0.05

CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task SIIM --img_size 224 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO SIIM/' \
    --output_dir "output/SIIM/100/" --data_volume '100' --num_steps 3000  --eval_batch_size 512 \
    --learning_rate 5e-4 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 512 --weight_decay 0.05





# Finetune seg on RSNA
CUDA_VISIBLE_DEVICES=7 python train.py --name ecamp --stage train --model vit_base_patch16 --task RSNA --img_size 224 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO RSNA/' \
    --output_dir "output/RSNA/1/" --data_volume '1' --num_steps 3000  --eval_batch_size 512 \
    --learning_rate 3e-4 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 96 --weight_decay 0.05

CUDA_VISIBLE_DEVICES=5 python train.py --name ecamp --stage train --model vit_base_patch16 --task RSNA --img_size 224 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO RSNA/' \
    --output_dir "output/RSNA/10/" --data_volume '10' --num_steps 1000  --eval_batch_size 512 \
    --learning_rate 5e-4 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 256 --weight_decay 0.05

CUDA_VISIBLE_DEVICES=6 python train.py --name ecamp --stage train --model vit_base_patch16 --task RSNA --img_size 224 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16.pth' --dataset_path '$PATH TO RSNA/' \
    --output_dir "output/RSNA/100/" --data_volume '100' --num_steps 1000  --eval_batch_size 512 \
    --learning_rate 3e-3 --warmup_steps 100 --fp16 --fp16_opt_level O2 --train_batch_size 512 --weight_decay 0.05





###### Finetune FFAIR ECAMP on RIGA

# FFAIR ECAMP
CUDA_VISIBLE_DEVICES=5 python train_RIGA.py --name ecamp --stage train --model vit_base_patch16 --task RIGA --img_size 224 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16_fundus.pth' --dataset_path '$PATH TO RIGA' \
    --output_dir "output/RIGA/ECAMP/1/" --data_volume '1' --num_steps 500  --eval_batch_size 95 \
    --learning_rate 5e-4 --warmup_steps 15 --fp16 --fp16_opt_level O2 --train_batch_size 5 --weight_decay 0.05

CUDA_VISIBLE_DEVICES=5 python train_RIGA.py --name ecamp --stage train --model vit_base_patch16 --task RIGA --img_size 224 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16_fundus.pth' --dataset_path '$PATH TO RIGA' \
    --output_dir "output/RIGA/ECAMP/10/" --data_volume '10' --num_steps 500  --eval_batch_size 95 \
    --learning_rate 5e-4 --warmup_steps 15 --fp16 --fp16_opt_level O2 --train_batch_size 56 --weight_decay 0.05

CUDA_VISIBLE_DEVICES=6 python train_RIGA.py --name ecamp --stage train --model vit_base_patch16 --task RIGA --img_size 224 \
    --pretrained_path '$PATH TO ECAMP_ViT_Base_16_fundus.pth' --dataset_path '$PATH TO RIGA' \
    --output_dir "output/RIGA/ECAMP/100/" --data_volume '100' --num_steps 1000  --eval_batch_size 95 \
    --learning_rate 5e-4 --warmup_steps 20 --fp16 --fp16_opt_level O2 --train_batch_size 128 --weight_decay 0.05