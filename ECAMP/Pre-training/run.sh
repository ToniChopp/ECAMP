# source activate ECAMP
# cd /$YOUR_CODE_DIR/ECAMP/Pre-training/
# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
python main_pretrain.py \
    --num_workers 16 \
    --accum_iter 8 \
    --batch_size 256 \
    --model ecamp \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 115 \
    --warmup_epochs 40 \
    --lr 1.5e-4 --weight_decay 0.05 \
    --resume ./dataset/mae_vit_base.pth \
    --data_path ./dataset/ \
    --output_dir ../output/ \
    --description "ECAMP pretraining"
