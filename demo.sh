#!/bin/sh

#---------------------------------------WHST--------------------------------------#
# train baseline (Unet)
#nohup python -u main.py --model Unet --batch_size 2 --hidden_dim 32 --num_pool 4 --dropout 0.0 --output_dir logs/unet --device cuda --GPU_ids 0 >out0 &

# train WHST, do not return intermediate features 
#nohup python -u main.py --model WHST --batch_size 2 --output_dir logs/whst --device cuda --GPU_ids 2 >out2 &

# train WHST, return intermediate features
nohup python -u main.py --model WHST --return_interm --batch_size 8 --num_pool 4 --enc_layers 6 --dec_layers 6 --hidden_dim 256 --embedding_size 16 --patch_size 1 --output_dir logs/whst466-onepos --device cuda --GPU_ids 1 >out1 &
