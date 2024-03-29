CUDA_VISIBLE_DEVICES=3 \
  python train_cl.py \
  --model_name bert_spc_cl \
  --dataset cl_mams_2X3 \
  --num_epoch 50 \
  --seed 755 \
  --lr 2e-5 \
  --is_test 0 \
  --type cl2X3


# "cl_acl2014_2X3" "cl_res2014_2X3" "cl_laptop2014_2X3" "cl_res2015_2X3" "cl_res2016_2X3" "cl_mams_2X3".