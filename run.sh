# # —————— Train for 外部video ——————
# # exp1 - m1+m2+meta
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '12' \
#     --use_meta  \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext_video/m12_meta \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename

# # # exp2 - m1+m2
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '12' \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext_video/m12 \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename

# # exp3 - m1+meta
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '1' \
#     --use_meta  \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext_video/m1_meta \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename

# # # exp4 - m2+meta
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '2' \
#     --use_meta  \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext_video/m2_meta \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename

# # exp5 - m1
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '1' \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext_video/m1 \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename

# # exp6 - m2
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '2' \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext_video/m2 \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename


# —————— Train for 外部静态 ——————
# exp1 - m1+m2+meta
CUDA_VISIBLE_DEVICES=0 python main.py \
    -bs 32 \
    -nw 4 \
    --modal '12' \
    --use_meta \
    --int \
    --epoch 200 \
    --use_trainning \
    --res_dir /data16/linrun/BA_results/test_on_train/new_ext/m12_meta \
    --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename

# # exp2 - m1+m2
# CUDA_VISIBLE_DEVICES=3 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '12' \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext/m12 \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename

# # exp3 - m1+meta
# CUDA_VISIBLE_DEVICES=6 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '1' \
#     --use_meta  \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext/m1_meta \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename

# # exp4 - m2+meta
# CUDA_VISIBLE_DEVICES=6 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '2' \
#     --use_meta  \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext/m2_meta \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename

# # exp5 - m1
# CUDA_VISIBLE_DEVICES=1 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '1' \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext/m1 \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename

# # exp6 - m2
# CUDA_VISIBLE_DEVICES=2 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '2' \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/ext/m2 \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename


# # —————— Train for 内部 ——————
# # exp1 - m12 meta
# CUDA_VISIBLE_DEVICES=3 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '12' \
#     --use_meta  \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/int/m12_meta \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_for_train_val

# # exp2 - m12
# CUDA_VISIBLE_DEVICES=3 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '12' \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/int/m12 \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_for_train_val

# # exp3 - m1 meta
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '1' \
#     --use_meta  \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/int/m1_meta \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_for_train_val

# # exp4 - m2 meta
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '2' \
#     --use_meta  \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/int/m2_meta \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_for_train_val

# # exp5 - m1
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '1' \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/int/m1 \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_for_train_val

# # exp6 - m2
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     -bs 32 \
#     -nw 4 \
#     --modal '2' \
#     --int \
#     --epoch 200 \
#     --use_trainning \
#     --res_dir /data16/linrun/BA_results/test_on_train/int/m2 \
#     --dataset /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_for_train_val