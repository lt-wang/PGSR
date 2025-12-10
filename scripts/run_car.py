import os

# scenes = ["2024_06_04_13_44_39","2024_10_08_09_17_32_anonymous_special_vehicles"]
scenes = ["2024_10_08_09_17_32_anonymous_special_vehicles"]
data_base_path='data/3drealcar'
out_base_path='output/3drealcar'
out_name='test_wo_normal_prior'
gpu_id=0

for scene in scenes:


    common_args = "-r2 --ncc_scale 0.5"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train_car.py -s {data_base_path}/{scene}/colmap_processed/pcd_rescale -m {out_base_path}/{scene}/{out_name} {common_args}'
    # print(cmd)
    # os.system(cmd)

    common_args = "--num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} {common_args} --skip_test'
    print(cmd)
    os.system(cmd)

