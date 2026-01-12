datapath=./datasets/Real-IAD/realiad_256

datasets=("multi_class_0.2")
# datasets=("mint" "button_battery" "toy_brick")
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

# python main.py --dataset real_iad --data_path ./datasets/Real-IAD/realiad_256 --exp_json_path ./datasets/Real-IAD/realiad_jsons_fuiad_0.4 --noise 0  "${dataset_flags[@]}"  --faiss_on_gpu --save_segmentation_images

python main.py --dataset real_iad --gpu 0 --data_path ./datasets/Real-IAD/realiad_256 --bank_size 0.02 --exp_json_path ./datasets/Real-IAD/realiad_jsons_multiclass --noise 0  "${dataset_flags[@]}"  --faiss_on_gpu --norm_data True --shuffle True
