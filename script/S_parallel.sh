start_gpu=0
gpu_nums_per_iter=3

for preLen in 96 192 336 720
do

export CUDA_VISIBLE_DEVICES=$((0 * $gpu_nums_per_iter / 6+ $start_gpu))
python -u main.py \
  --is_training True \
  --root_path all_six_datasets/electricity \
  --data_path electricity.csv \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 2e-4 &

export CUDA_VISIBLE_DEVICES=$((1 * $gpu_nums_per_iter / 6+ $start_gpu))
python -u main.py \
  --is_training True \
  --root_path all_six_datasets/traffic \
  --data_path traffic.csv \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 1e-3 &

export CUDA_VISIBLE_DEVICES=$((2 * $gpu_nums_per_iter / 6+ $start_gpu))
python -u main.py \
  --is_training True \
  --root_path all_six_datasets/exchange_rate \
  --data_path exchange_rate.csv \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 5e-4 &

export CUDA_VISIBLE_DEVICES=$((3 * $gpu_nums_per_iter / 6+ $start_gpu))
python -u main.py \
  --is_training True \
  --root_path all_six_datasets/ETT-small \
  --data_path ETTm2.csv \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 3e-4 &

export CUDA_VISIBLE_DEVICES=$((4 * $gpu_nums_per_iter / 6+ $start_gpu))
python -u main.py \
  --is_training True \
  --root_path all_six_datasets/weather \
  --data_path weather.csv \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 5e-4 &
done

export CUDA_VISIBLE_DEVICES=$((5 * $gpu_nums_per_iter / 6+ $start_gpu))
for preLen in 24 36 48 60
do
python -u main.py \
  --is_training True \
  --root_path all_six_datasets/illness \
  --data_path national_illness.csv \
  --data custom \
  --features S \
  --seq_len 36 \
  --label_len 36 \
  --pred_len $preLen \
  --learning_rate 5e-4 &
done

