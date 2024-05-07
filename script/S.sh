export CUDA_VISIBLE_DEVICES=0

for preLen in 96 192 336 720
do

python -u main.py \
  --is_training True \
  --root_path all_six_datasets/electricity \
  --data_path electricity.csv \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 2e-4

python -u main.py \
  --is_training True \
  --root_path all_six_datasets/traffic \
  --data_path traffic.csv \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 1e-3


python -u main.py \
  --is_training True \
  --root_path all_six_datasets/exchange_rate \
  --data_path exchange_rate.csv \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 5e-4

python -u main.py \
  --is_training True \
  --root_path all_six_datasets/ETT-small \
  --data_path ETTm2.csv \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 3e-4

python -u main.py \
  --is_training True \
  --root_path all_six_datasets/weather \
  --data_path weather.csv \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 5e-4
done

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
  --learning_rate 5e-4
done

