#!/bin/bash

source .venv/bin/activate # TODO if a different environment is adopted, please fix this line.

# array of model params in this order : model_dim num_heads num_layers batch_size lr dropout notation task
configs=(
  "512 4 5 256 0.0001 0.3 fragsmiles forward"
  "512 4 5 256 0.0001 0.3 safe forward"
  "256 4 3 512 0.0010 0.3 selfies forward"
  "256 4 4 512 0.0010 0.3 smiles forward"
  "512 4 5 256 0.0001 0.3 fragsmiles backward"
  "256 4 4 512 0.0010 0.3 safe backward"
  "256 4 3 256 0.0010 0.3 selfies backward"
  "256 4 4 512 0.0010 0.3 smiles backward"
)

# Loop sulle configurazioni
for config in "${configs[@]}"; do
  read model_dim num_heads num_layers batch_size lr dropout notation task <<< "$config"

  echo "Running training with: model_dim=$model_dim, num_heads=$num_heads, num_layers=$num_layers, batch_size=$batch_size, lr=$lr, dropout=$dropout, notation=$notation, task=$task"

  python scripts/train.py \
    --model_dim "$model_dim" \
    --num_heads "$num_heads" \
    --num_layers "$num_layers" \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --dropout "$dropout" \
    --notation "$notation" \
    --task "$task"

  echo "Running prediction with: model_dim=$model_dim, num_heads=$num_heads, num_layers=$num_layers, batch_size=$batch_size, lr=$lr, dropout=$dropout, notation=$notation, task=$task"

  python scripts/predict.py \
    --model_dim "$model_dim" \
    --num_heads "$num_heads" \
    --num_layers "$num_layers" \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --dropout "$dropout" \
    --notation "$notation" \
    --task "$task"

   python scripts/convert_predictions_strict.py \
    --model_dim "$model_dim" \
    --num_heads "$num_heads" \
    --num_layers "$num_layers" \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --dropout "$dropout" \
    --notation "$notation" \
    --task "$task"
done
