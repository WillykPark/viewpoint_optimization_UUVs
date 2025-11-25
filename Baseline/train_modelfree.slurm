#!/bin/bash
#SBATCH --job-name=ppo_sonar
#SBATCH --output=./results/ppo_output.txt
#SBATCH --error=./results/ppo_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-turin
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000mb
#SBATCH --time=24:00:00

module load conda
conda activate RLenv   # ← 너의 PPO 환경 이름

# GPU 디버깅용
echo "CUDA available?"
python - <<EOF
import torch
print("cuda available:", torch.cuda.is_available())
print("gpu name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
EOF

echo "======================= START PPO TRAINING ======================="

python training_model_free.py

echo "======================= TRAIN DONE ======================="