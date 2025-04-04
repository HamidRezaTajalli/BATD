#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu_a100
##SBATCH --partition=gpu_h100
#SBATCH --time=0-00:15:00
#SBATCH --mem=2GB
#SBATCH --output=script_logging/slurm_%A.out
#SBATCH --mail-type=END,FAIL                     # send email when job ends or fails
#SBATCH --mail-user=hamidreza.tajalli@ru.nl      # email address



# Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0


# srun python example_poker.py --dataset_name poker --model_name tabnet --target_label 1 --mu 1.0 --beta 0.1 --lambd 0.1 --epsilon 0.02 --exp_num 0

# srun python clean_train.py --dataset higgs --model saint --method ordinal

# srun python NC.py --dataset_name covtype --model_name ftt --target_label 0




# srun torchrun --standalone --nproc_per_node=4 /home/htajalli/prjs0962/repos/BA_NODE/temp_test_torchrun.py

# # pip install git+https://github.com/huggingface/transformers
# srun $HOME/TAConvDR/component3_retriever/bm25/evaluation.py \
#     --index_dir_path "corpus/indexes" \
#     --result_qrel_path "component3_retriever/results" \
#     --gold_qrel_path "component3_retriever/data/topiocqa/dev/qrel_gold.trec" \
#     --dataset_name "topiocqa" \
#     --query_format "human_rewritten" \
#     --seed 42


# # dataset_name = ["topiocqa", "inscit", "qrecc"]
# # query_format = ['original', 'human_rewritten', 'all_history', 'same_topic']


# srun python step_by_step.py --dataset_name covtype --model_name ftt --target_label 5 --mu 1.0 --beta 0.1 --lambd 0.1 --epsilon 0.01 --exp_num 0

# srun python ss.py --dataset_name eye_movement --model_name ftt --target_label 1 --mu 0.5 --beta 0.1 --lambd 0.1 --epsilon 0.05 --exp_num 0
