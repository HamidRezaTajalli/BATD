from pathlib import Path
import subprocess


job_executer_files_path = Path("./job_executer_files")
if not job_executer_files_path.exists():
    job_executer_files_path.mkdir(parents=True, exist_ok=True)


template_file_address = Path("./job_executer.sh")

method_list = ["ohe", "converted", "ordinal"]
dataset_list = ["aci", "bm", "higgs", "credit_card", "diabetes", "covtype", "eye_movement", "kdd99"]
model_list = ["tabnet", "ftt", "saint"] 
dataset_list = ['aci']


trigger_size_list = [0.08]
prune_rate_list = [0.3, 0.5]
epsilon_list = [0.01, 0.02, 0.05, 0.1]
epsilon_list = [0.02]
mu_list = [0.2, 0.5, 1.0] 
mu_list = [1.0]
beta = 0.1
lambd = 0.1
num_exp = 1
exp_num_list = range(0, num_exp)
target_labels = [1]

######## step by step experiments #############################
##############################################################

for exp_num in exp_num_list:
    for dataset in dataset_list:
        for model in model_list:
            for target_label in target_labels:
                for epsilon in epsilon_list:
                    for mu in mu_list:
                        job_script_file = f"exp_{exp_num}_{dataset}_{model}_{target_label}_{epsilon}_{mu}.sh"
                        job_script_file_address = job_executer_files_path / Path(job_script_file)

                        # Read the template and append the command to run the experiment
                        with open(template_file_address, 'r') as template_file:
                            template_content = template_file.read()
                        
                        # write the content to the job script file
                        with open(job_script_file_address, 'w') as job_script_file:
                            job_script_file.write(template_content)

                        # create the command to run the experiment
                        command = f"srun python beatrix_files/Beatrix_tabular.py --dataset_name {dataset} --model_name {model} --target_label {target_label} --mu {mu} --beta {beta} --lambd {lambd} --epsilon {epsilon} --exp_num {exp_num}"

                        # append the command to the job script file
                        with open(job_script_file_address, 'a') as job_script_file:
                            job_script_file.write("\n")  # Ensure there's a newline before adding the command
                            job_script_file.write(command)
                            
                        # Make the script executable
                        subprocess.run(['chmod', '+x', str(job_script_file_address)])
                        # Submit the job script to SLURM
                        subprocess.run(['sbatch', str(job_script_file_address)])






