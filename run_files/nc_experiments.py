from pathlib import Path
import subprocess


job_executer_files_path = Path("./job_executer_files")
if not job_executer_files_path.exists():
    job_executer_files_path.mkdir(parents=True, exist_ok=True)


template_file_address = Path("./job_executer.sh")


attack_type_list = ["catback", "none", 'badnet', 'tabdoor']
dataset_list = ["aci", "bm", "higgs", "credit_card", "diabetes", "covtype", "eye_movement", "kdd99"]
model_list = ["tabnet", "xgboost", "ftt", "saint"] 

dataset_list = ["bm"]
model_list = ['tabnet', 'ftt', 'saint']
attack_type_list = ["none"]

trigger_size_list = [0.08]
epsilon_list = [0.01, 0.02, 0.05, 0.1]
epsilon_list = [0.02]
mu_list = [0.2, 0.5, 1.0] 
mu_list = [1.0]
beta = 0.1
lambd = 0.1
num_exp = 1
exp_num_list = range(0, num_exp)
target_labels = [1]




######## Neural Cleanse experiments #############################
##############################################################

for exp_num in exp_num_list:
    for dataset in dataset_list:
        for model in model_list:
            for target_label in target_labels:
                for epsilon in epsilon_list:
                    for mu in mu_list:
                        for trigger_size in trigger_size_list:
                            for attack_type in attack_type_list:
                                job_script_file = f"exp_{exp_num}_{dataset}_{model}_{target_label}_{epsilon}_{mu}_{trigger_size}_{attack_type}.sh"
                                job_script_file_address = job_executer_files_path / Path(job_script_file)

                                # Read the template and append the command to run the experiment
                                with open(template_file_address, 'r') as template_file:
                                    template_lines = template_file.readlines()
                                
                                # Define the parameters we want to set based on dataset
                                if dataset == 'higgs':
                                    mem_param = '#SBATCH --mem=32GB'
                                    cpu_param = '#SBATCH --cpus-per-task=2'
                                    time_param = '#SBATCH --time=0-01:59:00'
                                elif dataset == 'covtype':
                                    mem_param = '#SBATCH --mem=16GB'
                                    cpu_param = '#SBATCH --cpus-per-task=2'
                                    time_param = '#SBATCH --time=0-01:15:00'
                                elif dataset == 'credit_card':
                                    mem_param = '#SBATCH --mem=16GB'
                                    cpu_param = '#SBATCH --cpus-per-task=2'
                                    time_param = '#SBATCH --time=0-00:35:00'
                                else:
                                    mem_param = '#SBATCH --mem=8GB'
                                    cpu_param = '#SBATCH --cpus-per-task=2'
                                    time_param = '#SBATCH --time=0-00:25:00'
                                
                                # Process each line and modify SBATCH parameters as needed
                                modified_lines = []
                                mem_found = False
                                cpu_found = False
                                time_found = False
                                
                                for line in template_lines:
                                    if line.strip().startswith('#SBATCH --mem='):
                                        modified_lines.append(mem_param + '\n')
                                        mem_found = True
                                    elif line.strip().startswith('#SBATCH --cpus-per-task='):
                                        modified_lines.append(cpu_param + '\n')
                                        cpu_found = True
                                    elif line.strip().startswith('#SBATCH --time='):
                                        modified_lines.append(time_param + '\n')
                                        time_found = True
                                    else:
                                        modified_lines.append(line)
                                
                                # Add any missing parameters at the end of SBATCH section
                                if not mem_found or not cpu_found or not time_found:
                                    for i, line in enumerate(modified_lines):
                                        if line.strip().startswith('#SBATCH') and (i+1 < len(modified_lines) and not modified_lines[i+1].strip().startswith('#SBATCH')):
                                            # This is the last SBATCH line, add missing parameters after it
                                            if not mem_found:
                                                modified_lines.insert(i+1, mem_param + '\n')
                                                i += 1
                                            if not cpu_found:
                                                modified_lines.insert(i+1, cpu_param + '\n')
                                                i += 1
                                            if not time_found:
                                                modified_lines.insert(i+1, time_param + '\n')
                                            break
                                
                                # Create the command to run the experiment
                                command = f"srun python NC.py --dataset_name {dataset} --model_name {model} --target_label {target_label} --mu {mu} --beta {beta} --lambd {lambd} --epsilon {epsilon} --trigger_size {trigger_size} --attack_type {attack_type} --exp_num {exp_num}"
                                
                                # Append the command to the modified content
                                modified_lines.append(f"\n\n{command}\n")
                                
                                # Write the modified content to the job script file
                                with open(job_script_file_address, 'w') as job_file:
                                    job_file.writelines(modified_lines)

                                # Make the script executable
                                subprocess.run(['chmod', '+x', str(job_script_file_address)])
                                # Submit the job script to SLURM
                                subprocess.run(['sbatch', str(job_script_file_address)])