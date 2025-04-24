import subprocess
import os

if __name__ == '__main__':
    seeds = [i for i in range(6, 40)]
    job_script_directory = "/network/scratch/g/guangyuan.wang/comp579_final_proj/temp_job_scripts"
    os.makedirs(job_script_directory, exist_ok=True)
    
    for seed in seeds:
        time = '24:00:00'
        python_run_command = f"python lmcdqn.py --seed {seed}"
        job_script_content = f'''#!/usr/bin/bash
        echo Start time
        echo "`date +%Y:%m:%d-%H:%M:%S`"
        module unload python
        module load anaconda
        cd /network/scratch/g/guangyuan.wang/comp579_final_proj/
        conda activate ./condaenv
        echo {python_run_command}
        {python_run_command}
        echo Stop time
        echo "`date +%Y:%m:%d-%H:%M:%S`"
        '''

        job_name = f"lmcdqn-{seed}"
        job_script_filename = os.path.join(
            job_script_directory, f"{job_name}.sh")

        with open(job_script_filename, 'w') as job_script_file:
            job_script_file.write(job_script_content)

        launch_command = f'sbatch --job-name={job_name} --time={time} --gres=gpu:1 -c 2 --mem=24G --output={job_name}.out {job_script_filename}'
        subprocess.run(launch_command, shell=True,
                        executable='/usr/bin/bash')