#!/bin/bash
while IFS=$' ' read IDX HOUR CORE MEM CFG_NAME CFG_DIR
do
STD_OUTPUT_FILE="/projects/p30309/RL/dppo/script/clusters/std_output/${IDX}.log"

JOB=`sbatch << EOJ
#!/bin/bash
#SBATCH -J ${IDX}
#SBATCH -A p30309
#SBATCH -p gengpu
#SBATCH -t ${HOUR}:59:59
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}G
#SBATCH --cpus-per-task=${CORE}
#SBATCH --output=${STD_OUTPUT_FILE}
#SBATCH --mail-type=BEGIN,FAIL,END,REQUEUE #BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=zkghhg@gmail.com

#Delete any preceding space after 'EOJ'. OW, there will be some error.

# unload any modules that carried over from your command line session
module purge

# Set your working directory
cd /projects/p30309/RL/dppo/   #$PBS_O_WORKDIR

# load modules you need to use
# module load python/anaconda3.6
# module load mamba
# module load python-miniconda3/4.12.0

# source /home/kzy816/miniconda3_4_12_0_conda.sh

conda init bash

which conda

conda activate "dppo_310"

export LD_LIBRARY_PATH="/home/kzy816/.mujoco/mujoco210/bin:/usr/lib/nvidia:\$LD_LIBRARY_PATH"
export DPPO_DATA_DIR="/projects/p30309/RL/dppo/data"
export DPPO_LOG_DIR="/projects/p30309/RL/dppo/log"

echo "=== Debug LD_LIBRARY_PATH ==="
echo "LD_LIBRARY_PATH: \$LD_LIBRARY_PATH"
echo "Python path: \$(which python)"

# # A command you actually want to execute:
# java -jar <someinput> <someoutput>
# # Another command you actually want to execute, if needed:
# python myscript.py

which python

python -uB ./script/run.py --config-name=${CFG_NAME} --config-dir=${CFG_DIR} wandb=null

EOJ
`

# print out the job id for reference later
echo "JobID = ${JOB} for indices ${IDX} and parameters ${CFG_NAME}, ${CFG_DIR} submitted on `date`"

sleep 0.5

done < /projects/p30309/RL/dppo/script/clusters/quest/param.info
# done < ./command_script/param_unet_exp.info
# done < ./command_script/param_trex_online.info
# done < ./command_script/param_ar_lin_2d.info
exit

# make this file executable and then run from the command line
# chmod u+x submit.sh
# ./submit.sh
# The last line of params.txt have to be an empty line.
