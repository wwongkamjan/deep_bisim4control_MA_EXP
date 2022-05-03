#!/bin/bash
#SBATCH --job-name=pong_bisim                         # sets the job name
#SBATCH --output=out/pong_bisim_out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=out/pong_bisim_err_out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=48:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=medium                                          # set QOS, this will determine what resources can be requested
#SBATCH --mem=16GB                                              # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --account=furongh
#SBATCH --mail-user=wwongkam@terpmail.umd.edu
#SBATCH --mail-type=ALL
module add cuda/9.2.148
module add Python3/3.6.4
module add ffmpeg/4.2.1

export MJLIB_PATH=/nfshomes/wwongkam/.mujoco/mujoco200/bin/libmujoco200.so

srun bash -c "hostname;  python train_MA.py --img_source video --eval_resource_files './distractors/images/drivingcar1.mp4' --resource_files './distractors/images/drivingcar1.mp4' --total_frames 1000 --encoder_type pixel --decoder_type identity --action_repeat 4 --save_video --save_tb --work_dir ./log --seed 10"

# once the end of the batch script is reached your job allocation will be revoked