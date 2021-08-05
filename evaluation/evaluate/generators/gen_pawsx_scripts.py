
bl = "\\" 
dir_name = "DIR_NAME='pawsx'_${BATCH_SIZE_PER_GPU}_${WEIGHT_DECAY}_${LEARN_RATE}_$(date +'%m-%d-%y_%H-%M')"
models = [{'m':'bertin'}, {'m':'beto'}, {'m':'bne-base'}, {'m':'bne-base_new'}, {'m':'bne-large', 'grad':2}, {'m':'mbert'}, {'m':'electra'}]
batches = [8, 16]
weight_dec = ['0.1', '0.01']
learn_rate = ['0.00001','0.00003', '0.00005']

for model_d in models:
    model = model_d['m']
    grad = model_d['grad'] if 'grad' in model_d else 1
    for batch in batches:
        batch_noGrad = int(batch / grad)
        for weight in weight_dec:
            for lr in learn_rate:
                with open(f"{model}_pawsx_batch{batch}_lr{lr}_decay{weight}.sh", "w") as f:
                    f.write(f"""#!/bin/bash
#SBATCH --job-name="{model}_pawsx_batch{batch}_lr{lr}_decay{weight}"
#SBATCH -D .
#SBATCH --output=../logs/{model}_pawsx_batch{batch}_lr{lr}_decay{weight}_%j.out
#SBATCH --error=../logs/{model}_pawsx_batch{batch}_lr{lr}_decay{weight}_%j.err
#SBATCH --ntasks=1
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task=128
#SBATCH --time=2-0:00:00

module load gcc/10.2.0 rocm/4.0.1 intel/2018.4 python/3.7.4

source ../env/bin/activate

export LD_LIBRARY_PATH=/gpfs/projects/bsc88/projects/bne/eval_amd/scripts_to_run/external-lib:$LD_LIBRARY_PATH

SEED=1
NUM_EPOCHS=5
BATCH_SIZE={batch_noGrad}
GRADIENT_ACC_STEPS={grad}
BATCH_SIZE_PER_GPU=$(( $BATCH_SIZE*$GRADIENT_ACC_STEPS ))
LEARN_RATE={lr}
WARMUP=0.06
WEIGHT_DECAY={weight}
MAX_SEQ_LENGTH=512

MODEL='../models/{model}'
OUTPUT_DIR='../output/{model}'
LOGGING_DIR='../tb/{model}'
CACHE_DIR='/gpfs/scratch/bsc88/bsc88344/cache_{model}'
{dir_name}

export MPLCONFIGDIR=$CACHE_DIR/$DIR_NAME/matplotlib
export HF_HOME=$CACHE_DIR/$DIR_NAME/huggingface
rm -rf $MPLCONFIGDIR

python ../run_glue.py --model_name_or_path $MODEL --seed $SEED {bl}
                                          --dataset_script_path ../scripts/paws-x.py --dataset_config_name es {bl}
                                          --task_name mrpc --do_train --do_eval --do_predict {bl}
                                          --num_train_epochs $NUM_EPOCHS --gradient_accumulation_steps $GRADIENT_ACC_STEPS --per_device_train_batch_size $BATCH_SIZE --max_seq_length $MAX_SEQ_LENGTH {bl}
                                          --learning_rate $LEARN_RATE {bl}
                                         --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY {bl}
                                          --output_dir $OUTPUT_DIR/$DIR_NAME --overwrite_output_dir {bl}
                                          --logging_dir $LOGGING_DIR/$DIR_NAME --logging_strategy epoch {bl}
                                          --cache_dir $CACHE_DIR/$DIR_NAME --overwrite_cache {bl}
                                          --metric_for_best_model accuracy --evaluation_strategy epoch --load_best_model_at_end
                """)
                    
