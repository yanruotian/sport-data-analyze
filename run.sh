# srun -G 1 -p rtx2080 bash run.sh | tee run.log

which python3

LR="5e-5"
SEED=100
BATCH_SIZE=1
INPUT_PATH="/data_new/private/yanruotian/time-analyze/splited/综合数据"

OPTS=""
OPTS+=" --seed ${SEED}"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --epoch 50"
OPTS+=" --input-path ${INPUT_PATH}"
OPTS+=" --output-path outputs"
OPTS+=" --seq-len 15"
OPTS+=" --eval-step 500"
OPTS+=" --device cuda"
OPTS+=" --train-proportion 0.4"

CMD="python3 -m sport ${OPTS}"
echo ${CMD}
${CMD}
