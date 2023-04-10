# srun -G 1 -p rtx2080 bash run.sh

# 注意：本脚本版本落后于run.ps1，需要修改后使用

which python3

LR="1e-4"
SEED=100
# MODEL_TYPE="transformer"
MODEL_TYPE="lstm"
CRITERIA="macro-f1"
BATCH_SIZE=1
INPUT_PATH="splited/综合数据"
OUTPUT_DIR="outputs/${MODEL_TYPE}/criteria=${CRITERIA}/lr=${LR}/seed=${SEED}"
mkdir -p ${OUTPUT_DIR}

OPTS=""
OPTS+=" --criteria ${CRITERIA}"
OPTS+=" --model-type ${MODEL_TYPE}"
OPTS+=" --seed ${SEED}"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --epoch 50"
OPTS+=" --input-path ${INPUT_PATH}"
OPTS+=" --output-path ${OUTPUT_DIR}"
OPTS+=" --seq-len 5"
OPTS+=" --eval-step 500"
OPTS+=" --device cuda"
OPTS+=" --train-proportion 0.6"

CMD="python3 -m sport ${OPTS}"
echo ${CMD}
${CMD} | tee ${OUTPUT_DIR}/run.log
