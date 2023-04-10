where.exe python.exe

Set-Variable DATASET "conHR"
Set-Variable LR "5e-3"
Set-Variable SEED 42
Set-Variable MODEL_TYPE "lstm"
Set-Variable CRITERIA "macro-f1"
Set-Variable BATCH_SIZE 64
Set-Variable INPUT_PATH "datasets/splited/${DATASET}/fake"
Set-Variable OUTPUT_DIR "outputs/${DATASET}/${MODEL_TYPE}/criteria=${CRITERIA}/lr=${LR}/seed=${SEED}"

Write-Output $OUTPUT_DIR

if (-not (Test-Path $OUTPUT_DIR)) {
    mkdir -p $OUTPUT_DIR
}

Set-Variable OPTS (
    " --criteria ${CRITERIA}" +
    " --model-type ${MODEL_TYPE}" +
    " --seed ${SEED}" +
    " --lr ${LR}" +
    " --batch-size ${BATCH_SIZE}" +
    " --epoch 50" +
    " --input-path ${INPUT_PATH}" +
    " --output-path ${OUTPUT_DIR}" +
    " --seq-len 10" +
    " --eval-step 100" +
    " --device cuda" +
    " --train-proportion 0.6" +
    " --early-stopping 10" +
    " --dataset ${DATASET}"
)

Set-Variable CMD "python.exe -m sport ${OPTS}"
Set-Variable LOG_PATH "${OUTPUT_DIR}/run.log"
Write-Output "${CMD}"
Write-Output "log path = ${LOG_PATH}"
powershell.exe -Command "${CMD} > ${LOG_PATH}"
