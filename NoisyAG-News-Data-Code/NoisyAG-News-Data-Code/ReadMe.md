# ReadMe

This is a new benchmark focused on instance-dependent noise in text classification tasks. We selected five models—BERT, RoBERTa, BART, XLNet, and DeBERTa—and combined them with six noise-handling methods: CT, CM, CMGT, LS, NLS, and BTLS to classify the NoisyAG-News dataset.

#### Runtime Environment

```python
python                       3.9.0
keras                        2.9.0
nvidia-cuda-runtime-cu12     12.1.105
nvidia-cudnn-cu12            8.9.2.26
torch                        2.1.2
torchnet                     0.0.4
torchvision                  0.16.2
```

#### Launch

Run the following script in the console or execute the shell script. Various parameters can be modified as needed.

```python
#!/bin/bash
# xlnet-base-cased roberta-base bert-base-uncased  bart-base  deberta-v3-base
modelNames=("bart-base")

# 
methods=(  'ct' ) 

# 
#dataSets=("gt" "best" "mid" "worst")
dataSets=(  "GtSample50000" )

# 
noiseType="NoneNoise"
NoiseLevel="0.0"

CUDA_DEVICE=5
mySeed=2345

# 
for modelName in "${modelNames[@]}"; do
  for method in "${methods[@]}"; do
    for dataSet in "${dataSets[@]}"; do
      logFile="./pLog/${modelName}/${method}/${dataSet}_${noiseType}_${NoiseLevel}_${mySeed}.log"
      
      echo "Running modelName=${modelName}, method=${method}, dataSet=${dataSet} device ${CUDA_DEVICE}"

      if [ "$method" == "ls" ] && [ "$dataSet" == "worst" ]; then
        smoothing_factor=0.38
      elif [ "$method" == "ls" ] && [ "$dataSet" == "mid" ]; then
        smoothing_factor=0.2
      elif [ "$method" == "ls" ] && [ "$dataSet" == "best" ]; then
        smoothing_factor=0.1
      elif [ "$method" == "nls" ] && [ "$dataSet" == "worst" ]; then
        smoothing_factor=-0.38
      elif [ "$method" == "nls" ] && [ "$dataSet" == "mid" ]; then
        smoothing_factor=-0.2
      elif [ "$method" == "nls" ] && [ "$dataSet" == "best" ]; then
        smoothing_factor=-0.1
      elif [ "$dataSet" == "gt" ]; then
        smoothing_factor=0.0
      else
        smoothing_factor=0.0
      fi
        
      echo "smoothing_factor is ${smoothing_factor}"

      CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup python NoisyAG_News.py \
      --saveData \
      --dataset $dataSet \
      --log_root "./log/" \
      --data_root "./data/" \
      --trainer_name $method \
      --model_name $modelName \
      --gen_val \
      --nl_batch_size 32 \
      --eval_batch_size 32 \
      --gradient_accumulation_steps 1 \
      --max_sen_len 64 \
      --lr 0.00002 \
      --num_training_steps 8000 \
      --patience 36 \
      --eval_freq 100 \
      --store_model 0 \
      --noise_level $NoiseLevel \
      --noise_type $noiseType \
      --smoothing_factor $smoothing_factor \
      --manualSeed $mySeed \
      >> ${logFile} 2>&1 &
      # 
      sleep 3

    done
  done
done
```
