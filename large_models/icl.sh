MODEL=${MODEL:-facebook/opt-13b}

python run.py --model_name $MODEL --task_name $TASK --output_dir result/tmp --tag icl --num_train 32 --num_eval 1000 --load_float16 --verbose "$@"
