#!/bin/bash
source .venv/bin/activate

run=0

for loss_scale in 1 5 10 25 50 100 500 1000 4000 8000 16000 32000 64000 128000; do
for man_width in 3; do
for batch_size in 64; do
for lr in 0.1; do
for bk_exp_width in 4; do
for fw_exp_width in 2; do
for clip in 0.01 0.1 1 10 100; do
for batchnorm in batchnorm id; do
for round2 in stochastic nearest ; do
	if [ $batchnorm == "batchnorm" ]; then
		run=0
	elif [ $batchnorm == "id" ]; then
		run=1
	elif [ $batchnorm == "shift_norm" ]; then
		run=2
	fi
	CUDA_VISIBLE_DEVICES=$run python mix_precision_train.py \
		-w $man_width -e $man_width -g $man_width -a $man_width \
		--seed $run \
		--weight-ew $fw_exp_width \
		--error-ew $bk_exp_width \
		--gradient-ew $fw_exp_width \
		--activation-ew $bk_exp_width \
		--loss-scale $loss_scale \
		--log-path results/results_new_batchnorm_2/f${fw_exp_width}_b${bk_exp_width}_log_${run}_clip \
		--seed $run \
		--epochs 200 \
		--weight-round $round2 \
		--error-round $round2 \
		--gradient-round $round2 \
		--activation-round $round2 \
		--learning-rate $lr \
		--momentum 0.9 \
		--batch-size $batch_size \
		--clip $clip \
		--batchnorm $batchnorm \
		--mix-precision True & 
done
done
	wait
done
done
done
done
done
done
done
