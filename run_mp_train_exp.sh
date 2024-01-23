#!/bin/bash
source .venv/bin/activate

run=0

# for loss_scale in 1 5 10 25 50 100 500 1024 2048 4096 8192 16384 32768 65536 131072; do
for loss_scale in 2048 ; do
for man_width in 3; do
for batch_size in 64; do
for lr in 0.1; do
for bk_exp_width in 4; do
for fw_exp_width in 2; do
for momentum in 0 0.7 0.9 ; do
for batchnorm in batchnorm id; do
for round2 in stochastic nearest ; do
for clip in 0.1 ; do
	if [ $momentum == "0" ]; then
		run=0
	elif [ $round2 == "0.7" ]; then
		run=1
	elif [ $round2 == "0.9" ]; then
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
		--log-path results/normvsbatchnorm_lossscale_clip\
		--seed $run \
		--epochs 200 \
		--weight-round $round2 \
		--error-round $round2 \
		--gradient-round $round2 \
		--activation-round $round2 \
		--learning-rate $lr \
		--momentum $momentum \
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
done
