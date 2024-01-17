run_one() {
  ssh  $1  "cd /vol/bitbucket/tl2020/QPytorch &&  tmux new-session -d -s myTempSession ./run_mp_train_exp.sh $2"
}

run_one gpu31 256000
run_one gpu32 16000
run_one gpu33 32000
run_one gpu34 64000
run_one gpu36 128000
