#/bin/bash

cd /vol/bitbucket/tl2020/QPytorch_result_removed
source .venv/bin/activate

id=$1

python mix_precision_train.py $@:2

curl -X POST -H 'Content-type: application/json' --data "{"worker_address\":\"$id\"}" http://gpu31.doc.ic.ac.uk:4443
