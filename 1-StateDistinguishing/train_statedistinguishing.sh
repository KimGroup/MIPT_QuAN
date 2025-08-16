
ch=4
h_dim=16
n_head=4
lr=1e-4
dropout=0.1


model_num=$1
for run in 0 1 2 3
do
for gamma in $5
do
for fraction in $4
do
for set in $3
do
for L in $2
do
let batchsize=2048/$set
epoch=100
cmd="python3 MIPT_QuAN/1-StateDistinguishing/train_1.py -setsize $set \
-L $L -init 1 -gamma $gamma -fraction $fraction \
-epoch $epoch -batchsize $batchsize -lr $lr \
-modelnum $model_num -hdim $h_dim -nhead $n_head -ch $ch \
-dim_outputs 1 -p_outputs 1 -drop $dropout \
-wandb_name 'MIPT_${model_num}_L${L}_g${gamma}_run_${run}' \
-prefix '/home/hk684' -saveprefix g2_saved_models_MIPT $6 "
echo $cmd
eval $cmd
done
done
done
done
done
