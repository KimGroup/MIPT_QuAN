ch=4
h_dim=16
n_head=1
lr=1e-4
dropout=0.1

gamma1=0.100
gamma2=0.850

model_num=$1
for run in 0 1 2 3 4 5 6 7
do
for fraction in $4
do
for set in $3
do
for L in $2
do
let batchsize=32768/$set
let epoch=$(echo "scale=0; 8000000/$fraction" | bc -l)
cmd="python3 /MIPT_QuAN/2-PhaseRecognition/train_2.py -test -setsize $set -noise '_noise' \
-L $L -init 1 -gamma1 $gamma1 -gamma2 $gamma2 -fraction $fraction \
-epoch $epoch -batchsize $batchsize -lr $lr \
-modelnum $model_num -hdim $h_dim -nhead $n_head -ch $ch\
-dim_outputs 1 -p_outputs 1 -drop $dropout \
-wandb_name 'MIPT_${model_num}_L${L}_zerosz_one_noise_g${gamma1}vs${gamma2}_f${fraction}_run_${run}' \
-saveprefix g2_saved_models_MIPT $5 "
echo $cmd
eval $cmd
done
done
done
done
