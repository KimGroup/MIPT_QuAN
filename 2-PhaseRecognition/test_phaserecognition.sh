echo "Start: bash ML_MIPT/trainL_gamma2.sh $1 $2 $3 $4 $5 $6 "
ch=4
h_dim=16
n_head=1
lr=1.6e-4
dropout=0.1

ker=2
gamma1=0.100
gamma2=0.850
protocol='one'

model_num=$1

for run in $4
do
for fraction in $5
do
for fractionmodel in $6
do
for set in $3
do
for L in $2
do
let batchsize=32768/$set
cmd="python3 /MIPT_QuAN/2-PhaseRecognition/test_2.py -test -setsize $set \
-noise '_noise' -datanoise 'noisy' -modelnoise 'noiseless' \
-L $L -init 1 -gamma1 $gamma1 -gamma2 $gamma2 -fraction $fraction -fractionmodel $fractionmodel \
-batchsize $batchsize -lr $lr \
-modelnum $model_num -hdim $h_dim -nhead $n_head -ch $ch \
-dim_outputs 1 -p_outputs 1 -drop $dropout \
-wandb_name 'MIPT_${model_num}_L${L}_zerosz_one_noise_g0.100vs0.850_f${fraction}_run_${run}' \
-saveprefix g2_saved_models_MIPT $7 "
echo $cmd
eval $cmd
done
done
done
done
done