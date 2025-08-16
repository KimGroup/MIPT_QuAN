ch=4
h_dim=16
n_head=1
lr=1e-4
dropout=0.1

gamma1=0.100
gamma2=0.850
fraction=40000
model_num=$1
additional=$5
for run in 7
do
for fractionmodel in $4
do
for set in $3
do
for L in $2
do
let batchsize=32768/$set
cmd="python3 MIPT_QuAN/2-PhaseRecognition/test_attn.py -test -setsize $set \
-L $L -init 1 -gamma1 $gamma1 -gamma2 $gamma2 -fraction $fraction -fractionmodel $fractionmodel \
-batchsize $batchsize -lr $lr \
-modelnum $model_num -hdim $h_dim -nhead $n_head -ch $ch \
-dim_outputs 1 -p_outputs 1 -drop $dropout \
-wandb_name 'MIPT_${model_num}_L${L}_zerosz_${protocol}_g0.100vs${gamma2}_f${fractionmodel}_run_${run}' \
-saveprefix g2_saved_models_MIPT $additional "
echo $cmd
eval $cmd
done
done
done
done


cmd="python3 MIPT_QuAN/2-PhaseRecognition/plot_attn.py -runs 6 -Ogammas 0.05,0.9 -ts 2,12 "
echo $cmd
eval $cmd
