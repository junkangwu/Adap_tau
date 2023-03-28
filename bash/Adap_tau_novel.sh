#!/bin/bash 
# 单边测试
dataset="$1"
lr="$2"
l2="$3"
context_hops="$4"
logdir="./log/"
# w_init="$4"
# temperature="$5"
n_negs="$5"
bsz="$6"
loss_fn="Adap_tau_Loss"
drop_bool="$7"
t_1="${8}"
t_2="${9}"
sampling_method="${10}"
gpus="${11}"
cnt="${12}"

generate_mode="${13}"
gnn="${14}"
tau_mode="${15}"
cd ..

if [[ $gnn = "mf" ]]
then
        gnn_name="MF"
elif [[ $gnn = "lgn" ]]
then
        gnn_name="LGN"
else
        echo "NO loss"
        exit 1
fi
# python main_ustc_v2.py
if [[ $drop_bool = "drop" ]]
then
        echo "start to drop embedding"
        name1="${dataset}_${gnn_name}_mode_${tau_mode}_${sampling_method}_${generate_mode}_TAU_${loss_fn}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_t1_${t_1}_t2_${t_2}_drop"
        echo $name1
        CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                --batch_size $bsz --gpu_id 0 --logdir $logdir \
                --n_negs  $n_negs --l2 $l2 --mess_dropout True --mess_dropout_rate 0.1  \
                --loss_fn $loss_fn --sampling_method $sampling_method \
                --generate_mode $generate_mode --u_norm --i_norm --tau_mode $tau_mode\
                --temperature $t_1 --temperature_2 $t_2 --cnt_lr $cnt --context_hops $context_hops\
                > ./outputs/${name1}.log
else
        echo "start to drop embedding"
        name1="${dataset}_${gnn_name}_mode_${tau_mode}_${sampling_method}_${generate_mode}_TAU_${loss_fn}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_t1_${t_1}_t2_${t_2}_nodrop"
        echo $name1
        CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                --batch_size $bsz --gpu_id 0 --logdir $logdir \
                --n_negs  $n_negs --l2 $l2  \
                --loss_fn $loss_fn --sampling_method $sampling_method \
                --generate_mode $generate_mode --u_norm --i_norm --tau_mode $tau_mode \
                --temperature $t_1 --temperature_2 $t_2 --cnt_lr $cnt --context_hops $context_hops\
                > ./outputs/${name1}.log
fi


