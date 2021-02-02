#!/bin/bash
resolution=(512)
for((i=0;i<=0;i++)); 
do 
    echo $i
    for((j=0;j<=13;j++));
    do
            epoch=$[100 + 100 * $j]
            var=`echo $epoch|awk '{printf("%04d",$0)}'`
            echo logs/network/ICL_512/checkpoints/model_epoch_${var}.pth
            python experiment_scripts/test_sdf.py --checkpoint_path=logs/network/ICL_512/checkpoints/model_epoch_${var}.pth --experiment_name=../model/ICL_512_model --resolution=512 #${resolution[i]}        
            #echo ${resolution[i]} ${var}
            #mv logs/model/ICL_512_model/test.ply logs/model/ICL_512_model/epoch${var}_${resolution[i]}.ply 
    done
done
