#!/bin/bash
resolution=(128 256 512 1024)
for((i=0;i<=3;i++)); 
do
    for((j=0;j<=9;j++));
    do
            epoch=$[100 + 1000 * $j]
            var=`echo $epoch|awk '{printf("%04d",$0)}'`
            python experiment_scripts/test_sdf.py --checkpoint_path=logs/experiment_1_clean/checkpoints/model_epoch_${var}.pth --experiment_name=experiment_1_model_clean --resolution=${resolution[i]}        
            echo ${resolution[i]} ${var}
            mv logs/experiment_1_model_clean/test.ply logs/experiment_1_model_clean/epoch${var}_${resolution[i]}.ply 
    done
done
