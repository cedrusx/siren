#!/bin/bash
for((i=2;i<=4;i++));
do
        last=$[$i-1]
        #echo $last
        echo "s/num_hidden_layers=${last}/num_hidden_layers=${i}/g"
        perl -i -pe "s/num_hidden_layers=${last}/num_hidden_layers=${i}/g" modules.py
        for((j=0;j<=2;j++));
        do
             last_channel=$[2**$j*32]
             echo "s/hidden_features=${last_channel}/hidden_features=$[2**$j*64]/g"
             perl -i -pe "s/hidden_features=${last_channel}/hidden_features=$[2**$j*64]/g" modules.py
                   for((k=0;k<=3;k++));
                   do   
                        last_term=${k}
                        echo "s/-1e${last_term}/-1e$[${k}+1]/g"
                        perl -i -pe "s/-1e${last_term}/-1e$[${k}+1]/g" loss_functions.py
                        python experiment_scripts/train_sdf.py --model_type=sine --point_cloud_path=data/ICL_clean_0.xyz --batch_size=150000 --experiment_name=single_frame_$[2**$j*64]_${i}_1e$[${k}+1] --epochs_til_ckpt=100
                        python experiment_scripts/test_sdf.py --checkpoint_path=logs/single_frame_$[2**$j*64]_${i}_1e$[${k}+1]/checkpoints/model_current.pth --experiment_name=experiment_1_rec --resolution=512
                        mv logs/experiment_1_rec/test.ply logs/experiment_1_rec/single_frame_$[2**$j*64]_${i}_1e$[${k}+1].ply
                   done
        done
done
