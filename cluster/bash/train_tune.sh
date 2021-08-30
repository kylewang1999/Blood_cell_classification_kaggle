python train_custom_tune.py --model_path ./eval-darts-hybrid-reorg-20210828-063008/weights.pt --arch DARTS_TS_BC_50EPOCH --save FOO

mv .basophil basophil
mv .erythroblast erythroblast
mv .ig ig
mv .platelet platelet

mv eosinophil 0_eosinophil
mv lymphocyte 1_lymphocyte
mv monocyte 2_monocyte
mv neutrophil 3_neutrophil