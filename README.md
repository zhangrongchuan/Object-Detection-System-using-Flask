## Training
### *TERO*
In order to reproduce the result(tMRR) on different datasets with GPU, run the following command,
```cmd
python Main.py --model TERO --dim 500 --lr 0.1 --gamma 120 --loss logloss --timedisc 0 —gpu --gran 2 --valid 3 --t_eval_type t_mrr --dataset ICEWS14 --eta 10  
python Main.py --model TERO --dim 500 --lr 0.1 --gamma 120 --loss logloss --timedisc 0 —gpu --gran 2 --valid 3 --t_eval_type t_mrr --dataset ICEWS05-15 --eta 10  
python Main.py --model TERO --dim 500 --lr 0.1 --gamma 120 --loss logloss --timedisc 0 —gpu --gran 2 --valid 3 --t_eval_type t_mrr --dataset gdelt --eta 10  
```
In order to reproduce the result(MRR-T, MAE) on different datasets with GPU, run the following command,
```cmd
python Main.py --model TERO --dim 500 --lr 0.1 --gamma 120 --loss logloss --timedisc 0 —gpu --gran 2 --task TimePrediction --valid 3 --dataset ICEWS14 --eta 10  
python Main.py --model TERO --dim 500 --lr 0.1 --gamma 120 --loss logloss --timedisc 0 —gpu --gran 2 --task TimePrediction --valid 3 --dataset ICEWS05-15 --eta 10  
python Main.py --model TERO --dim 500 --lr 0.1 --gamma 120 --loss logloss --timedisc 0 —gpu --gran 2 --task TimePrediction --valid 3 --dataset gdelt --eta 10  
```

### *ATISE*
In order to reproduce the result(tMRR) on different datasets with GPU, run the following command,
```cmd
python Main.py --model ATISE --dim 500 --lr 0.00003 --gamma 120 --loss logloss --timedisc 0 --gpu --gran 3 --cmin 0.003 --valid 3 --t_eval_type t_mrr --dataset ICEWS14
python Main.py --model ATISE --dim 500 --lr 0.00003 --gamma 120 --loss logloss --timedisc 0 --gpu --gran 3 --cmin 0.003 --valid 3 --t_eval_type t_mrr --dataset ICEWS05-15
python Main.py --model ATISE --dim 500 --lr 0.00003 --gamma 120 --loss logloss --timedisc 0 --gpu --gran 3 --cmin 0.003 --valid 3 --t_eval_type t_mrr --dataset gdelt
```

In order to reproduce the result(MRR-T, MAE) on different datasets with GPU, run the following command,
```cmd
python Main.py --model ATISE --dim 500 --lr 0.00003 --gamma 120 --loss logloss --timedisc 0 --gpu --gran 3 --cmin 0.003 --task TimePrediction --valid 3 --dataset ICEWS14
python Main.py --model ATISE --dim 500 --lr 0.00003 --gamma 120 --loss logloss --timedisc 0 --gpu --gran 3 --cmin 0.003 --task TimePrediction --valid 3 --dataset ICEWS05-15
python Main.py --model ATISE --dim 500 --lr 0.00003 --gamma 120 --loss logloss --timedisc 0 --gpu --gran 3 --cmin 0.003 --task TimePrediction --valid 3 --dataset gdelt
```

### *TComplEx*
In order to reproduce the result(MRR-T, MAE, tMRR) on different datasets with GPU, run the following command,
```cmd
python tkbc/learner.py --model TComplEx --rank 1500 --emb_reg 1e-2 --time_reg 1e-1 --time_eval --gpu --max_epochs 500 --dataset ICEWS14
python tkbc/learner.py --model TComplEx --rank 1500 --emb_reg 1e-2 --time_reg 1e-1 --time_eval --gpu --max_epochs 500 --dataset ICEWS05-15
python tkbc/learner.py --model TComplEx --rank 1500 --emb_reg 1e-2 --time_reg 1e-1 --time_eval --gpu --max_epochs 500 --dataset gdelt
```

### *TNTComplEx*
In order to reproduce the result(MRR-T, MAE, tMRR) on different datasets with GPU, run the following command,
```cmd
python tkbc/learner.py --model TNTComplEx --rank 1500 --emb_reg 1e-2 --time_reg 1e-1 --time_eval --gpu --max_epochs 500 --dataset ICEWS14
python tkbc/learner.py --model TNTComplEx --rank 1500 --emb_reg 1e-2 --time_reg 1e-1 --time_eval --gpu --max_epochs 500 --dataset ICEWS05-15
python tkbc/learner.py --model TNTComplEx --rank 1500 --emb_reg 1e-2 --time_reg 1e-1 --time_eval --gpu --max_epochs 500 --dataset gdelt
```

### *TimePlex*
In order to reproduce the result(MRR-T, MAE, tMRR) on different datasets with GPU, run the following command to train first,
```cmd
python main.py -d icews14 -m TimePlex_base -a '{"embedding_dim":200, "srt_wt": 5.0, "ort_wt": 5.0, "sot_wt": 5.0, "time_reg_wt":1.0, "emb_reg_wt":0.005}' -l crossentropy_loss_AllNeg -r 0.1 -b 1000 -x 2000 -n 0 -v 1 -q 0 -y 500 -g_reg 2 -g 1.0 --filter_method time-str -e 250 --flag_add_reverse 1 --save_dir icews14_timeplex_base
python main.py -d icews05-15 -m TimePlex_base -a '{"embedding_dim":200, "srt_wt": 5.0, "ort_wt": 5.0, "sot_wt": 5.0, "time_reg_wt":5.0, "emb_reg_wt":0.005}' -l crossentropy_loss_AllNeg -r 0.1 -b 1000 -x 2000 -n 0 -v 1 -q 0 -y 500 -g_reg 2 -g 1.0 --filter_method time-str -e 250 --flag_add_reverse 1 --save_dir icews05-15_timeplex_base
python main.py -d gdelt -m TimePlex_base -a '{"embedding_dim":200, "srt_wt": 5.0, "ort_wt": 5.0, "sot_wt": 5.0, "time_reg_wt":5.0, "emb_reg_wt":0.005}' -l crossentropy_loss_AllNeg -r 0.1 -b 1000 -x 2000 -n 0 -v 1 -q 0 -y 500 -g_reg 2 -g 1.0 --filter_method time-str -e 250 --flag_add_reverse 1 --save_dir gdelt_timeplex_base
```
And then, run the following command to test to generate the result,
```cmd
python main.py -d icews14 -m TimePlex_base --resume_from_save "./models/icews14_timeplex_base/best_valid_model.pt"  --mode test --filter_method time-str -y 40 --flag_add_reverse 1
python main.py -d icews05-15 -m TimePlex_base --resume_from_save "./models/icews05-15_timeplex_base/best_valid_model.pt"  --mode test --filter_method time-str -y 40 --flag_add_reverse 1
python main.py -m TimePlex_base -a '{"embedding_dim":200, "srt_wt": 5.0, "ort_wt": 5.0, "sot_wt": 5.0, "time_reg_wt":5.0, "emb_reg_wt":0.005}' -l crossentropy_loss_AllNeg -r 0.1 -b 1000 -x 2000 -n 0 -v 1 -q 0 -y 500 -g_reg 2 -g 1.0 --filter_method time-str -e 250 --flag_add_reverse 1 --save_dir gdelt_timeplex_base -d gdelt -pr 1 -pt 1 -x 500
