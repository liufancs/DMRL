# Disentangled Multimodal Representation Learning for Recommendation

This is our implementation for the paper:

Fan Liu*, Huilin Chen, Zhiyong Cheng, Anan Liu, Liqiang Nie, Mohan Kankanhalli. [Disentangled Multimodal Representation Learning for Recommendation](https://arxiv.org/pdf/2203.05406.pdf). IEEE Transactions on Multimedia. (“*”= Corresponding author)

**Please cite our paper if you use our codes. Thanks!**

### Dataset
We provide five processed datasets: Amazon-Office, Amazon-Clothing, Amazon-Baby, Amazon-ToysGames, Amazon-Sports.

All of the above datasets could be downloaded from :
- Google Drive [Link](https://drive.google.com/drive/folders/1EmehilbrTMbW5pV2RIHNhopV_hnupvDj?usp=sharing)

# Office
Run DMRL.py
```
python DMRL.py --dataset Office --hidden_layer_dim_a 256 --hidden_layer_dim_b 128 --learning_rate 0.0001 --factors 4 --decay_r 1e-2 --decay_c 1e-0
```
# Baby
Run DMRL.py
```
python DMRL.py --dataset Baby --hidden_layer_dim_a 256 --hidden_layer_dim_b 128 --learning_rate 0.0001 --factors 4  --decay_r 1e-0 --decay_c 1e-3
```
# Clothing
Run DMRL.py
```
python DMRL.py --dataset Clothing --hidden_layer_dim_a 512 --hidden_layer_dim_b 256 --learning_rate 0.0001 --factors 4  --decay_r 1e+1 --decay_c 1e-3
```
# ToysGames
Run DMRL.py
```
python DMRL.py --dataset ToysGames --hidden_layer_dim_a 256 --hidden_layer_dim_b 128 --learning_rate 0.0001 --factors 4  --decay_r 1e-0 --decay_c 1e-3
```
# Sports
Run DMRL.py
```
python DMRL.py --dataset Sports --hidden_layer_dim_a 256 --hidden_layer_dim_b 128 --learning_rate 0.0001 --factors 4  --decay_r 1e-0 --decay_c 1e-0
```

Last Update Date: DEC. 04, 2022
