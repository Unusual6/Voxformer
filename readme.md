train:
tools/dist_train.sh CONFIG_path 2 

test:
tools/dist_test.sh CONFIG_path CHECKPOINT_path 2

Vis_system:
1. cd tools/vis_system
2. python main.py
3. open http://127.0.0.1:5000