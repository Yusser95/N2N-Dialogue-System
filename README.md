# N2N-Dialogue-System
 ## implementation of the paper ”A Network-based End-to-End Trainable Task-oriented ## Dialogue System”

# environemnt
## python3.9.9

# files structure
## │   .gitignore
## │   README.md
## │   requirements.txt
## │   test.ipynb
## │
## └───src
##     │   belief_tracker.py
##     │   cli.py
##     │   config.json
##     │   database_operator.py
##     │   data_loader.py
##     │   generation_network.py
##     │   intent_network.py
##     │   n2n_dialogue_system.py
##     │   policy_network.py
##     │   train.py
##     │   utils.py
##     │   __init__.py
##     │
##     ├───data
##     │       CamRest.json
##     │       CamRestOTGY.json
##     │       groups.json
##     │       sys_vocab.json
##     │       vocab_tiny.model
##     │       vocab_tiny.model.w2i
##     │       woz2_dev.delexed.json
##     │       woz2_test.json
##     │       woz2_train.delexed.json
##     │
##     └───models
##             tracker_5401.model

# Run
## download the following files and extract them in the corrosponding directories

## src/data
## https://drive.google.com/file/d/1IvByN8Efs0x5A7dzpRyzDVjnBwAeABT_/view?usp=sharing

## src/models
## https://drive.google.com/file/d/1CzAq2DohPW-CLZnacddvnUc6dFk6szXV/view?usp=sharing



## you can run from notebook 'test.ipynb'
## or
## pip install -r requirements.txt
## python train.py
## python cli.py

# reference
## A Network-based End-to-End Trainable Task-oriented Dialogue System; Wen, Tsung-Hsien, et. al.

## https://github.com/edward-zhu/dialog
