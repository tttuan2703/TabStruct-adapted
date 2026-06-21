#!/bin/bash

bash script/train/synthetic_jobs.sh
bash script/train/wikisql_train_jobs.sh
bash script/test/compositional_jobs.sh
bash script/test/robustness_jobs.sh
bash script/test/structure_jobs.sh
bash script/test/wikisql_test_jobs.sh