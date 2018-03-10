#!/bin/bash

# python hcq_densenet169.py
# python hcq_densenet201.py

python resume_model.py > result_test_single_subject_81.txt
python resume_model.py > result_validation_single_subject_81.txt
python resume_model.py > result_train_single_subject_81.txt