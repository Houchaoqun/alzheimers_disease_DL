#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import time
import datetime

import shutil
from shutil import copy
import random

def backup_txt_rename(txt_path):
	if os.path.exists(txt_path):
		i = datetime.datetime.now()
		date = str(i.year) + str("%02d"%i.month) + str("%02d"%i.day) + str("%02d"%i.hour) + str("%02d"%i.minute) + str("%02d"%i.second)
		new_name = txt_path +".bak" + date
		os.rename(txt_path, new_name)
		# copy(new_name, "./backup/")
		print("copied and deleted file, new_name = {}".format(new_name))

def create_dir(dir_path):
	if not os.path.exists(dir_path):
		print("Create dir = {}".format(dir_path))
		os.makedirs(dir_path)

def get_subject_id(slice_path_root, _dir, fold_name, save_path):
	
	save_file_txt_path = os.path.join(save_path, _dir+"_"+fold_name+".txt")
	print(save_file_txt_path)
	backup_txt_rename(save_file_txt_path)
	count = 0
	exists_list = []

	labels = ["AD", "NC"]
	for lable in labels:
		slice_list_path = os.path.join(slice_path_root, _dir, lable)
		slice_list = os.listdir(slice_list_path)
		print("slice_list_path = {}".format(slice_list_path))
		with open(save_file_txt_path, "a+") as save_file_txt:
			for slice_item in slice_list:
				slice_name = slice_item.split(".")[0]
				unique_id = slice_name.split("_")[0][0:4] ## To unique the subject id: id = 072 both in AD and NC
				subject_name = unique_id + "_" + slice_name.split("_")[1]
				is_save = True

				# print("xx")
				for exists_item in exists_list:
					# exists_item = exists_item.replace("\n", "")
					# exists_item = exists_item.replace(" ", "")
					# subject_name = subject_name.replace(" ", "")
					# print("exists_item = {}, subject_name = {}".format(exists_item, subject_name))
					if (exists_item == subject_name):
						is_save = False
						break
				# print(slice_item)

				if(is_save):
					print(subject_name)
					exists_list.append(subject_name)
					save_file_txt.writelines(subject_name + "," + lable + "\n")
					count = count + 1
			print("count = {}".format(count))


if __name__=="__main__":
	### 
	run_flag = 'AD'
	save_path = "/home/reserch/documents/deeplearning/alzheimers_disease_DL/pytorch/subject_id"
	# fold_name = "single_subject_data_fold_01_entropy_except_zero"

	
	root_path = "/home/reserch/documents/deeplearning/alzheimers_disease/ADNI-825-Slice/experiments_FineTunning"
	fold_name = "single_subject_data_fold_01_train_val_test_entropy_keep_SliceNum_81"
	slice_path_root = os.path.join(root_path, fold_name)

	get_subject_id(slice_path_root, "train", fold_name, save_path)
	get_subject_id(slice_path_root, "validation", fold_name, save_path)
	get_subject_id(slice_path_root, "test", fold_name, save_path)


### run it 
### python .\specified_subject_get_slice_train_val_test.py > specified_subject_get_slice_train_val_test.txt