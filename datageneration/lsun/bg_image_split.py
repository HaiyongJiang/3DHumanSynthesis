import os
import random
# src = "/root/zzy/3DDataGenerate/SUN/SUN397"
src = "/home/meitu/jhy/projects/surreal/datageneration/lsun/camvid_701/"
sub_folder_list = os.listdir(src)
train_file_path = os.path.join(src,"train_img.txt")
test_file_path = os.path.join(src,"test_img.txt")
train_list = []
test_list = []
train_file = open(train_file_path,"w")
test_file = open(test_file_path,"w")

for sub_folder_name in sub_folder_list:
    print("list files in folder: " + sub_folder_name)
    try:
        sub_folder_path = os.path.join(src,sub_folder_name)
        image_list = os.listdir(sub_folder_path)
	print("found files: #%d"%(len(image_list)))
        random.shuffle(image_list)
        image_list_len = len(image_list)
	for image_id in range(0,image_list_len):
	    fname = os.path.join(sub_folder_path, image_list[image_id])
	    if image_list[image_id].split(".")[-1]!="png":
		continue
	    if image_id < int(image_list_len*0.8):
		train_file.write(fname + "\n")
	    else:
		test_file.write(fname+"\n")
    except:
        continue

train_file.close()
test_file.close()
