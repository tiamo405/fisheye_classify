import os
import shutil
import random

# Đường dẫn tới folder gốc A
folder_A_path = "/mnt/nvme0n1/phuongnam/fisheye_classify/duoc_si_PMC"

# Đường dẫn tới folder đích X
folder_X_path = "/mnt/nvme0n1/phuongnam/fisheye_classify/data"

# Tạo các thư mục train, val và test trong folder X
os.makedirs(os.path.join(folder_X_path, 'train', 'black'))
os.makedirs(os.path.join(folder_X_path, 'train', 'white'))
os.makedirs(os.path.join(folder_X_path, 'train', 'customer'))

os.makedirs(os.path.join(folder_X_path, 'val', 'black'))
os.makedirs(os.path.join(folder_X_path, 'val', 'white'))
os.makedirs(os.path.join(folder_X_path, 'val', 'customer'))

os.makedirs(os.path.join(folder_X_path, 'test', 'black'))
os.makedirs(os.path.join(folder_X_path, 'test', 'white'))
os.makedirs(os.path.join(folder_X_path, 'test', 'customer'))


# Tỷ lệ dữ liệu train, val và test (ví dụ: 70-15-15)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Lấy danh sách tệp tin trong folder black và white
files_B = os.listdir(os.path.join(folder_A_path, 'black'))
files_C = os.listdir(os.path.join(folder_A_path, 'white'))
files_D = os.listdir(os.path.join(folder_A_path, 'customer'))


# Sắp xếp danh sách tệp tin ngẫu nhiên
random.shuffle(files_B)
random.shuffle(files_C)
random.shuffle(files_D)


# Tính toán số lượng tệp tin cho từng tập dữ liệu
total_files_B = len(files_B)
total_files_C = len(files_C)
total_files_D = len(files_D)


train_count_B = int(train_ratio * total_files_B)
val_count_B = int(val_ratio * total_files_B)
test_count_B = total_files_B - train_count_B - val_count_B

train_count_C = int(train_ratio * total_files_C)
val_count_C = int(val_ratio * total_files_C)
test_count_C = total_files_C - train_count_C - val_count_C

train_count_D = int(train_ratio * total_files_D)
val_count_D = int(val_ratio * total_files_D)
test_count_D = total_files_D - train_count_D - val_count_D

# Di chuyển tệp tin vào các thư mục tương ứng trong folder X
for i in range(train_count_B):
    shutil.copy(os.path.join(folder_A_path, 'black', files_B[i]), os.path.join(folder_X_path, 'train', 'black', files_B[i]))

for i in range(train_count_C):
    shutil.copy(os.path.join(folder_A_path, 'white', files_C[i]), os.path.join(folder_X_path, 'train', 'white', files_C[i]))

for i in range(train_count_C):
    shutil.copy(os.path.join(folder_A_path, 'customer', files_D[i]), os.path.join(folder_X_path, 'train', 'customer', files_D[i]))


for i in range(train_count_B, train_count_B + val_count_B):
    shutil.copy(os.path.join(folder_A_path, 'black', files_B[i]), os.path.join(folder_X_path, 'val', 'black', files_B[i]))

for i in range(train_count_C, train_count_C + val_count_C):
    shutil.copy(os.path.join(folder_A_path, 'white', files_C[i]), os.path.join(folder_X_path, 'val', 'white', files_C[i]))

for i in range(train_count_C, train_count_C + val_count_C):
    shutil.copy(os.path.join(folder_A_path, 'customer', files_D[i]), os.path.join(folder_X_path, 'val', 'customer', files_D[i]))



for i in range(train_count_B + val_count_B, total_files_B):
    shutil.copy(os.path.join(folder_A_path, 'black', files_B[i]), os.path.join(folder_X_path, 'test', 'black', files_B[i]))

for i in range(train_count_C + val_count_C, total_files_C):
    shutil.copy(os.path.join(folder_A_path, 'white', files_C[i]), os.path.join(folder_X_path, 'test', 'white', files_C[i]))

for i in range(train_count_C + val_count_C, total_files_C):
    shutil.copy(os.path.join(folder_A_path, 'customer', files_D[i]), os.path.join(folder_X_path, 'test', 'customer', files_D[i]))

print("Chia dữ liệu thành công!")

