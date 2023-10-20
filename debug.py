import torch

# Kiểm tra xem CUDA có khả dụng không
if torch.cuda.is_available():
    # PyTorch có thể sử dụng CUDA
    device = torch.device("cuda")
    print("CUDA is available!")
else:
    # PyTorch sử dụng CPU vì CUDA không khả dụng
    device = torch.device("cpu")
    print("CUDA is not available, using CPU instead.")
