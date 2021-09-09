import subprocess
from datetime import datetime
import time
import os

print("STARTING GRIDSEARCH...")
print(f"[{datetime.now()}]")
start = time.time()

bs_list     = [32, 64, 128, 256]        # Batch Size
eps_list    = [1, 20, 30, 40, 50]          # Epochs
m_list      = [0.1, 0.5, 1.0, 3.0]      # Margin
lr_list     = [0.1]                     # Learning Rate

current_abs_path = os.path.abspath(os.getcwd())

# Iterate through Grid Search
for lr in lr_list:
    for eps in eps_list:
        for m in m_list:
            for bs in bs_list:
                print(f"RUNING... BS:{bs} EPS:{eps} M:{m} LR:{lr}")
                # subprocess.call(f'python3 /{current_abs_path}/TripletNetwork_MNIST_Argparse.py -bs={bs} -eps={eps} -lr={lr} -m={m}',shell=True)
                subprocess.call(f'python3 /{current_abs_path}/TripletNetwork_Fashion_Argparse.py -bs={bs} -eps={eps} -lr={lr} -m={m}',shell=True)
                # subprocess.call(f'python3 /{current_abs_path}/TripletNetwork_Cifar10_Argparse.py -bs={bs} -eps={eps} -lr={lr} -m={m}',shell=True)

print("FINISHING GRIDSEARCH...")
end = time.time()-start
print(f"[{datetime.now()}]")
print(f"\nTotal time = {int(end//3600):02d}:{int((end//60))%60:02d}:{end%60:.6f}")
