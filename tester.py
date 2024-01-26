import subprocess
import time
import random

def run_sequential(random_seed):
    start = time.time()
    result = subprocess.run(['./bin/k-means', '100', '10000', str(random_seed)], stdout=subprocess.PIPE, text=True)
    end = time.time()
    output = result.stdout.strip()
    return output, round(end - start, 4)

def run_parallel(random_seed):
    start = time.time()
    result = subprocess.run(['./bin/parallel-k-means', '100',  '10000', str(random_seed)], stdout=subprocess.PIPE, text=True)
    end = time.time()
    output = result.stdout.strip()
    return output, round(end - start, 4)

for i in range(10):
    # Generate a random seed
    random_seed = random.randint(0, 2**32 - 1)
    sequential_output, sequential_time = run_sequential(random_seed)
    parallel_output, parallel_time = run_parallel(random_seed)
    with open("result.txt", "a") as file:
        file.write("S: {}".format(sequential_time))
        file.write(" {}\n".format(sequential_output))
        file.write("P: {}".format(parallel_time))
        file.write(" {}\n".format(parallel_output))
