import subprocess
import time
import random
import matplotlib.pyplot as plt

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

# List to save results
sequential_times = []
parallel_times = []
iterations = []
for i in range(20):
    # Generate a random seed
    random_seed = random.randint(0, 2**32 - 1)
    sequential_output, sequential_time = run_sequential(random_seed)
    sequential_times.append(sequential_time)
    parallel_output, parallel_time = run_parallel(random_seed)
    parallel_times.append(parallel_time)
    iterations.append(parallel_output)
    with open("result.txt", "a") as file:
        file.write("S: {}".format(sequential_time))
        file.write(" {}\n".format(sequential_output))
        file.write("P: {}".format(parallel_time))
        file.write(" {}\n".format(parallel_output))

# Compute the speedup
speedup = []
for i in range(len(sequential_times)):
    speedup.append(round(sequential_times[i] / parallel_times[i], 4))

print("Mean speedup: {}".format(round(sum(speedup) / len(speedup), 4)))

# Prepare the result to plot
sequential_data = list(zip(iterations, sequential_times))
parallel_data = list(zip(iterations, parallel_times))
sequential_data = sorted(sequential_data, key=lambda x: x[0])
parallel_data = sorted(parallel_data, key=lambda x: x[0])
iterations = [data[0] for data in parallel_data]
sequential_times = [data[1] for data in sequential_data]
parallel_times = [data[1] for data in parallel_data]

# Plot the results
plt.plot(iterations, sequential_times, label="Sequential", marker='o')
plt.plot(iterations, parallel_times, label="Parallel", marker='o')
plt.xlabel('Iteration Number')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. Iteration Number')
plt.legend()
plt.show()