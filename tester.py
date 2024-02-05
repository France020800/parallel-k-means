import subprocess
import time
import random
import matplotlib.pyplot as plt

def run_sequential(random_seed):
    result = subprocess.run(['./bin/k-means', '10', '100000', str(random_seed)], stdout=subprocess.PIPE, text=True)
    execution_time = result.stdout.strip()
    return round(float(execution_time), 4)

def run_parallel(random_seed, num_threads):
    result = subprocess.run(['./bin/parallel-k-means', '10',  '100000', str(random_seed), str(num_threads)], stdout=subprocess.PIPE, text=True)
    execution_time = result.stdout.strip()
    return round(float(execution_time), 4)

# List to save results
speedups = []
# Generate a random seed
seed = 42
sequential_time = run_sequential(seed)
for i in range(1, 10):
    parallel_time = run_parallel(seed, i)
    speedups.append(round(sequential_time/parallel_time, 4))
    print(f'Speedup: {speedups[i-1]}\nThreads number: {i}')
    with open("result.txt", "a") as file:
        file.write("Speedup: {}s\n".format(speedups[i-1]))
        file.write("Threads number: {}\n".format(i))

print("Mean speedup: {}".format(round(sum(speedups) / len(speedups), 4)))



# Plot the results
plt.plot(range(1, 10), speedups, label="Speedup", marker='o')
plt.plot(range(1,10), range(1,10), label="Ideal", marker='o', color='coral')
plt.xlabel('Threads Number')
plt.ylabel('Speedup')
plt.title('Speedup vs. Threads Number')
plt.legend()
plt.show()
