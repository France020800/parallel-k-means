import subprocess
import os
import matplotlib.pyplot as plt

def run_sequential(random_seed):
    result = subprocess.run(['./bin/k-means', '10', '1000000', str(random_seed)], stdout=subprocess.PIPE, text=True)
    output = result.stdout.strip()
    return output

def run_parallel(random_seed, num_threads):
    result = subprocess.run(['./bin/parallel-k-means', '10',  '1000000', str(random_seed), str(num_threads)], stdout=subprocess.PIPE, text=True)
    output = result.stdout.strip()
    return output

# Create the environment
os.system('makedir bin')
os.system('g++ -o bin/parallel-k-means -fopenmp parallel-k-means.cpp')
os.system('g++ -o bin/k-means k-means.cpp')

# List to save results
speedups = []
# Generate a random seed
seed = 42
sequential_time = run_sequential(seed)
print(f'Sequential time: {sequential_time}')
for i in range(1, 10):
    parallel_time = run_parallel(seed, i)
    speedups.append(round(sequential_time/parallel_time, 4))
    print(f'Speedup: {speedups[i-1]}\nThreads number: {i}')
    with open("result.txt", "a") as file:
        file.write("Speedup: {}s\n".format(speedups[i-1]))
        file.write("Threads number: {}\n".format(i))

print("Mean speedup: {}".format(round(sum(speedups) / len(speedups), 4)))



# Plot the results
plt.plot(range(1, 10), speedups, label="Sequential", marker='o')
plt.xlabel('Threads Number')
plt.ylabel('Speedup')
plt.title('Speedup vs. Threads Number\n10 clusters and 1M points')
plt.legend()
plt.show()
