#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;

class Point
{
private:
    int x;
    int y;
    int z;
    int clusterId;

public:
    Point(int x, int y, int z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    int get_x()
    {
        return this->x;
    }
    int get_y()
    {
        return this->y;
    }
    int get_z()
    {
        return this->z;
    }
    int get_clusterId()
    {
        return this->clusterId;
    }
    void set_clusterId(int clusterId)
    {
        this->clusterId = clusterId;
    }
};

void print_centroids(vector<Point> centroids)
{
    for (int i = 0; i < centroids.size(); i++)
    {
        printf("Centroid %d: (%d, %d)\n", i, centroids[i].get_x(), centroids[i].get_y());
    }
}

int main(int argc, char const *argv[])
{
    if (argc < 3)
    {
        printf("Error: command-line argument count mismatch. \n ./k-means <K> <num_points> \n");
        return 1;
    }

    // Seed the random number generator
    std::mt19937 gen;
    if (argc >= 4)
    {
        unsigned int seed = atoi(argv[3]); // Change this to any desired seed value
        gen = std::mt19937(seed);
    }
    else
    {
        random_device rd;
        gen = std::mt19937(rd());
    }

    if (argc == 5)
    {
        omp_set_num_threads(atoi(argv[4]));
    } else {
        int num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
    }
    

    // Define the distribution (0 to 100 inclusive)
    uniform_int_distribution<int> distribution(0, 100000);

    // Create a random list of num_points points
    int num_points = atoi(argv[2]);
    vector<Point> points;
    for (int i = 0; i < num_points; i++)
    {
        points.push_back(Point(distribution(gen), distribution(gen), distribution(gen)));
    }

    // Declare k random points as centroids
    int K = atoi(argv[1]);
    vector<Point> centroids;
    for (int i = 0; i < K; i++)
    {
        // Get K random points from the list of points
        int random_index = rand() % points.size();
        centroids.push_back(points[random_index]);
    }

    // printf("Initial centroids:\n");
    // print_centroids(centroids);

    // Start the timer
    // auto start = chrono::high_resolution_clock::now();

    bool complete = false;
    int iter = 0;
    while (!complete)
    {
        // Assign each point to a centroid
        #pragma omp parallel for
        for (int i = 0; i < points.size(); i++)
        {
            int min_distance = 1000000;
            int min_index = 0;
            #pragma omp parallel for reduction(min:min_distance) shared(min_index)
            for (int j = 0; j < K; j++)
            {
                int distance = (int)sqrt(pow(points[i].get_x() - centroids[j].get_x(), 2) + pow(points[i].get_y() - centroids[j].get_y(), 2) + pow(points[i].get_z() - centroids[j].get_z(), 2));
                if (distance < min_distance)
                {
                    min_distance = distance;
                    min_index = j;
                }
            }
            points[i].set_clusterId(min_index);
        }

        // Recalculate the centroids
        complete = true;
        #pragma omp parallel for
        for (int i = 0; i < K; i++)
        {
            int sum_x = 0;
            int sum_y = 0;
            int sum_z = 0;
            int count = 0;
            #pragma omp parallel for reduction(+:sum_x,sum_y,sum_z,count)
            for (int j = 0; j < points.size(); j++)
            {
                if (points[j].get_clusterId() == i)
                {
                    sum_x += points[j].get_x();
                    sum_y += points[j].get_y();
                    sum_z += points[j].get_z();
                    count++;
                }
            }
            if (centroids[i].get_x() != sum_x / count || centroids[i].get_y() != sum_y / count, centroids[i].get_z() != sum_z / count)
            {
                complete = false;
                centroids[i] = Point(sum_x / count, sum_y / count, sum_z / count);
            }
        }
        iter++;
    }

    // printf("Final centroids:\n");
    // print_centroids(centroids);

    // Stop the timer
    // auto stop = chrono::high_resolution_clock::now();

    // Calculate the execution time
    // std::chrono::duration<double> duration = stop - start;

    // printf("Program complete after %d iterations.\n", iter);
    // printf("Execution time: %.4f seconds\n", duration.count());
    printf("%d", iter);

    return 0;
}
