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
public:
    double x;
    double y;
    double z;
    int clusterId;

public:
    Point(double x, double y, double z, int cluster)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->clusterId = cluster;
    }
};

class Centroid
{
public:
    double x;
    double y;
    double z;
    int num_points;

public:
    Centroid(double x, double y, double z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};

void print_centroids(vector<Point> centroids)
{
    for (int i = 0; i < centroids.size(); i++)
    {
        printf("Centroid %d: (%d, %d)\n", i, centroids[i].x, centroids[i].y);
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

    if (argc < 4)
    {
        random_device rd;
        gen = std::mt19937(rd());
        
    }
    else
    {
        unsigned int seed = atoi(argv[3]); // Change this to any desired seed value
        gen = std::mt19937(seed);
    }

    if (argc == 5)
    {
        omp_set_num_threads(atoi(argv[4]));
    }
    else
    {
        int num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
    }
    //printf("Number of threads: %d\n", omp_get_max_threads());

    // Define the distribution
    uniform_real_distribution<double> distribution(0.0, 1000.0);

    // Create a random list of num_point points
    int num_points = atoi(argv[2]);
    vector<Point> points;
    for (int i = 0; i < num_points; i++)
    {
        points.push_back(Point(distribution(gen), distribution(gen), distribution(gen), -2));
    }

    // Create k random centroids
    int K = atoi(argv[1]);
    vector<Centroid> centroids;
    for (int i = 0; i < K; i++)
    {
        // Get K random points from the list of points
        int random_index = rand() % points.size();
        centroids.push_back(Centroid(points[random_index].x,
                                     points[random_index].y,
                                     points[random_index].z));
    }

    vector<Centroid> cumulate_centroids;
    cumulate_centroids.resize(K, Centroid(0, 0, 0));
    
    auto start = chrono::high_resolution_clock::now();

    int iter = 0;
    bool changed = true;
    while (changed)
    {
        changed = false;
        #pragma omp parallel for shared(points, centroids, cumulate_centroids, changed)
        for (int i = 0; i < points.size(); i++)
        {
            double min_distance = INFINITY;
            int centroid_index = -1;
            for (int j = 0; j < centroids.size(); j++)
            {
                double distance = pow(points[i].x - centroids[j].x, 2) +
                                  pow(points[i].y - centroids[j].y, 2) +
                                  pow(points[i].z - centroids[j].z, 2);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    centroid_index = j;
                }
            }
            if (centroid_index != points[i].clusterId)
            {
                #pragma omp atomic write
                points[i].clusterId = centroid_index;
                #pragma omp atomic write
                changed = true;
            }

            #pragma omp atomic
            cumulate_centroids[centroid_index].x += points[i].x;
            #pragma omp atomic
            cumulate_centroids[centroid_index].y += points[i].y;
            #pragma omp atomic
            cumulate_centroids[centroid_index].z += points[i].z;
            #pragma omp atomic 
            cumulate_centroids[centroid_index].num_points += 1;
        }
        if (changed)
        {
            //#pragma omp parallel for shared(centroids, cumulate_centroids)
            for (int i = 0; i < centroids.size(); i++)
            {
                centroids[i].x = cumulate_centroids[i].x / cumulate_centroids[i].num_points;
                centroids[i].y = cumulate_centroids[i].y / cumulate_centroids[i].num_points;
                centroids[i].z = cumulate_centroids[i].z / cumulate_centroids[i].num_points;
            }
            iter++;
        }
        cumulate_centroids.clear();
        cumulate_centroids.resize(centroids.size(), Centroid(0, 0, 0));
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    printf("%f\n", elapsed.count());

    return 0;
}
