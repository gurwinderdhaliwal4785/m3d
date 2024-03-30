#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <map>
#include <algorithm>
#include <mpi.h>

using namespace std;

const int NUM_TRAFFIC_LIGHTS = 5;
const int TOP_N_CONGESTED = 5;

// Function to read traffic data from input file
vector<tuple<string, int, int>> read_traffic_data(const string& filename) {
    vector<tuple<string, int, int>> data;
    ifstream input_file(filename);
    if (!input_file.is_open()) {
        cerr << "Error opening input file." << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    while (getline(input_file, line)) {
        istringstream iss(line);
        string timestamp;
        int traffic_light_id, cars_passed;
        if (!(iss >> timestamp >> traffic_light_id >> cars_passed)) {
            cerr << "Error reading input data." << endl;
            continue;
        }
        data.push_back(make_tuple(timestamp, traffic_light_id, cars_passed));
    }
    return data;
}

// Function to find top N congested traffic lights
vector<pair<int, int>> find_top_congested(const map<int, int>& congested_traffic_lights) {
    vector<pair<int, int>> sorted_congested(congested_traffic_lights.begin(), congested_traffic_lights.end());
    partial_sort(sorted_congested.begin(), sorted_congested.begin() + min(TOP_N_CONGESTED, (int)sorted_congested.size()), sorted_congested.end(),
                 [](const pair<int, int> &a, const pair<int, int> &b) { return a.second > b.second; });
    return sorted_congested;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    string input_file = argv[1];

    // Read traffic data
    vector<tuple<string, int, int>> all_traffic_data;
    if (rank == 0) {
        all_traffic_data = read_traffic_data(input_file);
    }

    // Broadcast traffic data to all processes
    int data_size = all_traffic_data.size();
    MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        all_traffic_data.resize(data_size);
    }
    MPI_Bcast(all_traffic_data.data(), data_size * sizeof(tuple<string, int, int>), MPI_CHAR, 0, MPI_COMM_WORLD);

    // Process traffic data
    map<int, int> congested_traffic_lights;
    for (const auto& data : all_traffic_data) {
        int traffic_light_id = get<1>(data);
        int cars_passed = get<2>(data);
        congested_traffic_lights[traffic_light_id] += cars_passed;
    }

    // Find top N congested traffic lights
    vector<pair<int, int>> top_congested = find_top_congested(congested_traffic_lights);

    // Gather top congested traffic lights from all processes
    vector<pair<int, int>> all_top_congested(size * TOP_N_CONGESTED);
    MPI_Gather(top_congested.data(), TOP_N_CONGESTED * sizeof(pair<int, int>), MPI_CHAR,
               all_top_congested.data(), TOP_N_CONGESTED * sizeof(pair<int, int>), MPI_CHAR,
               0, MPI_COMM_WORLD);

    // Write output to file
    if (rank == 0) {
        ofstream output_file("output.txt");
        if (!output_file.is_open()) {
            cerr << "Error opening output file." << endl;
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < all_top_congested.size(); ++i) {
            output_file << "Rank " << i / TOP_N_CONGESTED << ": ";
            output_file << "Traffic Light " << all_top_congested[i].first << ", Cars: " << all_top_congested[i].second << endl;
        }
        cout << "Output written to output.txt" << endl;
    }

    MPI_Finalize();
    return 0;
}
