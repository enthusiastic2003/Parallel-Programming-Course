#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

using namespace std;
const int CHARS = 100000000;
// const int CHARS = 1000000;
const int max_hash_collisions = 1024;
vector<char> strA(CHARS);
vector<char> strB(CHARS);

struct SubstringSearchResult {
    bool found;
    int length;
    int indexA;
    int indexB;
};

__device__ uint64_t substringHasher(
    const uint64_t* __restrict__ hash, const uint64_t* __restrict__ powers, const int startIdx,const int endIdx) {
        if(startIdx == 0){
            return hash[endIdx];
        }
        else{
            return hash[endIdx] - hash[startIdx - 1] * powers[endIdx - startIdx + 1];
        }
}

__global__ void parallelSearch(const uint64_t* __restrict__ hashes, const uint* __restrict__ indices , int totalCount, int boundary, int* d_matchIdxA, int* d_matchIdxB, int* d_counter){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=totalCount - 1){
        return;
    }
    //hashes contains hashes of substrings of both A and B, sorted by hash value.
    // hashes is sorted, so we only need to compare with the next one
    if(hashes[idx] == hashes[idx+1]){
        uint pos1 = indices[idx];
        uint pos2 = indices[idx+1];

        bool is1posA = pos1 < boundary; // is first substring from A?
        bool is2posA = pos2 < boundary; // is second substring from A?

        if(is1posA != is2posA){
            int matchIdx = atomicAdd(d_counter, 1);
            int idxA = is1posA ? pos1 : pos2; // if pos1 is from A, then idxA = pos1, else idxA = pos2
            int idxB = is1posA ? pos2 - boundary : pos1 - boundary; // if pos1 is from A, then idxB = pos2 - boundary, else idxB = pos1 - boundary
            if(matchIdx < max_hash_collisions) {
                d_matchIdxA[matchIdx] = idxA;
                d_matchIdxB[matchIdx] = idxB;

            }   
        }
    }
}

__global__ void computeRollingHashes(const uint64_t* __restrict__ hashA, const uint64_t* __restrict__ hashB, uint64_t* d_substringHashes, const uint64_t* __restrict__ powers, int length, int totalSubstrings, int countPrefixHashes){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=totalSubstrings){
        return;
    }

    if(idx < countPrefixHashes){
        d_substringHashes[idx] = substringHasher(hashA, powers, idx, idx+length-1);
    } else {
        int idxB = idx - countPrefixHashes;
        d_substringHashes[idx] = substringHasher(hashB, powers, idxB, idxB+length-1);
    }
}


int main(int argc, char* argv[])
{
    long long seed = (argc < 2) ? 123456LL : stoll(argv[1]);
    cout << "#####################################################################" << endl;
    cout << "Seed: " << seed << endl;
    // Reverse the seed
    long long reversed_seed = 0;
    long long temp = seed;
    while (temp > 0) {
        reversed_seed = reversed_seed * 10 + temp % 10;
        temp /= 10;
    }

    // Create a string variable filled with random characters

    uniform_int_distribution<> dis(65, 90); // A-Z only

    mt19937 gen(seed);
    mt19937 gen2(reversed_seed);

    for (int i = 0; i < CHARS; ++i) {
        strA[i] = dis(gen);
        strB[i] = dis(gen2);
    }

    // Begin timing
    auto start = chrono::high_resolution_clock::now();
    // Implementing rolling hash
    int base = 121;

    // calculate power array
    vector<uint64_t> powers(CHARS+1);

    for(int i = 0; i < CHARS+1; ++i) {
        powers[i] = (i == 0) ? 1 : (powers[i-1] * base);
    }

    vector<uint64_t> hashA(CHARS);
    vector<uint64_t> hashB(CHARS);

    for(int i = 0; i < CHARS; ++i) {
        hashA[i] = (i == 0) ? strA[i] : (hashA[i-1] * base + strA[i]);
        hashB[i] = (i == 0) ? strB[i] : (hashB[i-1] * base + strB[i]);
    }

    int lo = 1, hi = CHARS;
    int mid;

    // TO send to GPU: strA,strB, hashA, hashB, powers, mid
    // We will use thrust to copy data to the GPU


    uint64_t* d_hashA;
    uint64_t* d_hashB;
    uint64_t* d_powers;

    cudaMalloc(&d_hashA, CHARS * sizeof(uint64_t));
    cudaMalloc(&d_hashB, CHARS * sizeof(uint64_t));
    cudaMalloc(&d_powers, (CHARS+1) * sizeof(uint64_t));

    cudaMemcpy(/* destination */ d_hashA, /* source */ hashA.data(), /* size */ CHARS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(/* destination */ d_hashB, /* source */ hashB.data(), /* size */ CHARS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(/* destination */ d_powers, /* source */ powers.data(), /* size */ (CHARS+1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    int* d_matchIdxA;
    int* d_matchIdxB;
    int *d_counter;

    cudaMalloc(&d_matchIdxA, max_hash_collisions * sizeof(int));
    cudaMalloc(&d_matchIdxB, max_hash_collisions * sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));

    cudaMemset(d_counter, 0, sizeof(int));

    uint64_t* d_substringHashes;
    cudaMalloc(&d_substringHashes , 2*CHARS * sizeof(uint64_t));
    uint* d_sortedIndices;
    cudaMalloc(&d_sortedIndices, 2*CHARS * sizeof(uint));
    thrust::device_ptr<uint64_t> t_hashes(d_substringHashes);
    thrust::device_ptr<uint> t_indices(d_sortedIndices);

    vector<SubstringSearchResult> lastSearchResults;
    while(lo<=hi){
        mid = (lo + hi) / 2;
        // Reset counter for this iteration
        cudaMemset(d_counter, 0, sizeof(int));
        // Call the kernel to search for common substrings of length mid
        vector<SubstringSearchResult> searchResults;
        int countPrefixHashes = (CHARS - mid + 1); // Number of substrings of length mid in both strings
        int totalSubstrings = 2 * countPrefixHashes; // Total substrings from both A and B
        thrust::sequence(t_indices, t_indices + totalSubstrings);
        // Create a prefix of prefixSortedHashA and sortedIndices
        int threadsPerBlock = 256;
        int blocks = (totalSubstrings + threadsPerBlock - 1) / threadsPerBlock;

        // Compute rolling hashes for all substrings of length mid in both A and B, store in d_substringHashes
        computeRollingHashes<<<blocks, threadsPerBlock>>>(d_hashA, d_hashB, d_substringHashes, d_powers, mid, totalSubstrings, countPrefixHashes);

        thrust::sort_by_key(t_hashes, t_hashes + totalSubstrings, t_indices);

        parallelSearch<<<blocks, threadsPerBlock>>>(
            d_substringHashes,
            d_sortedIndices, 
            totalSubstrings,
            countPrefixHashes,
            d_matchIdxA,
            d_matchIdxB,
            d_counter
        );

        int h_counter = 0;
        cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

        if(h_counter > max_hash_collisions) {
            
            throw runtime_error("Too many hash collisions, increase max_hash_collisions or optimize hash function");
        
        } else if (h_counter > 0 && h_counter < max_hash_collisions) {
            // Copy the candidate matches back to host
            vector<int> h_matchIdxA(h_counter);
            vector<int> h_matchIdxB(h_counter);
            cudaMemcpy(h_matchIdxA.data(), d_matchIdxA, h_counter*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_matchIdxB.data(), d_matchIdxB, h_counter*sizeof(int), cudaMemcpyDeviceToHost);

            for(int i =0;i<h_counter; i++){
                int startA = h_matchIdxA[i];
                int startB = h_matchIdxB[i];
                bool match = true;
                for(int j = 0;j<mid;j++){
                    if(strA[startA+j]!=strB[startB + j]){
                        match = false;
                        break;
                    }
                }
                if(match){
                    searchResults.push_back({true, mid, startA, startB});
                }
            }

        }

        if(searchResults.empty() == false) {
            lastSearchResults = searchResults;
            lo = mid + 1; // Search in the upper half
        } else {
            hi = mid - 1; // Search in the lower half
        }

    }

    // End timing
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "\n==================== Results ====================" << endl;
    if (lastSearchResults.empty()) {
        cout << "No common substring found." << endl;
    } else {
        cout << "Total matches found             : " << lastSearchResults.size() << endl;
        cout << "Longest common substring length : " << lastSearchResults.front().length << endl;
        cout << "Time taken                      : " << duration << " ms" << endl;

        const auto& result = lastSearchResults.front();
        cout << "\n--- Match 1 ---" << endl;
        cout << "String A start index            : " << result.indexA << endl;
        cout << "String B start index            : " << result.indexB << endl;

        int printLen = min(result.length, 100);
        cout << "Substring from A                : " << string(strA.begin() + result.indexA, strA.begin() + result.indexA + printLen);
        if (result.length > 100) cout << "...";
        cout << endl;
        cout << "Substring from B                : " << string(strB.begin() + result.indexB, strB.begin() + result.indexB + printLen);
        if (result.length > 100) cout << "...";
        cout << endl;
        cout << "================================================" << endl;
    }

    cudaFree(d_hashA);
    cudaFree(d_hashB);
    cudaFree(d_powers);
    cudaFree(d_matchIdxA);
    cudaFree(d_matchIdxB);
    cudaFree(d_counter);
    cudaFree(d_substringHashes);
    cudaFree(d_sortedIndices);

}