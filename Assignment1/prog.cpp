#include <iostream>
#include <omp.h>
#include <string>
#include <random>
#include <vector>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <algorithm>
#include <parallel/algorithm>

using namespace std;

const int CHARS = 100000000; // 100 million characters
// const int CHARS = 100000; // 100 thousand characters
vector<char> strA(CHARS);
vector<char> strB(CHARS);

struct HashTableEntry {

    uint64_t hash;
    int startIndex;

    bool operator<(const HashTableEntry& other) const {
        return hash < other.hash;
    }

    bool operator==(const HashTableEntry& other) const {
        return hash == other.hash;
    }
    
};

struct SubstringSearchResult {
    bool found;
    int length;
    int indexA;
    int indexB;
};


// A function to calculate the hash of a substring
uint64_t calculate_hash(const vector<uint64_t>& hash, int start, int end, const vector<uint64_t>& powers) {
    if (start == 0) {
        return hash[end];
    }
    return hash[end] - hash[start - 1] * powers[end - start + 1];
}

bool check_string_equality(const vector<char>& strA, const vector<char>& strB, int startA, int startB, int length) {
    for(int i = 0; i < length; ++i) {
        if(strA[startA + i] != strB[startB + i]) {
            return false;
        }
    }
    return true;
}

SubstringSearchResult substring_of_length_exists(int length, const vector<uint64_t>& hashA, const vector<uint64_t>& hashB, const vector<uint64_t>& powers, vector<HashTableEntry>& hashTable)  // ← Reuse!
{

    if (length == 0) return {true, 0, 0, 0}; // An empty substring is always common
    int count = hashA.size() - length + 1;

    hashTable.resize(count);


    #pragma omp parallel for schedule(static)
    for(int i = 0; i<=hashA.size() - length; ++i){
        uint64_t hash_value = calculate_hash(hashA, i, i + length - 1, powers);
        HashTableEntry entry{hash_value, i};
        hashTable[i] = entry;
    }

    __gnu_parallel::sort(hashTable.begin(), hashTable.end());

    bool found = false;
    int foundIndexA = -1;
    int foundIndexB = -1;
    
    // Now we find the hash values for strB and check for matches in the sorted hash table
    #pragma omp parallel for shared(found, foundIndexA, foundIndexB)
    for(int i = 0; i <= hashB.size() - length; ++i) {
        if (found) continue; // If we already found a common substring, skip further checks
        uint64_t hash_value = calculate_hash(hashB, i, i + length - 1, powers);
        HashTableEntry entry{hash_value, i};

        auto it = lower_bound(hashTable.begin(), hashTable.end(), entry);
        while(it != hashTable.end() && it->hash == entry.hash) {
            if (found) break;
            if(check_string_equality(strA, strB, it->startIndex, i, length)) {
                #pragma omp atomic write
                found = true;
                #pragma omp critical
                {
                    if (foundIndexA == -1) {
                        foundIndexA = it->startIndex;
                        foundIndexB = i;
                    }
                }
                break;
            }
            ++it;
        }

    }
    if(found){
        return {true, length, foundIndexA, foundIndexB};
    }
    return {false, 0, -1, -1};
}

int main(int argc, char* argv[])
{

    long long seed = (argc < 2) ? 123456LL : stoll(argv[1]);
    cout << "#####################################################################" << endl;
    cout << "Seed: " << seed << endl;
    cout << "Number of threads: " << omp_get_max_threads() << std::endl;

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

    int lo = 0, hi = CHARS;
    int mid = (lo + hi) / 2;

    SubstringSearchResult bestResult = {false, 0, -1, -1}; // Store best here
    vector<HashTableEntry> hashTable;
    hashTable.reserve(CHARS);
    while(lo<=hi) {
        mid = (lo + hi) / 2;
        SubstringSearchResult result = substring_of_length_exists(mid, hashA, hashB, powers, hashTable);
        if (result.found) {
            bestResult = result;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    auto end = chrono::high_resolution_clock::now();

    cout << "Length of longest common substring: " << hi << endl;
    cout << "Substring in strA: " << string(strA.begin() + bestResult.indexA, strA.begin() + bestResult.indexA + bestResult.length) << " ;  Start: " << bestResult.indexA << endl;
    cout << "Substring in strB: " << string(strB.begin() + bestResult.indexB, strB.begin() + bestResult.indexB + bestResult.length) << " ;  Start: " << bestResult.indexB << endl;

    std::cout << "Time taken: " 
              << chrono::duration_cast<chrono::milliseconds>(end - start).count() 
              << " ms" << std::endl;
    cout << "####################################################################" << endl;
}
