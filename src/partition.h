/*
 * partition.h
 *
 *  Created on: Dec 29, 2017
 *      Author: Sameer
 */

#include <armadillo>
using namespace std;
using namespace arma;

#ifndef PARTITION_H_
#define PARTITION_H_
typedef class Partition* LPPartition;
class Partition
{
	//data members
public:
	int nObs; // number of indices
	int K; // number of clusters
	int* cluster_config; // sizes of each cluster
    int** clusters; // the actual clusters  // might be easier to use vectors.
	int* cluster_assignment; // size nObs, tells us which cluster each index belongs to
	int** pairwise_assignment; // array of pairwise assignments
	double* log_like; // size K. holds the log-likelihood evaluated on each cluster
	double* log_prior; // size K. holds the log-prior evaluated on each cluster
	double* beta_hat; // will have size nObs. holds the point estimates of beta in each blockgroup
	// methods
public:
	Partition(); // constructor
	Partition(LPPartition initial_partition); // constructor. create a new partition from an old one
	~Partition(); // destructor
public:
	void Copy_Partition(LPPartition initial_partition);
//	void Initialize_Partition(int n, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta);
	void Initialize_Partition(int n);
//	void Initialize_Partition2(int id,  mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta);
	void Initialize_Partition2(int id);
	void get_pairwise();
//	void log_likelihood(int cluster_id, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu);
	void log_likelihood(int cluster_id);
//	void log_pi_ep(int cluster_id, double eta);
	void log_pi_ep(int cluster_id);
//	void beta_postmean(int cluster_id, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu);
	void beta_postmean(int cluster_id);
	void Print_Partition();
  void Print_Partition_ToFile(string file);
	void Print_Means();
//	void Split(int split_k, int* new_cluster1, int* new_cluster2, int size1, int size2, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta); // split cluster k into two parts: new_cluster1, new_cluster2
	void Split(int split_k, int* new_cluster1, int* new_cluster2, int size1, int size2); // split cluster k into two parts: new_cluster1, new_cluster2
//	void Merge(int k_1, int k_2, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta); // merge cluster max(k_1, k_2) into min(k_1, k_2)
	void Merge(int k_1, int k_2); // merge cluster max(k_1, k_2) into min(k_1, k_2)

	// splits cluster split_k into two parts (new_cluster1, new_cluster2). Then it merges new_cluster1 and existing cluster k_star_1, and new_cluster2 with existing cluster k_star_2
//	void Split_and_Merge(int split_k, int* new_cluster1, int* new_cluster2, int size1, int size2, int k_star_1, int k_star_2, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta);
	void Split_and_Merge(int split_k, int* new_cluster1, int* new_cluster2, int size1, int size2, int k_star_1, int k_star_2);

//	void Modify(int cl_ind, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta);
	void Modify(int cl_ind);


	void Find_Splits(int cluster_id);
};



#endif /* PARTITION_H_ */
