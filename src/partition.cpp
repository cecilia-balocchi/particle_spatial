/*
 * partition.cpp
 *
 *  Created on: Dec 29, 2017
 *      Author: Sameer
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <limits.h>
#include <armadillo>
#include <time.h>
#include "partition.h"
#include "various_functions.h"
using namespace std;

using namespace arma;

extern arma::mat Y;
extern arma::mat X;
extern arma::mat A_block;
// Set the hyper-parameters
extern double rho;
extern double a;
extern double b;
extern double nu;
extern double alpha;
extern double eta;

//class Partition::Begins
Partition::Partition(){
  nObs = 0;
  K = 0;
  cluster_config = NULL;
  clusters = NULL;
  cluster_assignment = NULL;
  pairwise_assignment = NULL;
  log_like = NULL;
  log_prior = NULL;
  beta_hat = NULL;
  return;
}
// initializes a partition
Partition::Partition(LPPartition initial_partition){
  nObs = initial_partition->nObs;
  K = initial_partition->K;
  cluster_config = new int[K];
  clusters = new int*[K];
  cluster_assignment = new int[nObs];
  pairwise_assignment = new int*[nObs];
  log_like = new double[K];
  log_prior = new double[K];
  beta_hat = new double[nObs];

  for(int k = 0; k < K; k++){
	cluster_config[k] = initial_partition->cluster_config[k];
	log_like[k] = initial_partition->log_like[k];
	log_prior[k] = initial_partition->log_prior[k];
	clusters[k] = new int[cluster_config[k]];
	for(int i = 0; i < cluster_config[k]; i++){
	  clusters[k][i] = initial_partition->clusters[k][i];
	}
  }

  for(int i = 0; i < nObs; i++){
	cluster_assignment[i] = initial_partition->cluster_assignment[i];
	beta_hat[i] = initial_partition->beta_hat[i];
	pairwise_assignment[i] = new int[nObs];
	for(int j = 0; j < nObs; j++){
	  pairwise_assignment[i][j] = initial_partition->pairwise_assignment[i][j];
	}
  }
  return;
}

Partition::~Partition(){
	int i;
	delete[] cluster_config; cluster_config = NULL;
	for(i = 0; i < K; i++){
		delete[] clusters[i]; clusters[i] = NULL;
	}
	delete[] clusters; clusters = NULL;
	delete[] cluster_assignment; cluster_assignment = NULL;
	for(i = 0; i < nObs; i++){
		delete[] pairwise_assignment[i]; pairwise_assignment[i] = NULL;
	}
	delete[] pairwise_assignment; pairwise_assignment = NULL;
	delete[] log_like; log_like = NULL;
	delete[] log_prior; log_prior = NULL;
	delete[] beta_hat; beta_hat = NULL;
	return;
}

/*
Partition::~Partition(){

  delete[] cluster_config;
  delete[] cluster_assignment;
  delete[] log_like;
  delete[] log_prior;
  delete[] beta_hat;

  for(int k = 0; k < K; k++){
	  delete[] clusters[k];
  }
  delete[] clusters;
  for(int i = 0; i < nObs; i++){
	  delete[] pairwise_assignment[i];
  }
  delete[] pairwise_assignment;
}
*/

void Partition::Copy_Partition(LPPartition initial_partition){
  delete[] cluster_config;
  delete[] log_like;
  delete[] log_prior;
  //std::cout << "[Copy_Partition]: Freed cluster_config, log_like, log_prior" << std::endl;

  for(int k = 0; k < K; k++){
	delete[] clusters[k];
  }
  delete[] clusters;
  //std::cout << "[Copy_Partitoin]: freed clusters" << std::endl;
  K = initial_partition->K;

  cluster_config = new int[K];
  log_like = new double[K];
  log_prior = new double[K];
  clusters = new int*[K];

  for(int i = 0; i < nObs; i++){
	cluster_assignment[i] = initial_partition->cluster_assignment[i];
	beta_hat[i] = initial_partition->beta_hat[i];
	for(int j = 0 ; j < nObs; j++){
	  pairwise_assignment[i][j] = initial_partition->pairwise_assignment[i][j];
	}
  }
  for(int k = 0; k < K; k++){
	cluster_config[k] = initial_partition->cluster_config[k];
	clusters[k] = new int[cluster_config[k]];
	for(int i = 0; i < cluster_config[k]; i++){
	  clusters[k][i] = initial_partition->clusters[k][i];
	}
	log_like[k] = initial_partition->log_like[k];
	log_prior[k] = initial_partition->log_prior[k];
  }
  //std::cout << "[Copy_Partition]: Finished Copy_Partition" << std::endl;

  return;
}
//void Partition::Initialize_Partition(int n, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta){
void Partition::Initialize_Partition(int n){
  nObs = n;
  K = 1;
  cluster_config = new int[K];
  cluster_config[0] = nObs; // initial partition has 1 cluster of size nObs
  clusters = new int*[K];
  cluster_assignment = new int[nObs];

  log_like = new double[K];
  log_prior = new double[K];
  beta_hat = new double[nObs];

  for(int k = 0; k < K; k++){
	clusters[k] = new int[cluster_config[k]];
  }
  // now since K = 1, we only have one cluster:
  for(int i = 0; i < nObs; i++){
	clusters[0][i] = i;
  }
  //memset(cluster_assignment, 1, nObs*sizeof(int));
  for(int i = 0; i < nObs; i++){
	cluster_assignment[i] = 0; // initial partition has 1 cluster
  }
  get_pairwise();

  // log_likelihood and log_prior
  for(int k = 0; k < K; k++){
//	log_likelihood(k,Y, X, A_block, rho, a, b, alpha, nu);
//	log_pi_ep(k, eta);
//	beta_postmean(k, Y, X, A_block, rho, a, b, alpha, nu);
	log_likelihood(k);
	log_pi_ep(k);
	beta_postmean(k);
  }
  return;
}

// useful just to try a handful for partitions
// corresponding to the example with square grid
// here n = 100
//void Partition::Initialize_Partition2(int id,  mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta){
void Partition::Initialize_Partition2(int id){

  nObs = 100;

  if(id == 0){
	K = 3;
	cluster_config = new int[K];
	cluster_config[0] = 25;
	cluster_config[1] = 25;
	cluster_config[2] = 50;
	clusters = new int*[K];
	clusters[0] = new int[25];
	clusters[1] = new int[25];
	clusters[2] = new int[50];

	log_like = new double[K];
	log_prior = new double[K];
	beta_hat = new double[nObs];
	cluster_assignment = new int[nObs];

	int counter0 = 0;
	int counter1 = 0;
	for(int i = 0; i < 5; i++){
	  for(int j = 0; j < 5; j++){
		clusters[0][counter0] = 10*i + j;
		clusters[1][counter1] = 10*(i+5) + j;
		counter0++;
		counter1++;
		cluster_assignment[10*i+j] = 0;
		cluster_assignment[10*(i+5) + j] = 1;
	  }
	}
	int counter2 = 0;
	for(int i = 0; i < 10; i++){
	  for(int j = 5; j < 10; j++){
		clusters[2][counter2] = 10*i + j;
		cluster_assignment[10*i + j] = 2;
		counter2++;
	  }
	}
  } else if(id == 1){
	// everything belongs to a single cluster
	//Initialize_Partition(100, Y,X, A_block, rho, a, b, alpha, nu, eta);
	Initialize_Partition(100);
  } else if(id == 2){
	// put original clusters 1 and 2 together
	nObs = 100;
		K = 2;
		cluster_config = new int[K];
		cluster_config[0] = 50;
		cluster_config[1] = 50;
		clusters = new int*[K];
		clusters[0] = new int[50];
		clusters[1] = new int[50];
		log_like = new double[K];
		log_prior = new double[K];
		beta_hat = new double[nObs];

		cluster_assignment = new int[nObs];

        int counter0 = 0;

		for(int i = 0; i < 5; i++){
			for(int j = 0; j < 5; j++){
				clusters[0][counter0] = 10*i + j;
				counter0++;
				clusters[0][counter0] = 10*(i+5) + j;
				counter0++;
				cluster_assignment[10*i+j] = 0;
				cluster_assignment[10*(i+5) + j] = 0;
			}
		}
		int counter1 = 0;
		for(int i = 0; i < 10; i++){
			for(int j = 5; j < 10; j++){
				clusters[1][counter1] = 10*i + j;
				cluster_assignment[10*i + j] = 1;
				counter1++;
			}
		}
	} else if(id == 3){
		// everything is in its own cluster
		nObs = 100;
		K = 100;
		cluster_config = new int[K];
		clusters = new int*[K];
		cluster_assignment = new int[nObs];
		log_like = new double[K];
		log_prior = new double[K];
		beta_hat = new double[nObs];

		for(int i = 0; i < nObs; i++){
			cluster_config[i] = 1;
			clusters[i] = new int[1];
			clusters[i][0] = i;
			cluster_assignment[i] = i;
		}
	} else if(id == 4){
	  nObs = 100;
	  K = 2;
	  cluster_config = new int[K];
	  cluster_config[0] = 60;
	  cluster_config[1] = 40;
	  clusters = new int*[K];
	  clusters[0] = new int[60];
	  clusters[1] = new int[40];
	  log_like = new double[K];
	  log_prior = new double[K];
	  beta_hat = new double[nObs];
	  cluster_assignment = new int[nObs];

      int counter0 = 0;
	  for(int i = 0; i < 10; i++){
	    for(int j = 0; j < 2; j++){
		  clusters[0][counter0] = 10*i + j;
		  counter0++;
		  clusters[0][counter0] = 10*i + (j+4);
		  counter0++;
		  clusters[0][counter0] = 10*i + (j+8);
		  counter0++;
		  cluster_assignment[10*i+j] = 0;
		  cluster_assignment[10*i + (j+4)] = 0;
		  cluster_assignment[10*i + (j+8)] = 0;
	    }
	  }
	  int counter1 = 0;
	  for(int i = 0; i < 10; i++){
		for(int j = 2; j < 4; j++){
		  clusters[1][counter1] = 10*i + j;
		  cluster_assignment[10*i + j] = 1;
		  counter1++;
		}
	    for(int j = 6; j < 8; j++){
		  clusters[1][counter1] = 10*i + j;
		  cluster_assignment[10*i + j] = 1;
		  counter1++;
		}
	  }
	}
	if(id != 1){
		get_pairwise();
		for(int k = 0; k < K; k++){
			//std::cout << "[Initialize_Partition]: k = " << k << std::endl;
			//tmp = log_likelihood(k, Y, X, A_block, rho, a, b, alpha, nu);
			//std::cout << log_like[k] << std::endl;
	//		log_likelihood(k, Y, X, A_block, rho, a, b, alpha, nu);
			log_likelihood(k);

		//	std::cout << "[Initialize_Partition]: computed log-likelihood:" << log_like[k] << std::endl;
	//		log_pi_ep(k, eta);
			log_pi_ep(k);
			//beta_postmean(k, Y, X, A_block, rho, a, b, alpha, nu);
			beta_postmean(k);
		}
	}
	return;
}



// Function to get the pairwise assignments
void Partition::get_pairwise(){
  pairwise_assignment = new int*[nObs];
  for(int i = 0; i < nObs; i++){
    pairwise_assignment[i] = new int[nObs];
  }
  for(int i = 0; i < nObs; i++){
	  for(int j = i; j < nObs; j++){
		  if(cluster_assignment[i] == cluster_assignment[j]){
			  pairwise_assignment[i][j] = 1;
			  pairwise_assignment[j][i] = 1;
		  } else{
			  pairwise_assignment[i][j] = 0;
			  pairwise_assignment[j][i] = 0;
		  }
	  }
  }
  return;

}
// Function to print out the partition
void Partition::Print_Partition(){
  std::cout << "Number of clusters K: " << K << std::endl;
  std::cout << "Size of clusters:";
  for(int k = 0; k < K; k++){
	std::cout << cluster_config[k] << " ";
  }
  std::cout << std::endl;
  std::cout << "Clusters:" << std::endl;
  for(int k = 0; k < K; k++){
	std::cout << "Cluster " << k  << " : ";
	for(int j = 0; j < cluster_config[k]; j++){
	  std::cout << clusters[k][j] << " ";
	}
	std::cout << std::endl;
  }
  double log_post = 0.0;
  for(int k = 0; k < K; k++){
		log_post += log_like[k] + log_prior[k];
  }
  std::cout << "Log-posterior : " << log_post << std::endl;
/*
	std::cout << "Cluster assignment: ";
	for(int i = 0; i < nObs; i++){
		std::cout << cluster_assignment[i] << " ";
	}
*/
	std::cout << std::endl;
	std::cout << "Log-likelihood:" ;
	for(int k = 0; k < K; k++){
		std::cout << log_like[k] << " ";
	}
	std::cout << std::endl;
	std::cout << "Log-prior:" ;
	for(int k = 0; k < K; k++){
		std::cout <<  log_prior[k] << " ";
	}
	std::cout << std::endl;
	std::cout << "Log-post:" ;
	for(int k = 0; k < K; k++){
		std::cout << log_like[k] + log_prior[k] << " ";
	}
	std::cout << std::endl;

	return;

}

void Partition::Print_Partition_ToFile(string file){
  ofstream myfile;
  myfile.open(file, std::ios_base::app);
  for(int i = 0; i < nObs-1; i++){
    for(int j = i+1; j < nObs; j++){
      myfile << pairwise_assignment[i][j] << ",";
    }
  }
  myfile << endl;
  myfile.close();
  return;
}

void Partition::Print_Means(){
	std::cout << "Posterior means of beta:" << std::endl;
	for(int i = 0; i < nObs; i++){
		std::cout << setprecision(6) << beta_hat[i] << " ";
	}
	std::cout << std::endl;

}

// Compute the log_likelihood
//void Partition::log_likelihood(int cluster_id, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu){
void Partition::log_likelihood(int cluster_id){
	int cluster_size = cluster_config[cluster_id];
	int T = Y.n_cols; // number of columns in Y
	mat I_T(T,T);
	I_T.eye(); // identity matrix
	double quad_form = 0.0;
	double numerator = 0.0;
	double denominator = 0.0;
	double Omega_log_det = 0;
	double Omega_log_det_sgn = 0;
	vec Y_vec = zeros<vec>(cluster_size*T);
	mat X_mat = zeros<mat>(cluster_size*T, cluster_size);
	mat Sigma_Y_k = zeros<mat>(cluster_size*T, cluster_size*T);
	mat Omega_Y_k = zeros<mat>(cluster_size*T, cluster_size*T);
	mat I_k(cluster_size, cluster_size);
	I_k.eye();
	if(cluster_size == 1){
		for(int t = 0; t < T; t++){
			Y_vec(t) = Y(clusters[cluster_id][0],t);
		}
		for(int t = 0; t < T; t++){
			X_mat(t,0) = X(clusters[cluster_id][0],t);
		}
		mat tXX = X_mat * X_mat.t();

		Sigma_Y_k = (a/(1.0 - rho) + b) * tXX + I_T;
		Omega_Y_k = inv_sympd(Sigma_Y_k);

		quad_form = as_scalar(Y_vec.t() * Omega_Y_k * Y_vec);

		Omega_log_det = 0;
		Omega_log_det_sgn = 0;
		log_det(Omega_log_det, Omega_log_det_sgn, Omega_Y_k);

		numerator = lgamma(alpha + ( (double) cluster_size * T/2)) + alpha * log(nu);
		denominator = lgamma(alpha) + (alpha + ( (double) cluster_size * T/2)) * log(nu + quad_form);
	} else {
		// i^th column contains all 0's except for entries i*T to (i+1)*T - 1
		for(int i = 0; i < cluster_size; i++){
			for(int j = 0; j < T; j++){
				X_mat(i*T + j,i) = X(clusters[cluster_id][i],j);
			}
		}
		for(int i = 0; i < cluster_size; i++){
			for(int j = 0; j < T; j++){
				Y_vec(i*T + j) = Y(clusters[cluster_id][i],j);
			}
		}

		mat A_block_k = Submatrix(A_block, cluster_size, cluster_size, clusters[cluster_id], clusters[cluster_id]);
		// get rowsums
		vec row_sums = zeros<vec>(cluster_size);
		for(int i = 0; i < cluster_size; i++){
			for(int j = 0; j < cluster_size; j++){
				row_sums(i) += A_block_k(i,j);
			}
		}
		mat D = diagmat(row_sums);
		mat A_star_k = D - A_block_k;
		mat Sigma_star_k = rho * A_star_k;
		for(int i = 0; i < cluster_size; i++){
			Sigma_star_k(i,i) += 1 - rho;
		}

		mat Omega_star_k = inv_sympd(Sigma_star_k);
		mat J = ones<mat>(cluster_size, 1);
		mat temp = X_mat * J; // this can be simplified greatly I feel.
		mat I_nT = zeros<mat>(cluster_size*T, cluster_size*T);
		I_nT.eye();
		mat Sigma_Y_k = I_nT + a * X_mat * Omega_star_k * X_mat.t() + b * temp * temp.t();

		// mat Omega_Y_k = inv_sympd(Sigma_Y_k);
		// quad_form = as_scalar(Y_vec.t() * Omega_Y_k * Y_vec);
		mat Omega_Y_k_Y = solve(Sigma_Y_k, Y_vec);
		quad_form = as_scalar(Y_vec.t() * Omega_Y_k_Y);

		Omega_log_det = 0;
		Omega_log_det_sgn = 0;
		// log_det(Omega_log_det, Omega_log_det_sgn, Omega_Y_k);
		log_det(Omega_log_det, Omega_log_det_sgn, Sigma_Y_k);

		numerator = lgamma(alpha +  ( (double) cluster_size * T/2)) + alpha * log(nu);
		denominator = lgamma(alpha) + (alpha + ( (double)cluster_size * T/2)) * log(nu + quad_form);
	}
	log_like[cluster_id] = 0.5 * Omega_log_det + numerator - denominator;
}

//void Partition::beta_postmean(int cluster_id, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu){
void Partition::beta_postmean(int cluster_id){

	int cluster_size = cluster_config[cluster_id];
	int T = Y.n_cols;
	if(cluster_size == 1){
		//std::cout << clusters[cluster_id][0] << std::endl;
		// this is a bit of a kluge
		// probably could've just done everything with row-vectors
		vec x(T);
		vec y(T);
		for(int t = 0; t < T; t++){
			x(t) = X(clusters[cluster_id][0],t);
			y(t) = Y(clusters[cluster_id][0],t);
		}
		double Sigma_beta_k = a/(1 - rho) + b;
		double Omega_beta_k = 1/Sigma_beta_k;
		//std::cout << "Trying to form tXX" << std::endl;
		double tXX = as_scalar(x.t() * x);
		double tXY = as_scalar(x.t() * y);
		beta_hat[clusters[cluster_id][0]] = 1/(tXX + Omega_beta_k) * tXY;
	} else {
		mat A_block_k = Submatrix(A_block, cluster_size, cluster_size, clusters[cluster_id], clusters[cluster_id]);
		// get rowsums
		vec row_sums = zeros<vec>(cluster_size);
		for(int i = 0; i < cluster_size; i++){
			for(int j = 0; j < cluster_size; j++){
				row_sums(i) += A_block_k(i,j);
			}
		}
		mat D = diagmat(row_sums);
		mat A_star_k = D - A_block_k;
		mat Sigma_star_k = rho * A_star_k;
		for(int i = 0; i < cluster_size; i++){
			Sigma_star_k(i,i) += 1 - rho;
		}
		mat Omega_star_k = inv_sympd(Sigma_star_k);
		mat ones_mat = ones<mat>(cluster_size, cluster_size);
		mat Sigma_beta_k = a * Omega_star_k + b * ones_mat;
		mat Omega_beta_k = inv_sympd(Sigma_beta_k);
		mat X_mat = zeros<mat>(cluster_size*T,cluster_size); // fill this in later

		for(int i = 0; i < cluster_size; i++){
		  for(int j = 0; j < T; j++){
			  X_mat(i*T + j,i) = X(clusters[cluster_id][i],j);
		  }
		}
		vec Y_vec = zeros<vec>(cluster_size*T);
		for(int i = 0; i < cluster_size; i++){
			for(int j = 0; j < T; j++){
				Y_vec(i*T + j) = Y(clusters[cluster_id][i],j);
			}
		}
		mat tXY = X_mat.t() * Y_vec;
		mat tXX = X_mat.t() * X_mat;
		mat temp = inv_sympd(tXX + Omega_beta_k);
		vec tmp_beta_hat = temp * tXY;
		for(int i = 0; i < cluster_size; i++){
			beta_hat[clusters[cluster_id][i]] = tmp_beta_hat(i);
		}
	}
}


//void Partition::log_pi_ep(int cluster_id, double eta){
void Partition::log_pi_ep(int cluster_id){
	log_prior[cluster_id] = log(eta) + lgamma(cluster_config[cluster_id]);
}


// Function will split the cluster into two parts
// new cluster1 will still be called cluster k
// new cluster2 will be called cluster K+1
//void Partition::Split(int split_k, int* new_cluster1, int* new_cluster2, int size1, int size2, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta){
void Partition::Split(int split_k, int* new_cluster1, int* new_cluster2, int size1, int size2){

  // create a temporary copy of the main attributes
  int orig_K = K;
  int* orig_cluster_config = new int[orig_K];
  int** orig_clusters = new int*[orig_K];
  int* orig_cluster_assignment = new int[nObs];
  double* orig_log_like = new double[orig_K];
  double* orig_log_prior = new double[orig_K];
  double* orig_beta_hat = new double[nObs];

  for(int k = 0; k < orig_K; k++){
	orig_cluster_config[k] = cluster_config[k];
	//orig_cluster_assignment[kk] = cluster_assignment[kk];
	orig_clusters[k] = new int[cluster_config[k]];
	for(int i = 0; i < cluster_config[k]; i++){
	  orig_clusters[k][i] = clusters[k][i];
	}
	orig_log_like[k] = log_like[k];
	orig_log_prior[k] = log_prior[k];
  }
  for(int i = 0; i < nObs; i++){
	orig_cluster_assignment[i] = cluster_assignment[i];
	orig_beta_hat[i] = beta_hat[i];
  }
  // clear up the memory from the original values and re-initialize
  // no need to delete pairwsie_assignment or cluster_assignment; the sizes are fixed
  K = orig_K+1; // we now have K+1 clusters
  //std::cout << "K = " << K << std::endl;
  delete[] cluster_config;
  //std::cout << "freed cluster_config" << std::endl;
  cluster_config = new int[K];

  for(int k = 0; k < K-1; k++){
	delete[] clusters[k];
  }
  delete[] clusters;
  //std::cout << "freed clusters" << std::endl;
  clusters = new int*[K];
  delete[] log_like;
  log_like = new double[K];
  delete[] log_prior;
  log_prior = new double[K];

  // how big are the new clusters?

  //std::cout << "size1 = " << size1 << " size2 = " << size2 << std::endl;
  // update cluster_config
  for(int k = 0; k < K; k++){
	if(k == split_k){
	  cluster_config[k] = size1;
	} else if(k == K-1){
	  cluster_config[k] = size2;
	} else{
	  cluster_config[k] = orig_cluster_config[k];
	}
  }
  //std::cout << "[Split]: Updated cluster_config" << std::endl;
  //std::cout << "K = " << K << std::endl;
  for(int k = 0; k < K; k++){
	clusters[k] = new int[cluster_config[k]];
	if(k == split_k){
	  for(int i = 0; i < size1; i++){
		clusters[k][i] = new_cluster1[i];
	  }

	} else if(k == K - 1){ // remember the 0-indexing...
	  for(int i = 0; i < size2;i++){
		clusters[k][i] = new_cluster2[i];
	  }
	} else{
	  for(int i = 0; i < cluster_config[k]; i++){
		clusters[k][i] = orig_clusters[k][i];
	  }
	}
  }

  //std::cout << "[Split]: Updated clusters" << std::endl;

  // now update new_cluster_assignments.
  for(int i = 0; i < nObs; i++){
	if(orig_cluster_assignment[i] != split_k){
	  cluster_assignment[i] = orig_cluster_assignment[i];
	}
  }
  for(int ii = 0; ii < size1; ii++){
	cluster_assignment[new_cluster1[ii]] = split_k;
  }
  for(int ii = 0; ii < size2; ii++){
	cluster_assignment[new_cluster2[ii]] = K - 1; // remember, cluster labels go from 0 to K-1.
  }
  //std::cout << "[Split]: updated cluster_assignemnt" << std::endl;
  // update the pairwise allocations
  get_pairwise();
  //update the log-likelihood and log-prior now
  //std::cout << "[Split]: Updated cluster_assignments and pairwise allocations" << std::endl;
  for(int k = 0; k < K; k++){
	if(k == split_k){ // need to re-compute
	  //std::cout << "[Split]: recomputing split_k's log_likelihood" << std::endl;
//	  log_likelihood(k, Y, X, A_block, rho, a, b, alpha, nu);
//	  log_pi_ep(split_k, eta);
//	  beta_postmean(k, Y, X, A_block, rho, a, b, alpha, nu);
	  log_likelihood(k);
	  log_pi_ep(split_k);
	  beta_postmean(k);
	} else if(k == K-1){
	  //std::cout << "[Split]: recomputing K-1's log_likelihood" << std::endl;
//	  log_likelihood(k, Y, X, A_block, rho, a, b, alpha, nu);
//	  log_pi_ep(k, eta);
//	  beta_postmean(k, Y, X, A_block, rho, a, b, alpha,nu);
	  log_likelihood(k);
	  log_pi_ep(k);
	  beta_postmean(k);
	} else{
	  log_like[k] = orig_log_like[k];
	  log_prior[k] = orig_log_prior[k];
	  for(int i = 0; i < cluster_config[k]; i++){
		beta_hat[clusters[k][i]] = orig_beta_hat[clusters[k][i]];
	  }
	}
  }

  // free up memory by deleting the local copies
  delete[] orig_cluster_config;
  for(int kk = 0; kk < K-1; kk++){ //original_clusters has length K and not K+1.
	delete[] orig_clusters[kk];
  }
  delete[] orig_clusters;
  delete[] orig_cluster_assignment;
  delete[] orig_log_like;
  delete[] orig_log_prior;
  delete[] orig_beta_hat;
  return;
}

//void Partition::Merge(int k_1, int k_2, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta){
void Partition::Merge(int k_1, int k_2){

  int k_max = max(k_1, k_2);
  int k_min = min(k_1, k_2);
  int new_cluster_size = cluster_config[k_min] + cluster_config[k_max];

    // make a pointer to the new merged cluster
  int* new_merged_cluster = new int[new_cluster_size];
  for(int i = 0; i < cluster_config[k_min]; i++){
	new_merged_cluster[i] = clusters[k_min][i];
  }
  for(int i = 0; i < cluster_config[k_max]; i++){
	new_merged_cluster[cluster_config[k_min] + i] = clusters[k_max][i];
  }
  // Update cluster_assignment
  // for original cluster k_max: this now becomes k_min
  // for clusters with original label greater than k_max, we need to decrement by 1
  int tmp_assignment = 0;
  for(int i = 0; i < nObs; i++){
	if(cluster_assignment[i] > k_max){ // we decrement cluster label by 1
	  tmp_assignment = cluster_assignment[i];
	  cluster_assignment[i] = tmp_assignment - 1;
	} else if(cluster_assignment[i] == k_max){
	  cluster_assignment[i] = k_min;
	}
  }
  // make a temporary copy of clusters and cluster_config
  int orig_K = K;
  int* orig_cluster_config = new int[orig_K];
  int** orig_clusters = new int*[orig_K];
  int* orig_cluster_assignment = new int[nObs];
  double* orig_log_like = new double[orig_K];
  double* orig_log_prior = new double[orig_K];
  double* orig_beta_hat = new double[nObs];

  for(int kk = 0; kk < orig_K; kk++){
	orig_cluster_config[kk] = cluster_config[kk];
	orig_clusters[kk] = new int[cluster_config[kk]];
	for(int i = 0; i < cluster_config[kk]; i++){
	  orig_clusters[kk][i] = clusters[kk][i];
	}
	orig_log_like[kk] = log_like[kk];
	orig_log_prior[kk] = log_prior[kk];
  }
  for(int i = 0; i < nObs; i++){
	orig_cluster_assignment[i] = cluster_assignment[i];
	orig_beta_hat[i] = beta_hat[i];
  }
  //std::cout << "[Merge] : Initialize orig_beta_hat" << std::endl;
  delete[] cluster_config;
  cluster_config = new int[K-1];

  for(int kk = 0; kk < orig_K; kk++){
	delete[] clusters[kk];
  }
  delete[] clusters;
  clusters = new int*[K-1];
  delete[] log_like;
  delete[] log_prior;
  log_like = new double[K-1];
  log_prior = new double[K-1];

  // looping over the OLD labels
  // remember the labels don't change until we get to k_max
  // this loop visits every cluster EXCEPT k_max
  for(int kk = 0; kk < orig_K; kk++){
	if(kk == k_min){
	  cluster_config[kk] = new_cluster_size;
	  clusters[kk] = new int[cluster_config[kk]];
	  for(int i = 0; i < cluster_config[kk]; i++){
		clusters[kk][i] = new_merged_cluster[i];
	  }
	  // we have updated clusters
	  // so we can now compute the log-likelihood and log-prior and posterior means of beta
	  log_likelihood(kk);
	  log_pi_ep(kk);
	  beta_postmean(kk);
	} else if(kk < k_max){
	  cluster_config[kk] = orig_cluster_config[kk];
	  clusters[kk] = new int[cluster_config[kk]];
	  for(int i = 0; i < cluster_config[kk]; i++){
		clusters[kk][i] = orig_clusters[kk][i];
		beta_hat[clusters[kk][i]] = orig_beta_hat[clusters[kk][i]];
	  }
	  log_like[kk] = orig_log_like[kk];
	  log_prior[kk] = orig_log_prior[kk];
	} else if(kk > k_max){
	  cluster_config[kk-1] = orig_cluster_config[kk];
	  clusters[kk-1] = new int[cluster_config[kk-1]];// ceci changed cluster_config[kk] into cluster_config[kk-1]
	  for(int i = 0; i < cluster_config[kk-1]; i++){ // ceci changed cluster_config[kk] into cluster_config[kk-1]
		clusters[kk-1][i] = orig_clusters[kk][i];
		beta_hat[clusters[kk-1][i]] = orig_beta_hat[clusters[kk-1][i]];
	  }
	  log_like[kk-1] = orig_log_like[kk];
	  log_prior[kk-1] = orig_log_prior[kk];
	}
  }
  get_pairwise();

  // update K
  K = orig_K - 1;
  // clean-up the memory
  delete[] orig_cluster_config;
  //std::cout << "freed orig_cluster_config" << std::endl;
  for(int kk = 0; kk < orig_K; kk++){ //original_clusters has length K and not K+1.
	delete[] orig_clusters[kk];
  }
  delete[] orig_clusters;
  //std::cout << "Freed orig_clusters" << std::endl;
  delete[] orig_cluster_assignment;
  delete[] orig_log_like;
  delete[] orig_log_prior;
  delete[] orig_beta_hat;
}

//void Partition::Split_and_Merge(int split_k, int* new_cluster1, int* new_cluster2, int size1, int size2, int k_star_1, int k_star_2, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta){
void Partition::Split_and_Merge(int split_k, int* new_cluster1, int* new_cluster2, int size1, int size2, int k_star_1, int k_star_2){

  //std::cout << "[Split_and_Merge]: split_k = " << split_k << " k_star_1 = " << k_star_1 << " k_star_2 = " << k_star_2 << std::endl;
  if(k_star_1 != k_star_2){
//	Split(split_k, new_cluster1, new_cluster2, size1, size2, Y, X, A_block, rho, a, b, alpha, nu, eta);
	Split(split_k, new_cluster1, new_cluster2, size1, size2);

	//std::cout << "[Split_and_Merge]: After the split, the cluster is:" << std::endl;
	//Print_Partition();
	if((split_k == k_star_1) & (split_k != k_star_2)){ // leave new_cluster1 alone and just attempt to merge new_cluster2 with k_star_2
	  // leave new_cluster1 by itself and merge k_star_2 with new_cluster2
//	  Merge(K-1, k_star_2, Y, X, A_block, rho, a, b, alpha, nu, eta); // remember new_cluster2's label is K-1
	  Merge(K-1, k_star_2); // remember new_cluster2's label is K-1

	} else if( (split_k != k_star_1) & (split_k == k_star_2)){
	  // leave new_clsuter2 by itself and merge k_star_1 with new_cluster1
	  // remember new_cluster1's label is still split_k;
//	  Merge(split_k, k_star_1, Y, X, A_block, rho, a, b, alpha, nu, eta);
	  Merge(split_k, k_star_1);

	} else if((split_k != k_star_1) & (split_k != k_star_2) & (k_star_2 > max(split_k, k_star_1))){
	  // We need to perform two merges
//	  Merge(split_k, k_star_1, Y, X, A_block, rho, a, b, alpha, nu, eta);
	  Merge(split_k, k_star_1);
	  // k_star_2's label is now decremented by 1
	  // new_cluster2's label is now K
//	  Merge(K, k_star_2 - 1, Y, X, A_block, rho, a, b, alpha, nu, eta);
	  Merge(K, k_star_2 - 1);

	} else if((split_k != k_star_1) & (split_k != k_star_2) & (k_star_2 < max(split_k, k_star_1))){
//	  Merge(split_k, k_star_1, Y, X, A_block, rho, a, b, alpha, nu, eta);
	  Merge(split_k, k_star_1);

//	  Merge(K, k_star_2, Y, X, A_block, rho, a, b, alpha, nu, eta);
	  Merge(K, k_star_2);

	}

  }
  return;
}

// Function to modify a partition if any of the clusters are disconnected
//void Partition::Modify(int cl_ind, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta){
void Partition::Modify(int cl_ind){
  // cl_ind is the index of the cluster that needs to be modified
  if(cluster_config[cl_ind] > 1)
  {
    int n = cluster_config[cl_ind];
    int *index = clusters[cl_ind];
    mat A_cluster = Submatrix(A_block, n, n, index, index);
    int *components = new int[n];
    int *count = new int;
    Connected_Components(A_cluster, n, components, count);
    if(*count > 1)
    {
      //I will use split iteratively, first split 0 from !=0
      //then among the remaining, split 1 from !=1, etc
      *count = *count - 1;
      int *new_components, *index1, *index2, *i_ind, n1, n2;
      for(int tmp = 0; tmp < *count; tmp++)
      {
        index1 = new int[n];
        index2 = new int[n];
        i_ind = new int[n];
        n1 = 0;
        n2 = 0;
        for(int i = 0; i < n; i++){
          if(components[i] == tmp){
            index1[n1] = index[i];
            n1++;
          } else {
            index2[n2] = index[i];
            i_ind[n2] = i;
            n2++;
          }
        }
//        Split(cl_ind, index1, index2, n1, n2, Y, X, A_block, rho, a, b, alpha, nu, eta);
        Split(cl_ind, index1, index2, n1, n2);
        if(tmp > 0)
          delete[] index;
        index = index2;
        n = n2;
        new_components = new int[n2];
        for(int j = 0; j < n2; j++)
          new_components[j] = components[i_ind[j]];
        delete[] components;
        components = new_components;
        cl_ind = K-1;
        delete[] index1;
      }
      delete[] index2;
    }
    delete[] components;
    delete count;
  }

}



void Partition::Find_Splits(int cluster_id, int **index1_ptr, int **index2_ptr, int &n1, int &n2){
  int n_cl = cluster_config[cluster_id];
  //double* beta_hat_cluster = new double[n_cl]; // we may get rid of this
  arma::mat A_block_cluster = Submatrix(A_block, n_cl, n_cl, clusters[cluster_id], clusters[cluster_id]);
  arma::mat beta_sim = zeros<mat>(n_cl, n_cl);
  arma::vec beta_hat_cluster(n_cl); // an armadillo vector holding the beta-hats
  for(int i = 0; i < n_cl; i++){
  	beta_hat_cluster(i) = beta_hat[clusters[cluster_id][i]];
  }
  double beta_hat_var = arma::var(beta_hat_cluster); // variance of the beta_hats within the cluster
  // populate the beta_similarity matrix
  double error = 0.0;
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,beta_hat_var);
  for(int i = 0; i < n_cl - 1; i++){
  	for(int j = i; j < n_cl; j++){
      // error = distribution(generator);
  	  beta_sim(i,j) = exp(-1 * (beta_hat_cluster(i) - beta_hat_cluster(j) + error) * (beta_hat_cluster(i) - beta_hat_cluster(j) + error)/(2 * beta_hat_var));
  	  beta_sim(j,i) = exp(-1 * (beta_hat_cluster(i) - beta_hat_cluster(j) + error) * (beta_hat_cluster(i) - beta_hat_cluster(j) + error)/(2 * beta_hat_var));
  	}
  }
  arma::mat diag_ncl(n_cl,n_cl,fill::eye);
  arma::mat W_beta_cl =  diag_ncl + beta_sim % A_block_cluster;
  arma::mat Dinv_sqrt = arma::diagmat(1/sqrt(arma::sum(W_beta_cl, 1)));
  arma::mat L = diag_ncl - Dinv_sqrt * W_beta_cl * Dinv_sqrt;
  arma::vec eigval; // the smallest eigenvalues are the first two
  arma::mat eigvec;
  eig_sym(eigval, eigvec, L);
  mat U = eigvec.cols(0,1);
  U = arma::diagmat(1/sqrt(arma::sum(arma::square(U), 1))) * U;
  arma::mat means;
  // kmeans(means, U.t(), 2, random_subset, 10, false);
  bool status = arma::kmeans(means, U.t(), 2, random_subset, 10, false);
  if(status == false)
    cout << "clustering failed" << endl;
  int * membership = which_is_nearest(means, U.t());
  (*index1_ptr) = new int[n_cl];
  (*index2_ptr) = new int[n_cl];
  n1 = 0;
  n2 = 0;
  for(int i = 0; i < n_cl; i++){
    if(membership[i] == 0){
      (*index1_ptr)[n1] = clusters[cluster_id][i];
      (n1)++;
    }
    else {
      (*index2_ptr)[n2] = clusters[cluster_id][i];
      (n2)++;
     }
   }
   delete[] membership;
   Split(cluster_id, *index1_ptr, *index2_ptr, n1, n2);
   // delete[] index1;
   // delete[] index2;
}

void Partition::K_Splits(int k){
  arma::vec beta_hat_vec(nObs); // an armadillo vector holding the beta-hats
  for(int i = 0; i < nObs; i++){
    beta_hat_vec(i) = beta_hat[i];
  }
  // cout << beta_hat_vec << endl;
  double beta_hat_var = arma::var(beta_hat_vec); // variance of the beta_hats within the cluster
  
  // double error = 0.0;
  // std::default_random_engine generator;
  // std::normal_distribution<double> distribution(0.0,beta_hat_var);
  arma::mat beta_sim = zeros<mat>(nObs, nObs);
  for(int i = 0; i < nObs - 1; i++){
    for(int j = i; j < nObs; j++){
      // error = distribution(generator);
      // beta_sim(i,j) = exp(-1 * (beta_hat_vec(i) - beta_hat_vec(j) + error) * (beta_hat_vec(i) - beta_hat_vec(j) + error)/(2 * beta_hat_var));
      // beta_sim(j,i) = exp(-1 * (beta_hat_vec(i) - beta_hat_vec(j) + error) * (beta_hat_vec(i) - beta_hat_vec(j) + error)/(2 * beta_hat_var));
      beta_sim(i,j) = exp(-1 * (beta_hat_vec(i) - beta_hat_vec(j)) * (beta_hat_vec(i) - beta_hat_vec(j))/(2 * beta_hat_var));
      beta_sim(j,i) = exp(-1 * (beta_hat_vec(i) - beta_hat_vec(j)) * (beta_hat_vec(i) - beta_hat_vec(j))/(2 * beta_hat_var));
    }
  }
  arma::mat diag_n(nObs,nObs,fill::eye);
  arma::mat W_beta_cl =  diag_n + beta_sim % A_block;
  arma::mat Dinv_sqrt = arma::diagmat(1/sqrt(arma::sum(W_beta_cl, 1)));
  arma::mat L = diag_n - Dinv_sqrt * W_beta_cl * Dinv_sqrt;
  arma::vec eigval; // the smallest eigenvalues are the first two
  arma::mat eigvec;
  eig_sym(eigval, eigvec, L);
  mat U = eigvec.cols(0,k-1);
  U = arma::diagmat(1/sqrt(arma::sum(arma::square(U), 1))) * U;
  arma::mat means;
  // kmeans(means, U.t(), 2, random_subset, 10, false);
  bool status = arma::kmeans(means, U.t(), k, random_subset, 10, false);
  if(status == false)
    cout << "clustering failed" << endl;
  int * membership = which_is_nearest_k(means, U.t());

  int** indices = new int*[k];
  int* ns = new int[k];
  for(int j = 0; j<k; j++){
    indices[j] = new int[nObs];
    ns[j] = 0;
  }
  
  for(int i = 0; i < nObs; i++){
    for(int j = 0; j < k; j++){
      if(membership[i] == j){
void Partition::K_Splits(int k){
  arma::vec beta_hat_vec(nObs); // an armadillo vector holding the beta-hats
  for(int i = 0; i < nObs; i++){
    beta_hat_vec(i) = beta_hat[i];
  }
  // cout << beta_hat_vec << endl;
  double beta_hat_var = arma::var(beta_hat_vec); // variance of the beta_hats within the cluster
  
  // double error = 0.0;
  // std::default_random_engine generator;
  // std::normal_distribution<double> distribution(0.0,beta_hat_var);
  arma::mat beta_sim = zeros<mat>(nObs, nObs);
  for(int i = 0; i < nObs - 1; i++){
    for(int j = i; j < nObs; j++){
      // error = distribution(generator);
      // beta_sim(i,j) = exp(-1 * (beta_hat_vec(i) - beta_hat_vec(j) + error) * (beta_hat_vec(i) - beta_hat_vec(j) + error)/(2 * beta_hat_var));
      // beta_sim(j,i) = exp(-1 * (beta_hat_vec(i) - beta_hat_vec(j) + error) * (beta_hat_vec(i) - beta_hat_vec(j) + error)/(2 * beta_hat_var));
      beta_sim(i,j) = exp(-1 * (beta_hat_vec(i) - beta_hat_vec(j)) * (beta_hat_vec(i) - beta_hat_vec(j))/(2 * beta_hat_var));
      beta_sim(j,i) = exp(-1 * (beta_hat_vec(i) - beta_hat_vec(j)) * (beta_hat_vec(i) - beta_hat_vec(j))/(2 * beta_hat_var));
    }
  }
  arma::mat diag_n(nObs,nObs,fill::eye);
  arma::mat W_beta_cl =  diag_n + beta_sim % A_block;
  arma::mat Dinv_sqrt = arma::diagmat(1/sqrt(arma::sum(W_beta_cl, 1)));
  arma::mat L = diag_n - Dinv_sqrt * W_beta_cl * Dinv_sqrt;
  arma::vec eigval; // the smallest eigenvalues are the first two
  arma::mat eigvec;
  eig_sym(eigval, eigvec, L);
  mat U = eigvec.cols(0,k-1);
  U = arma::diagmat(1/sqrt(arma::sum(arma::square(U), 1))) * U;
  arma::mat means;
  // kmeans(means, U.t(), 2, random_subset, 10, false);
  int** indices = new int*[k];
  int* ns = new int[k];
  int * membership;
  bool FLAG = true;
  while(FLAG){
    bool status = arma::kmeans(means, U.t(), k, random_subset, 10, false);
    if(status == false){
      cout << "clustering failed" << endl;
      // go directly to next iteration
    } else {
      membership = which_is_nearest_k(means, U.t());

      for(int j = 0; j<k; j++){
        indices[j] = new int[nObs];
        ns[j] = 0;
      }

      for(int i = 0; i < nObs; i++){
        for(int j = 0; j < k; j++){
          if(membership[i] == j){
            indices[j][ns[j]] = i;
            (ns[j])++;
          }
        }
      }

      FLAG = false; // if not proven wrong I will end the loop
      for(int j = 0; j < k; j++){
        mat A_cluster = Submatrix(A_block, ns[j], ns[j], indices[j], indices[j]);
        int *components = new int[ ns[j] ];
        int *count = new int;
        Connected_Components(A_cluster, ns[j],components,count);
        if((*count)!=1){
          FLAG = true; // I need to iterate again
          cout << "cluster " << j <<": disconnected components, I will iterate again." << endl;
        }
      }
      if(FLAG == true){
        delete[] membership;
      }
    }
  }

  // I probably should not use split and create a brand new function
  // Split(cluster_id, *index1_ptr, *index2_ptr, n1, n2);
  int orig_K = K;
  K = k;
  delete[] cluster_config; cluster_config = NULL;
  for(int i = 0; i < orig_K; i++){
    delete[] clusters[i]; clusters[i] = NULL;
  }
  delete[] clusters; clusters = NULL;
  delete[] log_like; log_like = NULL;
  delete[] log_prior; log_prior = NULL;
  for(int i = 0; i < nObs; i++){
    delete[] pairwise_assignment[i]; pairwise_assignment[i] = NULL;
  }
  delete[] pairwise_assignment; pairwise_assignment = NULL;

  cluster_config = new int[K];
  clusters = new int*[K];
  for(int i = 0; i < K; i++){
    cluster_config[i] = ns[i];
    clusters[i] = new int[ns[i]];
    for(int j = 0; j < ns[i]; j++){
      clusters[i][j] = indices[i][j];
    }
  }
  for(int i = 0; i < nObs; i++){
    cluster_assignment[i] = membership[i];
  }
  get_pairwise();
  log_like = new double[K];
  log_prior = new double[K];
  for(int i = 0; i < K; i++){
    log_likelihood(i);
    log_pi_ep(i);
    beta_postmean(i);
  }

  delete[] membership;
  delete[] ns;
  for(int j = 0; j<k; j++){
    delete[] indices[j];
  }
  delete[] indices;

  return;
}


/*
Partition::Partition(LPPartition InitialPartition){
	nObs = InitialPartition->nObs;
	K = InitialPartition->K;
	cluster_config = InitialPartition->cluster_config;
	cluster_membership = InitialPartition->cluster_membership;
//	pairwise_assignment = InitialPartition->pairwise_assignment;
	return;
}
*/



