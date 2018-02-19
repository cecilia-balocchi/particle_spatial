/*
 * partition_functions.cpp
 *
 *  Created on: Dec 29, 2017
 *      Author: Sameer
 */



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <vector>
#include <algorithm>
#include "partition.h"
#include "various_functions.h"
using namespace std;


// None of the methods here actually need any of this data
// instead, they call methods which use the data and we already have
// the appropriate extern calls in the corresponding headers

//extern arma::mat Y;
//extern arma::mat X;
extern arma::mat A_block; // we actually need this one!
// Set the hyper-parameters
//extern double rho;
//extern double a;
//extern double b;
//extern double nu;
//extern double alpha;
//extern double eta;


// Compare two partitions and see if they are equal
int Partition_Equal(Partition *partition1, Partition *partition2){
  int flag = 1;
    // simpler to just compare pairwise allocation
  for(int i = 0; i < partition1->nObs; i++){
	for(int j = 0; j < partition1->nObs; j++){
	  if(partition1->pairwise_assignment[i][j] != partition2->pairwise_assignment[i][j]){
		flag = 0;
		break;
	  }
	}
	if(flag == 0) break;
  }
  return flag;
}

double beta_bar(Partition *partition, int k){
  arma::vec beta_cl(partition->cluster_config[k]);
  for(int i = 0; i < partition->cluster_config[k]; i++){
    beta_cl(i) = partition->beta_hat[ partition->clusters[k][i] ];
  }
  return arma::mean(beta_cl);
}

// function to compute entropy
// when we replace the (current_l)^th particle with the candidate_clusters
//
double Entropy(unsigned current_l, Partition* candidate_particle, std::vector<LPPartition> particle_set, std::vector<double> w){
  unsigned L = particle_set.size();
  // need to loop over to extract the unique partitions
  std::vector<LPPartition> unik_particles;
  std::vector<double> p_star;

  unik_particles.push_back(candidate_particle);
  p_star.push_back(w[current_l]);

  // in a sense, we are replacing particle_set[current_l] with candidate_particle
  // by adding it to the unik_particles vector


  int num_unik_particles = 1;
  int counter = 0;
  for(unsigned l = 0; l < L; l++){ // loop over current particle set
	counter = 0;
	if(l != current_l){
	//std::cout << "[getUnik]: l = " << l << std::endl;
	  for(unsigned ul = 0; ul < num_unik_particles; ul++){
	    if(Partition_Equal(particle_set[l], unik_particles[ul]) == 1){
	    // l^th partition is equal to the ul^th unique partition
	    //std::cout << "particle " << l << " is equal to unik particle " << ul << std::endl;
	      p_star[ul] += w[l]; // update p_star
	      break;
        } else {
	      counter++;
        }
	  }

	  //std::cout << "[getUnik]: counter = " << counter << std::endl;
	  if(counter == num_unik_particles){
	    //std::cout << "we found a new unique particle!" << std::endl;
	    //particle_set[l]->Print_Partition();
	    // we have found a new unique particle
	    unik_particles.push_back(particle_set[l]);
	    p_star.push_back(w[l]);
	    num_unik_particles++;
      }
    }
  }
  double entropy = 0.0;
  //std::cout << "p_star = " ;
  for(unsigned ul = 0; ul < num_unik_particles; ul++){
	  //std::cout << p_star[ul] << " " ;
	  entropy += p_star[ul] * log(p_star[ul]);
  }
  //std::cout << std::endl;
  return -1.0 * entropy;

}

// a silly function to add up log-likelihoods and log-priors
double total_log_post(LPPartition partition){
	double log_post = 0.0;
	for(int k = 0; k < partition->K; k++){
		log_post += partition->log_like[k] + partition->log_prior[k];
	}
	return log_post;
}

double total_log_like(LPPartition partition){

  double log_like = 0.0;
  for(int k = 0; k < partition->K; k++){
	log_like += partition->log_like[k];
  }
  return log_like;
}
double total_log_prior(LPPartition partition){
  double log_prior = 0.0;
  for(int k = 0; k < partition->K; k++){
	log_prior += partition->log_prior[k];
  }
  return log_prior;
}

double Binder_Loss(LPPartition partition1, LPPartition partition2){
  int n = partition1->nObs;
  int K1 = partition1->K;
  int K2 = partition2->K;
  // need to get the matrix of counts n_ij which counts the number of indices that belong to cluster i in partition1 and cluster j in partition2
  mat counts(K1, K2, fill::zeros);

  for(int i = 0; i < n; i++){
	counts(partition1->cluster_assignment[i], partition2->cluster_assignment[i])++;
  }
  //counts.print();

  double loss = 0.0;
  // Binder loss is quadratic in the elements of the matrix of counts
  // Need sum of squares of row sums + sum of squared columns sums and then subtract sum of squares of elements of counts
  // row sums of counts: just the configuration of partition1
  // column sums of counts: just the
  for(int k = 0; k < K1; k++){
	loss += 0.5 * ( (double) partition1->cluster_config[k])*( (double) partition1->cluster_config[k]);
  }
  for(int k = 0; k < K2; k++){
	loss += 0.5 * ( (double) partition2->cluster_config[k]) * ( (double) partition2->cluster_config[k]);
  }
  loss -= accu(pow(counts, 2));

  // Wade and Ghahramani consider a scaled Binder loss and we will as well
  // Property 4 in their paper asserts that this binder loss will always be between 0 and 1 - 1/n
  loss *= 2.0/( (double) n* (double) n);
  return loss;

}

double  VI_Loss(LPPartition partition1, LPPartition partition2){
  int n = partition1->nObs;
  int K1 = partition1->K;
  int K2 = partition2->K;
  // need to get the matrix of counts n_ij which counts the number of indices that belong to cluster i in partition1 and cluster j in partition2
  mat counts(K1, K2, fill::zeros);

  for(int i = 0; i < n; i++){
	counts(partition1->cluster_assignment[i], partition2->cluster_assignment[i])++;
  }
  double loss = 0.0;
  for(int k = 0; k < K1; k++){
	loss += ( (double) partition1->cluster_config[k])/( (double) n) * log( ((double) partition1->cluster_config[k])/( (double) n));
  }
  for(int k = 0; k < K2; k++){
	loss += ( (double) partition2->cluster_config[k])/( (double) n) * log(((double) partition2->cluster_config[k])/((double) n));
  }
  for(int k1 = 0; k1 < K1; k1 ++){
	for(int k2 = 0; k2 < K2; k2++){
	  if(counts(k1, k2) != 0){ // 0 * log(0) = 0 so if any counts
		  loss -= 2.0 * ( (double) counts(k1, k2)) / ((double) n) * log( ((double) counts(k1, k2))/ ( (double) n));
	  }
	}
  }
  return loss;
}
double Binder_DPP(unsigned current_l, Partition* candidate_particle, std::vector<LPPartition> particle_set){

  // The first step is to get all of the unique particles when we replace the current_l^th particle with candidate particle
  unsigned L = particle_set.size();
  // need to loop over to extract the unique partitions
  std::vector<LPPartition> unik_particles;
  std::vector<double> p_star;

  unik_particles.push_back(candidate_particle);
  int num_unik_particles = 1;
  int counter = 0;
  for(unsigned l = 0; l < L; l++){ // loop over current particle set
	counter = 0;
	if(l != current_l){
	  for(unsigned ul = 0; ul < num_unik_particles; ul++){
	    if(Partition_Equal(particle_set[l], unik_particles[ul]) == 1){
	    // l^th partition is equal to the ul^th unique partition
	    //std::cout << "particle " << l << " is equal to unik particle " << ul << std::endl;
	    //p_star[ul] += w[l]; // update p_star
	      break;
        } else {
	      counter++;
        }
	  }
	  if(counter == num_unik_particles){
	    unik_particles.push_back(particle_set[l]);
	    num_unik_particles++;
      }
    }
  }

  mat kernel(num_unik_particles, num_unik_particles, fill::zeros);
  double dist = 0.0;
  double kernel_log_det = 0.0;
  double kernel_log_det_sgn = 0.0;
  for(int l = 0; l < num_unik_particles; l++){
	for(int ll = 0; ll < num_unik_particles; ll++){
	  dist = Binder_Loss(unik_particles[l], unik_particles[ll]);
	  kernel(l,ll) = exp(-0.5 * dist * dist);
	}
  }
  log_det(kernel_log_det, kernel_log_det_sgn, kernel);

/*
  double dist = 0.0;
  if(num_unik_particles == 1){
	dist = 0.0; // everything is equal to everything else
  } else {
  // now loop over the unique particles and compute pairwise distances
    for(unsigned l = 0; l < num_unik_particles - 1; l++){
	  for(unsigned ll = l+1; ll < num_unik_particles; ll++){
	    dist += Binder_Loss(unik_particles[l], unik_particles[ll]);
	  }
    }
  }
*/
  return kernel_log_det;
}
double VI_DPP(unsigned current_l, Partition* candidate_particle, std::vector<LPPartition> particle_set){
  // The first step is to get all of the unique particles when we replace the current_l^th particle with candidate particle
  unsigned L = particle_set.size();
  // need to loop over to extract the unique partitions
  std::vector<LPPartition> unik_particles;
  std::vector<int> particle_counts;


  unik_particles.push_back(candidate_particle);
  particle_counts.push_back(1);
  int num_unik_particles = 1;
  int counter = 0;
  for(unsigned l = 0; l < L; l++){ // loop over current particle set
	counter = 0;
	if(l != current_l){
	  for(unsigned ul = 0; ul < num_unik_particles; ul++){
	    if(Partition_Equal(particle_set[l], unik_particles[ul]) == 1){
	    // l^th partition is equal to the ul^th unique partition
	    //std::cout << "particle " << l << " is equal to unik particle " << ul << std::endl;
	    //p_star[ul] += w[l]; // update p_star
	      particle_counts[ul]++;
	      break;
        } else {
	      counter++;
        }
	  }
	  if(counter == num_unik_particles){
	    unik_particles.push_back(particle_set[l]);
	    particle_counts.push_back(1);
	    num_unik_particles++;
      }
    }
  }
  if(num_unik_particles == 1){
	return(-pow(10.0, 3.0));
  } else{

	mat kernel(num_unik_particles, num_unik_particles, fill::zeros);
	double dist = 0.0;
	double kernel_log_det = 0.0;
	double kernel_log_det_sgn = 0.0;
	for(int l = 0; l < num_unik_particles; l++){
	  for(int ll = 0; ll < num_unik_particles; ll++){
		dist = VI_Loss(unik_particles[l], unik_particles[ll]);
		//kernel(l,ll) = exp(-1.0 * dist); // large distance means the particles are not similar, hence entry in kernel matrix should be small
		kernel(l,ll) = exp(-1.0 * dist * particle_counts[l] * particle_counts[ll]);
	  }
	}
	log_det(kernel_log_det, kernel_log_det_sgn, kernel);
	return(kernel_log_det);
  }
/*
  double dist = 0.0;
  if(num_unik_particles == 1){
	dist = 0.0; // everything is equal to everything else
  } else {
  // now loop over the unique particles and compute pairwise distances
    for(unsigned l = 0; l < num_unik_particles - 1; l++){
	  for(unsigned ll = l+1; ll < num_unik_particles; ll++){
	    dist += VI_Loss(unik_particles[l], unik_particles[ll]);
	  }
    }
  }
  return dist;
*/
}


// get Local candidate
//
//void get_island(LPPartition candidate, int current_l, vector<LPPartition> particle_set, vector<double> w, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta){
void get_island(LPPartition candidate, int current_l, std::vector<LPPartition> particle_set, std::vector<double> w, double lambda, double xi){

  LPPartition max_candidate = new Partition(particle_set[current_l]); // holds the running "best local candidate"
  double max_objective = 0.0; // holds the objective value of the "best local candidate"
  LPPartition tmp_candidate = new Partition(particle_set[current_l]); // the running candidate
  double tmp_objective = 0.0; // objective value of the running candidate

  max_objective = w[current_l]*total_log_post(max_candidate) + lambda * Entropy(current_l, max_candidate, particle_set, w) + xi * VI_DPP(current_l, max_candidate, particle_set);
//	std::cout << "[get_local]: max obj = " << setprecision(8) << max_objective << std::endl;
  int size1 = 0;
  int size2 = 0;
  int* new_cluster1 = new int[1];
  int* new_cluster2 = new int[1];
  int split_k = 0;


//	std::cout << "starting to loop" << std::endl;
  for(int i = 0; i < particle_set[current_l]->nObs; i++){
     split_k = particle_set[current_l]->cluster_assignment[i];
     delete tmp_candidate; // delete the old value of tmp_candidate .. i can either use Copy_Partition or do a delete[] and new call

     try{
       tmp_candidate = new Partition(particle_set[current_l]);
     }
     catch(const std::bad_alloc& e){
       cout << "######## EXCEPTION 3 ########"  << e.what() << endl;
     }

     if(particle_set[current_l]->cluster_config[split_k] > 1){ // only attempt to split the cluster if there are at least 2 elements
       size1 = tmp_candidate->cluster_config[split_k] - 1;
       size2 = 1;
       delete[] new_cluster1;
       delete[] new_cluster2;

       try{
         new_cluster1 = new int[size1];
         new_cluster2 = new int[size2];
       }
       catch(const std::bad_alloc& e){
         cout << "######## EXCEPTION 12b ########"  << e.what() << endl;
       }

       new_cluster2[0] = i;
       int counter = 0;
       for(int ii = 0; ii < tmp_candidate->cluster_config[split_k]; ii++){
         if(tmp_candidate->clusters[split_k][ii] != i){
           new_cluster1[counter] = tmp_candidate->clusters[split_k][ii];
           counter++;
         }
       }
       // now actually perform the split
       //    tmp_candidate->Split(split_k, new_cluster1, new_cluster2, size1, size2, Y, X, A_block, rho, a, b, alpha, nu, eta);
       tmp_candidate->Split(split_k, new_cluster1, new_cluster2, size1, size2);
       //tmp_candidate->Print_Partition();

       tmp_objective = w[current_l]*total_log_post(tmp_candidate) + lambda * Entropy(current_l, tmp_candidate, particle_set, w) + xi * VI_DPP(current_l, tmp_candidate, particle_set);
       std::cout << "[get_island]: i = " << i << " max_obj = " << setprecision(8) << max_objective << " tmp_obj = " << setprecision(8) << tmp_objective << std::endl;
       std::cout << "			 log_like = " << setprecision(8) << total_log_like(tmp_candidate) << "  log_prior = " << setprecision(8) << total_log_prior(tmp_candidate) << std::endl;
       std::cout << "			 Entropy = " << setprecision(8) << lambda * Entropy(current_l, tmp_candidate, particle_set, w) << setprecision(8) << "  VI_DPP = " << setprecision(8) << xi * VI_DPP(current_l, tmp_candidate, particle_set) << std::endl;

       //    std::cout << "[get_local]: i = " << i << " max_obj = " << setprecision(8) << max_objective << " tmp_obj = " << setprecision(8) << tmp_objective << std::endl;
       if(tmp_objective > max_objective){
         //max_candidate->Copy_Partition(tmp_candidate);
         delete max_candidate;
         try{
           max_candidate = new Partition(tmp_candidate);
         }
         catch(const std::bad_alloc& e){
           cout << "######## EXCEPTION 4 ########"  << e.what() << endl;
         }
         max_objective = tmp_objective;
         //    std::cout << "max_obj is now: " << max_objective << std::endl;
         //max_candidate->Print_Partition();
       }
     }
   }
/*
  for(int i = 0; i < particle_set[current_l]->nObs; i++){
	//std::cout << "[get_local]: Starting i = " << i << std::endl;
	split_k = particle_set[current_l]->cluster_assignment[i];
	//std::cout << "[get_local]: Re-setting tmp_candidate" << std::endl;

	delete tmp_candidate; // delete the old value of tmp_candidate .. i can either use Copy_Partition or do a delete[] and new call
	tmp_candidate = new Partition(particle_set[current_l]);
	//tmp_candidate->Copy_Partition(particle_set[current_l]); // re-set tmp_candidate to the partition we're trying to update
	//std::cout << "[getLocal]: finished making copy" << std::endl;
	//std::cout << "[get_local]: i = " << i << " is in cluster " << split_k << std::endl;
	if(particle_set[current_l]->cluster_config[split_k] > 1){ // only attempt to split the cluster if there are at least 2 elements
	  //std::cout << "[getLocal]: will try to split now!" << std::endl;
	  size1 = tmp_candidate->cluster_config[split_k] - 1;
	  size2 = 1;
	  // free up the old versions of new_cluster1 and new_cluster2
	  //std::cout << "[getLocal]: about to delete something" << std::endl;
	  delete[] new_cluster1;
	  delete[] new_cluster2;
	  new_cluster1 = new int[size1];
	  new_cluster2 = new int[size2];
	  //std::cout << "[getLocal]: trying to re-allocated new_cluster1 and new_cluster2" << std::endl;

	  new_cluster2[0] = i;
	  int counter = 0;
	  for(int ii = 0; ii < tmp_candidate->cluster_config[split_k]; ii++){
		if(tmp_candidate->clusters[split_k][ii] != i){
		  new_cluster1[counter] = tmp_candidate->clusters[split_k][ii];
		  counter++;
		}
	  }
	  // now actually perform the split
	  tmp_candidate->Split(split_k, new_cluster1, new_cluster2, size1, size2);
	  tmp_objective = w[current_l]*total_log_post(tmp_candidate) + lambda * Entropy(current_l, tmp_candidate, particle_set, w);
	  //std::cout << "[get_local]: i = " << i << " max_obj = " << setprecision(8) << max_objective << " tmp_obj = " << setprecision(8) << tmp_objective << std::endl;
	  if(tmp_objective > max_objective){
		//max_candidate->Copy_Partition(tmp_candidate);
		delete max_candidate;
		max_candidate = new Partition(tmp_candidate);
		max_objective = tmp_objective;
	  }
	}
  }
*/

  candidate->Copy_Partition(max_candidate);
  delete max_candidate;
  delete tmp_candidate;
  delete[] new_cluster1;
  delete[] new_cluster2;
}
//void get_border(LPPartition candidate, int current_l, vector<LPPartition> particle_set, vector<double> w, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta){
void get_border(LPPartition candidate, int current_l, std::vector<LPPartition> particle_set, std::vector<double> w, double lambda, double xi){

  LPPartition max_candidate = new Partition(particle_set[current_l]); // holds the running "best local candidate"
  double max_objective = 0.0; // holds the objective value of the "best local candidate"

  LPPartition tmp_candidate = new Partition(particle_set[current_l]); // the running candidate
  double tmp_objective = 0.0; // objective value of the running candidate

  max_objective = w[current_l]*total_log_post(max_candidate) + lambda * Entropy(current_l, max_candidate, particle_set, w) + xi * VI_DPP(current_l, max_candidate, particle_set);
  //std::cout << "[get_border]: max obj = " << setprecision(8) << max_objective << std::endl;
  int size1 = 0;
  int size2 = 0;
  int* new_cluster1 = new int[1];
  int* new_cluster2 = new int[1];
  int split_k = 0;
  int k_star_1 = 0;
  int k_star_2 = 0;

  vector<int> neighboring_clusters; // holds the labels
  int neighbor_counter = 0;

  for(int i = 0; i < particle_set[current_l]->nObs; i++){
    split_k = particle_set[current_l]->cluster_assignment[i];
    neighbor_counter = 0;
    neighboring_clusters.clear();
    if(particle_set[current_l]->cluster_config[split_k] > 1){ // only attempt if there are at least 2 clusters. Otherwise this is just a Merge move
      for(int k = 0; k < particle_set[current_l]->K;k++){
        if(k != split_k){ // loop over the elements of cluster k to see if any are adjacent to i
          for(int j = 0; j < particle_set[current_l]->cluster_config[k]; j++){
            if(A_block(i, particle_set[current_l]->clusters[k][j]) == 1){ // j^th blockgroup of cluster k is adjacent to block group i
              neighboring_clusters.push_back(k); // add k to the vector of adjacent
              neighbor_counter++;
              break; // once we know that i is adjacent to something in cluster k we can stop
            }
          }
        }
      }
    }
    if(neighbor_counter > 0){
      // we are splitting i away from its cluster
      // let us define all of the necessary components for that split
      size1 = particle_set[current_l]->cluster_config[split_k] - 1;
      size2 = 1;

      try{
        new_cluster1 = new int[size1];
        new_cluster2 = new int[size2];
      }
      catch(const std::bad_alloc& e){
        cout << "######## EXCEPTION 34b ########"  << e.what() << endl;
      }

      new_cluster2[0] = i;
      int counter = 0;
      for(int ii = 0; ii < particle_set[current_l]->cluster_config[split_k]; ii++){
        if(particle_set[current_l]->clusters[split_k][ii] != i){
          new_cluster1[counter] = particle_set[current_l]->clusters[split_k][ii];
          counter++;
        }
      }
      // new_cluster1 contains all of the things in cluster split_k EXCEPT for i
      // it will be "merged" with itself (i.e. k_star_1 = split_k)
      k_star_1 = split_k;
      for(int nc = 0; nc < neighbor_counter; nc++){
      k_star_2 = neighboring_clusters[nc]; // this is the new cluster that we are adding i to!
      std::cout << "[get_border]: Attempting to move i = " << i << " from cluster " << split_k << " to cluster " << k_star_2 << std::endl;
      delete tmp_candidate;

      try{
        tmp_candidate = new Partition(particle_set[current_l]);
      }
      catch(const std::bad_alloc& e){
        cout << "######## EXCEPTION 5 ########"  << e.what() << endl;
      }

      //tmp_candidate->Split_and_Merge(split_k, new_cluster1, new_cluster2, size1, size2, k_star_1, k_star_2, Y, X, A_block, rho, a, b, alpha, nu, eta);
      tmp_candidate->Split_and_Merge(split_k, new_cluster1, new_cluster2, size1, size2, k_star_1, k_star_2);

      //std::cout << "[get_border]: tmp_candidate is :" << std::endl;
      //tmp_candidate->Print_Partition();
      tmp_objective = w[current_l]*total_log_post(tmp_candidate) + lambda * Entropy(current_l, tmp_candidate, particle_set, w) + xi * VI_DPP(current_l, tmp_candidate, particle_set);
      std::cout << "             max_obj = " << setprecision(8) << max_objective << " tmp_obj = " << setprecision(8) << tmp_objective << std::endl;
      std::cout << "			 log_like = " << setprecision(8) << total_log_like(tmp_candidate) << "  log_prior = " << setprecision(8) << total_log_prior(tmp_candidate) << std::endl;
      std::cout << "			 Entropy = " << setprecision(8) << lambda * Entropy(current_l, tmp_candidate, particle_set, w) << setprecision(8) << "  VI_DPP = " << setprecision(8) << xi * VI_DPP(current_l, tmp_candidate, particle_set) << std::endl;

      if(tmp_objective > max_objective){
        delete max_candidate;
        try{
          max_candidate = new Partition(tmp_candidate);
        }
        catch(const std::bad_alloc& e){
          cout << "######## EXCEPTION 6 ########"  << e.what() << endl;
        }


        max_objective = tmp_objective;
  //      std::cout << "max_obj is now: " << max_objective << std::endl;
        //max_candidate->Print_Partition();
      }
      } // closes loop over the neighboring clusters
    } // closes if statement that checks if there are neighboring clusters
    }

/*

  for(int i = 0; i < particle_set[current_l]->nObs; i++){
	split_k = particle_set[current_l]->cluster_assignment[i];
	neighbor_counter = 0;
	neighboring_clusters.clear();
	if(particle_set[current_l]->cluster_config[split_k] > 1){ // only attempt if there are at least 2 clusters. Otherwise this is just a Merge move
	  for(int k = 0; k < particle_set[current_l]->K;k++){
		if(k != split_k){ // loop over the elements of cluster k to see if any are adjacent to i
		  for(int j = 0; j < particle_set[current_l]->cluster_config[k]; j++){
			if(A_block(i, particle_set[current_l]->clusters[k][j]) == 1){ // j^th blockgroup of cluster k is adjacent to block group i
			  neighboring_clusters.push_back(k); // add k to the vector of adjacent
			  neighbor_counter++;
			  break; // once we know that i is adjacent to something in cluster k we can stop
			}
		  }
		}
	  }
	}
	if(neighbor_counter > 0){

	  std::cout << "[get_border]: i = " << i << "  borders clusters: ";
	  for(int nc = 0; nc < neighbor_counter; nc++){
		std::cout << neighboring_clusters[nc] << " ";
	  }
	  std::cout << std::endl;

	  // we are splitting i away from its cluster
	  // let us define all of the necessary components for that split
	  size1 = particle_set[current_l]->cluster_config[split_k] - 1;
	  size2 = 1;
	  delete[] new_cluster1;
	  delete[] new_cluster2;
	  new_cluster1 = new int[size1];
	  new_cluster2 = new int[size2];

	  new_cluster2[0] = i;
	  int counter = 0;
	  for(int ii = 0; ii < particle_set[current_l]->cluster_config[split_k]; ii++){
		if(particle_set[current_l]->clusters[split_k][ii] != i){
		  new_cluster1[counter] = particle_set[current_l]->clusters[split_k][ii];
		  counter++;
		}
	  }

	  // new_cluster1 contains all of the things in cluster split_k EXCEPT for i
	  // it will be "merged" with itself (i.e. k_star_1 = split_k)
	  k_star_1 = split_k;
	  for(int nc = 0; nc < neighbor_counter; nc++){
		k_star_2 = neighboring_clusters[nc]; // this is the new cluster that we are adding i to!
		//std::cout << "[get_border]: Attempting to move i = " << i << " from cluster " << split_k << " to cluster " << k_star_2 << std::endl;
		delete tmp_candidate;
		tmp_candidate = new Partition(particle_set[current_l]);
		tmp_candidate->Split_and_Merge(split_k, new_cluster1, new_cluster2, size1, size2, k_star_1, k_star_2);
		//std::cout << "[get_border]: tmp_candidate is :" << std::endl;
		//tmp_candidate->Print_Partition();
		tmp_objective = w[current_l]*total_log_post(tmp_candidate) + lambda * Entropy(current_l, tmp_candidate, particle_set, w);
		if(tmp_objective > max_objective){
		  delete max_candidate;
		  max_candidate = new Partition(tmp_candidate);
		  max_objective = tmp_objective;
		}
	  } // closes loop over the neighboring clusters
	} // closes if statement that checks if there are neighboring clusters
  } // closes for loop over all blockgroups
*/
  candidate->Copy_Partition(max_candidate); // set the candidate equal to the one with maximal objective function
  delete max_candidate;
  delete tmp_candidate;
  delete[] new_cluster1;
  delete[] new_cluster2;
}


//void get_merge(LPPartition candidate, int current_l, vector<LPPartition> particle_set, vector<double> w, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta){
void get_merge(LPPartition candidate, int current_l, std::vector<LPPartition> particle_set, std::vector<double> w, double lambda, double xi){

  LPPartition max_candidate = new Partition(particle_set[current_l]); // holds the running "best local candidate"
  double max_objective = 0.0; // holds the objective value of the "best local candidate"

  LPPartition tmp_candidate = new Partition(particle_set[current_l]); // the running candidate
  double tmp_objective = 0.0; // objective value of the running candidate

  max_objective = w[current_l]*total_log_post(max_candidate) + lambda * Entropy(current_l, max_candidate, particle_set, w);
  mat tmp_A;
  bool adj_clusters;
  if(particle_set[current_l]->K > 1){ // only makes sense to merge clusters if there are at least 2
    for(int k = 0; k < particle_set[current_l]->K - 1 ; k++){
      for(int kk = k+1; kk < particle_set[current_l]->K; kk++){
        //std::cout << "[get_merge] : Comparing cluster " << k << " and " << kk << std::endl;
        tmp_A = Submatrix(A_block, particle_set[current_l]->cluster_config[k], particle_set[current_l]->cluster_config[kk], particle_set[current_l]->clusters[k], particle_set[current_l]->clusters[kk]);
        // check if any element of tmp_A is equal to
        adj_clusters = any(vectorise(tmp_A == 1.0));
        //std::cout << neighboring_cluster << std::endl;
        if(adj_clusters){ // propose merging clusters k and kk!
          delete tmp_candidate;

          try{
            tmp_candidate = new Partition(particle_set[current_l]);
          }
          catch(const std::bad_alloc& e){
            cout << "######## EXCEPTION 7 ########"  << e.what() << endl;
          }
          //tmp_candidate->Merge(k,kk, Y, X, A_block, rho, a, b, alpha, nu, eta);
          tmp_candidate->Merge(k,kk);
          //tmp_candidate->Print_Partition();
          tmp_objective = w[current_l]*total_log_post(tmp_candidate) + lambda * Entropy(current_l, tmp_candidate, particle_set, w) + xi*VI_DPP(current_l, tmp_candidate, particle_set);
          std::cout << "[get_merge]: attempting to merge clusters " << k << " and " << kk << std::endl;
          std::cout << "             max_obj = " << setprecision(8) << max_objective << " tmp_obj = " << setprecision(8) << tmp_objective << std::endl;
          std::cout << "			    log_like = " << setprecision(8) << total_log_like(tmp_candidate) << "  log_prior = " << setprecision(8) << total_log_prior(tmp_candidate) << std::endl;
          std::cout << "			    Entropy = " << setprecision(8) << lambda * Entropy(current_l, tmp_candidate, particle_set, w) << setprecision(8) << "  VI_DPP = " << setprecision(8) << xi * VI_DPP(current_l, tmp_candidate, particle_set) << std::endl;

            if(tmp_objective > max_objective){
            delete max_candidate;
            try{
              max_candidate = new Partition(tmp_candidate);
            }
            catch(const std::bad_alloc& e){
              cout << "######## EXCEPTION 8 ########"  << e.what() << endl;
            }
            max_objective = tmp_objective;
            //std::cout << "max_obj is now: " << max_objective << std::endl;
            //max_candidate->Print_Partition();
          } // closes if statement that updates max_candidate
        } // closes if statement that proposes the merge
      } // closes inner loop over remaining clusters
    } // closes outer loop over clusters
  }

  /*
  // Sweep over all clusters and find the ones that are adjacent
  if(particle_set[current_l]->K > 1){ // only makes sense to merge clusters if there are at least 2

	for(int k = 0; k < particle_set[current_l]->K - 1 ; k++){
	  for(int kk = k+1; kk < particle_set[current_l]->K; kk++){
		//std::cout << "[get_merge] : Comparing cluster " << k << " and " << kk << std::endl;
		tmp_A = Submatrix(A_block, particle_set[current_l]->cluster_config[k], particle_set[current_l]->cluster_config[kk], particle_set[current_l]->clusters[k], particle_set[current_l]->clusters[kk]);
		// check if any element of tmp_A is equal to
		adj_clusters = any(vectorise(tmp_A == 1.0));
		//std::cout << neighboring_cluster << std::endl;
		if(adj_clusters){ // propose merging clusters k and kk!
		  delete tmp_candidate;
		  tmp_candidate = new Partition(particle_set[current_l]);
		  tmp_candidate->Merge(k,kk);
		  //tmp_candidate->Print_Partition();
		  tmp_objective = w[current_l]*total_log_post(tmp_candidate) + lambda * Entropy(current_l, tmp_candidate, particle_set, w);
		  //std::cout << "[get_merge]: tmp_objective = " << setprecision(8) << tmp_objective << "  max_objective = " << setprecision(8) << max_objective << std::endl;
		  if(tmp_objective > max_objective){
			delete max_candidate;
			max_candidate = new Partition(tmp_candidate);
			max_objective = tmp_objective;
		  } // closes if statement that updates max_candidate
		} // closes if statement that proposes the merge
	  } // closes inner loop over remaining clusters
	} // closes outer loop over clusters
  } // closes if statement that checks the number of
*/
  candidate->Copy_Partition(max_candidate);
  delete max_candidate;
  delete tmp_candidate;
}
// Split candidates
void get_split(LPPartition candidate, int current_l, std::vector<LPPartition> particle_set, std::vector<double> w, double lambda, double xi){
  LPPartition max_candidate = new Partition(particle_set[current_l]); // holds the running "best local candidate"
  double max_objective = 0.0; // holds the objective value of the "best local candidate"

  LPPartition tmp_candidate = new Partition(particle_set[current_l]); // the running candidate
  double tmp_objective = 0.0; // objective value of the running candidate

  max_objective = w[current_l]*total_log_post(max_candidate) + lambda * Entropy(current_l, max_candidate, particle_set, w) + xi * VI_DPP(current_l, max_candidate, particle_set);
  int size1 = 0;
  int size2 = 0;
  int* new_cluster1;
  int* new_cluster2;
  try{
    new_cluster1 = new int[1];
    new_cluster2 = new int[1];
  }
  catch(const std::bad_alloc& e){
    cout << "######## EXCEPTION 12c ########"  << e.what() << endl;
  }
  for(int k = 0; k < particle_set[current_l]->K; k++){
    if(particle_set[current_l]->cluster_config[k] > 1){ // only makes sense to split clusters
      int n = particle_set[current_l]->nObs;
      int n_cl = particle_set[current_l]->cluster_config[k];
      arma::mat A_block_cluster = Submatrix(A_block, n_cl, n_cl, particle_set[current_l]->clusters[k], particle_set[current_l]->clusters[k]);
      double* beta_hat_cluster = new double[n_cl];
      if(beta_hat_cluster == nullptr) cout << "######## EXCEPTION 5b ########" << endl;
      for(int i = 0; i < n_cl; i++){
        beta_hat_cluster[i] = particle_set[current_l]->beta_hat[ particle_set[current_l]->clusters[k][i] ];
      }
      arma::mat dist = Distance_matrix(beta_hat_cluster, n_cl);
      arma::vec beta_hat_cluster_vec(n_cl);
      for(int i = 0; i < n_cl; i++){
        beta_hat_cluster_vec(i) = beta_hat_cluster[i];
      }
      delete[] beta_hat_cluster;
      arma::mat beta_sim = exp(- square(dist) / (2 * arma::var(beta_hat_cluster_vec)));
      arma::mat diag_ncl(n_cl,n_cl,fill::eye);
      arma::mat W_beta_cl =  diag_ncl + beta_sim % A_block_cluster;
      arma::mat Dinv_sqrt = arma::diagmat(1/sqrt(arma::sum(W_beta_cl, 1)));
      arma::mat L = diag_ncl - Dinv_sqrt * W_beta_cl * Dinv_sqrt;
      arma::vec eigval; // the smallest eigenvalues are the first two
      arma::mat eigvec;
      arma::eig_sym(eigval, eigvec, L);
      arma::mat U = eigvec.cols(0,1);
      U = arma::diagmat(1/sqrt(arma::sum(arma::square(U), 1))) * U;
      arma::mat means;
      // kmeans(means, U.t(), 2, random_subset, 10, false);
      bool status = arma::kmeans(means, U.t(), 2, random_subset, 10, false);
      if(status == false)
        cout << "clustering failed" << endl;
      int * membership = which_is_nearest(means, U.t());
      int *index1 = new int[n_cl];
      if(index1 == nullptr) cout << "######## EXCEPTION 6b ########" << endl;
      int *index2 = new int[n_cl];
      if(index2 == nullptr) cout << "######## EXCEPTION 7b ########" << endl;
      int n1 = 0, n2 = 0;
      for(int i = 0; i < n_cl; i++){
        if(membership[i] == 0){
          index1[n1] = particle_set[current_l]->clusters[k][i];
          n1++;
        } else {
          index2[n2] = particle_set[current_l]->clusters[k][i];
          n2++;
        }
      }
      delete[] membership;

      // we now know how to split cluster k
      // first we initialize the tmp_candidate
      delete tmp_candidate;
      try{
        tmp_candidate = new Partition(particle_set[current_l]);
      }
      catch(const std::bad_alloc& e){
        cout << "######## EXCEPTION 9 ########"  << e.what() << endl;
      }


      tmp_objective = 0.0;
      // now just try splitting
      tmp_candidate->Split(k, index1, index2, n1, n2);
      tmp_objective = w[current_l]*total_log_post(tmp_candidate) + lambda * Entropy(current_l, tmp_candidate, particle_set, w) + xi*VI_DPP(current_l, tmp_candidate, particle_set);
      std::cout << "[get_split]: attempting to split cluster " << k  << std::endl;
      std::cout << "             max_obj = " << setprecision(8) << max_objective << " tmp_obj = " << setprecision(8) << tmp_objective << std::endl;
      std::cout << "			    log_like = " << setprecision(8) << total_log_like(tmp_candidate) << "  log_prior = " << setprecision(8) << total_log_prior(tmp_candidate) << std::endl;
      std::cout << "			    Entropy = " << setprecision(8) << lambda * Entropy(current_l, tmp_candidate, particle_set, w) << setprecision(8) << "  VI_DPP = " << setprecision(8) << xi * VI_DPP(current_l, tmp_candidate, particle_set) << std::endl;

      if(tmp_objective > max_objective){
        delete max_candidate;
        try{
          max_candidate = new Partition(tmp_candidate);
        }
        catch(const std::bad_alloc& e){
          cout << "######## EXCEPTION 10 ########"  << e.what() << endl;
        }
        max_objective = tmp_objective;
      }
      // now let's try splitting and merging
      // First split and merge: keep index1 by itself (and labelled cluster k) and merge index2 into the closest existing cluster that is adjacent to it and has similar average beta-hat (requires identify k_star_2)
      // Second split and merge: keep index2 by itself (and labelled cluster K) and merge index1 into closest existing cluster that is adjacent to it and has similar average beta-hat (requires identifying k_star_1)
      // Third split and merge: merge both index1 and index2 into existing clusters (so long as they're not merging into the same one!!)

      delete[] index1;
      delete[] index2;
    }
  }

  candidate->Copy_Partition(max_candidate);
  delete[] new_cluster1;
  delete[] new_cluster2;
  delete max_candidate;
  delete tmp_candidate;
}

