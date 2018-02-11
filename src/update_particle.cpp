/*
 * update_particle.cpp
 *
 *  Created on: Jan 13, 2018
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
#include "partition_functions.h"
#include "update_particle.h"
extern arma::mat A_block;
extern double lambda;
extern double xi;

using namespace std;

//void update_w(vector<LPPartition> particle_set, vector<double>& w, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta ){
void update_w(std::vector<LPPartition> particle_set, std::vector<double>& w){
  // First we identify the unik partitions
  double max_log_post = 0.0;
  double tmp_log_post = 0.0;
  double tmp_norm = 0.0;
  double tmp_p_star = 0.0;
  unsigned L = particle_set.size();
  // need to loop over to extract the unique partitions
  std::vector<LPPartition> unik_particles;
  unik_particles.push_back(particle_set[0]);

  std::vector<int> particle_assignment(L); // will tell us which unik particle each particle in the particle set is
  particle_assignment[0] = 0;

//  vector<int> particle_counts; // counts multiplicity of the unique particles
//  particle_counts.push_back(1);

  std::vector<double> p_star;
  std::vector<double> log_post;

  log_post.push_back(total_log_post(particle_set[0]));
  p_star.push_back(0);

  //std::cout << "[update_w]: log_post[0] = " << log_post[0] << std::endl;
  max_log_post = total_log_post(particle_set[0]);

  int num_unik_particles = 1;
  int counter = 0;
  for(unsigned l = 1; l < L; l++){ // loop over current particle set
  counter = 0;
  for(unsigned ul = 0; ul < num_unik_particles; ul++){
//    std::cout << "[update_w]: l = " << l << " ul = " << ul << "   " << Partition_Equal(particle_set[l], unik_particles[ul]) << std::endl;
    if(Partition_Equal(particle_set[l], unik_particles[ul]) == 1){
      // l^th partition is equal to the ul^th unique partition
      //std::cout << "particle " << l << " is equal to unik particle " << ul << std::endl;
      particle_assignment[l] = ul;
//      particle_counts[ul]++;
//      std::cout << "particle_counts[" << ul << "] = " << particle_counts[ul] << std::endl;
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
    particle_assignment[l] = num_unik_particles; // the labels are off-set by 1
    p_star.push_back(0.0); // for now, we will populate p_star with 0's
    tmp_log_post = total_log_post(particle_set[l]);
    log_post.push_back(tmp_log_post);
    if(tmp_log_post > max_log_post){
    max_log_post = tmp_log_post;
    }
    //particle_counts.push_back(1);
    //std::cout << "particle_counts[" << num_unik_particles << "] = " << particle_counts[num_unik_particles] << std::endl;
    num_unik_particles++;
    }
  }

  int particle_counts[num_unik_particles];
  int tmp_count = 0;
  for(int ul = 0; ul < num_unik_particles; ul++){
  tmp_count = 0;
  for(int l = 0; l < L; l++){
    if(particle_assignment[l] == ul){
    tmp_count++;
    }
  }
  particle_counts[ul] = tmp_count;
  }

/*
  std::cout << "[update_w] : Particle Assignment is " << std::endl;
  for(int l = 0; l < L ; l++){
  std::cout << particle_assignment[l] << " " ;
  }
  std::cout << std::endl;
  std::cout << "[update_w] : log_post is " << std::endl;
  for(int ul = 0; ul < num_unik_particles; ul++){
  std::cout << log_post[ul] << "    ";
  }
  std::cout << std::endl;
  std::cout << "[update_w] : particle_counts is " << std::endl;
  for(int ul = 0; ul < num_unik_particles; ul++){
  std::cout << particle_counts[ul] << "  " ;
  }
  std::cout << std::endl;
*/
//  std::cout << "[update_w] : max_log_post is " << max_log_post << std::endl;
//  std::cout << "p_star: " ;
  for(int ul = 0; ul < num_unik_particles; ul++){
    tmp_log_post = log_post[ul] - max_log_post;
    tmp_p_star = exp(1/lambda * tmp_log_post); // introduce the 1/lambda
    tmp_norm += tmp_p_star;
    p_star[ul] = tmp_p_star;
//  std::cout << p_star[ul] << "  " ;
  }
//  std::cout << "[update_w] : After re-normalization p_star : " << std::endl;
  for(int ul = 0; ul < num_unik_particles; ul++){
  p_star[ul] /= tmp_norm;
//  std::cout << p_star[ul] << "   " ;
  }
//  std::cout << std::endl;
//  std::cout << "[update_w]: w is" << std::endl;
  for(int l = 0; l < L; l++){
//  std::cout << "particle_assignment[l] = " << particle_assignment[l] << "  particle_counts[particle_assignment[l]] = " << particle_counts[particle_assignment[l]] << std::endl;
//  std::cout << p_star[particle_assignment[l]] << std::endl;
//  std::cout << p_star[particle_assignment[l]]/particle_counts[particle_assignment[l]] << std::endl;
  w[l] = (double) p_star[particle_assignment[l]]/particle_counts[particle_assignment[l]];
//  std::cout << "[update_w]: w[" << l << "] = " << w[l] << std::endl;
  }
  //std::cout << std::endl;
  return;
}






//void update_particle(LPPartition candidate, int current_l, vector<LPPartition> particle_set, vector<double> w, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta){
void update_particle(LPPartition candidate, int current_l, std::vector<LPPartition> particle_set, std::vector<double> w){
  LPPartition max_candidate;
  LPPartition tmp_candidate;
  try{
    max_candidate = new Partition(particle_set[current_l]); // holds the running "best local candidate"
  }
  catch(const std::bad_alloc& e){
    cout << "######## EXCEPTION 1 ########"  << e.what() << endl;
  }
  double max_objective = 0.0; // holds the objective value of the "best local candidate"
  
  try{
    tmp_candidate = new Partition(particle_set[current_l]); // the running candidate
  }
  catch(const std::bad_alloc& e){
    cout << "######## EXCEPTION 2 ########"  << e.what() << endl;
  }
  double tmp_objective = 0.0; // objective value of the running candidate

  max_objective = w[current_l]*total_log_post(max_candidate) + lambda * Entropy(current_l, max_candidate, particle_set, w) + xi * VI_DPP(current_l, max_candidate, particle_set);

  // Initialize the stuff for splitting and merging clusters
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
  int split_k = 0;
  int k_star_1 = 0;
  int k_star_2 = 0;
  // Initialize stuff for border candidates
  std::vector<int> neighboring_clusters; // holds the labels of clusters adjacent to block-group i
  int neighbor_counter = 0;
  // Initialize stuff for the merge candidates
  mat tmp_A;
  bool adj_clusters;
   
    // First try to create islands
  //  std::cout<< "[update_particle] : Trying island candidates now" << std::endl;
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
      tmp_candidate->Modify(split_k);
      //tmp_candidate->Print_Partition();

      tmp_objective = w[current_l]*total_log_post(tmp_candidate) + lambda * Entropy(current_l, tmp_candidate, particle_set, w) + xi * VI_DPP(current_l, tmp_candidate, particle_set);
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
    
    // Now try to move border elements
    //std::cout<< "[update_particle] : Trying border candidates now" << std::endl;
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
      //std::cout << "[get_border]: Attempting to move i = " << i << " from cluster " << split_k << " to cluster " << k_star_2 << std::endl;
      delete tmp_candidate;
      
      try{
        tmp_candidate = new Partition(particle_set[current_l]);
      }
      catch(const std::bad_alloc& e){
        cout << "######## EXCEPTION 5 ########"  << e.what() << endl;
      }

      //tmp_candidate->Split_and_Merge(split_k, new_cluster1, new_cluster2, size1, size2, k_star_1, k_star_2, Y, X, A_block, rho, a, b, alpha, nu, eta);
      tmp_candidate->Split_and_Merge(split_k, new_cluster1, new_cluster2, size1, size2, k_star_1, k_star_2);
      tmp_candidate->Modify(split_k);
      //std::cout << "[get_border]: tmp_candidate is :" << std::endl;
      //tmp_candidate->Print_Partition();
      tmp_objective = w[current_l]*total_log_post(tmp_candidate) + lambda * Entropy(current_l, tmp_candidate, particle_set, w) + xi * VI_DPP(current_l, tmp_candidate, particle_set);
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
    } // closes loop over the blockgroups
    
    // Now try merge candidates
    //std::cout << "[update_particle]: Trying merge candidates now" << std::endl;

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
            //std::cout << "[get_merge]: tmp_objective = " << setprecision(8) << tmp_objective << "  max_objective = " << setprecision(8) << max_objective << std::endl;
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
    
    // Now do the heavy lifting: we will loop over the clusters, run spectral clustering on the beta_hat's within the cluster, and propose various split and merge candidates
    
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
  
  // catch(const std::bad_alloc& e){
  //  cout << "----------------EXCEPTION IN UPDATE PARTICLE: " << e.what() << endl;
  // }
  
}

