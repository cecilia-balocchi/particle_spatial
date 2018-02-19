/*
 * ensm_partition.cpp
 *
 *  Created on: Feb 14, 2018
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
//extern arma::mat X;
//extern arma::mat Y;
//extern arma::mat A_block;
//extern double rho;
//extern double a;
//extern double b;
//extern double nu;
//extern double alpha;
//extern double eta;
extern int max_iter;

void ensm_partition(vector<LPPartition>& particle_set, double lambda, double xi){
  int iter = 0;
  int flag = 0;
  int conv_counter = 0; // counter to see how many particles remain unchanged
  int L = particle_set.size();

  vector<double> w(L); // importance weights
  for(unsigned l = 0; l < L; l++){
    w[l] = (double) 1/L;
  }  double old_objective = 0.0;
  double objective = 0.0;

  objective = lambda * Entropy(0, particle_set[0], particle_set, w) + xi * VI_DPP(0, particle_set[0], particle_set);
  for(int l = 0; l < L; l++){
    objective += w[l] * total_log_post(particle_set[l]);
  }
  LPPartition candidate = new Partition(particle_set[0]);

  while((iter < max_iter) & (flag == 0)){

    // try to update the particle
    // compute the old objective value
    std::cout << "[particle_spatial]: Starting iter = " << iter << std::endl;
    old_objective = objective;
    // sweep over the particle set
    conv_counter = 0;
    for(int l = 0; l < L; l++){
      std::cout << "[particle_spatial]: updating particle " << l << std::endl;
      // free up the candidate
      delete candidate;
      try{
        candidate = new Partition(particle_set[l]);
      }
      catch(const std::bad_alloc& e){
        cout << "EXCEPTION IN PARTICLE SPATIAL"  << e.what() << endl;
      }
      update_particle(candidate, l, particle_set, w, lambda, xi);
      conv_counter += Partition_Equal(particle_set[l], candidate); // check if the current particle is the best
      particle_set[l]->Copy_Partition(candidate);
    }

    // now let us update w
    update_w(particle_set, w, lambda, xi);

    // now compute the objective
    objective = lambda * Entropy(0, particle_set[0], particle_set, w) + xi * VI_DPP(0, particle_set[0], particle_set); // compute the entropy
    for(int l = 0; l < L; l++){
      objective += w[l] * total_log_post(particle_set[l]);
    }

    std::cout << std::endl;
    std::cout << "[ensm_partition]: obj = " << setprecision(8) << objective << "    old_obj = " << setprecision(8) << old_objective << std::endl;
    std::cout << "[ensm_partition]: percent increase in objective = " << setprecision(6) << 100*fabs((objective - old_objective)/old_objective) << std::endl;
    std::cout << "[ensm_partition]: number of stationary particles = " << conv_counter << std::endl;
    std::cout << "[ensm_partition]: importance weights :  " << std::endl;
    for(int l = 0; l < L; l++){
      std::cout << setprecision(6) << w[l] << "   " ;
    }
    std::cout << std::endl;
    // check for convergence
    if(objective < old_objective){
      std::cout << "WARNING THE OBJECTIVE DECREASED" << std::endl;
    }
    flag = 0;
    if((conv_counter == L) || ( fabs((objective - old_objective)/old_objective) < 1e-6)){
      flag = 1;
    }
    iter++;
  }


}


