/*
 * particle_spatial.cpp
 *
 *  Created on: Jan 9, 2018
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
#include <armadillo>
#include "partition.h"
#include "partition_functions.h"
//#include "matrix_functions.h"
#include "various_functions.h"
#include "update_particle.h"
using namespace std;
using namespace arma;


// I want these parameters to be accessible to everybody
// This allows me to avoid passing them as arguments
arma::mat Y;
arma::mat X;
arma::mat A_block;
// Set the hyper-parameters
double rho = 0.9;
double a = 1.0;
double b = 1.0;
double nu = 1.0;
double alpha = nu/2;
double eta = 1.0;
double lambda = 1.0;
double xi = 1.0;


int main(){
  // Some stuff for the main loop
  int iter = 0;
  int max_iter = 10;
  int flag = 0;
  int conv_counter = 0; // counter to see how many particles remain unchanged

  double old_objective = 0.0;
  double objective = 0.0;

  // Load in the important data
  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  printf ( "Starting date and time are: %s", asctime (timeinfo) );

  //mat Y, X, A_block;
  Y.load("Y_jan10.csv", csv_ascii);
  X.load("X_jan10.csv", csv_ascii);
  A_block.load("A_block_jan10.csv", csv_ascii);
  std::cout << "Loaded the matrices" << std::endl;
  int n = Y.n_rows;

  LPPartition gamma_0 = new Partition;
  gamma_0->Initialize_Partition2(0);
  LPPartition gamma_1 = new Partition;
  gamma_1->Initialize_Partition2(1);

  LPPartition gamma_2 = new Partition;
  gamma_2->Initialize_Partition2(2);


  LPPartition gamma_3 = new Partition;
  gamma_3->Initialize_Partition2(3);

  LPPartition gamma_4 = new Partition;
  gamma_4->Initialize_Partition2(4);
/*
  cout << "Binder loss between gamma_0 and gamma_0 = " << Binder_Loss(gamma_0, gamma_0) << endl;
  cout << "Binder loss between gamma_0 and gamma_1 = " << Binder_Loss(gamma_0, gamma_1) << endl;
  cout << "Binder loss between gamma_0 and gamma_2 = " << Binder_Loss(gamma_0, gamma_2) << endl;
  cout << "Binder loss between gamma_0 and gamma_3 = " << Binder_Loss(gamma_0, gamma_3) << endl;
  cout << "Binder loss between gamma_1 and gamma_3 = " << Binder_Loss(gamma_1, gamma_3) << endl;

  cout << "VI loss between gamma_0 and gamma_0 = " << VI_Loss(gamma_0, gamma_0) << endl;
  cout << "VI loss between gamma_0 and gamma_1 = " << VI_Loss(gamma_0, gamma_1) << endl;
  cout << "VI loss between gamma_0 and gamma_2 = " << VI_Loss(gamma_0, gamma_2) << endl;
  cout << "VI loss between gamma_0 and gamma_3 = " << VI_Loss(gamma_0, gamma_3) << endl;
  cout << "VI loss between gamma_1 and gamma_3 = " << VI_Loss(gamma_1, gamma_3) << endl;
*/


  //Binder_Loss(gamma_0, gamma_2);


  // make our particle set
  int L = 10;
  vector<LPPartition> particle_set(L);
  vector<double> w(L); // importance weights
  for(unsigned l = 0; l < L; l++){
    particle_set[l] = new Partition(gamma_1);
    w[l] = (double) 1/L;
  }

  objective = lambda * Entropy(0, particle_set[0], particle_set, w) + xi * VI_DPP(0, particle_set[0], particle_set);
  for(int l = 0; l < L; l++){
    objective += w[l] * total_log_post(particle_set[l]);
  }
  cout << "Entropy = " << Entropy(0, particle_set[0], particle_set, w) << endl;
  cout << "VI_DPP = " << VI_DPP(0, particle_set[0], particle_set) << endl;
  cout << "Objective = " << objective << endl;

  LPPartition island_candidate = new Partition();
  island_candidate->Initialize_Partition(n);
  get_island(island_candidate, 0, particle_set, w);

  LPPartition border_candidate = new Partition();
  border_candidate->Initialize_Partition(n);
  get_border(border_candidate, 0, particle_set, w);

  LPPartition split_candidate = new Partition();
  split_candidate->Initialize_Partition(n);
  get_split(split_candidate, 0, particle_set, w);

  island_candidate->Print_Partition();
  border_candidate->Print_Partition();
  split_candidate->Print_Partition();




/*
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
      update_particle(candidate, l, particle_set, w);
      conv_counter += Partition_Equal(particle_set[l], candidate); // check if the current particle is the best
      particle_set[l]->Copy_Partition(candidate);
    }


    // now let us update w
    update_w(particle_set, w);

    // now compute the objective
    objective = lambda * Entropy(0, particle_set[0], particle_set, w) + xi * VI_DPP(0, particle_set[0], particle_set); // compute the entropy
    for(int l = 0; l < L; l++){
      objective += w[l] * total_log_post(particle_set[l]);
    }

    std::cout << std::endl;
    std::cout << "[particle_spatial]: obj = " << setprecision(8) << objective << "    old_obj = " << setprecision(8) << old_objective << std::endl;
    std::cout << "[particle_spatial]: percent increase in objective = " << setprecision(6) << 100*fabs((objective - old_objective)/old_objective) << std::endl;
    std::cout << "[particle_spatial]: number of stationary particles = " << conv_counter << std::endl;
    std::cout << "[particle_spatial]: importance weights :  " << std::endl;
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
  delete candidate;

  for(int l = 0; l < L; l++){
    particle_set[l]->Print_Partition();
    cout << endl;
  }

  for(int l = 0; l < L; l++){
    delete particle_set[l];
  }
*/
  // time to clean things up
  delete gamma_0;
  delete gamma_1;
  delete gamma_2;
  delete gamma_3;
  delete gamma_4;


  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  printf ( "Ending date and time are: %s", asctime (timeinfo) );

  return 0;
}


