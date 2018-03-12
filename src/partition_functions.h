/*
 * partition_functions.h
 *
 *  Created on: Dec 30, 2017
 *      Author: Sameer
 */

int Partition_Equal(Partition *partition1, Partition *partition2);
double beta_bar(Partition *partition, int k);
double Entropy(unsigned current_l, Partition* candidate_particle, std::vector<LPPartition> particle_set, std::vector<double> w);
double total_log_post(LPPartition partition);
double total_log_like(LPPartition partition);
double total_log_prior(LPPartition partition);
double Binder_Loss(LPPartition partition1, LPPartition partition2);
double VI_Loss(LPPartition partition1, LPPartition partition2);
double VI_Avg(unsigned current_l, Partition* candidate_particle, std::vector<LPPartition> particle_set);

double Binder_DPP(unsigned current_l, Partition* candidate_particle, std::vector<LPPartition> particle_set);
double VI_DPP(unsigned current_l, Partition* candidate_particle, std::vector<LPPartition> particle_set);



// The below functions are really only used for testing purposes!
// create an "island" out of each block-groupd
//void get_island(LPPartition candidate, int current_l, vector<LPPartition> particle_set, vector<double> w, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta);
//void get_border(LPPartition candidate, int current_l, vector<LPPartition> particle_set, vector<double> w, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta);
//void get_merge(LPPartition candidate, int current_l, vector<LPPartition> particle_set, vector<double> w, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta);

void get_island(LPPartition candidate, int current_l, std::vector<LPPartition> particle_set, std::vector<double> w, double lambda, double xi);
void get_border(LPPartition candidate, int current_l, std::vector<LPPartition> particle_set, std::vector<double> w, double lambda, double xi);
void get_merge(LPPartition candidate, int current_l, std::vector<LPPartition> particle_set, std::vector<double> w, double lambda, double xi);
void get_split(LPPartition candidate, int current_l, std::vector<LPPartition> particle_set, std::vector<double> w, double lambda, double xi);



