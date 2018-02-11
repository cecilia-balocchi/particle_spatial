/*
 * update_particle.h
 *
 *  Created on: Jan 13, 2018
 *      Author: Sameer
 */

#ifndef UPDATE_PARTICLE_H_
#define UPDATE_PARTICLE_H_

//void update_particle(LPPartition candidate, int current_l, vector<LPPartition> particle_set, vector<double> w, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta);
//void update_w(vector<LPPartition> particle_set, vector<double>& w, mat Y, mat X, mat A_block, double rho, double a, double b, double alpha, double nu, double eta );
void update_particle(LPPartition candidate, int current_l, std::vector<LPPartition> particle_set, std::vector<double> w);
void update_w(std::vector<LPPartition> particle_set, std::vector<double>& w);



#endif /* UPDATE_PARTICLE_H_ */
