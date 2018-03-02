/*
 * various_functions.h
 *
 *  Created on: January 23, 2018
 *      Author: Sameer
 */


#ifndef VARIOUS_FUNCTIONS_H_
#define VARIOUS_FUNCTIONS_H_

arma::mat Submatrix(arma::mat M, int n_rows, int n_cols, int* row_index, int* col_index);
void Connected_Components(arma::mat M, int n, int* components, int* count);
void DFSUtil(arma::mat M, int n, int v, bool* visited, int* components, int* count);
int* which_is_nearest(arma::mat centroids, arma::mat data);
int* which_is_nearest_k(arma::mat centroids, arma::mat data);
arma::mat Distance_matrix(double* beta_hat, int nObs);

#endif /* VARIOUS_FUNCTIONS_H_ */
