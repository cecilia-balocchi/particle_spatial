/*
 * various_functions.cpp
 *
 *  Created on: January 23, 2018
 *      Author: Sameer
 *  
 */

#include <iostream>
#include <iomanip>
#include <time.h>
#include <vector>
#include <algorithm>
#include <armadillo>
#include "various_functions.h"
using namespace std;
using namespace arma;


// we need to restrict rows AND columns with index

mat Submatrix(mat M, int n_row, int n_col, int* row_index, int* col_index){

  mat N(n_row, n_col);
  for(int i = 0; i < n_row; i++){
    for(int j = 0; j < n_col; j++){
	  N(i,j) = M(row_index[i],col_index[j]);
	}
  }

  return N;
}

void Connected_Components(mat M, int n, int* components, int* count)
{
	// Mark all the vertices as not visited
	bool *visited = new bool[n];
	*count = 0;
	for(int v = 0; v < n; v++)
		visited[v] = false;
	for(int v = 0; v < n; v++)
	{
		if(visited[v] == false)
		{
			DFSUtil(M, n, v, visited, components, count);
			(*count)++;
		}
	}
	delete[] visited;
}
void DFSUtil(mat M, int n, int v, bool* visited, int* components, int* count)
{
	visited[v] = true;
	components[v] = *count;
	for(int i = 0; i < n; i++)
		if(M(v,i) == 1.0)
			if(visited[i] == false)
				DFSUtil(M, n, i, visited, components, count);
}

int* which_is_nearest(arma::mat centroids, arma::mat data){
  // this is written for only two centroids
  int *membership;
  membership = new int[data.n_cols];
  for(int point_ind = 0; point_ind < data.n_cols; point_ind++){
    double dist0 = arma::norm(data.col(point_ind) - centroids.col(0));
    double dist1 = arma::norm(data.col(point_ind) - centroids.col(1));
    if(dist0 <= dist1){
      membership[point_ind] = 0;
    } else {
      membership[point_ind] = 1;
    }
  }
  return membership;
}

mat Distance_matrix(double* beta_hat, int nObs){
  mat dist(nObs,nObs, fill::zeros);
  for(int i = 0; i < nObs; i++)
  {
    for(int j = i+1; j < nObs; j++)
    {
      dist(i,j) = abs(beta_hat[i] - beta_hat[j]);
      dist(j,i) = dist(i,j);
    }
  }
  return dist;
}