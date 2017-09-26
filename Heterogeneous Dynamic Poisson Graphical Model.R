##########################################################################
# Heterogeneous Dynamic Poisson Graphical Model
# Auther: Jing Chu
# Date: June 2017
##########################################################################
# This code randomly generate a sample for heterogeneous dynamic
# transcription factor networks and implemented a Hierarchical Dynamic
# Poisson Graphical Model for nonparametric Bayesian learning of the
# network. In particular, an efficient parallel Markov chain Monte Carlo
# algorithm for posterior computation is developed
##########################################################################


##########################################################################
#Generate dataset (K clusters, R replicates, D conditions)
##########################################################################
rm(list=ls(all=TRUE))

N <- 5000
#number of genomic locations

D <- 3
#number of conditions

R <- 2
#number of replicates

K <- 2
#number of clusters

p <- 7
#number of nodes

Pi <- rep(NA, K)
#Pi[k] represents the probability that one object is assigned to cluster k

Lambda <- array(NA, dim = c(D, K, p, p))
#Lambda[ d, k,,] represents the probability structure in cluster k under condition d

cla <- rep(NA, N)
# cla[n] shows the latent class (cluster) membership of object n

Y <- array(NA, dim = c(D, R, N, p, p))
# Y[d,r,n,,] denotes the latent read counts for object n in r-th replicate under condition d 

X <- array(NA, dim = c(D, R, N, p))
# X[d,r,n,] denotes the observed read counts for object n in r-th replicate under condition d


set.seed(02222017)


#Setting values for cluster proportion
Pi <- c(0.6, 0.4)

#Setting values for network structures and intensity parameters

#cluster 1

Lambda[1,1,,] <- matrix( 	 c(2, 4, 0, 0, 0, 0, 0,
				   4, 2, 4, 0, 0, 0, 0,
				   0, 4, 2, 0, 0, 0, 0,
				   0, 0, 0, 2, 4, 0, 0,
				   0, 0, 0, 4, 2, 4, 0,
				   0, 0, 0, 0, 4, 2, 4,
				   0, 0, 0, 0, 0, 4, 2), nrow = p, byrow = T )

Lambda[2,1,,] <- matrix( 	 c(2, 0, 1, 0, 0, 0, 0,
				   0, 2, 0, 0, 2, 0, 0,
				   1, 0, 2, 1, 0, 0, 0,
				   0, 0, 1, 2, 1, 0, 0,
				   0, 2, 0, 1, 2, 3, 0,
				   0, 0, 0, 0, 3, 2, 0,
				   0, 0, 0, 0, 0, 0, 2), nrow = p, byrow = T )



Lambda[3,1,,] <- matrix( 	 c(2, 1, 0, 0, 0, 0, 0,
				   1, 2, 3, 0, 0, 0, 0,
				   0, 3, 2, 0, 0, 0, 0,
				   0, 0, 0, 2, 1, 0, 0,
				   0, 0, 0, 1, 2, 1, 0,
				   0, 0, 0, 0, 1, 2, 0,
				   0, 0, 0, 0, 0, 0, 2), nrow = p, byrow = T )


#cluster 2

Lambda[1,2,,] <- matrix( 	 c(2, 4, 0, 0, 0, 0, 0,
				   4, 2, 4, 0, 0, 0, 0,
				   0, 4, 2, 0, 0, 2, 0,
				   0, 0, 0, 2, 4, 0, 0,
				   0, 0, 0, 4, 2, 4, 0,
				   0, 0, 2, 0, 4, 2, 2,
				   0, 0, 0, 0, 0, 2, 2), nrow = p, byrow = T )

Lambda[2,2,,] <- matrix( 	 c(2, 0, 3, 0, 0, 0, 0,
				   0, 2, 0, 0, 0, 0, 0,
				   3, 0, 2, 0, 0, 0, 0,
				   0, 0, 0, 2, 1, 0, 0,
				   0, 0, 0, 1, 2, 3, 0,
				   0, 0, 0, 0, 3, 2, 3,
				   0, 0, 0, 0, 0, 3, 2), nrow = p, byrow = T )

Lambda[3,2,,] <- matrix( 	 c(2, 1, 0, 0, 0, 0, 0,
				   1, 2, 3, 0, 0, 3, 0,
				   0, 3, 2, 0, 3, 0, 0,
				   0, 0, 0, 2, 1, 0, 0,
				   0, 0, 3, 1, 2, 1, 0,
				   0, 3, 0, 0, 1, 2, 0,
				   0, 0, 0, 0, 0, 0, 2), nrow = p, byrow = T )

#generate X
for(n in 1:N) {
	cla[n] <- sample(1:K, 1, prob = Pi)
	for(d in 1:D) {
			for(i in 1:p) {
				for(j in i:p) {
					Y[d, ,n,i,j] <- rpois(R, lambda = Lambda[d,cla[n], i, j])
					if(j != i)
						Y[d, ,n,j,i] <- Y[d, ,n,i,j]
				}
				X[d, ,n,i] <- rowSums(Y[d, ,n,i,])
			}
	}
}

###################################################################
#Dirichlet Process Mixture Model (Sampling via Block Gibbs Sampler)
###################################################################

###############################
#several functions
###############################

#sample from a Dirichlet distribution
Dirich <- function(alpha) {
	num <- length(alpha)
	y <- rgamma(num , shape = alpha , rate = 1)
	return(y/sum(y))
	
}

################################		
#sample from base distribution H 
################################

sample_H <- function(par_0, par_1, par_2, par_p_t) {
	#par_0 is the parameter vector when there is no edge
	#par_1 is the parameter vector when there is an edge
	#par_2 is the parameter vector associated with the node
	#par_p_t is the parameter indicating the probability of the presence of an edge

	L_mat <- array(NA, dim = c(D,p,p))
	Lambda_mat <- array(NA, dim = c(D,p,p))
	for(d in 1:D)
		for(i in 1:p)
			for(j in i:p){
				if(j == i) {
					Lambda_mat[d,i,j] <- rgamma(1, shape = par_2[1], rate = par_2[2])
				} else {
					L_mat[d, i, j] <- sample(c(0, 1), 1, prob = c( 1 - par_p_t, par_p_t))
					if(L_mat[d, i, j] == 1){
						Lambda_mat[d,i,j] <- rgamma(1, shape = par_1[1], rate = par_1[2])
					}else{
						Lambda_mat[d,i,j] <- rgamma(1, shape = par_0[1], rate = par_0[2])
					}
					L_mat[d, j, i] <- L_mat[d, i, j]
					Lambda_mat[d,j,i] <- Lambda_mat[d,i,j] 
				}
			}

	updatelist <- list("L" = L_mat, "Lambda" = Lambda_mat) 
	return(updatelist)
}

 
###############################		
# conditional density of Y_t
###############################
log_dsty_Y <- function(Y_t, Lambda_mat){
	s <- 0
	for(d in 1:D)
		for(i in 1:p)
			for(j in i:p){
				tmp <- sapply(1:R, function(r){ dpois(Y_t[d, r, i, j], lambda = Lambda_mat[d, i, j], log = TRUE) } )
				s <- s  + sum(tmp)
			}
 
	

	return(s)
}

###############################		
#sample cla
###############################	
sample_cla <- function(Lambda_t, P_dp_t, Y_t, n, M){


	ss <- sapply( 1:M, function(m){
					log_dsty_Y(Y_t[,,n,,], Lambda_t[m,,,])
				  } )
	ss_max <- max(ss)
	ss_new <- ss - ss_max
	Prob <- P_dp_t * exp(ss_new)  
	tmp <- sample(1:M, 1, prob=Prob)	

	return(tmp)


}

###############################		
#sample P_dp
###############################

sample_P_dp <- function(cla_t, alpha, M){
	M_arr <- sapply(1:M, function(m){sum(cla_t == m)})
	V_arr <- sapply(1:(M-1), function(i){rbeta(1, shape1 = alpha/M + M_arr[i],
							shape2 = alpha/M*(M - i) + sum(M_arr[(i+1):M]))} )
	P_dp <- rep(NA, M-1)
	
	for(i in 1:(M-1)){
		if(i == 1) {
			P_dp[i] <- V_arr[i]
		}else P_dp[i] <- ( 1 - sum(P_dp[1:(i-1)]) )*V_arr[i]
	}
	
	return( c(P_dp, 1 - sum(P_dp)) )
}

###############################		
#sample Lambda_m
###############################


sample_L_m <- function(Lambda_t, par_p_t, m){
	#par_p_t is fixed
	#m is the table number
	#Lambda_t is the underlying p by p lambda matrix

	L <- array(NA, dim = c( D, p, p))

		for(d in 1:D) {
			for(i in 1:(p-1)) {
				for(j in (i+1):p) {

					tmp <- dgamma(Lambda_t[m, d, i, j], shape = c(par_1[1], par_0[1]), rate = c(par_1[2], 
								par_0[2]), log = TRUE)
					if(sum(tmp == -Inf) == 2){
						tmp_new <- c(1/2,1/2)
					}else{
						tmp_max <- max(tmp)
						tmp_new <- tmp - tmp_max
					}	
					prob <- c(par_p_t, 1 - par_p_t)*exp(tmp_new)
					L[ d, i, j] <- sample(c(1,0), 1, prob = prob)
					L[ d, j, i] <- L[d, i, j]
				}
			}
		}
	
	
	return(L)
}
							
sample_Lambda_m <- function(Y_t, Lambda_t, L_t, cla_t,m,  par_0, par_1, par_2){
 	
	Lambda <- array(NA, dim = c(D, p, p))

		ind <- which(cla_t == m)
		total <- length(ind)
		for(d in 1:D)
			for(i in 1:p)
				for(j in i:p){
					if(j==i){
						tmp <- sum(Y_t[d,,ind,i,i])
						Lambda[ d,i,i] <- rgamma(1,shape=par_2[1]+tmp, rate=par_2[2]+total*R)
					}else{
						tmp <- sum(Y_t[d,,ind,i,j])
						if(L_t[m ,d, i, j] == 1){
							Lambda[d,i,j] <- rgamma(1,shape=par_1[1]+tmp, rate=par_1[2]+total*R)
						}else{
							Lambda[ d,i,j] <- rgamma(1,shape=par_0[1]+tmp, rate=par_0[2]+total*R)
						}

						Lambda[d,j,i] <- Lambda[d,i,j]
					}
				}
	

	
	return(Lambda)
	
}

###############################
#sample Y[,,n,,] a via Metropolis-Hasting step
###############################
#proposal
Proposal <- function(Y_old) {
	Y_star <- Y_old
	ind <- sort(sample(1:p,2))
	r <- runif(1)
	if(r <= 1/2){
		Y_star[ind[1],ind[1]] <- Y_star[ind[1],ind[1]] - 1
		Y_star[ind[1],ind[2]] <- Y_star[ind[1],ind[2]] + 1
		Y_star[ind[2],ind[1]] <- Y_star[ind[1],ind[2]]
		Y_star[ind[2],ind[2]] <- Y_star[ind[2],ind[2]] - 1
	}else{
		Y_star[ind[1],ind[1]] <- Y_star[ind[1],ind[1]] + 1
		Y_star[ind[1],ind[2]] <- Y_star[ind[1],ind[2]] - 1
		Y_star[ind[2],ind[1]] <- Y_star[ind[1],ind[2]]
		Y_star[ind[2],ind[2]] <- Y_star[ind[2],ind[2]] + 1

	}	


	if(sum(Y_star[ind,ind]<0)>0){
		return("FALSE")
	}else return(Y_star)
}

post_dratio <- function(d, r, n,  Y_proposal, Y, Lambda,class) {
	s <- 0
	for(i in 1:p){
		for(j in i:p) {
			s <- s + log(Lambda[class,d,i,j]) * (Y_proposal[i,j]-Y[d,r,n,i,j]) +
				 lfactorial(Y[d,r,n,i,j]) - lfactorial(Y_proposal[i,j])

			
		}
	}
	return(exp(s))
}




sample_Y <- function(d, r, n, Lambda, cla,  Y) {
	Y_proposal <- Proposal(Y[d,r,n,,])
	if(is.matrix(Y_proposal) == FALSE) {
		return(Y[d,r,n,,])
	} else {
		
		ratio <- min( post_dratio(d,r,n,Y_proposal,Y,Lambda,cla[n]) ,1)
      	tmp <- sample(c(0,1),1,prob = c(1-ratio , ratio))
		if(tmp == 1) {
	
			return(Y_proposal)
		} else {
			return(Y[d,r,n,,])
		}
	}
}


#install parallel package 

library(parallel)
		
###############################
#parameters in the priors
###############################
alpha <- 2
M <- 5


par_0 <- c(2,20)
par_1 <- c(2,1)
par_2 <- c(3,1)

###############################
#initial values for parameters
###############################

P_dp_t <- Dirich(rep(alpha/M,M))
par_p_t <- 1/4
Lambda_t <- array( NA, dim = c(M, D, p, p) )
L_t <- array(NA, dim = c(M, D, p, p))

for(m in 1:M){
	Lambda_t[m,,,] <- sample_H(par_0, par_1, par_2, par_p_t)$Lambda
	L_t[m,,,] <- sample_H(par_0, par_1, par_2, par_p_t)$L 
}


cla_t <- sample(1:M, N, replace = TRUE)

Y_t <- array(0, dim = c(D, R, N, p, p))
for(d in 1:D) {
		for(n in 1:N) {
			for(i in 1:p) {
				Y_t[d, ,n,i,i] <- X[d, ,n,i]
			}
		}
}




##########################################################################
# Sampling
#
# Use "mclapply" function to parallelize the Gibbs sampler
# for Hierarchical Dynamic Poisson Graphical Models (HDPGM)
##########################################################################
mc = 20 #number of cores to carry out parallel computing

T <- 40000

lgth <- N/mc
t1 <- Sys.time()

S <- 20000

K_rec <- rep(NA, S)
cla_t_rec <- array(NA, dim = c(S, N))
L_t_rec <- array(NA, dim = c(S, M, D, p, p))
Lambda_t_rec <- array(NA, dim = c(S, M, D, p, p))
P_dp_rec <- array(NA, dim = c(S,M))



for( t in 1:T) {


	#sample network structures and intensity parameters
	cla_star <- unique(cla_t)
	cla_diff <- setdiff(1:M, cla_star)
	for(m in cla_diff) {
		Lambda_t[m,,,] <- sample_H(par_0, par_1, par_2, par_p_t)$Lambda
		L_t[m,,,] <- sample_H(par_0, par_1, par_2, par_p_t)$L
	}
	
	for(m in cla_star) {

		L_t[m,,,] <- sample_L_m(Lambda_t, par_p_t, m)
		Lambda_t[m,,,] <- sample_Lambda_m(Y_t, Lambda_t, L_t, cla_t, m, par_0, par_1, par_2)
	}

	#sample class membership
	lists_cla <- mclapply(1:mc,function(i){
		cla_t_core <- cla_t[(lgth*(i-1)+1):(lgth*i)]
		Y_t_core <- Y_t[,,(lgth*(i-1)+1):(lgth*i),,]
		P_dp_t_core <- P_dp_t 
		Lambda_t_core <- Lambda_t
		tmp <- NULL
		for(n in 1:lgth) {

			#sample cla
			tmp<-c(tmp,sample_cla(Lambda_t_core, P_dp_t_core, Y_t_core,n,M))
		}
		tmp		
	}, mc.cores=mc )
	
	for(i in 1:mc){
		cla_t[(lgth*(i-1)+1):(lgth*i)] <- lists_cla[[i]]
	
	}


	#sample latent counts
	for(d in 1:D){
		for(r in 1:R){

			lists_Y <- mclapply(1:mc,function(i){
				cla_t_core <- cla_t[(lgth*(i-1)+1):(lgth*i)]
				Y_t_core <- Y_t[,,(lgth*(i-1)+1):(lgth*i),,]
		
				Lambda_t_core <- Lambda_t
		
				tmp <- NULL
				for(n in 1:lgth) {

					#sample cla
					tmp<-cbind(tmp,as.matrix(sample_Y(d,r,n, Lambda_t_core, cla_t_core, Y_t_core)))
				}
				tmp		
			}, mc.cores=mc )


			for(i in 1:mc)
				for(j in (lgth*(i-1)+1):(lgth*i)) 
					Y_t[d,r,j,,] <- lists_Y[[i]][,((j-lgth*(i-1)-1)*p+1):((j-lgth*(i-1))*p)]

		}
	}


	#sample cluster proportion
	P_dp_t <- sample_P_dp(cla_t, alpha, M)

	if(t > T - S){
		K_rec[t - (T-S)] <- length(table(cla_t))
		cla_t_rec[t - (T-S), ] <- cla_t 

		L_t_rec[t - (T-S) , , , , ] <- L_t
		Lambda_t_rec[t - (T-S) , , , , ] <- Lambda_t
		P_dp_rec[t - (T-S), ] <- P_dp_t
	}



	print("====================================================")
	print(t)
	
	print(table(cla_t))
	
	for(m in 1:M){
		for( d in 1:D){
			print(L_t[m,d,,])
		}
	}
                                                                                                                                                                                                                                                                                                                          
}

t2 <- Sys.time()
print("time used")
t2 - t1


