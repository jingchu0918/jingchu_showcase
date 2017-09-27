//////////////////////////////////////////////////////////////////////////
/*
/* Heterogeneous Dynamic Poisson Graphical Model
/* Auther: Jing Chu
/* Date: June 2017
/*
//////////////////////////////////////////////////////////////////////////
/*
/* This code randomly generate a sample for heterogeneous dynamic
/* transcription factor networks and implemented a Hierarchical Dynamic
/* Poisson Graphical Model for nonparametric Bayesian learning of the
/* network. In particular, an efficient parallel Markov chain Monte Carlo
/* algorithm for posterior computation is developed
/*
//////////////////////////////////////////////////////////////////////////
/*
/* Message Passing Interface (MPI) is used for parallel computing
/*
//////////////////////////////////////////////////////////////////////////



#include <mpi.h>
#include <string.h>

#define N 2000
/* number of genomic locations */
#define D 3
/* number of conditions */
#define R 2
/* number of replicates */
#define K 2
/* number of clusters */
#define p 7
/* number of nodes */
#define M 5
#define T 10000
/*T=40000*/
#define S 2000
/*S=20000*/
#define mc 20
/*number of cores to carry out parallel computing*/
#define lgth N/mc

/* functions */

int poissonRandom(double expectedValue) {
    double limit = exp(-expectedValue);
    int k = 0; /*count of iteration*/
    double tmp=1.0;  /*pseudo random number*/
    double u;
    do {
        k+=1;
        u = (double)rand()/(double)(RAND_MAX/1);
        tmp *=u;
    } while (tmp>limit);
    return(k-1);
}

double logfactorial(int x){
    int i;
    double tmp=0.0;
    if(x<=0){
        return 1;
    }
    else{
        for (i=1;i<=x;i++){
            tmp += log(i);
        }
        return tmp;
    }
}

double logpoissonDensity(int x,double lambda){
    return x*log(lambda)-lambda-logfactorial(x);
}

int bernoulliRandom(double prob) {
    if(prob < 0 || prob > 1) return -1;
    double u = (double)rand()/(double)(RAND_MAX/1);
    if(u < prob) return 1;
    return 0;
}

double exponentialRandom(double lambda){
    double u = (double)rand()/(double)(RAND_MAX/1);
    return (-log(u)/lambda);
}

double gammaRandom(double shape, double scale){
    int i,n=shape;
    double delta=shape-n,tmp=0;
    for (i=0;i<n;i++){
        tmp+=exponentialRandom(1.0);
    }
    double eta,xi=0,U,V,W;
    
    if (delta>0.0){
        do{
            U = (double)rand()/(double)(RAND_MAX/1);
            V = (double)rand()/(double)(RAND_MAX/1);
            W = (double)rand()/(double)(RAND_MAX/1);
            
            if (U<=exp(1)/(exp(1)+delta)) {
                xi=pow(V,1/delta);
                eta=W*pow(xi,delta-1);
            }
            else {
                xi=1-log(V);
                eta=W*exp(-xi);
            }
        } while (eta>pow(xi,delta-1)*pow(exp(1),-xi));
    }
    return scale*(xi+tmp);
}


double loggammaDensity(double x,double shape, double rate){
    return shape*log(rate)+(shape-1)*log(x)-rate*x-log(tgammaf(shape));
}

double betaRandom(double alpha, double beta){
    double x=gammaRandom(alpha, 1.0);
    double y=gammaRandom(beta, 1.0);
    return x/(x+y);
    
}

/* sample c(1:n) with prob[n] */
int discreteRandom(int n, double prob_original[n]){
    double sum=0;
    int i,ans=0;
    double prob[n];
    for (i=0;i<n;i++){sum+=prob_original[i];}
    for (i=0;i<n;i++) {prob[i]=prob_original[i]/sum;}
    double cum_prob[n+1];
    cum_prob[0]=0;
    for (i=1;i<n;i++){
        cum_prob[i]=cum_prob[i-1]+prob[i-1];
    }
    cum_prob[n]=1;
    double u = (double)rand()/(double)(RAND_MAX/1);
    for (i=0;i<n;i++){
        if ((u>=cum_prob[i])&&(u<=cum_prob[i+1])) {
            ans=i+1;
            break;
        }
    }
    return ans;
}


/*sample from a Dirichlet distribution*/
void Dirich(double alpha_prob[M],double P_dirich[M]){
    double y[M],sum=0;
    int i;
    for (i=0;i<M;i++){
        y[i]=gammaRandom(alpha_prob[i], 1);
        sum+=y[i];
    }
    for (i=0;i<M;i++){
        P_dirich[i]=y[i]/sum;
    }
}

/* sample from base distribution H */
void sample_H(double par_0[2],double par_1[2],double par_2[2],double par_p_t,int L[D][p][p],double Lambda[D][p][p]){
    /* par_0 is the parameter vector when there is no edge
     par_1 is the parameter vector when there is an edge
     par_2 is the parameter vector associated with the node
     par_p_t is the parameter indicating the probability of the presence of an edge */
    int d,i,j;
    for (d=0;d<D;d++){
        for (i=0;i<p;i++){
            for (j=i;j<p;j++){
                if (j==i) {Lambda[d][i][j]=gammaRandom(par_2[0], 1.0/par_2[1]);
                    L[d][i][i]=2;}
                else {
                    L[d][i][j]=bernoulliRandom(par_p_t);
                    if (L[d][i][j]==1){Lambda[d][i][j]=gammaRandom(par_1[0],1.0/par_1[1]);}
                    else {Lambda[d][i][j]=gammaRandom(par_0[0],1.0/par_0[1]);}
                    L[d][j][i]=L[d][i][j];
                    Lambda[d][j][i]=Lambda[d][i][j];
                }
            }
        }
    }
}


/* conditional density of Y_t */
double log_dsty_Y(int Y_t[D][R][p][p], double Lambda_mat[D][p][p]){
    int d,i,j,r;
    double s,tmp;
    s=0.0;
    
    for (d=0;d<D;d++){
        for (i=0;i<p;i++){
            for (j=i;j<p;j++){
                for (r=0;r<R;r++){
                    tmp=logpoissonDensity(Y_t[d][r][i][j],Lambda_mat[d][i][j]);
                    s=s+tmp;
                }
                
            }
        }
        
    }
    
    return(s);
}

double maximum(double array[],int size)
{
    int i;
    double max;
    max=array[0];
    for (i=1;i<size;i++){
        if (array[i]>max){max=array[i];}
    }
    return max;
}

/* sample  cla and returns an integer from 1:M */
int sample_cla(double Lambda_t[M][D][p][p],double P_dp_t[M],int Y_t[D][R][lgth][p][p],int n){
    int m,d,i,j,r;
    int tmp;
    double ss[M],ss_new[M],Prob[M];
    double ss_max;
    double Lambda_t_input[D][p][p];
    int Y_t_input[D][R][p][p];
    for (d=0;d<D;d++){
        for (r=0;r<R;r++){
            for (i=0;i<p;i++){
                for (j=0;j<p;j++){
                    Y_t_input[d][r][i][j]=Y_t[d][r][n][i][j];
                }
            }
        }
    }
    for (m=0;m<M;m++){
        for (d=0;d<D;d++){
            for (i=0;i<p;i++){
                for (j=0;j<p;j++){
                    Lambda_t_input[d][i][j]=Lambda_t[m][d][i][j];
                }
            }
        }
        
        ss[m]=log_dsty_Y(Y_t_input,Lambda_t_input);
    }
    
    ss_max=maximum(ss,M);
    for (m=0;m<M;m++){
        ss_new[m]=ss[m]-ss_max;
        Prob[m]=P_dp_t[m]*exp(ss_new[m]);
    }
    
    tmp=discreteRandom(M,Prob);
    return(tmp);
}



/* sample P_dp */
void sample_P_dp(int cla_t[N],double alpha,double P_dp[M]){
    int M_arr[M]={0};
    double V_arr[M-1],sum;
    int m,n,i,j;
    for (n=0;n<N;n++){
        for (m=1;m<=M;m++){
            if (cla_t[n]==m) {
                M_arr[m-1]+=1;
                break;
            }
        }
    }
    
    
    for (i=0;i<M-1;i++){
        sum=0.0;
        for (j=i+1;j<M;j++){sum+=M_arr[j];}
        V_arr[i]=betaRandom(alpha/M+M_arr[i],alpha/M*(M-i-1)+sum);
    }
    P_dp[0]=V_arr[0];
    for (i=1;i<M-1;i++){
        sum=0;
        for (j=0;j<i;j++) {sum+=P_dp[j];}
        P_dp[i]=(1-sum)*V_arr[i];
    }
    sum=0;
    for (i=0;i<M-1;i++){sum += P_dp[i];}
    P_dp[M-1]=1-sum;
}





/* sample L_m  m from 1:M */
void sample_L_m(double Lambda_t[M][D][p][p],double par_p_t,int m,double par_0[2],double par_1[2],int L[D][p][p])
{
    /* par_p_t is fixed */
    /* m is the table number */
    /* Lambda_t is the underlying p by p lambda matrix */
    int d,i,j;
    double tmp[2],tmp_new[2],tmp_max,prob[2],sum;
    
    
    for (d=0;d<D;d++){
        L[d][p-1][p-1]=2;
        for (i=0;i<p-1;i++){
            for (j=i;j<p;j++){
                if (j==i)  {L[d][i][j]=2;}
                else{
                    tmp[0]=loggammaDensity(Lambda_t[m-1][d][i][j], par_1[0], par_1[1]);
                    tmp[1]=loggammaDensity(Lambda_t[m-1][d][i][j], par_0[0], par_0[1]);
                    if ((isinf(tmp[0])!=0)&&(isinf(tmp[1])!=0)) {tmp_new[0]=0.5;tmp_new[1]=0.5;}
                    else {
                        tmp_max=maximum(tmp,2);
                        tmp_new[0]=tmp[0]-tmp_max;
                        tmp_new[1]=tmp[1]-tmp_max;
                    }
                    prob[0]=par_p_t*exp(tmp_new[0]);
                    prob[1]=(1.0-par_p_t)*exp(tmp_new[1]);
                    sum=prob[0]+prob[1];
                    prob[0]=prob[0]/sum;
                    prob[1]=prob[1]/sum;
                    L[d][i][j]=bernoulliRandom(prob[0]);
                    L[d][j][i]=L[d][i][j];
                }
            }
        }
    }
    
}

// m>>1:M //
void sample_Lambda_m(int Y_t[D][R][N][p][p],int L_t[M][D][p][p],int cla_t[N],int m,double par_0[2],double par_1[2],double par_2[2],double Lambda[D][p][p])
{
    int total=0,i,d,j,k,tmp;
    int ind[N]={0};
    for (i=0;i<N;i++){
        if (cla_t[i]==m) {
            ind[total]=i;
            total+=1;
        }
    }
    
    for (d=0;d<D;d++){
        for (i=0;i<p;i++){
            for (j=i;j<p;j++){
                tmp=0;
                for (k=0;k<total;k++) {tmp +=Y_t[d][0][ind[k]][i][j]+Y_t[d][1][ind[k]][i][j];}
                if (j==i){
                    Lambda[d][i][i]=gammaRandom(par_2[0]+tmp,1.0/(par_2[1]+total*R));
                }else {
                    if (L_t[m-1][d][i][j]==1){
                        Lambda[d][i][j]=gammaRandom(par_1[0]+tmp,1.0/(par_1[1]+total*R));
                    }else {
                        Lambda[d][i][j]=gammaRandom(par_0[0]+tmp,1.0/(par_0[1]+total*R));
                    }
                    Lambda[d][j][i]=Lambda[d][i][j];
                }
            }
        }
    }
}



/* sample Y[,,n,,] a via Metropolis-Hasting step */
/* proposal */
int Proposal(int Y_old[p][p],int Y_star[p][p]){  /*mark=1 means false;mark=0 means update */
    int i,j;
    int mark=0;
    for (i=0;i<p;i++){
        for (j=0;j<p;j++){
            Y_star[i][j]=Y_old[i][j];
        }
    }
    
    int ind[2],t_ind;
    double prob[p];
    for (i=0;i<p;i++){prob[i]=1.0/p;}
    
    do{
        ind[0]=discreteRandom(p,prob)-1;
        ind[1]=discreteRandom(p,prob)-1;
    }while (ind[0]==ind[1]);
    
    if (ind[0]>ind[1]) {
        t_ind=ind[0];
        ind[0]=ind[1];
        ind[1]=t_ind;
    }
    double r;
    r=(double)rand()/(double)(RAND_MAX/1);
    if (r<=0.5){
        Y_star[ind[0]][ind[0]] = Y_star[ind[0]][ind[0]] - 1;
        Y_star[ind[0]][ind[1]] = Y_star[ind[0]][ind[1]] + 1;
        Y_star[ind[1]][ind[0]] = Y_star[ind[0]][ind[1]];
        Y_star[ind[1]][ind[1]] = Y_star[ind[1]][ind[1]] - 1;
    }
    else {
        Y_star[ind[0]][ind[0]] = Y_star[ind[0]][ind[0]] + 1;
        Y_star[ind[0]][ind[1]] = Y_star[ind[0]][ind[1]] - 1;
        Y_star[ind[1]][ind[0]] = Y_star[ind[0]][ind[1]];
        Y_star[ind[1]][ind[1]] = Y_star[ind[1]][ind[1]] + 1;
    }
    for (i=0;i<2;i++){
        for (j=0;j<2;j++){
            if (Y_star[ind[i]][ind[j]]<0) {mark=1;}
        }
    }
    return mark;
}


// class>>1:M //
double post_dratio(int d,int r,int n,int Y_proposal[p][p],int Y[D][R][lgth][p][p],double Lambda[M][D][p][p],int class){
    int i,j;
    double s=0.0;
    for (i=0;i<p;i++){
        for (j=i;j<p;j++){
            s=s+log(Lambda[class-1][d][i][j])*(Y_proposal[i][j]-Y[d][r][n][i][j])+logfactorial(Y[d][r][n][i][j])-logfactorial(Y_proposal[i][j]);
        }
    }
    return exp(s);
}

// n>>0:lgth-1 //
void sample_Y(int d,int r,int n,double Lambda[M][D][p][p],int cla[lgth],int Y[D][R][lgth][p][p],int Y_proposal[p][p]){
    int Y_input[p][p],Y_proposal_output[p][p],mark=0;
    int i,j;
    for (i=0;i<p;i++){
        for (j=0;j<p;j++){
            Y_input[i][j]=Y[d][r][n][i][j];
        }
    }
    
    
    mark=Proposal(Y_input,Y_proposal_output);  /*mark=1 means false;mark=0 means update */
    
    int tmp;
    double ratio;
    if (mark==1) {
        for (i=0;i<p;i++){
            for (j=0;j<p;j++){
                Y_proposal[i][j]=Y_input[i][j];
            }
        }
    }
    
    else {
        ratio=post_dratio(d,r,n,Y_proposal_output,Y,Lambda,cla[n]);
        if (ratio>1.0) {ratio=1.0;}
        tmp=bernoulliRandom(ratio);
        if (tmp==0){
            for (i=0;i<p;i++){
                for (j=0;j<p;j++){
                    Y_proposal[i][j]=Y_input[i][j];
                }
            }
        }
        else {
            for (i=0;i<p;i++){
                for (j=0;j<p;j++){
                    Y_proposal[i][j]=Y_proposal_output[i][j];
                }
            }
            
        }
    }
}



int unique(int cla_t[N],int cla_star[]){
    int i,j,mark,len;
    cla_star[0]=cla_t[0];
    len=1;
    for (i=1;i<N;i++){
        mark=0;
        for (j=0;j<len;j++){
            if (cla_star[j]==cla_t[i]){
                mark=1;
                break;
            }
        }
        if (mark==0) {
            cla_star[len]=cla_t[i];
            len+=1;
        }
    }
    return len;
}




int main(void) {
    MPI_Init(NULL,NULL);
    srand(28);
    
    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    double Pi[K];
    /* Pi[k] represents the probability that one object is assigned to cluster k */
    double Lambda[D][K][p][p];
    /* Lambda[ d, k,,] represents the probability structure in cluster k under condition d */
    int cla[N];
    //cla[n] shows the latent class (cluster) membership of object n
    int Y[D][R][N][p][p];
    /* Y[d,r,n,,] denotes the latent read counts for object n in r-th replicate under condition d */
    int X[D][R][N][p];
    /* X[d,r,n,] denotes the observed read counts for object n in r-th replicate under condition d */
    /* Generate dataset (K clusters, R replicates, D conditions) */
    int cla_t[N];
    int Y_t[D][R][N][p][p]={0};
    int K_rec[S];
    int cla_t_rec[S][N];
    int L_t_rec[S][M][D][p][p];
    double Lambda_t[M][D][p][p],updatelist_Lambda[D][p][p];
    int L_t[M][D][p][p],updatelist_L[D][p][p];
    
    double Lambda_t_rec[S][M][D][p][p];
    double P_dp_rec[S][M];
    
    double alpha=2.0;
    double alpha_prob[M];
    double par_0[2]={2.0,20.0};
    double par_1[2]={2.0,1.0};
    double par_2[2]={3.0,1.0};
    double P_dp_t[M];
    double par_p_t=1.0/4;
    int len=0,mark;
    int sample_L_m_output[D][p][p];
    double sample_Lambda_m_output[D][p][p];
    int cla_star[N];
    
    
    int Y_proposal[p][p];
    
    int cla_t_core[lgth];
    int Y_t_core[D][R][lgth][p][p];
    
    
    int i,j,m,n,d,r,t,nthread;
    
    //////////////*** Generate dataset (K clusters, R replicates, D conditions) ***///////////////

    
    ///////initial values in rank==0///////////
    
    if (world_rank==0) {
        
        Pi[0]=0.6;
        Pi[1]=0.4;
    
        /* original values of Lambda */
        FILE* fp;
        fp=fopen("Lambda.txt","r");
        for (j=0;j<K;j++){
            for (i=0;i<D;i++){
                for (m=0;m<p;m++){
                    for (n=0;n<p;n++){
                        fscanf(fp,"%lf",&Lambda[i][j][m][n]);
                    }
                    fscanf(fp,"\n");
                }
                fscanf(fp,"\n");
            }
        }
        fclose(fp);
    
        /* generate X */
        for (n=0;n<N;n++){
            cla[n]=discreteRandom(K,Pi);
            for (d=0;d<D;d++){
                for (i=0;i<p;i++){
                    for (j=i;j<p;j++){
                        for (r=0;r<R;r++){
                            Y[d][r][n][i][j]=poissonRandom(Lambda[d][cla[n]-1][i][j]);
                            if (j != i ) {Y[d][r][n][j][i]=Y[d][r][n][i][j];}
                        }
                    }
                    for (r=0;r<R;r++) {
                        X[d][r][n][i]=0;
                        for (j=0;j<p;j++) {X[d][r][n][i]+=Y[d][r][n][i][j];}
                    }
                }
            }
        }
        
        /*Dirichlet Process Mixture Model (Sampling via Block Gibbs Sampler)*/
        
        /*parameters in the priors*/
        
        for (i=0;i<M;i++){
            alpha_prob[i]=2.0 / M;
        }
        
        /*initial values for parameters */
        Dirich(alpha_prob,P_dp_t);
        memset(L_t,2,sizeof(L_t));
        for (m=0;m<M;m++){
            sample_H(par_0,par_1,par_2,par_p_t,updatelist_L,updatelist_Lambda);
            for (d=0;d<D;d++){
                for (i=0;i<p;i++){
                    for (j=0;j<p;j++){
                        Lambda_t[m][d][i][j]=updatelist_Lambda[d][i][j];
                        L_t[m][d][i][j]=updatelist_L[d][i][j];
                    }
                }
            }
        }
        
        double prob[M];
        for (m=0;m<M;m++){prob[m]=1.0/M;}
        for (n=0;n<N;n++){
            cla_t[n]=discreteRandom(M, prob);
        }
        
        for (d=0;d<D;d++){
            for (n=0;n<N;n++){
                for (i=0;i<p;i++){
                    for (r=0;r<R;r++){
                        Y_t[d][r][n][i][i]=X[d][r][n][i];
                    }
                }
            }
        }
        
        /////////////send Y_t_core and cla_t_core/////
        for (nthread=1;nthread<world_size;nthread++){
            
            for (n=lgth*nthread;n<lgth*(nthread+1);n++){
                for (d=0;d<D;d++){
                    for (r=0;r<R;r++){
                        for (i=0;i<p;i++){
                            for (j=0;j<p;j++){
                                Y_t_core[d][r][n-lgth*nthread][i][j]=Y_t[d][r][n][i][j];
                            }
                        }
                    }
                }
            }
            MPI_Send(&(cla_t[lgth*nthread]),lgth,MPI_INT,nthread,222,MPI_COMM_WORLD);
            MPI_Send(&(Y_t_core[0][0][0][0][0]),D*R*lgth*p*p,MPI_INT,nthread,333,MPI_COMM_WORLD);
        
        }
        //initial cla_t_core in 0//
        for (n=0;n<lgth;n++){cla_t_core[n]=cla_t[n];}
    }
    else {
        //receive cla_t_core//
        MPI_Recv(&(cla_t_core[0]),lgth,MPI_INT,0,222,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //receive Y_t_core//
        MPI_Recv(&(Y_t_core[0][0][0][0][0]), D*R*lgth*p*p, MPI_INT, 0, 333, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    
    /////////////////////*************  sampling ***************///////////////////////////////////////////
    
    
    
    for (t=0;t<T;t++){
        if (world_rank==0){
            
            
            //initial Y_t_core in 0//
            for (d=0;d<D;d++){
                for (r=0;r<R;r++){
                    for (n=0;n<lgth;n++){
                        for (i=0;i<p;i++){
                            for (j=0;j<p;j++){
                                Y_t_core[d][r][n][i][j]=Y_t[d][r][n][i][j];
                            }
                        }
                    }
                }
            }
            
            
            /* sample network structures and intensity parameters Lambda_t & L_t*/
            len=unique(cla_t,cla_star);
            for (m=1;m<=M;m++){
                mark=0;
                for (n=0;n<len;n++){
                    if (cla_star[n]==m){
                        mark=1;
                        break;}
                }
                if (mark==0) {                   //mark==0 means m is in cla_diff//
                    sample_H(par_0,par_1,par_2,par_p_t,updatelist_L,updatelist_Lambda);
                    for (d=0;d<D;d++){
                        for (i=0;i<p;i++){
                            for (j=0;j<p;j++){
                                Lambda_t[m-1][d][i][j]=updatelist_Lambda[d][i][j];
                                L_t[m-1][d][i][j]=updatelist_L[d][i][j];
                            }
                        }
                    }
                    
                }else {            //mark==1 means m is in cla_star//
                    sample_L_m(Lambda_t,par_p_t,m,par_0,par_1,sample_L_m_output);
                    for (d=0;d<D;d++){
                        for (i=0;i<p;i++){
                            for (j=0;j<p;j++){
                                L_t[m-1][d][i][j]=sample_L_m_output[d][i][j];
                            }
                        }
                    }
                    
                    
                    sample_Lambda_m(Y_t,L_t,cla_t,m,par_0,par_1,par_2,sample_Lambda_m_output);
                    for (d=0;d<D;d++){
                        for (i=0;i<p;i++){
                            for (j=0;j<p;j++){
                                Lambda_t[m-1][d][i][j]=sample_Lambda_m_output[d][i][j];
                            }
                        }
                    }
                    
                }
            }
            
            /////send Lambda_t & P_dp_t from 0///
            for (nthread=1;nthread<world_size;nthread++){
                MPI_Send(&(P_dp_t[0]),M,MPI_DOUBLE,nthread,000,MPI_COMM_WORLD);
                MPI_Send(&(Lambda_t[0][0][0][0]),M*D*p*p,MPI_DOUBLE,nthread,111,MPI_COMM_WORLD);
            }
        
        }
        
        
        else {
            //receive P_dp_t//
            MPI_Recv(&(P_dp_t[0]), M, MPI_DOUBLE, 0, 000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //receive Lambda_t//
            MPI_Recv(&(Lambda_t[0][0][0][0]), M*D*p*p,MPI_DOUBLE,0, 111, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        /* sample class membership cla_t_core*/
        for (n=0;n<lgth;n++){
            cla_t_core[n]=sample_cla(Lambda_t,P_dp_t,Y_t_core,n);
        }
        
        
        
        
        /* sample latent counts Y_t_core*/
        for (d=0;d<D;d++){
            for (r=0;r<R;r++){
                for (n=0;n<lgth;n++){
                    sample_Y(d,r,n,Lambda_t,cla_t_core,Y_t_core,Y_proposal);
                    for (i=0;i<p;i++){
                        for (j=0;j<p;j++){
                            Y_t_core[d][r][n][i][j]=Y_proposal[i][j];
                        }
                    }
                }
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        
        //send back updated cla_t_core & Y_t_core to 0//
        if (world_rank==0){
            
            //update cla_t and Y_t for  rank=0//
            for (n=0;n<lgth;n++){cla_t[n]=cla_t_core[n];}
            for (d=0;d<D;d++){
                for (r=0;r<R;r++){
                    for (n=0;n<lgth;n++){
                        for (i=0;i<p;i++){
                            for (j=0;j<p;j++){
                                Y_t[d][r][n][i][j]=Y_t_core[d][r][n][i][j];
                            }
                        }
                    }
                }
            }
            
            for (nthread=1;nthread<world_size;nthread++){
                //receive updated cla_t_core//
                MPI_Recv(&(cla_t[nthread*lgth]), lgth, MPI_INT, nthread, 444, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //receive updated Y_t_core//
                MPI_Recv(&(Y_t_core[0][0][0][0][0]), D*R*lgth*p*p, MPI_INT, nthread, 555, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //update cla_t & Y_t in 0//
                for (n=lgth*nthread;n<lgth*(nthread+1);n++){
                    for (d=0;d<D;d++){
                        for (r=0;r<R;r++){
                            for (i=0;i<p;i++){
                                for (j=0;j<p;j++){
                                    Y_t[d][r][n][i][j]=Y_t_core[d][r][n-lgth*nthread][i][j];
                                }
                            }
                        }
                    }
                }
            }
            
            
        }
        else {
            MPI_Send(&(cla_t_core[0]),lgth, MPI_INT, 0, 444, MPI_COMM_WORLD);
            MPI_Send(&(Y_t_core[0][0][0][0][0]), D*R*lgth*p*p, MPI_INT, 0, 555,MPI_COMM_WORLD);
        }
        
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        //update P_dp_t//
        if (world_rank==0){
    
//            printf("%d updated cla_t:",t);
//            for (n=0;n<N;n++){
//                printf("%d",cla_t[n]);
//                
//            }
//            printf("\n");
//            
//            for (m=0;m<M;m++){
//                for (d=0;d<D;d++){
//                    for (i=0;i<p;i++){
//                        for (j=0;j<p;j++){printf("%d ",L_t[m][d][i][j]);}
//                        printf("\n");
//                    }
//                    printf("\n");
//                }
//                printf("\n");
//            }
//            
            
            /* sample cluster proportion */
            sample_P_dp(cla_t,alpha,P_dp_t);
            
            if (t>=T-S) {
                len=unique(cla_t,cla_star);
                K_rec[t-(T-S)]=len;
                for (n=0;n<N;n++){
                    cla_t_rec[t-(T-S)][n]=cla_t[n];}
                for (m=0;m<M;m++){
                    for (d=0;d<D;d++){
                        for (i=0;i<p;i++){
                            for (j=0;j<p;j++){
                                L_t_rec[t-(T-S)][m][d][i][j]=L_t[m][d][i][j];
                                Lambda_t_rec[t-(T-S)][m][d][i][j]=Lambda_t[m][d][i][j];
                            }
                        }
                    }
                }
                for (m=0;m<M;m++){
                    P_dp_rec[t-(T-S)][m]=P_dp_t[m];
                }
                
                
                
//                printf("%d updated cla_t:",t);
//                for (n=0;n<N;n++){
//                    printf("%d",cla_t[n]);
//                    
//                }
//                printf("\n");
//                
//                for (m=0;m<M;m++){
//                    for (d=0;d<D;d++){
//                        for (i=0;i<p;i++){
//                            for (j=0;j<p;j++){printf("%d ",L_t[m][d][i][j]);}
//                            printf("\n");
//                        }
//                        printf("\n");
//                    }
//                    printf("\n");
//                }
                
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
    }
    
    
    
    
    MPI_Finalize();
    return 0;
}
