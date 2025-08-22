#include <iostream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "potentials/biased_kicker.hpp"
#include "potentials/random_kicker.hpp"
#include "potentials/inversepower_potential.hpp"


/*
 * SIP:Mean and variance tests for three particles
 */

class SIPMeanVarianceTestThreeParticles :  public ::testing::Test
{
public:
    std::vector<double> x = {-1.0,0.0, 0.0,1.0, 1.0,0.0}, radii = {sqrt(2)-1,sqrt(2)+1,sqrt(2)-1};
    double tol = 0.05;
    std::vector<double> d,avg;
    std::vector<double> boxv = {20.0,20.0};
    size_t natoms = 3, ndim = 2 , n_repeat = 100000;
    virtual void SetUp(){ 	
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif
        d.resize(natoms*ndim);
    }
    //-------------helper functions -------------------
    //calculate the distance between two vectors
    double dist_vec(std::vector<double>& v1, std::vector<double>& v2){
        if (v2.size() != v1.size()){
            throw std::runtime_error("dist_vec()::v1 and v2 must have the same size");
        }
        double sum = 0.0;
        for (size_t i=0;i<v1.size();i++){
            sum += (v1[i] - v2[i])*(v1[i] - v2[i]);
        }
        return sqrt(sum);
    }
    // calculate the distance bwtween two matrices
    double dist_mat(std::vector<std::vector<double> >& m1,std::vector<std::vector<double> >& m2, size_t ndim){
        double sum = 0.0;
        for (size_t i=0;i<ndim;i++){
            for (size_t j=0;j<ndim;j++){
                sum += (m1[i][j] - m2[i][j])*(m1[i][j] - m2[i][j]);
            }
        }
        return sqrt(sum);
    }
    // add v1 = v1 + v2
    void self_add(std::vector<double>& v1, std::vector<double>& v2){
        if (v2.size() != v1.size()){
            throw std::runtime_error("self_add()::v1 and v2 must have the same size");
        }
        for (size_t i=0;i<v1.size();i++){
            v1[i] += v2[i];
        }
    }
    // print a vector
    void print_vec(std::vector<double>& v1){
        std::cout<<"{"<<v1[0];
        for (size_t i=1;i<v1.size();i++){
            std::cout<<","<<v1[i];   
        }
        std::cout<<"}";
    }

    void get_mean_cov(std::vector<std::vector<double>>& disp_vec, std::vector<double>& mean_vec, 
                  std::vector<std::vector<double>>& cov_mat){
        // disp_vec is the list of displacement
        std::vector<double> avg = {0.,0.};
        std::vector<std::vector<double> > cov = {{0.,0.},{0.,0.}};
        for (auto d:disp_vec){
            self_add(avg,d);
        }
        self_mul(avg,double(1.0/disp_vec.size()));
        for(auto d:disp_vec){
            for (size_t i = 0 ;i < ndim; i++){
                for (size_t j = 0 ;j < ndim; j++){
                    cov[i][j] += (d[i]-avg[i])*(d[j]-avg[j])/disp_vec.size();
                }
            }
        }
        mean_vec = avg;
        cov_mat = cov;
    }

    void self_mul(std::vector<double>& v1, double n){
        for (size_t i=0;i<v1.size();i++){
            v1[i] *= n;
        }
    }

    // Function to compute the cross-covariance between two vector sets
    void get_cross_cov(const std::vector<std::vector<double>>& disp1, 
                        const std::vector<std::vector<double>>& disp2, 
                        std::vector<std::vector<double>>& cross_cov_mat) {
        std::vector<double> mean_disp1 = {0.,0.}, mean_disp2 = {0.,0.};
        std::vector<std::vector<double> > cross_cov = {{0.,0.},{0.,0.}};
        if (mean_disp1.size() != mean_disp2.size()){
            throw std::runtime_error("self_add()::disp1 and disp2 must have the same size");
        }
        for (auto d:disp1){
            self_add(mean_disp1,d);
        }
        for (auto d:disp2){
            self_add(mean_disp2,d);
        }
        self_mul(mean_disp1,double(1.0/disp1.size()));
        self_mul(mean_disp2,double(1.0/disp2.size()));

        for(size_t n = 0; n < disp1.size(); ++n){
            auto& d1 = disp1[n];
            auto& d2 = disp2[n];
            for (size_t i = 0; i < ndim; ++i) {
                for (size_t j = 0; j < ndim; ++j) {
                    cross_cov[i][j] += (d1[i] - mean_disp1[i])*(d2[j] - mean_disp2[j])/disp1.size();
                }
            }
        }
        cross_cov_mat = cross_cov;
    }

    void self_mul(std::vector<std::vector<double> >& m1,double n){
        for (size_t i = 0; i < m1.size();i++){
            for (size_t j = 0; j < m1[0].size();j++){
                m1[i][j] *= n;
            }
        }
    }
};

TEST_F(SIPMeanVarianceTestThreeParticles, ReciprocalBRO){
    double eps = 0.5; // kick_size
    ha::ReciprocalPairwiseBiasedKickerPeriodicCellLists<2> kicker(eps,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        kicker.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    // k0 = {-1/sqrt(2),-1/sqrt(2)}; mean0 = 0.5*eps*k0, cov0 = eps^2/12*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    // k2 = {1/sqrt(2),-1/sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    std::vector<double> ans_mean0 = {-1.0/sqrt(2),-1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,sqrt(2)}; 
    std::vector<double> ans_mean2 = {1.0/sqrt(2), -1.0/sqrt(2)};
    self_mul(ans_mean0,eps*0.5);
    self_mul(ans_mean1,eps*0.5);
    self_mul(ans_mean2,eps*0.5);
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,eps*eps/12.0);
    self_mul(ans_cov1,eps*eps/12.0);
    self_mul(ans_cov2,eps*eps/12.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2 ;
}

TEST_F(SIPMeanVarianceTestThreeParticles, NonReciprocalBRO){
    double eps = 0.5; // kick_size
    ha::NonReciprocalPairwiseBiasedKickerPeriodicCellLists<2> kicker(eps,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        kicker.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    // k0 = {-1/sqrt(2),-1/sqrt(2)}; mean0 = 0.5*eps*k0, cov0 = eps^2/12*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    // k2 = {1/sqrt(2),-1/sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    std::vector<double> ans_mean0 = {-1.0/sqrt(2),-1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,sqrt(2)}; 
    std::vector<double> ans_mean2 = {1.0/sqrt(2), -1.0/sqrt(2)};
    self_mul(ans_mean0,eps*0.5);
    self_mul(ans_mean1,eps*0.5);
    self_mul(ans_mean2,eps*0.5);
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,eps*eps/12.0);
    self_mul(ans_cov1,eps*eps/12.0);
    self_mul(ans_cov2,eps*eps/12.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2 ;
}

TEST_F(SIPMeanVarianceTestThreeParticles, ParticlewiseBRO){
    double eps = 0.5; // kick_size
    ha::ParticlewiseBiasedKickerPeriodicCellLists<2> kicker(eps,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        kicker.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    // k0 = {-1/sqrt(2),-1/sqrt(2)}; mean0 = 0.5*eps*k0, cov0 = eps^2/12*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    // k2 = {1/sqrt(2),-1/sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    std::vector<double> ans_mean0 = {-1.0/sqrt(2),-1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,sqrt(2)}; 
    std::vector<double> ans_mean2 = {1.0/sqrt(2), -1.0/sqrt(2)};
    self_mul(ans_mean0,eps*0.5);
    self_mul(ans_mean1,eps*0.5);
    self_mul(ans_mean2,eps*0.5);
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{0.0,0.0},{0.0,2.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,eps*eps/12.0);
    self_mul(ans_cov1,eps*eps/12.0);
    self_mul(ans_cov2,eps*eps/12.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}


TEST_F(SIPMeanVarianceTestThreeParticles, ReciprocalPairwiseNoise){
    double a = 4.0*sqrt(2), mpow = 2, alpha = -1.0, D0 = 0.5; 
    ha::InversePowerPeriodicReciprocalPairwiseNoiseCellLists<2> pot(mpow,a, alpha,D0,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_stochastic_force(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;

    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = alpha*a/(4*sqrt(2))*k0, cov0 = D0*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = alpha*a/(4*sqrt(2))*k1, cov1 = D0*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = alpha*a/(4*sqrt(2))*k0, cov2 = D0*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,alpha*a/sqrt(32.0));
    self_mul(ans_mean1,alpha*a/sqrt(32.0));
    self_mul(ans_mean2,alpha*a/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,a*a*D0/32.0);
    self_mul(ans_cov1,a*a*D0/32.0);
    self_mul(ans_cov2,a*a*D0/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}

TEST_F(SIPMeanVarianceTestThreeParticles, NonReciprocalPairwiseNoise){
    double a = 4.0*sqrt(2), mpow = 2, alpha = -1.0, D0 = 0.5; 
    ha::InversePowerPeriodicNonReciprocalPairwiseNoiseCellLists<2> pot(mpow,a, alpha,D0,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    std::vector<double> xx = {-1.0,0.0, 0.0,1.0, 1.0,0.0};
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_stochastic_force(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;

    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = alpha*a/(4*sqrt(2))*k0, cov0 = D0*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = alpha*a/(4*sqrt(2))*k1, cov1 = D0*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = alpha*a/(4*sqrt(2))*k0, cov2 = D0*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,alpha*a/sqrt(32.0));
    self_mul(ans_mean1,alpha*a/sqrt(32.0));
    self_mul(ans_mean2,alpha*a/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,a*a*D0/32.0);
    self_mul(ans_cov1,a*a*D0/32.0);
    self_mul(ans_cov2,a*a*D0/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}


TEST_F(SIPMeanVarianceTestThreeParticles, ParticlewiseNoise){
    double a = 4.0*sqrt(2), mpow = 2, alpha = -0.5, D0 = 0.5; 
    ha::InversePowerPeriodicParticlewiseNoiseCellLists<2> pot(mpow,a, alpha,D0,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_stochastic_force(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }

    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;

    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = alpha*a/(4*sqrt(2))*k0, cov0 = D0*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = alpha*a/(4*sqrt(2))*k1, cov1 = D0*a^2/32*(k0+k2)*(k0+k2)^T 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = alpha*a/(4*sqrt(2))*k0, cov2 = D0*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,alpha*a/sqrt(32.0));
    self_mul(ans_mean1,alpha*a/sqrt(32.0));
    self_mul(ans_mean2,alpha*a/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{0.0,0.0},{0.0,2.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,a*a*D0/32.0);
    self_mul(ans_cov1,a*a*D0/32.0);
    self_mul(ans_cov2,a*a*D0/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};
    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}

TEST_F(SIPMeanVarianceTestThreeParticles, ParticlewiseBatch){
    double a = 4.0*sqrt(2), mpow = 2, lr = -0.5, p = 0.5;         
    ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<2> pot(mpow,a,lr,p,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_batch_energy_gradient(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat; 
    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = lr*p*a/(4*sqrt(2))*k0, cov0 = lr*lr*p*(1-p)*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = lr*p*a/(4*sqrt(2))*k1, cov1 = lr*lr*p*(1-p)*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = lr*p*a/(4*sqrt(2))*k0, cov2 = lr*lr*p*(1-p)*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,lr*a*p/sqrt(32.0));
    self_mul(ans_mean1,lr*a*p/sqrt(32.0));
    self_mul(ans_mean2,lr*a*p/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{0.0,0.0},{0.0,2.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cov1,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cov2,a*a*lr*lr*p*(1-p)/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}

TEST_F(SIPMeanVarianceTestThreeParticles, PairwiseBatch){
    double a = 4.0*sqrt(2), mpow = 2, lr = -0.5, p = 0.5;         
    ha::InversePowerPeriodicProbabilisticPairBatchCellLists<2> pot(mpow,a,lr,p,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_batch_energy_gradient(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat; 
    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = lr*p*a/(4*sqrt(2))*k0, cov0 = lr*lr*p*(1-p)*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = lr*p*a/(4*sqrt(2))*k1, cov1 = lr*lr*p*(1-p)*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = lr*p*a/(4*sqrt(2))*k0, cov2 = lr*lr*p*(1-p)*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,lr*a*p/sqrt(32.0));
    self_mul(ans_mean1,lr*a*p/sqrt(32.0));
    self_mul(ans_mean2,lr*a*p/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cov1,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cov2,a*a*lr*lr*p*(1-p)/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}


TEST_F(SIPMeanVarianceTestThreeParticles, CorrelatedBRO){
    double eps = 0.5; // kick_size
    double corr = 0.5; // noise correlation
    ha::CorrelatedPairwiseBiasedKickerPeriodicCellLists<2> kicker(eps,corr,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        kicker.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);

    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    std::vector<std::vector<double> > cross_cov_mat;
    // k0 = {-1/sqrt(2),-1/sqrt(2)}; mean0 = 0.5*eps*k0, cov0 = eps^2/12*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    // k2 = {1/sqrt(2),-1/sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    std::vector<double> ans_mean0 = {-1.0/sqrt(2),-1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,sqrt(2)}; 
    std::vector<double> ans_mean2 = {1.0/sqrt(2), -1.0/sqrt(2)};
    self_mul(ans_mean0,eps*0.5);
    self_mul(ans_mean1,eps*0.5);
    self_mul(ans_mean2,eps*0.5);
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    std::vector<std::vector<double> > ans_cross_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cross_cov1 = {{0.5,-0.5},{-0.5,0.5}};
    std::vector<std::vector<double> > ans_cross_cov2 = {{0.0,0.0},{0.0,0.0}};
    self_mul(ans_cov0,eps*eps/12.0);
    self_mul(ans_cov1,eps*eps/12.0);
    self_mul(ans_cov2,eps*eps/12.0);
    self_mul(ans_cross_cov0,(-1)*corr*eps*eps/12.0);
    self_mul(ans_cross_cov1,(-1)*corr*eps*eps/12.0);
    self_mul(ans_cross_cov2,(-1)*corr*eps*eps/12.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    get_cross_cov(disp_vec0,disp_vec1,cross_cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    double rerr3 = dist_mat(cross_cov_mat,ans_cross_cov0,ndim)/dist_mat(ans_cross_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    EXPECT_LT(rerr3,tol) << "atom 0:relative err for cross_cov_mat = " <<rerr3 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    get_cross_cov(disp_vec1,disp_vec2,cross_cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    rerr3 = dist_mat(cross_cov_mat,ans_cross_cov1,ndim)/dist_mat(ans_cross_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
    EXPECT_LT(rerr3,tol) << "atom 1:relative err for cross_cov_mat = " <<rerr3 ;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    get_cross_cov(disp_vec2,disp_vec0,cross_cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    rerr3 = dist_mat(cross_cov_mat,ans_cross_cov2,ndim)-dist_mat(ans_cross_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2 ;    
    EXPECT_LT(rerr3,tol) << "atom 2:relative err for cross_cov_mat = " <<rerr3 ;

}


TEST_F(SIPMeanVarianceTestThreeParticles, CorrelatedPairwiseNoise){
    double a = 4.0*sqrt(2), mpow = 2, alpha = -1.0, D0 = 0.5, corr = 0.5, Dtherm = 0.0;
    ha::InversePowerPeriodicCorrelatedPairwiseNoiseCellLists<2> pot(mpow,a,alpha,D0,corr,Dtherm,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_stochastic_force(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    std::vector<std::vector<double> > cross_cov_mat;

    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = alpha*a/(4*sqrt(2))*k0, cov0 = D0*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = alpha*a/(4*sqrt(2))*k1, cov1 = D0*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = alpha*a/(4*sqrt(2))*k0, cov2 = D0*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,alpha*a/sqrt(32.0));
    self_mul(ans_mean1,alpha*a/sqrt(32.0));
    self_mul(ans_mean2,alpha*a/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    std::vector<std::vector<double> > ans_cross_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cross_cov1 = {{0.5,-0.5},{-0.5,0.5}};
    std::vector<std::vector<double> > ans_cross_cov2 = {{0.0,0.0},{0.0,0.0}};
    self_mul(ans_cov0,a*a*D0/32.0);
    self_mul(ans_cov1,a*a*D0/32.0);
    self_mul(ans_cov2,a*a*D0/32.0);
    self_mul(ans_cross_cov0,(-1)*corr*a*a*D0/32.0);
    self_mul(ans_cross_cov1,(-1)*corr*a*a*D0/32.0);
    self_mul(ans_cross_cov2,(-1)*corr*a*a*D0/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    get_cross_cov(disp_vec0,disp_vec1,cross_cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    double rerr3 = dist_mat(cross_cov_mat,ans_cross_cov0,ndim)/dist_mat(ans_cross_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    EXPECT_LT(rerr3,tol) << "atom 0:relative err for cross_cov_mat = " <<rerr3 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    get_cross_cov(disp_vec1,disp_vec2,cross_cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    rerr3 = dist_mat(cross_cov_mat,ans_cross_cov1,ndim)/dist_mat(ans_cross_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    EXPECT_LT(rerr3,tol) << "atom 1:relative err for cross_cov_mat = " <<rerr3 ;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    get_cross_cov(disp_vec2,disp_vec0,cross_cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    rerr3 = dist_mat(cross_cov_mat,ans_cross_cov2,ndim)-dist_mat(ans_cross_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
    EXPECT_LT(rerr3,tol) << "atom 2:relative err for cross_cov_mat = " <<rerr3 ;
}



TEST_F(SIPMeanVarianceTestThreeParticles, CorrelatedRO){
    double eps = 0.5; // kick_size
    double corr_mag = 0.5; // noise correlation in magnitude of kicks
    double corr_dir = 1.0; // noise correlation in direction of kicks
    double corr = corr_mag*corr_dir;
    ha::CorrelatedPairwiseRandomKickerPeriodicCellLists<2> random_kicker(eps,corr_mag,corr_dir,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);

    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        random_kicker.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }

    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    std::vector<std::vector<double> > cross_cov_mat;
    std::vector<double> ans_mean0 = {0.,0.};
    std::vector<double> ans_mean1 = {0.,0.};
    std::vector<double> ans_mean2 = {0.,0.};
    std::vector<std::vector<double> > ans_cov0 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cross_cov0 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cross_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cross_cov2 = {{0.0,0.0},{0.0,0.0}};
    self_mul(ans_cov0,eps*eps/(3.0*ndim));
    self_mul(ans_cov1,2.0*eps*eps/(3.0*ndim));
    self_mul(ans_cov2,eps*eps/(3.0*ndim));
    self_mul(ans_cross_cov0,(-1.0)*(corr+3)*eps*eps/(12.0*ndim));
    self_mul(ans_cross_cov1,(-1.0)*(corr+3)*eps*eps/(12.0*ndim));
    self_mul(ans_cross_cov2,(-1.0)*(corr+3)*eps*eps/(12.0*ndim));
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    get_cross_cov(disp_vec0,disp_vec1,cross_cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)-dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    double rerr3 = dist_mat(cross_cov_mat,ans_cross_cov0,ndim)/dist_mat(ans_cross_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    EXPECT_LT(rerr3,tol) << "atom 0:relative err for cross_cov_mat = " <<rerr3 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    get_cross_cov(disp_vec1,disp_vec2,cross_cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)-dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    rerr3 = dist_mat(cross_cov_mat,ans_cross_cov1,ndim)/dist_mat(ans_cross_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
    EXPECT_LT(rerr3,tol) << "atom 1:relative err for cross_cov_mat = " <<rerr3 ;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    get_cross_cov(disp_vec2,disp_vec0,cross_cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)-dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    rerr3 = dist_mat(cross_cov_mat,ans_cross_cov2,ndim)-dist_mat(ans_cross_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2 ;    
    EXPECT_LT(rerr3,tol) << "atom 2:relative err for cross_cov_mat = " <<rerr3 ;
}



TEST_F(SIPMeanVarianceTestThreeParticles, CorrelatedPairwiseUndirectedNoise){
    double a = 4.0*sqrt(2), mpow = 2, alpha = 0.0, D0 = 0.5, corr = 0.5, Dtherm = 0.0; 
    ha::InversePowerPeriodicCorrelatedPairwiseUndirectedNoiseCellLists<2> pot(mpow,a,alpha,D0,corr,Dtherm,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_stochastic_force(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    std::vector<std::vector<double> > cross_cov_mat;

    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = alpha*a/(4*sqrt(2))*k0, cov0 = D0*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = alpha*a/(4*sqrt(2))*k1, cov1 = D0*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = alpha*a/(4*sqrt(2))*k0, cov2 = D0*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {0.,0.};
    std::vector<double> ans_mean1 = {0.,0.};
    std::vector<double> ans_mean2 = {0.,0.};
    std::vector<std::vector<double> > ans_cov0 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cross_cov0 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cross_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cross_cov2 = {{0.0,0.0},{0.0,0.0}};
    self_mul(ans_cov0,a*a*D0/32.0);
    self_mul(ans_cov1,2.0*a*a*D0/32.0);
    self_mul(ans_cov2,a*a*D0/32.0);
    self_mul(ans_cross_cov0,(-1)*corr*a*a*D0/32.0);
    self_mul(ans_cross_cov1,(-1)*corr*a*a*D0/32.0);
    self_mul(ans_cross_cov2,(-1)*corr*a*a*D0/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    get_cross_cov(disp_vec0,disp_vec1,cross_cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)-dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    double rerr3 = dist_mat(cross_cov_mat,ans_cross_cov0,ndim)/dist_mat(ans_cross_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    EXPECT_LT(rerr3,tol) << "atom 0:relative err for cross_cov_mat = " <<rerr3 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    get_cross_cov(disp_vec1,disp_vec2,cross_cov_mat); 
    rerr1 = dist_vec(mean_vec,ans_mean1)-dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    rerr3 = dist_mat(cross_cov_mat,ans_cross_cov1,ndim)/dist_mat(ans_cross_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    EXPECT_LT(rerr3,tol) << "atom 1:relative err for cross_cov_mat = " <<rerr3 ;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    get_cross_cov(disp_vec2,disp_vec0,cross_cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)-dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    rerr3 = dist_mat(cross_cov_mat,ans_cross_cov2,ndim)-dist_mat(ans_cross_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
    EXPECT_LT(rerr3,tol) << "atom 2:relative err for cross_cov_mat = " <<rerr3 ;
}



TEST_F(SIPMeanVarianceTestThreeParticles, CorrelatedBatch){
    double a = 4.0*sqrt(2), mpow = 2, lr = -0.5, p = 0.1, corr = 0.8;         
    ha::InversePowerPeriodicCorrelatedProbabilisticPairBatchCellLists<2> pot(mpow,a,lr,p,corr,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_batch_energy_gradient(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat; 
    std::vector<std::vector<double> > cross_cov_mat;
    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = lr*p*a/(4*sqrt(2))*k0, cov0 = lr*lr*p*(1-p)*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = lr*p*a/(4*sqrt(2))*k1, cov1 = lr*lr*p*(1-p)*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = lr*p*a/(4*sqrt(2))*k0, cov2 = lr*lr*p*(1-p)*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,lr*a*p/sqrt(32.0));
    self_mul(ans_mean1,lr*a*p/sqrt(32.0));
    self_mul(ans_mean2,lr*a*p/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    std::vector<std::vector<double> > ans_cross_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cross_cov1 = {{0.5,-0.5},{-0.5,0.5}};
    std::vector<std::vector<double> > ans_cross_cov2 = {{0.0,0.0},{0.0,0.0}};
    self_mul(ans_cov0,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cov1,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cov2,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cross_cov0,(-1)*corr*a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cross_cov1,(-1)*corr*a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cross_cov2,(-1)*corr*a*a*lr*lr*p*(1-p)/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    get_cross_cov(disp_vec0,disp_vec1,cross_cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    double rerr3 = dist_mat(cross_cov_mat,ans_cross_cov0,ndim)/dist_mat(ans_cross_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    EXPECT_LT(rerr3,tol) << "atom 0:relative err for cross_cov_mat = " <<rerr3 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    get_cross_cov(disp_vec1,disp_vec2,cross_cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    rerr3 = dist_mat(cross_cov_mat,ans_cross_cov1,ndim)/dist_mat(ans_cross_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
    EXPECT_LT(rerr3,tol) << "atom 1:relative err for cross_cov_mat = " <<rerr3 ;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    get_cross_cov(disp_vec2,disp_vec0,cross_cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    rerr3 = dist_mat(cross_cov_mat,ans_cross_cov2,ndim)-dist_mat(ans_cross_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2 ;    
    EXPECT_LT(rerr3,tol) << "atom 2:relative err for cross_cov_mat = " <<rerr3 ;
}




TEST_F(SIPMeanVarianceTestThreeParticles, CorrelatedPairwiseNoiseThermal){
    double a = 4.0*sqrt(2), mpow = 2, alpha = -1.0, D0 = 0.0, corr = 0.5, Dtherm = 0.5; 
    ha::InversePowerPeriodicCorrelatedPairwiseNoiseCellLists<2> pot(mpow,a,alpha,D0,corr,Dtherm,radii,boxv);
    std::vector<std::vector<double>> disp_vec0; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    std::vector<double> d(natoms * ndim);   

    for (size_t i = 0; i < n_repeat; i++){
        std::fill(d.begin(), d.end(), 0.0); 
        pot.get_thermal_displacement_only(d);
        std::vector<double> d0;
        for (size_t k = 0; k < ndim; k++) {
            d0.push_back(d[k]);
        }
        disp_vec0.push_back(d0);
    }

    std::vector<double> ans_mean = {0.0, 0.0};
    double expected_variance = 2.0*Dtherm;
    std::vector<std::vector<double>> ans_cov = {{expected_variance, 0.0}, {0.0, expected_variance}};

    std::vector<double> mean_vec;
    std::vector<std::vector<double>> cov_mat;
    get_mean_cov(disp_vec0, mean_vec, cov_mat);

    std::vector<double> zero_vec = {0.,0.};
    std::vector<std::vector<double>> zero_mat = {{0.,0.},{0.,0.}};

    double rerr_mean = dist_vec(mean_vec,ans_mean)-dist_vec(ans_mean,zero_vec);
    double rerr_cov = dist_mat(cov_mat, ans_cov, ndim) / dist_mat(ans_cov, zero_mat, ndim);

    EXPECT_LT(rerr_mean, tol) << "MA(q) noise: relative err for mean_vec = " << rerr_mean;
    EXPECT_LT(rerr_cov, tol) << "MA(q) noise: relative err for cov_mat = " << rerr_cov;
}



TEST_F(SIPMeanVarianceTestThreeParticles, CorrelatedPairwiseUndirectedNoiseThermal){
    double a = 4.0*sqrt(2), mpow = 2, alpha = -1.0, D0 = 0.0, corr = 0.5, Dtherm = 0.5; 
    ha::InversePowerPeriodicCorrelatedPairwiseUndirectedNoiseCellLists<2> pot(mpow,a,alpha,D0,corr,Dtherm,radii,boxv);
    std::vector<std::vector<double>> disp_vec0; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    std::vector<double> d(natoms * ndim);   

    for (size_t i = 0; i < n_repeat; i++){
        std::fill(d.begin(), d.end(), 0.0); 
        pot.get_thermal_displacement_only(d);
        std::vector<double> d0;
        for (size_t k = 0; k < ndim; k++) {
            d0.push_back(d[k]);
        }
        disp_vec0.push_back(d0);
    }

    std::vector<double> ans_mean = {0.0, 0.0};
    double expected_variance = 2.0*Dtherm;
    std::vector<std::vector<double>> ans_cov = {{expected_variance, 0.0}, {0.0, expected_variance}};

    std::vector<double> mean_vec;
    std::vector<std::vector<double>> cov_mat;
    get_mean_cov(disp_vec0, mean_vec, cov_mat);

    std::vector<double> zero_vec = {0.,0.};
    std::vector<std::vector<double>> zero_mat = {{0.,0.},{0.,0.}};

    double rerr_mean = dist_vec(mean_vec,ans_mean)-dist_vec(ans_mean,zero_vec);
    double rerr_cov = dist_mat(cov_mat, ans_cov, ndim) / dist_mat(ans_cov, zero_mat, ndim);

    EXPECT_LT(rerr_mean, tol) << "MA(q) noise: relative err for mean_vec = " << rerr_mean;
    EXPECT_LT(rerr_cov, tol) << "MA(q) noise: relative err for cov_mat = " << rerr_cov;
}

