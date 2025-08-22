#include <iostream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "potentials/biased_kicker.hpp"
#include "potentials/inversepower_potential.hpp"


/*
 * InversePower tests
 */

class SIPSgdBatchTest :  public ::testing::Test
{
public:
    std::vector<double> x, radii, boxv = {1.0,1.0,1.0};    
    std::vector<double> d;
    size_t natoms = 100, ndim = 3;
    virtual void SetUp(){ 	
        #ifdef _OPENMP
        omp_set_num_threads(2);
        #endif
        x.resize(natoms*ndim);
        radii.resize(natoms);
        d.resize(natoms*ndim);
        for (auto & r:radii){
            r = 0.1;
        } 
        uniformly_fill(x,1.0);
    }
    double dist_vec(std::vector<double>& v1, std::vector<double>& v2){
        if (v2.size() != v1.size()){
            throw std::runtime_error("v1 and v2 must have the same size");
        }
        double sum = 0.0;
        for (size_t i=0;i<v1.size();i++){
            sum += (v1[i] - v2[i])*(v1[i] - v2[i]);
        }
        return sqrt(sum);
    }

    double norm(std::vector<double>& v1){
        double sum = 0.0;
        for (size_t i=0;i<v1.size();i++){
            sum += v1[i]*v1[i];
        }
        return sqrt(sum);
    }
    void uniformly_fill(std::vector<double>& x, double rng){
        UniformNoise rnd(0.0,rng);
        for (size_t i = 0 ; i < x.size(); i++){
            x[i] = rnd.rand();
        }
    }
    void inversepower2_pbc(double mpow,double a,
                    std::vector<double> x1, std::vector<double> x2,
                    std::vector<double>& g1, std::vector<double>& g2,
                    double r1, double r2,
                    std::vector<double>& boxv){
        size_t ndim = boxv.size();
        // deal with the pbc
        for (size_t k = 0; k < ndim; k++){
            while (x1[k] - x2[k] > 0.5*boxv[k]){
                x1[k] -= boxv[k];
            }
            while (x1[k] - x2[k] < -0.5*boxv[k]){
                x1[k] += boxv[k];
            }
        }
        double r = dist_vec(x1,x2);
        if (r >= r1+r2){
            for (size_t k = 0; k < ndim; k++){
                g1[k] = 0.0;
                g2[k] = 0.0;
            }
        }
        else{
            double v = a*std::pow(1 - r/(r1+r2),mpow - 1.0)/(r1+r2);
            for (size_t k = 0; k < ndim; k++){
                g1[k] = v*(x2[k] - x1[k])/r;
                g2[k] = v*(x1[k] - x2[k])/r;
            }
        }
    }
};

TEST_F(SIPSgdBatchTest, PairwiseSGD){
    std::vector<double> d_ref;
    d_ref.resize(natoms*ndim);
    double a = 4.0*sqrt(2), mpow = 2, lr = 1.0, p = 0.9; 
    ha::InversePowerPeriodicProbabilisticPairBatchCellLists<3> pot(mpow, a, lr, p, radii,boxv);
    
    pot.get_batch_energy_gradient(x,d);
    std::set<std::array<size_t,2> > batch_pairs;
    pot.get_batch_pairs(batch_pairs);
    for (auto pair:batch_pairs){
        std::vector<double> x1,x2;
        std::vector<double> g1,g2;
        x1.resize(ndim);
        x2.resize(ndim);
        g1.resize(ndim);
        g2.resize(ndim);
        for (size_t k = 0; k < ndim; k++){
            x1[k] = x[pair[0]*ndim + k];
            x2[k] = x[pair[1]*ndim + k];             
        } 
        inversepower2_pbc(mpow,a,x1,x2,g1,g2,radii[pair[0]],radii[pair[1]],boxv);
        for (size_t k = 0; k < ndim; k++){
            d_ref[pair[0]*ndim + k] += lr*g1[k];
            d_ref[pair[1]*ndim + k] += lr*g2[k];            
        } 
    }
    double e = dist_vec(d,d_ref)/norm(d_ref);
    EXPECT_LT(e,1e-12) << "the relative error between sgd gradient and accumulated graident = " << e;
}

TEST_F(SIPSgdBatchTest, ParticlewiseSGD){
    std::vector<double> d_ref;
    std::vector<double> g;
    d_ref.resize(natoms*ndim);
    g.resize(natoms*ndim);
    double a = 4.0*sqrt(2), mpow = 2, lr = 1.0, p = 0.8; 
    ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<3> pot(mpow, a, lr, p, radii,boxv);
    ha::InversePowerPeriodicCellLists<3> pot_ref(mpow, a,radii,boxv);
    pot.get_batch_energy_gradient(x,d);
    pot_ref.get_energy_gradient(x,g);
    std::vector<size_t> batch_particles;
    pot.get_batch_particles(batch_particles);
    for (auto i:batch_particles){
        for (size_t k = 0; k < ndim; k++){
            d_ref[i*ndim + k] = g[i*ndim + k];
        }
    }
    double e = dist_vec(d,d_ref)/norm(d_ref);
    EXPECT_LT(e,1e-12) << "the relative error between sgd gradient and accumulated graident = " << e;
}
