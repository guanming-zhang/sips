#ifndef SGD_ALGORITHMS_HPP
#define SDG_ALGORITHMS_HPP
#include "utils/json.hpp"
#include "base_algorithm.hpp"
#include "potentials/inversepower_potential.hpp"

template<size_t ndim>
class inversepower_probabilistic_pairwise_sgd_clist: public base_potential_algorithm<ha::InversePowerPeriodicProbabilisticPairBatchCellLists<ndim> >{
public: 
    inversepower_probabilistic_pairwise_sgd_clist(double pow, double eps, double lr, double prob,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicProbabilisticPairBatchCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicProbabilisticPairBatchCellLists<ndim> >(pow,eps,lr,prob,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
};

template<size_t ndim>
class inversepower_probabilistic_particlewise_sgd_clist: public base_potential_algorithm<ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<ndim> >{
public: 
    inversepower_probabilistic_particlewise_sgd_clist(double pow, double eps, double lr, double prob,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<ndim> >(pow,eps,lr,prob,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
};

template<size_t ndim>
class inversepower_correlated_probabilistic_pairwise_sgd_clist: public base_potential_algorithm<ha::InversePowerPeriodicCorrelatedProbabilisticPairBatchCellLists<ndim> >{
public: 
    inversepower_correlated_probabilistic_pairwise_sgd_clist(double pow, double eps, double lr, double prob, double correlation,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicCorrelatedProbabilisticPairBatchCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicCorrelatedProbabilisticPairBatchCellLists<ndim> >(pow,eps,lr,prob,correlation,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
};



#endif
