#ifndef RO_ALGORITHMS_HPP
#define RO_ALGORITHMS_HPP
#include <vector>
#include <string>
#include <fstream>
#include "utils/json.hpp"
#include "hyperalg/vecN.hpp"
#include "base_algorithm.hpp"
#include "potentials/random_kicker.hpp"

// template<size_t ndim>
// class nonreciprocal_pairwise_bro_clist: public base_nonpotential_algorithm<ha::NonReciprocalPairwiseBiasedKickerPeriodicCellLists<ndim> >{
// public: 
//     double m_kick_size;
//     nonreciprocal_pairwise_bro_clist(double kick_size, std::vector<double> const radii, 
//                   std::vector<double> const boxv,std::vector<double> init_coords,
//                   const double ncellx_scale=1.0, const bool balance_omp=true):
//              base_nonpotential_algorithm<ha::NonReciprocalPairwiseBiasedKickerPeriodicCellLists<ndim> >(
//                     std::make_shared<ha::NonReciprocalPairwiseBiasedKickerPeriodicCellLists<ndim>>(kick_size,radii,boxv,ncellx_scale,balance_omp),
//                      boxv,init_coords)
//                 {}
// };

// template<size_t ndim>
// class reciprocal_pairwise_bro_clist: public base_nonpotential_algorithm<ha::ReciprocalPairwiseBiasedKickerPeriodicCellLists<ndim> >{
// public: 
//     double m_kick_size;
//     reciprocal_pairwise_bro_clist(double kick_size, std::vector<double> const radii, 
//                   std::vector<double> const boxv,std::vector<double> init_coords,
//                   const double ncellx_scale=1.0, const bool balance_omp=true):
//              base_nonpotential_algorithm<ha::ReciprocalPairwiseBiasedKickerPeriodicCellLists<ndim> >(
//                     std::make_shared<ha::ReciprocalPairwiseBiasedKickerPeriodicCellLists<ndim>>(kick_size,radii,boxv,ncellx_scale,balance_omp),
//                      boxv,init_coords)
//                 {}
// };

// template<size_t ndim>
// class pariticlewise_bro_clist: public base_nonpotential_algorithm<ha::ParticlewiseBiasedKickerPeriodicCellLists<ndim> >{
// public: 
//     double m_kick_size;
//     pariticlewise_bro_clist(double kick_size, std::vector<double> const radii, 
//                   std::vector<double> const boxv,std::vector<double> init_coords,
//                   const double ncellx_scale=1.0, const bool balance_omp=true):
//              base_nonpotential_algorithm<ha::ParticlewiseBiasedKickerPeriodicCellLists<ndim> >(
//                     std::make_shared<ha::ParticlewiseBiasedKickerPeriodicCellLists<ndim>>(kick_size,radii,boxv,ncellx_scale,balance_omp),
//                      boxv,init_coords)
//                 {}
// };

template<size_t ndim>
class correlated_pairwise_ro_clist: public base_nonpotential_algorithm<ha::CorrelatedPairwiseRandomKickerPeriodicCellLists<ndim> >{
public: 
    double m_kick_size;
    correlated_pairwise_ro_clist(double kick_size, double correlation_mag, double correlation_dir, std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
             base_nonpotential_algorithm<ha::CorrelatedPairwiseRandomKickerPeriodicCellLists<ndim> >(
                    std::make_shared<ha::CorrelatedPairwiseRandomKickerPeriodicCellLists<ndim>>(kick_size,correlation_mag,correlation_dir,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
};




#endif
