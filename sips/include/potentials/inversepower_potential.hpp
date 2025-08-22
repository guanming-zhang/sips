#ifndef HYPERALG_INVERSEPOWER_HPP
#define HYPERALG_INVERSEPOWER_HPP

#include <string>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <math.h>
#include <vector>

#include "hyperalg/simple_pairwise_potential.hpp"
#include "hyperalg/cell_list_potential.hpp"
#include "hyperalg/distance.hpp"
#include "hyperalg/meta_pow.hpp"
#include "hyperalg/base_interaction.hpp"
#include "sips/cell_list_potential_with_probabilistic_batch.hpp"
#include "sips/cell_list_potential_with_correlated_probabilistic_batch.hpp"
#include "sips/cell_list_potential_with_noise.hpp"
#include "sips/cell_list_potential_with_undirected_noise.hpp"
//#include "sips/cell_list_potential_with_selective_batch.hpp"

namespace ha{
    
class InversePowerInteraction : public ha::BaseInteraction{
public:
    const double m_pow; // Inverse power exponent
    const double m_eps;
    
    InversePowerInteraction(double a, double eps)
    : m_pow(a),
      m_eps(eps)
    {}

    /* calculate energy from distance squared */
    virtual double energy(double r2, const double radius_sum) const
    {
      double E;
      const double r = std::sqrt(r2);
      if (r >= radius_sum) {
          E = 0.;
      }
      else {
          E = std::pow((1 -r/radius_sum), m_pow) * m_eps/m_pow;
      }
      return E;
    }

    /* calculate energy and gradient from distance squared, gradient is in -(dv/drij)/|rij| */
    virtual double energy_gradient(double r2, double *gij, const double radius_sum) const
    {
      double E;
      const double r = std::sqrt(r2);
      //here we should not use "if (r2 >= radius_sum*radius_sum)" in the condition
      //because it is possible that r2 is smaller than radius_sum^2 by a very small amout(~1e-14) and the else part is excuted
      //but sqrt(r2) and radius_sum are exactly the same in the machine due to the lower persition of sqrt()
      //this will cause a problem when dividing (r-radius_sum) 
      if (r >= radius_sum) {
          E = 0.;
          *gij = 0.;
      }
      else if (abs(m_pow - 1.0) < 1e-6) {
          E = (1 - r/radius_sum)*m_eps;
          *gij = m_eps/(radius_sum*r);
      }
      else {
          const double factor = std::pow((1 -r/radius_sum), m_pow) * m_eps;
          E =  factor / m_pow;
          *gij =  - factor / ((r-radius_sum)*r);
      }
      return E;
    }

    virtual double energy_gradient_hessian(double r2, double *gij, double *hij, const double radius_sum) const
    {
      double E;
      const double r = std::sqrt(r2);
      if (r >= radius_sum) {
          E = 0.;
          *gij = 0;
          *hij=0;
      }
      else {
          const double factor = std::pow((1 -r/radius_sum), m_pow) * m_eps;
          const double denom = 1.0 / (r-radius_sum);
          E =  factor / m_pow;
          *gij =  - factor * denom / r ;
          *hij = (m_pow-1) * factor * denom * denom;
      }
      return E;
    }
};

template<size_t ndim>
class InversePowerCartesian : public ha::SimplePairwisePotential<ha::InversePowerInteraction, ha::cartesian_distance<ndim>>{
    public:
    const size_t m_ndim;
    InversePowerCartesian(double a, double eps, const std::vector<double> radii)
    : SimplePairwisePotential<ha::InversePowerInteraction, ha::cartesian_distance<ndim>>
    (std::make_shared<ha::InversePowerInteraction>(a, eps),
    std::make_shared<ha::cartesian_distance<ndim>>(),
    radii),
    m_ndim(ndim)
    {}
};

template<size_t ndim>
class InversePowerPeriodic : public ha::SimplePairwisePotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>>{
    public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    InversePowerPeriodic(double a, double eps, const std::vector<double> radii, const std::vector<double> boxv)
    : SimplePairwisePotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>>
    (std::make_shared<ha::InversePowerInteraction>(a, eps),
    std::make_shared<ha::periodic_distance<ndim>>(boxv),
    radii),
    m_boxv(boxv),
    m_ndim(ndim)
    {}
};

template <size_t ndim>
class InversePowerPeriodicCellLists : public CellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    InversePowerPeriodicCellLists(double pow, double eps,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        : CellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim> >
        (std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        balance_omp),
        m_boxv(boxv),
        m_ndim(ndim)
    {}
};

/* ----------- inverse power potential with stochastic batches --------- */ 

//selective, pairwise batch, this class is obselete
/* 
template <size_t ndim>
class InversePowerPeriodicSelectiveBatchCellLists : public BatchCellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    InversePowerPeriodicBatchCellLists(double pow, double eps,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        : BatchCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim> >(std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        balance_omp),
        m_boxv(boxv)
    {}
};
*/
/* ----------- inverse power potential with probabilistic batch --------- */ 
template <size_t ndim>
class InversePowerPeriodicProbabilisticPairBatchCellLists : public PairwiseProbabilisticCellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    InversePowerPeriodicProbabilisticPairBatchCellLists(double pow, double eps, double lr, double prob,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        : PairwiseProbabilisticCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim> >(std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        lr,
        prob,
        balance_omp),
        m_boxv(boxv),
        m_ndim(ndim)
    {}
    virtual void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        PairwiseProbabilisticCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim> >::get_batch_energy_gradient(coords,disp);

    }

};



template <size_t ndim>
class InversePowerPeriodicCorrelatedProbabilisticPairBatchCellLists : public PairwiseCorrelatedProbabilisticCellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim>,CorrelatedBernoulliNoise > {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    InversePowerPeriodicCorrelatedProbabilisticPairBatchCellLists(double pow, double eps, double lr, double prob, double correlation,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        : PairwiseCorrelatedProbabilisticCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>, CorrelatedBernoulliNoise >(std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        lr,
        prob,
        balance_omp),
        m_boxv(boxv),
        m_ndim(ndim)
    {
        for (size_t i=0;i<this->m_ppegAcc.m_noises.size();i++){
            this->m_ppegAcc.m_noises[i]->set_correlation_and_p(correlation, prob);
        }
    }
    virtual void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        PairwiseCorrelatedProbabilisticCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>, CorrelatedBernoulliNoise >::get_batch_energy_gradient(coords,disp);

    }
};






/* ----------- inverse power potential with probabilistic particle batch --------- */ 
template <size_t ndim>
class InversePowerPeriodicProbabilisticParticleBatchCellLists : public CellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    InversePowerPeriodicProbabilisticParticleBatchCellLists(double pow, double eps, double lr, double prob,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        :CellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim> >(std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        balance_omp),
        m_boxv(boxv),
        m_lr(lr),
        m_prob(prob),
        m_ndim(ndim)
    {
        //create the random generators
        #ifdef _OPENMP
        for (size_t i =0 ;i < omp_get_max_threads();++i){
            UniformNoise* ptr = new UniformNoise(0.0,1.0);
            m_rand_generators.push_back(ptr);
        }
        m_batch_particles.resize(omp_get_max_threads());
        #else
        UniformNoise* ptr = new UniformNoise(0.0,1.0);
        m_rand_generators.push_back(ptr);
        m_batch_particles.resize(1);
        #endif  
    }       
    
    virtual double get_batch_energy_gradient(std::vector<double> const & coords, std::vector<double> & grad){
        // update the displacement
        CellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim> >::get_energy_gradient(coords,grad);
        size_t natoms = coords.size()/ndim;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            size_t num_threads = omp_get_max_threads();
            size_t n_per_threads = ndim/num_threads;
            size_t thread_id = omp_get_thread_num();
            size_t iend,istart = thread_id*n_per_threads;
            m_batch_particles[thread_id].clear();
            // the last thread
            if (thread_id == num_threads - 1){
                iend = natoms;
            }
            else{
                iend = istart + n_per_threads;
            }
            for (size_t i=istart;i<iend;i++){
                if (m_prob < m_rand_generators[thread_id]->rand()){
                    for (size_t k=0;k<ndim;k++){
                        grad[i*ndim+k] = 0.0;
                    }
                }
                else{
                    m_batch_particles[thread_id].push_back(i);
                    for (size_t k=0;k<ndim;k++){
                        grad[i*ndim+k] *= m_lr;
                    }
                }
            }
        }
        #else
        m_batch_particles[0].clear();
        for (size_t i=0;i< natoms; i++){
            double rnd = m_rand_generators[0]->rand();
            if (rnd < m_prob){
                for (size_t k=0;k<ndim;k++){
                    grad[i*ndim+k] = 0.0;
                }
            }
            else{
                m_batch_particles[0].push_back(i);
                for (size_t k=0;k<ndim;k++){
                    grad[i*ndim+k] *= m_lr;
                }
            }
        }
        #endif  
        // for particlewise batch, batch energy is undefined
        return -1.0;
    }

    void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        get_batch_energy_gradient(coords,disp);
    }

    void get_batch_particles(std::vector<size_t>& particle_batch){
        m_tot_batch_particles.clear();
        for (auto batch:m_batch_particles){
            for (auto atom_i: batch){
                m_tot_batch_particles.push_back(atom_i);
            }
        }
        particle_batch = m_tot_batch_particles;
    }
protected:
    double m_prob, m_lr;
    std::vector<std::vector<size_t> > m_batch_particles;
    std::vector<size_t> m_tot_batch_particles;
    std::vector<UniformNoise*> m_rand_generators; 

};


/* ----------- inverse power potential with pairwise muitiplicative noise --------- */ 
template <size_t ndim>
class InversePowerPeriodicReciprocalPairwiseNoiseCellLists : public ha::PairwiseNoisyCellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim>,ReciprocalGaussianNoise> {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    InversePowerPeriodicReciprocalPairwiseNoiseCellLists(double pow, double eps, double alpha, double D0,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        :  PairwiseNoisyCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>, ReciprocalGaussianNoise>(
            std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        alpha,
        sqrt(D0),
        balance_omp),
        m_boxv(boxv),
        m_ndim(ndim)
    {}
    virtual void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        PairwiseNoisyCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>, ReciprocalGaussianNoise>::get_stochastic_force(coords,disp);
    }

};

template <size_t ndim>
class InversePowerPeriodicNonReciprocalPairwiseNoiseCellLists : public PairwiseNoisyCellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim>,NonReciprocalGaussianNoise > {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    InversePowerPeriodicNonReciprocalPairwiseNoiseCellLists(double pow, double eps, double alpha, double D0,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        :  PairwiseNoisyCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>, NonReciprocalGaussianNoise>(
            std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        alpha,
        sqrt(D0),
        balance_omp),
        m_boxv(boxv),
        m_ndim(ndim)
    {}
    virtual void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
         PairwiseNoisyCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>, NonReciprocalGaussianNoise>::get_stochastic_force(coords,disp);

    }

};


template <size_t ndim>
class InversePowerPeriodicCorrelatedPairwiseNoiseCellLists : public ha::PairwiseNoisyCellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim>,CorrelatedGaussianNoise> {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    InversePowerPeriodicCorrelatedPairwiseNoiseCellLists(double pow, double eps, double alpha, double D0, double correlation, double Dtherm,
        std::vector<double> const radii, std::vector<double> const boxv, 
        const double ncellx_scale=1.0, const bool balance_omp=true)
        :  PairwiseNoisyCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>, CorrelatedGaussianNoise>(
            std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        alpha,
        sqrt(D0),
        balance_omp),
        m_boxv(boxv), m_ndim(ndim), m_Dtherm(Dtherm),
        m_thermal_noise_generator(0.0, 1.0)
    {
        for (size_t i=0;i<this->m_psfAcc.m_noises.size();i++){
            this->m_psfAcc.m_noises[i]->set_correlation(correlation);
        }
    }
    virtual void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        PairwiseNoisyCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>, CorrelatedGaussianNoise>::get_stochastic_force(coords,disp);
        std::vector<double> thermal_noise(disp.size());
        for (size_t i = 0; i < thermal_noise.size(); ++i) {
            thermal_noise[i] = m_thermal_noise_generator.rand();
        }
        for (size_t i = 0; i < disp.size(); i++) {
            disp[i] += sqrt(2.0*m_Dtherm)*thermal_noise[i];
        }
    }

    // only for testing purposes
    void get_thermal_displacement_only(std::vector<double> & disp)
    {
        std::vector<double> thermal_noise(disp.size());
        for (size_t i = 0; i < thermal_noise.size(); ++i) {
            thermal_noise[i] = m_thermal_noise_generator.rand();
        }
        for (size_t i = 0; i < disp.size(); i++) {
            disp[i] += sqrt(2.0*m_Dtherm)*thermal_noise[i];
        }
    }

private:
    double m_Dtherm;
    GaussianNoise m_thermal_noise_generator;

};


template <size_t ndim>
class InversePowerPeriodicCorrelatedPairwiseUndirectedNoiseCellLists : public ha::PairwiseUndirectedNoisyCellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim>,CorrelatedGaussianNoise> {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    InversePowerPeriodicCorrelatedPairwiseUndirectedNoiseCellLists(double pow, double eps, double alpha, double D0, double correlation, double Dtherm,
        std::vector<double> const radii, std::vector<double> const boxv, 
        const double ncellx_scale=1.0, const bool balance_omp=true)
        :  PairwiseUndirectedNoisyCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>, CorrelatedGaussianNoise>(
            std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        alpha,
        sqrt(D0),
        balance_omp),
        m_boxv(boxv), m_ndim(ndim), m_Dtherm(Dtherm),
        m_thermal_noise_generator(0.0, 1.0)
    {
        for (size_t i=0;i<this->m_psfAcc.m_noises.size();i++){
            this->m_psfAcc.m_noises[i]->set_correlation(correlation);
        }
    }
    virtual void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        PairwiseUndirectedNoisyCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>, CorrelatedGaussianNoise>::get_stochastic_force(coords,disp);
        std::vector<double> thermal_noise(disp.size());
        for (size_t i = 0; i < thermal_noise.size(); ++i) {
            thermal_noise[i] = m_thermal_noise_generator.rand();
        }
        for (size_t i = 0; i < disp.size(); i++) {
            disp[i] += sqrt(2.0*m_Dtherm)*thermal_noise[i];
        }
    }

    // only for testing purposes
    void get_thermal_displacement_only(std::vector<double> & disp)
    {
        std::vector<double> thermal_noise(disp.size());
        for (size_t i = 0; i < thermal_noise.size(); ++i) {
            thermal_noise[i] = m_thermal_noise_generator.rand();
        }
        for (size_t i = 0; i < disp.size(); i++) {
            disp[i] += sqrt(2.0*m_Dtherm)*thermal_noise[i];
        }
    }

private:
    double m_Dtherm;
    GaussianNoise m_thermal_noise_generator;

};


template <size_t ndim>
class InversePowerPeriodicParticlewiseNoiseCellLists : public CellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    InversePowerPeriodicParticlewiseNoiseCellLists(double pow, double eps, double alpha, double D0,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        :  CellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim> >(
            std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        balance_omp),
        m_alpha(alpha),
        m_D0(D0),
        m_boxv(boxv),
        m_ndim(ndim){
        //create the random generators
        #ifdef _OPENMP
        for (size_t i =0 ;i < omp_get_max_threads();++i){
            GaussianNoise* ptr = new GaussianNoise(0.0,1.0,sqrt(D0));
            m_rand_generators.push_back(ptr);
        }
        #else
        GaussianNoise* ptr = new GaussianNoise(0.0,1.0,sqrt(D0));
        m_rand_generators.push_back(ptr);
        #endif  
    }
    virtual void get_stochastic_force(std::vector<double> const & coords, std::vector<double> & grad){
        CellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim> >::get_energy_gradient(coords,grad);
        size_t natoms = coords.size()/ndim;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            size_t num_threads = omp_get_max_threads();
            size_t n_per_threads = natoms/num_threads;
            size_t thread_id = omp_get_thread_num();
            size_t iend,istart = thread_id*n_per_threads;
            if (thread_id == num_threads - 1){
                iend = natoms;
            }
            else{
                iend = istart + n_per_threads;
            }
            for (size_t i=istart;i<iend;i++){
                ha::VecN<ndim, double> xi; //xi is the gaussian noise
                const size_t xi_off = ndim * i;
                // noise = sqrt(g*g^T)*xi_ij = 1/|g|*g*g^T*xi_ij 
                // g := |g|, dot = dot(g,xi)
                double g = 0.0, dot = 0.0;            
                for (size_t k = 0; k < ndim; ++k) {
                    dot += grad[xi_off + k]*m_rand_generators[thread_id]->rand();
                    g += grad[xi_off + k]*grad[xi_off + k]; 
                }
                g = sqrt(g) + 1e-12; // avoid g=0
                for (size_t k = 0; k < ndim; ++k) {
                    grad[xi_off + k] *= m_alpha + dot/g;
                }
            }
        }
        #else
        for (size_t i=0;i<natoms;i++){
            ha::VecN<ndim, double> xi; //xi is the gaussian noise
            const size_t xi_off = ndim * i;
            // noise = sqrt(g*g^T)*xi_ij = 1/|g|*g*g^T*xi_ij 
            // g := |g|, dot = dot(g,xi)
            double g = 0.0, dot = 0.0;            
            for (size_t k = 0; k < ndim; ++k) {
                dot += grad[xi_off + k]*m_rand_generators[0]->rand();
                g += grad[xi_off + k]*grad[xi_off + k]; 
            }
            g = sqrt(g) + 1e-12; // avoid g=0
            for (size_t k = 0; k < ndim; ++k) {
                grad[xi_off + k] *= m_alpha + dot/g;
            }
        }
        #endif  
    }
    virtual void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        get_stochastic_force(coords,disp);
    }
    protected:
        std::vector<GaussianNoise*> m_rand_generators; 
        double m_alpha,m_D0;

};

}


#endif
