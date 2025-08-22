#ifndef SIPS_CELL_LIST_POTENTIAL_WITH_UNDIRECTED_NOISE_HPP
#define SIPS_CELL_LIST_POTENTIAL_WITH_UNDIRECTED_NOISE_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <omp.h>
#include <math.h>
#include <random>

#include "hyperalg/distance.hpp"
#include "hyperalg/cell_list_potential.hpp"
#include "hyperalg/vecN.hpp"
#include "utils/noise.hpp"

namespace ha{

/**
 * class which accumulates the energy and gradient one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy, typename noise_type>
class PairwiseUndirectedStochasticForceAccumulator{
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;
    // create different generators for different threads to avoid race condition
    // std::vector<noise_type*> m_noises; 
    const std::vector<double> * m_coords;
    const std::vector<double> m_radii;
    std::vector<double*> m_energies;
    double m_noise_scale;
    double m_grad_scale_factor; // the scale_factor for the gradient

public:
    std::vector<noise_type*> m_noises; 
    std::vector<double> * m_gradient;
    ~PairwiseUndirectedStochasticForceAccumulator()
    {
        for(auto & energy : m_energies) {
            delete energy;
        }

        for (auto & noise: m_noises){
            delete noise;
        }
    }

    PairwiseUndirectedStochasticForceAccumulator(std::shared_ptr<pairwise_interaction> & interaction,
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii,
            double grad_scale_factor,
            double noise_scale )
        : m_interaction(interaction),
          m_dist(dist),
          m_radii(radii),
          m_grad_scale_factor(grad_scale_factor),
          m_noise_scale(noise_scale)    
        {
        #ifdef _OPENMP
        m_energies = std::vector<double*>(omp_get_max_threads());
        m_noises = std::vector<noise_type*>(omp_get_max_threads());
        // here the noise generator should not be created in parallel,
        // therefor, to avoid the same random seed.
        for (size_t i = 0; i < omp_get_max_threads(); i++){
            m_noises[i] = new noise_type(0.0,1.0,noise_scale);
        }
        #pragma omp parallel
        {
            m_energies[omp_get_thread_num()] = new double();
        }
        #else
        m_energies = std::vector<double*>(1);
        m_energies[0] = new double();
        m_noises = std::vector<noise_type*>(1);
        m_noises[0] = new noise_type(0.0,1.0,noise_scale);
        #endif
    }

    void reset_data(const std::vector<double> * coords, std::vector<double> * gradient) {
        m_coords = coords;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            *m_energies[omp_get_thread_num()] = 0;
        }
        #else
        *m_energies[0] = 0;
        #endif
        m_gradient = gradient;
    }
    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        ha::VecN<m_ndim, double> dr,xi_ij,xi_ji; //xi is the gaussian noise
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        m_dist->get_rij(dr.data(), m_coords->data() + xi_off, m_coords->data() + xj_off);
        double r2 = 0;
        for (size_t k = 0; k < m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        #ifdef _OPENMP
        for (size_t k = 0; k < m_ndim; ++k) {
            m_noises[isubdom]->rand2(&xi_ij[k],&xi_ji[k]);
        }
        #else
        for (size_t k = 0; k < m_ndim; ++k) {
            m_noises[0]->rand2(&xi_ij[k],&xi_ji[k]);
        }
        #endif
        double gij;
        double radius_sum = 0;
        if(m_radii.size() > 0) {
            radius_sum = m_radii[atom_i] + m_radii[atom_j];
        }
        
        if (sqrt(r2) < radius_sum) {
            #ifdef _OPENMP
            *m_energies[isubdom] += m_interaction->energy_gradient(r2, &gij, radius_sum);
            #else
            *m_energies[0] += m_interaction->energy_gradient(r2, &gij, radius_sum);
            #endif
            // double g = 0.0, dot_ij = 0.0, dot_ji = 0.0; 
            // for (size_t k = 0; k < m_ndim; ++k) {
            //     dr[k] *= gij; // dr = gradient 
            //     dot_ij +=  dr[k]*xi_ij[k];
            //     dot_ji +=  dr[k]*xi_ji[k];
            //     g += dr[k]*dr[k]; 
            // }
            // g = sqrt(g); // g is nonzero since sqrt(r^2) < radius_sum
            for (size_t k = 0; k < m_ndim; ++k) {
                (*m_gradient)[xi_off + k] -= (dr[k]*m_grad_scale_factor) + xi_ij[k];
                (*m_gradient)[xj_off + k] += (dr[k]*m_grad_scale_factor) + xi_ji[k];
            }
            
        }
    }

    double get_energy() {
        double energy = 0;
        for(size_t i = 0; i < m_energies.size(); ++i) {
            energy += *m_energies[i];
        }
        return energy;
    }
};


/**
 * Potential to loop over the list of atom pairs generated with the
 * cell list implementation in cell_lists.h.
 * This should also do the cell list construction and refresh, such that
 * the interface is the same for the user as with SimplePairwise.
 */
template <typename pairwise_interaction, typename distance_policy, typename noise_type>
class PairwiseUndirectedNoisyCellListPotential : public ha::CellListPotential<pairwise_interaction,distance_policy> {
protected:
   PairwiseUndirectedStochasticForceAccumulator<pairwise_interaction, distance_policy, noise_type> m_psfAcc;
public:
    ~PairwiseUndirectedNoisyCellListPotential(){}
     PairwiseUndirectedNoisyCellListPotential(
            std::shared_ptr<pairwise_interaction> interaction,
            std::shared_ptr<distance_policy> dist,
            std::vector<double> const & boxvec,
            double rcut, double ncellx_scale,
            const std::vector<double> radii,
            double grad_scale_factor = 1.0,
            double noise_scale = 1.0,
            const bool balance_omp=true)
        :CellListPotential<pairwise_interaction, distance_policy>(interaction, dist, boxvec,rcut, ncellx_scale,radii,balance_omp),
          m_psfAcc(interaction, dist, radii, grad_scale_factor, noise_scale)
            {
         }

    virtual double get_stochastic_force(std::vector<double> const & coords, std::vector<double> & grad)
    {
        const size_t natoms = coords.size() / this->m_ndim;
        if (this->m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (coords.size() != grad.size()) {
            throw std::invalid_argument("the gradient has the wrong size");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            std::fill(grad.begin(), grad.end(), NAN);
            return NAN;
        }

        this->update_iterator(coords);
        std::fill(grad.begin(), grad.end(), 0.);
        m_psfAcc.reset_data(&coords, &grad);
        auto looper = this->m_cell_lists.get_atom_pair_looper(m_psfAcc);

        looper.loop_through_atom_pairs();

        return m_psfAcc.get_energy();
    }

};

} //namespace ha

#endif 