#ifndef SIPS_CELL_LIST_POTENTIAL_WITH_PROBABILISTIC_BATCH_HPP
#define SIPS_CELL_LIST_POTENTIAL_WITH_PROBABILISTIC_BATCH_HPP

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
#include <set>
#include "hyperalg/distance.hpp"
#include "hyperalg/cell_list_potential.hpp"
#include "hyperalg/vecN.hpp"
#include "utils/noise.hpp"

namespace ha{

/**
 * class which accumulates the energy and gradient one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class PairwiseProbabilisticEnergyGradientAccumulator{
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double> * m_coords;
    const std::vector<double> m_radii;
    std::vector<double*> m_energies;
    std::vector<UniformNoise*> m_rands;
    double m_grad_scale_factor; // the scale_factor for the gradient
    double m_prob; // the probability for a pair to be selected

public:
    std::vector<double> * m_gradient;
    /** a vector for the batch of pairs m_batch_pairs[isubdom] = a set of pairs such as {{atom1,atom2},{atom2,atom5} ...}*/
    std::vector<std::set<std::array<size_t,2> > > m_batch_pairs; 
    ~PairwiseProbabilisticEnergyGradientAccumulator()
    {
        for(auto & energy : m_energies) {
            delete energy;
        }
        for (auto & r: m_rands){
            delete r;
        }
    }

    PairwiseProbabilisticEnergyGradientAccumulator(std::shared_ptr<pairwise_interaction> & interaction,
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii,
            double grad_scale_factor,
            double prob)
        : m_interaction(interaction),
          m_dist(dist),
          m_radii(radii),
          m_grad_scale_factor(grad_scale_factor),
          m_prob(prob)
    {
        #ifdef _OPENMP
        m_energies = std::vector<double*>(omp_get_max_threads());
        m_rands = std::vector<UniformNoise*>(omp_get_max_threads());
        #pragma omp parallel
        {
            m_energies[omp_get_thread_num()] = new double();
            m_rands[omp_get_thread_num()] = new UniformNoise(0.0,1.0);
        }
        #else
        m_energies = std::vector<double*>(1);
        m_energies[0] = new double();
        m_rands = std::vector<UniformNoise*>(1);
        m_rands[0] = new UniformNoise(0.0,1.0);
        #endif

        #ifdef _OPENMP
            m_batch_pairs.resize(omp_get_max_threads());
        #else
            m_batch_pairs.resize(1);
        #endif 
    }

    void reset_data(const std::vector<double> * coords, std::vector<double> * gradient) {
        m_coords = coords;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            *m_energies[omp_get_thread_num()] = 0;
            m_batch_pairs[omp_get_thread_num()].clear();
        }
        #else
        *m_energies[0] = 0;
        m_batch_pairs[0].clear();
        #endif
        m_gradient = gradient;
    }
    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        ha::VecN<m_ndim, double> dr;
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        m_dist->get_rij(dr.data(), m_coords->data() + xi_off, m_coords->data() + xj_off);
        double r2 = 0;
        for (size_t k = 0; k < m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        double gij;
        double radius_sum = 0;
        if(m_radii.size() > 0) {
            radius_sum = m_radii[atom_i] + m_radii[atom_j];
        }
        double rnd;
        //use critical to prevent the race condition for random number generation
        #ifdef _OPENMP
        rnd = m_rands[isubdom]->rand();
        #else
        rnd = m_rands[0]->rand();
        #endif
        if ((sqrt(r2) < radius_sum) and (rnd < m_prob)) {
            #ifdef _OPENMP
            *m_energies[isubdom] += m_interaction->energy_gradient(r2, &gij, radius_sum);
            m_batch_pairs[isubdom].insert({atom_i,atom_j});
            #else
            *m_energies[0] += m_interaction->energy_gradient(r2, &gij, radius_sum);
            m_batch_pairs[0].insert({atom_i,atom_j});
            #endif
            for (size_t k = 0; k < m_ndim; ++k) {
                dr[k] *= gij*m_grad_scale_factor;
                (*m_gradient)[xi_off + k] -= dr[k];
                (*m_gradient)[xj_off + k] += dr[k];
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




template <typename pairwise_interaction, typename distance_policy>
class PairwiseProbabilisticCellListPotential : public CellListPotential<pairwise_interaction,distance_policy> {
protected:
    PairwiseProbabilisticEnergyGradientAccumulator<pairwise_interaction, distance_policy> m_ppegAcc;
public:
    double m_prob;
    std::set<std::array<size_t,2> > m_tot_batch_pairs; 

    ~PairwiseProbabilisticCellListPotential(){}
     PairwiseProbabilisticCellListPotential(
            std::shared_ptr<pairwise_interaction> interaction,
            std::shared_ptr<distance_policy> dist,
            std::vector<double> const & boxvec,
            double rcut, double ncellx_scale,
            const std::vector<double> radii,
            double grad_scale_factor,
            double prob,
            const bool balance_omp=true)
        :CellListPotential<pairwise_interaction, distance_policy>(interaction, dist, boxvec,rcut, ncellx_scale,radii,balance_omp),
         m_ppegAcc(interaction, dist, radii, grad_scale_factor,prob),
         m_prob(prob){}

    virtual double get_batch_energy_gradient(std::vector<double> const & coords, std::vector<double> & grad)
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
        m_ppegAcc.reset_data(&coords, &grad);
        auto looper = this->m_cell_lists.get_atom_pair_looper(m_ppegAcc);

        looper.loop_through_atom_pairs();

        return m_ppegAcc.get_energy();
    }

    virtual void get_batch_pairs(std::set<std::array<size_t,2> > & batch_pairs){
        m_tot_batch_pairs.clear();
        for (size_t isubdom=0;isubdom<m_ppegAcc.m_batch_pairs.size();isubdom++){
            for (auto pair:m_ppegAcc.m_batch_pairs[isubdom]){
                m_tot_batch_pairs.insert(pair);
            }
        }
        batch_pairs = m_tot_batch_pairs;
    }
    
};






} //namespace ha

#endif //#ifndef 
