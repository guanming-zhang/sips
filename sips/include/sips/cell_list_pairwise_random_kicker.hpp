#ifndef SIPS_CELL_LIST_PAIRWISE_RANDOM_KICKER_HPP
#define SIPS_CELL_LIST_PAIRWISE_RANDOM_KICKER_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <omp.h>
#include <math.h>
#include <random>

#include "hyperalg/distance.hpp"
#include "hyperalg/cell_lists.hpp"
#include "hyperalg/vecN.hpp"
#include "hyperalg/cell_list_potential.hpp"
#include "utils/noise.hpp"

namespace ha{

/**
 * class which accumulates the paiwise random kick
 */
template <typename noise_mag_type, typename noise_dir_type, typename distance_policy>
class PairwiseRandomKickAccumulator {
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    double m_noise_scale;
    // std::vector<noise_mag_type*> m_noises;
    // std::vector<noise_dir_type*> d_noises;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double> * m_coords;
    const std::vector<double> m_radii;

public:
    std::vector<noise_mag_type*> m_noises;
    std::vector<noise_dir_type*> d_noises;
    /** record the total kick for each particle*/
    std::vector<double> *m_kick; 

    ~PairwiseRandomKickAccumulator(){
        for (auto & noise: m_noises){
            delete noise;
        }
        for (auto & noise: d_noises){
            delete noise;
        }
    }

    PairwiseRandomKickAccumulator(
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii,
            double noise_scale)
        : m_noise_scale(noise_scale),
          m_dist(dist),
          m_radii(radii)
    {
        #ifdef _OPENMP
        m_noises = std::vector<noise_mag_type*>(omp_get_max_threads());
        d_noises = std::vector<noise_dir_type*>(omp_get_max_threads());
        for (size_t i =0; i < omp_get_max_threads(); i++){
            m_noises[i] = new noise_mag_type(0.0,1.0,noise_scale);
            d_noises[i] = new noise_dir_type(0.0,1.0);     // only scale the magnitude of noise, and not direction
        }
        #else
        m_noises = std::vector<noise_mag_type*>(1);
        d_noises = std::vector<noise_dir_type*>(1);
        m_noises[0] = new noise_mag_type(0.0,1.0,noise_scale);
        d_noises[0] = new noise_dir_type(0.0,1.0);         // only scale the magnitude of noise, and not direction

        #endif
    }

    void reset_data(const std::vector<double> * coords, std::vector<double> * kick) {
        m_coords = coords;
        m_kick = kick;
    }
    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        ha::VecN<m_ndim, double> dr,xi_ij,xi_ji;  //xi is a random vector
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        m_dist->get_rij(dr.data(), m_coords->data() + xi_off, m_coords->data() + xj_off);
        double r2 = 0;
        for (size_t k = 0; k < m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        #ifdef _OPENMP
        for (size_t k = 0; k < m_ndim; ++k) {
            d_noises[isubdom]->rand2(&xi_ij[k],&xi_ji[k]);
        }
        #else
        for (size_t k = 0; k < m_ndim; ++k) {
            d_noises[0]->rand2(&xi_ij[k],&xi_ji[k]);
        }
        #endif

        // Calculate the norm of xi_ij and xi_ji
        double norm_xi_ij = 0.0, norm_xi_ji = 0.0;
        for (size_t k = 0; k < m_ndim; ++k) {
            norm_xi_ij += xi_ij[k] * xi_ij[k];
            norm_xi_ji += xi_ji[k] * xi_ji[k];
        }
        norm_xi_ij = sqrt(norm_xi_ij);
        norm_xi_ji = sqrt(norm_xi_ji);

        // Normalize xi_ij and xi_ji to make them unit vectors uniformly sampled on a unit hypersphere
        for (size_t k = 0; k < m_ndim; ++k) {
            xi_ij[k] /= norm_xi_ij;
            xi_ji[k] /= norm_xi_ji;
        }

        double gij;
        double radius_sum = 0;
        if(m_radii.size() > 0) {
            radius_sum = m_radii[atom_i] + m_radii[atom_j];
        }

        if (sqrt(r2)<radius_sum){
            double rnd1,rnd2;
            #if _OPENMP
            m_noises[isubdom]->rand2(&rnd1,&rnd2);
            #else
            m_noises[0]->rand2(&rnd1,&rnd2);
            #endif
            for (size_t k = 0; k < m_ndim; ++k) {
                (*m_kick)[xi_off + k] += xi_ij[k]*rnd1;
                (*m_kick)[xj_off + k] -= xi_ji[k]*rnd2;
            }
        }
    }
};



/**
 * Potential to loop over the list of atom pairs generated with the
 * cell list implementation in cell_lists.h.
 * This should also do the cell list construction and refresh, such that
 * the interface is the same for the user as with SimplePairwise.
 */
template <typename noise_mag_type, typename noise_dir_type, typename distance_policy>
class CellListPairwiseRandomKicker{
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    const std::vector<double> m_radii;
    ha::CellLists<distance_policy> m_cell_lists;
    std::shared_ptr<distance_policy> m_dist;

    PairwiseRandomKickAccumulator<noise_mag_type, noise_dir_type, distance_policy> m_kAcc;
public:
    ~CellListPairwiseRandomKicker() {}
    CellListPairwiseRandomKicker(
            std::shared_ptr<distance_policy> dist,
            std::vector<double> const & boxvec,
            double rcut, double ncellx_scale,
            const std::vector<double> radii,
            double noise_scale,
            const bool balance_omp=true)
        : m_radii(radii),
          m_cell_lists(dist, boxvec, rcut, ncellx_scale, balance_omp),
          m_dist(dist),
          m_kAcc(dist, m_radii, noise_scale)
    {}

    double sum_radii(const size_t atom_i, const size_t atom_j) const {
        if(m_radii.size() == 0) {
            return 0;
        } else {
            return m_radii[atom_i] + m_radii[atom_j];
        }
    }

    size_t get_ndim() { return m_ndim; }


    void get_kick(std::vector<double> const & coords, std::vector<double> & kick)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (coords.size() != kick.size()) {
            throw std::invalid_argument("the kick vector has the wrong size");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            std::fill(kick.begin(), kick.end(), NAN);
            return;
        }

        update_iterator(coords);
        std::fill(kick.begin(), kick.end(), 0.);
        m_kAcc.reset_data(&coords, &kick);
        auto looper = m_cell_lists.get_atom_pair_looper(m_kAcc);

        looper.loop_through_atom_pairs();
    }

    void get_neighbors(std::vector<double> const & coords,
                                std::vector<std::vector<size_t>> & neighbor_indss,
                                std::vector<std::vector<std::vector<double>>> & neighbor_distss,
                                const double cutoff_factor = 1.0)
    {
        size_t natoms = coords.size() / m_ndim;
        std::vector<short> include_atoms(natoms, 1);
        get_neighbors_picky(coords, neighbor_indss, neighbor_distss, include_atoms, cutoff_factor);
    }

    void get_neighbors_picky(std::vector<double> const & coords,
                                      std::vector<std::vector<size_t>> & neighbor_indss,
                                      std::vector<std::vector<std::vector<double>>> & neighbor_distss,
                                      std::vector<short> const & include_atoms,
                                      const double cutoff_factor = 1.0)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (natoms != include_atoms.size()) {
            throw std::runtime_error("include_atoms.size() is not equal to the number of atoms");
        }
        if (m_radii.size() == 0) {
            throw std::runtime_error("Can't calculate neighbors, because the "
                                     "used interaction doesn't use radii. ");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            return;
        }

        update_iterator(coords);
        NeighborAccumulator<distance_policy> accumulator(m_dist, coords, m_radii, (1) * cutoff_factor, include_atoms);
        auto looper = m_cell_lists.get_atom_pair_looper(accumulator);

        looper.loop_through_atom_pairs();

        neighbor_indss = accumulator.m_neighbor_indss;
        neighbor_distss = accumulator.m_neighbor_distss;
    }

    double get_rmsd(std::vector<double> const & coords1,std::vector<double> const & coords2){
        double rmsd = 0.0;
        size_t natoms = coords1.size()/m_ndim;
        if (coords1.size() != coords2.size()){
            throw std::runtime_error("CellListPotential::get_rmsd() the size of two coordinates must be the same");
        } 
        if (m_ndim * natoms != coords1.size()) {
            throw std::runtime_error("CellListPotential::get_rmsd() coords.size() is not divisible by the number of dimensions");
        }

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:rmsd)
        for (size_t i = 0; i < natoms; i++){
            const size_t xi_off = m_ndim * i;
            ha::VecN<m_ndim, double> dr;
            m_dist->get_rij(dr.data(), coords1.data() + xi_off, coords2.data() + xi_off);
            double r2 = 0.0;
            for (size_t k = 0; k < m_ndim; ++k) {
                r2 += dr[k] * dr[k];
            }
            rmsd += r2;
        }
        #else
        for (size_t i = 0; i < natoms; i++){
            const size_t xi_off = m_ndim * i;
            ha::VecN<m_ndim, double> dr;
            m_dist->get_rij(dr.data(), coords1.data() + xi_off, coords2.data() + xi_off);
            double r2 = 0.0;
            for (size_t k = 0; k < m_ndim; ++k) {
                r2 += dr[k] * dr[k];
            }
            rmsd += r2;
        }
        #endif
        rmsd = sqrt(rmsd/natoms);
        return rmsd;
    }

    double get_average_radius(){
        double r_mean = 0.0;
        #ifdef _OPENMP
        #pragma parallel for reduction(+:r_mean)
        for (size_t i = 0;i<m_radii.size();i++){
            r_mean += m_radii[i];
        }
        r_mean /= this->m_natoms;
        #else
        for (size_t i = 0;i<m_radii.size();i++){
            r_mean += m_radii[i];
        }
        r_mean /= this->m_natoms;
        #endif
        return r_mean;
    }


protected:
    virtual void update_iterator(std::vector<double> const & coords)
    {
        m_cell_lists.update(coords);
    }
};

} //namespace ha

#endif 
