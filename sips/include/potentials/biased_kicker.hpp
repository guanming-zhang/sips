#ifndef SIP_BIASED_KICKER_CELL_LISTS_HPP
#define SIP_BIASED_KICKER_CELL_LISTS_HPP
#include <algorithm>
#include "sips/cell_list_unit_kicker.hpp"
#include "sips/cell_list_pairwise_kicker.hpp"

// this class is designated for the particlewise non-reciprocal BRO
namespace ha{
template <size_t ndim>
class ParticlewiseBiasedKickerPeriodicCellLists : public CellListUnitKicker<ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    double m_kick_size;
    const size_t m_ndim;
    ~ParticlewiseBiasedKickerPeriodicCellLists(){
        for (size_t i=0;i<m_rand_generators.size();i++){
            delete m_rand_generators[i];
        }   
    }
    ParticlewiseBiasedKickerPeriodicCellLists(double kick_size,
                std::vector<double> const radii, std::vector<double> const boxv,
                const double ncellx_scale=1.0, const bool balance_omp=true)
        : CellListUnitKicker<ha::periodic_distance<ndim> >
        (std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        balance_omp),
        m_kick_size(kick_size),
        m_boxv(boxv),
        m_ndim(ndim){
            m_rand_generators.clear();
            #ifdef _OPENMP
            for (size_t i =0 ;i < omp_get_max_threads();++i){
                UniformNoise* ptr = new UniformNoise(0.0,1.0,kick_size);
                m_rand_generators.push_back(ptr);
            }
            #else
                UniformNoise* ptr = new UniformNoise(0.0,1.0,kick_size);
                m_rand_generators.push_back(ptr);
            #endif  
        }

    virtual void get_kick(std::vector<double> const & coords, std::vector<double> & kick){
        CellListUnitKicker<ha::periodic_distance<ndim> >::get_kick(coords,kick);
        size_t n_atoms = kick.size()/ndim;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            size_t num_threads = omp_get_max_threads();
            size_t n_per_threads = n_atoms/num_threads;
            size_t thread_id = omp_get_thread_num();
            size_t iend,istart = thread_id*n_per_threads;
            if (thread_id == num_threads - 1){
                iend = n_atoms;
            }
            else{
                iend = istart + n_per_threads;
            }
            for (size_t i=istart;i<iend;i++){
                double rnd = m_rand_generators[thread_id]->rand();
                for (size_t k=0;k<ndim;k++){
                    kick[i*ndim+k] *= rnd;
                }
            }
        }
        #else
        for (size_t i=0;i<n_atoms;i++){
            double rnd = m_rand_generators[0]->rand();
            for (size_t k=0;k<ndim;k++){
                kick[i*ndim+k] *= rnd;
            }
        }
        #endif
    }
    void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        get_kick(coords,disp);
    }
    protected:
        std::vector<UniformNoise*> m_rand_generators;   
};

//Reciprocal Pairwise BRO
template <size_t ndim>
class ReciprocalPairwiseBiasedKickerPeriodicCellLists : public ha::CellListPairwiseKicker<ReciprocalUniformNoise,ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    ReciprocalPairwiseBiasedKickerPeriodicCellLists(double kick_size,
                std::vector<double> const radii, std::vector<double> const boxv,
                const double ncellx_scale=1.0, const bool balance_omp=true)
        : CellListPairwiseKicker<ReciprocalUniformNoise,ha::periodic_distance<ndim> >
        (std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        kick_size,
        balance_omp),
        m_boxv(boxv),
        m_ndim(ndim){}
    void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        CellListPairwiseKicker<ReciprocalUniformNoise,ha::periodic_distance<ndim> >::get_kick(coords,disp);
    }
};

//Nonreciprocal Pairwise BRO
template <size_t ndim>
class NonReciprocalPairwiseBiasedKickerPeriodicCellLists : public ha::CellListPairwiseKicker<NonReciprocalUniformNoise,ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    NonReciprocalPairwiseBiasedKickerPeriodicCellLists(double kick_size,
                std::vector<double> const radii, std::vector<double> const boxv,
                const double ncellx_scale=1.0, const bool balance_omp=true)
        : CellListPairwiseKicker<NonReciprocalUniformNoise,ha::periodic_distance<ndim> >
        (std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        kick_size,
        balance_omp),
        m_boxv(boxv),
        m_ndim(ndim){}
    void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        CellListPairwiseKicker<NonReciprocalUniformNoise,ha::periodic_distance<ndim> >::get_kick(coords,disp);
    }
};

//Correlated Pairwise BRO
template <size_t ndim>
class CorrelatedPairwiseBiasedKickerPeriodicCellLists : public ha::CellListPairwiseKicker<CorrelatedUniformNoise,ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    const size_t m_ndim;
    CorrelatedPairwiseBiasedKickerPeriodicCellLists(double kick_size, double correlation,
                std::vector<double> const radii, std::vector<double> const boxv,
                const double ncellx_scale=1.0, const bool balance_omp=true)
        : CellListPairwiseKicker<CorrelatedUniformNoise,ha::periodic_distance<ndim> >
        (std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        kick_size,
        balance_omp),
        m_boxv(boxv),
        m_ndim(ndim)
    {
        for (size_t i=0;i<this->m_kAcc.m_noises.size();i++){
            this->m_kAcc.m_noises[i]->set_correlation(correlation);
        }
    }
    void get_displacement(std::vector<double> const & coords, std::vector<double> & disp){
        CellListPairwiseKicker<CorrelatedUniformNoise,ha::periodic_distance<ndim> >::get_kick(coords,disp);
    }
};

}
#endif