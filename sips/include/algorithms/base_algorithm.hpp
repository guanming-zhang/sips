#ifndef ALGORITHMS_BASE_ALGORITHM_HPP
#define ALGORITHMS_BASE_ALGORITHM_HPP
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include "utils/noise.hpp"
#include "utils/json.hpp"
#include "hyperalg/vecN.hpp"
#include "utils/file_helper.hpp"
template<typename T_pot>
class base_algorithm{
public:
    std::vector<double> m_boxv;
    size_t m_natoms;
    size_t m_ndim;
    base_algorithm(
        std::shared_ptr<T_pot> pot_ptr,    
        std::vector<double> boxv,
        std::vector<double> init_coords
        ):
        m_boxv(boxv),
        m_coords(init_coords),
        m_init_coords(init_coords),
        m_current_step(0.0),
        m_cutoff_factor(1.0),
        m_json(),
        m_potential(pot_ptr){
            m_ndim = m_potential->m_ndim;
            m_natoms = init_coords.size()/m_ndim;
            if (m_natoms*m_ndim != init_coords.size()){
                throw std::runtime_error("the size of the coords must be divisiable by ndim");
            }
            m_grad.assign(m_ndim*m_natoms, 0.0);
            m_nbr.resize(m_natoms);
            m_nbr_dists.resize(m_natoms);
            m_json.clear();
            m_activity_series.clear();
            m_step_series.clear();
            zoom_rate = 0;
            zoom_end = 0;
            zoom_rate = 1;
        }
    virtual void one_step(){   
        (this->m_potential)->get_displacement(this->m_coords,this->m_grad);
        #ifdef _OPENMP
        #pragma omp parallel for
        for (size_t i = 0; i<m_coords.size();i++){
            this->m_coords[i] += this->m_grad[i];
        }
        #else
        for (size_t i = 0; i<this->m_coords.size();i++){
            this->m_coords[i] += this->m_grad[i];
        }
        #endif  
    }

    virtual void run(size_t n_steps, size_t n_save, size_t n_rec, 
                    size_t current_step = 0, double cutoff_factor = 1.0,
                    std::string output_dir='./',
                    std::string save_mode = 'concise',
                    bool compression = false){
        throw std::runtime_error("need to implement/override this method");
    }
    virtual void get_neighbors(){
        m_potential->get_neighbors(this->m_coords,this->m_nbr,this->m_nbr_dists,m_cutoff_factor);
    }

    virtual double get_activity(){
        get_neighbors();
        size_t n_active = 0;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:n_active)
        for (size_t i=0;i< this->m_natoms; i++){
            if (this->m_nbr[i].size() > 0){
                n_active++;
            }
        }
        #else
        for (size_t i=0;i<this->m_natoms; i++){
            if (this->m_nbr[i].size() > 0){
                n_active++;
            }
        }
        #endif
        return double(n_active)/double(this->m_natoms);
    }
    virtual double update_data_json(std::string save_mode){
        double activity = get_activity();
        this->m_json["x"] = this->m_coords;
        this->m_json["iter"] = this->m_current_step;
        this->m_json["activity"] = activity;
        if (save_mode != "concise"){
            // did not call get_neighbors() since it is called explicitly in 
            // get_activity()
            (this->m_json)["neighbor"] = this->m_nbr; 
            (this->m_json)["displacement"] = this->m_grad;
        }
        return activity;
    }
    
    virtual double update_stat(){
        double activity = get_activity();
        this->m_activity_series.push_back(activity);
        this->m_step_series.push_back(this->m_current_step);
        return activity;
    }
    virtual void update_stat_json(){
        this->m_json["time_steps"] = this->m_step_series;
        this->m_json["activities"] = this->m_activity_series;
    }
    virtual void write_json_to_file(std::string dir,std::string file){
        std::ofstream out_file;
        out_file.open(path_join(dir,file));
        if (out_file.is_open()){
            out_file << m_json.dump();
            out_file.close();
        }
        else{
            throw std::runtime_error("cannot open file:" + path_join(dir,file));
        }
    }

    virtual void set_zoom_steps(size_t s_start,size_t s_end, size_t r){
        zoom_start = s_start;
        zoom_end = s_end;
        zoom_rate = r;
    }

protected:
    std::vector<double> m_coords,m_init_coords;
    std::vector<double> m_grad;
    std::vector< std::vector<size_t> > m_nbr;
    std::vector<std::vector<std::vector<double>>> m_nbr_dists;
    std::vector<double> m_activity_series;
    std::vector<unsigned> m_step_series;
    size_t m_current_step;
    size_t m_nactive;
    double m_cutoff_factor;
    nlohmann::json m_json;
    std::shared_ptr<T_pot> m_potential;
    // if zoom_rate > 1 and zoom_start<current_step<zoom_end
    // save the data every n_save/zoom_rate steps
    size_t zoom_start,zoom_end,zoom_rate;
};


/*-----------------------------base nonpotential algorithm------------------------------------------*/
template<typename T_kicker>
class base_nonpotential_algorithm:public base_algorithm<T_kicker>{
public:
    base_nonpotential_algorithm(
        std::shared_ptr<T_kicker> kicker_ptr,    
        std::vector<double> boxv,
        std::vector<double> init_coords
        ):
        base_algorithm<T_kicker>(kicker_ptr,boxv,init_coords){}

    virtual void one_step(){ 
        base_algorithm<T_kicker>::one_step();
    }

    virtual double get_activity(){
        return base_algorithm<T_kicker>::get_activity();
    }
    virtual double update_data_json(std::string save_mode){
        return base_algorithm<T_kicker>::update_data_json(save_mode);
    }
    
    virtual double update_stat(){
        return base_algorithm<T_kicker>::update_stat();
    }

    virtual void update_stat_json(){
        return base_algorithm<T_kicker>::update_stat_json();
    }

    void write_json_to_file(std::string dir, std::string file){
        base_algorithm<T_kicker>::write_json_to_file(dir,file);
    }

    virtual void run(size_t n_steps, size_t n_save, size_t n_rec, 
                    size_t current_step = 0, double cutoff_factor = 1.0,
                     std::string output_dir='./', 
                     std::string save_mode = 'concise',
                     bool compression = false){

        this->m_current_step = current_step;
        this->m_cutoff_factor = cutoff_factor;
        bool save_now = false;
        while (this->m_current_step < n_steps){
            std::string output_file = "data_iter" 
                                      + std::to_string(this->m_current_step);
            // check if saving data or not
            if ((this->zoom_rate>1) and (this->m_current_step >= this->zoom_start) 
               and (this->m_current_step <= this->zoom_end)){
               save_now = (this->m_current_step*this->zoom_rate % n_save == 0);
            }
            else{
               save_now = (this->m_current_step % n_save == 0);
            }
            // write the data
            if (save_now){
                this->m_json.clear();
                double activity = update_data_json(save_mode);
                std::cout<<"step = "<< this->m_current_step;
                std::cout<<", activity = "<< activity <<std::endl;
                if (compression){
                    write_json_to_file(output_dir,"data.json");
                    compress_file(output_dir,output_file + ".zip","data.json");
                    rm_file(output_dir,"data.json");
                }
                else{
                    write_json_to_file(output_dir,output_file +".json");
                }
                if (activity < 1.0/double(this->m_natoms)){
                    break;
                }
            }
            // calcuate the statisticsupdate_stat()
            if (this->m_current_step % n_rec == 0){
                double activity = update_stat();
                if (activity < 1.0/double(this->m_natoms)){
                    update_data_json(save_mode);
                    if (compression){
                        write_json_to_file(output_dir,"data.json");
                        compress_file(output_dir,output_file + ".zip","data.json");
                        rm_file(output_dir,"data.json");
                    }
                    else{
                        write_json_to_file(output_dir,output_file +".json");
                    }
                    this->m_json.clear();
                    break;
                }
            }
            // check if absorbing every 1000 steps
            if (this->m_current_step % 1000 == 0){
                if (get_activity() < 1.0/this->m_natoms){
                    this->m_json.clear();
                    update_data_json(save_mode);
                    if (compression){
                        write_json_to_file(output_dir,"data.json");
                        compress_file(output_dir,output_file + ".zip","data.json");
                        rm_file(output_dir,"data.json");
                    }
                    else{
                        write_json_to_file(output_dir,output_file +".json");
                    }
                    break;
                }
            }
            one_step();
            this->m_current_step ++;
        }
        this->m_json.clear();
        update_stat_json();
        write_json_to_file(output_dir,"stat_data.json");
    }
};
/*-----------------------------[end]base nonpotential algorithm---------------------------------------*/


/*-----------------------------base potential algorithm------------------------------------------*/
template<typename T_pot>
class base_potential_algorithm:public base_algorithm<T_pot>{
public:
    base_potential_algorithm(
            std::shared_ptr<T_pot> pot_ptr,    
            std::vector<double> boxv,
            std::vector<double> init_coords
            ): 
            base_algorithm<T_pot>(pot_ptr,boxv,init_coords){
                m_energy_series.clear();
                m_energy_fluc_series.clear();
            }
    virtual void one_step(){   
        base_algorithm<T_pot>::one_step(); 
    }
    virtual double get_activity(){
        return base_algorithm<T_pot>::get_activity();
    }
    virtual double update_data_json(std::string save_mode){
        this->m_json["energy"] = (this->m_potential)->get_energy(this->m_coords);
        return base_algorithm<T_pot>::update_data_json(save_mode);
    }
    
    virtual double update_stat(){
        m_energy_series.push_back((this->m_potential)->get_energy(this->m_coords));
        m_energy_fluc_series.push_back(avg_energy_flucuation());
        return base_algorithm<T_pot>::update_stat();
    }

    virtual void update_stat_json(){
        this->m_json["energy"] = m_energy_series;
        // this->m_json["energy_flucuation"] = avg_energy_flucuation();
        this->m_json["energy_flucuation"] = m_energy_fluc_series;
        return base_algorithm<T_pot>::update_stat_json();
    }

    void write_json_to_file(std::string dir, std::string file){
        base_algorithm<T_pot>::write_json_to_file(dir,file);
    }

    virtual void run(size_t n_steps, size_t n_save, size_t n_rec, 
                    size_t current_step = 0, double cutoff_factor = 1.0,
                    std::string output_dir='./', 
                    std::string save_mode = 'concise',
                    bool compression = false){
        this->m_current_step = current_step;
        this->m_cutoff_factor = cutoff_factor;
        
        bool save_now = false;
        while (this->m_current_step < n_steps){
            std::string output_file = "data_iter" 
                                      + std::to_string(this->m_current_step);
            // check if saving data or not
            if ((this->zoom_rate>1) and (this->m_current_step >= this->zoom_start) 
               and (this->m_current_step <= this->zoom_end)){
               save_now = (this->m_current_step*this->zoom_rate % n_save == 0);
            }
            else{
               save_now = (this->m_current_step % n_save == 0);
            }
            // write the data
            if (save_now){
                this->m_json.clear();
                double activity = update_data_json(save_mode);
                std::cout<<"step = "<<this-> m_current_step
                        <<", activity = "<< activity << std::endl;
                if (compression){
                    write_json_to_file(output_dir,"data.json");
                    compress_file(output_dir,output_file + ".zip","data.json");
                    rm_file(output_dir,"data.json");
                }
                else{
                    write_json_to_file(output_dir,output_file +".json");
                }
                if (activity < 1.0/double(this->m_natoms)){
                    break;
                }
            }
            // calcuate the statisticsupdate_supdate_tat()
            if (this->m_current_step % n_rec == 0){
                double activity = update_stat();
                if (activity < 1.0/double(this->m_natoms)){
                    update_data_json(save_mode);
                    if (compression){
                        write_json_to_file(output_dir,"data.json");
                        compress_file(output_dir,output_file + ".zip","data.json");
                        rm_file(output_dir,"data.json");
                    }
                    else{
                        write_json_to_file(output_dir,output_file +".json");
                    }
                    this->m_json.clear();
                    break;
                }
            }
            // check if absorbing every 1000 steps
            if (this->m_current_step % 1000 == 0){
                if (get_activity() < 1.0/this->m_natoms){
                    this->m_json.clear();
                    update_data_json(save_mode);
                    if (compression){
                        write_json_to_file(output_dir,"data.json");
                        compress_file(output_dir,output_file + ".zip","data.json");
                        rm_file(output_dir,"data.json");
                    }
                    else{
                        write_json_to_file(output_dir,output_file +".json");
                    }
                    break;
                }
            }
            one_step();
            this->m_current_step ++;
        }
        this->m_json.clear();
        update_stat_json();
        write_json_to_file(output_dir,"stat_data.json");
    }
     
    double avg_energy_flucuation(double sigma = 0.02,size_t n_repeat = 5000){
            double delta = 0.0;
            sigma *= (this->m_potential)->get_average_radius();
            GaussianNoise noise(0.0,1.0,sigma);
            double energy_ref = (this->m_potential)->get_energy(this->m_coords);
            for (size_t i = 0;i < n_repeat;i++){
                std::vector<double> new_coords(this->m_coords.size());
                // perturb the coordinates
                for (size_t i =0 ;i < new_coords.size(); i++){
                    new_coords[i] = this->m_coords[i] + noise.rand();
                }
                double energy_new = (this->m_potential)->get_energy(new_coords);
                delta += energy_new - energy_ref;
            }
        return delta/(this->m_natoms*sigma*sigma*n_repeat);         
    }     
                                                             
protected:
    std::vector<double> m_energy_series; 
    std::vector<double> m_energy_fluc_series; 
};
/*-----------------------------[end]base sgd algorithm---------------------------------------*/
#endif
