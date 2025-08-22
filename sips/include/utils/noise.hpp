#ifndef NOISE_HPP
#define NOISE_HPP
#include <random>
#include <sstream>
#include <cmath>  
#include <iostream>

template<typename distribution>
class BaseNoise{
    public:
    virtual ~BaseNoise(){}
    BaseNoise():
        m_gen(std::random_device{}())
        {
            m_state << m_gen;
        }
    // abstract function to generate one number
    virtual double rand(){
        return m_dis(m_gen); 
    }

    virtual void set_seed(unsigned s){
        m_gen.seed(s);
    }

    virtual void recover(){
        m_state >> m_gen;
    }

    protected:
        std::mt19937 m_gen;
        distribution m_dis;
        std::stringstream m_state;
};

class GaussianNoise:public BaseNoise<std::normal_distribution<double>>{
    public:
    GaussianNoise(double mean=0.0,double sigma=1.0,double scale = 1.0): m_scale(scale)
    {
            m_dis.reset();
            m_dis.param(std::normal_distribution<double>::param_type(mean, sigma));
    }

    double rand() override {
        return m_scale*m_dis(m_gen);
    }

    private:
    double m_scale;
};

class UniformNoise:public BaseNoise<std::uniform_real_distribution<double>>{
    public:
    UniformNoise(double low = 0.0,double high = 1.0,double scale = 1.0): m_scale(scale){
            m_dis.reset();
            m_dis.param(std::uniform_real_distribution<double>::param_type(low, high));
    }

    double rand() override {
        return m_scale * m_dis(m_gen);
    }

    private:
        double m_scale;
};

class ReciprocalGaussianNoise:public BaseNoise<std::normal_distribution<double>>{
    public:
    ReciprocalGaussianNoise(double mean = 0.0,double sigma = 1.0, double scale = 1.0): m_scale(scale){
            m_dis.reset();
            m_dis.param(std::normal_distribution<double>::param_type(mean, sigma));
    }
    void rand2(double* a, double* b){
        *a = m_scale*rand();
        *b = *a;
    }
    private:
        double m_scale;
};


class ReciprocalUniformNoise:public BaseNoise<std::uniform_real_distribution<double>>{
    public:
    ReciprocalUniformNoise(double low = 0.0,double high = 1.0,double scale = 1.0): m_scale(scale){
            m_dis.reset();
            m_dis.param(std::uniform_real_distribution<double>::param_type(low, high));
    }
    void rand2(double* a, double* b){
        *a = m_scale*rand();
        *b = *a;
    }
    private:
        double m_scale;
};


class NonReciprocalGaussianNoise:public BaseNoise<std::normal_distribution<double>>{
    public:
    NonReciprocalGaussianNoise(double mean = 0.0,double sigma = 1.0, double scale = 1.0): m_scale(scale){
            m_dis.reset();
            m_dis.param(std::normal_distribution<double>::param_type(mean, sigma));
    }
    void rand2(double* a, double* b){
        *a = m_scale*rand();
        *b = m_scale*rand();
    }
    private:
        double m_scale;
};


class NonReciprocalUniformNoise:public BaseNoise<std::uniform_real_distribution<double>>{
    public:
    NonReciprocalUniformNoise(double low = 0.0,double high = 1.0,double scale = 1.0): m_scale(scale){
            m_dis.reset();
            m_dis.param(std::uniform_real_distribution<double>::param_type(low, high));
    }
    void rand2(double* a, double* b){
        *a = m_scale*rand();
        *b = m_scale*rand();
    }
    private:
        double m_scale;
};


class CorrelatedGaussianNoise:public BaseNoise<std::normal_distribution<double>>{
    double m_correlation;  // Correlation parameter
    public:
    CorrelatedGaussianNoise(double mean = 0.0,double sigma = 1.0,double scale = 1.0,double correlation = 0.0)
        : m_scale(scale),m_correlation(correlation){
        m_dis.reset();
        m_dis.param(std::normal_distribution<double>::param_type(mean, sigma));
    }

    void set_correlation(double correlation){
        m_correlation = correlation;
    }
    
    void rand2(double* a, double* b){
        // Generate two independent standard normal random variables
        *a = rand();
        *b = rand();

        // Transform a,b into a bivariate normal distribution with the given correlation - m_correlation
        // Note than a,b have both mean 0 and variance 1, independently
        *b = m_correlation*(*a) + std::sqrt(1 - m_correlation*m_correlation)*(*b);

        *a = m_scale*(*a);
        *b = m_scale*(*b);
    }
    private:
        double m_scale;
};


class CorrelatedUniformNoise:public BaseNoise<std::normal_distribution<double>>{
    double m_correlation;  // Correlation parameter
    public:
    CorrelatedUniformNoise(double mean = 0.0,double sigma = 1.0,double scale = 1.0,double correlation = 0.0)
        : m_scale(scale),m_correlation(correlation){
        m_dis.reset();
        m_dis.param(std::normal_distribution<double>::param_type(mean, sigma));
    }

    void set_correlation(double correlation){
        // m_correlation = correlation;
        m_correlation = 2.0*sin((correlation*M_PI)/6.0);
    }
    
    void rand2(double* a, double* b){
        // Generate two independent standard normal random variables
        *a = rand();
        *b = rand();

        // Transform a,b into a bivariate normal distribution with the given correlation - m_correlation
        // Note than a,b have both mean 0 and variance 1, independently
        *b = m_correlation*(*a) + std::sqrt(1 - (m_correlation*m_correlation))*(*b);

        // Transform a,b into a bivariate uniform distribution with the given correlation - m_correlation
        // Note than a,b have both U[0,1] and have mean 1/2 and variance 1/12, independently
        *a = 0.5 * (1.0 + std::erf((*a) / std::sqrt(2.0)));
        *b = 0.5 * (1.0 + std::erf((*b) / std::sqrt(2.0)));

        *a = m_scale*(*a);
        *b = m_scale*(*b);
    }
    private:
        double m_scale;
};


class CorrelatedBernoulliNoise:public BaseNoise<std::uniform_real_distribution<double>>{
    double m_correlation;  // Correlation parameter
    double m_p;            // probability parameter (prob of getting 1)
    public:
    CorrelatedBernoulliNoise(double low = 0.0,double high = 1.0,double scale = 1.0,double correlation = 0.0,double p = 0.0)
        : m_scale(scale),m_correlation(correlation),m_p(p){
        m_dis.reset();
        m_dis.param(std::uniform_real_distribution<double>::param_type(low, high));
    }

    void set_correlation_and_p(double correlation, double p){
        m_correlation = correlation;
        m_p = p;
    }
    
    void rand2(double* a, double* b){

        // Calculate joint probabilities
        double p00 = (1-m_p)*(1-m_p) + m_correlation*m_p*(1-m_p);
        double p01 = 1 - m_p - p00;
        double p10 = 1 - m_p - p00;
        // double p11 = p00 + (2*m_p) - 1;
        
        // Generate a standard uniform random variable
        double u = rand();

        if (u < p00) {
            *a = 0.0;
            *b = 0.0;
        } else if (u < p00+p01) {
            *a = 0.0;
            *b = 1.0;
        } else if (u < p00+p01+p10) {
            *a = 1.0;
            *b = 0.0;
        } else {
            *a = 1.0;
            *b = 1.0;
        }

    }
    private:
        double m_scale;
};


#endif

