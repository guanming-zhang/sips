#include "algorithms/bro_algorithms.hpp"
#include "algorithms/sgd_algorithms.hpp"
#include "algorithms/stochastic_algorithms.hpp"
#include "utils/noise.hpp"
#include <algorithm>
#include <iostream>

int main(){

    double corr_g = 0.5, corr_u = 0.5;

    CorrelatedGaussianNoise gen1(0.0, 1.0);
    CorrelatedUniformNoise gen2(0.0, 1.0);

    gen1.set_correlation(corr_g);
    gen2.set_correlation(corr_u);

    double x1, x2;
    double u1, u2;
    double n_repeat = 10000;
    double mean_g = 0.0, cov_g = 0.0, mean_u = 0.0, cov_u = 0.0, cross_cov_ug = 0.0;

    for (size_t i = 0; i < n_repeat; i++) {
        gen1.rand2(&x1, &x2);
        gen2.rand2(&u1, &u2);

        mean_g += x1 / n_repeat;
        cov_g += (x1 * x2) / n_repeat;
        mean_u += u1 / n_repeat;
        cov_u += ((u1*u2 - 0.25)/(1.0/12.0)) / n_repeat;
        cross_cov_ug += (x1 * u1) / n_repeat;
    }

    std::cout << "mean_gaussian = " << mean_g << std::endl;
    std::cout << "cov_gaussian = " << cov_g << std::endl;
    std::cout << "mean_uni = " << mean_u << std::endl;
    std::cout << "cov_uni = " << cov_u << std::endl;
    std::cout << "cross_cov_gaussian_uni = " << cross_cov_ug << std::endl;

    return 0;
}