This codebase is derived from https://github.com/guanming-zhang/sips, created by Guanming Zhang. The original license and copyright notices have been preserved. 
This version contains implementation of pairwise, correlated random-organizing systems (random organization, biased random organization, and stochastic gradient descent).

░██████╗██╗██████╗░░██████╗
██╔════╝██║██╔══██╗██╔════╝
╚█████╗░██║██████╔╝╚█████╗░
░╚═══██╗██║██╔═══╝░░╚═══██╗
██████╔╝██║██║░░░░░██████╔╝
╚═════╝░╚═╝╚═╝░░░░░╚═════╝░ 

𝕤𝕥𝕠𝕔𝕙𝕒𝕤𝕥𝕚𝕔𝕒𝕝𝕝𝕪 𝕚𝕟𝕥𝕖𝕣𝕒𝕔𝕥𝕚𝕟𝕘 𝕡𝕒𝕣𝕥𝕚𝕔𝕝𝕖 𝕤𝕪𝕤𝕥𝕖𝕞𝕤

Preparation
install pybind11, cmake, openmp, python
install finufft(python package)

How to build the library?
a) cd sips
b) mkdir build & cd build
c) conda activate <env> 
   (make sure the python version is consistent during building and running,
    inconsistency may trigger "module not found" error)
d) cmake ..
e) cmake --build .

How to run the examples?
a) cd scripts
b) python run.py ./examples/pairwise_sgd
(the directory should contain the input file info.json,
 you may modify the info.json as you need)

How to build and run the C++ gtests?(for testing only)
a) cd sips/cpptests/source & mkdir build
b) cmake ..
c) cmake --build . 
d) ./test_main --gtest_filter=SIP*

What are we simulating? 
We are simulating dynamical equations for particle systems.

structure of the C++ code(sips)
1) biased random orgainization 
2) stochastic gradient descent for particles 
3) random organization
3) stochastic process with multiplicative noise 

folders
--include:
----algorigms:
    base_algorithms.hpp
    bro_algorithms.hpp
        1) BRO with particle-wise kick
        2) BRO with reciprocal, pairwise kick
        3) BRO with nonreciprocal, pairwise kick (*likely to be Sam's method)
        4) BRO with correlated, pairwise kick
    sgd_algorithms.hpp
        1) SGD with selective, pairwise batch
        2) SGD with probablistic, pairwise batch
        3) SGD with selective, particlewise batch
        4) SGD with probablistic, particlewise batch
        5) SGD with probabilistic, correlated pairwise batch
    ro_algorithms.hpp
        1) RO with correlated, pairwise kick
    stocastic_algotithms.hpp
        1) particlewise stochastic dynamics
        2) pairwise stochastic dynamics
        3) correlated pairwise stochastic dynamics with thermal noise
----hyperalg:
        hyperalg classes
----sips:
        stocastical-force(kicks) classes based on hyperalg classes
----potentials:
        inversepower.hpp
----src:
        py_wrapper.cpp 
        (use pybind11 to wrap the algorithm classes)

----scratch:
    scratch.cpp
        Description: use the stcratch and test some simple functions
                     !!! do not use add_directories() to include this directory
                         in the project cmake file. It is just a scratch.

*SIPS project is based on hyperalg potentials, at the same time, keeping the hyerperalg 
as untouched as possible.
*nolhmann json hearder(https://github.com/nlohmann/json) is applied for processing json files. 
it is compatible with C++11 and C++14 standard, sometimes not work well with C++17(also depends on 
the compilor).
   


