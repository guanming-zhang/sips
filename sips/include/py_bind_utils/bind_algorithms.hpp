#ifndef BIND_SIPS_ALGORITHMS_HPP
#define BIND_SIPS_ALGORITHMS_HPP

#include "algorithms/bro_algorithms.hpp"
#include "algorithms/ro_algorithms.hpp"
#include "algorithms/sgd_algorithms.hpp"
#include "algorithms/stochastic_algorithms.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// this is the helper function for py_bind [CORRELATED_RO] classes
// the constructor format matches [CORRELATED_RO] classes only
template <typename T>
void bind_correlated_ro(py::module& m,std::string class_name){
    py::class_<T>(m,class_name.c_str())
    .def(py::init<double, double, double, std::vector<double> const, 
                  std::vector<double> const,std::vector<double>,
                  const double, const bool>(),
                  py::arg("kick_size"),py::arg("corr_mag"),py::arg("corr_dir"),py::arg("radii"),
                  py::arg("box_size"),py::arg("x0"),
                  py::arg("ncellx_scale")=1.0,py::arg("balance_omp")=true)
    .def("run",&T::run,"run the algorithm",
         py::arg("n_steps"),py::arg("n_save"),py::arg("n_rec"),
         py::arg("starting_step")=0,
         py::arg("cutoff")=1.0,
         py::arg("output_dir") = "./",
         py::arg("save_mode") = "concise",
         py::arg("compression") = false)
    .def("set_zoom_steps",&T::set_zoom_steps,
          py::arg("starting_step"),py::arg("ending_step"),py::arg("zoom_rate"));
}

// this is the helper function for py_bind [BRO] classes
// the constructor format matches [BRO] classes only
template <typename T>
void bind_bro(py::module& m,std::string class_name){
    py::class_<T>(m,class_name.c_str())
    .def(py::init<double, std::vector<double> const, 
                  std::vector<double> const,std::vector<double>,
                  const double, const bool>(),
                  py::arg("kick_size"),py::arg("radii"),
                  py::arg("box_size"),py::arg("x0"),
                  py::arg("ncellx_scale")=1.0,py::arg("balance_omp")=true)
    .def("run",&T::run,"run the algorithm",
         py::arg("n_steps"),py::arg("n_save"),py::arg("n_rec"),
         py::arg("starting_step")=0,
         py::arg("cutoff")=1.0,
         py::arg("output_dir") = "./",
         py::arg("save_mode") = "concise",
         py::arg("compression") = false)
    .def("set_zoom_steps",&T::set_zoom_steps,
          py::arg("starting_step"),py::arg("ending_step"),py::arg("zoom_rate"));
}

// this is the helper function for py_bind [CORRELATED_BRO] classes
// the constructor format matches [CORRELATED_BRO] classes only
template <typename T>
void bind_correlated_bro(py::module& m,std::string class_name){
    py::class_<T>(m,class_name.c_str())
    .def(py::init<double, double, std::vector<double> const, 
                  std::vector<double> const,std::vector<double>,
                  const double, const bool>(),
                  py::arg("kick_size"),py::arg("corr"),py::arg("radii"),
                  py::arg("box_size"),py::arg("x0"),
                  py::arg("ncellx_scale")=1.0,py::arg("balance_omp")=true)
    .def("run",&T::run,"run the algorithm",
         py::arg("n_steps"),py::arg("n_save"),py::arg("n_rec"),
         py::arg("starting_step")=0,
         py::arg("cutoff")=1.0,
         py::arg("output_dir") = "./",
         py::arg("save_mode") = "concise",
         py::arg("compression") = false)
    .def("set_zoom_steps",&T::set_zoom_steps,
          py::arg("starting_step"),py::arg("ending_step"),py::arg("zoom_rate"));
}

// this is the helper function for py_bind [INVERSEPOWER_SGD] classes
// the constructor format matches [INVERSEPOWER_SGD] classes only
template <typename T>
void bind_inversepower_sgd(py::module& m,std::string class_name){
    py::class_<T>(m,class_name.c_str())
    .def(py::init<double, double, double, double,
                  std::vector<double> const, 
                  std::vector<double> const,
                  std::vector<double>,
                  const double, const bool>(),
                  py::arg("mpow"),py::arg("a"),py::arg("lr"),py::arg("prob"),
                  py::arg("radii"),
                  py::arg("box_size"),py::arg("x0"),
                  py::arg("ncellx_scale")=1.0,py::arg("balance_omp")=true)
    .def("run",&T::run,"run the algorithm",
         py::arg("n_steps"),py::arg("n_save"),py::arg("n_rec"),
         py::arg("starting_step")=0,
         py::arg("cutoff")=1.0,
         py::arg("output_dir") = "./",
         py::arg("save_mode") = "concise",
         py::arg("compression") = false)
    .def("set_zoom_steps",&T::set_zoom_steps,
          py::arg("starting_step"),py::arg("ending_step"),py::arg("zoom_rate"));
}

// this is the helper function for py_bind [INVERSEPOWER_CORRELATED_SGD] classes
// the constructor format matches [INVERSEPOWER_CORRELATED_SGD] classes only
template <typename T>
void bind_inversepower_correlated_sgd(py::module& m,std::string class_name){
    py::class_<T>(m,class_name.c_str())
    .def(py::init<double, double, double, double, double,
                  std::vector<double> const, 
                  std::vector<double> const,
                  std::vector<double>,
                  const double, const bool>(),
                  py::arg("mpow"),py::arg("a"),py::arg("lr"),py::arg("prob"),py::arg("corr"),
                  py::arg("radii"),
                  py::arg("box_size"),py::arg("x0"),
                  py::arg("ncellx_scale")=1.0,py::arg("balance_omp")=true)
    .def("run",&T::run,"run the algorithm",
         py::arg("n_steps"),py::arg("n_save"),py::arg("n_rec"),
         py::arg("starting_step")=0,
         py::arg("cutoff")=1.0,
         py::arg("output_dir") = "./",
         py::arg("save_mode") = "concise",
         py::arg("compression") = false)
    .def("set_zoom_steps",&T::set_zoom_steps,
          py::arg("starting_step"),py::arg("ending_step"),py::arg("zoom_rate"));
}

// this is the helper function for py_bind [INVERSEPOWER_STOCHASTICDYN] classes
// the constructor format matches [INVERSEPOWER_STOCHASTICDYN] classes only
template <typename T>
void bind_inversepower_sd(py::module& m,std::string class_name){
    py::class_<T>(m,class_name.c_str())
    .def(py::init<double, double, double, double,
                  std::vector<double> const, 
                  std::vector<double> const,
                  std::vector<double>,
                  const double, const bool>(),
                  py::arg("mpow"),py::arg("a"),py::arg("alpha"),py::arg("D"),
                  py::arg("radii"),
                  py::arg("box_size"),py::arg("x0"),
                  py::arg("ncellx_scale")=1.0,py::arg("balance_omp")=true)
    .def("run",&T::run,"run the algorithm",
         py::arg("n_steps"),py::arg("n_save"),py::arg("n_rec"),
         py::arg("starting_step")=0,
         py::arg("cutoff")=1.0,
         py::arg("output_dir") = "./",
         py::arg("save_mode") = "concise",
         py::arg("compression") = false)
    .def("set_zoom_steps",&T::set_zoom_steps,
          py::arg("starting_step"),py::arg("ending_step"),py::arg("zoom_rate"));
}

// this is the helper function for py_bind [INVERSEPOWER_CORR_STOCHASTICDYN] classes
// the constructor format matches [INVERSEPOWER_CORR_STOCHASTICDYN] classes only
template <typename T>
void bind_inversepower_correlated_sd(py::module& m,std::string class_name){
    py::class_<T>(m,class_name.c_str())
    .def(py::init<double, double, double, double, double, double,
                  std::vector<double> const, 
                  std::vector<double> const,
                  std::vector<double>,
                  const double, const bool>(),
                  py::arg("mpow"),py::arg("a"),py::arg("alpha"),py::arg("D"),py::arg("corr"),py::arg("Dtherm"),
                  py::arg("radii"),
                  py::arg("box_size"),py::arg("x0"),
                  py::arg("ncellx_scale")=1.0,py::arg("balance_omp")=true)
    .def("run",&T::run,"run the algorithm",
         py::arg("n_steps"),py::arg("n_save"),py::arg("n_rec"),
         py::arg("starting_step")=0,
         py::arg("cutoff")=1.0,
         py::arg("output_dir") = "./",
         py::arg("save_mode") = "concise",
         py::arg("compression") = false)
    .def("set_zoom_steps",&T::set_zoom_steps,
          py::arg("starting_step"),py::arg("ending_step"),py::arg("zoom_rate"));
}

// this is the helper function for py_bind [INVERSEPOWER_CORR_UNDIRECTED_STOCHASTICDYN] classes
// the constructor format matches [INVERSEPOWER_CORR_UNDIRECTED_STOCHASTICDYN] classes only
template <typename T>
void bind_inversepower_correlated_undirected_sd(py::module& m,std::string class_name){
    py::class_<T>(m,class_name.c_str())
    .def(py::init<double, double, double, double, double, double,
                  std::vector<double> const, 
                  std::vector<double> const,
                  std::vector<double>,
                  const double, const bool>(),
                  py::arg("mpow"),py::arg("a"),py::arg("alpha"),py::arg("D"),py::arg("corr"),py::arg("Dtherm"),
                  py::arg("radii"),
                  py::arg("box_size"),py::arg("x0"),
                  py::arg("ncellx_scale")=1.0,py::arg("balance_omp")=true)
    .def("run",&T::run,"run the algorithm",
         py::arg("n_steps"),py::arg("n_save"),py::arg("n_rec"),
         py::arg("starting_step")=0,
         py::arg("cutoff")=1.0,
         py::arg("output_dir") = "./",
         py::arg("save_mode") = "concise",
         py::arg("compression") = false)
    .def("set_zoom_steps",&T::set_zoom_steps,
          py::arg("starting_step"),py::arg("ending_step"),py::arg("zoom_rate"));
}

#endif
