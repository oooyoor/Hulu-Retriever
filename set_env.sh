#!/bin/bash
if [ ! -d "external" ]; then
    mkdir external
fi
cd external
if [ ! -d "hnswlib" ]; then
    git submodule add git@github.com:nmslib/hnswlib.git
fi
if [ ! -d "spdlog" ]; then
    git submodule add git@github.com:gabime/spdlog.git
fi
if [ ! -f "json.hpp" ]; then    
    wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
fi