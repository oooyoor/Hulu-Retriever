#!/bin/bash
set -e

EXTERNAL_DIR="external"
mkdir -p "$EXTERNAL_DIR"
if ! git config --file .gitmodules --name-only --get-regexp "submodule.external/hnswlib.path" >/dev/null 2>&1; then
    git submodule add git@github.com:nmslib/hnswlib.git external/hnswlib
fi
if ! git config --file .gitmodules --name-only --get-regexp "submodule.external/spdlog.path" >/dev/null 2>&1; then
    git submodule add git@github.com:gabime/spdlog.git external/spdlog
fi
git submodule update --init --recursive
JSON_PATH="$EXTERNAL_DIR/json.hpp"
if [ ! -f "$JSON_PATH" ]; then
    wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -O "$JSON_PATH"
fi