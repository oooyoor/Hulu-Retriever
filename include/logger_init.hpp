#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <filesystem>
#include "global_config.hpp"

inline void init_logger(bool is_debug)
{
    std::filesystem::create_directories("logs");

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/run.log", true);
    file_sink->set_level(spdlog::level::debug);

    std::vector<spdlog::sink_ptr> sinks { console_sink, file_sink };
    auto logger = std::make_shared<spdlog::logger>("global_logger", sinks.begin(), sinks.end());

    logger->set_level(is_debug ? spdlog::level::debug : spdlog::level::info);
    logger->flush_on(spdlog::level::debug);

    GlobalConfig::instance().logger = logger;
}
