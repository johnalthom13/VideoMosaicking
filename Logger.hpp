#ifndef LOGGER_HPP
#define LOGGER_HPP
#include <iostream>

static bool debugging_enabled = true;

#define LOG_START(x) (std::cout << "Starting: " << (x) << std::endl)
#define LOG_FINISH(x) (std::cout << "Done: " << (x) << std::endl)
#define LOG_DEBUG(x) do { \
if (debugging_enabled) { std::cout << x << std::endl; } \
} while (0)
#endif LOGGER_HPP
