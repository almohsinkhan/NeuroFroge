#include "execution.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <atomic>
#include <stdexcept>

namespace {

std::atomic<std::size_t> threshold{1};
std::atomic<int> configured_threads{1};

}

namespace execution {

bool openmpAvailable() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

std::size_t parallelThreshold() {
    return threshold.load();
}

void setParallelThreshold(std::size_t value) {
    if (value == 0) {
        throw std::invalid_argument(
            "Parallel threshold must be positive"
        );
    }

    threshold.store(value);
}

int threadCount() {
    return configured_threads.load();
}

void setThreadCount(int value) {
    if (value <= 0) {
        throw std::invalid_argument(
            "Thread count must be positive"
        );
    }

    configured_threads.store(value);

#ifdef _OPENMP
    omp_set_num_threads(value);
#endif
}

}
