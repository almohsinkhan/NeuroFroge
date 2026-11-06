#pragma once

#include <cstddef>

namespace execution {

bool openmpAvailable();

std::size_t parallelThreshold();
void setParallelThreshold(std::size_t threshold);

int threadCount();
void setThreadCount(int thread_count);

template <typename Function>
void parallelFor(std::size_t work_items, Function function) {
#ifdef _OPENMP
    if (work_items >= parallelThreshold()) {
        #pragma omp parallel for
        for (long long index = 0;
             index < static_cast<long long>(work_items);
             index++) {
            function(static_cast<std::size_t>(index));
        }
        return;
    }
#endif

    for (std::size_t index = 0; index < work_items; index++) {
        function(index);
    }
}

}
