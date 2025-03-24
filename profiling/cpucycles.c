#include "cpucycles.h"

static int fd_cycles, fd_instructions;
// static int fd_stalled_cycles_frontend, fd_stalled_cycles_backend;

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd,
                   flags);
}

static int init_perf_counter(int *fd, int config)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    *fd = perf_event_open(&attr, 0, -1, -1, 0);
    if (*fd == -1) {
        perror("perf_event_open");
        return -1;
    }
    return 0;
}

static inline void start_counter(int fd)
{
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
}

static inline void stop_counter(int fd)
{
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
}

static inline uint64_t read_counter(int fd)
{
    uint64_t count;
    if (read(fd, &count, sizeof(count)) == -1) {
        perror("read_counter");
        return 0;
    }
    return count;
}

// SSR: start, stop, read
#define IMPL_SSR(NAME)                  \
    void start_##NAME()                 \
    {                                   \
        start_counter(fd_##NAME);       \
    }                                   \
    void stop_##NAME()                  \
    {                                   \
        stop_counter(fd_##NAME);        \
    }                                   \
    uint64_t read_##NAME()              \
    {                                   \
        return read_counter(fd_##NAME); \
    }

int init_perf_counters()
{
    if (init_perf_counter(&fd_cycles, PERF_COUNT_HW_CPU_CYCLES) == -1) {
        perror("init_perf_counter fd_cycles");
        return -1;
    }

    if (init_perf_counter(&fd_instructions, PERF_COUNT_HW_INSTRUCTIONS) ==
        -1) {
        perror("init_perf_counter fd_instructions");
        return -1;
    }

    // if (init_perf_counter(&fd_stalled_cycles_frontend,
    //                       PERF_COUNT_HW_STALLED_CYCLES_FRONTEND) == -1) {
    //     perror("init_perf_counter fd_stalled_cycles_frontend");
    //     return -1;
    // }

    // if (init_perf_counter(&fd_stalled_cycles_backend,
    //                       PERF_COUNT_HW_STALLED_CYCLES_BACKEND) == -1) {
    //     perror("init_perf_counter fd_stalled_cycles_backend");
    //     return -1;
    // }
}

IMPL_SSR(cycles)
IMPL_SSR(instructions)
// IMPL_SSR(stalled_cycles_frontend)
// IMPL_SSR(stalled_cycles_backend)
