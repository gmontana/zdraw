// Wired-memory ballast for the memory-ladder sweeps (ledger
// memory-ladder-w0-instrument-20261006). Allocates N GiB of anonymous memory in
// 256 MiB chunks, mlocks each chunk so it cannot be compressed or paged out, and
// blocks until SIGTERM/SIGINT. When mlock is refused (RLIMIT_MEMLOCK), the
// chunk stays allocated, filled with incompressible pseudo-random data, and a
// background pass touches every page every two seconds so it stays resident.
// Prints the wired and the merely-touched GiB.
//
//   cc -O2 -o ballast ballast.c
//   ./ballast 100        # leave hw.memsize - 100 GiB free, roughly
//
// This approximates memory pressure (macOS has no cgroups); the engine under
// test runs in another process and reports its own footprint, wall time and
// hash, which is what tools/ballast_sweep.sh records.
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static volatile sig_atomic_t stop = 0;
static void fill_random(void* p, size_t n) {
    uint64_t x = 0x9E3779B97F4A7C15ULL ^ (uint64_t)(uintptr_t)p;
    uint64_t* w = p;
    for (size_t i = 0; i < n / sizeof(uint64_t); i++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        w[i] = x;
    }
}
static void on_signal(int sig) { (void)sig; stop = 1; }

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: ballast <GiB>\n");
        return 2;
    }
    double want_gib = atof(argv[1]);
    const size_t chunk = (size_t)256 << 20;
    size_t want = (size_t)(want_gib * (double)(1ULL << 30));
    size_t wired = 0, touched = 0;
    size_t count = want / chunk + 1;
    void** chunks = calloc(count, sizeof(void*));
    size_t n = 0;
    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);
    while (wired + touched < want && n < count) {
        void* p = mmap(NULL, chunk, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p == MAP_FAILED) break;
        fill_random(p, chunk); /* incompressible: the memory compressor cannot
                                  fold a constant pattern away and fake the pressure */
        chunks[n++] = p;
        if (mlock(p, chunk) == 0) wired += chunk;
        else touched += chunk;
    }
    printf("ballast: wired %.2f GiB, touched-only %.2f GiB (asked %.2f)\n",
           wired / 1073741824.0, touched / 1073741824.0, want_gib);
    fflush(stdout);
    while (!stop) {
        sleep(2);
        if (touched == 0) continue;
        for (size_t i = 0; i < n; i++) {
            volatile unsigned char* b = chunks[i];
            for (size_t off = 0; off < chunk; off += 16384) b[off] = b[off] + 1;
        }
    }
    printf("ballast: released\n");
    return 0;
}
