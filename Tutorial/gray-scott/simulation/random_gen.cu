#include "random_gen.h"

int curand_init_gen(double * devdata, int n, curandGenerator_t *gen) {
    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(gen, CURAND_RNG_PSEUDO_MT19937));
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(*gen, 1234ULL));
}

int gen_curand_values(double *devdata, int n, curandGenerator_t gen) {
    /* Generate n doubles on device */
    CURAND_CALL(curandGenerateUniformDouble(gen, devdata, n));
    
    printf("Done with curand\n");
    fflush(stdout);
    return EXIT_SUCCESS;
}

int curand_cleanup(curandGenerator_t gen) {
    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
}

