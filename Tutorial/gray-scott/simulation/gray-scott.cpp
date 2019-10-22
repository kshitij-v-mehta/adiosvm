// The solver is based on Hiroshi Watanabe's 2D Gray-Scott reaction diffusion
// code available at:
// https://github.com/kaityo256/sevendayshpc/tree/master/day5

#include <mpi.h>
#include <random>
#include <iostream>

#include "gray-scott.h"

#ifdef _OPENACC
#include <cuda.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#endif

GrayScott::GrayScott(const Settings &settings, MPI_Comm comm)
    : settings(settings), comm(comm), rand_dev(), mt_gen(rand_dev()),
      uniform_dist(-1.0, 1.0)
{
}

GrayScott::~GrayScott() {

}

void GrayScott::init()
{
    init_mpi();
    init_field();

#ifdef _OPENACC
    //curand_init((unsigned long long)clock(), 0, 0, &cuRand_state);
    curand_init((unsigned long long)MPI_Wtime(), 0, 0, &cuRand_state);
#endif
}

void GrayScott::iterate()
{
    exchange(u, v);
    calc(u, v, u2, v2);

    // Swap u and u2
    double *tmp;
    tmp = u;
    u = u2;
    u2 = tmp;

    // Swap v and v2
    tmp = v;
    v = v2;
    v2 = tmp;
}

double* GrayScott::u_ghost() const { return u; }

double* GrayScott::v_ghost() const { return v; }

double* GrayScott::u_noghost() const { return data_noghost(u); }

double* GrayScott::v_noghost() const { return data_noghost(v); }

void GrayScott::u_noghost(double *u_no_ghost) const
{
    data_noghost(u, u_no_ghost);
}

void GrayScott::v_noghost(double *v_no_ghost) const
{
    data_noghost(v, v_no_ghost);
}

double*
GrayScott::data_noghost(const double *data) const
{
    double *buf = (double*) malloc (size_x * size_y * size_z * sizeof(double));
    if (NULL == buf) {
        std::cerr << "Could not allocate buf in data_noghost" << std::endl;
        MPI_Abort(comm, -1);
    }
    data_no_ghost_common(data, buf);
    return buf;
}

void GrayScott::data_noghost(const double *data,
                             double *data_no_ghost) const
{
    data_no_ghost_common(data, data_no_ghost);
}

void GrayScott::data_no_ghost_common(const double *data,
                                     double *data_no_ghost) const
{
    for (int z = 1; z < size_z + 1; z++) {
        for (int y = 1; y < size_y + 1; y++) {
            for (int x = 1; x < size_x + 1; x++) {
                data_no_ghost[(x - 1) + (y - 1) * size_x +
                              (z - 1) * size_x * size_y] = data[l2i(x, y, z)];
            }
        }
    }
}

void GrayScott::init_field()
{
    V = (size_x + 2) * (size_y + 2) * (size_z + 2);
    u = (double*) malloc (V * sizeof(double));
    v = (double*) malloc (V * sizeof(double));
    u2 = (double*) malloc (V * sizeof(double));
    v2 = (double*) malloc (V * sizeof(double));

    if ( (NULL == u) || (NULL == v) || (NULL == u2) || (NULL == v2) ) {
        std::cerr << "Could not allocate arrays" << std::endl;
        MPI_Abort(comm, -1);
    }

#pragma acc parallel loop independent
    for (int i=0; i<V; i++) {
        u[i] = 1.0;
        v[i] = 0.0;
        u2[i] = 0.0;
        v2[i] = 0.0;
    }

    const int d = 6;
#pragma acc parallel loop independent collapse(3)
    for (int z = settings.L / 2 - d; z < settings.L / 2 + d; z++) {
        for (int y = settings.L / 2 - d; y < settings.L / 2 + d; y++) {
            for (int x = settings.L / 2 - d; x < settings.L / 2 + d; x++) {
                if (!is_inside(x, y, z)) continue;
                int i = g2i(x, y, z);
                u[i] = 0.25;
                v[i] = 0.33;
            }
        }
    }
}

#pragma acc routine
double GrayScott::calcU(double tu, double tv) const
{
    return -tu * tv * tv + settings.F * (1.0 - tu);
}

#pragma acc routine
double GrayScott::calcV(double tu, double tv) const
{
    return tu * tv * tv - (settings.F + settings.k) * tv;
}

#pragma acc routine
double GrayScott::laplacian(int x, int y, int z,
                            double *s) const
{
    double ts = 0.0;
    ts += s[l2i(x - 1, y, z)];
    ts += s[l2i(x + 1, y, z)];
    ts += s[l2i(x, y - 1, z)];
    ts += s[l2i(x, y + 1, z)];
    ts += s[l2i(x, y, z - 1)];
    ts += s[l2i(x, y, z + 1)];
    ts += -6.0 * s[l2i(x, y, z)];

    return ts / 6.0;
}

void GrayScott::calc(double *u, double *v, double *u2, double *v2)
{
#pragma acc parallel loop independent collapse(3)
    for (int z = 1; z < size_z + 1; z++) {
        for (int y = 1; y < size_y + 1; y++) {
            for (int x = 1; x < size_x + 1; x++) {
                const int i = l2i(x, y, z);
                double du = 0.0;
                double dv = 0.0;
                du = settings.Du * laplacian(x, y, z, u);
                dv = settings.Dv * laplacian(x, y, z, v);
                du += calcU(u[i], v[i]);
                dv += calcV(u[i], v[i]);
#ifdef _OPENACC
                du += settings.noise * curand_uniform_double(&cuRand_state);
#else
                du += settings.noise * uniform_dist(mt_gen);
#endif
                u2[i] = u[i] + du * settings.dt;
                v2[i] = v[i] + dv * settings.dt;
            }
        }
    }
}

void GrayScott::init_mpi()
{
    int dims[3] = {};
    const int periods[3] = {1, 1, 1};
    int coords[3] = {};

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    MPI_Dims_create(procs, 3, dims);
    npx = dims[0];
    npy = dims[1];
    npz = dims[2];

    MPI_Cart_create(comm, 3, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    px = coords[0];
    py = coords[1];
    pz = coords[2];

    size_x = (settings.L + npx - 1) / npx;
    size_y = (settings.L + npy - 1) / npy;
    size_z = (settings.L + npz - 1) / npz;

    offset_x = size_x * px;
    offset_y = size_y * py;
    offset_z = size_z * pz;

    if (px == npx - 1) {
        size_x -= size_x * npx - settings.L;
    }
    if (py == npy - 1) {
        size_y -= size_y * npy - settings.L;
    }
    if (pz == npz - 1) {
        size_z -= size_z * npz - settings.L;
    }

    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    MPI_Cart_shift(cart_comm, 1, 1, &down, &up);
    MPI_Cart_shift(cart_comm, 2, 1, &south, &north);

    // XY faces: size_x * (size_y + 2)
    MPI_Type_vector(size_y + 2, size_x, size_x + 2, MPI_DOUBLE, &xy_face_type);
    MPI_Type_commit(&xy_face_type);

    // XZ faces: size_x * size_z
    MPI_Type_vector(size_z, size_x, (size_x + 2) * (size_y + 2), MPI_DOUBLE,
                    &xz_face_type);
    MPI_Type_commit(&xz_face_type);

    // YZ faces: (size_y + 2) * (size_z + 2)
    MPI_Type_vector((size_y + 2) * (size_z + 2), 1, size_x + 2, MPI_DOUBLE,
                    &yz_face_type);
    MPI_Type_commit(&yz_face_type);
}

void GrayScott::exchange_xy(double *local_data) const
{
    MPI_Status st;

    // Send XY face z=size_z to north and receive z=0 from south
    MPI_Sendrecv(&local_data[l2i(1, 0, size_z)], 1, xy_face_type, north, 1,
                 &local_data[l2i(1, 0, 0)], 1, xy_face_type, south, 1,
                 cart_comm, &st);
    // Send XY face z=1 to south and receive z=size_z+1 from north
    MPI_Sendrecv(&local_data[l2i(1, 0, 1)], 1, xy_face_type, south, 1,
                 &local_data[l2i(1, 0, size_z + 1)], 1, xy_face_type, north, 1,
                 cart_comm, &st);
}

void GrayScott::exchange_xz(double *local_data) const
{
    MPI_Status st;

    // Send XZ face y=size_y to up and receive y=0 from down
    MPI_Sendrecv(&local_data[l2i(1, size_y, 1)], 1, xz_face_type, up, 2,
                 &local_data[l2i(1, 0, 1)], 1, xz_face_type, down, 2, cart_comm,
                 &st);
    // Send XZ face y=1 to down and receive y=size_y+1 from up
    MPI_Sendrecv(&local_data[l2i(1, 1, 1)], 1, xz_face_type, down, 2,
                 &local_data[l2i(1, size_y + 1, 1)], 1, xz_face_type, up, 2,
                 cart_comm, &st);
}

void GrayScott::exchange_yz(double *local_data) const
{
    MPI_Status st;

    // Send YZ face x=size_x to east and receive x=0 from west
    MPI_Sendrecv(&local_data[l2i(size_x, 0, 0)], 1, yz_face_type, east, 3,
                 &local_data[l2i(0, 0, 0)], 1, yz_face_type, west, 3, cart_comm,
                 &st);
    // Send YZ face x=1 to west and receive x=size_x+1 from east
    MPI_Sendrecv(&local_data[l2i(1, 0, 0)], 1, yz_face_type, west, 3,
                 &local_data[l2i(size_x + 1, 0, 0)], 1, yz_face_type, east, 3,
                 cart_comm, &st);
}

void GrayScott::exchange(double *u, double *v) const
{
    exchange_xy(u);
    exchange_xz(u);
    exchange_yz(u);

    exchange_xy(v);
    exchange_xz(v);
    exchange_yz(v);
}

