#include <iostream>
#include <fstream>
#include <mpi.h>
#include <vector>

#include <adios2.h>

#include "gray-scott.h"

void print_io_settings(const adios2::IO &io)
{
    std::cout << "Simulation writes data using engine type:              " << io.EngineType() << std::endl;
}

void print_settings(const Settings &s)
{
    std::cout << "grid:             " << s.L << "x" << s.L << "x" << s.L
              << std::endl;
    std::cout << "steps:            " << s.steps << std::endl;
    std::cout << "plotgap:          " << s.plotgap << std::endl;
    std::cout << "F:                " << s.F << std::endl;
    std::cout << "k:                " << s.k << std::endl;
    std::cout << "dt:               " << s.dt << std::endl;
    std::cout << "Du:               " << s.Du << std::endl;
    std::cout << "Dv:               " << s.Dv << std::endl;
    std::cout << "noise:            " << s.noise << std::endl;
    std::cout << "output:           " << s.output << std::endl;
    std::cout << "adios_config:     " << s.adios_config << std::endl;
}

void print_simulator_settings(const GrayScott &s)
{
    std::cout << "decomposition:    " << s.npx << "x" << s.npy << "x" << s.npz
              << std::endl;
    std::cout << "grid per process: " << s.size_x << "x" << s.size_y << "x"
              << s.size_z << std::endl;
}

double start_timer(MPI_Comm comm)
{
    MPI_Barrier(comm);
    return MPI_Wtime();
}

void end_timer(double timer_start, int rank, MPI_Comm comm, std::string profile_target, std::string app_name, std::ofstream &my_timer_log)
{
    double timer_end_my = MPI_Wtime();
    MPI_Barrier(comm);
    double timer_end_barrier = MPI_Wtime();
    my_timer_log << app_name << " PE " << std::to_string(rank) << " time for " << profile_target << ": my: " << std::to_string(timer_end_my - timer_start) << ", barrier: " << std::to_string(timer_end_barrier - timer_start) << std::endl;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, procs, wrank;
    double timestamp_t1;
    std::ofstream my_timer_log;

    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    const unsigned int color = 1;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Too few arguments" << std::endl;
            std::cerr << "Usage: grayscott settings.json" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    Settings settings = Settings::from_json(argv[1]);

    if (settings.L % procs != 0) {
        if (rank == 0) {
            std::cerr << "L must be divisible by the number of processes"
                      << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    GrayScott sim(settings, comm);

    sim.init();
    std::string my_io_log_filename = argv[0];
    my_io_log_filename += "_timer_log_PE_" + std::to_string(rank) + ".txt";
    my_timer_log.open(my_io_log_filename, std::ios::trunc);
    if (!my_timer_log.is_open()) {
        std::cout << "FATAL ERROR: Could not open timer log " << my_io_log_filename << ". Quitting." << std::endl;
        MPI_Finalize();
        return -1;
    }

    adios2::ADIOS adios(settings.adios_config, comm, adios2::DebugON);

    adios2::IO io = adios.DeclareIO("SimulationOutput");

    if (rank == 0) {
        print_io_settings(io);
        std::cout << "========================================" << std::endl;
        print_settings(settings);
        print_simulator_settings(sim);
        std::cout << "========================================" << std::endl;
    }

    io.DefineAttribute<double>("F", settings.F);
    io.DefineAttribute<double>("k", settings.k);
    io.DefineAttribute<double>("dt", settings.dt);
    io.DefineAttribute<double>("Du", settings.Du);
    io.DefineAttribute<double>("Dv", settings.Dv);
    io.DefineAttribute<double>("noise", settings.noise);

    adios2::Variable<double> varU = io.DefineVariable<double>(
        "U", {sim.npz * sim.size_z, sim.npy * sim.size_y, sim.npx * sim.size_x},
        {sim.pz * sim.size_z, sim.py * sim.size_y, sim.px * sim.size_x},
        {sim.size_z, sim.size_y, sim.size_x});

    adios2::Variable<double> varV = io.DefineVariable<double>(
        "V", {sim.npz * sim.size_z, sim.npy * sim.size_y, sim.npx * sim.size_x},
        {sim.pz * sim.size_z, sim.py * sim.size_y, sim.px * sim.size_x},
        {sim.size_z, sim.size_y, sim.size_x});

    adios2::Variable<int> varStep = io.DefineVariable<int>("step");

    timestamp_t1 = start_timer(comm);
    adios2::Engine writer = io.Open(settings.output, adios2::Mode::Write);
    end_timer(timestamp_t1, rank, comm, "adios2 open", argv[0], my_timer_log);
    

    for (int i = 0; i < settings.steps; i++) {
        my_timer_log << "Step " << std::to_string(i) << std::endl;
        timestamp_t1 = start_timer(comm);
        sim.iterate();
        end_timer(timestamp_t1, rank, comm, "sim.iterate", argv[0], my_timer_log);

        if (i % settings.plotgap == 0) {
            if (rank == 0) {
                std::cout << "Simulation at step " << i 
                          << " writing output step     " << i/settings.plotgap 
                          << std::endl;
            }
            timestamp_t1 = start_timer(comm);
            std::vector<double> u = sim.u_noghost();
            std::vector<double> v = sim.v_noghost();
            end_timer(timestamp_t1, rank, comm, "noghost copy", argv[0], my_timer_log);

            timestamp_t1 = start_timer(comm);
            writer.BeginStep();
            writer.Put<int>(varStep, &i);
            writer.Put<double>(varU, u.data());
            writer.Put<double>(varV, v.data());
            writer.EndStep();
            end_timer(timestamp_t1, rank, comm, "begin- and end-step", argv[0], my_timer_log);
        }
    }

    timestamp_t1 = start_timer(comm);
    writer.Close();
    end_timer(timestamp_t1, rank, comm, "adios2 close", argv[0], my_timer_log);

    my_timer_log.close();
    MPI_Finalize();
}

