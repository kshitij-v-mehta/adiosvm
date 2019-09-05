#include <fstream>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <sstream>
#include <vector>

#include <adios2.h>
#include <mpi.h>

#include "../common/timer.hpp"
#include "gray-scott.h"
#include "writer.h"

void print_io_settings(const adios2::IO &io)
{
    std::cout << "Simulation writes data using engine type:              "
              << io.EngineType() << std::endl;
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
    std::cout << "process layout:   " << s.npx << "x" << s.npy << "x" << s.npz
              << std::endl;
    std::cout << "local grid size:  " << s.size_x << "x" << s.size_y << "x"
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
    double timestamp_t1, timestamp_start_app, timestamp_step_start, timestamp_prev_step;
    std::ofstream my_timer_log;
    std::string prog_name = "gray_scott";

    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    const unsigned int color = 1;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    timestamp_start_app = MPI_Wtime();

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Too few arguments" << std::endl;
            std::cerr << "Usage: gray-scott settings.json" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    Settings settings = Settings::from_json(argv[1]);

    GrayScott sim(settings, comm);
    sim.init();
    std::string my_io_log_filename = prog_name;
    my_io_log_filename += "_timer_log_PE_" + std::to_string(rank) + ".txt";
    my_timer_log.open(my_io_log_filename, std::ios::trunc);
    if (!my_timer_log.is_open()) {
        std::cout << "FATAL ERROR: Could not open timer log " << my_io_log_filename << ". Quitting." << std::endl;
        MPI_Finalize();
        return -1;
    }

    adios2::ADIOS adios(settings.adios_config, comm, adios2::DebugON);
    adios2::IO io_main = adios.DeclareIO("SimulationOutput");
    adios2::IO io_ckpt = adios.DeclareIO("SimulationCheckpoint");

    Writer writer_main(settings, sim, io_main);
    Writer writer_ckpt(settings, sim, io_ckpt);

    writer_main.open(settings.output);

    if (rank == 0) {
        print_io_settings(io_main);
        std::cout << "========================================" << std::endl;
        print_settings(settings);
        print_simulator_settings(sim);
        std::cout << "========================================" << std::endl;
    }

#ifdef ENABLE_TIMERS
    Timer timer_total;
    Timer timer_compute;
    Timer timer_write;

    std::ostringstream log_fname;
    log_fname << "gray_scott_pe_" << rank << ".log";

    std::ofstream log(log_fname.str());
    log << "step\ttotal_gs\tcompute_gs\twrite_gs" << std::endl;
#endif

    for (int i = 0; i < settings.steps;) {
#ifdef ENABLE_TIMERS
        MPI_Barrier(comm);
        timer_total.start();
        timer_compute.start();
#endif

        for (int j = 0; j < settings.plotgap; j++) {
            timestamp_step_start = MPI_Wtime();
            if (i == 0) timestamp_prev_step = timestamp_step_start;
            my_timer_log << prog_name << " ------ Step " << std::to_string(i) << " gap after previous: " << timestamp_step_start - timestamp_prev_step << ", gap from app start: " << timestamp_step_start - timestamp_start_app << " ------" << std::endl;
            timestamp_prev_step = timestamp_step_start;

            timestamp_t1 = start_timer(comm);
            sim.iterate();
            end_timer(timestamp_t1, rank, comm, "sim.iterate", prog_name, my_timer_log);
            i++;
        }

#ifdef ENABLE_TIMERS
        double time_compute = timer_compute.stop();
        MPI_Barrier(comm);
        timer_write.start();
#endif

        if (rank == 0) {
            std::cout << "Simulation at step " << i
                      << " writing output step     " << i / settings.plotgap
                      << std::endl;
        }

        writer_main.write(i, sim);

        if (settings.checkpoint &&
            i % (settings.plotgap * settings.checkpoint_freq) == 0) {
            writer_ckpt.open(settings.checkpoint_output);
            writer_ckpt.write(i, sim);
            writer_ckpt.close();
        }

#ifdef ENABLE_TIMERS
        double time_write = timer_write.stop();
        double time_step = timer_total.stop();
        MPI_Barrier(comm);

        log << i << "\t" << time_step << "\t" << time_compute << "\t"
            << time_write << std::endl;
#endif
    }

    timestamp_t1 = start_timer(comm);
    writer_main.close();
    end_timer(timestamp_t1, rank, comm, "adios2 close", prog_name, my_timer_log);

    my_timer_log.close();

#ifdef ENABLE_TIMERS
    log << "total\t" << timer_total.elapsed() << "\t" << timer_compute.elapsed()
        << "\t" << timer_write.elapsed() << std::endl;

    log.close();
#endif

    std::cout << prog_name << " PE " << rank << " total execution time before barrier: " << MPI_Wtime() - timestamp_start_app << std::endl;
    MPI_Barrier(comm);
    if (rank == 0) std::cout << prog_name << " total execution time: " << MPI_Wtime() - timestamp_start_app << std::endl;
    MPI_Finalize();
}

