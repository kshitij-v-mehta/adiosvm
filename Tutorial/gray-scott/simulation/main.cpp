#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <adios2.h>
#include <mpi.h>

#include "../common/timer.hpp"
#include "gray-scott.h"
#include "writer.h"
#include <stdlib.h>

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
    std::cout << "write_data        " << s.write_data << std::endl;
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

std::string indented_string(std::string msg) {
    int n = 32;
    std::string indent(n-msg.length(), ' ');
    return msg+indent;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, procs, wrank;

    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    const unsigned int color = 1;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    double start_time, cur_time;
    int fileio_time = std::stoi(argv[2]);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Too few arguments" << std::endl;
            std::cerr << "Usage: gray-scott settings.json" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    std::ostringstream log_fname;
    log_fname << "gray_scott_pe_" << rank << ".log";

    std::ofstream log(log_fname.str());
    
    Settings settings = Settings::from_json(argv[1]);

    GrayScott sim(settings, log, comm);
    sim.init();

    adios2::ADIOS adios(settings.adios_config, comm, adios2::DebugON);
    adios2::IO io_main = adios.DeclareIO("SimulationOutput");
    adios2::IO io_ckpt = adios.DeclareIO("SimulationCheckpoint");

    Writer writer_main(settings, sim, io_main);
    Writer writer_ckpt(settings, sim, io_ckpt);

    if (settings.write_data) writer_main.open(settings.output);

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
    Timer timer_comm;
    Timer timer_write;

    double time_step, time_compute, time_comm, time_write;

    log << indented_string("step")
        << indented_string("start_timestamp (seconds)") 
        << indented_string("total_gs (ms)") 
        << indented_string("compute_gs (ms)") 
        << indented_string("comm_gs (ms)") 
        << indented_string("write_gs (ms)") 
        << std::endl;
#endif

    start_time = MPI_Wtime();
    for (int i = 0; i < settings.steps;) {
#ifdef ENABLE_TIMERS
        timer_total.start();
        cur_time = MPI_Wtime() - start_time;

        for (int j = 0; j < settings.plotgap; j++) {
            sim.iterate(&timer_compute, &timer_comm, &time_compute, &time_comm, fileio_time);
            i++;
        }

        timer_write.start();
#else
        for (int j = 0; j < settings.plotgap; j++) {
            sim.iterate();
            i++;
        }
#endif

        if (rank == 0) {
            std::cout << "[" << cur_time << "] \t"
                      << "Simulation at step " << i
                      << " writing output step     " << i / settings.plotgap
                      << std::endl;
        }

       if (settings.write_data) writer_main.write(i, sim);

        if (settings.checkpoint &&
            i % (settings.plotgap * settings.checkpoint_freq) == 0) {
            writer_ckpt.open(settings.checkpoint_output);
            writer_ckpt.write(i, sim);
            writer_ckpt.close();
        }

#ifdef ENABLE_TIMERS
        time_write = timer_write.stop();
        time_step = timer_total.stop();

        log << indented_string(std::to_string(i)) 
            << indented_string(std::to_string(cur_time))
            << indented_string(std::to_string(time_step)) 
            << indented_string(std::to_string(time_compute))
            << indented_string(std::to_string(time_comm))
            << indented_string(std::to_string(time_write)) 
            << std::endl;
#endif
    }

    if (settings.write_data) writer_main.close();

#ifdef ENABLE_TIMERS
    log << indented_string("total") 
        << indented_string("")
        << indented_string(std::to_string(timer_total.elapsed()))
        << indented_string(std::to_string(timer_compute.elapsed()))
        << indented_string(std::to_string(timer_comm.elapsed()))
        << indented_string(std::to_string(timer_write.elapsed()))
        << std::endl;

    log.close();
#endif

    MPI_Finalize();
}
