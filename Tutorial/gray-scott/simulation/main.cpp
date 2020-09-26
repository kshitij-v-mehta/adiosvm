#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

#include <adios2.h>
#include <mpi.h>

#include "../common/timer.hpp"
#include "gray-scott.h"
#include "writer.h"

#define MIN_ACCURACY 1E-4
#define MAX_ACCURACY 1E-20
double accuracy = 1E-10;
double cur_accuracy = accuracy;

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
    std::cout << "io_threshold:     " << s.io_threshold_percent << std::endl;
    std::cout << "adios_config:     " << s.adios_config << std::endl;
    std::cout << "policy id:        " << s.policy_id << std::endl;
}

void print_simulator_settings(const GrayScott &s)
{
    std::cout << "process layout:   " << s.npx << "x" << s.npy << "x" << s.npz
        << std::endl;
    std::cout << "local grid size:  " << s.size_x << "x" << s.size_y << "x"
        << s.size_z << std::endl;
}

bool policy_1(double total_time, double write_time, double io_threshold_percent, int rank, MPI_Comm comm) {
    if (total_time == 0.0)
        return true;

    double io_threshold = io_threshold_percent/100;
    double write_frac = write_time/total_time;
    double global_write_frac = 0.0;

    MPI_Allreduce(&write_frac, &global_write_frac, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0)
        std::cout << "global write frac: " << global_write_frac << std::endl;

    if(isnan(global_write_frac))
        return true;

    if (global_write_frac <= io_threshold)
        return true;

    return false;
}

bool controller(double total_time, double write_time, int policy_id, double io_threshold_percent, int rank, MPI_Comm comm)
{
    if (policy_id == 1) {
        policy_1(total_time, write_time, io_threshold_percent, rank, comm);
    }

    // else policy 3 on compression
    if (total_time == 0.0)
        return true;

    double io_threshold = io_threshold_percent/100;
    double write_frac = write_time/total_time;
    double global_write_frac = 0.0;

    MPI_Allreduce(&write_frac, &global_write_frac, 1, MPI_DOUBLE, MPI_MAX, comm);

    if(isnan(global_write_frac))
        return true;

    if (global_write_frac <= io_threshold) {
        if (cur_accuracy == accuracy) {
            accuracy = accuracy/10;
            if (accuracy < MAX_ACCURACY)
                accuracy = accuracy * 10;
        }
        
        cur_accuracy = accuracy;
        return true;
    }
    else {
        if (cur_accuracy == accuracy)
            accuracy = accuracy*10;
            if (accuracy > MIN_ACCURACY) {
                accuracy = accuracy/10;
            }
    }

    return false;
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

    double start_time, cur_time, total_time, write_time;

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

    adios2::ADIOS adios(comm, adios2::DebugON);

    if (rank == 0) {
        //print_io_settings(io_bp4);
        std::cout << "========================================" << std::endl;
        print_settings(settings);
        print_simulator_settings(sim);
        std::cout << "========================================" << std::endl;
    }

    Timer timer_total;
    Timer timer_compute;
    Timer timer_write;

    std::ostringstream log_fname;
    log_fname << "gray_scott_pe_" << rank << ".log";

    std::ofstream log(log_fname.str());
    log << "step\ttotal_gs\tcompute_gs\twrite_gs" << std::endl;

    total_time = 0.0;
    write_time = 0.0;
    start_time = MPI_Wtime();

    // -------------------------------------------------------------------- //
    for (int i = 0; i < settings.steps;) {
        bool write_this_step = true;
        double time_write = 0.0;
        
        MPI_Barrier(comm);

        timer_total.start();
        
        timer_compute.start();

        for (int j = 0; j < settings.plotgap; j++) {
            sim.iterate();
            i++;
        }
        
        if (rank == 0)
            std::cout << "Processing step " << i << std::endl;

        double time_compute = timer_compute.stop();

        write_this_step = controller(total_time, write_time, settings.policy_id, settings.io_threshold_percent, rank, comm);
        if (write_this_step) {
            timer_write.start();

            if (rank == 0) {
                cur_time = MPI_Wtime() - start_time;
                std::cout << "[" << cur_time << "] \t"
                    << "Simulation at step " << i
                    << " writing output step     " << i / settings.plotgap
                    << std::endl;
            }

            // Declare IO
            std::ostringstream _io_obj_name;
            _io_obj_name << "Simout-" << i ;
            std::string io_obj_name = _io_obj_name.str();
            adios2::IO io_main = adios.DeclareIO(io_obj_name);
            io_main.SetEngine("BP4");
            io_main.SetParameter("SubStreams", "128");
            
            if (settings.policy_id == 3) {
                if (rank == 0)
                    std::cout << "GS: Accuracy for step " << i << " set to " << accuracy << std::endl;
            }

            // Create output filename
            std::ostringstream _out_fname;
            _out_fname << "gs-" << i << ".bp";

            // Create writer object and open file
            adios2::Operator op = adios.DefineOperator(io_obj_name, "sz");
            Writer writer_main(settings, sim, io_main, accuracy);
            if (settings.policy_id == 3) 
                Writer writer_main(settings, sim, io_main, op, accuracy);

            std::string out_fname = _out_fname.str();
            writer_main.open(out_fname);

            // Write and close
            writer_main.write(i, sim);
            writer_main.close();

            time_write = timer_write.stop();
            write_time += timer_write.elapsed();
        }

        double time_step = timer_total.stop();
        MPI_Barrier(comm);
        total_time += timer_total.elapsed();

        log << i << "\t" << time_step << "\t" << time_compute << "\t"
            << time_write << std::endl;
    }
    // -------------------------------------------------------------------- //

    log << "total\t" << timer_total.elapsed() << "\t" << timer_compute.elapsed()
        << "\t" << timer_write.elapsed() << std::endl;

    log.close();

    MPI_Finalize();
}
