#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <adios2.h>
#include <mpi.h>

#include "../common/timer.hpp"
#include "gray-scott.h"
#include "writer.h"

#define INSITUMPI 1
#define BP4 2

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

bool controller(double total_time, double write_time, MPI_Comm comm)
{
    double allow_ratio = 0.3;
    double write_frac = write_time/total_time;
    double global_write_frac = 0.0;

    MPI_Allreduce(&write_frac, &global_write_frac, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (global_write_frac <= 0.4) return true;
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

    double start_time, cur_time;

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

    /* if (rank == 0) {
        print_io_settings(io_bp4);
        std::cout << "========================================" << std::endl;
        print_settings(settings);
        print_simulator_settings(sim);
        std::cout << "========================================" << std::endl;
    } */

    Timer timer_total;
    Timer timer_compute;
    Timer timer_write;

    std::ostringstream log_fname;
    log_fname << "gray_scott_pe_" << rank << ".log";

    std::ofstream log(log_fname.str());
    log << "step\ttotal_gs\tcompute_gs\twrite_gs" << std::endl;

    int pending = -1;
    int ack = -1;
    MPI_Request request;
    MPI_Status status;
    int flag = -1;
    int engineid = -1;
    start_time = MPI_Wtime();

    // -------------------------------------------------------------------- //
    for (int i = 0; i < settings.steps; i++) {
        bool write_this_step = true;
        double time_write = 0.0;
        
        MPI_Barrier(comm);
        timer_total.start();
        
        timer_compute.start();
        sim.iterate();
        double time_compute = timer_compute.stop();

        write_this_step = true;
        if (write_this_step) {
            timer_write.start();

            // Declare IO
            std::ostringstream _io_obj_name;
            _io_obj_name << "Simout-" << i ;
            std::string io_obj_name = _io_obj_name.str();
            adios2::IO io_main = adios.DeclareIO(io_obj_name);
            
            // Wait if there is an outstanding request
            engineid = INSITUMPI;
            if (rank == 0) {
                if (pending > -1) {
                    MPI_Test(&request, &flag, &status);
                    if (!flag)  //pdf_calc has not returned
                        engineid = BP4;
                    else
                        pending = 0;
                }
            }
            MPI_Bcast(&engineid, 1, MPI_INT, 0, comm);
            
            //Send file id to the pdf_calc root
            if ((engineid == INSITUMPI) && (rank == 0))
                MPI_Send(&i, 1, MPI_INT, procs, 0, MPI_COMM_WORLD);

            if (engineid == INSITUMPI)
                io_main.SetEngine("InSituMPI");
            else
                io_main.SetEngine("BP4");
            io_main.SetParameter("SubStreams", "128");

            if (rank == 0) {
                std::string enginename = "InSituMPI";
                if (engineid == BP4) enginename = "BP4";
                cur_time = MPI_Wtime() - start_time;
                std::cout << "GS: [" << cur_time << "]\t sending file " << i << " to " << enginename << std::endl;
            }
            
            // Create output filename
            std::ostringstream _out_fname;
            _out_fname << "gs-" << i << ".bp";

            // Create writer object and open file
            Writer writer_main(settings, sim, io_main);
            std::string out_fname = _out_fname.str();
            writer_main.open(out_fname);

            // Write and close
            writer_main.write(i, sim);
            writer_main.close();

            // Post a non-blocking recv from pdf_calc about completion
            if ((engineid == INSITUMPI) && (rank == 0)) {
                pending = 1;
                MPI_Irecv(&ack, 1, MPI_INT, procs, 0, MPI_COMM_WORLD, &request);
            }

            time_write = timer_write.stop();
        }

        double time_step = timer_total.stop();
        MPI_Barrier(comm);

        log << i << "\t" << time_step << "\t" << time_compute << "\t"
            << time_write << std::endl;
    }
    
    //Send term signal to the pdf_calc root
    int signal = -1;
    if (rank == 0)
        MPI_Send(&signal, 1, MPI_INT, procs, 0, MPI_COMM_WORLD);

    // -------------------------------------------------------------------- //

    log << "total\t" << timer_total.elapsed() << "\t" << timer_compute.elapsed()
        << "\t" << timer_write.elapsed() << std::endl;

    log.close();

    MPI_Finalize();
}
