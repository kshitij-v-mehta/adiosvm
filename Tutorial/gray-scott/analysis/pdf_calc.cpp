/*
 * Analysis code for the Gray-Scott application.
 * Reads variable U and V, and computes the PDF for each 2D slices of U and V.
 * Writes the computed PDFs using ADIOS.
 *
 * Norbert Podhorszki, pnorbert@ornl.gov
 *
 */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <string>
#include <thread>

#include "adios2.h"


bool epsilon(double d) { return (d < 1.0e-20); }
bool epsilon(float d) { return (d < 1.0e-20); }

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

/*
 * Function to compute the PDF of a 2D slice
 */
template <class T> 
void compute_pdf(const std::vector<T> &data,
                           const std::vector<std::size_t> &shape,
                           const size_t start,
                           const size_t count,
                           const size_t nbins,
                           const T min,
                           const T max,
                           std::vector<T> &pdf,
                           std::vector<T> &bins)
{
    if(shape.size() != 3)
        throw std::invalid_argument("ERROR: shape is expected to be 3D\n");
    
    size_t slice_size = shape[1] * shape[2];
    pdf.resize( count * nbins );
    bins.resize (nbins);
    
    size_t start_data = 0;
    size_t start_pdf = 0;

    T binWidth = (max - min)/nbins;
    for (auto i = 0; i < nbins; ++i )
    {
        bins[i] = min + (i * binWidth);
    }

    if (nbins == 1)
    {
        // special case: only one bin
        for (auto i = 0; i < count; ++i )
        {
            pdf[i] = slice_size;
        }
        return;
    }

    if (epsilon(max-min) || epsilon(binWidth))
    {
        // special case: constant array
        for (auto i = 0; i < count; ++i )
        {
            pdf[i*nbins + (nbins/2)] = slice_size;
        }
        return;
    }


    for (auto i = 0; i < count; ++i )
    {
        // Calculate a PDF for 'nbins' bins for values between 'min' and 'max'
        // from data[ start_data .. start_data+slice_size-1 ]
        // into pdf[ start_pdf .. start_pdf+nbins-1 ]
        for (auto j = 0; j < slice_size; ++j )
        {
            if (data[start_data+j] > max || data[start_data+j] < min) 
            {
                std::cout << " data[" << start*slice_size+start_data+j << "] = " 
                    <<  data[start_data+j] << " is out of [min,max] = ["
                    << min << "," << max << "]" << std::endl;
            }
            size_t bin = static_cast<size_t>(std::floor((data[start_data+j] - min)/binWidth));
            if (bin == nbins)
            {
                bin = nbins - 1;
            }
            ++pdf[start_pdf+bin];
        }
        start_pdf += nbins;
        start_data += slice_size;
    }
    return;
}

/*
 * Print info to the user on how to invoke the application
 */
void printUsage()
{
    std::cout
        << "Usage: pdf_calc input output [N] [output_inputdata]\n"
        << "  input:   Name of the input file handle for reading data\n"
        << "  output:  Name of the output file to which data must be written\n"
        << "  N:       Number of bins for the PDF calculation, default = 1000\n"
        << "  output_inputdata: YES will write the original variables besides the analysis results\n\n";
}

/*
 * MAIN
 */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, comm_size, wrank;
    double timestamp_t1;
    std::ofstream my_timer_log;
    std::string prog_name = "pdf_calc";
    double timestamp_app_start, timestamp_step_current, timestamp_step_prev;

    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    const unsigned int color = 2;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    if (argc < 3)
    {
        std::cout << "Not enough arguments\n";
        if (rank == 0)
            printUsage();
        MPI_Finalize();
        return 0;
    }

    timestamp_app_start = MPI_Wtime();
    std::string in_filename;
    std::string out_filename;
    size_t nbins = 1000;
    bool write_inputvars = false;
    in_filename = argv[1];
    out_filename = argv[2];

    if (argc >= 4)
    {
        int value = std::stoi(argv[3]);
        if (value > 0)
            nbins = static_cast<size_t>(value);
    }

    if (argc >= 5)
    {
        std::string value = argv[4];
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        if (value == "yes")
            write_inputvars = true;
    }

    std::string my_io_log_filename = prog_name;
    my_io_log_filename += "_timer_log_PE_" + std::to_string(rank) + ".txt";
    my_timer_log.open(my_io_log_filename, std::ios::trunc);
    if (!my_timer_log.is_open()) {
        std::cout << "FATAL ERROR: Could not open timer log " << my_io_log_filename << ". Quitting." << std::endl;
        MPI_Finalize();
        return -1;
    }


    std::size_t u_global_size, v_global_size;
    std::size_t u_local_size, v_local_size;
    
    bool firstStep = true;

    std::vector<std::size_t> shape;
    
    std::vector<double> u;
    std::vector<double> v;
    int simStep;

    std::vector<double> pdf_u;
    std::vector<double> pdf_v;
    std::vector<double> bins_u;
    std::vector<double> bins_v;
    
    // adios2 variable declarations
    adios2::Variable<double> var_u_in, var_v_in;
    adios2::Variable<int> var_step_in;
    adios2::Variable<double> var_u_pdf, var_v_pdf;
    adios2::Variable<double> var_u_bins, var_v_bins;
    adios2::Variable<int> var_step_out;
    adios2::Variable<double> var_u_out, var_v_out;

    // adios2 io object and engine init
    adios2::ADIOS ad ("adios2.xml", comm, adios2::DebugON);

    // IO objects for reading and writing
    adios2::IO reader_io = ad.DeclareIO("SimulationOutput");
    adios2::IO writer_io = ad.DeclareIO("PDFAnalysisOutput");
    if (!rank) 
    {
        std::cout << "PDF analysis reads from Simulation using engine type:  " << reader_io.EngineType() << std::endl;
        std::cout << "PDF analysis writes using engine type:                 " << writer_io.EngineType() << std::endl;
    }

    // Engines for reading and writing
    timestamp_t1 = start_timer(comm);
    adios2::Engine reader = reader_io.Open(in_filename, adios2::Mode::Read, comm);
    end_timer(timestamp_t1, rank, comm, "reader adios2 open", prog_name, my_timer_log);
    
    timestamp_t1 = start_timer(comm);
    adios2::Engine writer = writer_io.Open(out_filename, adios2::Mode::Write, comm);
    end_timer(timestamp_t1, rank, comm, "writer adios2 open", prog_name, my_timer_log);

    bool shouldIWrite = (!rank || reader_io.EngineType() == "HDF5");

    // read data per timestep
    int stepAnalysis = 0;
    timestamp_step_prev = -1;
    while(true) {

        // Begin step
        timestamp_t1 = start_timer(comm);

        adios2::StepStatus read_status = reader.BeginStep(adios2::StepMode::NextAvailable, 10.0f);
        
        if (read_status == adios2::StepStatus::NotReady)
        {
            my_timer_log << "Step not ready " << std::endl;
            // std::cout << "Stream not ready yet. Waiting...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK)
        {
            my_timer_log << "Exiting loop" << std::endl;
            break;
        }
 
        int stepSimOut = reader.CurrentStep();
        timestamp_step_current = MPI_Wtime();
        if (timestamp_step_prev == -1) timestamp_step_prev = timestamp_step_current;
        my_timer_log << prog_name << " ------ Current Step " << stepSimOut << ", after wait " << MPI_Wtime() - timestamp_t1 << ", gap from previous step " << timestamp_step_current - timestamp_step_prev << ", gap from app start: " << timestamp_step_current - timestamp_app_start << std::endl;
        timestamp_step_prev = timestamp_step_current;

        // Inquire variable and set the selection at the first step only
        // This assumes that the variable dimensions do not change across timesteps

        // Inquire variable
        var_u_in = reader_io.InquireVariable<double>("U");
        var_v_in = reader_io.InquireVariable<double>("V");
        var_step_in = reader_io.InquireVariable<int>("step");

        

        std::pair<double, double> minmax_u =  var_u_in.MinMax();
        std::pair<double, double> minmax_v =  var_v_in.MinMax();

        shape = var_u_in.Shape();

        // Calculate global and local sizes of U and V
        u_global_size = shape[0] * shape[1] * shape[2];
        u_local_size  = u_global_size/comm_size;
        v_global_size = shape[0] * shape[1] * shape[2];
        v_local_size  = v_global_size/comm_size;

        size_t count1 = shape[0]/comm_size;
        size_t start1 = count1 * rank;
        if (rank == comm_size-1) {
            // last process need to read all the rest of slices
            count1 = shape[0] - count1 * (comm_size - 1);
        }

        /*std::cout << "  rank " << rank << " slice start={" <<  start1 
            << ",0,0} count={" << count1  << "," << shape[1] << "," << shape[2]
            << "}" << std::endl;*/

        // Set selection
        var_u_in.SetSelection(adios2::Box<adios2::Dims>(
                    {start1,0,0},
                    {count1, shape[1], shape[2]}));
        var_v_in.SetSelection(adios2::Box<adios2::Dims>(
                    {start1,0,0},
                    {count1, shape[1], shape[2]}));

        // Declare variables to output
        if (firstStep) {
            var_u_pdf = writer_io.DefineVariable<double> ("U/pdf",
                    { shape[0], nbins },
                    { start1, 0 },
                    { count1, nbins } );
            var_v_pdf = writer_io.DefineVariable<double> ("V/pdf",
                    { shape[0], nbins },
                    { start1, 0},
                    { count1, nbins } );

            if (shouldIWrite)
            {
                var_u_bins = writer_io.DefineVariable<double> ("U/bins",
                        { nbins }, { 0 }, { nbins } );
                var_v_bins = writer_io.DefineVariable<double> ("V/bins",
                        { nbins }, { 0 }, { nbins } );
                var_step_out = writer_io.DefineVariable<int> ("step");
            }


            if ( write_inputvars) {
                var_u_out = writer_io.DefineVariable<double> ("U",
                        { shape[0], shape[1], shape[2] },
                        { start1, 0, 0 },
                        { count1, shape[1], shape[2] } );
                var_v_out = writer_io.DefineVariable<double> ("V",
                        { shape[0], shape[1], shape[2] },
                        { start1, 0, 0 },
                        { count1, shape[1], shape[2] } );

            }
            firstStep = false;
        }


        // Read adios2 data
        reader.Get<double>(var_u_in, u);
        reader.Get<double>(var_v_in, v);
        if (shouldIWrite)
        {
            reader.Get<int>(var_step_in, &simStep);
        }

        // End adios2 step
        reader.EndStep();
        end_timer(timestamp_t1, rank, comm, "reader begin- and end-step", prog_name, my_timer_log);

        if (!rank)
        {
            std::cout << "PDF Analysis step " << stepAnalysis
                << " processing sim output step "
                << stepSimOut << " sim compute step " << simStep << std::endl;
        }

        // HDF5 engine does not provide min/max. Let's calculate it
        if (reader_io.EngineType() == "HDF5")
        {
            auto mmu = std::minmax_element(u.begin(), u.end());
            minmax_u = std::make_pair(*mmu.first, *mmu.second);
            auto mmv = std::minmax_element(v.begin(), v.end());
            minmax_v = std::make_pair(*mmv.first, *mmv.second);
        }

        // Compute PDF
        timestamp_t1 = start_timer(comm);
        std::vector<double> pdf_u;
        std::vector<double> bins_u;
        compute_pdf(u, shape, start1, count1, nbins, minmax_u.first, minmax_u.second, pdf_u, bins_u);
        end_timer(timestamp_t1, rank, comm, "compute pdf u", prog_name, my_timer_log);

        timestamp_t1 = start_timer(comm);
        std::vector<double> pdf_v;
        std::vector<double> bins_v;
        compute_pdf(v, shape, start1, count1, nbins, minmax_v.first, minmax_v.second, pdf_v, bins_v);
        end_timer(timestamp_t1, rank, comm, "compute pdf v", prog_name, my_timer_log);

        // write U, V, and their norms out
        timestamp_t1 = start_timer(comm);
        writer.BeginStep ();
        writer.Put<double> (var_u_pdf, pdf_u.data());
        writer.Put<double> (var_v_pdf, pdf_v.data());
        if (shouldIWrite)
        {
            writer.Put<double> (var_u_bins, bins_u.data());
            writer.Put<double> (var_v_bins, bins_v.data());
            writer.Put<int> (var_step_out, simStep);
        }
        if (write_inputvars) {
            writer.Put<double> (var_u_out, u.data());
            writer.Put<double> (var_v_out, v.data());
        }
        writer.EndStep ();
        end_timer(timestamp_t1, rank, comm, "writer begin- and end-step", prog_name, my_timer_log);

        ++stepAnalysis;
    }

    // cleanup
    timestamp_t1 = start_timer(comm);
    reader.Close();
    end_timer(timestamp_t1, rank, comm, "reader adios close", prog_name, my_timer_log);

    timestamp_t1 = start_timer(comm);
    writer.Close();
    end_timer(timestamp_t1, rank, comm, "writer adios close", prog_name, my_timer_log);

    MPI_Barrier(comm);
    if (rank == 0) std::cout << prog_name << " total execution time: " << MPI_Wtime() - timestamp_app_start << std::endl;
    MPI_Finalize();
    return 0;
}

