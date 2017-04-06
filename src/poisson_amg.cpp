// Program for solving Poisson's equation with 
// BoomerAMG preconditioning CGM.
// Run with e.g.:
//      mpirun -np 4 poissonAMG 

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    // Initialize MPI:
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Parse command line options:
    const char *mesh_file = "../meshes/cube.msh";
    int order = 1;
    
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                    "Finite element order");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0){
        args.PrintOptions(cout);
    }

    // Read the mesh:
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // Refine mesh:
    int ref_levels =
        (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
    for (int l=0; l<ref_levels; l++){
        mesh->UniformRefinement();
    }

    // Partion mesh and further refine:
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    {
        int par_ref_levels = 2;
        for (int l=0; l < par_ref_levels; l++){
            pmesh->UniformRefinement();
        }
    }
    cout << "OK in process: " << myid << endl;
    return 0;
}
