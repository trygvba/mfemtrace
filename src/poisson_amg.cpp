// Program for solving Poisson's equation with 
// BoomerAMG preconditioning CGM.
// Run with e.g.:
//      mpirun -np 4 ./poissonAMG 

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
    // Define Finite Element spaces:
    FiniteElementCollection *fec;
    fec = new H1_FECollection(order, dim);
    ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
    HYPRE_Int size = fespace->GlobalTrueVSize();
    if (myid == 0){
        cout << "Number of finite element unknowns: " << size << endl;
    }
    
    // Determine list of BC dofs:
    Array<int> ess_tdof_list;
    if (pmesh->bdr_attributes.Size()){
        Array<int> ess_bdr(pmesh->bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Set up linear RHS:
    ParLinearForm *b = new ParLinearForm(fespace);
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    // Define solution vector:
    ParGridFunction x(fespace);
    x = 0.0;
    
    // Set up bilinear form:
    ParBilinearForm *a = new ParBilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));
    a->Assemble();
    
    HypreParMatrix A;
    Vector B, X;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

    if (myid == 0){
        cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
    }

    // Define parallel PCG:
    HypreSolver *amg = new HypreBoomerAMG(A);
    HyprePCG *pcg = new HyprePCG(A);
    pcg->SetTol(1e-12);
    pcg->SetMaxIter(200);
    pcg->SetPreconditioner(*amg);
    pcg->Mult(B,X);

    // Recover x:
    a->RecoverFEMSolution(X, *b, x);
    {
      ostringstream mesh_name, sol_name;
      mesh_name << "AMGmesh." << setfill('0') << setw(6) << myid;
      sol_name << "AMGsol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
    }
    // Free used memory:
    delete pcg;
    delete amg;
    delete a;
    delete b;
    delete fespace;
    delete fec;
    delete pmesh;

    MPI_Finalize();
    return 0;
}
