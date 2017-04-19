// Program for solving Poisson's equation with MFEM:

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]){
    // Parse command line options:
    const char *mesh_file = "../meshes/cube.msh";
    int order = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                    "Finite element order.");
    
    args.Parse();
    if (!args.Good()){
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    // Read the mesh:
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // Refine mesh:
    {
        int ref_levels =
            (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++){
            mesh->UniformRefinement();
        }
    }

    // Print edges:
    cout << "Number of edges: " << mesh->GetNEdges() << endl;
    Array<int> e;
    Table *edges = mesh->GetEdgeVertexTable();
    for (int l=0; l<mesh->GetNEdges(); l++){
        mesh->GetEdgeVertices(l, e);
        cout << "Edge #" << l << ": (" << e[0] << ", " << e[1] << ")" << endl; 
    }

    // Define Finite Element space:
    FiniteElementCollection *fec;
    fec = new H1_FECollection(order,dim);

    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
    cout << "Number of finite element unknowns: "
         << fespace->GetTrueVSize() << endl;

    // Determine boundary DOFs:
    Array<int> ess_tdof_list;
    if (mesh->bdr_attributes.Size()){
        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Set up linear RHS:
    LinearForm *b = new LinearForm(fespace);
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    // Set up bilinear form:
    GridFunction x(fespace);
    x = 0.0;
    BilinearForm *a = new BilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));
    a->Assemble();

    SparseMatrix A;
    Vector B, X;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

    cout << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
    GSSmoother M(A);
    PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
#else
    UMFPackSolver umf_solver;
    umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    umf_solver.SetOperator(A);
    umf_solver.Mult(B,X);
#endif

    // Recover solution:
    a->RecoverFEMSolution(X, *b, x);
    // Save solution:
    ofstream mesh_ofs("poisson_solution.vtk");
    mesh_ofs.precision(8);
    mesh->PrintVTK(mesh_ofs, 1, 2);
    x.SaveVTK(mesh_ofs, "solution", 1);

    // Free used memory:
    delete a;
    delete b;
    delete fespace;
    delete fec;
    delete mesh;

    return 0;
}
