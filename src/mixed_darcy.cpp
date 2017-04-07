
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;


// Define source terms and boundary functions:
void fFun(const Vector &x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);


int main(int argc, char *argv[]){
    // Parse command line options:
    const char *mesh_file = "../meshes/cube.msh";
    int order = 1;
    
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                    "Finite Element order.");

    args.Parse();
    if (!args.Good()){
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    // Read mesh:
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // Refine:
    {
        int ref_levels = 
            (int)floor(log(20000./mesh->GetNE())/log(2.)/dim);
        for (int l=0; l < ref_levels; l++){
            mesh->UniformRefinement();
        }
    }
    
    // Define finite element spaces:
    FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
    FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

    FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
    FiniteElementSpace *W_space = new FiniteElementSpace(mesh, l2_coll);

    // Define block structure:
    Array<int> block_offsets(3);    // Number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = W_space->GetVSize();
    block_offsets.PartialSum();
    
    cout << "*********************************************************\n";
    cout << "dim(R) = " << R_space->GetVSize() << "\n";
    cout << "dim(W) = " << W_space->GetVSize() << "\n";
    cout << "*********************************************************\n";

    // Define coefficients:
    ConstantCoefficient k(1.0);

    // Define RHS:
    VectorFunctionCoefficient fcoeff(dim, fFun);
    FunctionCoefficient fnatcoeff(f_natural);
    FunctionCoefficient gcoeff(gFun);

    // Allocate memory for solution and rhs:
    BlockVector x(block_offsets), rhs(block_offsets);
    
    LinearForm *fform(new LinearForm);
    fform->Update(R_space, rhs.GetBlock(0),0);
    fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
    fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
    fform->Assemble();

    LinearForm *gform(new LinearForm);
    gform->Update(W_space, rhs.GetBlock(1), 0);
    gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
    gform->Assemble();

    // Assemble FE matrices:
    BilinearForm *mVarf(new BilinearForm(R_space));
    MixedBilinearForm *bVarf(new MixedBilinearForm(R_space, W_space));
    
    mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
    mVarf->Assemble();
    mVarf->Finalize();
    SparseMatrix &M(mVarf->SpMat());

    bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    bVarf->Assemble();
    bVarf->Finalize();
    SparseMatrix & B(bVarf->SpMat());
    B *= -1;
    SparseMatrix *BT = Transpose(B);

    BlockMatrix darcyMatrix(block_offsets);
    darcyMatrix.SetBlock(0,0, &M);
    darcyMatrix.SetBlock(0,1, BT);
    darcyMatrix.SetBlock(1,0, &B);

    // Construct preconditioner:
    SparseMatrix *MinvBt = Transpose(B);
    Vector Md(M.Height());
    M.GetDiag(Md);
    for (int i=0; i < Md.Size(); i++){
        MinvBt->ScaleRow(i, 1./Md(i));
    }
    SparseMatrix *S = Mult(B, *MinvBt);

    Solver *invM, *invS;
    invM = new DSmoother(M);
#ifndef MFEM_USE_SUITESPARSE
    invS = new GSSmoother(*S);
#else
    invS = new UMFPackSolver(*S);
#endif
    invM->iterative_mode = false;
    invS->iterative_mode = false;

    BlockDiagonalPreconditioner darcyPrec(block_offsets);
    darcyPrec.SetDiagonalBlock(0, invM);

    // Solve with MINRES:
    int maxIter(1000);
    double rtol(1.e-6);
    double atol(1.e-10);

    MINRESSolver solver;
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(maxIter);
    solver.SetOperator(darcyMatrix);
    solver.SetPreconditioner(darcyPrec);
    solver.SetPrintLevel(1);
    x = 0.0;
    solver.Mult(rhs, x);
    
    if (solver.GetConverged()){
        cout << "MINRES converged in " << solver.GetNumIterations()
             << " iterations with a residual norm of " << solver.GetFinalNorm() << endl;
    }
    else{
        cout << "MINRES was unable to converge in " << solver.GetNumIterations()
             << " iterations. Residual norm is " << solver.GetFinalNorm() << endl;
    }
    // Create grid functions:
    GridFunction u,p;
    u.MakeRef(R_space, x.GetBlock(0),0);
    p.MakeRef(W_space, x.GetBlock(1),0);

    int order_quad = max(2, 2*order+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i<Geometry::NumGeom; ++i){
        irs[i] = &(IntRules.Get(i, order_quad));
    }
    
    // Save mesh and solution:
    ofstream darcy_ofs("darcy_solution.vtk");
    darcy_ofs.precision(8);
    mesh->PrintVTK(darcy_ofs, 1, 2);
    u.SaveVTK(darcy_ofs, "u", 1);
    p.SaveVTK(darcy_ofs, "p", 1);
    
    // Free memory:
    delete invM;
    delete invS;
    delete bVarf;
    delete mVarf;
    delete gform;
    delete fform;
    delete W_space;
    delete R_space;
    delete l2_coll;
    delete hdiv_coll;
    delete mesh;
    return 0;
}

void fFun(const Vector & x, Vector & f){
    f(0) = 10.*x(1)*(1.-x(0));
    f(1) = -10.*x(0)*(1.-x(1));
    f(2) = -1.0;
}

double gFun(const Vector & x){
    return 0.;
}

double f_natural(const Vector & x){
    return 0.;
}
