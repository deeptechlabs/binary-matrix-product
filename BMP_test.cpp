#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>

#include "BMP.h"

using Eigen::MatrixXf;
using namespace std;

float float_sign(float x)
{
    // return (x>=0);
    return 2. * (x>=0) - 1.;
}

int main(int argc, char* argv[])
{   
    /*
    cout << endl << "-- Testing concatenate and deconcatenate part 1 --" << endl;
    
    float * A = new float[32];
    for(int i = 0; i<32; i++)
    {
        A[i] = rand() % 2;
    }
    cout<<endl<<"A = ["<<endl;
    for(int i = 0; i<32; i++)
    {
        cout<<A[i]<<"  ";
    }
    cout<<endl<<"]"<<endl;
    
    unsigned long int a = concatenate(A);
    cout<<endl<<"concatenate(A) = "<<a<<endl;
    
    A = deconcatenate(a);
    cout<<endl<<"deconcatenate(concatenate(A)) = ["<<endl;
    for(int i = 0; i<32; i++)
    {
        cout<<A[i]<<"  ";
    }
    cout<<endl<<"]"<<endl;
    
    
    cout << endl << "-- Testing concatenate and deconcatenate part 2 --" << endl;
    
    int N = 1024;
    
    // A is a matrix filled with 1 and 0s.
    MatrixXf B(N,N);
    B.setRandom();
    B = B.unaryExpr(ptr_fun(float_sign));
    
    cout<<endl<<"B.maxCoeff() = " << B.maxCoeff();
    cout<<endl<<"B.minCoeff() = " << B.minCoeff();
    cout<<endl<<"B.sum() = " << B.sum();
    // cout<<endl<<"deconcatenate(concatenate(B)).maxCoeff() = " << deconcatenate(concatenate(B)).maxCoeff();
    // cout<<endl<<"deconcatenate(concatenate(B)).minCoeff() = " << deconcatenate(concatenate(B)).minCoeff();
    cout<<endl<<"deconcatenate(concatenate(B)).sum() = " << deconcatenate(concatenate(B)).sum();
    cout<<endl<<"(B-deconcatenate(concatenate(B))).sum() = " << (B-deconcatenate(concatenate(B))).sum()<<endl<<endl;
    
    */
    
	cout << endl << "-- Testing Binary Matrix Product for Deep Learning --" << endl;	
    
    //Eigen::setNbThreads(4);
    const int threads_core = atoi(argv[1]);
    Eigen::setNbThreads(threads_core);
        
    //int N = 8192;
    int N = atoi(argv[2]); //4096;
    
    const int layer_sizes = atoi(argv[3]);
    
    std::ofstream outfile;

    for (auto layer_size = 0; layer_size != layer_sizes; layer_size++) {
        outfile.open("./results.txt", std::ios_base::app);
        MatrixXf A(N,N);
        // A.setZero();
        A.setRandom();
        A = A.unaryExpr(ptr_fun(float_sign));
        //cout <<endl<<"A max = " <<A.maxCoeff();
        //cout <<endl<<"A min = " <<A.minCoeff();
        //cout <<endl<<"A sum = " <<A.sum();

        MatrixXf B(N,N);
        // B.setZero();
        B.setRandom();
        B = B.unaryExpr(ptr_fun(float_sign));
        //cout <<endl<<"B max = " <<B.maxCoeff();
        //cout <<endl<<"B min = " <<B.minCoeff();
        
        //cout <<endl<<"A B diff = " <<(A-B).sum();
        
        MatrixXf C1(N,N);
        MatrixXf C2(N,N);
        
        double binary_start_time = omp_get_wtime();
        
        // at first loop initialize C1 as A

        
        // do forward propagation
        //
        //

        cout << endl << "Layer size = " << layer_size << endl;
        
        C1 = A;
        for (int i = 0; i < layer_size; i++) {
            //cout << "Computing layer " << i << endl;
            C1 = BMP(C1,B);
        }
        
        double binary_end_time = omp_get_wtime();

        // C1 = A*B;
        double binary_elapsed_time = binary_end_time - binary_start_time;
        
        cout <<endl<< "Binary matrix product elapsed_time = " << binary_elapsed_time << endl;
        
        double normal_start_time = omp_get_wtime();
        
        cout << endl << "Conducting normal Eigen SGEMM" << endl;

        // do forward propagation
        //
        cout << "Layer size = " << layer_size << endl;

        C2 = A;
        for (int i = 0; i < layer_size; i++) {
            //cout << "Computing layer " << i << endl;
            C2 = C2*B;
            // C2 = BMP(A,B);
        }

        // for resnet sometimes the layers are 10 layers deep
        // speed up

        double normal_end_time = omp_get_wtime();
        double normal_elapsed_time = normal_end_time - normal_start_time;
        cout << endl << "Eigen SGEMM elapsed_time = " << normal_elapsed_time<< endl;
        
        double speedup = normal_elapsed_time / binary_elapsed_time;

        cout << endl << "Speed up " << speedup << " x " << endl;
        //cout << endl <<"C1 sum = " << C1.sum();
        //cout << endl<<"C2 sum = " << C2.sum();
        //cout << endl<<"Mean difference = " << (C1-C2).mean()<<endl<<endl;
        outfile << layer_size << " "
               << N << " "
               << binary_elapsed_time << " "
               << normal_elapsed_time << " "
               << speedup << std::endl;

        outfile.close();
    }
    std::cout << "Experiment finished" << std::endl;
    return 0;
}    
    
