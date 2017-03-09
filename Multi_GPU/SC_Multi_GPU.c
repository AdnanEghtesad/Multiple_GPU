#include <stdio.h>
#include <time.h>
#include <math.h>
#include <sys/time.h> 
#include <stdlib.h>

#ifdef _OPENACC
#include <openacc.h>
#endif /*_OPENACC*/



#define NX 4096         
#define NY 4096 
#define iter_max 1000   
#define MAX_NUM_DEVICES 4
//#define num_devices 4


#ifdef USE_DOUBLE
    typedef double real;
    #define fmaxr fmax
    #define fabsr fabs
    #define expr exp
#else
    typedef float real;
    #define fmaxr fmaxf
    #define fabsr fabsf
    #define expr expf
#endif

//  using namespace std; 
int main () {
//int i,j;
//int n,m;
double tol,error;
int iter;
int dn;
  real A[NX][NY];
  real Anew[NX][NY];
  
    static double acc_time;
    static int acc_n;
    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);
  
	tol=.01;
	iter=0;
        error=2*tol;


int num_devices = 1;
#if _OPENACC
    acc_device_t device_type = acc_get_device_type();
    if ( acc_device_nvidia == device_type )
    {
        num_devices=acc_get_num_devices(acc_device_nvidia);
        for ( int dn = 0; dn < num_devices; dn++ )
        {
            acc_set_device_num(dn,acc_device_nvidia);
            acc_init(acc_device_nvidia);
        }
    }
    else
    {
        acc_init(device_type);
    }
#endif /*_OPENACC*/

    int ix_start = 1;
    int ix_end   = (NX - 1);

    // Ensure correctness if NY%num_devices != 0
    int chunk_size = ceil( (1.0*NY)/num_devices );

    int iy_start[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];

    iy_start[0] = 1;
    for ( int dn = 1; dn < num_devices; dn++ )
    {
        iy_start[dn] = iy_start[(dn-1)] + chunk_size;
        iy_end[dn-1] = iy_start[dn];
    }
    iy_end[(num_devices-1)] = NY-1;
    
    
 
	
  /* 
  
  //OpenACC Warm-up
    for ( int dn = 0; dn < num_devices; dn++ )
    {
        #pragma acc set device_num(dn)
        #pragma acc kernels
        for( int i = 0; i< NX; i++)
        {
            for( int j = 0; j < NX; j++ )
            {
              
               A[i][j]=i*2.f+j;
            }
        }
    }

  */	
	 
	 
        
    
        
        for( int i = 0; i< NX; i++)
        {
            for( int j = 0; j < NX; j++ )
            {
              
               A[i][j]=i*2.0+j;
            }
        }
    
 
   #ifdef Single_GPU
   #pragma acc enter data copyin(A[0:NX*NY]),create(Anew[0:NX*NY])
   #endif
   while ( error > tol && iter < iter_max ) 
   {

     
 //   !$omp parallel do shared(m, n, Anew,A) reduction( max:error )
    

   #ifdef  Multi_GPU    
    for ( int dn = 0; dn < num_devices; dn++ )
    {
        #pragma acc set device_num(dn)
        #pragma acc enter data copyin(A[(iy_start[dn]-1):(iy_end[dn]-iy_start[dn])+2][0:NX] ) create(Anew[iy_start[dn]:(iy_end[dn]-iy_start[dn])][0:NX])
    }	 
   #endif
   
   #ifdef Multi_GPUU   
   #pragma acc enter data copyin(A[0:N1*N2]),create(Anew[0:N1*N2])
   #endif



 #ifdef Multi_GPU

    for(int dn=0;dn<num_devices;dn++)
    {
     
    #pragma acc set device_num(dn)
    #pragma acc kernels async
    
    for (int iy = iy_start[dn]; iy < iy_end[dn]; iy++)
           
    for( int ix = ix_start; ix < ix_end; ix++ )
                
    {  
    { 
      Anew[iy][ix] = 0.25  *( A[iy+1][ix] + A[iy-1][ix] + 
      A[iy][ix-1] + A[iy][ix+1] );
     // error = fmaxf( error, fabsf(Anew[i][j]-A[i][j]) );
      }
      }
      
   }   
    
   
 #elif Single_GPU
  
    #pragma acc kernels
    #ifdef COLLAPSE
    #pragma acc loop collapse(2)
    #endif
    
    
    for(int i=1;i<NX-1;i++){
    for(int j=1;j<NY-1;j++){  
     
      Anew[i][j] = 0.25  *( A[i+1][j] + A[i-1][j] + 
      A[i][j-1] + A[i][j+1] );
     // error = fmaxf( error, fabsf(Anew[i][j]-A[i][j]) );
      }
    }
   
#elif serial

    for(int i=1;i<NX-1;i++){
    for(int j=1;j<NY-1;j++){  
     
      Anew[i][j] = 0.25  *( A[i+1][j] + A[i-1][j] + 
      A[i][j-1] + A[i][j+1] );
     // error = fmaxf( error, fabsf(Anew[i][j]-A[i][j]) );
      }
    }
   
 #endif   
 

  
         if(iter % 100==0 ) {	  
         // printf("%d, %g\n", iter, error);
          printf("%d\n",iter);
	 }  
        iter++;
     
   
   
   
 #if Multi_GPUU
    for(int dn=0;dn<num_devices;dn++)
    {
   
    #pragma acc kernels async
     
   
    for (int iy = iy_start[dn]; iy < iy_end[dn]; iy++)
          
    for( int ix = ix_start; ix < ix_end; ix++ )
    {
    {
      A[iy][ix] = Anew[iy][ix];
    }
    }
    }
    printf("here");
    getchar();
   
 #elif Single_GPU
 #pragma acc kernels
   
    #ifdef COLLAPSE
    #pragma acc loop collapse(2)
    #endif
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){  
     
      A[i][j] = Anew[i][j];
     // error = fmaxf( error, fabsf(Anew[i][j]-A[i][j]) );
      }
    }
    
   
   
 #elif serial
 
 for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){  
     
      A[i][j] = Anew[i][j];
     // error = fmaxf( error, fabsf(Anew[i][j]-A[i][j]) );
      }
    }
   }  
   
   
   
 #endif   


     
   }//while ( error > tol && iter < iter_max )  
   #ifdef Single_GPU
   #pragma acc exit data copyout(A[0:NX*NY]),delete(Anew[0:NX*NY])
   #endif


    #ifdef Multi_GPU
    for(int dn=0; dn<num_devices;dn++) 
    {
   #pragma acc set device_num(dn)
   #pragma acc wait
    }
    #endif


  #ifdef Multi_GPU    
   for ( int dn = 0; dn < num_devices; dn++ )
    {
        #pragma acc set device_num(dn)
        #pragma acc exit data copyout(A[(iy_start[dn]-1):(iy_end[dn]-iy_start[dn])+2][0:NX]) delete(Anew[iy_start[dn]:(iy_end[dn]-iy_start[dn])][0:NX])
    } 
  #endif
  #ifdef Multi_GPUU   
  #pragma acc exit data copyout(A[0:NX*NY]),delete(Anew[0:NX*NY])
  #endif

    clock_gettime(CLOCK_REALTIME, &now);
    acc_time += (now.tv_sec+now.tv_nsec*1e-9) - (tmstart.tv_sec+tmstart.tv_nsec*1e-9);
    acc_n++;

    printf("avg energy_calculation avg time: %g total time %g\n", acc_time / acc_n, acc_time);
    
    
return 0;
 
}
